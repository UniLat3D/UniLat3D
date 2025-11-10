# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import numpy as np
import pccm

from cumm import dtypes
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import bases, constants, core, layout, thread_map
from cumm.gemm.arch import tensorop


def seq(*vals):
    return np.array([*vals], dtype=np.int64)


class WarpMmaTuring(bases.WarpMma):
    def __init__(self,
                 warp_tile_shape: Tuple[int, int, int],
                 inst_shape: Tuple[int, int, int],
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_c: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 top_dtype_a: Optional[dtypes.DType] = None,
                 top_dtype_b: Optional[dtypes.DType] = None,
                 top_dtype_c: Optional[dtypes.DType] = None,
                 acc_is_rowmajor: bool = False):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.warp_tile_shape = seq(*warp_tile_shape)
        self.inst_shape = seq(*inst_shape)

        self.mma_iters = self.warp_tile_shape // self.inst_shape  # type: np.ndarray
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.top_dtype_a = top_dtype_a or self.dtype_a
        self.top_dtype_b = top_dtype_b or self.dtype_b
        self.top_dtype_c = top_dtype_c or self.dtype_c

        self.mn = warp_tile_shape[0] * warp_tile_shape[1]
        self.km = warp_tile_shape[2] * warp_tile_shape[0]
        self.kn = warp_tile_shape[2] * warp_tile_shape[1]
        self.fragment_a_t = self.array_type(
            str(dtype_a), self.inst_shape[2] * warp_tile_shape[0] // 32)
        self.fragment_b_t = self.array_type(
            str(dtype_b), self.inst_shape[2] * warp_tile_shape[1] // 32)
        self.fragment_c_t = self.array_type(
            str(dtype_c), warp_tile_shape[0] * warp_tile_shape[1] // 32)

        self.mma = tensorop.MmaSync(inst_shape, 32, self.top_dtype_a, self.top_dtype_b, self.top_dtype_c,
                                    trans_a, trans_b, trans_c)
        self.add_param_class("tensorop", self.mma, "InstMma")
        self.acc_is_rowmajor = acc_is_rowmajor

    def min_arch(self) -> Optional[Tuple[int, int]]:
        return self.mma.min_arch()

    def array_type(self, dtype: str, count: int):
        return core.array_type(dtype, count)

    def python_ctor(self):
        return self

    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator(self):
        code = pccm.code()
        code.arg("D", f"{self.fragment_c_t}&")
        code.arg("A", f"{self.fragment_a_t} const &")
        code.arg("B", f"{self.fragment_b_t} const &")
        code.arg("C", f"{self.fragment_c_t} const &")

        code.raw(f"""
        InstMma mma;
        D = C;
        {self.mma.fragment_a_t} const *ptr_A = reinterpret_cast<{self.mma.fragment_a_t} const *>(&A);
        {self.mma.fragment_b_t} const *ptr_B = reinterpret_cast<{self.mma.fragment_b_t} const *>(&B);
        {self.mma.fragment_c_t} *ptr_D = reinterpret_cast<{self.mma.fragment_c_t} *>(&D);
        """)
        with code.macro_if_("defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)"):
            with code.range_("n", self.mma_iters[1], "TV_PRAGMA_UNROLL"):
                with code.range_("m", self.mma_iters[0], "TV_PRAGMA_UNROLL"):
                    code.raw(
                        f"int m_serpentine = ((n % 2) ? ({self.mma_iters[0]} - 1 - m) : m);"
                    )
                    if self.acc_is_rowmajor:
                        code.raw(f"""
                        mma(ptr_D[n + m_serpentine * {self.mma_iters[1]}],
                            ptr_A[m_serpentine],
                            ptr_B[n],
                            ptr_D[n + m_serpentine * {self.mma_iters[1]}]);
                        """)
                    else:
                        code.raw(f"""
                        mma(ptr_D[m_serpentine + n * {self.mma_iters[0]}],
                            ptr_A[m_serpentine],
                            ptr_B[n],
                            ptr_D[m_serpentine + n * {self.mma_iters[0]}]);
                        """)

        with code.macro_else_if_(
                "defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)"):
            with code.range_("m", self.mma_iters[0], "TV_PRAGMA_UNROLL"):
                with code.range_("n", self.mma_iters[1], "TV_PRAGMA_UNROLL"):
                    code.raw(
                        f"int n_serpentine = ((m % 2) ? ({self.mma_iters[1]} - 1 - n) : n);"
                    )
                    if self.acc_is_rowmajor:
                        code.raw(f"""
                        mma(ptr_D[n_serpentine + m * {self.mma_iters[1]}],
                            ptr_A[m],
                            ptr_B[n_serpentine],
                            ptr_D[n_serpentine + m * {self.mma_iters[1]}]);
                        """)
                    else:
                        code.raw(f"""
                        mma(ptr_D[m + n_serpentine * {self.mma_iters[0]}],
                            ptr_A[m],
                            ptr_B[n_serpentine],
                            ptr_D[m + n_serpentine * {self.mma_iters[0]}]);
                        """)

        code.macro_endif_()
        return code

    async def __call__(self, D: ArrayPtr, A: ArrayPtr, B: ArrayPtr,
                       C: ArrayPtr):
        D.data.numpy_view()[:] = C.data.numpy_view()
        ptr_A = A.change_access_size(self.mma.fragment_a_count)
        ptr_B = B.change_access_size(self.mma.fragment_b_count)
        ptr_D = D.change_access_size(self.mma.fragment_c_count)
        for n in range(self.mma_iters[1]):
            for m in range(self.mma_iters[0]):
                if n % 2:
                    m_serpentine = self.mma_iters[0] - 1 - m
                else:
                    m_serpentine = m
                if self.acc_is_rowmajor:
                    await self.mma(ptr_D[n + m_serpentine * self.mma_iters[1]],
                                   ptr_A[m_serpentine], ptr_B[n],
                                   ptr_D[n + m_serpentine * self.mma_iters[1]])
                else:
                    await self.mma(ptr_D[m_serpentine + n * self.mma_iters[0]],
                                   ptr_A[m_serpentine], ptr_B[n],
                                   ptr_D[m_serpentine + n * self.mma_iters[0]])
