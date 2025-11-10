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

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import bases, constants, core, layout, thread_map
from cumm.gemm.arch import tensorop


def seq(*vals):
    return np.array([*vals], dtype=np.int64)


class WarpMmaVolta(bases.WarpMma):
    def __init__(self, warp_tile_shape: Tuple[int, int,
                                              int], dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType, dtype_c: dtypes.DType, trans_a: bool,
                 trans_b: bool, trans_c: bool):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.warp_tile_shape = seq(warp_tile_shape[0], warp_tile_shape[1],
                                   warp_tile_shape[2])
        self.interleaved_mma_shape = seq(32, 32, 4)
        self.inst_shape = seq(16, 16, 4)

        self.mma_iters = self.interleaved_mma_shape // self.inst_shape  # type: np.ndarray
        self.mma_tile_iters = self.warp_tile_shape // self.interleaved_mma_shape  # type: np.ndarray
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.mn = warp_tile_shape[0] * warp_tile_shape[1]
        self.km = warp_tile_shape[2] * warp_tile_shape[0]
        self.kn = warp_tile_shape[2] * warp_tile_shape[1]
        self.fragment_a_count = self.inst_shape[2] * self.mma_iters[
            0] * self.mma_tile_iters[0]
        self.fragment_b_count = self.inst_shape[2] * self.mma_iters[
            1] * self.mma_tile_iters[1]
        self.fragment_c_count = self.mma_iters.prod(
        ) * self.mma_tile_iters.prod() * 8

        self.fragment_a_t = core.array_type(str(dtype_a),
                                            self.fragment_a_count)
        self.fragment_b_t = core.array_type(str(dtype_b),
                                            self.fragment_b_count)
        self.fragment_c_t = core.array_type(str(dtype_c),
                                            self.fragment_c_count)
        self.mma = tensorop.MmaSync((8, 8, 4), 8, dtype_a, dtype_b, dtype_c,
                                    trans_a, trans_b, trans_c)
        self.add_param_class("tensorop", self.mma, "InstMma")

    def python_ctor(self):
        return self

    def min_arch(self) -> Optional[Tuple[int, int]]:
        return self.mma.min_arch()

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
        TV_PRAGMA_UNROLL
        for (int outer_col = 0; outer_col < {self.mma_tile_iters[1]}; ++outer_col) {{
            TV_PRAGMA_UNROLL
            for (int inner_col = 0; inner_col < {self.mma_iters[1]}; ++inner_col) {{
                TV_PRAGMA_UNROLL
                for (int outer_row = 0; outer_row < {self.mma_tile_iters[0]};
                    ++outer_row) {{
                    TV_PRAGMA_UNROLL
                    for (int inner_row = 0; inner_row < {self.mma_iters[0]}; ++inner_row) {{
                        int op_col = inner_col + {self.mma_iters[1]} * outer_col;
                        // Column-major serpentine sequence to maximize reuse of A operand.
                        int inner_row_serp = inner_row;
                        int outer_row_serp = outer_row;
                        if (op_col & 1) {{
                            inner_row_serp = {self.mma_iters[0]} - inner_row - 1;
                            outer_row_serp = {self.mma_tile_iters[0]} - outer_row - 1;
                        }}
                        int op_row = inner_row_serp + {self.mma_iters[0]} * outer_row_serp;
                        // op_idx: [kMmaTileIterations[1], kMmaTileIterations[0],
                        // kMmaIterations[1], kMmaIterations[0]]

                        int op_idx =
                            inner_row_serp +
                            {self.mma_iters[0]} *
                                (inner_col +
                                {self.mma_iters[1]} *
                                    (outer_row_serp + {self.mma_tile_iters[0]} * outer_col));
                        mma(ptr_D[op_idx], ptr_A[op_row], ptr_B[op_col], ptr_D[op_idx]);

                    }}
                }}
            }}
        }}
        """)
        return code

    async def __call__(self, D: ArrayPtr, A: ArrayPtr, B: ArrayPtr,
                       C: ArrayPtr):
        D.data.numpy_view()[:] = C.data.numpy_view()
        ptr_A = A.change_access_size(self.mma.fragment_a_count)
        ptr_B = B.change_access_size(self.mma.fragment_b_count)
        ptr_D = D.change_access_size(self.mma.fragment_c_count)
        for outer_col in range(self.mma_tile_iters[1]):
            for inner_col in range(self.mma_iters[1]):
                for outer_row in range(self.mma_tile_iters[0]):
                    for inner_row in range(self.mma_iters[0]):
                        op_col = inner_col + self.mma_iters[1] * outer_col
                        # Column-major serpentine sequence to maximize reuse of A operand.
                        inner_row_serp = inner_row
                        outer_row_serp = outer_row
                        if (op_col & 1):
                            inner_row_serp = self.mma_iters[0] - inner_row - 1
                            outer_row_serp = self.mma_tile_iters[
                                0] - outer_row - 1

                        op_row = inner_row_serp + self.mma_iters[
                            0] * outer_row_serp
                        # op_idx: [kMmaTileIterations[1], kMmaTileIterations[0],
                        # kMmaIterations[1], kMmaIterations[0]]

                        op_idx = (inner_row_serp + self.mma_iters[0] *
                                  (inner_col + self.mma_iters[1] *
                                   (outer_row_serp +
                                    self.mma_tile_iters[0] * outer_col)))
                        # if cudasim.threadIdx().x == 0:
                        #     print(outer_col, inner_col, outer_row, inner_row)
                        await self.mma(ptr_D[op_idx], ptr_A[op_row],
                                       ptr_B[op_col], ptr_D[op_idx])
