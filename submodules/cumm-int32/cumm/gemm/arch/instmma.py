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

from typing import List, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import constants, core


class InstMma(pccm.ParameterizedClass):
    def __init__(self, shape: Tuple[int, int, int], num_threads: int,
                 dtype_a: dtypes.DType, dtype_b: dtypes.DType,
                 dtype_c: dtypes.DType, trans_a: bool, trans_b: bool,
                 trans_c: bool):
        # TODO merge mma sync
        super().__init__()
        self.shape = (shape[0], shape[1], shape[2])
        self.shape = shape
        self.num_threads = num_threads
        if num_threads == 8:
            assert shape[0] == 8 and shape[1] == 8 and shape[2] == 4
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.mn = shape[0] * shape[1]
        self.km = shape[2] * shape[0]
        self.kn = shape[2] * shape[1]

        element_count_c = self.mn // self.num_threads
        element_count_a = self.km // self.num_threads
        element_count_b = self.kn // self.num_threads

        self.fragment_a_t = core.array_type(str(dtype_a), element_count_a)
        self.fragment_b_t = core.array_type(str(dtype_b), element_count_b)
        self.fragment_c_t = core.array_type(str(dtype_c), element_count_c)

    def python_ctor(self):
        return self

    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator(self):
        code = pccm.code()
        code.arg("d", f"{self.fragment_c_t}&")
        code.arg("a", f"{self.fragment_a_t} const &")
        code.arg("b", f"{self.fragment_b_t} const &")
        code.arg("c", f"{self.fragment_c_t} const &")
        dabc = (self.dtype_a, self.dtype_b, self.dtype_c)
        if self.shape == (1, 1, 1):
            if dabc == (dtypes.float16, dtypes.float16, dtypes.float32):
                code.raw("d[0] = float(a[0]) * float(b[0]) + c[0];")
            elif dabc == (dtypes.bfloat16, dtypes.bfloat16, dtypes.float32):
                code.raw("d[0] = float(a[0]) * float(b[0]) + c[0];")
            else:
                code.raw("d[0] = a[0] * b[0] + c[0];")
        elif self.shape == (1, 1, 4) or self.shape == (1, 1, 2):
            code.raw(f"""
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
            unsigned const &A = reinterpret_cast<unsigned const &>(a);
            unsigned const &B = reinterpret_cast<unsigned const &>(b);
            """)
            if dabc == (dtypes.int8, dtypes.int8, dtypes.int32):
                code.raw(f"""
                asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                            : "=r"(d[0])
                            : "r"(A), "r"(B), "r"(c[0]));
                """)
            elif dabc == (dtypes.int16, dtypes.int16, dtypes.int32):
                code.raw(f"""
                asm volatile("dp2a.s32.s32 %0, %1, %2, %3;"
                            : "=r"(d[0])
                            : "r"(A), "r"(B), "r"(c[0]));
                """)
            else:
                raise NotImplementedError
            code.raw(f"""
            #else
                d[0] = c[0];
                TV_PRAGMA_UNROLL
                for (int k = 0; k < {self.shape[2]}; ++k) {{
                    d[0] += a[k] * b[k];
                }}
            #endif
            """)
        elif self.shape == (2, 1, 1):
            if dabc == (dtypes.float16, dtypes.float16, dtypes.float16):
                code.raw(f"""
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
                    __half2 const & A = reinterpret_cast<__half2 const &>(a);
                    __half2 B = __half2half2(reinterpret_cast<__half const &>(b));
                    __half2 const & C = reinterpret_cast<__half2 const &>(c);
                    __half2 D = __hfma2(A, B, C);
                    d = reinterpret_cast<{core.array_type(self.dtype_c, 2)} &>(D);
                #else
                    TV_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {{
                        d[i] = a[i] * b[0] + c[i];
                    }}
                #endif
                """)
            elif dabc == (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16):
                code.raw(f"""
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
                    __nv_bfloat162 const & A = reinterpret_cast<__nv_bfloat162 const &>(a);
                    __nv_bfloat162 B = __half2half2(reinterpret_cast<__nv_bfloat16 const &>(b));
                    __nv_bfloat162 const & C = reinterpret_cast<__nv_bfloat162 const &>(c);
                    __nv_bfloat162 D = __hfma2(A, B, C);
                    d = reinterpret_cast<{core.array_type(self.dtype_c, 2)} &>(D);
                #else
                    TV_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {{
                        d[i] = a[i] * b[0] + c[i];
                    }}
                #endif
                """)
            else:
                raise NotImplementedError

        elif self.shape == (1, 2, 1):
            assert dabc == (dtypes.float16, dtypes.float16, dtypes.float16)
            assert not self.trans_c
            if dabc == (dtypes.float16, dtypes.float16, dtypes.float16):
                code.raw(f"""
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
                    __half2 const & A = __half2half2(reinterpret_cast<__half const &>(a));
                    __half2 B = reinterpret_cast<__half2 const &>(b);
                    __half2 const & C = reinterpret_cast<__half2 const &>(c);
                    __half2 D = __hfma2(A, B, C);
                    d = reinterpret_cast<{core.array_type(self.dtype_c, 2)} &>(D);
                #else
                    TV_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {{
                        d[i] = a[0] * b[i] + c[i];
                    }}
                #endif
                """)
            elif dabc == (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16):
                code.raw(f"""
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
                    __nv_bfloat162 const & A = __half2half2(reinterpret_cast<__nv_bfloat16 const &>(a));
                    __nv_bfloat162 B = reinterpret_cast<__nv_bfloat162 const &>(b);
                    __nv_bfloat162 const & C = reinterpret_cast<__half2 const &>(c);
                    __nv_bfloat162 D = __hfma2(A, B, C);
                    d = reinterpret_cast<{core.array_type(self.dtype_c, 2)} &>(D);
                #else
                    TV_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {{
                        d[i] = a[0] * b[i] + c[i];
                    }}
                #endif
                """)
            else:
                raise NotImplementedError

        elif self.shape == (2, 2, 1):
            assert self.trans_a and not self.trans_b

            if dabc == (dtypes.float16, dtypes.float16, dtypes.float16):
                code.raw(f"""
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
                """)

                if self.trans_c:
                    code.raw(f"""
                    __half2 const & A = reinterpret_cast<__half2 const &>(a);
                    __half2 Blo = __low2half2(reinterpret_cast<__half2 const &>(b));
                    __half2 Bhi = __high2half2(reinterpret_cast<__half2 const &>(b));

                    __half2 const *C = reinterpret_cast<__half2 const *>(&c);
                    __half2 Dlo = __hfma2(A, Blo, C[0]);
                    __half2 Dhi = __hfma2(A, Bhi, C[1]);
                    """)
                else:
                    code.raw(f"""
                    __half2 Alo = __low2half2(reinterpret_cast<__half2 const &>(a));
                    __half2 Ahi = __high2half2(reinterpret_cast<__half2 const &>(a));
                    __half2 const & B = reinterpret_cast<__half2 const &>(b);
                    
                    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

                    __half2 Dlo = __hfma2(Alo, B, C[0]);
                    __half2 Dhi = __hfma2(Ahi, B, C[1]);
                    """)
            elif dabc == (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16):
                code.raw(f"""
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
                """)
                if self.trans_c:
                    code.raw(f"""
                    __nv_bfloat162 const & A = reinterpret_cast<__nv_bfloat162 const &>(a);
                    __nv_bfloat162 Blo = __low2half2(reinterpret_cast<__nv_bfloat162 const &>(b));
                    __nv_bfloat162 Bhi = __high2half2(reinterpret_cast<__nv_bfloat162 const &>(b));

                    __nv_bfloat162 const *C = reinterpret_cast<__nv_bfloat162 const *>(&c);
                    __nv_bfloat162 Dlo = __hfma2(A, Blo, C[0]);
                    __nv_bfloat162 Dhi = __hfma2(A, Bhi, C[1]);
                    """)
                else:
                    code.raw(f"""
                    __nv_bfloat162 Alo = __low2half2(reinterpret_cast<__nv_bfloat162 const &>(a));
                    __nv_bfloat162 Ahi = __high2half2(reinterpret_cast<__nv_bfloat162 const &>(a));
                    __nv_bfloat162 const & B = reinterpret_cast<__nv_bfloat162 const &>(b);
                    
                    __nv_bfloat162 const *C = reinterpret_cast<__nv_bfloat162 const *>(&c);

                    __nv_bfloat162 Dlo = __hfma2(Alo, B, C[0]);
                    __nv_bfloat162 Dhi = __hfma2(Ahi, B, C[1]);
                    """)
            else:
                raise NotImplementedError
            code.raw(f"""
            {core.array_type(self.dtype_c, 2)} * D = reinterpret_cast<{core.array_type(self.dtype_c, 2)} *>(&d);
            D[0] = reinterpret_cast<{core.array_type(self.dtype_c, 2)} const &>(Dlo);
            D[1] = reinterpret_cast<{core.array_type(self.dtype_c, 2)} const &>(Dhi);
            """)
            code.raw("#else")
            if self.trans_c:
                code.raw(f"""
                TV_PRAGMA_UNROLL
                for (int j = 0; j < 2; ++j) {{
                    TV_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {{
                        d[i + 2 * j] = a[i] * b[j] + c[i + 2 * j];
                    }}
                }}
                """)
            else:
                code.raw(f"""
                TV_PRAGMA_UNROLL
                for (int i = 0; i < 2; ++i) {{
                    TV_PRAGMA_UNROLL
                    for (int j = 0; j < 2; ++j) {{
                        d[i * 2 + j] = a[i] * b[j] + c[i * 2 + j];
                    }}
                }}
                """)
            code.raw("#endif")
        else:
            raise NotImplementedError
        return code

    def __call__(self, d: ArrayPtr, a: ArrayPtr, b: ArrayPtr, c: ArrayPtr):
        if self.shape == (1, 1, 1):
            dabc = (self.dtype_a, self.dtype_b, self.dtype_c)
            if dabc == (dtypes.float16, dtypes.float16, dtypes.float32):
                d.data.numpy_view()[0] = float(a.data.numpy_view()[0]) * float(
                    b.data.numpy_view()[0]) + c.data.numpy_view()[0]
            else:
                d.data.numpy_view()[0] = a.data.numpy_view(
                )[0] * b.data.numpy_view()[0] + c.data.numpy_view()[0]
        elif self.shape == (1, 1, 4) or self.shape == (1, 1, 2):
            d.data.numpy_view()[0] = c.data.numpy_view()[0]
            for k in range(self.shape[2]):
                d.data.numpy_view(
                )[0] += a.data.numpy_view()[k] * b.data.numpy_view()[k]
        elif self.shape == (2, 1, 1):
            for i in range(2):
                d.data.numpy_view()[i] = a.data.numpy_view(
                )[i] * b.data.numpy_view()[0] + c.data.numpy_view()[i]
        elif self.shape == (1, 2, 1):
            for i in range(2):
                d.data.numpy_view()[i] = a.data.numpy_view(
                )[0] * b.data.numpy_view()[i] + c.data.numpy_view()[i]
        elif self.shape == (2, 2, 1):
            for j in range(2):
                for i in range(2):
                    d.data.numpy_view()[i + 2 * j] = a.data.numpy_view(
                    )[i] * b.data.numpy_view()[j] + c.data.numpy_view()[i +
                                                                        2 * j]
        else:
            raise NotImplementedError
