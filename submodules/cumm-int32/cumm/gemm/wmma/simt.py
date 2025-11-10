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

from typing import List, Tuple, Union

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import bases, constants, core, layout, thread_map
from cumm.gemm.arch import instmma


class WarpMmaSimt(bases.WarpMma):
    def __init__(self, thread_mma_shape: Tuple[int, int,
                                               int], dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType, dtype_c: dtypes.DType, trans_a: bool,
                 trans_b: bool, trans_c: bool):
        # TODO merge mma sync
        super().__init__()
        self.add_dependency(TensorViewNVRTC, layout.RowMajor,
                            layout.ColumnMajor)
        self.thread_mma_shape = (thread_mma_shape[0], thread_mma_shape[1],
                                 thread_mma_shape[2])
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.mn = thread_mma_shape[0] * thread_mma_shape[1]
        self.km = thread_mma_shape[2] * thread_mma_shape[0]
        self.kn = thread_mma_shape[2] * thread_mma_shape[1]

        self.fragment_a_t = self.array_type(str(dtype_a), self.km)
        self.fragment_b_t = self.array_type(str(dtype_b), self.kn)
        self.fragment_c_t = self.array_type(str(dtype_c), self.mn)
        dabc = (dtype_a, dtype_b, dtype_c)
        use_dp4a = (thread_mma_shape[2] % 4 == 0
                    and dabc == (dtypes.int8, dtypes.int8, dtypes.int32))
        use_dp4a &= (trans_a and not trans_b) or (trans_b and not trans_a)
        use_hfma2 = (not trans_c and thread_mma_shape[1] % 4 == 0)
        use_hfma2 |= (trans_c and thread_mma_shape[0] % 4 == 0)
        use_hfma2 &= dabc == (dtypes.float16, dtypes.float16, dtypes.float16)
        # use_hfma2 = False
        if use_dp4a:
            self.mma = instmma.InstMma((1, 1, 4), 1, dtype_a, dtype_b, dtype_c,
                                       trans_a, trans_b, trans_c)
        elif use_hfma2:
            inst_shape = (2, 1, 1) if trans_c else (1, 2, 1)
            self.mma = instmma.InstMma(inst_shape, 1, dtype_a, dtype_b,
                                       dtype_c, trans_a, trans_b, trans_c)
        else:
            self.mma = instmma.InstMma((1, 1, 1), 1, dtype_a, dtype_b, dtype_c,
                                       trans_a, trans_b, trans_c)
        self.use_hfma2 = use_hfma2
        self.use_dp4a = use_dp4a
        self.add_param_class("instmma", self.mma, "InstMma")

    def array_type(self, dtype: Union[str, dtypes.DType], count: int):
        return core.array_type(dtype, count)

    def python_ctor(self):
        return self

    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator(self):
        la = "ColumnMajor" if self.trans_a else "RowMajor"
        lb = "ColumnMajor" if self.trans_b else "RowMajor"
        lc = "ColumnMajor" if self.trans_c else "RowMajor"
        mma_iters = np.array(self.thread_mma_shape) // np.array(self.mma.shape)

        code = pccm.code()
        code.arg("D", f"{self.fragment_c_t}&")
        code.arg("A", f"{self.fragment_a_t} const &")
        code.arg("B", f"{self.fragment_b_t} const &")
        code.arg("C", f"{self.fragment_c_t} const &")
        if self.use_dp4a:
            code.raw(f"""
            constexpr {lc} layoutC = {lc}::from_shape({{{self.thread_mma_shape[0]}, {self.thread_mma_shape[1]}}});
            InstMma mma;
            D = C;
            {self.array_type(dtypes.int8, 4)} const *ptr_A =
                reinterpret_cast<{self.array_type(dtypes.int8, 4)} const *>(&A);
            {self.array_type(dtypes.int8, 4)} const *ptr_B =
                reinterpret_cast<{self.array_type(dtypes.int8, 4)} const *>(&B);
            """)
            with code.range_("k", mma_iters[2], "TV_PRAGMA_UNROLL"):
                with code.range_("n", mma_iters[1], "TV_PRAGMA_UNROLL"):
                    with code.range_("m", mma_iters[0], "TV_PRAGMA_UNROLL"):
                        code.raw(f"""
                        {self.array_type(dtypes.int32, 1)} tmp =
                            reinterpret_cast<{self.array_type(dtypes.int32, 1)} &>(D[layoutC(m, n)]);
                        """)
                        if self.trans_b and not self.trans_a:
                            code.raw(f"""
                            mma(tmp, ptr_A[m + k * {self.thread_mma_shape[0]}], ptr_B[n + k * {self.thread_mma_shape[0]}], tmp);
                            """)
                        else:
                            code.raw(f"""
                            mma(tmp, ptr_A[m * {self.thread_mma_shape[2]} / {self.mma.shape[2]} + k],
                                ptr_B[n * {self.thread_mma_shape[2]} / {self.mma.shape[2]} + k], tmp);
                            """)
                        code.raw("""
                        D[layoutC(m, n)] = reinterpret_cast<int32_t &>(tmp);
                        """)
        elif self.use_hfma2:
            ccount = self.mma.shape[0] * self.mma.shape[1]

            code.raw(f"""
            InstMma mma;
            {self.array_type(self.dtype_c, ccount)} *ptr_D =
                reinterpret_cast<{self.array_type(self.dtype_c, ccount)} *>(&D);
            {self.array_type(self.dtype_a, self.mma.shape[0])} const *ptr_A =
                reinterpret_cast<{self.array_type(self.dtype_a, self.mma.shape[0])} const *>(&A);
            {self.array_type(self.dtype_b, self.mma.shape[1])} const *ptr_B =
                reinterpret_cast<{self.array_type(self.dtype_b, self.mma.shape[1])} const *>(&B);
            """)
            if self.trans_c:
                with code.range_("k", mma_iters[2], "TV_PRAGMA_UNROLL"):
                    with code.range_("m", mma_iters[0], "TV_PRAGMA_UNROLL"):
                        with code.range_("n", mma_iters[1],
                                         "TV_PRAGMA_UNROLL"):
                            code.raw(f"""
                            {self.array_type(dtypes.float16, 2)} tmp;
                            {self.array_type(dtypes.float16, 2)} *ptr_tmp = &tmp;
                            """)
                            if not self.trans_a:
                                code.raw(f"""
                                ptr_tmp[0] = ptr_D[n * {self.thread_mma_shape[0]} / 2 + m];
                                {self.array_type(dtypes.float16, 2)} tmp_A;
                                // row major A, read 2 elements in 2 row
                                tmp_A[0] = (*ptr_A)[2 * m * {self.thread_mma_shape[2]} + k];
                                tmp_A[1] = (*ptr_A)[(2 * m + 1) * {self.thread_mma_shape[2]} + k];
                                """)
                                if not self.trans_b:
                                    code.raw(
                                        f"mma(tmp, tmp_A, ptr_B[k * {self.thread_mma_shape[1]} + n], tmp);"
                                    )
                                else:
                                    code.raw(
                                        f"mma(tmp, tmp_A, ptr_B[n * {self.thread_mma_shape[2]} + k], tmp);"
                                    )
                            else:
                                code.raw(f"""
                                ptr_tmp[0] = ptr_D[n * {self.thread_mma_shape[0]} / 2 + m];
                                """)
                                if not self.trans_b:
                                    code.raw(f"""
                                    mma(tmp, ptr_A[k * {self.thread_mma_shape[0]} / 2 + m], ptr_B[k * {self.thread_mma_shape[1]} + n],
                                        tmp);
                                    """)
                                else:
                                    code.raw(f"""
                                    mma(tmp, ptr_A[k * {self.thread_mma_shape[0]} / 2 + m], ptr_B[n * {self.thread_mma_shape[2]} + k],
                                        tmp);
                                    """)
                            code.raw(f"""
                            ptr_D[m + n * {self.thread_mma_shape[0]} / 2] = ptr_tmp[0];
                            """)

            else:
                with code.range_("k", mma_iters[2], "TV_PRAGMA_UNROLL"):
                    with code.range_("n", mma_iters[1], "TV_PRAGMA_UNROLL"):
                        with code.range_("m", mma_iters[0],
                                         "TV_PRAGMA_UNROLL"):
                            code.raw(f"""
                            {self.array_type(dtypes.float16, 2)} tmp;
                            {self.array_type(dtypes.float16, 2)} *ptr_tmp = &tmp;
                            """)
                            if not self.trans_b:
                                code.raw(f"""
                                ptr_tmp[0] = ptr_D[m * {self.thread_mma_shape[1]} / 2 + n];
                                """)
                                if not self.trans_a:
                                    code.raw(f"""
                                    mma(tmp, ptr_A[m * {self.thread_mma_shape[2]} + k], ptr_B[k * {self.thread_mma_shape[1]} / 2 + n],
                                        tmp);
                                    """)
                                else:
                                    code.raw(f"""
                                    mma(tmp, ptr_A[k * {self.thread_mma_shape[0]} + m], ptr_B[k * {self.thread_mma_shape[1]} / 2 + n],
                                        tmp);
                                    """)

                            else:
                                code.raw(f"""
                                ptr_tmp[0] = ptr_D[m * {self.thread_mma_shape[1]} / 2 + n];
                                // col major B, read 2 elements in 2 row
                                {self.array_type(dtypes.float16, 2)} tmp_B;
                                tmp_B[0] = (*ptr_B)[2 * n * {self.thread_mma_shape[2]} + k];
                                tmp_B[1] = (*ptr_B)[(2 * n + 1) * {self.thread_mma_shape[2]} + k];
                                """)
                                if not self.trans_a:
                                    code.raw(
                                        f"mma(tmp, ptr_A[m * {self.thread_mma_shape[2]} + k], tmp_B, tmp);"
                                    )
                                else:
                                    code.raw(
                                        f"mma(tmp, ptr_A[k * {self.thread_mma_shape[0]} + m], tmp_B, tmp);"
                                    )
                            code.raw(f"""
                            ptr_D[m * {self.thread_mma_shape[1]} / 2 + n] = ptr_tmp[0];
                            """)
        else:
            # fall back to basic impl
            code.raw(f"""
            constexpr {la} layoutA = {la}::from_shape({{{self.thread_mma_shape[0]}, {self.thread_mma_shape[2]}}});
            constexpr {lb} layoutB = {lb}::from_shape({{{self.thread_mma_shape[2]}, {self.thread_mma_shape[1]}}});
            constexpr {lc} layoutC = {lc}::from_shape({{{self.thread_mma_shape[0]}, {self.thread_mma_shape[1]}}});
            InstMma mma;
            D = C;
            TV_PRAGMA_UNROLL
            for (int k = 0; k < {self.thread_mma_shape[2]}; ++k) {{
                TV_PRAGMA_UNROLL
                for (int n = 0; n < {self.thread_mma_shape[1]}; ++n) {{
                    TV_PRAGMA_UNROLL
                    for (int m = 0; m < {self.thread_mma_shape[0]}; ++m) {{
                        // what's this????
                        // Column-major serpentine sequence to maximize reuse of A operand.
                        // "mma_tensor_op_sm70.h:243"
                        int m_serpentine = (n % 2) ? ({self.thread_mma_shape[0]} - 1 - m) : m;
                        {self.mma.fragment_c_t} d;
                        {self.mma.fragment_a_t} a;
                        {self.mma.fragment_b_t} b;
                        d[0] = D[layoutC(m_serpentine, n)];
                        a[0] = A[layoutA(m_serpentine, k)];
                        b[0] = B[layoutB(k, n)];
                        mma(d, a, b, d);
                        D[layoutC(m_serpentine, n)] = d[0];
                    }}
                }}
            }}

            """)
        return code

    async def __call__(self, D: ArrayPtr, A: ArrayPtr, B: ArrayPtr,
                       C: ArrayPtr):
        mma_iters = np.array(self.thread_mma_shape) // np.array(self.mma.shape)
        la = layout.ColumnMajor() if self.trans_a else layout.RowMajor()
        lb = layout.ColumnMajor() if self.trans_b else layout.RowMajor()
        lc = layout.ColumnMajor() if self.trans_c else layout.RowMajor()
        inst_mma = self.mma

        if self.use_dp4a:
            D.data.numpy_view()[:] = C.data.numpy_view()
            layoutC = lc.from_shape_python(self.thread_mma_shape[:2])
            ptr_A = A.change_access_size(4)
            ptr_B = B.change_access_size(4)

            for k in range(mma_iters[2]):
                for n in range(mma_iters[1]):
                    for m in range(mma_iters[0]):
                        # detach from a array ptr
                        tmp = D[layoutC(m, n)].copy()

                        if self.trans_b and not self.trans_a:
                            inst_mma(tmp,
                                     ptr_A[m + k * self.thread_mma_shape[0]],
                                     ptr_B[n + k * self.thread_mma_shape[0]],
                                     tmp)
                        else:
                            inst_mma(
                                tmp, ptr_A[m * self.thread_mma_shape[2] //
                                           self.mma.shape[2] + k],
                                ptr_B[n * self.thread_mma_shape[2] //
                                      self.mma.shape[2] + k], tmp)
                        D[layoutC(m, n)] = tmp
        elif self.use_hfma2:
            ccount = self.mma.shape[0] * self.mma.shape[1]

            D.data.numpy_view()[:] = C.data.numpy_view()
            ptr_D = D.change_access_size(ccount)
            ptr_A = A.change_access_size(self.mma.shape[0])
            ptr_B = B.change_access_size(self.mma.shape[1])

            if self.trans_c:
                for k in range(mma_iters[2]):
                    for m in range(mma_iters[0]):
                        for n in range(mma_iters[1]):
                            tmp = ptr_D[n * self.thread_mma_shape[0] // 2 +
                                        m].copy()
                            if not self.trans_a:
                                tmp_A = ArrayPtr(dtypes.float16.tv_dtype, 2)
                                tmp_A[0] = ptr_A[2 * m *
                                                 self.thread_mma_shape[2] + k]
                                tmp_A[1] = ptr_A[(2 * m + 1) *
                                                 self.thread_mma_shape[2] + k]
                                if not self.trans_b:
                                    self.mma(
                                        tmp, tmp_A,
                                        ptr_B[k * self.thread_mma_shape[1] +
                                              n], tmp)
                                else:
                                    self.mma(
                                        tmp, tmp_A,
                                        ptr_B[n * self.thread_mma_shape[2] +
                                              k], tmp)
                            else:
                                if not self.trans_b:
                                    self.mma(
                                        tmp,
                                        ptr_A[k * self.thread_mma_shape[0] // 2
                                              + m],
                                        ptr_B[k * self.thread_mma_shape[1] +
                                              n], tmp)
                                else:
                                    self.mma(
                                        tmp,
                                        ptr_A[k * self.thread_mma_shape[0] // 2
                                              + m],
                                        ptr_B[n * self.thread_mma_shape[2] +
                                              k], tmp)
                            ptr_D[m + n * self.thread_mma_shape[0] // 2] = tmp
            else:
                for k in range(mma_iters[2]):
                    for n in range(mma_iters[1]):
                        for m in range(mma_iters[0]):
                            tmp = ptr_D[m * self.thread_mma_shape[1] // 2 +
                                        n].copy()
                            if self.trans_b:
                                tmp_B = ArrayPtr(dtypes.float16.tv_dtype, 2)
                                tmp_B[0] = ptr_B[2 * n *
                                                 self.thread_mma_shape[2] + k]
                                tmp_B[1] = ptr_B[(2 * n + 1) *
                                                 self.thread_mma_shape[2] + k]
                                if not self.trans_a:
                                    self.mma(
                                        tmp,
                                        ptr_A[m * self.thread_mma_shape[2] +
                                              k], tmp_B, tmp)
                                else:
                                    self.mma(
                                        tmp,
                                        ptr_A[k * self.thread_mma_shape[0] +
                                              m], tmp_B, tmp)
                            else:
                                if not self.trans_a:
                                    self.mma(
                                        tmp,
                                        ptr_A[m * self.thread_mma_shape[2] +
                                              k],
                                        ptr_B[k * self.thread_mma_shape[1] // 2
                                              + n], tmp)
                                else:
                                    self.mma(
                                        tmp,
                                        ptr_A[k * self.thread_mma_shape[0] +
                                              m],
                                        ptr_B[k * self.thread_mma_shape[1] // 2
                                              + n], tmp)
                            ptr_D[m * self.thread_mma_shape[1] // 2 + n] = tmp
        else:
            M, N, K = self.thread_mma_shape[0], self.thread_mma_shape[
                1], self.thread_mma_shape[2]
            # print(M, N, K)
            layoutA = la.from_shape_python([M, K])
            layoutB = lb.from_shape_python([K, N])
            layoutC = lc.from_shape_python([M, N])
            if self.trans_a:
                A_mat = A.data.numpy_view().reshape(K, M).T
            else:
                A_mat = A.data.numpy_view().reshape(M, K)
            if self.trans_b:
                B_mat = B.data.numpy_view().reshape(N, K).T
            else:
                B_mat = B.data.numpy_view().reshape(K, N)
            C_mat = A_mat @ B_mat
            if self.trans_c:
                D.data.numpy_view()[:] = (C.data.numpy_view() +
                                          C_mat.T.reshape(-1))
            else:
                D.data.numpy_view()[:] = (C.data.numpy_view() +
                                          C_mat.reshape(-1))
            # if cudasim.threadIdx().x == 0:
            #     acc = D.data.numpy_view()
            #     print(A.data.numpy_view())
            #     # print(A_mat.T)
            #     print(acc.mean(), acc.min(), acc.max())

            return

            # layoutA = la.from_shape_python([M, K])
            # layoutB = lb.from_shape_python([K, N])
            # layoutC = lc.from_shape_python([M, N])
            # D.data.copy_(C.data)
            # for k in range(self.thread_mma_shape[2]):
            #     for n in range(self.thread_mma_shape[1]):
            #         for m in range(self.thread_mma_shape[0]):
            #             if n % 2 != 0:
            #                 m_serpentine = (self.thread_mma_shape[0] - 1 - m)
            #             else:
            #                 m_serpentine = m
            #             d = D[layoutC(m_serpentine, n)].copy()
            #             a = A[layoutA(m_serpentine, k)].copy()
            #             b = B[layoutB(k, n)].copy()
            #             inst_mma(d, a, b, d)
            #             D[layoutC(m_serpentine, n)] = d
