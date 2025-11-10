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

from typing import List, Optional

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import bases, constants, layout, thread_map
from cumm.gemm.core import MetaArray, metaseq, seq


def seq(*vals):
    return np.array([*vals], dtype=np.int64)


def div_up(a, b):
    return (a + b - 1) // b


class VoltaMmaOutParams(object):
    def __init__(self, dtype: dtypes.DType, warp_tile_shape: MetaArray[int]):
        self.warp_tile_shape = warp_tile_shape
        self.interleaved_wmma_shape = metaseq(32, 32, 4)
        self.inst_shape = metaseq(16, 16, 4)
        self.mma_iters = self.interleaved_wmma_shape // self.inst_shape
        self.mma_tile_iters = warp_tile_shape // self.interleaved_wmma_shape
        self.is_float_acc = dtype == dtypes.float32
        self.rows_per_mma_tile = 2
        self.element_per_acc = 2 if self.is_float_acc else 4
        self.element_per_mma = 8
        self.acc_per_interleave_tile = 8 if self.is_float_acc else 4
        self.iterations = self.mma_tile_iters[0] * self.mma_iters[0]


class OutFragIterVolta(bases.GemmOutFragIterator):
    def __init__(self, dtype: dtypes.DType, warp_tile_shape: MetaArray[int]):
        self.params = VoltaMmaOutParams(dtype, warp_tile_shape)
        element_count = self.params.element_per_acc * self.params.acc_per_interleave_tile * self.params.mma_tile_iters[
            1]
        super().__init__(dtype, element_count, self.params.element_per_acc, -1)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)

        self.warp_tile_shape = warp_tile_shape

        self.add_member("index_", "int")
        self.add_member("src_ptr_", self.const_access_pointer)

        # cudasim members
        self.index_ = 0
        self.src_ptr_: Optional[ArrayPtr] = None

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"")
        code.arg("src_ptr", "const void*")
        code.ctor_init(
            "src_ptr_",
            f"reinterpret_cast<{self.const_access_pointer}>(src_ptr)")
        code.ctor_init("index_", "0")
        return code

    def python_ctor(self, src_ptr: ArrayPtr):
        new_obj = OutFragIterVolta(self.dtype, self.warp_tile_shape)
        new_obj.index_ = 0
        new_obj.src_ptr_ = src_ptr.change_access_size(self.element_per_acc)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(
            f"{self.access_pointer} frag_ptr = reinterpret_cast<{self.access_pointer}>(&frag);"
        )
        if not self.params.is_float_acc:
            code.raw(f"""
            static constexpr int kAccessesPerMma =
                {self.params.element_per_mma} / {self.element_per_acc};
            TV_PRAGMA_UNROLL
            for (int tile_n = 0; tile_n < {self.params.mma_tile_iters[1]}; ++tile_n) {{
                int tile_access_idx =
                    (tile_n * {self.params.mma_tile_iters[0]} + (index_ & 2) / 2) *
                    {self.params.mma_iters[0]} * {self.params.mma_iters[1]} * kAccessesPerMma;

                TV_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < {self.params.mma_iters[1]} * kAccessesPerMma;
                    ++mma_n) {{
                    int mma_access_idx =
                        ((mma_n & 1) * 2 + (index_ & 1)) * kAccessesPerMma +
                        (mma_n & 2) / 2;
                    frag_ptr[tile_n * {self.params.mma_iters[1]} * kAccessesPerMma + mma_n] =
                        src_ptr_[tile_access_idx + mma_access_idx];
                }}
            }}
            """)
        else:
            code.raw(f"""
            constexpr int kRegsPerMmaRow = 2;
            TV_PRAGMA_UNROLL
            for (int reg_row = 0; reg_row < {self.params.rows_per_mma_tile}; ++reg_row) {{
                TV_PRAGMA_UNROLL
                for (int tile_n = 0; tile_n < {self.params.mma_tile_iters[1]}; ++tile_n) {{
                    TV_PRAGMA_UNROLL
                    for (int mma_n = 0; mma_n < {self.params.mma_iters[1]} * 2; ++mma_n) {{
                        // (index_ & 1): 01010101
                        // (index_ & 0b10): 00110011
                        // (index_ & 2) * Policy::MmaIterations::kCount / 2:
                        // 00220022
                        // (mma_n & 1) * 2: 02020202
                        // (mma_n & 2) * 2: 00220022

                        int mma_idx = (index_ & 1) +
                                    (index_ & 2) * {self.params.mma_iters[:2].prod()} / 2 +
                                    (tile_n * {self.params.mma_tile_iters[0]}) *
                                        {self.params.mma_iters[:2].prod()} +
                                    (mma_n & 1) * 2;

                        int reg_offset = reg_row * kRegsPerMmaRow + (mma_n & 2) * 2;
                        int reg_idx = mma_idx * {self.params.element_per_mma} + reg_offset;
                        *frag_ptr = src_ptr_[reg_idx / {self.element_per_acc}];
                        ++frag_ptr;
                    }}
                }}
            }}
            """)
        code.arg("frag", f"{self.fragment_t} &")
        code.arg("index_offset", str(self.index_t), "0")
        return code

    def load_python(self, frag: ArrayPtr, index_offset: int = 0):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        if not self.params.is_float_acc:
            kAccessesPerMma = self.params.element_per_mma // self.element_per_acc
            for tile_n in range(self.params.mma_tile_iters[1]):
                tile_access_idx = ((tile_n * self.params.mma_tile_iters[0] +
                                    (self.index_ & 2) // 2) *
                                   self.params.mma_iters[0] *
                                   self.params.mma_iters[1] * kAccessesPerMma)

                for mma_n in range(self.params.mma_iters[1] * kAccessesPerMma):
                    mma_access_idx = (((mma_n & 1) * 2 +
                                       (self.index_ & 1)) * kAccessesPerMma +
                                      (mma_n & 2) // 2)
                    frag_ptr[tile_n * self.params.mma_iters[1] *
                             kAccessesPerMma +
                             mma_n] = self.src_ptr_[tile_access_idx +
                                                    mma_access_idx]

        else:
            kRegsPerMmaRow = 2
            idx = 0
            for reg_row in range(self.params.rows_per_mma_tile):
                for tile_n in range(self.params.mma_tile_iters[1]):
                    for mma_n in range(self.params.mma_iters[1] * 2):
                        # (index_ & 1): 01010101
                        # (index_ & 0b10): 00110011
                        # (index_ & 2) * Policy::MmaIterations::kCount / 2:
                        # 00220022
                        # (mma_n & 1) * 2: 02020202
                        # (mma_n & 2) * 2: 00220022

                        mma_idx = ((self.index_ & 1) + (self.index_ & 2) *
                                   self.params.mma_iters[:2].prod() // 2 +
                                   (tile_n * self.params.mma_tile_iters[0]) *
                                   self.params.mma_iters[:2].prod() +
                                   (mma_n & 1) * 2)

                        reg_offset = reg_row * kRegsPerMmaRow + (mma_n & 2) * 2
                        reg_idx = mma_idx * self.params.element_per_mma + reg_offset
                        frag_ptr[idx] = self.src_ptr_[reg_idx //
                                                      self.element_per_acc]

                        # frag_ptr += 1
                        idx += 1

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        ++index_;
        return *this;
        """).ret(f"{self.class_name}&")
        return code

    def increment_python(self):
        self.index_ += 1
        return self


class OutWarpTileIteratorVolta(bases.GemmOutWarpIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int]):
        self.params = VoltaMmaOutParams(dtype, warp_tile_shape)
        element_count = self.params.element_per_acc * self.params.acc_per_interleave_tile * self.params.mma_tile_iters[
            1]
        super().__init__(dtype, element_count, self.params.element_per_acc, -1)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)

        self.tile_shape = tile_shape
        self.warp_tile_shape = warp_tile_shape

        self.padding = metaseq(0, self.element_per_acc)
        self.stride_in_access = (tile_shape[1] +
                                 self.padding[1]) // self.element_per_acc
        self.lanes_in_quad = 4
        self.rows_in_quad = 4
        self.rows_per_quad = 4
        self.column_per_quad = 8
        self.accesses_per_quad = self.column_per_quad // self.element_per_acc
        self.access_quad_delta = 16

        self.add_member("pointer_", self.access_pointer)
        # cudasim members
        self.pointer_: Optional[ArrayPtr] = None

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        if not self.params.is_float_acc:
            code.raw(f"""
            int quad_id = lane_idx / {self.lanes_in_quad}; // 0000 1111 2222 3333 ....
            int lane_in_quad = (lane_idx % {self.lanes_in_quad});

            int quad_row_idx = ((quad_id & 4) >> 1) + (quad_id & 1); //
            int quad_col_idx = ((quad_id & 2) >> 1);

            int row = quad_row_idx * {self.rows_per_quad} + lane_in_quad;
            int column = quad_col_idx * {self.column_per_quad} / {self.element_per_acc};
            pointer_ = reinterpret_cast<{self.access_pointer}>(ptr) + row * {self.stride_in_access} +
                        column;
            """)
        else:
            code.raw(f"""
            int quad_id = lane_idx / {self.lanes_in_quad};
            int lane_in_quad = (lane_idx % {self.lanes_in_quad});

            int const kQuadRowDelta = 4;
            int const kQuadColumnDelta = 2 * {self.params.mma_iters[1]};

            int quad_row_offset = ((quad_id & 4) / 2 + (quad_id & 1)) * kQuadRowDelta;
            int quad_column_offset = (quad_id & 2) / 2 * kQuadColumnDelta;

            int thread_row_offset = (lane_in_quad & 1);
            int thread_column_offset = (lane_in_quad & 2) / 2;

            int row = quad_row_offset + thread_row_offset;
            int column = quad_column_offset + thread_column_offset;
            pointer_ = reinterpret_cast<{self.access_pointer}>(ptr) + row * {self.stride_in_access} +
                        column;
            """)

        code.raw(f"""
        add_warp_offset(warp_offset_m, warp_offset_n);
        """)
        code.arg("ptr", self.pointer)
        code.arg("warp_offset_m,warp_offset_n,lane_idx", "int")
        return code

    def python_ctor(self, ptr: ArrayPtr, warp_offset_m: int,
                    warp_offset_n: int, lane_idx: int):
        new_obj = OutWarpTileIteratorVolta(self.dtype, self.tile_shape,
                                           self.warp_tile_shape)
        if not self.params.is_float_acc:
            quad_id = lane_idx // self.lanes_in_quad  # 0000 1111 2222 3333 ....
            lane_in_quad = (lane_idx % self.lanes_in_quad)

            quad_row_idx = ((quad_id & 4) >> 1) + (quad_id & 1)
            quad_col_idx = ((quad_id & 2) >> 1)

            row = quad_row_idx * self.rows_per_quad + lane_in_quad
            column = quad_col_idx * self.column_per_quad // self.element_per_acc
            new_obj.pointer_ = ptr.change_access_size(
                self.element_per_acc) + row * self.stride_in_access + column
        else:
            quad_id = lane_idx // self.lanes_in_quad
            lane_in_quad = lane_idx % self.lanes_in_quad

            kQuadRowDelta = 4
            kQuadColumnDelta = 2 * self.params.mma_iters[1]

            quad_row_offset = ((quad_id & 4) // 2 +
                               (quad_id & 1)) * kQuadRowDelta
            quad_column_offset = (quad_id & 2) // 2 * kQuadColumnDelta

            thread_row_offset = (lane_in_quad & 1)
            thread_column_offset = (lane_in_quad & 2) // 2

            row = quad_row_offset + thread_row_offset
            column = quad_column_offset + thread_column_offset
            new_obj.pointer_ = ptr.change_access_size(
                self.element_per_acc) + row * self.stride_in_access + column
        new_obj.add_warp_offset_python(warp_offset_m, warp_offset_n)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_warp_offset(self):
        code = pccm.FunctionCode(f"""
        pointer_ +=
            (warp_m * {self.params.inst_shape[0]} * ({self.tile_shape[1]} + {self.padding[1]}) + warp_n * {self.warp_tile_shape[1]}) /
            {self.element_per_acc};
        """)
        return code.arg("warp_m, warp_n", "int")

    def add_warp_offset_python(self, warp_m, warp_n):
        self.pointer_ += (warp_m * self.params.inst_shape[0] *
                          (self.tile_shape[1] + self.padding[1]) + warp_n *
                          self.warp_tile_shape[1]) // self.element_per_acc

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(
            f"{self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);"
        )
        if self.params.is_float_acc:
            code.raw(f"""
            int const kAccessesPerRow = {self.params.mma_tile_iters[1]} * {self.params.mma_iters[1]}  * 2;

            TV_PRAGMA_UNROLL
            for (int row_idx = 0; row_idx < {self.params.rows_per_mma_tile}; ++row_idx) {{

                TV_PRAGMA_UNROLL
                for (int access_idx = 0; access_idx < kAccessesPerRow; ++access_idx) {{
                    int frag_idx = row_idx * kAccessesPerRow + access_idx;

                    int ptr_column_offset = (access_idx & 1) * 2 +
                                            (access_idx & 2) * {self.params.mma_iters[1]}  * 2 +
                                            (access_idx & 4) * {self.params.mma_iters[1]}  * 2;

                    int ptr_row_offset = row_idx * 2;

                    int ptr_offset = ptr_row_offset * {self.stride_in_access} +
                                    ptr_column_offset +
                                    pointer_offset / {self.element_per_acc};

                    pointer_[ptr_offset] = frag_ptr[frag_idx];
                }}
            }}
            """)
        else:
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int tile_idx = 0; tile_idx < {self.params.mma_tile_iters[1]}; ++tile_idx) {{
                TV_PRAGMA_UNROLL
                for (int access_idx = 0; access_idx < {self.params.acc_per_interleave_tile}; ++access_idx) {{
                    int access_quad = access_idx / 2;
                    int access = access_idx % 2;
                    int ptr_offset =
                        tile_idx * {self.params.interleaved_wmma_shape[1]} / {self.element_per_acc} +
                        access_quad * {self.access_quad_delta} / {self.element_per_acc} + access +
                        pointer_offset / {self.element_per_acc};

                    int frag_idx = tile_idx * {self.params.acc_per_interleave_tile} + access_idx;
                    pointer_[ptr_offset] = frag_ptr[frag_idx];
                }}
            }}
            """)
        code.arg("frag", f"{self.fragment_t} const &")
        code.arg("pointer_offset", str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        if self.params.is_float_acc:
            kAccessesPerRow = self.params.mma_tile_iters[
                1] * self.params.mma_iters[1] * 2
            for row_idx in range(self.params.rows_per_mma_tile):

                for access_idx in range(kAccessesPerRow):
                    frag_idx = row_idx * kAccessesPerRow + access_idx

                    ptr_column_offset = (
                        (access_idx & 1) * 2 +
                        (access_idx & 2) * self.params.mma_iters[1] * 2 +
                        (access_idx & 4) * self.params.mma_iters[1] * 2)

                    ptr_row_offset = row_idx * 2

                    ptr_offset = (ptr_row_offset * self.stride_in_access +
                                  ptr_column_offset +
                                  pointer_offset // self.element_per_acc)

                    self.pointer_[ptr_offset] = frag_ptr[frag_idx]
                    ptr_addrs[frag_idx * frag_ptr.access_size:(frag_idx + 1) *
                              frag_ptr.access_size] = np.arange(
                                  (self.pointer_ + ptr_offset).offset,
                                  (self.pointer_ + ptr_offset).offset +
                                  frag_ptr.access_size)

        else:
            for tile_idx in range(self.params.mma_tile_iters[1]):
                for access_idx in range(self.params.acc_per_interleave_tile):
                    access_quad = access_idx // 2
                    access = access_idx % 2
                    ptr_offset = (
                        tile_idx * self.params.interleaved_wmma_shape[1] //
                        self.element_per_acc + access_quad *
                        self.access_quad_delta // self.element_per_acc +
                        access + pointer_offset // self.element_per_acc)

                    frag_idx = tile_idx * self.params.acc_per_interleave_tile + access_idx
                    self.pointer_[ptr_offset] = frag_ptr[frag_idx]
                    ptr_addrs[frag_idx * frag_ptr.access_size:(frag_idx + 1) *
                              frag_ptr.access_size] = np.arange(
                                  (self.pointer_ + ptr_offset).offset,
                                  (self.pointer_ + ptr_offset).offset +
                                  frag_ptr.access_size)
        return ptr_addrs

    async def store_python(self, frag: ArrayPtr):
        return await self.store_with_pointer_offset_python(frag, 0)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        code = pccm.FunctionCode(f"""
        store_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} const&")
        return code

    def add_pointer_offset_python(self, pointer_offset: int):
        self.pointer_ += pointer_offset // self.element_per_acc

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        pointer_ += pointer_offset / {self.element_per_acc};
        """)
        code.arg("pointer_offset", f"int")
        return code
