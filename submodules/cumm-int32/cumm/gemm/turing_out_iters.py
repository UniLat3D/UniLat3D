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

from typing import List, Optional, Union

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers
from cumm.gemm import bases, constants, layout, thread_map
from cumm.gemm.core import MetaArray, array_type, metaseq, seq


def div_up(a, b):
    return (a + b - 1) // b


class TensorOpOutParams(object):
    def __init__(self, dtype: dtypes.DType, warp_tile_shape: MetaArray[int],
                 wmma_shape: MetaArray[int]):
        self.warp_tile_shape = warp_tile_shape
        self.wmma_shape = wmma_shape
        self.mma_count = warp_tile_shape // wmma_shape
        self.element_per_acc = 2
        self.rows_per_iteration = 8
        self.iter_per_inst = wmma_shape[0] // self.rows_per_iteration
        self.num_iters = self.mma_count[0] * self.iter_per_inst
        self.acc_row_stride = self.element_per_acc
        self.acc_col_stride = self.element_per_acc * self.mma_count[
            0] * self.iter_per_inst


class OutFragIterTensorOp(bases.GemmOutFragIterator):
    def __init__(self, dtype: dtypes.DType, warp_tile_shape: MetaArray[int],
                 wmma_shape: MetaArray[int]):
        self.params = TensorOpOutParams(dtype, warp_tile_shape, wmma_shape)
        element_count = self.params.mma_count[1] * self.params.element_per_acc

        super().__init__(dtype, element_count, self.params.element_per_acc,
                         dtype.bitsize() * self.params.element_per_acc // 8)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.element_per_acc = self.params.element_per_acc
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
        new_obj = OutFragIterTensorOp(self.dtype, self.warp_tile_shape,
                                      self.params.wmma_shape)
        new_obj.index_ = 0
        new_obj.src_ptr_ = src_ptr.change_access_size(self.element_per_acc)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        # print(self.params.mma_count[1], self.params.acc_col_stride, self.params.element_per_acc)
        # raise NotImplementedError
        code = pccm.FunctionCode(
            f"{self.access_pointer} frag_ptr = reinterpret_cast<{self.access_pointer}>(&frag);"
        )
        code.raw(f"""
        int index = index_ + index_offset;
        TV_PRAGMA_UNROLL
        for (int n = 0; n < {self.params.mma_count[1]}; ++n) {{
            int accumulator_access_offset = 
                index + n * {self.params.acc_col_stride} / {self.element_per_acc};
            frag_ptr[n] = src_ptr_[accumulator_access_offset];
        }}
        """)
        code.arg("frag", f"{self.fragment_t} &")
        code.arg("index_offset", str(self.index_t), "0")
        return code

    def load_python(self, frag: ArrayPtr, index_offset: int = 0):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        index = self.index_ + index_offset
        for n in range(self.params.mma_count[1]):
            accumulator_access_offset = (
                index + n * self.params.acc_col_stride // self.element_per_acc)
            frag_ptr[n] = self.src_ptr_[accumulator_access_offset]

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


class OutWarpTileIteratorTensorOp(bases.GemmOutWarpIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int], wmma_shape: MetaArray[int]):
        self.params = TensorOpOutParams(dtype, warp_tile_shape, wmma_shape)
        element_count = self.params.mma_count[1] * self.params.element_per_acc

        super().__init__(dtype, element_count, self.params.element_per_acc,
                         dtype.bitsize() * self.params.element_per_acc // 8)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel, layout.RowMajor)

        self.tile_shape = tile_shape
        self.warp_tile_shape = warp_tile_shape
        self.wmma_shape = wmma_shape
        self.shape = metaseq(warp_tile_shape[1],
                             self.params.rows_per_iteration)
        self.lanes_in_quad = 4
        # self.padding = seq(0, self.lanes_in_quad * self.element_per_acc)
        # self.padding = metaseq(0, self.lanes_in_quad * self.params.element_per_acc)
        self.padding = metaseq(
            0, self.lanes_in_quad * self.params.element_per_acc)

        self.stride_in_access = (tile_shape[1] +
                                 self.padding[1]) // self.element_per_acc
        # print(tile_shape, padding, self.stride_in_access, self.element_per_acc)
        # raise NotImplementedError
        self.add_member("pointer_", self.access_pointer)
        # self.add_member("pointer_bkp_", self.access_pointer)

        self.add_member("layout_", "RowMajor")
        # cudasim members
        self.pointer_: Optional[ArrayPtr] = None
        self.layout_: Optional[layout.RowMajor] = None

    def __repr__(self):
        return (f"{self.class_name}[shape={self.shape}|"
                f"mc={self.params.mma_count}]")

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_offset_m,warp_offset_n,lane_idx", "int")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.access_pointer}>(ptr)")
        code.ctor_init("layout_", f"{self.stride_in_access}")
        code.raw(f"""
        // pointer_bkp_ = reinterpret_cast<{self.access_pointer}>(ptr);
        int quad_id = (lane_idx / {self.lanes_in_quad}); 
        int lane_in_quad = (lane_idx % {self.lanes_in_quad});
        pointer_ += layout_(quad_id, lane_in_quad);
        add_warp_offset(warp_offset_m, warp_offset_n);
        // tv::printf2_block_once(threadIdx.x, pointer_ - reinterpret_cast<{self.access_pointer}>(ptr));

        """)
        return code

    def python_ctor(self, ptr: ArrayPtr, warp_offset_m: int,
                    warp_offset_n: int, lane_idx: int):
        new_obj = OutWarpTileIteratorTensorOp(self.dtype, self.tile_shape,
                                              self.warp_tile_shape,
                                              self.wmma_shape)
        quad_id = lane_idx // self.lanes_in_quad  # 0000 1111 2222 3333 ....
        lane_in_quad = (lane_idx % self.lanes_in_quad)
        new_obj.layout_ = layout.RowMajor().python_ctor(self.stride_in_access)
        new_obj.pointer_ = ptr.change_access_size(
            self.element_per_acc) + new_obj.layout_(quad_id, lane_in_quad)
        new_obj.add_warp_offset_python(warp_offset_m, warp_offset_n)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_warp_offset(self):
        code = pccm.FunctionCode(f"""
        pointer_ += layout_(warp_m * {self.params.rows_per_iteration}, warp_n * 
            {self.warp_tile_shape[1]} / {self.element_per_acc});
        """)
        return code.arg("warp_m, warp_n", "int")

    def add_warp_offset_python(self, warp_m, warp_n):
        self.pointer_ += self.layout_(
            warp_m * self.params.rows_per_iteration,
            warp_n * self.warp_tile_shape[1] // self.element_per_acc)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(
            f"{self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);"
        )
        # print(self.params.mma_count[1], self.lanes_in_quad, self.element_per_acc)
        # raise NotImplementedError
        code.raw(f"""
        // tv::printf2_block_once(threadIdx.x, pointer_ - pointer_bkp_);
        TV_PRAGMA_UNROLL
        for (int n = 0; n < {self.params.mma_count[1]}; ++n) {{
            pointer_[n * {self.lanes_in_quad} + pointer_offset / {self.element_per_acc}] = frag_ptr[n];
        }}
        """)
        code.arg("frag", f"{self.fragment_t} const &")
        code.arg("pointer_offset", str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        # print(cudasim.threadIdx().x, self.pointer_.offset // self.element_per_acc)
        ptr_addrs = np.full((frag.length, ), -1, dtype=np.int32)
        frag_ptr = frag.change_access_size(self.element_per_acc)
        for n in range(self.params.mma_count[1]):
            self.pointer_[n * self.lanes_in_quad +
                          pointer_offset // self.element_per_acc] = frag_ptr[n]
            await checkers.smem_bank_conflicit_check(
                self.pointer_, n * self.lanes_in_quad +
                pointer_offset // self.element_per_acc)
            access_offset = n * self.lanes_in_quad + pointer_offset // self.element_per_acc
            if n == 0:
                ptr_addrs[n * frag_ptr.access_size:(n + 1) *
                          frag_ptr.access_size] = np.arange(
                              (self.pointer_ + access_offset).offset,
                              (self.pointer_ + access_offset).offset +
                              frag_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        code = pccm.FunctionCode(f"""
        store_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} const&")
        return code

    async def store_python(self, frag: ArrayPtr):
        return await self.store_with_pointer_offset_python(frag, 0)

    def add_pointer_offset_python(self, pointer_offset: int):
        self.pointer_ += pointer_offset // self.element_per_acc

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        pointer_ += pointer_offset / {self.element_per_acc};
        """)
        code.arg("pointer_offset", f"int")
        return code


class OutWarpTileIteratorTensorOpMixed(bases.GemmOutWarpIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int], wmma_shape: MetaArray[int],
                 out_count: int, contig_lanes: int):
        self.params = TensorOpOutParams(dtype, warp_tile_shape, wmma_shape)
        element_count = self.params.mma_count[1] * self.params.element_per_acc

        super().__init__(dtype, element_count, self.params.element_per_acc,
                         dtype.bitsize() * self.params.element_per_acc // 8)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel, layout.RowMajor)
        self.out_count = out_count
        self.contig_lanes = contig_lanes
        self.spec_s32_168 = dtype == dtypes.int32 and out_count == 16 and tile_shape[
            1] == 128
        self.spec_s32_88 = dtype == dtypes.int32 and out_count == 8 and tile_shape[
            1] == 64
        self.tile_shape = tile_shape
        self.warp_tile_shape = warp_tile_shape
        self.wmma_shape = wmma_shape
        self.shape = metaseq(warp_tile_shape[1],
                             self.params.rows_per_iteration)

        self.offset_count = 4
        if self.spec_s32_168 or self.spec_s32_88:
            self.pointer_count = 2
            assert self.params.mma_count[1] <= 8
        else:
            self.pointer_count = out_count * dtype.bitsize() // (min(
                128, out_count * dtype.bitsize()))
        assert self.pointer_count <= 4
        assert dtype.bitsize() == 32
        self.lanes_in_quad = 4
        # self.padding = seq(0, self.lanes_in_quad * self.element_per_acc)
        self.padding = metaseq(
            0, self.lanes_in_quad * self.params.element_per_acc)
        self.stride_in_access = (tile_shape[1] +
                                 self.padding[1]) // self.element_per_acc
        # print(tile_shape, padding, self.stride_in_access, self.element_per_acc)
        self.add_member("pointers_",
                        self.access_pointer,
                        array=f"[{self.pointer_count}]")
        if cudasim.enable_debug():
            self.add_member("smem_pointer_", self.const_pointer)

        self.add_member("layout_", "RowMajor")
        self.add_member("warp_column_", "int")
        if self.spec_s32_168:
            self.add_member("uniform_offset_",
                            "int",
                            array=f"[{self.offset_count}]")
        # cudasim members
        self.pointers_: List[Union[ArrayPtr,
                                   None]] = [None] * self.pointer_count
        self.layout_: Optional[layout.RowMajor] = None
        self.warp_column_: int = 0
        self.uniform_offset_: List[int] = [0] * self.offset_count

    def __repr__(self):
        return (f"{self.class_name}[shape={self.shape}|"
                f"mc={self.params.mma_count}]")

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_offset_m,warp_offset_n,lane_idx", "int")
        code.ctor_init("layout_", f"{self.stride_in_access}")
        code.ctor_init("warp_column_", f"0")
        if cudasim.enable_debug():
            code.ctor_init("smem_pointer_", f"ptr")

        code.raw(f"""
        int tensorop_row = lane_idx / {self.lanes_in_quad};
        int tensorop_col = lane_idx % {self.lanes_in_quad};
        auto pointer = reinterpret_cast<{self.access_pointer}>(ptr) + tensorop_row * {self.stride_in_access};
        """)
        with code.range_("i", self.pointer_count, "TV_PRAGMA_UNROLL"):
            if not self.spec_s32_168 and not self.spec_s32_88:
                code.raw(f"""
                // int swizzled_tensorop_col = (tensorop_col % 2) + ((
                //     (tensorop_col / 2) + i) % {self.pointer_count}) * 2
                int swizzled_tensorop_col = tensorop_col ^ (i * 2);
                """)
            else:
                code.raw(f"""
                int swizzled_tensorop_col = tensorop_col ^ (i * 2);
                """)
            code.raw(f"""
            pointers_[i] = pointer + swizzled_tensorop_col;
            """)
        if self.spec_s32_168:
            for i in range(self.offset_count):
                code.raw(f"uniform_offset_[{i}] = {i * 4};")
        code.raw(f"add_warp_offset(warp_offset_m, warp_offset_n);")
        return code

    def python_ctor(self, ptr: ArrayPtr, warp_offset_m: int,
                    warp_offset_n: int, lane_idx: int):
        new_obj = OutWarpTileIteratorTensorOpMixed(self.dtype, self.tile_shape,
                                                   self.warp_tile_shape,
                                                   self.wmma_shape,
                                                   self.out_count,
                                                   self.contig_lanes)
        tensorop_row = lane_idx // self.lanes_in_quad  # 0000 1111 2222 3333 ....
        tensorop_col = (lane_idx % self.lanes_in_quad)
        new_obj.layout_ = layout.RowMajor().python_ctor(self.stride_in_access)
        for i in range(self.pointer_count):
            pointer = ptr.change_access_size(
                self.element_per_acc) + tensorop_row * self.stride_in_access
            if not self.spec_s32_168 and not self.spec_s32_88:
                # use 2301 for pointer 1, else 0123
                # when tensorop_row == 1357, swizzle 0123 to 2301
                swizzled_tensorop_col = (tensorop_col % 2) + ((
                    (tensorop_col // 2) + i) % self.pointer_count) * 2
                swizzled_tensorop_col = tensorop_col ^ (i * 2)
            else:
                # for shared load, we need to save row1 to 2301
                # row2/row3 we need to swap tensorop part.
                swizzled_tensorop_col = tensorop_col ^ (i * 2)
            new_obj.pointers_[i] = pointer + swizzled_tensorop_col
        if self.spec_s32_168:
            for i in range(self.offset_count):
                new_obj.uniform_offset_[i] = i * 4

        new_obj.add_warp_offset_python(warp_offset_m, warp_offset_n)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_warp_offset(self):
        code = pccm.code()
        for i in range(self.pointer_count):
            code.raw(f"""
            auto offset_{i} = layout_(
                warp_m * {self.params.rows_per_iteration},
                warp_n * {self.warp_tile_shape[1]} / {self.element_per_acc});
            pointers_[{i}] += offset_{i};
            // tv::printf2_once<' ', 234>("OFFSET", {i}, offset_{i});
            """)
        if self.spec_s32_88:
            # TODO why swap ptr here?
            code.raw(f"""
            if (warp_n % 2 == 1){{
                auto tmp = pointers_[0];
                pointers_[0] = pointers_[1];
                pointers_[1] = tmp;
            }}
            """)
        code.raw(f"""
        warp_column_ += warp_n * {self.warp_tile_shape[1]};
        """)
        if self.spec_s32_168:
            for i in range(self.offset_count):
                # 1032 if warp_n == 1
                # why swizzle here?
                # because warp_n handle smem line 1357
                code.raw(f"""
                uniform_offset_[{i}] = ({i} ^ warp_n) * 4;
                """)
        return code.arg("warp_m, warp_n", "int")

    def add_warp_offset_python(self, warp_m: int, warp_n: int):
        offset = self.layout_(
            warp_m * self.params.rows_per_iteration,
            warp_n * self.warp_tile_shape[1] // self.element_per_acc)
        # if cudasim.threadIdx().x == 234:
        #     print("PFFSET", offset)
        for i in range(self.pointer_count):
            self.pointers_[i] += offset
        if self.spec_s32_88:
            # TODO why swap ptr here?
            if warp_n % 2 == 1:
                tmp = self.pointers_[0]
                self.pointers_[0] = self.pointers_[1]
                self.pointers_[1] = tmp

        self.warp_column_ += warp_n * self.warp_tile_shape[1]
        if self.spec_s32_168:
            for i in range(self.offset_count):
                # 1032 if warp_n == 1
                # why swizzle here?
                # because warp_n handle smem line 1357
                self.uniform_offset_[i] = (i ^ warp_n) * 4

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(
            f"{self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);"
        )
        with code.range_("n", self.params.mma_count[1], "TV_PRAGMA_UNROLL"):
            if not self.spec_s32_168 and not self.spec_s32_88:
                code.raw(f"""
                int column_idx = warp_column_ + n * {self.lanes_in_quad * self.element_per_acc};
                int smem_line_offset = column_idx * {self.dtype.itemsize()} / 128;
                int ptr_idx = smem_line_offset % {self.pointer_count};
                auto ptr = pointers_[ptr_idx];
                int offset = n * {self.lanes_in_quad} + pointer_offset / {self.element_per_acc};
                ptr[offset] = frag_ptr[n];
                """)
            elif self.spec_s32_168:
                code.raw(f"""
                int ptr_idx = n / 4;
                int offset_idx = n % 4;
                auto ptr = pointers_[ptr_idx];
                int offset = ( n / 4) * 16 + 
                    pointer_offset / {self.element_per_acc} + 
                    uniform_offset_[offset_idx];
                ptr[offset] = frag_ptr[n];
                """)
            else:
                code.raw(f"""
                int ptr_idx = n / 4;
                auto ptr = pointers_[ptr_idx];
                int offset = (n /
                          4) * 16 + pointer_offset / {self.element_per_acc} + (
                              n % 4) * 4;
                ptr[offset] = frag_ptr[n];
                """)
        code.arg("frag", f"{self.fragment_t} const &")
        code.arg("pointer_offset", str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        # print(cudasim.threadIdx().x, self.pointer_.offset // self.element_per_acc)
        ptr_addrs = np.full((frag.length, ), -1, dtype=np.int32)
        frag_ptr = frag.change_access_size(self.element_per_acc)
        for n in range(self.params.mma_count[1]):
            if not self.spec_s32_168 and not self.spec_s32_88:
                column_idx = self.warp_column_ + n * self.lanes_in_quad * self.element_per_acc
                smem_line_offset = column_idx * self.dtype.itemsize() // 128
                # if smem_line_offset == 1, use second (swizzled) pointer.
                ptr_idx = smem_line_offset % self.pointer_count
                ptr = self.pointers_[ptr_idx]
                offset = n * self.lanes_in_quad + pointer_offset // self.element_per_acc
                ptr[offset] = frag_ptr[n]
                await checkers.smem_bank_conflicit_check(ptr, offset)
            elif self.spec_s32_168:
                ptr_idx = n // 4
                offset_idx = n % 4
                # use second ptr for smem line 1357...
                # suck code
                assert ptr_idx < 2
                ptr = self.pointers_[ptr_idx]
                # uniform_offset_: 0 4 8 12, treat them as internal loop of 4.
                offset = (
                    n // 4
                ) * 16 + pointer_offset // self.element_per_acc + self.uniform_offset_[
                    offset_idx]
                ptr[offset] = frag_ptr[n]
                await checkers.smem_bank_conflicit_check(ptr, offset)
            else:
                ptr_idx = n // 4
                assert ptr_idx < 2
                ptr = self.pointers_[ptr_idx]
                offset = (n //
                          4) * 16 + pointer_offset // self.element_per_acc + (
                              n % 4) * 4
                ptr[offset] = frag_ptr[n]
                await checkers.smem_bank_conflicit_check(ptr, offset)
            if n % 4 == 0:
                ptr_addrs[n * frag_ptr.access_size:(n + 1) *
                          frag_ptr.access_size] = np.arange(
                              (ptr + offset).offset,
                              (ptr + offset).offset + frag_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        code = pccm.FunctionCode(f"""
        store_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} const&")
        return code

    async def store_python(self, frag: ArrayPtr):
        return await self.store_with_pointer_offset_python(frag, 0)

    def add_pointer_offset_python(self, pointer_offset: int):
        for i in range(self.pointer_count):
            self.pointers_[i] += pointer_offset // self.element_per_acc

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.code()
        code.arg("pointer_offset", f"int")
        for i in range(self.pointer_count):
            code.raw(
                f"pointers_[{i}] += pointer_offset / {self.element_per_acc};")
        return code


class OutSmemLoaderMixed(bases.GemmOutSmemLoader):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 tmap: thread_map.Out5DLinear, access_length: int, stride: int,
                 max_alignment: int, contig_lanes: int):
        element_count = tmap.iterations.prod() * access_length
        assert element_count != 0, str(tmap.iterations)
        num_sub_access = min(128 // dtype.bitsize(), access_length)
        min_alignment = access_length * dtype.itemsize()
        self.max_alignment = max_alignment
        self.alignment = max_alignment if max_alignment < min_alignment else min_alignment

        super().__init__(dtype, element_count, num_sub_access,
                         min(16, self.alignment))
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.tile_shape = tile_shape
        self.spec_s32_168 = dtype == dtypes.int32 and access_length == 16 and tile_shape[
            1] == 128
        self.spec_s32_88 = dtype == dtypes.int32 and access_length == 8 and tile_shape[
            1] == 64
        self.stride = stride
        self.contig_lanes = contig_lanes
        self.tmap = tmap
        self.iterations = tmap.iterations  # type: MetaArray[int]
        self.delta = tmap.delta  # type: MetaArray[int]
        self.element_per_acc_output = access_length
        self.stride_vec = stride // num_sub_access

        self.loads_per_access = self.element_per_acc_output // num_sub_access
        self.num_sub_access = num_sub_access
        self.add_member("pointers_",
                        self.const_access_pointer,
                        array=f"[{self.loads_per_access}]")
        if cudasim.enable_debug():
            self.add_member("smem_pointer_", self.const_access_pointer)

        if self.spec_s32_168 or self.spec_s32_88:
            assert self.iterations[4] == 1
        # cudasim members
        self.pointers_: List[Optional[ArrayPtr]] = [None
                                                    ] * self.loads_per_access
        self.smem_pointer_: Optional[ArrayPtr] = None

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        auto thread_offset = ThreadMap::initial_offset(thread_idx);
        auto pointer = reinterpret_cast<{self.const_access_pointer}>(ptr);
        """)
        code.arg("ptr", self.pointer)
        code.arg("thread_idx", "int")
        if cudasim.enable_debug():
            code.ctor_init(
                "smem_pointer_",
                f"reinterpret_cast<{self.const_access_pointer}>(ptr)")

        with code.range_("i", self.loads_per_access, "TV_PRAGMA_UNROLL"):
            if self.spec_s32_168:
                code.raw(f"""
                int lane_col_idx = thread_offset[1] / {self.element_per_acc_output};
                int lane_offset = (lane_col_idx << 2) | ((lane_col_idx / 2) ^ i);
                pointers_[i] = pointer + thread_offset[0] * {self.stride_vec} + lane_offset;
                """)
            elif self.spec_s32_88:
                code.raw(f"""
                int lane_col_idx = thread_offset[1] / {self.element_per_acc_output};
                int lane_offset = ((lane_col_idx % 8) * 2) | ((lane_col_idx / 4) ^ i);
                pointers_[i] = pointer + thread_offset[0] * {self.stride_vec} + lane_offset;
                """)
            else:
                code.raw(f"""
                int col_idx_in_subacc = (thread_offset[1] / {self.element_per_acc_output}) * {self.loads_per_access};
                int smem_line_offset = (col_idx_in_subacc * {self.num_sub_access} * {self.dtype.itemsize()} / 128) % {self.loads_per_access};
                col_idx_in_subacc += (smem_line_offset + i) % {self.loads_per_access};
                // tv::printf2_once<' ', {cudasim.debug_tx()}>(i, threadIdx.x, "col_idx_in_subacc", thread_offset[0], thread_offset[1], col_idx_in_subacc, thread_offset[0] * {self.stride_vec} + col_idx_in_subacc);
                pointers_[i] = pointer + thread_offset[0] * {self.stride_vec} + col_idx_in_subacc;
                """)
        return code

    def python_ctor(self, ptr: ArrayPtr, thread_idx: int):
        new_obj = OutSmemLoaderMixed(self.dtype, self.tile_shape, self.tmap,
                                     self.element_per_acc_output, self.stride,
                                     self.max_alignment, self.contig_lanes)
        thread_offset = new_obj.tmap.initial_offset_python(thread_idx)
        for i in range(self.loads_per_access):
            pointer = ptr.change_access_size(self.num_sub_access)
            if self.spec_s32_168:
                lane_col_idx = thread_offset[1] // self.element_per_acc_output
                lane_offset = (lane_col_idx << 2) | ((lane_col_idx // 2) ^ i)
                new_obj.pointers_[i] = pointer + thread_offset[
                    0] * new_obj.stride_vec + lane_offset
            elif self.spec_s32_88:
                lane_col_idx = thread_offset[1] // self.element_per_acc_output
                # 0246 1357
                # lane_offset = ((lane_col_idx & 0b111) << 1) + ((lane_col_idx >> 3) ^ i)
                lane_offset = ((lane_col_idx % 8) * 2) | (
                    (lane_col_idx // 4) ^ i)
                new_obj.pointers_[i] = pointer + thread_offset[
                    0] * new_obj.stride_vec + lane_offset
            else:
                col_idx_in_subacc = (
                    thread_offset[1] //
                    self.element_per_acc_output) * self.loads_per_access
                smem_line_offset = (col_idx_in_subacc * self.num_sub_access *
                                    self.dtype.itemsize() //
                                    128) % self.loads_per_access
                # smem_line_offset: for loads_per_access == 2, smem_line_offset == 01010101....
                # if smem_line_offset > 0, pointers is swizzled by simple shift.
                col_idx_in_subacc += (smem_line_offset +
                                      i) % self.loads_per_access
                new_obj.pointers_[i] = pointer + thread_offset[
                    0] * new_obj.stride_vec + col_idx_in_subacc
                # if cudasim.threadIdx().x == cudasim.debug_tx():
                #     cudasim.debug_print("col_idx_in_subacc", cudasim.threadIdx().x, thread_offset[0], thread_offset[1], thread_offset[0] * new_obj.stride_vec + col_idx_in_subacc, col_idx_in_subacc)
            assert new_obj.pointers_[i].length > 0
        # print(new_obj.pointer_)
        new_obj.smem_pointer_ = ptr.change_access_size(self.num_sub_access)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);

        TV_PRAGMA_UNROLL
        for (int cluster = 0; cluster < {self.iterations[1]}; ++cluster) {{

            TV_PRAGMA_UNROLL
            for (int group = 0; group < {self.iterations[2]}; ++group) {{

                TV_PRAGMA_UNROLL
                for (int row = 0; row < {self.iterations[3]}; ++row) {{
                    int row_ptr_offset = (
                        row * {self.delta[3] * self.stride_vec} +
                        group * {self.delta[2] * self.stride_vec} +
                        cluster * {self.delta[1] * self.stride_vec} +
                        pointer_offset / {self.num_sub_access});
                    int frag_row_idx = (row + {self.iterations[3]} *
                                    (group + {self.iterations[2]} * cluster));
                    TV_PRAGMA_UNROLL
                    for (int column = 0; column < {self.iterations[4]}; ++column) {{
                        int frag_idx = frag_row_idx * {self.iterations[4]} + column;
                        int vector_idx = ((column * {self.delta[4]} /
                                        {self.element_per_acc_output}) *
                                        {self.loads_per_access});
                        TV_PRAGMA_UNROLL
                        for (int v = 0; v < {self.loads_per_access}; ++v) {{
                            auto mem_ptr = pointers_[v] + row_ptr_offset;
                            // tv::printf2_once<' ', {cudasim.debug_tx()}>(cluster, group, row, column, v, frag_idx * {self.loads_per_access} + v, vector_idx, mem_ptr - smem_pointer_);

                            frag_ptr[frag_idx * {self.loads_per_access} +
                                     v] = (mem_ptr[vector_idx]);
                            // tv::print_ptr_once<int, 0, {self.num_sub_access}, {cudasim.debug_tx()}>(reinterpret_cast<const int*>(mem_ptr));
                        }}
                    }}
                }}
            }}
        }}
        """)
        code.arg("frag", f"{self.fragment_t} &").arg("pointer_offset",
                                                     str(self.index_t))
        return code

    async def load_with_pointer_offset_python(self, frag: ArrayPtr,
                                              pointer_offset: int):
        frag_ptr = frag.change_access_size(self.num_sub_access)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)
        for cluster in range(self.iterations[1]):
            for group in range(self.iterations[2]):
                for row in range(self.iterations[3]):
                    row_ptr_offset = (
                        row * self.delta[3] * self.stride_vec +
                        group * self.delta[2] * self.stride_vec +
                        cluster * self.delta[1] * self.stride_vec +
                        pointer_offset // self.num_sub_access)
                    frag_row_idx = (row + self.iterations[3] *
                                    (group + self.iterations[2] * cluster))

                    for column in range(self.iterations[4]):

                        frag_idx = frag_row_idx * self.iterations[4] + column
                        vector_idx = ((column * self.delta[4] //
                                       self.element_per_acc_output) *
                                      self.loads_per_access)
                        for v in range(self.loads_per_access):
                            mem_ptr = self.pointers_[v] + row_ptr_offset

                            frag_ptr[frag_idx * self.loads_per_access +
                                     v] = (mem_ptr[vector_idx])
                            # if cudasim.threadIdx().x == cudasim.debug_tx():
                            #     print(cluster, group, row, column, v, frag_idx * self.loads_per_access + v, vector_idx, mem_ptr.access_offset - self.smem_pointer_.access_offset)
                            #     data = mem_ptr[vector_idx]
                            #     print(data.data.numpy_view(), frag.data.numpy_view())
                            dst_offset = frag_idx * self.loads_per_access + v

                            # await checkers.smem_bank_conflicit_check(
                            #     mem_ptr, vector_idx)
                            if v == 0:
                                ptr_addrs[dst_offset *
                                          frag_ptr.access_size:(dst_offset +
                                                                1) *
                                          frag_ptr.access_size] = np.arange(
                                              (mem_ptr + vector_idx).offset,
                                              (mem_ptr + vector_idx).offset +
                                              frag_ptr.access_size)

        return ptr_addrs

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_pointer_offset_python(frag, 0)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} &")
        return code

    def add_pointer_offset_python(self, pointer_offset: int):
        for i in range(self.loads_per_access):
            self.pointers_[i] += pointer_offset // self.num_sub_access

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.code()
        code.arg("pointer_offset", f"int")
        for i in range(self.loads_per_access):
            code.raw(
                f"pointers_[{i}] += pointer_offset / {self.num_sub_access};")
        return code


def _ci_dev_wtf_mixed():
    for lane_col_idx in range(8):
        for i in range(2):
            # (lane_col_idx % 2) * 4: 0404040404040404
            # ((lane_col_idx // 2) * 8): 0088...
            # first |: 048C 048C
            # ((lane_col_idx // 2) ^ i):
            # lane_col_idx << 2 = 048...
            # i=0: 0011223344556677
            # i=1: 1100332255447766
            # i=2: ...
            lane_offset = ((lane_col_idx % 8) * 2) | ((lane_col_idx // 4) ^ i)
            print(lane_offset)


def _ci_dev_wtf_mixed2():
    num_pointer = 2
    for i in range(num_pointer):
        for lane_col_idx in range(8):

            # (lane_col_idx % 2) * 4: 0404040404040404
            # ((lane_col_idx // 2) * 8): 0088...
            # first |: 048C 048C
            # ((lane_col_idx // 2) ^ i):
            # lane_col_idx << 2 = 048...
            # i=0: 0011223344556677
            # i=1: 1100332255447766
            # i=2: ...
            lane_offset = ((lane_col_idx % 8) * 2) | ((lane_col_idx // 4) ^ i)
            print(lane_offset)
