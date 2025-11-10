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

# from codeai.astex.lineprof import lineprof_wrapper_cpp
from typing import List, Optional, Union

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers
from cumm.gemm import codeops, constants, layout, thread_map
from cumm.gemm.bases import (GemmIterator, GemmOutFragIterator,
                             GemmOutputIterator, GemmOutSmemLoader,
                             GemmOutWarpIterator)
from cumm.gemm.core import (MetaArray, aligned_array_type, array_type, metaseq,
                            seq)


def div_up(a, b):
    return (a + b - 1) // b


_LAYOUT_TYPES = Union[layout.ColumnMajor, layout.RowMajor,
                      layout.ColumnMajorInterleaved,
                      layout.RowMajorInterleaved]


class OutWarpTileIterator(GemmOutWarpIterator):
    def __init__(self,
                 dtype: dtypes.DType,
                 warp_tile_shape: MetaArray[int],
                 warp_shape: MetaArray[int],
                 lane_mma_shape: MetaArray[int],
                 lane_layout: _LAYOUT_TYPES,
                 stride: int,
                 scalar_store: bool = True):
        self.scalar_store = scalar_store
        self.access_length_wmma = lane_mma_shape[1]  # 4
        self.element_per_acc = 1 if scalar_store else self.access_length_wmma

        self.num_iterations = warp_tile_shape[0] // warp_shape[0]
        self.num_element_per_it = warp_tile_shape[1] // warp_shape[1]
        vector_length = warp_tile_shape[1] // warp_shape[1]
        self.num_access_per_it = vector_length // self.access_length_wmma

        super().__init__(dtype,
                         self.num_access_per_it * self.access_length_wmma,
                         self.element_per_acc,
                         dtype.itemsize() * self.element_per_acc)

        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.add_param_class("ns2", lane_layout,
                             "LaneLayout")  # TODO add a real layout class

        self.lane_layout = lane_layout
        self.warp_tile_shape = warp_tile_shape
        self.warp_shape = warp_shape
        self.lane_mma_shape = lane_mma_shape
        self.padding = metaseq(0, 4 * lane_mma_shape[1])
        if scalar_store:
            self.padding[1] += 1
        self.stride_raw = stride
        self.stride = stride + self.padding[1]

        self.add_member("pointer_", self.access_pointer)

        # cudasim members
        self.pointer_: Optional[ArrayPtr] = None

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        constexpr auto lane_layout = LaneLayout::from_shape({{{self.warp_shape[0]}, {self.warp_shape[1]}}});

        tv::array<int, 2> logical_offset{{warp_offset_m, warp_offset_n}};
        // 0, 1, 0, 1, 0, 1, ..., 2, 3, 2, 3, 2, 3
        int lane_offset_0 = lane_layout.inverse_0(lane_idx);
        // 0, 0, 1, 1, 2, 2, 3, 3, ...
        int lane_offset_1 = lane_layout.inverse_1(lane_idx);
        // saved to compacted shared memory, so logical_offset[0] * warp_shape[0],
        // not logical_offset[0] * warp_tile_shape[0]
        pointer_ = reinterpret_cast<{self.access_pointer}>(
            ptr + (logical_offset[0] * {self.warp_shape[0]} + lane_offset_0) * {self.stride} +
            logical_offset[1] * {self.warp_tile_shape[1]} +
            lane_offset_1 * {self.lane_mma_shape[1]});
        """)
        code.arg("ptr", self.pointer)
        code.arg("warp_offset_m,warp_offset_n,lane_idx", "int")
        return code

    def python_ctor(self, ptr: ArrayPtr, warp_offset_m: int,
                    warp_offset_n: int, lane_idx: int):
        new_obj = OutWarpTileIterator(self.dtype, self.warp_tile_shape,
                                      self.warp_shape, self.lane_mma_shape,
                                      self.lane_layout, self.stride_raw,
                                      self.scalar_store)
        lane_layout = new_obj.lane_layout.from_shape_python(
            new_obj.warp_shape[:2])
        logical_offset = seq(warp_offset_m, warp_offset_n)
        lane_offset_0 = lane_layout.inverse_0_python(lane_idx)
        lane_offset_1 = lane_layout.inverse_1_python(lane_idx)
        new_obj.pointer_ = (
            ptr + (logical_offset[0] * new_obj.warp_shape[0] + lane_offset_0) *
            new_obj.stride + logical_offset[1] * new_obj.warp_tile_shape[1] +
            lane_offset_1 * new_obj.lane_mma_shape[1]).change_access_size(
                new_obj.element_per_acc)
        off = ((logical_offset[0] * new_obj.warp_shape[0] + lane_offset_0) *
               new_obj.stride +
               logical_offset[1] * new_obj.warp_tile_shape[1] +
               lane_offset_1 * new_obj.lane_mma_shape[1])
        # new_obj.pointer_ +=
        # new_obj.pointer_ = new_obj.pointer_.change_access_size(
        #     new_obj.element_per_acc)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        // pointer_offset: element unit
        {self.const_access_pointer} dst_ptr =
            reinterpret_cast<{self.const_access_pointer}>(&frag);
        TV_PRAGMA_UNROLL
        for (int acc_idx = 0; acc_idx < {self.num_access_per_it}; ++acc_idx) {{
            if ({pccm.boolean(self.scalar_store)}) {{
                TV_PRAGMA_UNROLL
                for (int s = 0; s < {self.access_length_wmma}; ++s) {{
                    pointer_[acc_idx * {self.warp_shape[1]} * {self.access_length_wmma} + s +
                            pointer_offset] = dst_ptr[acc_idx * {self.access_length_wmma} + s];
                }}
            }} else {{
                pointer_[acc_idx * {self.warp_shape[1]} +
                        pointer_offset / {self.access_length_wmma}] = dst_ptr[acc_idx];
            }}
        }}
        """)
        code.arg("frag",
                 f"{self.fragment_t} const&").arg("pointer_offset",
                                                  str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        dst_ptr = frag.change_access_size(self.element_per_acc)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        for acc_idx in range(self.num_access_per_it):
            if self.scalar_store:
                for s in range(self.access_length_wmma):
                    self.pointer_[acc_idx * self.warp_shape[1] *
                                  self.access_length_wmma + s +
                                  pointer_offset] = dst_ptr[
                                      acc_idx * self.access_length_wmma + s]
            else:
                self.pointer_[acc_idx * self.warp_shape[1] + pointer_offset //
                              self.access_length_wmma] = dst_ptr[acc_idx]
                access_offset = (acc_idx * self.warp_shape[1] +
                                 pointer_offset // self.access_length_wmma)
                ptr_addrs[acc_idx * dst_ptr.access_size:(acc_idx + 1) *
                          dst_ptr.access_size] = np.arange(
                              (self.pointer_ + access_offset).offset,
                              (self.pointer_ + access_offset).offset +
                              dst_ptr.access_size)
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


class OutSmemLoader(GemmOutSmemLoader):
    def __init__(self, dtype: dtypes.DType, tmap: thread_map.Out5DLinear,
                 access_length: int, stride: int, max_alignment: int):
        self.tmap = tmap
        self.iterations = tmap.iterations  # type: MetaArray[int]
        self.delta = tmap.delta  # type: MetaArray[int]
        self.element_per_acc_output = access_length
        min_alignment = self.element_per_acc_output * dtype.itemsize()
        self.max_alignment = max_alignment
        alignment = max_alignment if max_alignment < min_alignment else min_alignment
        element_count = self.iterations.prod() * self.element_per_acc_output
        assert element_count != 0, str(tmap.iterations)
        num_sub_access = min(128 // dtype.bitsize(),
                             self.element_per_acc_output)
        self.loads_per_access = self.element_per_acc_output // num_sub_access
        self.num_sub_access = num_sub_access

        super().__init__(dtype,
                         self.iterations.prod() * self.element_per_acc_output,
                         num_sub_access, min(16, alignment))
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.stride = stride
        self.add_member("pointer_", self.pointer)

        # cudasim members
        self.pointer_ = None  # type: Optional[ArrayPtr]

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        auto thread_offset = ThreadMap::initial_offset(thread_idx);
        pointer_ = ptr + thread_offset[0] * {self.stride} + thread_offset[1];
        """)
        code.arg("ptr", self.pointer)
        code.arg("thread_idx", "int")
        return code

    def python_ctor(self, ptr: ArrayPtr, thread_idx: int):
        new_obj = OutSmemLoader(self.dtype, self.tmap,
                                self.element_per_acc_output, self.stride,
                                self.max_alignment)
        thread_offset = new_obj.tmap.initial_offset_python(thread_idx)
        new_obj.pointer_ = ptr + thread_offset[
            0] * new_obj.stride + thread_offset[1]
        assert new_obj.pointer_.length > 0
        # print(new_obj.pointer_)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int cluster = 0; cluster < {self.iterations[1]}; ++cluster) {{

            TV_PRAGMA_UNROLL
            for (int group = 0; group < {self.iterations[2]}; ++group) {{

                TV_PRAGMA_UNROLL
                for (int row = 0; row < {self.iterations[3]}; ++row) {{

                    {self.const_pointer} cur_pointer =
                        pointer_ + row * {self.delta[3]} * {self.stride} + group * {self.delta[2]} * {self.stride} +
                        cluster * {self.delta[1]} * {self.stride} + pointer_offset;
                    int frag_row_idx =
                        (row + {self.iterations[3]} * (group + {self.iterations[2]} * cluster));

                    {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
                    {self.access_t} const *memory_pointer =
                        reinterpret_cast<{self.access_t} const *>(cur_pointer);

                    TV_PRAGMA_UNROLL
                    for (int column = 0; column < {self.iterations[4]}; ++column) {{

                        int frag_idx = frag_row_idx * {self.iterations[4]} + column;

                        TV_PRAGMA_UNROLL
                        for (int v = 0; v < {self.loads_per_access}; ++v) {{
                            frag_ptr[frag_idx * {self.loads_per_access} + v] =
                                memory_pointer[(column * {self.delta[4]} / {self.element_per_acc_output}) *
                                                    {self.loads_per_access} +
                                                v];
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

                    cur_pointer = (self.pointer_ +
                                   row * self.delta[3] * self.stride +
                                   group * self.delta[2] * self.stride +
                                   cluster * self.delta[1] * self.stride +
                                   pointer_offset)

                    frag_row_idx = (row + self.iterations[3] *
                                    (group + self.iterations[2] * cluster))

                    memory_pointer = cur_pointer.change_access_size(
                        self.num_sub_access)
                    for column in range(self.iterations[4]):

                        frag_idx = frag_row_idx * self.iterations[4] + column

                        for v in range(self.loads_per_access):
                            mem_ptr = (memory_pointer +
                                       (column * self.delta[4] //
                                        self.element_per_acc_output) *
                                       self.loads_per_access + v)
                            frag_ptr[frag_idx * self.loads_per_access +
                                     v] = (mem_ptr[0])
                            dst_offset = frag_idx * self.loads_per_access + v
                            ptr_addrs[dst_offset *
                                      frag_ptr.access_size:(dst_offset + 1) *
                                      frag_ptr.access_size] = np.arange(
                                          mem_ptr.offset, mem_ptr.offset +
                                          frag_ptr.access_size)

                            # await checkers.smem_bank_conflicit_check(mem_ptr, 0)
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
        self.pointer_ += pointer_offset

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        pointer_ += pointer_offset;
        """)
        code.arg("pointer_offset", f"int")
        return code


def _add_out_iter_params_members(c: pccm.Class, long_index_t: str):
    c.add_member("stride", str(long_index_t))
    c.add_member("increment_row", str(long_index_t))
    c.add_member("increment_group", str(long_index_t))
    c.add_member("increment_cluster", str(long_index_t))
    c.add_member("advance_row", str(long_index_t))
    c.add_member("advance_group", str(long_index_t))
    c.add_member("advance_cluster", str(long_index_t))
    c.add_member("advance_tile", str(long_index_t))


class OutIteratorParams(pccm.ParameterizedClass):
    def __init__(self,
                 tmap: thread_map.Out5DLinear,
                 shuffle_in_stride: bool = False,
                 is_bias: bool = False):
        super().__init__()
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.long_index_t = dtypes.int64
        self.shuffle_in_stride = shuffle_in_stride
        if not is_bias:
            _add_out_iter_params_members(self, str(self.long_index_t))
            if self.shuffle_in_stride:
                self.add_member("indice_ptr_", "const int*")
        self.is_bias = is_bias
        #cudasim params
        self.stride = -1
        self.increment_row = 0
        self.increment_group = 0
        self.increment_cluster = 0
        self.advance_row = 0
        self.advance_group = 0
        self.advance_cluster = 0
        self.advance_tile = 0

    def __repr__(self) -> str:
        return f"OutP[s={self.stride}|{self.advance_tile}|{self.advance_cluster}|{self.advance_group}|{self.advance_row}]"

    def python_ctor(self, stride: int):
        new_obj = OutIteratorParams(self.tmap, self.shuffle_in_stride)
        new_obj.stride = stride
        increment_params = new_obj.tmap.iteration_inc_params_python(stride)
        advance_params = new_obj.tmap.iteration_advance_params_python(stride)

        new_obj.increment_cluster = increment_params[0]
        new_obj.increment_group = increment_params[1]
        new_obj.increment_row = increment_params[2]
        new_obj.advance_tile = advance_params[0]
        new_obj.advance_cluster = advance_params[1]
        new_obj.advance_group = advance_params[2]
        new_obj.advance_row = advance_params[3]
        return new_obj

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def default_ctor(self):
        return pccm.code()

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        if self.is_bias:
            return code.make_invalid()
        code.arg("stride_", "int")
        code.ctor_init("stride", "stride_")
        if self.shuffle_in_stride:
            code.arg("indice_ptr", "const int*", "nullptr")
            code.ctor_init("indice_ptr_", "indice_ptr")
        if not self.is_bias:
            code.raw("""
            auto increment_params = ThreadMap::iteration_inc_params(stride);
            auto advance_params = ThreadMap::iteration_advance_params(stride);

            increment_cluster = increment_params[0];
            increment_group = increment_params[1];
            increment_row = increment_params[2];
            advance_tile = advance_params[0];
            advance_cluster = advance_params[1];
            advance_group = advance_params[2];
            advance_row = advance_params[3];
            """)
        return code


class OutIterator(GemmOutputIterator):
    """
    Args:
        with_source: add source pointer to save registers
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 tmap: thread_map.Out5DLinear,
                 param_class: OutIteratorParams,
                 part_shape: MetaArray[int],
                 part_dilation: MetaArray[int],
                 access_length: int,
                 read_only: bool = False,
                 shuffle_in_stride: bool = False,
                 access_per_vector: int = 1,
                 with_source: bool = False):
        self.iterations = tmap.iterations  # type: MetaArray[int]
        self.delta = tmap.delta  # type: MetaArray[int]
        super().__init__(dtype,
                         self.iterations.prod() * access_length, access_length,
                         dtype.itemsize() * access_length, access_per_vector)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.read_only = read_only
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.params = param_class
        self.add_param_class("params", self.params, "Params")
        # shuffle_in_stride = False
        self.shuffle_in_stride = shuffle_in_stride
        self.part_shape = part_shape
        self.part_dilation = part_dilation
        self.tmap = tmap
        # if is_bias, all params in OutIteratorParams is zero.
        if with_source:
            assert not self.params.is_bias, "bias iter isn't related to source"
        self.is_bias = self.params.is_bias
        if read_only:
            self.add_member("pointer_", self.const_pointer)
        else:
            self.add_member("pointer_", self.pointer)
        if with_source:
            self.add_member("source_pointer_", self.const_pointer)
        self.with_source = with_source
        self.add_member("params_", "Params const&")
        self.add_member(
            "column_masks_",
            "bool",
            array=f"[{self.iterations[4]}][{self.access_per_vector}]")
        if not self.is_bias:
            self.add_member("extent_row_, thread_start_row_", str(self.index_t))
            self.add_member("counts_", "int", array="[3]")
        if self.shuffle_in_stride:
            self.add_member("indices_",
                            "int64_t",
                            array=f"[{self.iterations[1:4].prod()}]")

        # cudasim members
        self.pointer_ = None  # type: Optional[ArrayPtr]
        self.params_ = None  # type: Optional[OutIteratorParams]

        self.column_masks_ = [False] * self.iterations[4]
        self.extent_row_ = 0
        self.thread_start_row_ = 0
        self.counts_ = [0] * 3

    def get_params(self):
        return self.params

    def support_both_source_and_target(self) -> bool:
        return self.with_source 
        
    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        // pointer_bkp_ = ptr;
        """)
        if not self.is_bias:
            code.raw(f"""
            counts_[0] = 0;
            counts_[1] = 0;
            counts_[2] = 0;
            """)
        code.raw(f"""
        auto thread_offset = ThreadMap::initial_offset(thread_idx) + offset_2d;
        TV_PRAGMA_UNROLL
        for (int c = 0; c < {self.iterations[4]}; ++c) {{
            for (int v = 0; v < {self.access_per_vector}; ++v){{
                column_masks_[c][v] = ((thread_offset[1] + {self.delta[4]} * c + v * {self.element_per_acc}) < extent[1]);
            }}
        }}
        // tv::printf2_block_once("Outthread_offset ", threadIdx.x, thread_offset[0], thread_offset[1], (thread_offset[0]) * stride + thread_offset[1]);
        """)
        if not self.is_bias:
            code.raw(f"""
            extent_row_ = extent[0];
            thread_start_row_ = thread_offset[0];
            """)
        if self.shuffle_in_stride:
            assert not self.is_bias, "shuffle in stride don't support bias."
            with code.range_("cluster", str(self.iterations[1]),
                             "TV_PRAGMA_UNROLL"):
                with code.range_("group", str(self.iterations[2]),
                                 "TV_PRAGMA_UNROLL"):
                    with code.range_("row", str(self.iterations[3]),
                                     "TV_PRAGMA_UNROLL"):
                        code.raw(f"""
                        int idx = (row +  {self.iterations[3]} * (group +  {self.iterations[2]} * cluster));

                        int row_offset =
                            row * {self.delta[3]} + group * {self.delta[2]} + cluster * {self.delta[1]};
                        bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
                        indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
                        """)
            code.raw(f"""
            pointer_ = ptr + thread_offset[1];
            """)
            if self.with_source:
                code.raw(f"""
                source_pointer_ = source_ptr + thread_offset[1];
                """)
        else:
            if self.is_bias:
                code.raw(f"""
                pointer_ = ptr + thread_offset[1];
                """)
            else:
                code.raw(f"""
                pointer_ = ptr + (thread_offset[0]) * params.stride + thread_offset[1];
                """)
                if self.with_source:
                    code.raw(f"""
                    source_pointer_ = source_ptr + (thread_offset[0]) * params.stride + thread_offset[1];
                    """)

        code.arg("params", "Params const&")
        code.arg("ptr",
                 f"{self.const_pointer if self.read_only else self.pointer}")
        if self.with_source:
            code.arg("source_ptr",
                    f"{self.const_pointer}")

        code.arg("extent, offset_2d", "tv::array<int, 2>")
        code.arg("thread_idx", "int")
        code.ctor_init("params_", "params")
        return code

    def python_ctor(self, params: OutIteratorParams, ptr: ArrayPtr,
                    extent: MetaArray[int], offset_2d: MetaArray[int],
                    thread_idx: int) -> "OutIterator":
        new_obj = OutIterator(self.dtype, self.tmap, self.params,
                              self.part_shape, self.part_dilation,
                              self.element_per_acc, self.read_only,
                              self.shuffle_in_stride)
        new_obj.extent_row_ = extent[0]
        new_obj.counts_[0] = 0
        new_obj.counts_[1] = 0
        new_obj.counts_[2] = 0
        new_obj.params_ = params
        # print(params)
        thread_offset = new_obj.tmap.initial_offset_python(
            thread_idx) + offset_2d
        new_obj.thread_start_row_ = thread_offset[0]
        for c in range(new_obj.iterations[4]):
            new_obj.column_masks_[c] = (
                (thread_offset[1] + new_obj.delta[4] * c) < extent[1])
        # print(cudasim.threadIdx().x, increment_params, advance_params, thread_offset[0], thread_offset[1], (thread_offset[0]) * stride + thread_offset[1])
        new_obj.pointer_ = ptr + (
            thread_offset[0]) * params.stride + thread_offset[1]
        return new_obj

    def load_store_with_offset_template(self, store: bool):
        code = pccm.code()
        const_frag = "const" if store else ""
        const_mem = "" if store else "const"
        code.arg("frag", f"{self.fragment_t} {const_frag} &").arg(
            "offset", str(self.index_t))
        if self.with_source and not store:
            code.raw(f"""
            auto cur_pointer = source_pointer_;
            """)
        else:
            code.raw(f"""
            auto cur_pointer = pointer_;
            """)
        code.raw(f"""
        {self.access_t} {const_frag} *frag_ptr = reinterpret_cast<{self.access_t} {const_frag} *>(&frag);
        """)
        with code.range_("cluster", str(self.iterations[1]),
                         "TV_PRAGMA_UNROLL"):
            with code.range_("group", str(self.iterations[2]),
                             "TV_PRAGMA_UNROLL"):
                with code.range_("row", str(self.iterations[3]),
                                 "TV_PRAGMA_UNROLL"):
                    code.raw(f"""
                    int frag_row_idx =
                        (row +  {self.iterations[3]} * (group +  {self.iterations[2]} * cluster));
                    // delta: [Cluster, Group, Row]
                    """)
                    if not self.is_bias:
                        code.raw(f"""
                        int row_offset =
                            row * {self.delta[3]} + group * {self.delta[2]} + cluster * {self.delta[1]};

                        bool row_guard = ((row_offset + thread_start_row_) < extent_row_);
                        """)
                    if self.access_per_vector > 1:
                        if self.shuffle_in_stride:
                            code.raw(f"""
                            {self.dtype} {const_mem} *memory_pointer = cur_pointer + offset + indices_[frag_row_idx];
                            """)
                        else:
                            code.raw(f"""
                            {self.dtype} {const_mem} *memory_pointer = cur_pointer + offset;
                            """)
                        with code.range_("column", str(self.iterations[4]),
                                         "TV_PRAGMA_UNROLL"):
                            with code.range_("v", self.access_per_vector,
                                             "TV_PRAGMA_UNROLL"):
                                if self.is_bias:
                                    code.raw(
                                        f"bool guard = column_masks_[column][v];"
                                    )
                                else:
                                    code.raw(
                                        f"bool guard = row_guard && column_masks_[column][v];"
                                    )
                                if store:
                                    code.raw(f"""
                                    tv::gemm::global_store<{self.access_t}, sizeof({self.access_t})>(
                                        frag_ptr[frag_row_idx *  {self.iterations[4] * self.access_per_vector} + column * {self.access_per_vector} + v],
                                        memory_pointer + column * {self.delta[4]} + v * {self.element_per_acc},
                                        guard);
                                    """)
                                else:
                                    code.raw(f"""
                                    tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                                        frag_ptr[frag_row_idx *  {self.iterations[4] * self.access_per_vector} + column * {self.access_per_vector} + v],
                                        memory_pointer + column * {self.delta[4]} + v * {self.element_per_acc},
                                        guard);
                                    """)
                    else:
                        if self.shuffle_in_stride:
                            code.raw(f"""
                            {self.access_t} {const_mem} *memory_pointer =
                                reinterpret_cast<{self.access_t} {const_mem} *>(cur_pointer + offset + indices_[frag_row_idx]);
                            """)
                        else:
                            code.raw(f"""
                            {self.access_t} {const_mem} *memory_pointer =
                                reinterpret_cast<{self.access_t} {const_mem} *>(cur_pointer + offset);
                            """)

                        with code.range_("column", str(self.iterations[4]),
                                         "TV_PRAGMA_UNROLL"):
                            if self.is_bias:
                                code.raw(
                                    f"bool guard = column_masks_[column][0];"
                                )
                            else:
                                code.raw(
                                    f"bool guard = row_guard && column_masks_[column][0];"
                                )
                            if store:
                                code.raw(f"""
                                tv::gemm::global_store<{self.access_t}, sizeof({self.access_t})>(
                                    frag_ptr[frag_row_idx *  {self.iterations[4]} + column],
                                    (void *)&memory_pointer[column * {self.delta[4]} / {self.element_per_acc}],
                                    guard);
                                """)
                            else:
                                code.raw(f"""
                                tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                                    frag_ptr[frag_row_idx *  {self.iterations[4]} + column],
                                    (const void *)&memory_pointer[column * {self.delta[4]} / {self.element_per_acc}],
                                    guard);
                                """)


                    if not self.shuffle_in_stride and not self.is_bias:
                        code.raw(f"""
                        if (row + 1 <  {self.iterations[3]}) {{
                            cur_pointer += params_.increment_row;
                        }}
                        """)
                if not self.shuffle_in_stride and not self.is_bias:
                    code.raw(f"""
                    if (group + 1 <  {self.iterations[2]}) {{
                        cur_pointer += params_.increment_group;
                    }}
                    """)
            if not self.shuffle_in_stride and not self.is_bias:
                code.raw(f"""
                if (cluster + 1 <  {self.iterations[1]}) {{
                    cur_pointer += params_.increment_cluster;
                }}
                """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_offset(self):
        code = pccm.code()
        code.arg("frag",
                 f"{self.fragment_t} const &").arg("offset", str(self.index_t))
        if self.read_only:
            return code
        if self.is_bias:
            return code.make_invalid()
        return self.load_store_with_offset_template(True)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_offset(self):
        code = pccm.code()
        code.arg("frag", f"{self.fragment_t} &").arg("offset",
                                                     str(self.index_t))
        return self.load_store_with_offset_template(False)

    # @lineprof_wrapper_cpp
    def loadstore_with_offset_python(self, frag: ArrayPtr, offset: int,
                                     store: bool):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        cur_pointer = self.pointer_.shadow_copy()  # type: ArrayPtr
        # print(cudasim.threadIdx().x, cur_pointer)
        for cluster in range(self.iterations[1]):
            for group in range(self.iterations[2]):
                for row in range(self.iterations[3]):
                    frag_row_idx = (row + self.iterations[3] *
                                    (group + self.iterations[2] * cluster))
                    row_offset = row * self.delta[3] + group * self.delta[
                        2] + cluster * self.delta[1]

                    row_guard = ((row_offset + self.thread_start_row_) <
                                 self.extent_row_)
                    memory_pointer = (cur_pointer + offset).change_access_size(
                        self.element_per_acc)

                    for column in range(self.iterations[4]):
                        guard = row_guard and self.column_masks_[column]
                        if guard:
                            if store:
                                memory_pointer[
                                    column * self.delta[4] //
                                    self.element_per_acc] = frag_ptr[
                                        frag_row_idx * self.iterations[4] +
                                        column]
                            else:
                                frag_ptr[frag_row_idx * self.iterations[4] +
                                         column] = memory_pointer[
                                             column * self.delta[4] //
                                             self.element_per_acc]
                    if (row + 1 < self.iterations[3]):
                        cur_pointer += self.params_.increment_row

                if (group + 1 < self.iterations[2]):
                    cur_pointer += self.params_.increment_group

            if (cluster + 1 < self.iterations[1]):
                cur_pointer += self.params_.increment_cluster

    def store_python(self, frag: ArrayPtr):
        self.loadstore_with_offset_python(frag, 0, True)

    def load_python(self, frag: ArrayPtr):
        self.loadstore_with_offset_python(frag, 0, False)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        code = pccm.FunctionCode(f"""
        store_with_offset(frag, 0);
        """)
        if self.is_bias:
            return code.make_invalid()
        code.arg("frag", f"{self.fragment_t} const &")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} &")
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        cond = codeops.Condition(not self.shuffle_in_stride)
        cond_bothio = codeops.Condition(not self.shuffle_in_stride and self.with_source)

        if self.is_bias:
            code = pccm.FunctionCode()
        else:
            code = pccm.FunctionCode(f"""
            ++counts_[2];
            {cond("pointer_ += params_.advance_row;")}
            {cond_bothio("source_pointer_ += params_.advance_row;")}
            // kPartShape: [Tile, Cluster, Group, Row, Col]
            thread_start_row_ += {self.part_shape[3]};

            if (counts_[2] == {self.part_dilation[3]}) {{

            counts_[2] = 0;
            ++counts_[1];
            {cond("pointer_ += params_.advance_group;")}
            {cond_bothio("source_pointer_ += params_.advance_group;")}
            thread_start_row_ +=
                ({self.part_shape[2]} - 1) * {self.part_shape[3]} * {self.part_dilation[3]};

            if (counts_[1] == {self.part_dilation[2]}) {{

                counts_[1] = 0;
                ++counts_[0];
                {cond("pointer_ += params_.advance_cluster;")}
                {cond_bothio("source_pointer_ += params_.advance_cluster;")}
                thread_start_row_ +=
                    {self.part_dilation[2]} * {self.part_shape[2]} * {self.part_shape[3]} * {self.part_dilation[3]};

                if (counts_[0] == {self.part_dilation[1]}) {{
                counts_[0] = 0;
                {cond("pointer_ += params_.advance_tile;")}
                {cond_bothio("source_pointer_ += params_.advance_tile;")}
                }}
            }}
            }}
            """)

        if self.shuffle_in_stride:
            # update indices
            with code.range_("cluster", str(self.iterations[1]),
                             "TV_PRAGMA_UNROLL"):
                with code.range_("group", str(self.iterations[2]),
                                 "TV_PRAGMA_UNROLL"):
                    with code.range_("row", str(self.iterations[3]),
                                     "TV_PRAGMA_UNROLL"):
                        code.raw(f"""
                        int idx =
                            (row +  {self.iterations[3]} * (group +  {self.iterations[2]} * cluster));
                        int row_offset =
                            row * {self.delta[3]} + group * {self.delta[2]} + cluster * {self.delta[1]};
                        bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
                        indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
                        """)
        code.raw("return *this;")
        return code.ret(f"{self.class_name}&")

    def increment_python(self):
        self.counts_[2] += 1
        self.pointer_ += self.params_.advance_row
        # kPartShape: [Tile, Cluster, Group, Row, Col]
        self.thread_start_row_ += self.part_shape[3]

        if (self.counts_[2] == self.part_dilation[3]):

            self.counts_[2] = 0
            self.counts_[1] += 1
            self.pointer_ += self.params_.advance_group

            self.thread_start_row_ += (
                self.part_shape[2] -
                1) * self.part_shape[3] * self.part_dilation[3]

            if (self.counts_[1] == self.part_dilation[2]):

                self.counts_[1] = 0
                self.counts_[0] += 1
                self.pointer_ += self.params_.advance_cluster

                self.thread_start_row_ += self.part_dilation[
                    2] * self.part_shape[2] * self.part_shape[
                        3] * self.part_dilation[3]

                if (self.counts_[0] == self.part_dilation[1]):
                    self.counts_[0] = 0
                    self.pointer_ += self.params_.advance_tile


class OutFragIter(GemmOutFragIterator):
    def __init__(self, dtype: dtypes.DType, element_per_acc: int,
                 num_iteration: int):
        super().__init__(dtype, element_per_acc * num_iteration,
                         element_per_acc)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)

        self.num_iteration = num_iteration

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
        new_obj = OutFragIter(self.dtype, self.element_per_acc,
                              self.num_iteration)
        new_obj.index_ = 0
        new_obj.src_ptr_ = src_ptr.change_access_size(self.element_per_acc)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        {self.access_pointer} frag_ptr = reinterpret_cast<{self.access_pointer}>(&frag);
        TV_PRAGMA_UNROLL
        for (int n = 0; n < {self.num_iteration}; ++n) {{
            frag_ptr[n] = src_ptr_[index_ * {self.num_iteration} + n];
        }}
        """)
        code.arg("frag", f"{self.fragment_t} &")
        code.arg("index_offset", str(self.index_t), "0")
        return code

    def load_python(self, frag: ArrayPtr):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        for n in range(self.num_iteration):
            frag_ptr[n] = self.src_ptr_[self.index_ * self.num_iteration + n]

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
