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
from pccm.targets.cuda_ptx import RegDType

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewNVRTC
from cumm.constants import CUTLASS_MODE
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers
from cumm.gemm import bases, codeops, constants, layout, thread_map
from cumm.gemm.arch import memory
from cumm.gemm.core import MetaArray, array_type, metaseq, seq


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


_LAYOUT_TYPES = Union[layout.ColumnMajor, layout.RowMajor,
                      layout.ColumnMajorInterleaved,
                      layout.RowMajorInterleaved]


class WarpTileIterator(bases.GemmWarpIterator):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape_km_padded: MetaArray[int],
                 warp_tile_shape_km: MetaArray[int],
                 warp_shape: MetaArray[int],
                 lane_mma_shape_km: MetaArray[int],
                 smem_layout: _LAYOUT_TYPES,
                 lane_layout: _LAYOUT_TYPES,
                 padding: int,
                 left: bool = False,
                 partk: int = 1):
        element_per_acc = lane_mma_shape_km[0] * lane_mma_shape_km[1]
        self.sub_access_size = element_per_acc
        if padding > 0:
            if isinstance(
                    smem_layout,
                (layout.RowMajorInterleaved, layout.ColumnMajorInterleaved)):
                self.sub_access_size = min(element_per_acc,
                                           padding * smem_layout.interleave)
            else:
                self.sub_access_size = min(element_per_acc, padding)

        self.num_sub_access = element_per_acc // self.sub_access_size
        self.wmma_mat_shape_per_k_iter = metaseq(lane_mma_shape_km[0],
                                                 warp_tile_shape_km[1])
        warp_shape_contig = warp_shape[0] if left else warp_shape[1]
        # for each operand, every thread takes
        self.thread_mat_shape_per_k_iter = metaseq(
            self.wmma_mat_shape_per_k_iter[0],
            warp_tile_shape_km[1] // warp_shape_contig)
        # print(self.wmma_mat_shape_per_k_iter, tile_shape_km_padded, self.thread_mat_shape_per_k_iter)
        # raise NotImplementedError
        self.partk = partk
        self.thread_access_shape = metaseq(
            self.thread_mat_shape_per_k_iter[0] // lane_mma_shape_km[0],
            self.thread_mat_shape_per_k_iter[1] // lane_mma_shape_km[1])
        # print(self.thread_access_shape)
        self.thread_access_delta = self.wmma_mat_shape_per_k_iter // self.thread_access_shape // metaseq(
            lane_mma_shape_km[0], warp_shape_contig)
        element_count = self.thread_access_shape.prod() * element_per_acc

        super().__init__(dtype, element_count, self.sub_access_size)

        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.add_param_class("ns1", smem_layout,
                             "SmemLayout")  # TODO add a real layout class
        self.add_param_class("ns2", lane_layout,
                             "LaneLayout")  # TODO add a real layout class
        self.smem_layout = smem_layout
        self.lane_layout = lane_layout
        self.padding = padding

        self.tile_shape_km_padded = tile_shape_km_padded
        # print(tile_shape_km_padded)
        self.lane_mma_shape_km = lane_mma_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km

        self.warp_tile_shape = warp_tile_shape_km
        self.warp_shape = warp_shape
        self.lane_mma_shape = lane_mma_shape_km
        self.left = left
        self.num_k_iters = warp_tile_shape_km[0] // lane_mma_shape_km[0]
        self.add_member("pointer_", self.access_pointer)
        # self.add_member("pointer_bkp_", self.access_pointer)

        if partk > 1:
            self.add_member("wmma_k_index_", "int")

        smem_layout = self.smem_layout.from_shape_python(
            self.tile_shape_km_padded[:2])
        assert smem_layout.stride % self.sub_access_size == 0
        # cudasim members
        self.pointer_ = None  # type: Optional[ArrayPtr]
        # self.pointer_bkp_ = None  # type: Optional[ArrayPtr]

        self.wmma_k_index_ = 0

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        constexpr auto lane_layout = LaneLayout::from_shape({{{self.warp_shape[0]}, {self.warp_shape[1]}}});
        constexpr auto smem_layout = SmemLayout::from_shape({{{self.tile_shape_km_padded[0]}, {self.tile_shape_km_padded[1]}}});

        tv::array<int, 2> logical_offset{{warp_idx_k * {self.num_k_iters},
                                        warp_idx_residual * {self.wmma_mat_shape_per_k_iter[1]}}};
        // int lane_offset;
        if ({pccm.literal(self.left)}) {{
            // 0, 1, 0, 1, 0, 1, ..., 2, 3, 2, 3, 2, 3
            logical_offset[1] += lane_layout.inverse_0(lane_idx) * {self.lane_mma_shape[1]};
        }} else {{
            // 0, 0, 1, 1, 2, 2, 3, 3, ...
            logical_offset[1] += lane_layout.inverse_1(lane_idx) * {self.lane_mma_shape[1]};
        }}
        auto offset = smem_layout(logical_offset[0], logical_offset[1]);
        pointer_ = reinterpret_cast<{self.access_pointer}>(ptr + offset);

        """)
        # if self.left:
        #     code.raw(f"tv::printf2_block_once(threadIdx.x, offset);")
        code.arg("ptr",
                 self.pointer).arg("warp_idx_k,warp_idx_residual,lane_idx",
                                   "int")
        if self.partk > 1:
            code.ctor_init("wmma_k_index_", "0")
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_residual: int, lane_idx: int):
        new_obj = WarpTileIterator(self.dtype, self.tile_shape_km_padded,
                                   self.warp_tile_shape_km, self.warp_shape,
                                   self.lane_mma_shape_km, self.smem_layout,
                                   self.lane_layout, self.padding, self.left,
                                   self.partk)
        lane_layout = new_obj.lane_layout.from_shape_python(new_obj.warp_shape)
        smem_layout = new_obj.smem_layout.from_shape_python(
            new_obj.tile_shape_km_padded[:2])
        logical_offset = seq(
            warp_idx_k * new_obj.num_k_iters,
            warp_idx_residual * new_obj.wmma_mat_shape_per_k_iter[1])
        if new_obj.left:
            logical_offset[1] += lane_layout.inverse_0_python(
                lane_idx) * new_obj.lane_mma_shape[1]
        else:
            logical_offset[1] += lane_layout.inverse_1_python(
                lane_idx) * new_obj.lane_mma_shape[1]
        offset = smem_layout(logical_offset[0], logical_offset[1])
        new_obj.pointer_ = (ptr + offset).change_access_size(
            new_obj.sub_access_size)
        # new_obj.pointer_bkp_ = (ptr).change_access_size(
        #     new_obj.sub_access_size)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        static constexpr auto smem_layout = SmemLayout::from_shape({{{self.tile_shape_km_padded[0]}, {self.tile_shape_km_padded[1]}}});
        pointer_ += smem_layout(num * {self.wmma_mat_shape_per_k_iter[0]}, 0) / {self.sub_access_size};
        """)
        return code.arg("num", "int")

    def tile_increment_python(self, num: int):
        smem_layout = self.smem_layout.from_shape_python(
            self.tile_shape_km_padded[:2])

        # print(self.pointer_)
        # print(num, smem_layout)
        self.pointer_ += smem_layout(num * self.wmma_mat_shape_per_k_iter[0],
                                     0) // self.sub_access_size

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        # smem_layout = self.smem_layout.from_shape_python([self.tile_shape_km_padded[0], self.tile_shape_km_padded[1]])
        # code = pccm.FunctionCode(f"""
        # pointer_ += {smem_layout(self.wmma_mat_shape_per_k_iter[0], 0) // self.sub_access_size};
        # return *this;
        # """)

        code = pccm.FunctionCode(f"""
        static constexpr auto smem_layout = SmemLayout::from_shape({{{self.tile_shape_km_padded[0]}, {self.tile_shape_km_padded[1]}}});
        auto offset = 
        pointer_ += smem_layout({self.wmma_mat_shape_per_k_iter[0]}, 0) / {self.sub_access_size};
        
        """)
        if self.partk > 1:
            k_dist = (self.partk - 1) * self.num_k_iters
            code.raw(f"""
            ++wmma_k_index_;
            if (wmma_k_index_ == {self.num_k_iters}){{
                wmma_k_index_ = 0;
                pointer_ += smem_layout({k_dist * self.wmma_mat_shape_per_k_iter[0]}, 0) / {self.sub_access_size};
                // tile_increment({self.partk - 1});
            }}
            """)
        code.raw(f"return *this;")
        return code.ret("WarpTileIterator &")

    def increment_python(self):
        num_k_inc = self.num_k_iters
        smem_layout = self.smem_layout.from_shape_python(
            self.tile_shape_km_padded[:2])
        self.pointer_ += smem_layout(self.wmma_mat_shape_per_k_iter[0],
                                     0) // self.sub_access_size
        k_dist = (self.partk - 1) * self.num_k_iters

        if self.partk > 1:
            # partk == 1 don't need these code because
            # partk > 1 have other warp that handle following
            # k-iters.
            self.wmma_k_index_ += 1
            if self.wmma_k_index_ == self.num_k_iters:
                self.wmma_k_index_ = 0
                # jump to next stage
                self.pointer_ += smem_layout(
                    k_dist * self.wmma_mat_shape_per_k_iter[0],
                    0) // self.sub_access_size

        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        warp_mma_length = self.warp_shape[0] if self.left else self.warp_shape[
            1]
        code = pccm.FunctionCode(f"""
        static constexpr auto smem_layout = SmemLayout::from_shape({{{self.tile_shape_km_padded[0]}, {self.tile_shape_km_padded[1]}}});
        {self.access_pointer} dst_ptr = reinterpret_cast<{self.access_pointer}>(&frag);
        // kRow: 1, kCol: 2
        TV_PRAGMA_UNROLL
        for (int k = 0; k < {self.thread_access_shape[0]}; ++k) {{
            TV_PRAGMA_UNROLL
            for (int n = 0; n < {self.thread_access_shape[1]}; ++n) {{
                // ref offset: [8, 128 / 4], (0, n * 8) ~= (0, 32)
                // lane_id = 0, [0, 0-3], [0, 32-35]
                // lane_id = 1, [0, 0-3], [0, 32-35]
                // lane_id = 2, [0, 4-7], [0, 36-39]
                // lane_id = 3, [0, 4-7], [0, 36-39]
                // ...
                // lane_id = 31, [0, 28-31], [0, 60-63]
                auto offset = smem_layout(k * {self.lane_mma_shape[0]},
                                        n * ({pccm.boolean(self.left)} ? {self.warp_shape[0]} : {self.warp_shape[1]}) *
                                            {self.thread_access_delta[1]}) /
                            {self.sub_access_size};

                // auto offset = smem_layout(k * {self.lane_mma_shape[0]},
                //                         n * {warp_mma_length * self.thread_access_delta[1] // self.sub_access_size});
                TV_PRAGMA_UNROLL
                for (int sub = 0; sub < {self.num_sub_access}; ++sub){{
                    dst_ptr[k * {self.thread_access_shape[1] * self.num_sub_access} 
                        + n * {self.num_sub_access} + sub] =
                        pointer_[offset + sub + pointer_offset / {self.sub_access_size}];
                }}
            }}
        }}
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    async def load_with_pointer_offset_python(self, frag: ArrayPtr,
                                              pointer_offset: int):
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)
        smem_layout = self.smem_layout.from_shape_python(
            self.tile_shape_km_padded[:2])
        dst_ptr = frag.change_access_size(
            self.sub_access_size)  # type: ArrayPtr
        for k in range(self.thread_access_shape[0]):
            for n in range(self.thread_access_shape[1]):
                offset = smem_layout(
                    k * self.lane_mma_shape[0],
                    n *
                    (self.warp_shape[0] if self.left else self.warp_shape[1]) *
                    self.thread_access_delta[1]) // self.sub_access_size
                for sub in range(self.num_sub_access):
                    dst_offset = (
                        k * self.thread_access_shape[1] * self.num_sub_access +
                        n * self.num_sub_access + sub)
                    dst_ptr[dst_offset] = self.pointer_[offset + sub +
                                                        pointer_offset //
                                                        self.sub_access_size]
                    # if cudasim.threadIdx().x == 0:
                    #     print("self.pointer_.access_byte_size_", self.pointer_.access_byte_size_)
                    await checkers.smem_bank_conflicit_check(
                        self.pointer_,
                        (offset + sub +
                         pointer_offset // self.sub_access_size))
                    access_pointer = self.pointer_ + (
                        offset + sub + pointer_offset // self.sub_access_size)
                    ptr_addrs[dst_offset *
                              dst_ptr.access_size:(dst_offset + 1) *
                              dst_ptr.access_size] = np.arange(
                                  access_pointer.offset,
                                  access_pointer.offset + dst_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_pointer_offset_python(frag, 0)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_kgroup_index(self):
        code = pccm.code()
        code.arg("wmma_k", "int")
        if self.partk > 1:
            code.raw("wmma_k_index_ = wmma_k;")
        return code

    def set_wmma_k_index_python(self, wmma_k: int):
        self.wmma_k_index_ = wmma_k
        return


class SmemTileIteratorV2(bases.GemmSmemIterator):
    def __init__(self,
                 dtype: dtypes.DType,
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 tile_shape: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 advance_axis: int,
                 num_threads: int,
                 smem_shape: MetaArray[int],
                 transposed_input: bool = False):
        alignment = dtype.itemsize() * sub_tile_shape.prod()
        self.tile_shape = tile_shape
        self.tmap = tmap
        self.sub_tile_shape = sub_tile_shape
        self.access_shape = self.tile_shape // self.sub_tile_shape  # type: MetaArray[int]
        self.interleave = sub_tile_shape[0]

        access_shape = tmap.iterations
        delta = tmap.delta

        if transposed_input:
            access_shape = access_shape[::-1]
            delta = delta[::-1]
        delta = seq(delta[0] // self.interleave, delta[1] * self.interleave)
        self.thread_access_shape = access_shape
        self.iteration_delta = delta
        # print("self.thread_access_shape", self.thread_access_shape, self.iteration_delta)
        element_count = self.thread_access_shape.prod() * sub_tile_shape.prod()

        super().__init__(dtype, element_count, sub_tile_shape.prod(),
                         alignment)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.advance_axis = advance_axis
        self.num_threads = num_threads
        self.transposed_input = transposed_input

        self.interleave = sub_tile_shape[0]
        self.smem_shape = smem_shape
        self.smem_vis_shape = seq(smem_shape[0] // self.interleave,
                                  smem_shape[1] * self.interleave)
        self.static_stride = smem_shape[1]
        self.static_inc_strided = self.static_stride * self.interleave * self.iteration_delta[
            0] * self.dtype.itemsize()
        if self.advance_axis == 1:
            self.static_inc_advance = self.tile_shape[1] * self.dtype.itemsize(
            )
        else:
            self.static_inc_advance = self.static_stride * self.tile_shape[
                0] * self.dtype.itemsize()

        self.fragment_t = array_type(dtype, self.element_count)
        self.add_member("pointer_", self.byte_pointer)
        # self.add_member("stride_", str(self.index_t))
        # self.add_member("inc_strided_", str(self.index_t))
        # self.add_member("inc_advance_", str(self.index_t))

        # cudasim members
        self.pointer_ = None  # type: Optional[ArrayPtr]
        self.stride_ = 0
        self.inc_strided_ = 0
        self.inc_advance_ = 0

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        contig = 1
        strided = 0
        code = pccm.FunctionCode(f"""
        auto thread_offset = ThreadMap::initial_offset(thread_id);
        """)
        if self.transposed_input:
            code.raw(f"""
            int offset = (thread_offset[1] / {self.interleave}) * {self.smem_shape[1] * self.interleave}
                 + thread_offset[0] * {self.interleave};
            """)
        else:
            code.raw(f"""
            int offset = (thread_offset[0] / {self.interleave}) * {self.smem_shape[1] * self.interleave}
                + thread_offset[1] * {self.interleave};
            """)

        code.raw(f"""
        pointer_ = reinterpret_cast<{self.byte_pointer}>(ptr + offset);
        // for transposed input, kThreadAccessShape and kIterationDelta is
        // transposed too.
        // inc_strided_ = stride * {self.iteration_delta[strided]} * sizeof({self.dtype});
        // if ({pccm.boolean(self.advance_axis == contig)}) {{
        //     inc_advance_ = {self.tile_shape[contig]} * sizeof({self.dtype});
        // }} else {{
        //     inc_advance_ = {self.tile_shape[strided]} * stride * sizeof({self.dtype});
        // }}
        // tv::printf2_block_once(threadIdx.x, "inc_strided_", inc_strided_, inc_advance_, stride_, offset);
        """)  # .ctor_init("stride_", "stride")
        code.arg("stride", "int")
        code.arg("ptr", f"{self.dtype}*")
        code.arg("thread_id", "int")
        return code

    def python_ctor(self, stride: int, ptr: ArrayPtr, thread_id: int):
        new_obj = SmemTileIteratorV2(self.dtype, self.tmap, self.tile_shape,
                                     self.sub_tile_shape, self.advance_axis,
                                     self.num_threads, self.smem_shape,
                                     self.transposed_input)
        contig = 1
        strided = 0
        new_obj.stride_ = stride
        offset = 0  # type: int
        thread_offset = self.tmap.initial_offset_python(thread_id)
        # interleave is only used for int8 DP4A gemm.
        if new_obj.transposed_input:
            offset = (
                thread_offset[1] // self.interleave
            ) * stride * self.interleave + thread_offset[0] * self.interleave
        else:
            offset = (
                thread_offset[0] // self.interleave
            ) * stride * self.interleave + thread_offset[1] * self.interleave

        # print(offset)
        new_obj.pointer_ = (ptr + offset).change_access_byte_size(1)
        new_obj.inc_strided_ = stride * self.interleave * new_obj.iteration_delta[
            strided] * (new_obj.dtype.itemsize())
        if new_obj.advance_axis == contig:
            new_obj.inc_advance_ = new_obj.tile_shape[contig] * (
                new_obj.dtype.itemsize())
        else:
            new_obj.inc_advance_ = new_obj.tile_shape[strided] * stride * (
                new_obj.dtype.itemsize())
        # if cudasim.threadIdx().x == 0:
        # print(cudasim.threadIdx().x, "new_obj.inc_strided_", new_obj.inc_strided_, new_obj.inc_advance_, new_obj.stride_, offset)
        return new_obj

    def get_smem_vis_shape(self) -> MetaArray[int]:
        return self.smem_vis_shape

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        pointer_ += {self.static_inc_advance} * num;
        """)
        return code.arg("num", "int")

    def tile_increment_python(self, num: int):
        self.pointer_ += self.inc_advance_ * num

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        pointer_ +=  {self.static_inc_advance};
        return *this;
        """)
        return code.ret(f"{self.class_name}&")

    def increment_python(self):
        self.pointer_ += self.inc_advance_
        return self

    @pccm.cuda.member_function(name="operator--",
                               device=True,
                               forceinline=True)
    def operator_ss(self):
        code = pccm.FunctionCode(f"""
        pointer_ -=  {self.static_inc_advance};
        return *this;
        """)
        return code.ret(f"{self.class_name}&")

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        contig = 1
        strided = 0

        code = pccm.FunctionCode(f"""
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        {self.const_byte_pointer} byte_pointer =
            pointer_ + pointer_offset * sizeof({self.dtype});

        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.thread_access_shape[strided]}; ++s) {{
            {self.access_t} const *access_ptr =
                reinterpret_cast<{self.access_t} const *>(byte_pointer);
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.thread_access_shape[contig]}; ++c) {{
                int idx = c + s * {self.thread_access_shape[contig]};
                frag_ptr[idx] =
                    access_ptr[c * {self.iteration_delta[contig]} / {self.element_per_acc}];
            }}
            if (s < {self.thread_access_shape[strided]} - 1) {{
                byte_pointer +=  {self.static_inc_strided};
            }}
        }}
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        contig = 1
        strided = 0
        # TODO should we swap sc for transposed input?
        code = pccm.FunctionCode(f"""
        {self.access_t} const *frag_ptr = reinterpret_cast<{self.access_t} const *>(&frag);
        {self.byte_pointer} byte_pointer =
            pointer_ + pointer_offset * sizeof({self.dtype});

        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.thread_access_shape[strided]}; ++s) {{
            {self.access_t} *access_ptr =
                reinterpret_cast<{self.access_t} *>(byte_pointer);
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.thread_access_shape[contig]}; ++c) {{
                int idx = c + s * {self.thread_access_shape[contig]};
                access_ptr[c * {self.iteration_delta[contig]} / {self.element_per_acc}] =
                    frag_ptr[idx];
            }}
            if (s < {self.thread_access_shape[strided] - 1}) {{
                byte_pointer += {self.static_inc_strided};
            }}
        }}
        """)
        code.arg("frag",
                 f"{self.fragment_t} const &").arg("pointer_offset",
                                                   str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        contig = 1
        strided = 0
        ptr_addrs = np.zeros((frag.length), dtype=np.int32)

        frag_ptr = frag.change_access_size(
            self.sub_tile_shape.prod())  # type: ArrayPtr
        byte_pointer = self.pointer_ + pointer_offset * self.dtype.itemsize()
        for s in range(self.thread_access_shape[strided]):
            access_ptr = byte_pointer.change_access_size(
                self.sub_tile_shape.prod())
            for c in range(self.thread_access_shape[contig]):
                idx = c + s * self.thread_access_shape[contig]
                access_offset = c * self.iteration_delta[
                    contig] // self.element_per_acc

                await checkers.smem_bank_conflicit_check(
                    access_ptr + access_offset, 0)
                access_ptr[access_offset] = frag_ptr[idx]
                ptr_addrs[idx * frag_ptr.access_size:(idx + 1) *
                          frag_ptr.access_size] = np.arange(
                              (access_ptr + access_offset).offset,
                              (access_ptr + access_offset).offset +
                              frag_ptr.access_size)
            if s < self.thread_access_shape[strided] - 1:
                byte_pointer += self.inc_strided_
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        code = pccm.FunctionCode(f"""
        store_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} const &")
        return code

    async def store_python(self, frag: ArrayPtr):
        return await self.store_with_pointer_offset_python(frag, 0)


class MaskTileIteratorParams(pccm.ParameterizedClass):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 advance_axis: int,
                 shuffle_in_stride: bool = False):
        super().__init__()
        self.dtype = dtype
        self.long_index_t = dtypes.int64
        self.index_t = dtypes.int32
        self.tile_shape = tile_shape
        self.sub_tile_shape = sub_tile_shape
        self.advance_axis = advance_axis
        self.tmap = tmap
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        self.shuffle_in_stride = shuffle_in_stride

        #  self.add_member("stride_", str(self.index_t))
        self.add_member("stride_", str(self.index_t))
        self.add_member("inc_strided_", str(self.long_index_t))
        self.inc_advance_static = -1
        if self.advance_axis == 1:
            self.inc_advance_static = self.tile_shape[1] * self.dtype.itemsize(
            )
        else:
            self.add_member("inc_advance_", str(self.long_index_t))
        self.add_member("inc_next_", str(self.long_index_t))
        if shuffle_in_stride:
            self.add_member("indice_ptr_", f"{self.index_t} const *")

        # cudasim params
        self.stride_ = 0
        self.inc_strided_ = 0
        self.inc_advance_ = 0
        self.inc_next_ = 0

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def default_ctor(self):
        return pccm.code()

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        contig = 1
        strided = 0
        code = pccm.code()
        code.raw(f"""
        inc_strided_ = stride * {self.iteration_delta[strided]} * sizeof({self.dtype});
        """)
        if self.advance_axis == 0:
            code.raw(f"""
            inc_advance_ = {self.tile_shape[strided] * self.dtype.itemsize()} * stride;

            inc_next_ = inc_advance_ - ({self.thread_access_shape[strided] - 1}) *
                                            {self.iteration_delta[strided]} * stride *
                                            sizeof({self.dtype});

            """)
        else:
            code.raw(f"""
            inc_next_ = {self.inc_advance_static} - ({self.thread_access_shape[strided] - 1}) *
                                            {self.iteration_delta[strided]} * stride *
                                            sizeof({self.dtype});
            """)

        code.arg("stride", "int")
        code.ctor_init("stride_", "stride")
        if self.shuffle_in_stride:
            code.arg("indice_ptr", f"{self.index_t} const *", "nullptr")
            code.ctor_init("indice_ptr_", "indice_ptr")
        return code

    def python_ctor(self, stride: int):
        new_obj = MaskTileIteratorParams(self.dtype, self.tile_shape,
                                         self.sub_tile_shape, self.tmap,
                                         self.advance_axis)
        contig = 1
        strided = 0
        new_obj.stride_ = stride
        new_obj.inc_strided_ = stride * new_obj.iteration_delta[
            strided] * new_obj.dtype.itemsize()
        if new_obj.advance_axis == contig:
            new_obj.inc_advance_ = new_obj.tile_shape[
                contig] * new_obj.dtype.itemsize()
        else:
            new_obj.inc_advance_ = new_obj.tile_shape[
                strided] * stride * new_obj.dtype.itemsize()
        new_obj.inc_next_ = new_obj.inc_advance_ - (
            new_obj.thread_access_shape[strided] - 1
        ) * new_obj.iteration_delta[strided] * stride * self.dtype.itemsize()
        return new_obj


class MaskTileIterator(bases.GemmInputIterator):
    """
    # TODO gemm with gather
    # TODO support shuffle in k axis, just update indices in every k increment. ez
    shuffle_in_stride: for spatial sparse convolution.
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 param_class: MaskTileIteratorParams,
                 advance_axis: int,
                 num_sub_access: int,
                 transpose_load: bool = False,
                 last_residual: bool = True,
                 shuffle_in_stride: bool = False,
                 read_only: bool = True):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        self.access_per_vector = sub_tile_shape[1] // num_sub_access
        self.thread_tensor_stride = metaseq(
            self.thread_access_shape[1] * sub_tile_shape[0] *
            self.access_per_vector, sub_tile_shape[0] * self.access_per_vector,
            self.access_per_vector, 1)
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        # print("INPUT TAMP", tmap.iterations, tmap.delta, num_sub_access, )
        # raise NotImplementedError
        super().__init__(dtype, tmap, sub_tile_shape, element_count,
                         num_sub_access,
                         dtype.itemsize() * num_sub_access)
        if transpose_load:
            assert dtype == dtypes.int8
        self.read_only = read_only
        # shuffle_in_stride = False
        self.shuffle_in_stride = shuffle_in_stride
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.param_class = param_class
        self.add_param_class("maskiter", tmap, "ThreadMap")
        self.add_param_class("maskiter", param_class, "Params")
        self.tile_shape = tile_shape
        self.sub_tile_shape = sub_tile_shape
        self.advance_axis = advance_axis
        self.no_advance_axis = int(advance_axis == 0)
        self.num_sub_access = num_sub_access
        self.transpose_load = transpose_load
        self.last_residual = last_residual
        self.mask_count = self.thread_access_shape[
            0] * self.thread_access_shape[1] * sub_tile_shape[0]

        self.num_pred_per_byte = 4
        self.num_pred_per_32 = 4 * self.num_pred_per_byte
        self.num_pred_byte = div_up(self.mask_count, self.num_pred_per_byte)
        self.num_pred_32 = div_up(self.num_pred_byte, self.num_pred_per_byte)
        self.num_pred_mask = (1 << self.num_pred_per_byte) - 1
        self.mask_tensor_shape = metaseq(self.num_pred_32,
                                         self.num_pred_per_32,
                                         self.num_pred_per_byte)

        self.global_load = memory.GlobalLoad(self.element_per_acc *
                                             self.dtype.itemsize())
        self.add_param_class("maskiter", self.global_load, "GlobalLoad")

        self.add_member("pointer_", self.const_byte_pointer
                        if read_only else self.byte_pointer)  # 2 registers
        # self.add_member("pointer_bkp_", self.const_byte_pointer)

        self.add_member("params_", "Params const &")

        self.add_member("extent_", "tv::array<int, 2>")
        self.add_member("thread_offset_", "tv::array<int, 2>")
        self.add_member("residue_offset_", "int")
        self.add_member("is_residue_tile_", "bool")
        # self.add_member("residue_tile_idx_", "int")
        self.add_member("predicates_",
                        str(dtypes.uint32),
                        array=f"[{self.num_pred_32}]")
        if self.shuffle_in_stride:
            self.add_member(
                "indices_",
                str(dtypes.int32),
                array=f"[{self.tmap.iterations[0] * self.sub_tile_shape[0]}]")

        # cudasim members
        self.pointer_ = None  # type: Optional[ArrayPtr]
        self.params_ = None  # type: Optional[MaskTileIteratorParams]
        self.extent_ = seq(0, 0)
        self.thread_offset_ = seq(0, 0)
        self.residue_offset_ = seq(0, 0)

        self.is_residue_tile_ = False
        self.residue_tile_idx_ = 0
        self.predicates_ = np.zeros((self.num_pred_32, ), dtype=np.int32)
        self.is_left_ = True

    def get_params(self) -> pccm.ParameterizedClass:
        return self.param_class

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        contig = 1
        strided = 0
        code = pccm.code()
        # print("KADVANCE", self.param_class.inc_advance_static)
        code.raw(f"""
        int residue_size = (extent[{self.advance_axis}] - threadblock_offset[{self.advance_axis}]) %
                        {self.tile_shape[self.advance_axis]};
        if (!residue_size) {{
            residue_size = {self.tile_shape[self.advance_axis]};
        }}
        // tv::printf2_once(params_.inc_strided_, params_.inc_next_);
        """)
        if self.last_residual:
            if self.advance_axis == 1:
                code.raw(f"""
                residue_offset_ = residue_size;
                """)
            else:
                code.raw(f"""
                residue_offset_ = residue_size;
                """)
        else:
            # we minus 1 here because the range of that value is 1~.
            val_str = f"""
            (extent[{self.advance_axis}] - threadblock_offset[{self.advance_axis}] - 1) /
            {self.tile_shape[self.advance_axis]} * {self.tile_shape[self.advance_axis]}
            """
            if self.advance_axis == 1:
                code.raw(f"""
                residue_offset_ = {val_str};
                """)
            else:
                code.raw(f"""
                residue_offset_ = {val_str};
                """)
        min_val = f"""
        std::min(threadblock_offset[{self.advance_axis}] + residue_size, extent_[{self.advance_axis}])
        """

        if self.advance_axis == 1:
            code.raw(f"""
            tv::array<int, 2> residue_extent{{extent_[{self.no_advance_axis}], {min_val}}};
            """)
        else:
            code.raw(f"""
            tv::array<int, 2> residue_extent{{{min_val}, extent_[{self.no_advance_axis}]}};
            """)

        code.raw(f"""
        // residue tile always first k axis tile
        // thread_id / kAccessShape[kContig] is 'sub-tile' coord, so we need to
        // convert back to element coord
        thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);
        // auto init = ThreadMap::initial_offset(thread_id);
        if TV_IF_CONSTEXPR (!{pccm.boolean(self.last_residual)}) {{
            thread_offset_[{self.advance_axis}] += residue_offset_;
        }}
        // tv::printf2_block_once(threadIdx.x, thread_offset_[0] * extent_[1] + thread_offset_[1]);
        if TV_IF_CONSTEXPR (!{pccm.boolean(self.last_residual)}) {{
            compute_predicates_(extent, false);
        }} else {{
            compute_predicates_(residue_extent, false);
        }}
        """)
        if self.shuffle_in_stride:
            code.raw(f"""
            update_indices();
            add_pointer_offset(thread_offset_[1]);
            """)
            # if self.advance_axis == 0:
            #     code.raw(f"""
            #     params_.indice_ptr_ += thread_offset_[0];
            #     """)
        else:
            code.raw(f"""
            // here we can't use extent_[1] because splitk may split stride.
            add_pointer_offset(thread_offset_[0] * params.stride_ + thread_offset_[1]);
            """)
        code.arg("params", f"Params const &")
        code.arg("ptr",
                 f"{self.const_pointer if self.read_only else self.pointer}")
        code.arg("extent", "tv::array<int, 2>")
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")

        code.ctor_init("params_", "params")
        code.ctor_init(
            "pointer_",
            f"reinterpret_cast<{self.const_byte_pointer if self.read_only else self.byte_pointer}>(ptr)"
        )
        # code.ctor_init("pointer_bkp_",
        #                f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")

        code.ctor_init("extent_", "extent")
        code.ctor_init("is_residue_tile_", "true")

        return code

    def python_ctor(self,
                    params: MaskTileIteratorParams,
                    ptr: ArrayPtr,
                    extent: MetaArray[int],
                    thread_id: int,
                    tb_offset: MetaArray[int],
                    is_left: bool = True) -> "MaskTileIterator":
        new_obj = MaskTileIterator(self.dtype, self.tile_shape,
                                   self.sub_tile_shape, self.tmap, params,
                                   self.advance_axis, self.num_sub_access,
                                   self.transpose_load, self.last_residual,
                                   self.shuffle_in_stride, self.read_only)
        new_obj.params_ = params
        new_obj.pointer_ = ptr.change_access_byte_size(1)
        new_obj.extent_ = extent
        new_obj.is_residue_tile_ = True
        # if cudasim.threadIdx().x == 0:
        #     print(self.tmap)
        #     print(self.tile_shape, self.sub_tile_shape)
        #     print(cudasim.get_cuda_context().blockDim.count())

        residue_extent = seq(0, 0)
        residue_size = (extent[new_obj.advance_axis] -
                        tb_offset[new_obj.advance_axis]
                        ) % new_obj.tile_shape[new_obj.advance_axis]
        if residue_size == 0:
            residue_size = new_obj.tile_shape[new_obj.advance_axis]

        if (new_obj.last_residual):
            new_obj.residue_offset_[new_obj.no_advance_axis] = 0
            new_obj.residue_offset_[new_obj.advance_axis] = residue_size
        else:
            residue_tile_idx_ = (extent[new_obj.advance_axis] -
                                 tb_offset[new_obj.advance_axis] -
                                 1) // new_obj.tile_shape[new_obj.advance_axis]
            new_obj.residue_offset_[new_obj.no_advance_axis] = 0
            new_obj.residue_offset_[
                new_obj.advance_axis] = residue_tile_idx_ * new_obj.tile_shape[
                    new_obj.advance_axis]

        residue_extent[new_obj.no_advance_axis] = new_obj.extent_[
            new_obj.no_advance_axis]
        residue_extent[new_obj.advance_axis] = min(
            tb_offset[new_obj.advance_axis] + residue_size,
            new_obj.extent_[new_obj.advance_axis])

        new_obj.thread_offset_ = tb_offset + new_obj.tmap.initial_offset_python(
            thread_id)
        if (not new_obj.last_residual):
            new_obj.thread_offset_ += new_obj.residue_offset_
        if cudasim.debug_once():
            print(residue_size, new_obj.params_.stride_,
                  new_obj.thread_offset_, residue_extent)
        # print(ptr, thread_id, new_obj.params_.stride_, new_obj.thread_offset_)
        new_obj.add_pointer_offset_python(new_obj.thread_offset_[0] *
                                          new_obj.params_.stride_ +
                                          new_obj.thread_offset_[1])
        if (not new_obj.last_residual):
            new_obj.compute_predicates_python(extent, False)
        else:
            new_obj.compute_predicates_python(residue_extent, False)
        new_obj.is_left_ = is_left
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def update_indices(self):
        code = pccm.code()
        if not self.shuffle_in_stride:
            return code
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
            TV_PRAGMA_UNROLL
            for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                if (thread_offset_[0] + s * {self.tmap.delta[0]} + ss < extent_[0])
                    indices_[s * {self.sub_tile_shape[0]} + ss] = 
                        params_.indice_ptr_[thread_offset_[0] + 
                            s * {self.tmap.delta[0]} + ss] * 
                            params_.stride_ * {self.dtype.nbytes_str()};
                else{{
                    indices_[s * {self.sub_tile_shape[0]} + ss] = 0;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        pointer_ += sizeof({self.dtype}) * offset;
        """)
        return code.arg("offset", str(self.long_index_t))

    def add_pointer_offset_python(self, offset: int):
        self.pointer_ += self.dtype.itemsize() * offset

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.code()
        with code.if_("is_residue_tile_"):
            if self.last_residual:
                if self.advance_axis == 1:
                    code.raw(f"""
                    thread_offset_[{self.advance_axis}] += residue_offset_;
                    pointer_ += sizeof({self.dtype}) * residue_offset_;
                    """)
                else:
                    code.raw(f"""
                    thread_offset_[{self.advance_axis}] += residue_offset_;
                    """)
                    if not self.shuffle_in_stride:
                        code.raw(f"""
                        pointer_ += sizeof({self.dtype}) * params_.stride_ * residue_offset_;
                        """)

            else:
                if self.advance_axis == 1:
                    code.raw(f"""
                    thread_offset_[{self.advance_axis}] -= residue_offset_;
                    pointer_ -= sizeof({self.dtype}) * residue_offset_;
                    """)
                else:
                    code.raw(f"""
                    thread_offset_[{self.advance_axis}] -= residue_offset_;
                    """)
                    if not self.shuffle_in_stride:
                        code.raw(f"""
                        pointer_ -= sizeof({self.dtype}) * params_.stride_ * residue_offset_;
                        """)

            if self.advance_axis == 1:
                code.raw(f"""
                compute_predicates_(extent_, true);
                pointer_ += {self.param_class.inc_advance_static} * (num_tile - 1);
                """)
            else:
                code.raw(f"""
                compute_predicates_(extent_, true);
                """)
                if not self.shuffle_in_stride:
                    code.raw(f"""
                    pointer_ += params_.inc_advance_ * (num_tile - 1);
                    """)

        with code.else_():
            if self.advance_axis == 1:
                code.raw(f"""
                pointer_ += {self.param_class.inc_advance_static} * num_tile;
                """)
            else:
                if self.shuffle_in_stride:
                    code.raw(f"""
                    thread_offset_[0] += {self.tile_shape[0]} * num_tile;
                    """)
                else:
                    code.raw(f"""
                    pointer_ += params_.inc_advance_ * num_tile;
                    """)
        code.raw(f"""
        is_residue_tile_ = false;
        """)
        if self.shuffle_in_stride and self.advance_axis == 0:
            code.raw(f"""
            update_indices();
            """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num_tile: int):
        if (self.is_residue_tile_):
            # we only need calculate mask once because
            if (self.last_residual):
                self.thread_offset_ += self.residue_offset_
                self.pointer_ += self.dtype.itemsize() * (
                    self.params_.stride_ * self.residue_offset_[0] +
                    self.residue_offset_[1])
            else:
                self.thread_offset_ -= self.residue_offset_
                self.pointer_ -= self.dtype.itemsize() * (
                    self.params_.stride_ * self.residue_offset_[0] +
                    self.residue_offset_[1])

            self.compute_predicates_python(self.extent_, True)
            # residual tile has been added.
            self.pointer_ += self.params_.inc_advance_ * (num_tile - 1)
        else:
            self.pointer_ += self.params_.inc_advance_ * num_tile

        self.is_residue_tile_ = False

    @pccm.cuda.member_function(name="get",
                               device=True,
                               const=True,
                               forceinline=True)
    def get(self):
        contig = 1
        strided = 0
        code = pccm.code()
        const = "const" if self.read_only else ""
        if self.sub_tile_shape[strided] == 1:
            indice_s = "indices_[s] + " if self.shuffle_in_stride else ""
            code.raw(f"""
            return reinterpret_cast<{const} {self.access_t} *>(
                    pointer_ + {indice_s}
                    (c * {self.iteration_delta[contig]}) *
                        sizeof({self.dtype})) +
                v;
            """).arg("s,c,v", "int")

        else:
            code.arg("s,c,ss,v", "int")
            if not self.shuffle_in_stride:
                code.raw(f"""
                return reinterpret_cast<{const} {self.access_t} *>(
                        pointer_ +
                        (ss * params_.stride_ + c * {self.iteration_delta[contig]}) *
                            sizeof({self.dtype})) +
                    v;
                """)
            else:
                code.raw(f"""
                return reinterpret_cast<{const} {self.access_t} *>(
                        pointer_ + indices_[s * {self.sub_tile_shape[0]} + ss] + 
                        (c * {self.iteration_delta[contig]}) *
                            sizeof({self.dtype})) +
                    v;
                """)
        return code.ret(f"{const} {self.access_t} *")

    def get_python(self, c: int, ss: int, v: int) -> ArrayPtr:
        contig = 1
        strided = 0
        ptr = (self.pointer_ +
               (ss * self.params_.stride_ + c * self.iteration_delta[contig]) *
               self.dtype.itemsize())  # type: ArrayPtr
        return ptr.change_access_size(self.num_sub_access) + v

    @pccm.cuda.member_function(device=True, forceinline=True)
    def inc_stride(self):
        return pccm.FunctionCode(f"pointer_ += params_.inc_strided_; ")

    def inc_stride_python(self):
        self.pointer_ += self.params_.inc_strided_

    @pccm.cuda.member_function(device=True, forceinline=True)
    def end_iter(self):
        # back to initial location?
        if self.advance_axis == 1:
            return pccm.FunctionCode(f"""
            pointer_ += params_.inc_next_ - {self.param_class.inc_advance_static};
            """)
        else:
            return pccm.FunctionCode(f"""
            pointer_ += params_.inc_next_;
            pointer_ -= params_.inc_advance_;
            """)

    def end_iter_python(self):
        self.pointer_ += self.params_.inc_next_
        self.pointer_ -= self.params_.inc_advance_

    @pccm.cuda.member_function(device=True, forceinline=True)
    def valid(self):
        contig = 1
        strided = 0
        code = pccm.code()
        if self.sub_tile_shape[0] == 1:
            code.raw(f"""
            int scalar_index =
                s * {self.thread_tensor_stride[0]} + c * {self.thread_tensor_stride[1]} +
                v * {self.thread_tensor_stride[3]};
            """)
            code.arg("s,c,v", "int")
        else:
            code.raw(f"""
            int scalar_index =
                s * {self.thread_tensor_stride[0]} + c * {self.thread_tensor_stride[1]} +
                ss * {self.thread_tensor_stride[2]} + v * {self.thread_tensor_stride[3]};
            """)
            code.arg("s,c,ss,v", "int")

        code.raw(f"""
        int word_idx = scalar_index / {self.mask_tensor_shape[1]};
        int residual = scalar_index % {self.mask_tensor_shape[1]};
        int byte_idx = residual / {self.mask_tensor_shape[2]};
        int bit_idx = residual % {self.mask_tensor_shape[2]};
        bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
        return pred;
        """)
        return code.ret(f"bool")

    def valid_python(self, s: int, c: int, ss: int, v: int):
        scalar_index = (s * self.thread_tensor_stride[0] +
                        c * self.thread_tensor_stride[1] +
                        ss * self.thread_tensor_stride[2] +
                        v * self.thread_tensor_stride[3])
        word_idx = scalar_index // self.mask_tensor_shape[1]
        residual = scalar_index % self.mask_tensor_shape[1]
        byte_idx = residual // self.mask_tensor_shape[2]
        bit_idx = residual % self.mask_tensor_shape[2]
        pred = (self.predicates_[word_idx] & (1 <<
                                              (byte_idx * 8 + bit_idx))) != 0
        return pred

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask(self):
        return pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.num_pred_32}; ++i) {{
            predicates_[i] = 0u;
        }}
        """)

    def clear_mask_python(self):
        self.predicates_[:] = 0

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        tile_increment(1);
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        self.tile_increment_python(1)
        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, pointer_offset * sizeof({self.dtype}));
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))

        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        if self.read_only:
            return pccm.code()
        code = pccm.FunctionCode(f"""
        store_with_byte_offset(frag, pointer_offset * sizeof({self.dtype}));
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))

        return code

    def load_with_pointer_offset_python(self, frag: ArrayPtr,
                                        pointer_offset: int):
        self.load_with_byte_offset_python(
            frag, pointer_offset * self.dtype.itemsize())

    def loadstore_with_byte_offset_template(self, store: bool):
        """we don't handle transposed input for simt here because:
        if A is transposed (i.e. tile shape is [k, m])
        we load them to smem directly (always [k, m])
        if A isn't transposed, we need to transpose before save to smem.
        the problem is k is small enough (always < num_threads),
        so we don't need to perform any transpose operation for [1xn] -> [nx1].
        """
        contig = 1
        strided = 0
        code = pccm.cuda.PTXCode("")
        const_frag = "const" if store else ""
        const_mem = "" if store else "const"

        # print(self.thread_access_shape[strided], self.thread_access_shape[contig], self.access_per_vector, self.access_t)
        # print(self.tmap)
        code.raw(f"""
        {self.access_t} {const_frag}*frag_ptr = reinterpret_cast<{self.access_t} {const_frag} *>(&frag);
        """)
        io_ns_name = "cutlass::arch" if CUTLASS_MODE else "tv::gemm"
        if self.sub_tile_shape[0] == 1:
            # if not store:
            #     # with code.if_("((gemm_idx - 1) & 1) == 0"):
            #     with code.range_("s", self.thread_access_shape[strided], "TV_PRAGMA_UNROLL"):
            #         with code.asm_block() as asm:
            #             code.raw(f"""
            #             char {const_mem} *byte_ptr =
            #                 reinterpret_cast<char {const_mem} *>(get(s, 0, 0)) + byte_offset;
            #             bool pred = valid(s, 0, 0);

            #             """)
            #             reg = asm.ext_reg("(int)pred", RegDType.U32)
            #             g = asm.global_ptr("byte_ptr")

            #             with asm.pred_if("p", "ne", reg, 0):
            #                 asm.generic("prefetch.global.L2", [g])

            with code.range_("s", self.thread_access_shape[strided],
                             "TV_PRAGMA_UNROLL"):
                with code.range_("c", self.thread_access_shape[contig],
                                 "TV_PRAGMA_UNROLL"):
                    with code.range_("v", self.access_per_vector,
                                     "TV_PRAGMA_UNROLL"):
                        code.raw(f"""
                        int idx =
                            s * {self.thread_tensor_stride[0]} + c * {self.thread_tensor_stride[1]} +
                            v * {self.thread_tensor_stride[3]};
                        char {const_mem} *byte_ptr =
                            reinterpret_cast<char {const_mem} *>(get(s, c, v)) + byte_offset;
                        {self.access_t} {const_mem} *access_ptr =
                            reinterpret_cast<{self.access_t} {const_mem} *>(byte_ptr);
                        """)
                        if store:
                            code.raw(f"""
                            {io_ns_name}::global_store<{self.access_t}, sizeof({self.access_t})>(
                                frag_ptr[idx], access_ptr, valid(s, c, v));
                            """)
                        else:
                            code.raw(f"""
                            GlobalLoad::run(frag_ptr[idx], access_ptr, valid(s, c, v));
                            // {io_ns_name}::global_load<{self.access_t}, sizeof({self.access_t})>(
                            //    frag_ptr[idx], access_ptr, valid(s, c, v));
                            """)

                if not self.shuffle_in_stride:
                    code.raw(f"""
                    if (s != {self.thread_access_shape[strided]} - 1) {{
                        inc_stride();
                    }}
                    """)
            if not self.shuffle_in_stride:
                code.raw(f"end_iter();")
        else:
            with code.range_("s", self.thread_access_shape[strided],
                             "TV_PRAGMA_UNROLL"):
                with code.range_("c", self.thread_access_shape[contig],
                                 "TV_PRAGMA_UNROLL"):
                    with code.range_("ss", self.sub_tile_shape[0],
                                     "TV_PRAGMA_UNROLL"):
                        with code.range_("v", self.access_per_vector,
                                         "TV_PRAGMA_UNROLL"):
                            code.raw(f"""
                            int idx =
                                s * {self.thread_tensor_stride[0]} + c * {self.thread_tensor_stride[1]} +
                                ss * {self.thread_tensor_stride[2]} + v * {self.thread_tensor_stride[3]};
                            char {const_mem} *byte_ptr =
                                reinterpret_cast<char {const_mem} *>(get(s, c, ss, v)) + byte_offset;
                            {self.access_t} {const_mem} *access_ptr =
                                reinterpret_cast<{self.access_t} {const_mem} *>(byte_ptr);
                            """)
                            if store:
                                code.raw(f"""
                                {io_ns_name}::global_store<{self.access_t}, sizeof({self.access_t})>(
                                    frag_ptr[idx], access_ptr, valid(s, c, ss, v));
                                """)
                            else:
                                code.raw(f"""
                                {io_ns_name}::global_load<{self.access_t}, sizeof({self.access_t})>(
                                    frag_ptr[idx], access_ptr, valid(s, c, ss, v));
                                """)

                if not self.shuffle_in_stride:
                    code.raw(f"""
                    if (s != {self.thread_access_shape[strided]} - 1) {{
                        inc_stride();
                    }}
                """)
            if not self.shuffle_in_stride:
                code.raw(f"end_iter();")
        if self.transpose_load:
            code.raw(f"""
            using SubTileShape = tv::mp_list_int<{self.sub_tile_shape[0]}, {self.sub_tile_shape[1]}>;
            tv::gemm::transform::Transpose<{self.sub_tile_shape.prod()}, SubTileShape,
                                {self.dtype}>
                t;
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.thread_access_shape[strided]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.thread_access_shape[contig]}; ++c) {{
                    int idx = s * {self.thread_access_shape[contig]} + c;
                    t.transform(frag.data() + idx * {self.sub_tile_shape.prod()}, frag.data() + idx * {self.sub_tile_shape.prod()});
                }}
            }}
            """)
        code.arg("frag", f"{const_frag} {self.fragment_t}&").arg(
            "byte_offset", str(self.long_index_t))
        return code

    def can_multistage_load(self):
        return not self.transpose_load and self.sub_tile_shape[0] == 1 and self.access_per_vector == 1
        #  must be 2 of for it
    
    def enumurate_get_param(self, python = False):
        contig = 1
        strided = 0
        for s in range(self.thread_access_shape[strided]):
            for c in range(self.thread_access_shape[contig]):
                if python:
                    yield (s, c)
                else:
                    yield f"{s}, {c}"

    def load_ptr_with_param_python(self, s, c): # ret (ArrayPtr, bool: valid, ptr_addrs)
        assert(self.can_multistage_load())
        contig = 1
        strided = 0
        assert s < self.thread_access_shape[strided] and c < self.thread_access_shape[contig]
        byte_ptr = self.get_python(c, 0, 0).change_access_size(self.num_sub_access)
        valid = self.valid_python(s, c, 0, 0)
        ptr_addrs = np.full((self.num_sub_access,), -1, dtype=np.int32)
        ptr_addrs[:] = np.arange(byte_ptr.offset, byte_ptr.offset + self.num_sub_access)
        ret_pack = (byte_ptr, valid, ptr_addrs)
        if c == self.thread_access_shape[contig] - 1:  # should add inc
            if s != self.thread_access_shape[strided] - 1:
                self.inc_stride_python()
            else:
                self.end_iter_python()
        return ret_pack

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_ptr_with_param(self):
        code = pccm.FunctionCode()
        contig = 1
        strided = 0
        code.arg("s, c", "int")
        code.arg("valid_ref", "bool&")
        code.ret(f"const {self.access_t}*")
        if self.sub_tile_shape[0] == 1:
            code.raw("auto ret_ptr = get(s, c, 0);")
            code.raw("valid_ref = valid(s, c, 0);")
        else:
            code.raw("auto ret_ptr = get(s, c, 0, 0);")
            code.raw("valid_ref = valid(s, c, 0, 0);")
        if not self.shuffle_in_stride:
            code.raw(f"""
            if(c == {self.thread_access_shape[contig] - 1}){{
                if(s != {self.thread_access_shape[strided] - 1})
                    inc_stride();
                else
                    end_iter();
                }}
            """)
        code.raw("""
            return ret_ptr;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_byte_offset(self):
        return self.loadstore_with_byte_offset_template(False)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_byte_offset(self):
        if self.read_only:
            return pccm.code()
        return self.loadstore_with_byte_offset_template(True)

    def load_with_byte_offset_python(self, frag: ArrayPtr, byte_offset: int):
        frag_ptr = frag.change_access_size(self.num_sub_access)
        contig = 1
        strided = 0
        ptr_addrs = np.full((frag.length, ), -1, dtype=np.int32)
        for s in range(self.thread_access_shape[strided]):
            for c in range(self.thread_access_shape[contig]):
                for ss in range(self.sub_tile_shape[strided]):
                    for v in range(self.access_per_vector):
                        idx = (s * self.thread_tensor_stride[0] +
                               c * self.thread_tensor_stride[1] +
                               ss * self.thread_tensor_stride[2] +
                               v * self.thread_tensor_stride[3])
                        byte_ptr = self.get_python(
                            c, ss, v).change_access_byte_size(1) + byte_offset
                        access_ptr = byte_ptr.change_access_size(
                            self.num_sub_access)
                        if self.valid_python(s, c, ss, v):
                            frag_ptr[idx] = access_ptr[0]
                            ptr_addrs[idx * frag_ptr.access_size:(idx + 1) *
                                      frag_ptr.access_size] = np.arange(
                                          access_ptr.offset,
                                          access_ptr.offset +
                                          frag_ptr.access_size)

            if (s != self.thread_access_shape[strided] - 1):
                self.inc_stride_python()
        self.end_iter_python()
        subtile_prod = self.sub_tile_shape.prod()
        if self.transpose_load:
            for s in range(self.thread_access_shape[strided]):
                for c in range(self.thread_access_shape[contig]):
                    idx = s * self.thread_access_shape[contig] + c
                    data = frag.data.numpy_view()[idx *
                                                  subtile_prod:(idx + 1) *
                                                  subtile_prod].reshape(
                                                      *self.sub_tile_shape)
                    data_t = np.transpose(data, (1, 0))
                    frag.data.numpy_view()[idx * subtile_prod:(idx + 1) *
                                           subtile_prod] = data_t.reshape(-1)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        if self.read_only:
            return pccm.code()
        code = pccm.FunctionCode(f"""
        store_with_byte_offset(frag, 0);
        """)
        code.arg("frag", f"const {self.fragment_t}&")

        return code

    def load_python(self, frag: ArrayPtr):
        return self.load_with_byte_offset_python(frag, 0)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def compute_predicates_(self):
        contig = 1
        strided = 0
        if self.sub_tile_shape[0] == 1:
            code = pccm.FunctionCode(f"""
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.num_pred_32}; ++i) {{
                predicates_[i] = 0;
            }}
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.thread_access_shape[0]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.thread_access_shape[1]}; ++c) {{
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.access_per_vector}; ++v) {{
                        tv::array<int, 2> elem_coord{{
                            s * {self.iteration_delta[0]} + thread_offset_[0],
                            c * {self.iteration_delta[1]} + v * {self.num_sub_access} +
                                thread_offset_[1]}};
                        bool valid;
                        if (steady) {{
                            if ({self.advance_axis} == {contig}) {{
                                valid = elem_coord[{strided}] < extent[{strided}];
                            }} else {{
                                valid = elem_coord[{contig}] < extent[{contig}];
                            }}
                        }} else {{
                            valid = elem_coord[{strided}] < extent[{strided}] &&
                                    elem_coord[{contig}] < extent[{contig}];
                        }}
                        int scalar_index =
                            s * {self.thread_tensor_stride[0]} + c * {self.thread_tensor_stride[1]} +
                            v * {self.thread_tensor_stride[3]};
                        int word_idx = scalar_index / {self.mask_tensor_shape[1]};
                        int residual = scalar_index % {self.mask_tensor_shape[1]};
                        int byte_idx = residual / {self.mask_tensor_shape[2]};
                        int bit_idx = residual % {self.mask_tensor_shape[2]};
                        predicates_[word_idx] |=
                            (unsigned(valid) << (byte_idx * 8 + bit_idx));
                    }}
                }}
            }}
            """)
        else:
            code = pccm.FunctionCode(f"""
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.num_pred_32}; ++i) {{
                predicates_[i] = 0;
            }}
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.thread_access_shape[0]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.thread_access_shape[1]}; ++c) {{
                    TV_PRAGMA_UNROLL
                    for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss) {{
                        TV_PRAGMA_UNROLL
                        for (int v = 0; v < {self.access_per_vector}; ++v) {{
                            tv::array<int, 2> elem_coord{{
                                s * {self.iteration_delta[0]} + ss + thread_offset_[0],
                                c * {self.iteration_delta[1]} + v * {self.num_sub_access} +
                                    thread_offset_[1]}};
                            bool valid;
                            if (steady) {{
                                if ({self.advance_axis} == {contig}) {{
                                    valid = elem_coord[{strided}] < extent[{strided}];
                                }} else {{
                                    valid = elem_coord[{contig}] < extent[{contig}];
                                }}
                            }} else {{
                                valid = elem_coord[{strided}] < extent[{strided}] &&
                                        elem_coord[{contig}] < extent[{contig}];
                            }}
                            int scalar_index =
                                s * {self.thread_tensor_stride[0]} + c * {self.thread_tensor_stride[1]} +
                                ss * {self.thread_tensor_stride[2]} + v * {self.thread_tensor_stride[3]};
                            int word_idx = scalar_index / {self.mask_tensor_shape[1]};
                            int residual = scalar_index % {self.mask_tensor_shape[1]};
                            int byte_idx = residual / {self.mask_tensor_shape[2]};
                            int bit_idx = residual % {self.mask_tensor_shape[2]};
                            predicates_[word_idx] |=
                                (unsigned(valid) << (byte_idx * 8 + bit_idx));
                        }}
                    }}
                }}
            }}
            """)

        code.arg("extent", f"tv::array<int, 2>")
        code.arg("steady", f"bool", "false")
        return code

    def compute_predicates_python(self,
                                  extent: MetaArray[int],
                                  steady: bool = False):
        contig = 1
        strided = 0
        self.clear_mask_python()
        for s in range(self.thread_access_shape[strided]):
            for c in range(self.thread_access_shape[contig]):
                for ss in range(self.sub_tile_shape[strided]):
                    for v in range(self.access_per_vector):
                        elem_coord = seq(
                            s * self.iteration_delta[0] + ss +
                            self.thread_offset_[0],
                            c * self.iteration_delta[1] +
                            v * self.num_sub_access + self.thread_offset_[1])
                        valid = False
                        if (steady):
                            if (self.advance_axis == contig):
                                valid = elem_coord[strided] < extent[strided]
                            else:
                                valid = elem_coord[contig] < extent[contig]
                        else:
                            valid = elem_coord[strided] < extent[
                                strided] and elem_coord[contig] < extent[contig]
                        scalar_index = (s * self.thread_tensor_stride[0] +
                                        c * self.thread_tensor_stride[1] +
                                        ss * self.thread_tensor_stride[2] +
                                        v * self.thread_tensor_stride[3])
                        word_idx = scalar_index // self.mask_tensor_shape[1]
                        residual = scalar_index % self.mask_tensor_shape[1]
                        byte_idx = residual // self.mask_tensor_shape[2]
                        bit_idx = residual % self.mask_tensor_shape[2]
                        self.predicates_[word_idx] |= (
                            int(valid) << (byte_idx * 8 + bit_idx))
