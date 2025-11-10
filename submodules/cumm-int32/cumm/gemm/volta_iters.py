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

from typing import List

import numpy as np
import pccm

from cumm import dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers
from cumm.gemm import bases, constants, layout, layout_tensorop, thread_map
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.thread_map import PitchLinearWarpRaked


def div_up(a, b):
    return (a + b - 1) // b


class VoltaSmemTileIteratorCrosswise(bases.GemmSmemIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape_km: MetaArray[int],
                 tmap: PitchLinearWarpRaked, num_stage: int):
        self.layout = layout_tensorop.VoltaTensorOpCrosswise(dtype.bitsize())
        super().__init__(dtype, tmap.element_per_acc * tmap.iterations.prod(),
                         self.layout.element_per_acc)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)

        self.tile_shape_km = tile_shape_km
        self.tmap = tmap
        self.num_stage = num_stage

        self.add_param_class("tmap", tmap, "ThreadMap")
        self.add_param_class("layout", self.layout, "Layout")

        self.pointer_count = 2 if self.tmap.iterations[0] > 1 else 1
        self.iter_per_acc = tmap.element_per_acc // self.layout.element_per_acc
        self.contig_element_per_line = 4
        self.stride_of_volta_block = self.contig_element_per_line
        self.line_size = tile_shape_km[
            1] * self.stride_of_volta_block // self.layout.element_per_acc
        self.smem_vis_shape = [tile_shape_km[0] * num_stage, tile_shape_km[1]]

        self.alignment = dtype.itemsize() * self.element_per_acc

        self.add_member("pointers_",
                        self.access_pointer,
                        array=f"[{self.pointer_count}]")
        # cudasim members
        self.pointers_: List[ArrayPtr] = [None] * self.pointer_count

    def get_smem_vis_shape(self) -> MetaArray[int]:
        return seq(self.smem_vis_shape[0], self.smem_vis_shape[1])

    def python_ctor(self, stride: int, ptr: ArrayPtr, thread_id: int):
        new_obj = VoltaSmemTileIteratorCrosswise(self.dtype,
                                                 self.tile_shape_km, self.tmap,
                                                 self.num_stage)
        l = new_obj.layout.python_ctor(stride)
        thread_offset_base = new_obj.tmap.initial_offset_python(thread_id)
        for i in range(self.pointer_count):
            new_obj.pointers_[i] = (
                ptr + l(thread_offset_base[0] + i * self.tmap.warp_shape[0],
                        thread_offset_base[1])).change_access_size(
                            self.element_per_acc)
        return new_obj

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        auto layout = Layout(stride);
        auto thread_offset_base = ThreadMap::initial_offset(thread_id);
        // int offs[2];
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] = reinterpret_cast<{self.access_pointer}>(
                ptr + layout(thread_offset_base[0] + i * {self.tmap.warp_shape[0]},
                            thread_offset_base[1]));
        }}
        """)
        code.arg("stride", "int")
        code.arg("ptr", self.pointer)
        code.arg("thread_id", "int")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] +=
                num_tile * {self.tile_shape_km.prod()} / {self.element_per_acc};
        }}
        """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        for i in range(self.pointer_count):
            self.pointers_[i] += num * self.tile_shape_km.prod(
            ) // self.element_per_acc

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] += {self.tile_shape_km.prod()} / {self.element_per_acc};
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        for i in range(self.pointer_count):
            self.pointers_[i] += self.tile_shape_km.prod(
            ) // self.element_per_acc

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        {self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);

        {self.index_t} vec_pointer_offset = pointer_offset / {self.layout.element_per_acc};

        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.tmap.iterations[0]}; ++s) {{
            // TODO remove this
            {self.access_pointer} access_ptr = pointers_[(s & 1) ^ ((s >> 1) & 1)];
            // check next tile
            access_ptr += 16 * (s / 2);
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.tmap.iterations[1]}; ++c) {{
                TV_PRAGMA_UNROLL
                for (int sub = 0; sub < {self.iter_per_acc}; ++sub) {{
                    int idx =
                        sub + {self.iter_per_acc} * (c + s * {self.tmap.iterations[1]});
                    int access_offset =
                        c * {self.tmap.delta[1]} * {self.tile_shape_km[1]} / {self.element_per_acc} +
                        vec_pointer_offset + sub * {self.line_size};
                    // the right part (next k iter) should be put to next k block.
                    // access_offset += sub * {self.line_size};
                    access_ptr[access_offset] = frag_ptr[idx];
                }}
            }}
        }}
        """)
        code.arg("frag",
                 f"{self.fragment_t} const&").arg("pointer_offset",
                                                  str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        vec_pointer_offset = pointer_offset // self.layout.element_per_acc
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        for s in range(self.tmap.iterations[0]):
            access_ptr = self.pointers_[(s & 1) ^ ((s >> 1) & 1)].shadow_copy()
            access_ptr += 16 * (s // 2)
            for c in range(self.tmap.iterations[1]):
                for sub in range(self.iter_per_acc):
                    idx = sub + self.iter_per_acc * (
                        c + s * self.tmap.iterations[1])
                    access_offset = (
                        c * self.tmap.delta[1] * self.tile_shape_km[1] //
                        self.element_per_acc + vec_pointer_offset +
                        sub * self.line_size)
                    # the right part (next k iter) should be put to next k block.
                    # access_offset += sub * self.line_size
                    await checkers.smem_bank_conflicit_check(
                        access_ptr, access_offset)

                    access_ptr[access_offset] = frag_ptr[idx]
                    ptr_addrs[idx * frag_ptr.access_size:(idx + 1) *
                              frag_ptr.access_size] = np.arange(
                                  (access_ptr + access_offset).offset,
                                  (access_ptr + access_offset).offset +
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


class VoltaSmemTileIteratorCongruous(bases.GemmSmemIterator):
    def __init__(self, dtype: dtypes.DType, operand_a: bool,
                 tile_shape_km: MetaArray[int], tmap: PitchLinearWarpRaked,
                 num_stage: int):
        self.layout = layout_tensorop.VoltaTensorOpCongruous(
            operand_a, dtype.bitsize())
        super().__init__(dtype, tmap.element_per_acc * tmap.iterations.prod(),
                         self.layout.element_per_acc)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.operand_a = operand_a
        self.tile_shape_km = tile_shape_km
        self.tmap = tmap
        self.num_stage = num_stage

        self.add_param_class("tmap", tmap, "ThreadMap")
        self.add_param_class("layout", self.layout, "Layout")

        self.pointer_count = 2 if self.tmap.iterations[0] > 1 else 1
        self.iter_per_acc = tmap.element_per_acc // self.layout.element_per_acc
        self.contig_element_per_line = 4
        self.stride_of_volta_block = self.contig_element_per_line
        self.line_size = tile_shape_km[
            1] * self.stride_of_volta_block // self.layout.element_per_acc
        self.smem_vis_shape = [tile_shape_km[0] * num_stage, tile_shape_km[1]]

        self.alignment = dtype.itemsize() * self.element_per_acc
        self.add_member("pointers_",
                        self.access_pointer,
                        array=f"[{self.pointer_count}]")
        # cudasim members
        self.pointers_: List[ArrayPtr] = [None] * self.pointer_count

    def get_smem_vis_shape(self) -> MetaArray[int]:
        return seq(self.smem_vis_shape[0], self.smem_vis_shape[1])

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode(f"""
        auto layout = Layout(stride);
        auto thread_offset_base = ThreadMap::initial_offset(thread_id);
        // int offs[2];
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] = reinterpret_cast<{self.access_pointer}>(
                ptr + layout(thread_offset_base[0] + i * {self.tmap.warp_shape[0]},
                            thread_offset_base[1]));
        }}
        """)
        code.arg("stride", "int")
        code.arg("ptr", self.pointer)
        code.arg("thread_id", "int")
        return code

    def python_ctor(self, stride: int, ptr: ArrayPtr, thread_id: int):
        new_obj = VoltaSmemTileIteratorCongruous(self.dtype, self.operand_a,
                                                 self.tile_shape_km, self.tmap,
                                                 self.num_stage)
        l = new_obj.layout.python_ctor(stride)
        thread_offset_base = new_obj.tmap.initial_offset_python(thread_id)
        for i in range(self.pointer_count):
            new_obj.pointers_[i] = (
                ptr + l(thread_offset_base[0] + i * self.tmap.warp_shape[0],
                        thread_offset_base[1])).change_access_size(
                            self.element_per_acc)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] +=
                num_tile * {self.tile_shape_km.prod()} / {self.element_per_acc};
        }}
        """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        for i in range(self.pointer_count):
            self.pointers_[i] += num * self.tile_shape_km.prod(
            ) // self.element_per_acc

    def increment_python(self):
        for i in range(self.pointer_count):
            self.pointers_[i] += self.tile_shape_km.prod(
            ) // self.element_per_acc

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] += {self.tile_shape_km.prod()} / {self.element_per_acc};
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        {self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);

        {self.index_t} vec_pointer_offset = pointer_offset / {self.layout.element_per_acc};

        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.tmap.iterations[0]}; ++s) {{
            // TODO remove this
            {self.access_pointer} access_ptr = pointers_[s & 1];
            // check next tile
            int stride_idx = (s & ~1);
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.tmap.iterations[1]}; ++c) {{
                int idx = c + s * {self.tmap.iterations[1]};
                int access_offset =
                    stride_idx * {self.tmap.delta[0]} * {self.tile_shape_km[1]} / {self.tmap.element_per_acc} +
                    c * {self.tmap.delta[1]} / {self.tmap.element_per_acc} +
                    vec_pointer_offset;
                // tv::printf2_block_once(threadIdx.x, s, c, access_offset);

                access_ptr[access_offset] = frag_ptr[idx];
            }}
        }}
        """)
        code.arg("frag",
                 f"{self.fragment_t} const&").arg("pointer_offset",
                                                  str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               pointer_offset: int):
        frag_ptr = frag.change_access_size(self.element_per_acc)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        vec_pointer_offset = pointer_offset // self.layout.element_per_acc
        for s in range(self.tmap.iterations[0]):
            access_ptr = self.pointers_[(s & 1)].shadow_copy()
            stride_idx = (s & ~1)
            for c in range(self.tmap.iterations[1]):
                idx = c + s * self.tmap.iterations[1]
                access_offset = (
                    stride_idx * self.tmap.delta[0] * self.tile_shape_km[1] //
                    self.tmap.element_per_acc +
                    c * self.tmap.delta[1] // self.tmap.element_per_acc +
                    vec_pointer_offset)
                # the right part (next k iter) should be put to next k block.
                # access_offset += sub * self.line_size
                await checkers.smem_bank_conflicit_check(
                    access_ptr, access_offset)

                access_ptr[access_offset] = frag_ptr[idx]
                ptr_addrs[idx * frag_ptr.access_size:(idx + 1) *
                          frag_ptr.access_size] = np.arange(
                              (access_ptr + access_offset).offset,
                              (access_ptr + access_offset).offset +
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


class VoltaWarpTileIteratorCrosswise(bases.GemmWarpIterator):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape_km: MetaArray[int],
                 warp_tile_shape_km: MetaArray[int],
                 left: bool = False):
        self.advance_axis = 0 if left else 1
        self.interleaved_wmma_shape = metaseq(32, 32, 4)
        self.inst_shape = metaseq(16, 16, 4)

        element_count = warp_tile_shape_km[1] // self.interleaved_wmma_shape[
            self.advance_axis]
        element_count *= self.interleaved_wmma_shape[2]
        element_count *= self.interleaved_wmma_shape[
            self.advance_axis] // self.inst_shape[self.advance_axis]

        super().__init__(dtype, element_count, 8)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)

        self.tile_shape_km = tile_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km
        self.left = left

        self.num_warp_gemm_iters = warp_tile_shape_km[
            0] // self.interleaved_wmma_shape[2]
        self.contig_element_per_line = 4
        self.stride_of_volta_block = self.contig_element_per_line
        self.line_size = tile_shape_km[
            1] * self.stride_of_volta_block // self.element_per_acc
        self.lds_shape = metaseq(self.interleaved_wmma_shape[0], 1)
        self.lds_iterations = metaseq(
            warp_tile_shape_km[1] // self.lds_shape[0], 1)
        self.stride_in_access = tile_shape_km[1] // self.element_per_acc
        self.add_member("pointer_", self.const_access_pointer)
        self.add_member("byte_offset_, wmma_k_index_", self.index_t)

        # cudasim members
        self.pointer_: ArrayPtr = None
        self.byte_offset_ = -1
        self.wmma_k_index_ = -1

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_idx_k, warp_idx_mn, lane_idx", "int")
        code.ctor_init("wmma_k_index_", "0")
        code.raw(f"""
        int quad = (lane_idx / 4);
        int lane_in_quad = (lane_idx % 4);
        int access_contiguous;
        """)
        if self.left:
            code.raw("""
            // swizzle id: tid[4]|tid[1:0]|(tid[2]^tid[4])
            access_contiguous = ((quad & 0x4) << 1) + ((lane_in_quad) << 1) +
                                ((quad & 0x1) ^ ((quad & 0x4) >> 2));
            """)
        else:
            code.raw("""
            // swizzle id: tid[4]|tid[1:0]|tid[3]
            access_contiguous = ((quad & 0x4) << 1) + (lane_in_quad << 1) +
                                ((quad & 0x2) >> 1 ^ ((quad & 0x4) >> 2));
            """)
        code.raw(f"""
        byte_offset_ = access_contiguous * sizeof({self.dtype}) * {self.element_per_acc};
        pointer_ = reinterpret_cast<{self.const_access_pointer}>(ptr);
        add_warp_offset(warp_idx_k, warp_idx_mn);
        """)
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_mn: int, lane_idx: int):
        new_obj = VoltaWarpTileIteratorCrosswise(self.dtype,
                                                 self.tile_shape_km,
                                                 self.warp_tile_shape_km,
                                                 self.left)
        quad = lane_idx // 4
        lane_in_quad = (lane_idx % 4)
        if self.left:
            # swizzle id: tid[4]|tid[1:0]|(tid[2]^tid[4])
            access_contiguous = (((quad & 0x4) << 1) + ((lane_in_quad) << 1) +
                                 ((quad & 0x1) ^ ((quad & 0x4) >> 2)))
        else:
            # swizzle id: tid[4]|tid[1:0]|tid[3]
            access_contiguous = (((quad & 0x4) << 1) + (lane_in_quad << 1) +
                                 ((quad & 0x2) >> 1 ^ ((quad & 0x4) >> 2)))
        new_obj.byte_offset_ = access_contiguous * self.dtype.itemsize(
        ) * self.element_per_acc
        new_obj.pointer_ = ptr.change_access_size(self.element_per_acc)
        new_obj.add_warp_offset_python(warp_idx_k, warp_idx_mn)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_warp_offset(self):
        code = pccm.FunctionCode(f"""
        int mn_offset = warp_idx_mn;
        int k_offset = {self.num_warp_gemm_iters} * warp_idx_k;
        // kTileShapeKM: [K, M|N]
        // TODO better offset
        auto offset = k_offset * {self.interleaved_wmma_shape[2]} * {self.stride_in_access} +
                    mn_offset * {self.warp_tile_shape_km[1]} * 4 / {self.element_per_acc};
        // printf2_block_once(threadIdx.x, offset);
        pointer_ += offset;
        """)
        return code.arg("warp_idx_k, warp_idx_mn", "int")

    def add_warp_offset_python(self, warp_idx_k, warp_idx_mn):
        mn_offset = warp_idx_mn
        k_offset = self.num_warp_gemm_iters * warp_idx_k
        # kTileShapeKM: [K, M|N]
        # TODO better offset
        offset = (
            k_offset * self.interleaved_wmma_shape[2] * self.stride_in_access +
            mn_offset * self.warp_tile_shape_km[1] * 4 // self.element_per_acc)
        # printf2_block_once(threadIdx.x, offset)
        self.pointer_ += offset

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        // this function is only called when move warp iter back to start offset.
        // so we need to reset wmma_k_index_
        wmma_k_index_ = 0;
        // tv::printf2_block_once("tile_increment_warp", threadIdx.x, kLineSize * num, kLineSize, num);
        pointer_ += {self.line_size} * num;
        """)
        return code.arg("num", "int")

    def tile_increment_python(self, num: int):
        self.wmma_k_index_ = 0

        self.pointer_ += self.line_size * num

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        wmma_k_index_ = (wmma_k_index_ + 1) & 7;
        // handle permute (i)
        if (wmma_k_index_ == 4 || wmma_k_index_ == 0) {{
            // ptr swapped in k = 4-7, so we 'swap' ptr here.
            // byte_offset_ -=(+=) self.sizeof_element * self.kElementsPerAccess
            byte_offset_ ^= 1 * sizeof({self.dtype}) * {self.element_per_acc};
        }}
        pointer_ += {self.line_size};
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        self.wmma_k_index_ = (self.wmma_k_index_ + 1) & 7
        if (self.wmma_k_index_ == 4 or self.wmma_k_index_ == 0):
            self.byte_offset_ ^= 1 * self.dtype.itemsize(
            ) * self.element_per_acc
        self.pointer_ += self.line_size
        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        {self.index_t} byte_offset = pointer_offset * sizeof({self.dtype});
        {self.access_pointer} dst_ptr = reinterpret_cast<{self.access_pointer}>(&frag);
        // kRow: 1, kCol: 2
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.lds_iterations[0]}; ++s) {{
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.lds_iterations[1]}; ++c) {{
                int idx = c + s * {self.lds_iterations[1]};
                {self.const_access_pointer} source_ptr =
                    pointer_ + {self.lds_shape[1]} * c * {self.line_size} + {self.lds_shape[0]} * s / 2;

                char const *source_byte_ptr =
                    reinterpret_cast<char const *>(source_ptr) + byte_offset +
                    byte_offset_;
                dst_ptr[idx] = *(reinterpret_cast<{self.const_access_pointer}>(source_byte_ptr));
                if (wmma_k_index_ & 0x2) {{
                    uint64_t *low = reinterpret_cast<uint64_t *>(&frag) + idx * 2;
                    uint64_t *high = reinterpret_cast<uint64_t *>(&frag) + idx * 2 + 1;
                    uint64_t tmp = *low;
                    *low = *high;
                    *high = tmp;
                }}
            }}
        }}
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    async def load_with_pointer_offset_python(self, frag: ArrayPtr,
                                              pointer_offset: int):
        byte_offset = pointer_offset * self.dtype.itemsize()
        dst_ptr = frag.change_access_size(self.element_per_acc)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        for s in range(self.lds_iterations[0]):
            for c in range(self.lds_iterations[1]):
                idx = c + s * self.lds_iterations[1]  # type: int
                source_ptr = self.pointer_ + self.lds_shape[
                    1] * c * self.line_size + self.lds_shape[
                        0] * s // 2  # type: ArrayPtr
                source_byte_ptr = source_ptr.change_access_byte_size(
                    1) + byte_offset + self.byte_offset_
                dst_ptr[idx] = source_byte_ptr.change_access_size(
                    self.element_per_acc)[0]
                ptr_addrs[idx * dst_ptr.access_size:(idx + 1) *
                          dst_ptr.access_size] = np.arange(
                              source_byte_ptr.offset,
                              source_byte_ptr.offset + dst_ptr.access_size)
                await checkers.smem_bank_conflicit_check(
                    source_byte_ptr.change_access_size(self.element_per_acc),
                    0)

                if self.wmma_k_index_ & 0x2:
                    frag_uint64 = frag.change_access_byte_size(8)
                    low = frag_uint64 + idx * 2
                    high = frag_uint64 + idx * 2 + 1
                    tmp = low.copy()
                    low[0] = high[0]
                    high[0] = tmp[0]
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
        code = pccm.FunctionCode("wmma_k_index_ = wmma_k;")
        code.arg("wmma_k", "int")
        return code

    def set_wmma_k_index_python(self, wmma_k: int):
        self.wmma_k_index_ = wmma_k


class VoltaWarpTileIteratorCongruous(bases.GemmWarpIterator):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape_km: MetaArray[int],
                 warp_tile_shape_km: MetaArray[int],
                 left: bool = False):
        self.interleaved_wmma_shape = metaseq(32, 32, 4)
        self.advance_axis = 0 if left else 1
        self.inst_shape = metaseq(16, 16, 4)

        element_count = warp_tile_shape_km[1] // self.interleaved_wmma_shape[
            self.advance_axis]
        element_count *= self.interleaved_wmma_shape[2]
        element_count *= self.interleaved_wmma_shape[
            self.advance_axis] // self.inst_shape[self.advance_axis]

        super().__init__(dtype, element_count, 8)
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)

        self.tile_shape_km = tile_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km
        self.left = left

        self.num_warp_gemm_iters = warp_tile_shape_km[
            0] // self.interleaved_wmma_shape[2]
        self.contig_element_per_line = 4
        self.stride_of_volta_block = self.contig_element_per_line
        self.line_size = tile_shape_km[
            1] * self.stride_of_volta_block // self.element_per_acc
        self.lds_shape = metaseq(self.interleaved_wmma_shape[2],
                                 self.interleaved_wmma_shape[0])
        self.lds_iterations = metaseq(
            warp_tile_shape_km[1] // self.lds_shape[1],
            self.interleaved_wmma_shape[2] // self.lds_shape[0])
        if not left:
            self.lds_iterations = self.lds_iterations[::
                                                      -1]  # type: MetaArray[int]
        self.pointer_count = 2 if left else 1
        self.stride_in_access = tile_shape_km[1] // self.element_per_acc
        self.add_member("pointers_",
                        self.const_access_pointer,
                        array=f"[{self.pointer_count}]")
        self.add_member("byte_offset_", self.index_t)

        # cudasim members
        self.pointers_: List[ArrayPtr] = [None] * self.pointer_count
        self.byte_offset_ = -1

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_idx_k, warp_idx_mn, lane_idx", "int")
        if self.left:
            code.raw(f"""
            int vec_row = (lane_idx >> 4);       // tid[4]
            int vec_col = ((lane_idx & 4) >> 2); // tid[2]
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.pointer_count}; ++i) {{
                if (i == 1) {{
                    vec_row |= 2;
                }}
                int access_contiguous_idx = (vec_col << 2) | ((lane_idx & 3) ^ vec_row);
                int access_contiguous = access_contiguous_idx;

                int access_strided = vec_row;
                pointers_[i] = reinterpret_cast<{self.const_access_pointer}>(ptr) +
                            access_contiguous + access_strided * {self.stride_in_access};
            }}
            """)
        else:
            code.raw(f"""
            int access_strided = (lane_idx >> 3) & 0x3;
            int access_contiguous = ((lane_idx ^ (lane_idx >> 3)) & 0x3);
            pointers_[0] = reinterpret_cast<{self.const_access_pointer}>(ptr) +
                            access_contiguous + access_strided * {self.stride_in_access};
            """)
        code.raw(f"""
        add_warp_offset(warp_idx_k, warp_idx_mn);
        """)
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_mn: int, lane_idx: int):
        new_obj = VoltaWarpTileIteratorCongruous(self.dtype,
                                                 self.tile_shape_km,
                                                 self.warp_tile_shape_km,
                                                 self.left)
        if self.left:
            vec_row = (lane_idx >> 4)  # tid[4]
            vec_col = ((lane_idx & 4) >> 2)  # tid[2]
            for i in range(self.pointer_count):
                if (i == 1):
                    vec_row |= 2

                access_contiguous_idx = (vec_col << 2) | (
                    (lane_idx & 3) ^ vec_row)
                access_contiguous = access_contiguous_idx

                access_strided = vec_row
                new_obj.pointers_[i] = (
                    ptr.change_access_size(self.element_per_acc) +
                    access_contiguous +
                    access_strided * new_obj.stride_in_access)
        else:

            access_strided = (lane_idx >> 3) & 0x3
            access_contiguous = ((lane_idx ^ (lane_idx >> 3)) & 0x3)
            new_obj.pointers_[0] = (
                ptr.change_access_size(self.element_per_acc) +
                access_contiguous + access_strided * new_obj.stride_in_access)
        new_obj.byte_offset_ = 0
        new_obj.add_warp_offset_python(warp_idx_k, warp_idx_mn)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_warp_offset(self):
        code = pccm.FunctionCode(f"""
        int mn_offset = warp_idx_mn;
        int k_offset = {self.num_warp_gemm_iters} * warp_idx_k;
        // TODO why?
        if ({pccm.boolean(self.left)}) {{
            if ({self.warp_tile_shape_km[1]} == {self.lds_shape[1]}) {{
                if (mn_offset % 2) {{
                    auto tmp_pointer = pointers_[0];
                    pointers_[0] = pointers_[1];
                    pointers_[1] = tmp_pointer;
                }}
                mn_offset = mn_offset / 2 * 2;
            }}
        }}
        auto offset = k_offset * {self.stride_in_access} * {self.interleaved_wmma_shape[2]} *
                        {self.element_per_acc} +
                    mn_offset * {self.warp_tile_shape_km[1]};
        // if (!Left){{
        //   tv::printf2_block_once(threadIdx.x, offset);
        // }}
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] += offset / {self.element_per_acc};
        }}
        """)
        return code.arg("warp_idx_k, warp_idx_mn", "int")

    def add_warp_offset_python(self, warp_idx_k, warp_idx_mn):
        mn_offset = warp_idx_mn
        k_offset = self.num_warp_gemm_iters * warp_idx_k
        if self.left:
            if (self.warp_tile_shape_km[1] == self.lds_shape[1]):
                if (mn_offset % 2):
                    tmp_pointer = self.pointers_[0]
                    self.pointers_[0] = self.pointers_[1]
                    self.pointers_[1] = tmp_pointer

                mn_offset = mn_offset // 2 * 2

        offset = (k_offset * self.stride_in_access *
                  self.interleaved_wmma_shape[2] * self.element_per_acc +
                  mn_offset * self.warp_tile_shape_km[1])
        for i in range(self.pointer_count):
            self.pointers_[i] += offset // self.element_per_acc

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        // this function is only called when move warp iter back to start offset.
        // so we need to reset wmma_k_index_
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] += {self.stride_in_access} * {self.interleaved_wmma_shape[2]} * num;
        }}
        """)
        return code.arg("num", "int")

    def tile_increment_python(self, num: int):
        for i in range(self.pointer_count):
            self.pointers_[
                i] += num * self.stride_in_access * self.interleaved_wmma_shape[
                    2]

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointers_[i] += {self.stride_in_access} * {self.interleaved_wmma_shape[2]};
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        for i in range(self.pointer_count):
            self.pointers_[
                i] += self.stride_in_access * self.interleaved_wmma_shape[2]

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        // pointer_offset: element unit
        {self.index_t} byte_offset = pointer_offset * sizeof({self.dtype});
        {self.access_pointer} dst_ptr = reinterpret_cast<{self.access_pointer}>(&frag);
        // kRow: 1, kCol: 2
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.lds_iterations[0]}; ++s) {{
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.lds_iterations[1]}; ++c) {{
                int idx = c + s * {self.lds_iterations[1]};
                {self.const_access_pointer} source_ptr;
                if ({pccm.boolean(self.left)}) {{
                    source_ptr = pointers_[s & 1] + {self.lds_shape[1]} * c +
                                {self.lds_shape[0]} * (s / 2) * {self.stride_in_access};
                }} else {{
                    source_ptr = pointers_[0] + {self.lds_shape[1]} / {self.element_per_acc} * c +
                                {self.lds_shape[0]} * s * {self.stride_in_access};
                }}
                char const *source_byte_ptr =
                    reinterpret_cast<char const *>(source_ptr) + byte_offset +
                    byte_offset_;
                // if (Left){{
                //   auto  ppp = reinterpret_cast<const_pointer>(source_byte_ptr);
                //   tv::printf2_block_once(threadIdx.x, s, c, 
                //     reinterpret_cast<AccessType const*> (source_byte_ptr) - pointer_bkp_, 
                //     int(ppp[0]), int(ppp[1]), int(ppp[2]), int(ppp[3]));
                // }}

                dst_ptr[idx] = *(reinterpret_cast<{self.const_access_pointer}>(source_byte_ptr));
            }}
        }}
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    async def load_with_pointer_offset_python(self, frag: ArrayPtr,
                                              pointer_offset: int):
        byte_offset = pointer_offset * self.dtype.itemsize()
        dst_ptr = frag.change_access_size(self.element_per_acc)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        for s in range(self.lds_iterations[0]):
            for c in range(self.lds_iterations[1]):
                idx = c + s * self.lds_iterations[1]  # type: int
                if self.left:
                    source_ptr = (self.pointers_[s & 1] +
                                  self.lds_shape[1] * c + self.lds_shape[0] *
                                  (s // 2) * self.stride_in_access)
                else:
                    source_ptr = (
                        self.pointers_[0] +
                        self.lds_shape[1] // self.element_per_acc * c +
                        self.lds_shape[0] * s * self.stride_in_access)
                source_byte_ptr = source_ptr.change_access_byte_size(
                    1) + byte_offset + self.byte_offset_
                dst_ptr[idx] = source_byte_ptr.change_access_size(
                    self.element_per_acc)[0]
                await checkers.smem_bank_conflicit_check(
                    source_byte_ptr.change_access_size(self.element_per_acc),
                    0)
                ptr_addrs[idx * dst_ptr.access_size:(idx + 1) *
                          dst_ptr.access_size] = np.arange(
                              source_byte_ptr.offset,
                              source_byte_ptr.offset + dst_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_kgroup_index(self):
        code = pccm.code()
        code.arg("wmma_k", "int")
        return code

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_pointer_offset_python(frag, 0)

    def set_wmma_k_index_python(self, wmma_k: int):
        return
