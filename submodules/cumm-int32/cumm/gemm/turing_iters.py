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
"""
Tensor Op Iterator

Goal: Load a SUB MATRIX for tensor op.
regular way will cause bank conflicit if we read a submatrix from a warp.

Example 1: Load a 8x8 submatrix
For Warp Raked Layout [8, 4]:
a warp read 4 matrix to smem. 

"""

from typing import List

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorView
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers
from cumm.gemm import arch, bases, constants, layout_tensorop, thread_map
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.thread_map import PitchLinearWarpRaked

from .turing_my_iters import MyTensorOpLayout

# def seq(*vals) -> np.ndarray:
#     return np.array([*vals], dtype=np.int64)


def div_up(a, b):
    return (a + b - 1) // b


class SmemTileIterator(bases.GemmSmemIterator):
    def __init__(self,
                 is_crosswise: bool,
                 dtype: dtypes.DType,
                 tile_shape_km: MetaArray[int],
                 my_layout: MyTensorOpLayout,
                 smem_layout: layout_tensorop.TensorOpMultiplicand,
                 tmap: PitchLinearWarpRaked,
                 alignment: int = -1,
                 crosswise: int = 0,
                 num_stage: int = 2):
        super().__init__(dtype,
                         tmap.iterations.prod() * smem_layout.element_per_acc,
                         smem_layout.element_per_acc)
        # cultass shape: mk
        # our shape: km
        # A crosswise: [64, 32], congruous: [32, 64]
        # , congruous: [32, 128]
        self.my_layout = my_layout
        self.is_crosswise = is_crosswise
        if is_crosswise:
            assert crosswise != 0
        else:
            crosswise = 128 // self.dtype.itemsize()
        self.tile_shape_km = tile_shape_km
        self.num_stage = num_stage
        if is_crosswise:
            # col major smem
            self.smem_vis_shape = [
                tile_shape_km[0], tile_shape_km[1] * num_stage
            ]
        else:
            self.smem_vis_shape = [
                tile_shape_km[0] * num_stage, tile_shape_km[1]
            ]
        ss = smem_layout.static_stride * smem_layout.factor
        self.smem_vis_shape = [
            tile_shape_km[0] * num_stage * tile_shape_km[1] // ss, ss
        ]
        self.tmap = tmap
        if alignment == -1:
            alignment = dtype.bitsize() * tmap.element_per_acc // 8
        self.alignment = alignment
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.crosswise = crosswise

        self.access_size_bits = 128
        self.pointer_count = 2 if tmap.iterations[0] > 1 else 1
        self.layout = smem_layout
        self.smem_stride = smem_layout.static_stride
        self.add_param_class("layout", self.layout, "Layout")
        # if is_crosswise:
        #     print(self.pointer_count, self.my_layout)
        # dtype_uint4_count = self.element_count * dtype.itemsize() // 16
        # self.fragment_t = f"tv::array<int4, {dtype_uint4_count}>"

        if is_crosswise:
            # Total number of sections.  The memory is divided into stages.  One stage
            # can store one tile.  Stage is divided into sections.  Interleaved layout
            # can have multiple sections in a stage.  The rest layout only has one section
            # in a stage.
            self.add_member("sections_, sections_per_stage_", "int")
        self.add_member("stride_", self.index_t)
        self.add_member("pointer_",
                        self.access_pointer,
                        array=f"[{self.pointer_count}]")

        self.add_member("byte_offset_", self.index_t)
        # self.add_member("iteration_contiguous_, iteration_strided_", "int")
        # print(self.element_count, self.element_per_acc, self.tmap)
        # raise NotImplementedError
        # cudasim members
        self.sections_ = 0
        self.sections_per_stage_ = 0
        self.pointer_: List[ArrayPtr] = [None] * self.pointer_count
        self.stride_ = 0
        self.byte_offset_ = 0

    def get_smem_vis_shape(self) -> MetaArray[int]:
        return seq(self.smem_vis_shape[0], self.smem_vis_shape[1])

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        # TODO remove this argument
        code.arg("stride", "int")
        code.arg("ptr", self.pointer)
        code.arg("thread_id", "int")
        if self.is_crosswise:
            # Total number of sections.  The memory is divided into stages.  One stage
            # can store one tile.  Stage is divided into sections.  Interleaved layout
            # can have multiple sections in a stage.  The rest layout only has one section
            # in a stage.
            # self.smem_stride == num_stage * tile_shape_km[0] for crosswise
            # sections_ == sections_per_stage_ * num_stage
            code.ctor_init("sections_",
                           f"{self.smem_stride // self.crosswise}")
            # the sections is divided in k axis
            code.ctor_init("sections_per_stage_",
                           f"{self.tile_shape_km[0] // self.crosswise}")
            # num_stage * tile_shape_km[0] * self.layout.factor
            code.ctor_init(
                "stride_",
                f"{self.smem_stride} * {self.layout.factor} / {self.element_per_acc}"
            )
        else:
            code.ctor_init("stride_",
                           f"{self.smem_stride} / {self.element_per_acc}")
        code.ctor_init("byte_offset_", "0")
        # code.ctor_init("iteration_contiguous_", "0")
        # code.ctor_init("iteration_strided_", "0")

        code.raw(f"""
        auto thread_offset_base = ThreadMap::initial_offset(thread_id);
        auto layout = Layout({self.smem_stride});
        // int offs[2];
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointer_[i] = reinterpret_cast<{self.access_pointer}>(
                ptr + layout(thread_offset_base[0] + i * {self.tmap.warp_shape[0]},
                            thread_offset_base[1]));
            // offs[i] = layout(thread_offset_base[0] + i * {self.tmap.warp_shape[0]},
            //                 thread_offset_base[1]);
            // pointer_bkp_[i] = reinterpret_cast<{self.access_pointer}>(ptr);
        }}
        """)
        # code.raw("""
        # tv::printf2_block_once(threadIdx.x, stride, thread_offset_base[0], thread_offset_base[1], offs[0], offs[1]);
        # """)
        return code

    def python_ctor(self, stride: int, ptr: ArrayPtr, thread_id: int):
        new_obj = SmemTileIterator(self.is_crosswise, self.dtype,
                                   self.tile_shape_km, self.my_layout,
                                   self.layout, self.tmap, self.alignment,
                                   self.crosswise, self.num_stage)
        if new_obj.is_crosswise:
            new_obj.sections_ = new_obj.smem_stride // new_obj.crosswise
            new_obj.sections_per_stage_ = new_obj.tile_shape_km[
                0] // new_obj.crosswise
            new_obj.stride_ = new_obj.smem_stride * new_obj.layout.factor // new_obj.element_per_acc
        else:
            new_obj.stride_ = new_obj.smem_stride // new_obj.element_per_acc
        l = new_obj.layout.python_ctor(new_obj.smem_stride)
        thread_offset_base = new_obj.tmap.initial_offset_python(thread_id)
        for i in range(new_obj.pointer_count):
            off = l(thread_offset_base[0] + i * new_obj.tmap.warp_shape[0],
                    thread_offset_base[1])
            # if myoff != off:
            #     print("OFFSET DIFFERENT!!", myoff, off)
            new_obj.pointer_[i] = (
                ptr + l(thread_offset_base[0] + i * new_obj.tmap.warp_shape[0],
                        thread_offset_base[1])).change_access_size(
                            new_obj.element_per_acc)
            # if cudasim.threadIdx().x == 8:
            #     print(self.is_crosswise, new_obj.tmap.warp_shape, new_obj.smem_stride, thread_offset_base, l(thread_offset_base[0] + i * new_obj.tmap.warp_shape[0],
            #             thread_offset_base[1]))
            #     print("thread_offset_base =", thread_offset_base)

        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = pccm.FunctionCode(f"""
        {self.access_pointer} access_ptr = pointer_[s & 1];
        int stride_idx = (s & ~1);
        """)
        if self.is_crosswise:
            code.raw(f"""
            // kCrosswise elements in the contiguous dimension would span to a
            // shared memory cache line.
            int access_offset =
                stride_idx * {self.tmap.delta[0]} * stride_ / {self.layout.factor} +
                c * ({self.layout.tile_shape[1] * self.tmap.delta[1] // self.crosswise});
            // tv::printf2_block_once(threadIdx.x, s, c, access_offset);

            """)
        else:
            code.raw(f"""
            int access_offset = 
                stride_idx * {self.tmap.delta[0]} * stride_ +
                c * {self.tmap.delta[1] // self.tmap.element_per_acc};
            """)

        code.raw(f"""
        
        char *access_byte_ptr =
            reinterpret_cast<char *>(access_ptr + access_offset);
        return reinterpret_cast<{self.access_pointer}>(access_byte_ptr + byte_offset_);
        """).arg("s, c", "int")
        return code.ret(f"{self.access_pointer}")

    def get_python(self, s: int, c: int):
        access_ptr = self.pointer_[s & 1]
        # if (self.is_crosswise):
        #     access_ptr = self.pointer_[0] + s * 4 * self.stride_

        stride_idx = s & ~1
        if self.is_crosswise:
            access_offset = (stride_idx * self.tmap.delta[0] * self.stride_ //
                             self.layout.factor + c *
                             (self.tmap.delta[1] // self.crosswise) *
                             self.layout.tile_shape[1])
        else:
            access_offset = (
                stride_idx * self.tmap.delta[0] * self.stride_ +
                c * self.tmap.delta[1] // self.tmap.element_per_acc)
        ptr = ((access_ptr + access_offset).change_access_byte_size(1) +
               self.byte_offset_)
        return ptr.change_access_size(self.element_per_acc)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(
            f"byte_offset_ += offset * sizeof({self.dtype});")
        code.arg("offset", self.long_index_t)
        return code

    def add_pointer_offset_python(self, offset: int):
        self.byte_offset_ += offset * self.dtype.itemsize()

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_tile_offset(self):
        if self.is_crosswise:
            code = pccm.FunctionCode(f"""
            add_pointer_offset(c * sections_per_stage_ * stride_ *
                                {self.tmap.element_per_acc} / sections_ +
                                // s * Shape::kStrided * stride_ *
                                s * {self.tile_shape_km[1]} * stride_ *
                                {self.layout.element_per_acc});


            """).arg("s, c", "int")
        else:
            code = pccm.FunctionCode(f"""
            add_pointer_offset(c * {self.tile_shape_km[1]} +
                                s * {self.tile_shape_km[0]} * stride_ *
                                {self.layout.element_per_acc});
            """).arg("s, c", "int")
        return code

    def add_tile_offset_python(self, s: int, c: int):
        if self.is_crosswise:
            self.add_pointer_offset_python(
                c * self.sections_per_stage_ * self.stride_ *
                self.tmap.element_per_acc // self.sections_ +
                s * self.tile_shape_km[1] * self.stride_ *
                self.layout.element_per_acc)
            if cudasim.threadIdx().x == 0:
                cudasim.debug_print(
                    f"ADD TILE OFFSET CROSSWISE {s},{c},{self.stride_}",
                    c * self.sections_per_stage_ * self.stride_ *
                    self.tmap.element_per_acc // self.sections_ +
                    s * self.tile_shape_km[1] * self.stride_ *
                    self.layout.element_per_acc)
        else:
            self.add_pointer_offset_python(c * self.tile_shape_km[1] +
                                           s * self.tile_shape_km[0] *
                                           self.stride_ *
                                           self.layout.element_per_acc)
            if cudasim.threadIdx().x == 0:
                cudasim.debug_print(
                    f"ADD TILE OFFSET {s},{c},{self.stride_}",
                    c * self.tile_shape_km[1] + s * self.tile_shape_km[0] *
                    self.stride_ * self.layout.element_per_acc)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        if self.is_crosswise:
            code = pccm.FunctionCode(f"""
            add_tile_offset(0, num_tile);
            """)
        else:
            code = pccm.FunctionCode(f"""
            add_tile_offset(num_tile, 0);
            """)

        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        if self.is_crosswise:
            self.add_tile_offset_python(0, num)
        else:
            self.add_tile_offset_python(num, 0)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        store_with_byte_offset(frag, pointer_offset * {self.dtype.bitsize()} / 8);
        """)
        code.arg("frag",
                 f"{self.fragment_t} const&").arg("pointer_offset",
                                                  str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               offset: int):
        return await self.store_with_byte_offset_python(
            frag, offset * self.dtype.itemsize())

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_byte_offset(self):
        code = pccm.FunctionCode(f"""
        {self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);

        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.tmap.iterations[0]}; ++s) {{
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.tmap.iterations[1]}; ++c) {{
                int access_idx = c + s * {self.tmap.iterations[1]};
                char *byte_ptr = reinterpret_cast<char *>(get(s, c)) + byte_offset;
                {self.access_pointer} access_ptr = reinterpret_cast<{self.access_pointer}>(byte_ptr);
                *access_ptr = frag_ptr[access_idx];
            }}
        }}
        """)

        code.arg("frag",
                 f"{self.fragment_t} const&").arg("byte_offset",
                                                  str(self.index_t))
        return code

    async def store_with_byte_offset_python(self, frag: ArrayPtr,
                                            byte_offset: int):
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        frag_ptr = frag.change_access_size(self.element_per_acc)
        for s in range(self.tmap.iterations[0]):
            for c in range(self.tmap.iterations[1]):
                access_idx = c + s * self.tmap.iterations[1]
                byte_ptr = self.get_python(
                    s, c).change_access_byte_size(1) + byte_offset
                access_ptr = byte_ptr.change_access_size(self.element_per_acc)
                await checkers.smem_bank_conflicit_check(access_ptr, 0)
                access_ptr[0] = frag_ptr[access_idx]
                ptr_addrs[access_idx * frag_ptr.access_size:(access_idx + 1) *
                          frag_ptr.access_size] = np.arange(
                              access_ptr.offset,
                              access_ptr.offset + frag_ptr.access_size)
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

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        if self.is_crosswise:
            code = pccm.FunctionCode(f"""
            add_tile_offset(0, 1);
            return *this;
            """)
        else:
            code = pccm.FunctionCode(f"""
            add_tile_offset(1, 0);
            return *this;
            """)

        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        if self.is_crosswise:
            self.add_tile_offset_python(0, 1)
            return self
        else:
            self.add_tile_offset_python(1, 0)
            return self


class WarpIteratorCrosswise(bases.GemmWarpIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape_km: MetaArray[int],
                 my_layout: MyTensorOpLayout,
                 smem_layout: layout_tensorop.TensorOpMultiplicand,
                 warp_tile_shape_km: MetaArray[int], operand_a: bool,
                 inst_shape_km: MetaArray[int], mma_inst_delta: int,
                 partk: int):
        self.threads = 32

        element_count = warp_tile_shape_km[1] * inst_shape_km[0] // self.threads
        super().__init__(dtype, element_count, smem_layout.element_per_acc)
        # input: [m, k], [k, n]
        # swapped in wrapper for crosswise
        # warp tile shape: [32, 64, 32]
        # A crosswise: [32, 32], congruous: [32, 64]
        # B congruous: [32, 64], crosswise: [64, 32]

        # inst shape [16, 8, 8]
        # A congruous: 8, 16, crosswise: 16, 8
        # B congruous: 8, 8, crosswise: [8, 8]
        # cultass shape: mk
        # our shape: km
        self.my_layout = my_layout
        self.num_warp_gemm_iters = warp_tile_shape_km[0] // inst_shape_km[0]
        # TODO find out why reverse tile_shape_km
        self.warp_tile_shape_km_raw = warp_tile_shape_km.copy()
        self.inst_shape_km_raw = inst_shape_km.copy()

        warp_tile_shape_km = warp_tile_shape_km[::-1]
        # [16, 8, 8], [8, 16] -> [16, 8]
        inst_shape_km = inst_shape_km[::-1]
        self.tile_shape_km = tile_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km

        self.operand_a = operand_a
        self.inst_shape_km = inst_shape_km
        self.mma_inst_delta = mma_inst_delta
        self.partk = partk
        self.crosswise = smem_layout.crosswise

        self.layout = smem_layout
        self.smem_stride = smem_layout.static_stride

        self.lds_op_outer = self.layout.element_per_acc
        self.lds_op_inner = 8
        assert warp_tile_shape_km[0] % self.lds_op_outer == 0
        assert warp_tile_shape_km[1] % self.lds_op_inner == 0
        # lds_shape: number of sub matrix in ldmatrix inst
        # inst shape is mk ([16, 8]), so we need to load [16, 8]
        # so self.lds_op_outer * sizeof(T) must be 8 x f16?
        self.lds_shape = seq(1, inst_shape_km[1] // self.lds_op_outer)
        self.lds_shape[0] = arch.ldmatrix.LdMatrix.MaxNum // self.lds_shape[1]
        # warp_tile_shape_km: mk, [32, 32]
        if (self.lds_shape[0] * self.lds_op_inner) > warp_tile_shape_km[0]:
            # 4 // self.lds_shape[1]: lds_shape prod at most 4
            # 4 // self.lds_shape[1] * self.lds_op_inner: maximum lds stride
            # so if lds stride is too large, just use warp_tile_shape_km[0] // self.lds_op_inner
            self.lds_shape[0] = warp_tile_shape_km[0] // self.lds_op_inner

        self.lds_iters = seq(
            warp_tile_shape_km[0] // self.lds_op_inner // self.lds_shape[0], 1)
        # number of k per tile?
        self.k_groups_per_tile = self.layout.tile_shape[
            1] // self.layout.factor // self.lds_shape[1]  # type: int
        self.k_group_inc_mask = 0
        if self.k_groups_per_tile // self.partk > 1:
            self.k_group_inc_mask = (1 << int(
                np.log2(self.k_groups_per_tile // self.partk)) - 1) - 1
        # if cudasim.inside_cuda() and cudasim.threadIdx().x == 0:
        #     print(self.k_groups_per_tile, self.lds_shape, self.lds_iters, self.layout.factor)
        # print(self.lds_op_outer, self.layout.tile_shape)
        # print(self.class_name, self.lds_shape, self.lds_iters,
        #       self.k_groups_per_tile, self.layout.element_per_acc)
        self.add_member("sections_", "int")
        self.add_member("stride_", self.index_t)
        self.add_member("pointer_", self.const_access_pointer)
        # self.add_member("pointer_bkp_", self.const_access_pointer)

        self.add_member("byte_offset_", self.index_t)
        self.add_member("wmma_k_index_", "int")
        self.ldmatrix = arch.ldmatrix.LdMatrix(True, self.lds_shape.prod())
        self.add_param_class("ldsm", self.ldmatrix, "LdMatrix")
        # cudasim members
        self.pointer_: ArrayPtr = None
        self.sections_ = -1
        self.stride_ = 0
        self.byte_offset_ = -1
        self.wmma_k_index_ = -1

    def __repr__(self):
        return (f"WarpIteratorCrosswise[ldss={self.lds_shape}|"
                f"ldsi={self.lds_iters}|g={self.k_groups_per_tile}]")

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_idx_k, warp_idx_mn, lane_idx", "int")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_access_pointer}>(ptr)")
        # code.ctor_init("pointer_bkp_",
        #                f"reinterpret_cast<{self.const_access_pointer}>(ptr)")

        code.ctor_init("sections_", f"{self.smem_stride} / {self.crosswise}")
        code.ctor_init(
            "stride_",
            f"{self.smem_stride} * {self.layout.factor} / {self.element_per_acc}"
        )
        code.ctor_init("wmma_k_index_", "0")
        code.ctor_init("byte_offset_", "0")
        code.raw(f"""
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
            lane_idx = lane_idx % ({self.lds_shape.prod()} * {self.lds_op_inner});
        #endif
        int quad_quad = (lane_idx >> 4); // 0000 0000 0000 0000 1111 1111 1111 1111
        int quad_pair = (lane_idx >> 3); // 0000 0000 1111 1111 2222 2222 3333 3333
        int lane_in_pair = (lane_idx & 1); // lane_idx % 2
        int lane_in_quad = (lane_idx & 3); // lane_idx % 4
        int lane_in_quad_pair = (lane_idx & 7); // lane_idx % 8
        int lane_in_quad_quad = (lane_idx & 15); // lane_idx % 16

        int partition_contiguous_idx = -1;
        int access_contiguous_idx = -1;
        int access_strided_idx = -1;

        """)
        if self.layout.factor == 4:
            code.raw(f"""
            int factor_in_partition =
                ({self.layout.part_shape[1]} * {self.layout.factor} /
                {self.layout.tile_shape[1]});

            if ({self.lds_shape[0]} == {self.lds_shape.prod()}) {{
                // Integer matrix multiply 8816  A/B
                partition_contiguous_idx = lane_in_quad / factor_in_partition;
                access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                        (lane_in_quad_quad / {self.layout.factor}));
                access_strided_idx = lane_idx / {self.layout.factor};
            }}
            else if ({self.lds_shape[0]} ==
                            ({self.lds_shape.prod()} / 2) &&
                        {pccm.boolean(self.operand_a)}) {{
                // Integer matrix multiply 16832 A
                partition_contiguous_idx = lane_in_quad / factor_in_partition;
                access_strided_idx = lane_in_quad_quad / {self.layout.factor};
                access_contiguous_idx =
                    ((lane_in_pair * factor_in_partition + quad_quad) ^
                    access_strided_idx);
            }}
            else if ({self.lds_shape[0]} ==
                            ({self.lds_shape.prod()} / 2) &&
                        {pccm.boolean(not self.operand_a)}) {{
                // Integer matrix multiply 16832 B
                partition_contiguous_idx = lane_in_quad / factor_in_partition;
                access_strided_idx = lane_in_quad_pair / {self.layout.factor} + quad_quad * 2;
                access_contiguous_idx =
                    ((lane_in_pair * factor_in_partition + ((lane_idx & 8) >> 3)) ^
                    access_strided_idx);
            }}
            
            """)
        elif self.layout.factor == 2:
            # input shape: [?, 32]
            code.raw(f"""
            // Super Matrix multiply kBlock = 32
            if ({self.lds_shape[0]} == {self.lds_shape.prod()}) {{
                // Matrix multiply 1688 A/B
                // (Q stands for 1 8x128bit block).
                // Q0
                // Q1
                // Q2
                // Q3
                // Four blocks are next to each other in the strided dimension.
                partition_contiguous_idx = (lane_idx % {self.layout.factor});
                access_contiguous_idx = (lane_in_quad_pair / {self.layout.factor});
                access_strided_idx = lane_idx / {self.layout.factor};
            }}
            else if ({self.lds_shape[0]} ==
                            ({self.lds_shape.prod()} / 2) &&
                        {pccm.boolean(self.operand_a)}) {{
                // Matrix multiply 16816|1688.TF32 A
                // Q0 Q2
                // Q1 Q3
                partition_contiguous_idx = (lane_idx % {self.layout.factor});
                access_contiguous_idx =
                    (quad_quad ^ (lane_in_quad_pair / {self.layout.factor}));
                access_strided_idx = (lane_in_quad_quad / {self.layout.factor});
            }} else if ({self.lds_shape[0]} ==
                            ({self.lds_shape.prod()} / 2) &&
                        {pccm.boolean(not self.operand_a)}) {{
                // Matrix multiply 16816|1688.TF32 B
                // Q0 Q1
                // Q2 Q3
                partition_contiguous_idx = (lane_idx % {self.layout.factor});
                access_contiguous_idx =
                    ((quad_pair & 1) ^ (lane_in_quad_pair / {self.layout.factor}));
                access_strided_idx =
                    (lane_in_quad_pair + (lane_idx >> 4 << 3)) / {self.layout.factor};
            }}
            else if ({self.lds_shape[1]} == {self.lds_shape.prod()}) {{
                // Matrix multiply 16832.SP B
                // Q0 Q1 Q2 Q3
                partition_contiguous_idx = (lane_idx % {self.layout.factor});
                access_contiguous_idx =
                    (quad_pair ^ (lane_in_quad_pair / {self.layout.factor}));
                access_strided_idx = lane_in_quad_pair / {self.layout.factor};
            }}
            """)
        elif self.layout.factor == 1:
            code.raw(f"""
            // Super Matrix multiply kBlock = 64
            if ({self.lds_shape[0]} == {self.lds_shape.prod()}) {{
                // Q0
                // Q1
                // Q2
                // Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = lane_in_quad;
                access_strided_idx = lane_idx;
            }}
            else if ({self.lds_shape[0]} ==
                            ({self.lds_shape.prod()} / 2) &&
                        {pccm.boolean(self.operand_a)}) {{
                // Matrix multiply 16816|1688.TF32 A
                // Q0 Q2
                // Q1 Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = (quad_quad ^ lane_in_quad);
                access_strided_idx = lane_in_quad_quad;
            }} else if ({self.lds_shape[0]} ==
                            ({self.lds_shape.prod()} / 2) &&
                        {pccm.boolean(not self.operand_a)}) {{
                // Matrix multiply 16816|1688.TF32 B
                // Q0 Q1
                // Q2 Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair + (lane_idx >> 4 << 3);
            }} 
            else if ({self.lds_shape[1]} == {self.lds_shape.prod()}) {{
                // Matrix multiply 16832.SP B
                // Q0 Q1 Q2 Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = (quad_pair ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair;
            }}
            """)
        else:
            raise NotImplementedError

        code.raw(f"""
        int access_contiguous =
            partition_contiguous_idx * {self.layout.part_shape[1]} +
            access_contiguous_idx;

        int access_strided = access_strided_idx;
        byte_offset_ = (access_contiguous + access_strided * stride_) *
                    {self.dtype.bitsize()} * {self.element_per_acc} / 8;
        // tv::printf2_block_once(threadIdx.x, "PermuteK", byte_offset_);
        auto pointer_bkp_ = pointer_;
        auto byte_offset_bkp_ = byte_offset_;
        add_tile_offset(warp_idx_mn, {self.num_warp_gemm_iters} * warp_idx_k);
        // tv::printf2_block_once(threadIdx.x, "PermuteK", byte_offset_ - byte_offset_bkp_, pointer_ - pointer_bkp_);
        """)
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_mn: int, lane_idx: int):
        new_obj = WarpIteratorCrosswise(self.dtype, self.tile_shape_km,
                                        self.my_layout, self.layout,
                                        self.warp_tile_shape_km_raw,
                                        self.operand_a, self.inst_shape_km_raw,
                                        self.mma_inst_delta, self.partk)
        new_obj.pointer_ = ptr.change_access_size(new_obj.element_per_acc)
        new_obj.sections_ = new_obj.smem_stride // new_obj.crosswise
        new_obj.stride_ = new_obj.smem_stride * new_obj.layout.factor // new_obj.element_per_acc
        new_obj.wmma_k_index_ = 0
        new_obj.byte_offset_ = 0
        layout = new_obj.layout.python_ctor(new_obj.smem_stride)

        # turing
        lane_idx = lane_idx % (new_obj.lds_shape.prod() * new_obj.lds_op_inner)
        quad_quad = (lane_idx >> 4)  # 0[16] 1[16]
        quad_pair = (lane_idx >> 3)  # 0[8] 1[8] 2[8] 3[8]
        lane_in_pair = (lane_idx & 1)
        lane_in_quad = (lane_idx & 0b11)  # 0123 0123 0123 0123
        lane_in_quad_pair = (lane_idx & 0b111)  # 0123 4567 0123 4567
        lane_in_quad_quad = (lane_idx & 0b1111)  # 0-15 0-15

        partition_contiguous_idx = -1
        access_contiguous_idx = -1
        access_strided_idx = -1

        if new_obj.layout.factor == 4:
            factor_in_partition = ((new_obj.layout.part_shape[1] *
                                    new_obj.layout.factor //
                                    new_obj.layout.tile_shape[1]))
            # factor_in_partition == 2
            if (new_obj.lds_shape[0] == new_obj.lds_shape.prod()):
                # k = 0
                # 0 1 2 3
                #  4 5 6 7
                # 1 0 3 2
                #  5 4 7 6
                # 0 1 2 3
                #  4 5 6 7
                # 1 0 3 2
                #  5 4 7 6

                # Integer matrix multiply 8816  A/B
                # load 1/2/4 submatrix

                partition_contiguous_idx = lane_in_quad // factor_in_partition  # 0011 0011 0011 0011
                # access_strided_idx = lane_idx // warp_bfa_access_shape[1]
                access_strided_idx = lane_idx // new_obj.layout.factor

                access_contiguous_idx = (
                    (lane_in_pair * factor_in_partition)
                    ^  # 0202 0202 0202 0202 0202 0202 0202 0202
                    (lane_in_quad_quad // new_obj.layout.factor)
                )  # 0000 1111 2222 3333 0000 1111 2222 3333

                # stride: 0000 1111 2222 3333 4444 5555 6666 7777
                # acc_idx_c: 0202 1313 2020 3131 0202 1313 2020 3131
                # if 01 45, noop: ^ 0
                # if 23 67, switch:  ^ 1
                # if 1 3 5 7, += 1
                acc_idx_c = (lane_idx & 1) * 2  # 0202
                if (access_strided_idx // 2) & 1 == 1:
                    acc_idx_c ^= 0b10  # 0202 0202 2020 2020 0202 0202 2020 2020
                if access_strided_idx & 1 == 1:
                    acc_idx_c += 1
                assert acc_idx_c == access_contiguous_idx, f"{lane_idx}, {acc_idx_c}, {access_contiguous_idx}"

            elif (new_obj.lds_shape[0] == (new_obj.lds_shape.prod() // 2)
                  and new_obj.operand_a):
                # Integer matrix multiply 16832 A
                # Q0 Q2
                # Q1 Q3
                partition_contiguous_idx = lane_in_quad // factor_in_partition
                access_strided_idx = lane_in_quad_quad // new_obj.layout.factor
                # stride: 0000 1111 2222 3333 0000 1111 2222 3333

                # 0202.... + 0[16] [16] ^
                access_contiguous_idx = (
                    ((lane_in_pair * factor_in_partition + quad_quad)
                     ^ access_strided_idx))

            elif (new_obj.lds_shape[0] == (new_obj.lds_shape.prod() // 2)
                  and not new_obj.operand_a):
                # Integer matrix multiply 16832 B
                # Q0 Q1
                # Q2 Q3
                partition_contiguous_idx = lane_in_quad // factor_in_partition
                access_strided_idx = lane_in_quad_pair // new_obj.layout.factor + quad_quad * 2
                access_contiguous_idx = (((lane_in_pair * factor_in_partition +
                                           ((lane_idx & 8) >> 3))
                                          ^ access_strided_idx))

        elif new_obj.layout.factor == 2:
            # Super Matrix multiply kBlock = 32
            if (new_obj.lds_shape[0] == new_obj.lds_shape.prod()):
                # Matrix multiply 1688 A/B
                # (Q stands for 1 8x128bit block).
                # Q0
                # Q1
                # Q2
                # Q3
                # Four blocks are next to each other in the strided dimension.
                partition_contiguous_idx = (lane_idx % new_obj.layout.factor)
                # these lines are matching swizzle behavior in tensorop layout.
                # for k > 0, offset is handled in set_wmma_index
                access_contiguous_idx = (lane_in_quad_pair //
                                         new_obj.layout.factor
                                         )  # 00 11 22 33 00 11 22 33

                access_strided_idx = lane_idx // new_obj.layout.factor  # 00 11 22 33 ....

            elif (new_obj.lds_shape[0] == (new_obj.lds_shape.prod() // 2)
                  and new_obj.operand_a):

                # Matrix multiply 16816|1688.TF32 A
                # Q0 Q2 (check mma inst for more details.)
                # Q1 Q3
                partition_contiguous_idx = (lane_idx % new_obj.layout.factor)
                access_contiguous_idx = ((
                    quad_quad ^ (lane_in_quad_pair // new_obj.layout.factor)))
                access_strided_idx = (lane_in_quad_quad //
                                      new_obj.layout.factor)

            elif (new_obj.lds_shape[0] == (new_obj.lds_shape.prod() // 2)
                  and not new_obj.operand_a):
                # 16816: f16
                # Matrix multiply 16816|1688.TF32 B
                # Q0 Q1
                # Q2 Q3
                partition_contiguous_idx = (lane_idx % new_obj.layout.factor)
                access_contiguous_idx = (
                    ((quad_pair & 1) ^
                     (lane_in_quad_pair // new_obj.layout.factor)))
                access_strided_idx = ((lane_in_quad_pair +
                                       (lane_idx >> 4 << 3)) //
                                      new_obj.layout.factor)

            elif (new_obj.lds_shape[1] == new_obj.lds_shape.prod()):
                # Matrix multiply 16832.SP B
                # Q0 Q1 Q2 Q3

                partition_contiguous_idx = (lane_idx % new_obj.layout.factor)
                access_contiguous_idx = ((
                    quad_pair ^ (lane_in_quad_pair // new_obj.layout.factor)))
                access_strided_idx = lane_in_quad_pair // new_obj.layout.factor

        elif new_obj.layout.factor == 1:
            # Super Matrix multiply kBlock = 64
            if (new_obj.lds_shape[0] == new_obj.lds_shape.prod()):
                # Q0
                # Q1
                # Q2
                # Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2)
                access_contiguous_idx = lane_in_quad
                access_strided_idx = lane_idx

            elif (new_obj.lds_shape[0] == (new_obj.lds_shape.prod() // 2)
                  and new_obj.operand_a):
                # Matrix multiply 16816|1688.TF32 A
                # Q0 Q2
                # Q1 Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2)
                access_contiguous_idx = (quad_quad ^ lane_in_quad)
                access_strided_idx = lane_in_quad_quad
            elif (new_obj.lds_shape[0] == (new_obj.lds_shape.prod() // 2)
                  and not new_obj.operand_a):
                # Matrix multiply 16816|1688.TF32 B
                # Q0 Q1
                # Q2 Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2)
                access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad)
                access_strided_idx = lane_in_quad_pair + (lane_idx >> 4 << 3)

            elif (new_obj.lds_shape[1] == new_obj.lds_shape.prod()):
                # Matrix multiply 16832.SP B
                # Q0 Q1 Q2 Q3
                partition_contiguous_idx = (lane_in_quad_pair >> 2)
                access_contiguous_idx = (quad_pair ^ lane_in_quad)
                access_strided_idx = lane_in_quad_pair

        else:
            raise NotImplementedError
        access_contiguous = (
            partition_contiguous_idx * new_obj.layout.part_shape[1] +
            access_contiguous_idx)
        access_strided = access_strided_idx

        expected_offset = access_contiguous + access_strided_idx * new_obj.stride_
        ref_offset = layout.get_ldm_initial_offset_ref(lane_idx,
                                                       self.lds_shape,
                                                       not self.operand_a)
        assert ref_offset == expected_offset * self.element_per_acc, f"{lane_idx}, {expected_offset}, {ref_offset}"

        new_obj.byte_offset_ = (
            (access_contiguous + access_strided * new_obj.stride_) *
            new_obj.dtype.bitsize() * new_obj.element_per_acc // 8)
        # for k part, of partk == 2 and k of smem is 4, then num warp gemm iters is 2
        # so 0 1
        #     0 1
        new_obj.add_tile_offset_python(
            warp_idx_mn, new_obj.num_warp_gemm_iters * warp_idx_k)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_tile_offset(self):
        code = pccm.FunctionCode(f"""
        int mn_offset = warp_idx_mn;
        int k_offset = warp_idx_k;
        // tv::printf2_block_once(threadIdx.x, k_offset, mn_offset);
        int whole_tiles = mn_offset / {self.k_groups_per_tile};
        int k_groups_delta = mn_offset % {self.k_groups_per_tile};

        byte_offset_ ^= k_groups_delta * {self.dtype.bitsize()} *
                        {self.layout.element_per_acc} *
                        {self.lds_shape[1]} / 8;
        // tv::printf2_block_once(threadIdx.x, "premuteK", byte_offset_);

        pointer_ +=
            k_offset * stride_ * {self.warp_tile_shape_km[0]} / {self.layout.factor} +
            whole_tiles * stride_ / sections_;
        """)
        return code.arg("warp_idx_k, warp_idx_mn", "int")

    def add_tile_offset_python(self, warp_idx_k: int, warp_idx_mn: int):
        mn_offset = warp_idx_mn
        k_offset = warp_idx_k
        whole_tiles = mn_offset // self.k_groups_per_tile
        k_groups_delta = mn_offset % self.k_groups_per_tile
        self.byte_offset_ ^= int(k_groups_delta * self.dtype.bitsize() *
                                 self.layout.element_per_acc *
                                 self.lds_shape[1] // 8)
        self.pointer_ += (k_offset * self.stride_ *
                          self.warp_tile_shape_km[0] // self.layout.factor +
                          whole_tiles * self.my_layout.sw_shape[1])

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        add_tile_offset(0, num_tile);
        """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        return self.add_tile_offset_python(0, num)

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        // Integer matrix multiply 16832 Interleaved-32
        //   NONE
        // Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

        // Integer matrix multiply 8816  Interleaved-32
        //   ^1 ^1
        // Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
        // Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
        //   ^1 ^3 ^1 ^3
        // Matrix multiply 1688 kblock=64
        //   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

        // Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
        //   ^2 ^2
        // Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
        //   ^2 ^6 ^2 ^6
        if (({self.k_groups_per_tile} / {self.partk}) > 1) {{
            int mask = (({self.k_groups_per_tile} / {self.partk}) == 8)
                            ? 3
                            : ((({self.k_groups_per_tile} / {self.partk}) == 4) ? 1 : 0);

            if (((wmma_k_index_ & mask) % 2) == 0)
                byte_offset_ ^= 1 * {self.lds_shape[1]} *
                                {self.dtype.bitsize()} *
                                {self.layout.element_per_acc} / 8;
            else if ((wmma_k_index_ & mask) == 1)
                byte_offset_ ^= 3 * {self.lds_shape[1]} *
                                {self.dtype.bitsize()} *
                                {self.layout.element_per_acc} / 8;
            else if ((wmma_k_index_ & mask) == 3)
                byte_offset_ ^= 7 * {self.lds_shape[1]} *
                                {self.dtype.bitsize()} *
                                {self.layout.element_per_acc} / 8;
        }}

        wmma_k_index_++;
        // tv::printf2_block_once(threadIdx.x, "premuteK", byte_offset_);
        if (wmma_k_index_ == ({self.k_groups_per_tile} / {self.partk})) {{
            wmma_k_index_ = 0;
            // k group increment
            add_tile_offset(0, {self.k_groups_per_tile});
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):

        # Integer matrix multiply 16832 Interleaved-32
        #   NONE
        # Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

        # Integer matrix multiply 8816  Interleaved-32
        #   ^1 ^1
        # Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
        # Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
        #   ^1 ^3 ^1 ^3
        # Matrix multiply 1688 kblock=64
        #   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

        # Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
        #   ^2 ^2
        # Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
        #   ^2 ^6 ^2 ^6
        k_inc_width = self.lds_shape[1] * self.dtype.bitsize(
        ) * self.layout.element_per_acc // 8
        num_k_inc = (self.k_groups_per_tile // self.partk)
        if (num_k_inc > 1):
            # mask: largest number of bit for increment
            mask = self.k_group_inc_mask
            # if self.k_groups_per_tile // self.partk == 8:
            #     mask = 0b11
            # elif self.k_groups_per_tile // self.partk == 4:
            #     mask = 0b1
            # else:
            #     mask = 0
            # bit 0 advance
            self.byte_offset_ ^= layout_tensorop.swizzle_increment(
                self.wmma_k_index_ & mask, k_inc_width)
            # if (((self.wmma_k_index_ & mask) % 2) == 0):
            #     self.byte_offset_ ^= (1 * self.lds_shape[1] *
            #                           self.dtype.bitsize() *
            #                           self.layout.element_per_acc // 8)
            # # bit 1 advance
            # elif ((self.wmma_k_index_ & mask) == 1):
            #     self.byte_offset_ ^= (0b11 * self.lds_shape[1] *
            #                           self.dtype.bitsize() *
            #                           self.layout.element_per_acc // 8)
            # # bit 2 advance
            # elif ((self.wmma_k_index_ & mask) == 3):
            #     self.byte_offset_ ^= (0b111 * self.lds_shape[1] *
            #                           self.dtype.bitsize() *
            #                           self.layout.element_per_acc // 8)

        self.wmma_k_index_ += 1

        if (self.wmma_k_index_ == num_k_inc):
            self.wmma_k_index_ = 0
            # k group increment
            self.add_tile_offset_python(0, self.k_groups_per_tile)

        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_byte_offset(self):
        fetch_array_type = array_type("unsigned", self.lds_shape.prod())
        code = pccm.FunctionCode(f"""
        {fetch_array_type} *fetch_ptr =
            reinterpret_cast<{fetch_array_type} *>(&frag);
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.lds_iters[0]}; ++s) {{
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.lds_iters[1]}; ++c) {{
                int access_idx = c + s * {self.lds_iters[1]};

                {self.const_access_pointer} source_ptr =
                    pointer_ + {self.lds_shape[1]} * c +
                    {self.lds_op_inner} / {self.layout.factor} *
                        {self.lds_shape[0]} * s * stride_;
                auto off = {self.lds_shape[1]} * c +
                    {self.lds_op_inner} / {self.layout.factor} *
                        {self.lds_shape[0]} * s * stride_;

                char const *source_byte_ptr =
                    reinterpret_cast<char const *>(source_ptr) + byte_offset +
                    byte_offset_;
                auto debug_ptr = reinterpret_cast<{self.const_pointer}>(source_byte_ptr);
                // tv::printf2_block_once(threadIdx.x, s, c,access_idx,off,source_byte_ptr -  reinterpret_cast<char const *>(pointer_bkp_),
                //     float(debug_ptr[0]), float(debug_ptr[1]));

                LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
            }}
        }}
        // tv::print_fragment_meta_once<float>(frag);
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("byte_offset",
                                                    str(self.index_t))
        return code

    async def load_with_byte_offset_python(self, frag: ArrayPtr,
                                           byte_offset: int):
        fetch_ptr = frag.change_access_byte_size(self.lds_shape.prod() * 4)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        for s in range(self.lds_iters[0]):
            for c in range(self.lds_iters[1]):
                access_idx = c + s * self.lds_iters[1]
                source_ptr = (
                    self.pointer_ + self.lds_shape[1] * c +
                    self.lds_op_inner // self.layout.factor *
                    self.lds_shape[0] * s * self.stride_).change_access_size(
                        self.element_per_acc)

                source_byte_ptr = source_ptr.change_access_byte_size(
                    1) + byte_offset + self.byte_offset_
                await checkers.smem_bank_conflicit_check(fetch_ptr, access_idx)
                await self.ldmatrix(fetch_ptr[access_idx], source_byte_ptr)
                ptr_addrs[access_idx * fetch_ptr.access_size:(access_idx + 1) *
                          fetch_ptr.access_size] = np.arange(
                              source_byte_ptr.offset,
                              source_byte_ptr.offset + fetch_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, pointer_offset * sizeof({self.dtype}));
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_kgroup_index(self):
        code = pccm.FunctionCode(
            f"wmma_k_index_ = wmma_k % ({self.k_groups_per_tile} / {self.partk});"
        )
        code.arg("wmma_k", "int")
        return code

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_byte_offset_python(frag, 0)

    def set_wmma_k_index_python(self, wmma_k):
        self.wmma_k_index_ = wmma_k % (self.k_groups_per_tile // self.partk)


class WarpIteratorCongruous(bases.GemmWarpIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape_km: MetaArray[int],
                 my_layout: MyTensorOpLayout,
                 smem_layout: layout_tensorop.TensorOpMultiplicand,
                 warp_tile_shape_km: MetaArray[int], operand_a: bool,
                 inst_shape_km: MetaArray[int], mma_inst_delta: int,
                 partk: int):
        element_count = warp_tile_shape_km[1] * inst_shape_km[0] // self.threads
        super().__init__(dtype, element_count, smem_layout.element_per_acc)
        # cultass shape: mk
        # our shape: km
        self.my_layout = my_layout
        self.num_warp_gemm_iters = warp_tile_shape_km[0] // inst_shape_km[0]
        self.tile_shape_km = tile_shape_km
        self.warp_tile_shape_km_raw = warp_tile_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km
        self.operand_a = operand_a
        self.inst_shape_km = inst_shape_km
        self.inst_shape_km_raw = inst_shape_km

        self.mma_inst_delta = mma_inst_delta
        self.partk = partk

        self.layout = smem_layout
        self.smem_stride = smem_layout.static_stride
        self.is_spec_32 = smem_layout.element_size == 32 and smem_layout.crosswise == 32
        self.threads = 32
        num_threads = 32
        if self.is_spec_32:
            # TODO: only used for tf32 tensor ops
            # Determine number of elements along outer dimension per individual 32bit
            # shared memory load op.  Every one warp of 32bit shared memory load loads
            # 8x4 elements
            self.lds_op_inner = self.layout.tile_shape[0]
            self.lds_op_outer = num_threads // self.lds_op_inner
            # Number of 32 bit shared memory load instructions needed by one MMA instruction
            # 1688  A 2x2
            # 1688  B 1x2
            # 16816 B 1x4

            self.lds_shape = metaseq(inst_shape_km[0] // self.lds_op_inner,
                                     inst_shape_km[1] // self.lds_op_outer)
            self.lds_iters = metaseq(
                1, warp_tile_shape_km[1] // self.lds_shape[1] //
                self.lds_op_outer)
            self.pointer_count = self.layout.tile_shape[
                1] * self.layout.element_per_acc // self.lds_op_outer
        else:
            self.lds_op_outer = self.layout.element_per_acc
            self.lds_op_inner = 8
            self.lds_shape = metaseq(inst_shape_km[0] // self.lds_op_inner, 1)
            self.lds_shape[1] = 4 // self.lds_shape[0]
            self.lds_iters = metaseq(
                1, warp_tile_shape_km[1] // self.layout.element_per_acc //
                self.lds_shape[1])
            self.pointer_count = self.layout.tile_shape[1] // self.lds_shape[1]
        assert warp_tile_shape_km[1] % self.lds_op_outer == 0
        assert warp_tile_shape_km[0] % self.lds_op_inner == 0

        self.k_groups_per_tile = self.warp_tile_shape_km[0] // inst_shape_km[
            0]  # type: int
        # print(self.layout.tile_shape)
        # print(self.class_name, self.lds_shape, self.lds_iters,
        #       self.k_groups_per_tile)
        self.add_member("wmma_k_index_", "int")
        self.add_member("stride_", self.index_t)
        self.add_member("pointer_",
                        self.const_access_pointer,
                        array=f"[{self.pointer_count}]")
        self.add_member("byte_offset_", self.index_t)
        self.ldmatrix = arch.ldmatrix.LdMatrix(False, self.lds_shape.prod())

        self.add_param_class("ldsm", self.ldmatrix, "LdMatrix")

        # cudasim members
        self.pointer_: List[ArrayPtr] = [None] * self.pointer_count
        self.stride_ = 0
        self.byte_offset_ = -1
        self.wmma_k_index_ = -1

    def __repr__(self):
        return (f"WarpIteratorCongruous[ldss={self.lds_shape}|"
                f"ldsi={self.lds_iters}|g={self.k_groups_per_tile}]")

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_idx_k, warp_idx_mn, lane_idx", "int")
        code.ctor_init("stride_",
                       f"{self.smem_stride} / {self.element_per_acc}")
        code.ctor_init("wmma_k_index_", "0")
        code.ctor_init("byte_offset_", "0")
        if not self.is_spec_32:
            code.raw(f"""
            int quad_pair = (lane_idx >> 3);
            int quad_quad = (lane_idx >> 4);
            int lane_in_quad = (lane_idx & 3);
            int lane_in_quad_pair = (lane_idx & 7);
            int lane_in_quad_quad = (lane_idx & 15);

            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.pointer_count}; ++i) {{
                int partition_contiguous_idx = -1;
                int access_contiguous_idx = -1;
                int access_strided_idx = -1;

                if ({self.lds_shape[1]} == 4) {{
                    // Matrix multiply 1688 A/B
                    // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
                    // Four blocks are next to each other in the contiguous dimension.
                    partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
                    access_contiguous_idx = (quad_pair ^ lane_in_quad);
                    access_strided_idx = lane_in_quad_pair;
                }}
                else if ({self.lds_shape[1]} == 2 &&
                            {pccm.boolean(self.operand_a)}) {{
                    // Matrix multiply 16816 A
                    // Q0 Q2
                    // Q1 Q3
                    partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
                    access_contiguous_idx =
                        (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
                    access_strided_idx = lane_in_quad_pair + (lane_idx >> 4 << 3);
                }} else if ({self.lds_shape[1]} == 2 &&
                            {pccm.boolean(not self.operand_a)}) {{
                    // Matrix multiply 16816 B
                    // Q0 Q1
                    // Q2 Q3
                    partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
                    access_contiguous_idx = ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
                    access_strided_idx = lane_in_quad_quad;
                }} else if ({self.lds_shape[1]} == 1) {{
                    // Matrix multiply 16832.SP B
                    // Q0
                    // Q1
                    // Q2
                    // Q3
                    partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 2)); 
                    access_contiguous_idx = ((i & 3) ^ lane_in_quad); 
                    access_strided_idx = lane_idx; 
                }}

                int access_contiguous =
                    partition_contiguous_idx * {self.layout.part_shape[1]} +
                    access_contiguous_idx;

                int access_strided = access_strided_idx;

                pointer_[i] = reinterpret_cast<{self.const_access_pointer} >(ptr) +
                                access_contiguous + access_strided * stride_;
            }}
            add_tile_offset({self.num_warp_gemm_iters} * warp_idx_k, warp_idx_mn);
            """)
        else:
            code.raw(f"""
            for (int i = 0; i < {self.pointer_count}; ++i) {{
                int access_strided = lane_idx % {self.lds_op_inner};
                int access_contiguous = (lane_idx / {self.lds_op_inner}) +
                                        (access_strided ^ i) * {self.lds_op_outer};
                pointer_[i] = reinterpret_cast<{self.const_access_pointer} >(ptr) +
                                access_contiguous + access_strided * stride_;

            }}
            """)
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_mn: int, lane_idx: int):
        new_obj = WarpIteratorCongruous(self.dtype, self.tile_shape_km,
                                        self.my_layout, self.layout,
                                        self.warp_tile_shape_km_raw,
                                        self.operand_a, self.inst_shape_km_raw,
                                        self.mma_inst_delta, self.partk)
        new_obj.stride_ = self.smem_stride * self.layout.factor // self.element_per_acc
        new_obj.wmma_k_index_ = 0
        new_obj.byte_offset_ = 0
        assert not self.is_spec_32, "tf32 isn't supported yet."
        quad_quad = (lane_idx >> 4)  # 0[16] 1[16]
        quad_pair = (lane_idx >> 3)  # 0[8] 1[8] 2[8] 3[8]
        lane_in_pair = (lane_idx & 1)
        lane_in_quad = (lane_idx & 0b11)  # 0123 0123 0123 0123
        lane_in_quad_pair = (lane_idx & 0b111)  # 0123 4567 0123 4567
        lane_in_quad_quad = (lane_idx & 0b1111)  # 0-15 0-15
        layout = new_obj.layout.python_ctor(new_obj.smem_stride)

        for i in range(self.pointer_count):
            partition_contiguous_idx = -1
            access_contiguous_idx = -1
            access_strided_idx = -1
            if (self.lds_shape[1] == 4):
                # smem layout:
                # 0
                #  1
                #   2
                #    3
                #     4
                #      5
                #       6
                #        7
                #  Matrix multiply 1688 A/B
                #  Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
                #  Four blocks are next to each other in the contiguous dimension.
                #
                # stride: 01234567 01234567 ...
                # contig: 01234567 10325476 sw2 sw3
                # lane_in_quad_pair >> 2: 0000 1111 0000 1111 0000 1111 0000 1111
                # part 0: ...
                # part 1: switch
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i)
                access_contiguous_idx = (quad_pair ^ lane_in_quad)
                access_strided_idx = lane_in_quad_pair

            elif (self.lds_shape[1] == 2 and self.operand_a):
                #  Matrix multiply 16816 A
                #  Q0 Q2
                #  Q1 Q3
                # stride: 01234567 89ABCDEF 01234567 89ABCDEF
                # contig: 01234567 10325476 sw2 sw3

                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^
                                            (i >> 1))
                access_contiguous_idx = ((((quad_pair & 1) + ((i & 1) << 1))
                                          ^ lane_in_quad))
                access_strided_idx = lane_in_quad_pair + (lane_idx >> 4 << 3)
            elif (self.lds_shape[1] == 2 and not self.operand_a):
                #  Matrix multiply 16816 B
                #  Q0 Q1
                #  Q2 Q3
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^
                                            (i >> 1))
                access_contiguous_idx = ((quad_quad + ((i & 1) << 1))
                                         ^ lane_in_quad)
                access_strided_idx = lane_in_quad_quad
            elif (self.lds_shape[1] == 1):
                #  Matrix multiply 16832.SP B
                #  Q0
                #  Q1
                #  Q2
                #  Q3
                # (lane_in_quad_pair >> 2): 0000 1111 0000 1111 ^
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^
                                            (i >> 2))
                # access_contiguous_idx:
                # i == 0: 01230123.... + 00001111 * 4
                # i == 1: 10321032.... + 00001111 * 4
                # i == 2: 23012301.... + 00001111 * 4
                # i == 3: 32103210.... + 00001111 * 4
                # i == 4: 01230123.... + 11110000 * 4
                # ....

                access_contiguous_idx = ((i & 3) ^ lane_in_quad)
                access_strided_idx = lane_idx

            access_contiguous = (
                partition_contiguous_idx * self.layout.part_shape[1] +
                access_contiguous_idx)
            access_strided = access_strided_idx
            new_obj.pointer_[i] = (
                ptr.change_access_size(self.element_per_acc) +
                access_contiguous + access_strided * new_obj.stride_)
            expected_offset = access_contiguous + access_strided_idx * new_obj.stride_
            # if i == 0:
            ref_offset = layout.get_ldm_initial_offset_ref(
                lane_idx,
                self.lds_shape,
                self.operand_a,
                contig_offset=i * self.lds_shape[1] * self.lds_op_outer)
            assert ref_offset == expected_offset * self.element_per_acc, f"{lane_idx}, {expected_offset}, {ref_offset}"

        new_obj.add_tile_offset_python(self.num_warp_gemm_iters * warp_idx_k,
                                       warp_idx_mn)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(
            f"byte_offset_ += offset * sizeof({self.dtype});")
        code.arg("offset", self.long_index_t)
        return code

    def add_pointer_offset_python(self, offset: int):
        self.byte_offset_ += offset * self.dtype.itemsize()

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_tile_offset(self):
        code = pccm.code()
        if self.is_spec_32:
            code.raw(f"""
            constexpr int kContigEqual = {self.layout.tile_shape[1]} * {self.layout.element_per_acc} / 2;
            // Matrix multiply 1688 pointer_[0] <=> pointer_[4] pointer_[1] <=> pointer_[5]
            //           pointer_[2] <=> pointer_[6] pointer_[3] <=> pointer_[7]

            """)
        else:
            code.raw(f"""
            constexpr int kContigEqual = {self.layout.part_shape[1]} * {self.layout.element_per_acc};
            """)

        code.raw(f"""
        int mn_offset = warp_idx_mn;
        int k_offset = warp_idx_k;

        if ({self.warp_tile_shape_km[1]} == kContigEqual) {{
          if (warp_idx_mn % 2) {{
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.pointer_count} / 2; ++i) {{
              {self.const_access_pointer} tmp_pointer = pointer_[i];
              pointer_[i] = pointer_[i + {self.pointer_count} / 2];
              pointer_[i + {self.pointer_count} / 2] = tmp_pointer;
            }}
          }}
          mn_offset = (warp_idx_mn >> 1) << 1;
        }}
        """)
        if self.is_spec_32:
            code.raw(f"""
            int offset = (k_offset * {self.inst_shape_km[0]}) *
                            stride_  +
                        mn_offset * {self.warp_tile_shape_km[1]};

            """)
        else:
            code.raw(f"""
            int offset = (k_offset * {self.inst_shape_km[0]}) *
                            stride_ * {self.layout.element_per_acc} +
                        mn_offset * {self.warp_tile_shape_km[1]};

            """)
        code.raw(f"""
        add_pointer_offset(offset);
        """)
        return code.arg("warp_idx_k, warp_idx_mn", "int")

    def add_tile_offset_python(self, warp_idx_k: int, warp_idx_mn: int):
        mn_offset = warp_idx_mn
        k_offset = warp_idx_k
        # two warp handle one swizzle part.
        # for second warp, we need to swap pointers (swizzle part for second warp)
        if (self.warp_tile_shape_km[1] == self.layout.part_shape[1] *
                self.layout.element_per_acc):
            if (warp_idx_mn % 2):
                for i in range(self.pointer_count // 2):
                    tmp_pointer = self.pointer_[i]
                    self.pointer_[i] = self.pointer_[i +
                                                     self.pointer_count // 2]
                    self.pointer_[i + self.pointer_count // 2] = tmp_pointer
            # mn_offset: 00 22 44 66
            # this stmt is exists because we have add offset in init function.
            # so we skip second warp.
            # if (warp_idx_mn % 2):
            #     mn_offset ^= 1
            mn_offset = (warp_idx_mn >> 1) << 1
            # if mn_offset == 1:
            #     mn_offset = 0
            # mn_offset ^= 1

        self.add_pointer_offset_python((k_offset * self.inst_shape_km[0]) *
                                       self.stride_ *
                                       self.layout.element_per_acc +
                                       mn_offset * self.warp_tile_shape_km[1])

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        add_tile_offset(num_tile, 0);
        """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        return self.add_tile_offset_python(num, 0)

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        add_tile_offset(1, 0); // strided, contig
        // tv::printf2_block_once(threadIdx.x, "byte_offset_=", byte_offset_);
        if ({self.partk} > 1) {{
            ++wmma_k_index_;
            // Jump to next stage
            if (wmma_k_index_ == {self.k_groups_per_tile}) {{
                wmma_k_index_ = 0;
                add_tile_offset((({self.partk} - 1) * {self.k_groups_per_tile}), 0);
            }}
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        self.add_tile_offset_python(1, 0)
        if (self.partk > 1):
            self.wmma_k_index_ += 1
            # Jump to next stage
            if (self.wmma_k_index_ == self.k_groups_per_tile):
                self.wmma_k_index_ = 0
                self.add_tile_offset_python(
                    ((self.partk - 1) * self.k_groups_per_tile), 0)
        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_byte_offset(self):
        fetch_array_type = array_type("unsigned", self.lds_shape.prod())
        if not self.is_spec_32:
            code = pccm.FunctionCode(f"""
            {fetch_array_type} *fetch_ptr = 
            reinterpret_cast<{fetch_array_type} *>(&frag);

            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.lds_iters[0]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.lds_iters[1]}; ++c) {{

                    int access_idx = c + s * {self.lds_iters[1]};

                    {self.const_access_pointer} source_ptr =
                        pointer_[c % {self.pointer_count}] +
                        {self.layout.tile_shape[1]} * (c / {self.pointer_count}) +
                        {self.lds_shape[0]} * s * stride_;

                    char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

                    LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
                }}
            }}
            """)
        else:
            code = pccm.FunctionCode(f"""
            {self.dtype} *fetch_ptr = reinterpret_cast<{self.dtype} *>(&frag);

            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.lds_iters[0]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.lds_iters[1]}; ++c) {{
                    TV_PRAGMA_UNROLL
                    for (int ss = 0; ss < {self.lds_shape[0]}; ++ss) {{
                        TV_PRAGMA_UNROLL
                        for (int cc = 0; cc < {self.lds_shape[1]}; ++cc) {{
                            int access_idx =
                                cc + (ss + (c + s * {self.lds_iters[1]}) *
                                            {self.lds_shape[0]}) *
                                        {self.lds_shape[1]};
                            int access_idx_contiguous = cc + c * {self.lds_shape[1]};
                            int access_idx_strided =
                                (ss + s * {self.lds_shape[0]}) * {self.lds_op_inner};

                            {self.const_access_pointer} source_ptr =
                                pointer_[access_idx_contiguous % {self.pointer_count}] +
                                {self.layout.tile_shape[1]} * {self.layout.element_per_acc} *
                                    (access_idx_contiguous / {self.pointer_count}) +
                                access_idx_strided * stride_;

                            char const *source_byte_ptr =
                                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                                byte_offset_;

                            fetch_ptr[access_idx] =
                                *reinterpret_cast<{self.dtype} const *>(source_byte_ptr);
                        }}
                    }}
                }}
            }}
            """)

        code.arg("frag", f"{self.fragment_t}&").arg("byte_offset",
                                                    str(self.index_t))
        return code

    async def load_with_byte_offset_python(self, frag: ArrayPtr,
                                           byte_offset: int):
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)
        fetch_ptr = frag.change_access_byte_size(self.lds_shape.prod() * 4)
        for s in range(self.lds_iters[0]):
            for c in range(self.lds_iters[1]):
                access_idx = c + s * self.lds_iters[1]
                source_ptr = (self.pointer_[c % self.pointer_count] +
                              self.layout.tile_shape[1] *
                              (c // self.pointer_count) +
                              self.lds_shape[0] * s * self.stride_)

                source_byte_ptr = source_ptr.change_access_byte_size(
                    1) + byte_offset + self.byte_offset_
                await checkers.smem_bank_conflicit_check(fetch_ptr, access_idx)
                await self.ldmatrix(fetch_ptr[access_idx], source_byte_ptr)
                ptr_addrs[access_idx * fetch_ptr.access_size:(access_idx + 1) *
                          fetch_ptr.access_size] = np.arange(
                              source_byte_ptr.offset,
                              source_byte_ptr.offset + fetch_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, pointer_offset * sizeof({self.dtype}));
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_kgroup_index(self):
        code = pccm.code()
        code.arg("wmma_k", "int")
        return code

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_byte_offset_python(frag, 0)

    def set_wmma_k_index_python(self, wmma_k):
        return
