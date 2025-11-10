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
from cumm.common import GemmBasic, GemmBasicKernel, TensorView
from cumm.constants import CUTLASS_MODE
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers
from cumm.gemm import bases, codeops, constants, layout, thread_map
from cumm.gemm.arch import memory
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.mask_iters import MaskTileIteratorParams, div_up


class MaskTileIteratorGather(bases.GemmInputIterator):
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
                 read_only: bool = True,
                 have_output_ptr: bool = False,
                 is_scatter: bool = False):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        self.access_per_vector = sub_tile_shape[1] // num_sub_access
        self.thread_tensor_stride = metaseq(
            self.thread_access_shape[1] * sub_tile_shape[0] *
            self.access_per_vector, sub_tile_shape[0] * self.access_per_vector,
            self.access_per_vector, 1)
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        # print(tmap.iterations, tmap.delta)
        # raise NotImplementedError
        super().__init__(dtype, tmap, sub_tile_shape, element_count,
                         num_sub_access,
                         dtype.itemsize() * num_sub_access)
        if transpose_load:
            assert dtype == dtypes.int8
        self.read_only = read_only
        # shuffle_in_stride = False
        self.shuffle_in_stride = shuffle_in_stride
        self.add_dependency(TensorView, GemmBasicKernel)
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
        self.have_output_ptr = have_output_ptr
        if have_output_ptr:
            assert shuffle_in_stride, "output ptr only used for gather kernel"
        if is_scatter:
            assert have_output_ptr
            assert not self.read_only
        self.is_scatter = is_scatter
        self.add_member(
            "pointer_",
            self.const_byte_pointer if read_only else self.byte_pointer)
        if have_output_ptr:
            if self.is_scatter:
                self.add_member("out_pointer_", self.const_byte_pointer)
            else:
                self.add_member("out_pointer_", self.byte_pointer)

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
        code.raw(f"""
        int residue_size = (extent[{self.advance_axis}] - threadblock_offset[{self.advance_axis}]) %
                        {self.tile_shape[self.advance_axis]};
        if (!residue_size) {{
            residue_size = {self.tile_shape[self.advance_axis]};
        }}
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

        if TV_IF_CONSTEXPR (!{pccm.boolean(self.last_residual)}) {{
            compute_predicates_(extent, false);
        }} else {{
            compute_predicates_(residue_extent, false);
        }}
        """)
        if self.shuffle_in_stride:
            code.raw(f"""
            // we may use external gather instead of update indices.
            // for this situation, just set indice_ptr to nullptr.
            if (params_.indice_ptr_ == nullptr){{
                update_indices_identity();
            }}else{{
                update_indices();
            }}
            add_pointer_offset(thread_offset_[1]);
            """)
            if self.have_output_ptr:
                code.raw(f"""
                // here we can't use extent_[1] because splitk may split stride.
                add_output_pointer_offset(thread_offset_[0] * params.stride_ + thread_offset_[1]);
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
        output_const = "const " if self.is_scatter else ""
        if self.have_output_ptr:
            code.arg("output_ptr", f"{output_const} {self.pointer}")

        code.arg("extent", "tv::array<int, 2>")
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")

        code.ctor_init("params_", "params")
        code.ctor_init(
            "pointer_",
            f"reinterpret_cast<{self.const_byte_pointer if self.read_only else self.byte_pointer}>(ptr)"
        )
        if self.have_output_ptr:
            code.ctor_init(
                "out_pointer_",
                f"reinterpret_cast<{output_const}{self.byte_pointer}>(output_ptr)"
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
                    is_left: bool = True) -> "MaskTileIteratorGather":
        new_obj = MaskTileIteratorGather(
            self.dtype, self.tile_shape, self.sub_tile_shape, self.tmap,
            params, self.advance_axis, self.num_sub_access,
            self.transpose_load, self.last_residual, self.shuffle_in_stride,
            self.read_only)
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
                            params_.stride_ * {self.dtype.bitsize()} / 8;
                else{{
                    indices_[s * {self.sub_tile_shape[0]} + ss] = 0;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def update_indices_identity(self):
        """if indice ptr is nullptr, use this function to get identity offset.
        """
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
                        (thread_offset_[0] + s * {self.tmap.delta[0]} + ss) * 
                            params_.stride_ * {self.dtype.bitsize()} / 8;
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

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_output_pointer_offset(self):
        code = pccm.code()
        if self.have_output_ptr:
            code.raw(f"""
            out_pointer_ += sizeof({self.dtype}) * offset;
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
                    if self.shuffle_in_stride and self.have_output_ptr:
                        code.raw(f"""
                        out_pointer_ += sizeof({self.dtype}) * params_.stride_ * residue_offset_;
                        """)

            else:
                if self.advance_axis == 1:
                    code.raw(f"""
                    thread_offset_[{self.advance_axis}] -= residue_offset_;
                    pointer_ -= sizeof({self.dtype}) * residue_offset_;
                    """)
                    if self.shuffle_in_stride and self.have_output_ptr:
                        code.raw(f"""
                        out_pointer_ -= sizeof({self.dtype}) * residue_offset_;
                        """)

                else:
                    code.raw(f"""
                    thread_offset_[{self.advance_axis}] -= residue_offset_;
                    """)
                    if not self.shuffle_in_stride:
                        code.raw(f"""
                        pointer_ -= sizeof({self.dtype}) * params_.stride_ * residue_offset_;
                        """)
                    if self.shuffle_in_stride and self.have_output_ptr:
                        code.raw(f"""
                        out_pointer_ -= sizeof({self.dtype}) * params_.stride_ * residue_offset_;
                        """)

            if self.advance_axis == 1:
                code.raw(f"""
                compute_predicates_(extent_, true);
                pointer_ += {self.param_class.inc_advance_static} * (num_tile - 1);
                """)
                if self.shuffle_in_stride and self.have_output_ptr:
                    code.raw(f"""
                    out_pointer_ += {self.param_class.inc_advance_static} * (num_tile - 1);
                    """)

            else:
                code.raw(f"""
                compute_predicates_(extent_, true);
                """)
                if not self.shuffle_in_stride:
                    code.raw(f"""
                    pointer_ += params_.inc_advance_ * (num_tile - 1);
                    """)
                if self.shuffle_in_stride and self.have_output_ptr:
                    code.raw(f"""
                    out_pointer_ += params_.inc_advance_ * (num_tile - 1);
                    """)

        with code.else_():
            if self.advance_axis == 1:
                code.raw(f"""
                pointer_ += {self.param_class.inc_advance_static} * num_tile;
                """)
                if self.shuffle_in_stride and self.have_output_ptr:
                    code.raw(f"""
                    out_pointer_ += {self.param_class.inc_advance_static} * num_tile;
                    """)

            else:
                if self.shuffle_in_stride:
                    code.raw(f"""
                    thread_offset_[0] += {self.tile_shape[0]} * num_tile;
                    """)
                    if self.have_output_ptr:
                        code.raw(f"""
                        out_pointer_ += params_.inc_advance_ * num_tile;
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
                        (ss * extent_[1] + c * {self.iteration_delta[contig]}) *
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

    @pccm.cuda.member_function(name="get_output",
                               device=True,
                               const=True,
                               forceinline=True)
    def get_output(self):
        contig = 1
        strided = 0
        output_const = "const" if self.is_scatter else ""
        code = pccm.code()
        const = "const" if self.read_only else ""
        if not self.have_output_ptr:
            code.raw(f"return nullptr;")
            return code.ret(f"{self.access_t} *")
        assert self.sub_tile_shape[strided] == 1
        code.raw(f"""
        return reinterpret_cast<{output_const} {self.access_t} *>(
                out_pointer_ + 
                (c * {self.iteration_delta[contig]}) *
                    sizeof({self.dtype})) +
            v;
        """).arg("s,c,v", "int")
        return code.ret(f"{output_const} {self.access_t} *")

    def get_python(self, c: int, ss: int, v: int) -> ArrayPtr:
        contig = 1
        strided = 0
        ptr = (self.pointer_ +
               (ss * self.params_.stride_ + c * self.iteration_delta[contig]) *
               self.dtype.itemsize())  # type: ArrayPtr
        return ptr.change_access_size(self.num_sub_access) + v

    @pccm.cuda.member_function(device=True, forceinline=True)
    def inc_stride(self):
        if self.have_output_ptr and self.shuffle_in_stride:
            return pccm.FunctionCode(f"out_pointer_ += params_.inc_strided_; ")
        else:
            return pccm.FunctionCode(f"pointer_ += params_.inc_strided_; ")

    def inc_stride_python(self):
        self.pointer_ += self.params_.inc_strided_

    @pccm.cuda.member_function(device=True, forceinline=True)
    def end_iter(self):
        # back to initial location?
        if self.have_output_ptr and self.shuffle_in_stride:
            if self.advance_axis == 1:
                return pccm.FunctionCode(f"""
                out_pointer_ += params_.inc_next_ - {self.param_class.inc_advance_static};
                """)
            else:
                return pccm.FunctionCode(f"""
                out_pointer_ += params_.inc_next_;
                out_pointer_ -= params_.inc_advance_;
                """)
        else:
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

    def loadstore_with_byte_offset_template(self,
                                            store: bool,
                                            out_load: bool = False):
        contig = 1
        strided = 0
        code = pccm.cuda.PTXCode("")
        const_frag = "const" if store else ""
        const_mem = "" if store else "const"
        code.arg("frag", f"{const_frag} {self.fragment_t}&").arg(
            "byte_offset", str(self.long_index_t))
        if out_load:
            if not self.have_output_ptr or not self.is_scatter:
                return code

        # print(self.thread_access_shape[strided], self.thread_access_shape[contig], self.access_per_vector, self.access_t)
        # print(self.tmap)
        code.raw(f"""
        {self.access_t} {const_frag}*frag_ptr = reinterpret_cast<{self.access_t} {const_frag} *>(&frag);
        """)
        io_ns_name = "cutlass::arch" if CUTLASS_MODE else "tv::gemm"
        is_gather_store = store and self.have_output_ptr and not self.is_scatter
        is_scatter_load = not store and out_load and self.have_output_ptr
        if not self.have_output_ptr:
            get = "get"
        else:
            if self.is_scatter:
                if out_load:
                    get = "get_output"
                else:
                    get = "get" if store else "get"
            else:
                get = "get_output" if store else "get"

        if self.sub_tile_shape[0] == 1:
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
                            reinterpret_cast<char {const_mem} *>({get}(s, c, v)) + byte_offset;
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
                # for gather/scatter iters, we enable increment only if:
                # store output ptr or
                # is_scatter and load output
                if not self.shuffle_in_stride or is_scatter_load or is_gather_store:
                    code.raw(f"""
                    if (s != {self.thread_access_shape[strided]} - 1) {{
                        inc_stride();
                    }}
                    """)
            if not self.shuffle_in_stride or is_scatter_load or is_gather_store:
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
                                reinterpret_cast<char {const_mem} *>({get}(s, c, ss, v)) + byte_offset;
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

                if not self.shuffle_in_stride or is_gather_store:
                    code.raw(f"""
                    if (s != {self.thread_access_shape[strided]} - 1) {{
                        inc_stride();
                    }}
                """)
            if not self.shuffle_in_stride or is_gather_store:
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
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_byte_offset(self):
        return self.loadstore_with_byte_offset_template(False)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_output_with_byte_offset(self):
        return self.loadstore_with_byte_offset_template(False, True)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_byte_offset(self):
        if self.read_only and not self.have_output_ptr:
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
        if self.read_only and not self.have_output_ptr:
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
