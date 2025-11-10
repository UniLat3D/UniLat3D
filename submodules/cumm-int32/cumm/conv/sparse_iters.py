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

import contextlib
from email import iterators
from typing import List, Optional, Union

import numpy as np
import pccm
from pccm.core import FunctionCode
from pccm.targets.cuda_ptx import RegDType

from cumm import cudasim, dtypes
from cumm.common import (GemmBasic, GemmBasicKernel, TensorView,
                         TensorViewMath, TensorViewNVRTC)
from cumm.conv import bases, params
from cumm.conv.bases import LAYOUT_TYPES, ConvMode, ConvOpType
from cumm.gemm import codeops, constants, layout, thread_map
from cumm.gemm.arch.memory import GlobalLoad
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.mask import Mask


class SparseParams(bases.ConvIterParams):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape: MetaArray[int],
                 problem: params.ConvProblem,
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 is_output: bool = False,
                 increment_k_first: bool = False,
                 is_wgrad_out: bool = False,
                 is_wgrad_input: bool = False):
        super().__init__()
        self.add_dependency(TensorViewMath)
        self.dtype = dtype
        self.tmap = tmap
        self.tile_shape = tile_shape
        self.problem = problem
        self.ndim = problem.ndim
        self.is_output = is_output
        self.increment_k_first = increment_k_first
        self.is_wgrad_out = is_wgrad_out
        self.is_wgrad_input = is_wgrad_input
        self.add_param_class("params", problem, "ConvProblem")
        self.add_member("filter_c_delta", "int")
        if not is_wgrad_out and not is_wgrad_input:
            if increment_k_first:
                self.add_member("inc_c_next, inc_c_reset", "int")
            else:
                self.add_member("inc_c_next, inc_indice_reset", "int")

        self.add_member("indice_ptr_", "int const*")  # [RS, num_indices]
        self.add_member("mask_argsort_ptr_", "int const*")  # [RS, num_indices]
        self.add_member("RS", "int")

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("problem", "ConvProblem const&")
        code.arg("indice_ptr", "int const*")
        code.arg("mask_argsort_ptr", "int const*")

        code.ctor_init("indice_ptr_", "indice_ptr")
        code.ctor_init("mask_argsort_ptr_", "mask_argsort_ptr")
        C_or_K = "C" if self.problem.op_type == ConvOpType.kForward else "K"
        code.raw(f"""
        RS = problem.kernel_volume;
        filter_c_delta = {self.tile_shape[2]} * problem.split_k_slices;
        """)
        if not self.is_wgrad_out and not self.is_wgrad_input:
            code.raw(
                f"inc_c_next = filter_c_delta * {self.dtype.nbytes_str()} ;")
            if not self.increment_k_first:
                code.raw(f"""
                inc_indice_reset = problem.N * (1 - RS);
                """)
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def set_inc_reset_for_inc_k_first(self):
        code = FunctionCode()
        code.arg("gemm_iters_k", "int")
        if not self.increment_k_first:
            return code
        code.raw(f"""
        inc_c_reset = (- filter_c_delta) * gemm_iters_k * {self.dtype.nbytes_str()}  ;
        """)
        return code


class ForwardDgradSparseIOIterator(bases.ConvInputIterator):
    """for spatial sparse convolution
    Fwd/Dgrad: NRSC @ KRSC
    Wgrad: NK @ NRSC

    This class is deprecated.
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 increment_k_first: bool = False,
                 is_wgrad_out: bool = False,
                 is_wgrad_input: bool = False,
                 access_per_vector: int = 1):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype,
                         element_count,
                         sub_tile_shape[1],
                         -1,
                         access_per_vector=access_per_vector)
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.ndim = problem_size.ndim
        assert not (is_wgrad_input and is_wgrad_out), "error"
        self.is_wgrad = is_wgrad_out or is_wgrad_input
        self.is_wgrad_out = is_wgrad_out
        self.is_wgrad_input = is_wgrad_input
        if self.is_wgrad:
            assert sub_tile_shape[
                0] == 1, "backward weight don't support sub tile shape"
        else:
            assert tmap.iterations[1] == 1
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        is_output = op_type == ConvOpType.kBackwardInput
        self.params = SparseParams(dtype, tile_shape_mnk, problem_size, tmap,
                                   is_output, increment_k_first)
        self.tmap = tmap
        self.mask_cls = Mask(
            seq(self.tmap.iterations[0], self.tmap.iterations[1],
                self.sub_tile_shape[0], self.access_per_vector))
        self.add_param_class("mask", self.mask_cls, "Mask")

        assert tmap.iterations.prod() * self.sub_tile_shape[0] < 32
        self.gload = GlobalLoad(self.element_per_acc * self.dtype.itemsize(),
                                level="L2",
                                prefetch_size=128)
        self.add_param_class("gload", self.gload, "GlobalLoad")
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.increment_k_first = increment_k_first
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")
        self.invalid_indice_offset = self.dtype.bitsize(
        ) * self.element_per_acc
        self.add_member("params_", "Params &")
        self.add_member("problem_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)
        # self.add_member("origin_pointer_", self.const_byte_pointer)

        if not self.increment_k_first:
            self.add_member("filter_index_", f"int")
        if not self.is_wgrad or self.is_wgrad_input:
            self.add_member("indice_ptr_", f"int const*")

        if not self.is_wgrad:
            self.add_member("reduce_channel_offset_", "int")
            self.add_member("reduce_channel_offset_backup_", "int")

            self.add_member("mask_reset_backup_",
                            f"tv::array<uint32_t, {self.access_per_vector}>")
        else:
            self.add_member("stride_offset_", f"int")
        # self.add_member("origin_indice_ptr_", f"int const*")

        self.add_member("mask_",
                        f"tv::array<uint32_t, {self.access_per_vector}>")

        # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")
        self.add_member(
            "indices_",
            str(dtypes.int32),
            array=f"[{self.tmap.iterations[0] * self.sub_tile_shape[0]}]")

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params &")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")

        code.ctor_init("params_", "params")
        code.ctor_init("problem_", "problem_size")
        # code.ctor_init("reduce_channel_offset_", "0")
        if not self.increment_k_first:
            code.ctor_init("filter_index_", "0")
        if not self.is_wgrad or self.is_wgrad_input:
            code.ctor_init("indice_ptr_", "params.indice_ptr_")

        # code.ctor_init("origin_indice_ptr_", "params.indice_ptr_")
        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        if not self.is_wgrad:
            code.raw(f"""
            int stride_offset_ = thread_offset[0];
            """)
        else:
            code.raw(f"""
            stride_offset_ = thread_offset[0];
            """)
        code.raw(f"""
        // update_indices();
        pointer_ = reinterpret_cast<{self.const_byte_pointer}>(ptr + thread_offset[1]);
        // std::uintptr_t access_pointer_num = reinterpret_cast<std::uintptr_t>(pointer_);

        // if (access_pointer_num % 16 != 0){{
        //     tv::printf2_block_once("BBBBBBBBBBBBBBBBSFASF");
        // }}

        // origin_pointer_ = pointer_;
        params.mask_argsort_ptr_ += stride_offset_;
        // mask_ = 0;
        mask_.clear();
        """)

        if self.is_wgrad:
            C_or_K = "C" if self.is_wgrad_input else "K"
            with self.tmap.tmap_loop(code, "s", "c"):
                with code.range_("v", self.access_per_vector,
                                 "TV_PRAGMA_UNROLL"):
                    code.raw(f"""
                    uint32_t pred = ((stride_offset_ + s * {self.tmap.delta[0]}) < problem_.N) &&
                        (thread_offset[1] + c * {self.tmap.delta[1]} + v < problem_.{C_or_K});
                    mask_[v] |= (pred << (s * {self.tmap.iterations[1]} + c));
                    """)
        else:
            C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"

            code.raw(f"""
            reduce_channel_offset_ = thread_offset[1];
            reduce_channel_offset_backup_ = thread_offset[1];
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
                TV_PRAGMA_UNROLL
                for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.access_per_vector}; ++v){{
                        uint32_t pred = (stride_offset_ + s * {self.tmap.delta[0]} + ss) < problem_.N;
                        mask_[v] |= (pred << (s * {self.sub_tile_shape[0]} + ss));
                    }}
                }}
            }}
            """)
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int v = 0; v < {self.access_per_vector}; ++v){{
                mask_[v] = thread_offset[1] + v * {self.element_per_acc} >= problem_.{C_or_K} ? 0 : mask_[v];
            }}
            mask_reset_backup_ = mask_;
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def update_indices(self):
        code = pccm.cuda.PTXCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad_out:
            # if False:
            # wgrad out only need shuffle.
            for s in range(self.tmap.iterations[0]):
                for ss in range(self.sub_tile_shape[0]):
                    code.raw(f"uint32_t pred{s}_{ss};")
                    code.raw(
                        f"pred{s}_{ss} = mask_[0] & (1u << ({s * self.sub_tile_shape[0] * self.tmap.iterations[1]} + {ss}));"
                    )
                    with code.asm_block() as asm:
                        mask_ptr = asm.reg_ptr("indices_", RegDType.B32)
                        pred_ptr = asm.ext_reg(f"pred{s}_{ss}", RegDType.B32)
                        mask_arg_ptr = asm.global_ptr(
                            "params_.mask_argsort_ptr_")
                        with asm.pred_if("p", "ne", pred_ptr, 0):
                            asm.ld(
                                mask_arg_ptr +
                                (s * self.tmap.delta[0] + ss) * 4,
                                mask_ptr[s * self.sub_tile_shape[0] + ss])
                    # code.raw(f"""
                    # indices_[{s * self.sub_tile_shape[0] + ss}] = pred{s}_{ss} ? params_.mask_argsort_ptr_[{(s * self.tmap.delta[0] + ss)}] : 0;

                    # """)
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
                TV_PRAGMA_UNROLL
                for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                    indices_[s * {self.sub_tile_shape[0]} + ss] = indices_[s * {self.sub_tile_shape[0]} + ss] * 
                            problem_.K * {self.dtype.nbytes_str()} ;
                }}
            }}
            """)
        else:
            code.raw(
                f"int mask_inds[{self.tmap.iterations[0] * self.sub_tile_shape[0]}];"
            )
            code.raw(f"uint32_t pred;")
            for s in range(self.tmap.iterations[0]):
                for ss in range(self.sub_tile_shape[0]):
                    code.raw(
                        f"pred = mask_[0] & (1u << ({s * self.sub_tile_shape[0] * self.tmap.iterations[1]} + {ss}));"
                    )
                    with code.asm_block() as asm:
                        mask_ptr = asm.reg_ptr("mask_inds", RegDType.B32)
                        pred_ptr = asm.ext_reg("pred", RegDType.B32)
                        mask_arg_ptr = asm.global_ptr(
                            "params_.mask_argsort_ptr_")
                        with asm.pred_if("p", "ne", pred_ptr, 0):
                            asm.ld(
                                mask_arg_ptr +
                                (s * self.tmap.delta[0] + ss) * 4,
                                mask_ptr[s * self.sub_tile_shape[0] + ss])
                    # code.raw(f"""
                    # mask_inds[{s * self.sub_tile_shape[0] + ss}] = pred ? params_.mask_argsort_ptr_[{(s * self.tmap.delta[0] + ss)}] : 0;

                    # """)

            if self.is_wgrad_input:
                C_or_K = "C"
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
                TV_PRAGMA_UNROLL
                for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                    if (mask_[0] & (1u << (s * {self.sub_tile_shape[0] * self.tmap.iterations[1]} + ss))){{
                        indices_[s * {self.sub_tile_shape[0]} + ss] = 
                        indice_ptr_[mask_inds[s * {self.sub_tile_shape[0]} + ss]] * 
                            problem_.{C_or_K} * {self.dtype.nbytes_str()} ;
                    }}
                }}
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_if_not_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        code.arg("v", "int")
        code.raw(f"""
        mask_[v] = pred ? mask_[v] : 0;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_all_mask_if_not_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int v = 0; v < {self.access_per_vector}; ++v){{
            mask_[v] = pred ? mask_[v] : 0;
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_all_mask_if_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int v = 0; v < {self.access_per_vector}; ++v){{
            mask_[v] = pred ? 0:  mask_[v];
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_if_pred(self):
        """TODO why these code cause misaligned error?
        code.arg("pred", "bool")
        with code.asm_block() as asm:
            pred_ptr = asm.ext_reg("(int)pred", RegDType.B32)
            mask_ptr = asm.ext_reg("mask_[v]", RegDType.B32)
            with asm.pred_if("p", "ne", pred_ptr, 0):
                asm.mov(mask_ptr, 0)
        but following code is ok?
        """
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        code.arg("v", "int")
        code.raw(f"""
        mask_[v] = pred ? 0 : mask_[v];
        """)
        return code
        with code.asm_block() as asm:
            pred_ptr = asm.ext_reg("(int)pred", RegDType.B32)
            mask_ptr = asm.ext_reg("mask_[v]", RegDType.B32)
            reg = asm.reg("m", RegDType.B32)
            asm.mov(reg, mask_ptr)
            with asm.pred_if("p", "ne", pred_ptr, 0):
                asm.mov(reg, 0)
            asm.mov(mask_ptr, reg)

        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            code.raw(f"""
            increment_no_clear_mask();
            clear_mask_if_batch_unbound();
            """)
            return code

        if self.increment_k_first:
            return code
        else:
            code.raw(f"""
            if (++filter_index_ < params_.RS) {{
                indice_ptr_ += problem_.N;
                return;
            }}
            filter_index_ = 0;
            reduce_channel_offset_ += params_.filter_c_delta;
            pointer_ += params_.inc_c_next;
            indice_ptr_ += params_.inc_indice_reset;
            TV_PRAGMA_UNROLL
            for (int v = 0; v < {self.access_per_vector}; ++v){{
                clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_.{C_or_K}, v);
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_no_clear_mask(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            code.raw(f"""
            stride_offset_ += params_.filter_c_delta;
            params_.mask_argsort_ptr_ += params_.filter_c_delta;
            """)
            return code

        if self.increment_k_first:
            return code
        else:
            code.raw(f"""
            if (++filter_index_ < params_.RS) {{
                indice_ptr_ += problem_.N;
                return;
            }}
            filter_index_ = 0;
            reduce_channel_offset_ += params_.filter_c_delta;
            pointer_ += params_.inc_c_next;
            indice_ptr_ += params_.inc_indice_reset;
            TV_PRAGMA_UNROLL
            for (int v = 0; v < {self.access_per_vector}; ++v){{
                clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_.{C_or_K}, v);
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_if_batch_unbound(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            with self.tmap.tmap_loop(code, "s"):
                code.raw(f"""
                if (stride_offset_ + s * {self.tmap.delta[0]} >= problem_.N){{
                    uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1]});
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.access_per_vector}; ++v){{
                        mask_[v] = mask_[v] & (~mask);
                    }}
                    // mask_ = mask_ & (~mask);
                }}
                """)
            return code
        if self.increment_k_first:
            return code
        else:
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int v = 0; v < {self.access_per_vector}; ++v){{
                clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_.{C_or_K}, v);
            }}
            """)
        return code

    @pccm.cuda.member_function(name="operator+=",
                               device=True,
                               forceinline=True)
    def increment_num(self):
        code = FunctionCode()
        code.arg("num", "int")
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            code.raw(f"""
            stride_offset_ += num * params_.filter_c_delta;
            params_.mask_argsort_ptr_ += num * params_.filter_c_delta;
            """)
            with self.tmap.tmap_loop(code, "s"):
                code.raw(f"""
                if (stride_offset_ + s * {self.tmap.delta[0]} >= problem_.N){{
                    uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1]});
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.access_per_vector}; ++v){{
                        mask_[v] = mask_[v] & (~mask);
                    }}
                }}
                """)
            return code
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_k(self):
        code = FunctionCode()
        if not self.increment_k_first or self.is_wgrad:
            return code
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        code.raw(f"""
        pointer_ += params_.inc_c_next;
        reduce_channel_offset_ += params_.filter_c_delta;
        TV_PRAGMA_UNROLL
        for (int v = 0; v < {self.access_per_vector}; ++v){{
            clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_.{C_or_K}, v);
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_filter(self):
        code = FunctionCode()
        if not self.increment_k_first or self.is_wgrad_out:
            return code
        code.raw(f"""
        indice_ptr_ += problem_.N;
        """)
        return code

    @pccm.cuda.member_function(name="increment_filter",
                               device=True,
                               forceinline=True)
    def increment_filter_with_num(self):
        code = FunctionCode()
        code.arg("num", "int")
        if not self.increment_k_first or self.is_wgrad_out:
            return code
        code.raw(f"""
        indice_ptr_ += problem_.N * num;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def reset_k(self):
        code = FunctionCode()
        if not self.increment_k_first or self.is_wgrad:
            return code
        code.raw(f"""
        pointer_ += params_.inc_c_reset;
        mask_ = mask_reset_backup_;
        reduce_channel_offset_ = reduce_channel_offset_backup_;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get_indice_offset(self):
        code = FunctionCode()
        code.arg("stride, contig, ss", f"int")
        code.raw(f"""
        return indices_[stride * {self.sub_tile_shape[0]} + ss];
        """)
        return code.ret(f"int")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.arg("indice_offset", f"int")
        code.raw(f"""
        return reinterpret_cast<{self.const_access_pointer}>( pointer_ + indice_offset);
        """)
        code.ret(self.const_access_pointer)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s", "c"):
            with code.range_("ss", self.sub_tile_shape[0], "TV_PRAGMA_UNROLL"):
                with code.range_("v", self.access_per_vector,
                                 "TV_PRAGMA_UNROLL"):
                    code.raw(f"""
                    int mask_idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + 
                        c * {self.sub_tile_shape[0]} + ss;
                    int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0] * self.access_per_vector} + 
                        c * {self.sub_tile_shape[0] * self.access_per_vector} + ss * {self.access_per_vector} + v;
                    auto indice_offset = get_indice_offset(s, c, ss);
                    """)
                    if self.is_wgrad_out:
                        code.raw(
                            f"bool valid = bool(mask_[v] & (1u << mask_idx));")
                    else:
                        code.raw(
                            f"bool valid = bool(mask_[v] & (1u << mask_idx)) && (indice_offset >= 0);"
                        )
                    # if self.is_wgrad_out:
                    num_inds = self.tmap.iterations[0] * self.sub_tile_shape[0]
                    inds_unpack = ", ".join(
                        [f"indices_[{i}]" for i in range(num_inds)])
                    code.raw(f"""
                    auto access_pointer = reinterpret_cast<{self.const_access_pointer}>(pointer_ + indice_offset + 
                        c * {self.dtype.nbytes_str(self.tmap.delta[1] * self.dtype.bitsize())}) + v;
                    // std::uintptr_t access_pointer_num = reinterpret_cast<std::uintptr_t>(access_pointer);
                    // std::uintptr_t access_pointer_num2 = reinterpret_cast<std::uintptr_t>(pointer_);

                    // if (access_pointer_num % 16 != 0 && valid){{
                    //     tv::printf2(valid, s, access_pointer_num2 % 16, indice_offset, indice_offset%16, "AS", {inds_unpack}, "A", blockIdx.x, blockIdx.y, blockIdx.z);
                    // }}

                    // tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                    //    frag_ptr[idx], access_pointer, valid);
                    GlobalLoad::run(frag_ptr[idx], access_pointer, valid);
                    """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code
    
    def can_multistage_load(self):
        return self.sub_tile_shape[0] == 1 and self.access_per_vector == 1
    
    def enumurate_get_param(self, python=False):
        for s in range(self.tmap.iterations[0]):
            for c in range(self.tmap.iterations[1]):
                if python:
                    yield s, c
                else:
                    yield f"{s}, {c}"
    
    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_ptr_with_param(self):
        code = pccm.FunctionCode()
        code.arg("s, c", "int")
        code.arg("valid_ref", "bool&")
        code.ret(f"{self.const_access_pointer}")
        code.raw(f"""
            int mask_idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + 
                c * {self.sub_tile_shape[0]} + 0;
            auto indice_offset = get_indice_offset(s, c, 0);
        """)
        if self.is_wgrad_out:
            code.raw(
                f"valid_ref = bool(mask_[0] & (1u << mask_idx));")
        else:
            code.raw(
                f"valid_ref = bool(mask_[0] & (1u << mask_idx)) && (indice_offset >= 0);"
            )
        code.raw(f"""
            auto access_pointer = reinterpret_cast<{self.const_access_pointer}>(pointer_ + indice_offset + 
                c * {self.dtype.nbytes_str(self.tmap.delta[1] * self.dtype.bitsize())}) + 0;
            return access_pointer;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code


class ForwardDgradSparseIOIteratorV2Mask(bases.ConvInputIterator):
    """for spatial sparse convolution
    Fwd/Dgrad: NRSC @ KRSC
    Wgrad: NK @ NRSC
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 increment_k_first: bool = False,
                 is_wgrad_out: bool = False,
                 is_wgrad_input: bool = False,
                 access_per_vector: int = 1):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype,
                         element_count,
                         sub_tile_shape[1],
                         access_per_vector=access_per_vector)
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.ndim = problem_size.ndim
        assert not (is_wgrad_input and is_wgrad_out), "error"
        self.is_wgrad = is_wgrad_out or is_wgrad_input
        self.is_wgrad_out = is_wgrad_out
        self.is_wgrad_input = is_wgrad_input
        if self.is_wgrad:
            assert sub_tile_shape[
                0] == 1, "backward weight don't support sub tile shape"
        else:
            assert tmap.iterations[1] == 1
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        is_output = op_type == ConvOpType.kBackwardInput
        self.params = SparseParams(dtype, tile_shape_mnk, problem_size, tmap,
                                   is_output, increment_k_first)
        self.tmap = tmap
        self.mask_cls = Mask(
            seq(self.tmap.iterations[0], self.tmap.iterations[1],
                self.sub_tile_shape[0], self.access_per_vector))
        self.add_param_class("mask", self.mask_cls, "Mask")

        assert tmap.iterations.prod() * self.sub_tile_shape[0] < 32
        self.gload = GlobalLoad(self.element_per_acc * self.dtype.itemsize(),
                                level="L2",
                                prefetch_size=128)
        self.add_param_class("gload", self.gload, "GlobalLoad")
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.increment_k_first = increment_k_first
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")
        self.invalid_indice_offset = self.dtype.bitsize(
        ) * self.element_per_acc
        self.add_member("params_", "Params &")
        self.add_member("problem_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)
        # self.add_member("origin_pointer_", self.const_byte_pointer)

        if not self.increment_k_first:
            self.add_member("filter_index_", f"int")
        if not self.is_wgrad or self.is_wgrad_input:
            self.add_member("indice_ptr_", f"int const*")

        if not self.is_wgrad:
            self.add_member("reduce_channel_offset_", "int")
            self.add_member("reduce_channel_offset_backup_", "int")

            self.add_member("mask_reset_backup_", f"Mask")
        else:
            self.add_member("stride_offset_", f"int")
        # self.add_member("origin_indice_ptr_", f"int const*")

        self.add_member("mask_", f"Mask")

        # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")
        self.add_member(
            "indices_",
            str(dtypes.int32),
            array=f"[{self.tmap.iterations[0] * self.sub_tile_shape[0]}]")

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params &")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")

        code.ctor_init("params_", "params")
        code.ctor_init("problem_", "problem_size")
        # code.ctor_init("reduce_channel_offset_", "0")
        if not self.increment_k_first:
            code.ctor_init("filter_index_", "0")
        if not self.is_wgrad or self.is_wgrad_input:
            code.ctor_init("indice_ptr_", "params.indice_ptr_")

        # code.ctor_init("origin_indice_ptr_", "params.indice_ptr_")
        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        if not self.is_wgrad:
            code.raw(f"""
            int stride_offset_ = thread_offset[0];
            """)
        else:
            code.raw(f"""
            stride_offset_ = thread_offset[0];
            """)
        code.raw(f"""
        // update_indices();
        pointer_ = reinterpret_cast<{self.const_byte_pointer}>(ptr + thread_offset[1]);
        // origin_pointer_ = pointer_;
        params.mask_argsort_ptr_ += stride_offset_;
        // mask_ = 0;
        mask_.clear();
        """)

        if self.is_wgrad:
            C_or_K = "C" if self.is_wgrad_input else "K"
            with self.tmap.tmap_loop(code, "s", "c"):
                with code.range_("v", self.access_per_vector,
                                 "TV_PRAGMA_UNROLL"):
                    code.raw(f"""
                    uint32_t pred = ((stride_offset_ + s * {self.tmap.delta[0]}) < problem_.N) &&
                        (thread_offset[1] + c * {self.tmap.delta[1]} + v < problem_.{C_or_K});
                    mask_.set_coord(pred, s, c, 0, v);
                    // mask_[v] |= (pred << (s * {self.tmap.iterations[1]} + c));
                    """)
        else:
            C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"

            code.raw(f"""
            reduce_channel_offset_ = thread_offset[1];
            reduce_channel_offset_backup_ = thread_offset[1];
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
                TV_PRAGMA_UNROLL
                for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.access_per_vector}; ++v){{
                        uint32_t pred = (stride_offset_ + s * {self.tmap.delta[0]} + ss) < problem_.N;
                        // mask_[v] |= (pred << (s * {self.sub_tile_shape[0]} + ss));
                        mask_.set_coord(pred, s, 0, ss, v);
                    }}
                }}
            }}
            """)
            for v in range(self.access_per_vector):
                pred = f"thread_offset[1] + {v * self.element_per_acc} >= problem_.{C_or_K}"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (None, None, None, v))
            code.raw(f"""
            // TV_PRAGMA_UNROLL
            // for (int v = 0; v < {self.access_per_vector}; ++v){{
            //     mask_[v] = thread_offset[1] + v * {self.element_per_acc} >= problem_.{C_or_K} ? 0 : mask_[v];
            // }}
            mask_reset_backup_ = mask_;
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def update_indices(self):
        code = pccm.cuda.PTXCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad_out:
            # if False:
            # wgrad out only need shuffle.
            for s in range(self.tmap.iterations[0]):
                for ss in range(self.sub_tile_shape[0]):
                    code.raw(f"uint32_t pred{s}_{ss};")
                    # code.raw(
                    #     f"pred{s}_{ss} = mask_[0] & (1u << ({s} * {self.sub_tile_shape[0]} + {ss}));"
                    # )
                    code.raw(
                        f"pred{s}_{ss} = mask_.query_coord({s}, 0, {ss}, 0);")

                    with code.asm_block() as asm:
                        mask_ptr = asm.reg_ptr("indices_", RegDType.B32)
                        pred_ptr = asm.ext_reg(f"pred{s}_{ss}", RegDType.B32)
                        mask_arg_ptr = asm.global_ptr(
                            "params_.mask_argsort_ptr_")
                        with asm.pred_if("p", "ne", pred_ptr, 0):
                            asm.ld(
                                mask_arg_ptr +
                                (s * self.tmap.delta[0] + ss) * 4,
                                mask_ptr[s * self.sub_tile_shape[0] + ss])
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
                TV_PRAGMA_UNROLL
                for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                    indices_[s * {self.sub_tile_shape[0]} + ss] = indices_[s * {self.sub_tile_shape[0]} + ss] * 
                            problem_.K * {self.dtype.nbytes_str()} ;
                }}
            }}
            """)
        else:
            code.raw(
                f"int mask_inds[{self.tmap.iterations[0] * self.sub_tile_shape[0]}];"
            )
            code.raw(f"uint32_t pred;")
            for s in range(self.tmap.iterations[0]):
                for ss in range(self.sub_tile_shape[0]):
                    # code.raw(
                    #     f"pred = mask_[0] & (1u << ({s} * {self.sub_tile_shape[0]} + {ss}));"
                    # )
                    code.raw(f"pred = mask_.query_coord({s}, 0, {ss}, 0);")
                    with code.asm_block() as asm:
                        mask_ptr = asm.reg_ptr("mask_inds", RegDType.B32)
                        pred_ptr = asm.ext_reg("pred", RegDType.B32)
                        mask_arg_ptr = asm.global_ptr(
                            "params_.mask_argsort_ptr_")
                        with asm.pred_if("p", "ne", pred_ptr, 0):
                            asm.ld(
                                mask_arg_ptr +
                                (s * self.tmap.delta[0] + ss) * 4,
                                mask_ptr[s * self.sub_tile_shape[0] + ss])
            if self.is_wgrad_input:
                C_or_K = "C"
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
                TV_PRAGMA_UNROLL
                for (int ss = 0; ss < {self.sub_tile_shape[0]}; ++ss){{
                    // if (mask_[0] & (1u << (s * {self.sub_tile_shape[0]} + ss)))
                    if (mask_.query_coord(s, 0, ss, 0)){{
                        indices_[s * {self.sub_tile_shape[0]} + ss] = 
                        indice_ptr_[mask_inds[s * {self.sub_tile_shape[0]} + ss]] * 
                            problem_.{C_or_K} * {self.dtype.nbytes_str()} ;
                    }}
                }}
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_all_mask_if_not_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        code.raw(f"return mask_.clear_all_mask_if_not_pred(pred);")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_all_mask_if_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        code.raw(f"return mask_.clear_all_mask_if_pred(pred);")
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            code.raw(f"""
            increment_no_clear_mask();
            clear_mask_if_batch_unbound();
            """)
            return code

        if self.increment_k_first:
            return code
        else:
            code.raw(f"""
            if (++filter_index_ < params_.RS) {{
                indice_ptr_ += problem_.N;
                return;
            }}
            filter_index_ = 0;
            reduce_channel_offset_ += params_.filter_c_delta;
            pointer_ += params_.inc_c_next;
            indice_ptr_ += params_.inc_indice_reset;
            """)

            for v in range(self.access_per_vector):
                pred = f"reduce_channel_offset_ + {v * self.element_per_acc} >= problem_.{C_or_K}"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (None, None, None, v))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_no_clear_mask(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            code.raw(f"""
            stride_offset_ += params_.filter_c_delta;
            params_.mask_argsort_ptr_ += params_.filter_c_delta;
            """)
            return code

        if self.increment_k_first:
            return code
        else:
            code.raw(f"""
            if (++filter_index_ < params_.RS) {{
                indice_ptr_ += problem_.N;
                return;
            }}
            filter_index_ = 0;
            reduce_channel_offset_ += params_.filter_c_delta;
            pointer_ += params_.inc_c_next;
            indice_ptr_ += params_.inc_indice_reset;
            """)
            for v in range(self.access_per_vector):
                pred = f"reduce_channel_offset_ + {v * self.element_per_acc} >= problem_.{C_or_K}"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (None, None, None, v))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_if_batch_unbound(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            for s in range(self.tmap.iterations[0]):
                pred = f"stride_offset_ + {s * self.tmap.delta[0]} >= problem_.N"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (s, None, None, None))
            return code
        if self.increment_k_first:
            return code
        else:
            for v in range(self.access_per_vector):
                pred = f"reduce_channel_offset_ + {v * self.element_per_acc} >= problem_.{C_or_K}"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (None, None, None, v))
        return code

    @pccm.cuda.member_function(name="operator+=",
                               device=True,
                               forceinline=True)
    def increment_num(self):
        code = FunctionCode()
        code.arg("num", "int")
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if self.is_wgrad:
            code.raw(f"""
            stride_offset_ += num * params_.filter_c_delta;
            params_.mask_argsort_ptr_ += num * params_.filter_c_delta;
            """)
            for s in range(self.tmap.iterations[0]):
                pred = f"stride_offset_ + {s * self.tmap.delta[0]} >= problem_.N"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (s, None, None, None))
            return code
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_k(self):
        code = FunctionCode()
        if not self.increment_k_first or self.is_wgrad:
            return code
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        code.raw(f"""
        pointer_ += params_.inc_c_next;
        reduce_channel_offset_ += params_.filter_c_delta;
        """)
        for v in range(self.access_per_vector):
            pred = f"reduce_channel_offset_ + {v * self.element_per_acc} >= problem_.{C_or_K}"
            self.mask_cls.clear_mask_if_pred_template(code, pred,
                                                      (None, None, None, v))

        # code.raw(f"""
        # TV_PRAGMA_UNROLL
        # for (int v = 0; v < {self.access_per_vector}; ++v){{
        #     clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_.{C_or_K}, v);
        # }}
        # """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_filter(self):
        code = FunctionCode()
        if not self.increment_k_first or self.is_wgrad_out:
            return code
        code.raw(f"""
        indice_ptr_ += problem_.N;
        """)
        return code

    @pccm.cuda.member_function(name="increment_filter",
                               device=True,
                               forceinline=True)
    def increment_filter_with_num(self):
        code = FunctionCode()
        code.arg("num", "int")
        if not self.increment_k_first or self.is_wgrad_out:
            return code
        code.raw(f"""
        indice_ptr_ += problem_.N * num;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def reset_k(self):
        code = FunctionCode()
        if not self.increment_k_first or self.is_wgrad:
            return code
        code.raw(f"""
        pointer_ += params_.inc_c_reset;
        mask_ = mask_reset_backup_;
        reduce_channel_offset_ = reduce_channel_offset_backup_;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get_indice_offset(self):
        code = FunctionCode()
        code.arg("stride, contig, ss", f"int")
        code.raw(f"""
        return indices_[stride * {self.sub_tile_shape[0]} + ss];
        """)
        return code.ret(f"int")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.arg("indice_offset", f"int")
        code.raw(f"""
        return reinterpret_cast<{self.const_access_pointer}>( pointer_ + indice_offset);
        """)
        code.ret(self.const_access_pointer)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s", "c"):
            with code.range_("ss", self.sub_tile_shape[0], "TV_PRAGMA_UNROLL"):
                code.raw(f"auto indice_offset = get_indice_offset(s, c, ss);")
                with code.range_("v", self.access_per_vector,
                                 "TV_PRAGMA_UNROLL"):
                    code.raw(f"""
                    // int mask_idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + 
                    //     c * {self.sub_tile_shape[0]} + ss;
                    int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0] * self.access_per_vector} + 
                        c * {self.sub_tile_shape[0] * self.access_per_vector} + ss * {self.access_per_vector} + v;
                    
                    """)
                    if self.is_wgrad_out:
                        code.raw(
                            f"bool valid = bool(mask_.query_coord(s, c, ss, v));"
                        )
                    else:
                        code.raw(
                            f"bool valid = bool(mask_.query_coord(s, c, ss, v)) && (indice_offset >= 0);"
                        )
                    # if self.is_wgrad_out:
                    code.raw(f"""
                    auto access_pointer = reinterpret_cast<{self.const_access_pointer}>(pointer_ + indice_offset + 
                        c * {self.dtype.nbytes_str(self.tmap.delta[1] * self.dtype.bitsize())}) + v;
                    // tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                    //    frag_ptr[idx], access_pointer, valid);
                    GlobalLoad::run(frag_ptr[idx], access_pointer, valid);
                    """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code
