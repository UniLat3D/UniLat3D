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
from typing import List, Optional, Union

import numpy as np
import pccm
from pccm.core import FunctionCode
from pccm.targets.cuda_ptx import RegDType

from cumm import dtypes
from cumm.common import (GemmBasic, GemmBasicKernel, TensorViewMath,
                         TensorViewNVRTC)
from cumm.conv import bases, params
from cumm.conv.bases import LAYOUT_TYPES, ConvMode, ConvOpType
from cumm.gemm import codeops, constants, layout, thread_map
from cumm.gemm.arch.memory import GlobalLoad
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.mask import Mask


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


class AnalyticParams(bases.ConvIterParams):
    def __init__(self,
                 problem_size: params.ConvProblem,
                 input_layout: LAYOUT_TYPES,
                 is_output: bool = False,
                 has_rsc: bool = False):
        super().__init__()
        self.problem_size = problem_size
        self.is_output = is_output
        self.ndim = problem_size.ndim
        self.has_rsc = has_rsc
        self.add_param_class("input_layout", input_layout, "Layout")
        self.add_param_class("prob", problem_size, "ConvProblem")
        self.add_member("layout", "Layout")
        self.npq_layout = layout.TensorGeneric(problem_size.ndim + 1, True)
        self.add_param_class("layoutnpq", self.npq_layout, "LayoutNPQ")
        self.add_member("layout_npq", "LayoutNPQ")
        if has_rsc:
            self.add_member("layout_rsc", "LayoutNPQ")

        # self.add_member("filter_c_delta", "int")

        self.layout = input_layout

        # cudasim
        self.layout_: Optional[LAYOUT_TYPES] = None

    def python_ctor(self, conv_psize: params.ConvProblem,
                    layout: LAYOUT_TYPES):
        new_obj = AnalyticParams(self.problem_size, self.layout,
                                 self.is_output)
        new_obj.layout_ = layout
        return new_obj

    # @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    # def defctor(self):
    #     return pccm.code()

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("problem", "ConvProblem const&")
        code.arg("layout", "Layout const&")
        code.ctor_init("layout", "layout")
        if self.is_output:
            pqs = codeops.unpack("problem.output_dims", range(self.ndim))
        else:
            pqs = codeops.unpack("problem.input_dims", range(self.ndim))
        rss = codeops.unpack("problem.ksize", range(self.ndim))

        code.ctor_init("layout_npq",
                       f"LayoutNPQ::from_shape({{problem.N, {pqs}}})")
        if self.has_rsc:
            code.ctor_init("layout_rsc",
                           f"LayoutNPQ::from_shape({{{rss}, problem.C}})")
        # code.ctor_init("filter_c_delta", "")
        return code


class IOOptParams(bases.ConvIterParams):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape: MetaArray[int],
                 problem: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 is_output: bool = False):
        super().__init__()
        self.add_dependency(TensorViewMath)
        if not isinstance(input_layout, layout.TensorGeneric):
            raise NotImplementedError
        self.dtype = dtype
        self.tmap = tmap
        self.tile_shape = tile_shape
        self.problem = problem
        self.ndim = problem.ndim
        self.is_output = is_output
        self.add_param_class("input_layout", input_layout, "Layout")
        self.add_param_class("prob", problem, "ConvProblem")
        self.add_member("layout", "Layout")
        self.add_member("inc_next", "int64_t", array=f"[{self.ndim + 1}]")
        self.add_member("filter_c_delta", "int")
        self.npq_layout = layout.TensorGeneric(problem.ndim + 1, True)
        self.add_param_class("layoutnpq", self.npq_layout, "LayoutNPQ")
        self.add_member("layout_npq", "LayoutNPQ")

        self.layout = input_layout

        # cudasim
        self.layout_: Optional[LAYOUT_TYPES] = None
        self.inc_next_ = [0] * (problem.ndim + 1)
        self.filter_c_delta_ = 0

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("problem", "ConvProblem const&")
        code.arg("layout", "Layout const&")
        code.ctor_init("layout", "layout")
        if self.is_output:
            pqs = codeops.unpack("problem.output_dims", range(self.ndim))
        else:
            pqs = codeops.unpack("problem.input_dims", range(self.ndim))

        code.ctor_init("layout_npq",
                       f"LayoutNPQ::from_shape({{problem.N, {pqs}}})")
        if self.is_output:
            code.raw(f"""
            int conv_sign = problem.mode == tv::gemm::ConvMode::kConvolution ? 1 : -1;
            int prev_back = 0;
            """)
        else:
            code.raw(f"""
            int conv_sign = problem.mode == tv::gemm::ConvMode::kConvolution ? -1 : 1;
            int prev_back = 0;
            """)

        inc_next_plain = [
            f"layout.strides[{self.ndim - i}] * problem.dilation[{self.ndim - 1 - i}]"
            for i in range(self.ndim)
        ]

        for i in range(self.ndim):
            code.raw(f"""
            inc_next[{i}] = conv_sign * ({inc_next_plain[i]} - prev_back) * {self.dtype.bitsize()} / 8;
            prev_back += (problem.ksize[{self.ndim - 1 - i}] - 1) * {inc_next_plain[i]};
            """)
        code.raw(f"""
        inc_next[{self.ndim}] = {self.tile_shape[2]} * problem.split_k_slices - conv_sign * prev_back;
        inc_next[{self.ndim}] = inc_next[{self.ndim}] * {self.dtype.bitsize()} / 8;
        filter_c_delta = {self.tile_shape[2]} * problem.split_k_slices;
        """)
        return code

    def python_ctor(self, problem: params.ConvProblem, layout: LAYOUT_TYPES):
        new_obj = IOOptParams(self.dtype, self.tile_shape, self.problem,
                              self.layout, self.tmap, self.is_output)
        new_obj.layout_ = layout

        if problem.mode_ == ConvMode.kConvolution:
            conv_sign = -1
        else:
            conv_sign = 1
        inc_next_plain = [
            layout.strides[self.ndim - i - 1] *
            problem.dilation_[self.ndim - i - 1] for i in range(self.ndim)
        ]
        prev_back = 0
        for i in range(self.ndim):
            new_obj.inc_next_[i] = conv_sign * (inc_next_plain[i] - prev_back)
            prev_back += (problem.ksize_[self.ndim - 1 - i] -
                          1) * inc_next_plain[i]
        new_obj.inc_next_[self.ndim] = self.tile_shape[
            2] * problem.split_k_slices_ - conv_sign * prev_back
        new_obj.inc_next_ = [
            i * self.dtype.bitsize() // 8 for i in new_obj.inc_next_
        ]
        new_obj.filter_c_delta_ = self.tile_shape[2] * problem.split_k_slices_
        return new_obj


class WeightOptParams(bases.ConvIterParams):
    def __init__(self,
                 dtype: dtypes.DType,
                 tile_shape: MetaArray[int],
                 problem: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 increment_k_first: bool = False):
        super().__init__()
        self.add_dependency(TensorViewMath)
        if not isinstance(input_layout, layout.TensorGeneric):
            raise NotImplementedError
        self.dtype = dtype
        self.tmap = tmap
        self.tile_shape = tile_shape
        self.problem = problem
        self.ndim = problem.ndim
        self.op_type = problem.op_type
        self.increment_k_first = increment_k_first
        self.mask_sparse = problem.mask_sparse
        self.add_param_class("input_layout", input_layout, "Layout")
        self.add_param_class("prob", problem, "ConvProblem")
        self.add_member("layout", "Layout")
        self.add_member("inc_strided, inc_rs, inc_c", "int64_t")
        self.add_member("filter_c_delta", "int")

        if not increment_k_first:
            self.add_member("kernel_prod", "int")
        self.add_member("stride_rsc_bytes", "int")
        if increment_k_first:
            self.add_member("inc_c_reset", "int64_t")
        self.layout = input_layout
        # cudasim
        self.layout_: Optional[LAYOUT_TYPES] = None
        self.inc_strided_ = 0
        self.inc_rs_ = 0
        self.inc_c_ = 0
        self.filter_c_delta_ = 0

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("problem", "ConvProblem const&")
        code.arg("layout", "Layout const&")
        code.ctor_init("layout", "layout")
        mul_stride_if_bwd = ""
        # if forward, RSC, else, C
        RSC_if_fwd_else_C = f"layout.strides[0]"
        last_dim = 1 if self.mask_sparse else self.ndim
        C_if_fwd_else_KRSC = f"layout.strides[{last_dim}]"

        if self.op_type == ConvOpType.kBackwardInput:
            RSC_if_fwd_else_C = f"layout.strides[{last_dim}]"
            C_if_fwd_else_KRSC = f"layout.strides[0] * problem.K"
            # for weight opt params, the C is actually K, so we
            # need to multplie stride.
            mul_stride_if_bwd = " * layout.strides[0]"
        kernel_vol = "problem.kernel_volume" if self.problem.mask_sparse else "problem.ksize.op<tv::arrayops::prod>()"
        if self.increment_k_first:
            code.raw(f"""
            // int kernel_prod = {kernel_vol};
            filter_c_delta = {self.tile_shape[2]} * problem.split_k_slices;
            """)
        else:
            code.raw(f"""
            kernel_prod = {kernel_vol};
            filter_c_delta = {self.tile_shape[2]} * problem.split_k_slices;
            """)
        code.raw(f"""
        inc_strided = int64_t(layout.strides[0]) * {self.tmap.delta[0]};

        stride_rsc_bytes = layout.strides[0] * {self.dtype.bitsize()} / 8;
        """)
        if not self.increment_k_first:
            code.raw(f"""
            // back to strided start, then inc rs
            inc_rs = int64_t(layout.strides[{last_dim}])  - inc_strided * int64_t({self.tmap.iterations[0] - 1});
            inc_c = filter_c_delta{mul_stride_if_bwd};
            // back to rs start
            inc_c -= int64_t(kernel_prod - 1) * layout.strides[{last_dim}];
            // and strided start
            inc_c -= inc_strided * int64_t({self.tmap.iterations[0] - 1});
            """)
        else:
            code.raw(f"""
            // back to strided start, then inc c
            inc_c = filter_c_delta{mul_stride_if_bwd} - inc_strided * int64_t({self.tmap.iterations[0] - 1});
            inc_rs = int64_t(layout.strides[{last_dim}]);
            // inc_c_reset = -gemm_iters_k * filter_c_delta{mul_stride_if_bwd} * {self.dtype.bitsize()} / 8;
            """)
        code.raw(f"""
        inc_rs = inc_rs * {self.dtype.bitsize()} / 8;
        inc_strided = inc_strided * {self.dtype.bitsize()} / 8;
        inc_c = inc_c * {self.dtype.bitsize()} / 8;
        """)
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def set_inc_reset_for_inc_k_first(self):
        code = FunctionCode()
        code.arg("gemm_iters_k", "int", "-1")
        if not self.increment_k_first:
            return code
        mul_stride_if_bwd = ""
        if self.op_type == ConvOpType.kBackwardInput:
            mul_stride_if_bwd = " * layout.strides[0]"
        code.raw(f"""
        inc_c_reset = -gemm_iters_k * filter_c_delta{mul_stride_if_bwd} * {self.dtype.bitsize()} / 8;
        """)
        return code

    def python_ctor(self, problem: params.ConvProblem, layout: LAYOUT_TYPES):
        new_obj = WeightOptParams(self.dtype, self.tile_shape, self.problem,
                                  self.layout, self.tmap)
        new_obj.layout_ = layout

        new_obj.inc_strided_ = layout.strides[0] * self.tmap.delta[0]
        new_obj.inc_rs_ = layout.strides[
            self.ndim] - new_obj.inc_strided_ * (self.tmap.iterations[0] - 1)
        new_obj.filter_c_delta_ = self.tile_shape[2] * problem.split_k_slices_
        new_obj.inc_c_ = new_obj.filter_c_delta_

        new_obj.inc_c_ -= (int(np.prod(problem.ksize_)) -
                           1) * layout.strides[self.ndim]
        new_obj.inc_c_ -= new_obj.inc_strided_ * (self.tmap.iterations[0] - 1)

        new_obj.inc_c_ = new_obj.inc_c_ * self.dtype.bitsize() // 8
        new_obj.inc_rs_ = new_obj.inc_rs_ * self.dtype.bitsize() // 8
        new_obj.inc_strided_ = new_obj.inc_strided_ * self.dtype.bitsize() // 8
        return new_obj


class InputNPQIterator(bases.ConvInputIterator):
    """Why input (reduce NPQ) don't use mask?
    because iteration is done in NPQ axis,
    when k increments, the mn-mask is unstable because of NHW->NPQ conversion.
    we can't reuse pre computed mask in every k-iterations.

    for output-KRS/weight-KRS, reduction is done in KRS axis,
    the NPQ layout is stable, in every k iteration, we see same 
    NPQ->NHW layout.
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 optimized: bool = False,
                 transpose_load: bool = False):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype, element_count, sub_tile_shape[1])
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.optimized = optimized
        self.ndim = problem_size.ndim
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        self.params = AnalyticParams(problem_size,
                                     input_layout,
                                     is_output=True,
                                     has_rsc=True)
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")

        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)

        self.add_member("filter_rsc_offset_",
                        "int",
                        array=f"[{self.tmap.iterations[1]}][{self.ndim + 1}]")
        # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")
        self.add_member("npq_offset_", "int", array=f"[{tmap.iterations[0]}]")

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")

        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        with code.range_("c", str(self.tmap.iterations[1]),
                         "TV_PRAGMA_UNROLL"):
            code.raw(f"""
            int rsc_offset = thread_offset[1] + c * {self.tmap.delta[1]};
            params.layout_rsc.inverse(rsc_offset, filter_rsc_offset_[c]);
            """)
            if self.optimized:
                code.raw(f"auto& rsc_offset_per_c = filter_rsc_offset_[c];")
                for i in range(self.ndim):
                    code.raw(f"""
                    int r_{i} = rsc_offset_per_c[{i}];
                    if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                        r_{i} = (problem_size_.ksize[{i}] - 1 - rsc_offset_per_c[{i}]);
                    }}
                    rsc_offset_per_c[{i}] = - problem_size_.padding[{i}] + r_{i} * problem_size_.dilation[{i}];
                    """)
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.tmap.iterations[0]}; ++s){{
            npq_offset_[s] = thread_offset[0] + s * {self.tmap.delta[0]};
        }}
        """)
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        with self.tmap.tmap_loop(code, "s"):
            code.raw(f"""
            npq_offset_[s] += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def npqrs_to_nhwc(self):
        code = FunctionCode()
        code.arg("npq", f"const tv::array<int, {self.ndim + 1}>&")
        code.arg("rsc", f"const int*")
        for i in range(self.ndim):
            if self.optimized:
                code.raw(f"""
                int h_{i} = npq[{i + 1}] * problem_size_.stride[{i}] + rsc[{i}];
                """)
            else:
                code.raw(f"""
                int r_{i} = rsc[{i}];
                if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                    r_{i} = (problem_size_.ksize[{i}] - 1 - rsc[{i}]);
                }}
                int h_{i} = npq[{i + 1}] * problem_size_.stride[{i}] - problem_size_.padding[{i}] + r_{i} * problem_size_.dilation[{i}];
                """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{npq[0], {h0h1h2}, rsc[{self.ndim}]}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 2}>")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def at(self):
        code = FunctionCode()
        code.arg("stride, contig", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")

        code.raw(f"""
        auto npq = params_.layout_npq.inverse(npq_offset_[stride]);
        return npqrs_to_nhwc(npq, filter_rsc_offset_[contig]);
        """)

        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = FunctionCode()
        code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
        hw_valid = [
            f"indexes[{i + 1}] >= 0 && indexes[{i + 1}] < problem_size_.input_dims[{i}]"
            for i in range(self.ndim)
        ]
        code.raw(f"""
        return indexes[0] < problem_size_.N && 
            {' && '.join(hw_valid)} &&
            indexes[{self.ndim + 1}] < problem_size_.C;
        """)
        return code.ret(f"bool")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
        code.raw(f"""
        auto offset = params_.layout(indexes);
        return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s", "c"):
            code.raw(f"""
            int idx = s * {self.tmap.iterations[1]} + c;
            auto indexes = at(s, c);
            {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
            tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                frag_ptr[idx], access_ptr, valid(indexes));
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


class OutputNPQParams(bases.ConvIterParams):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 problem: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked]):
        super().__init__()
        self.add_dependency(TensorViewMath)
        if not isinstance(input_layout, layout.TensorGeneric):
            raise NotImplementedError
        self.dtype = dtype
        self.tmap = tmap
        self.tile_shape = tile_shape
        self.problem = problem
        self.ndim = problem.ndim
        self.add_param_class("input_layout", input_layout, "Layout")
        self.add_param_class("prob", problem, "ConvProblem")
        self.add_member("layout", "Layout")
        self.add_member("inc_strided, inc_contig", "int64_t")
        self.add_member("inc_k", "int64_t")
        self.add_member("NPQ", "int")

        self.layout = input_layout

    @pccm.cuda.constructor(device=True, host=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("problem", "ConvProblem const&")
        code.arg("layout", "Layout const&")
        code.ctor_init("layout", "layout")

        pqs_prod = codeops.unpack("problem.output_dims", range(self.ndim),
                                  " * ")
        code.ctor_init("NPQ", f"{pqs_prod} * problem.N")
        code.raw(f"""
        // NPQK NPQ is strided, K is contig
        inc_strided = {self.tmap.delta[0]} * layout.strides[{self.ndim}] * {self.dtype.bitsize()} / 8;
        inc_contig = {self.tmap.delta[1]} * {self.dtype.bitsize()} / 8;
        inc_k = {self.tile_shape[2]} * problem.split_k_slices * layout.strides[{self.ndim}] * {self.dtype.bitsize()} / 8;
        """)
        return code


class OutputNPQIterator(bases.ConvInputIterator):
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 optimized: bool = False,
                 transpose_load: bool = False):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype, element_count, sub_tile_shape[1])
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.optimized = optimized
        self.ndim = problem_size.ndim
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        if optimized:
            self.params = OutputNPQParams(dtype, tile_shape_mnk, problem_size,
                                          input_layout, tmap)
        else:
            self.params = AnalyticParams(problem_size,
                                         input_layout,
                                         is_output=True,
                                         has_rsc=False)
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")
        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)
        if not optimized:
            self.add_member("k_offset_",
                            "int",
                            array=f"[{self.tmap.iterations[1]}]")
            # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")
            self.add_member("npq_offset_",
                            "int",
                            array=f"[{tmap.iterations[0]}]")
        else:
            self.add_member("k_offset_", "int")
            # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")
            self.add_member("npq_offset_", "int")

            self.add_member("mask_", "int")

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")

        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        if self.optimized:
            code.raw(f"""
            mask_ = 0;
            k_offset_ = thread_offset[1];
            npq_offset_ = thread_offset[0];
            """)
            with self.tmap.tmap_loop(code, "s", "c"):
                code.raw(f"""
                int k_offset_iter = k_offset_ + c * {self.tmap.delta[1]};
                int npq_offset_iter = npq_offset_ + s * {self.tmap.delta[0]};
                int pred = npq_offset_iter < params.NPQ && k_offset_iter < problem_size.K;
                mask_ |= (pred << (s * {self.tmap.iterations[1]} + c));
                """)
            code.raw(f"""
            pointer_ += (npq_offset_ * problem_size.K + k_offset_) * {self.dtype.bitsize()} / 8;
            """)
        else:
            with code.range_("c", str(self.tmap.iterations[1]),
                             "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                k_offset_[c] = thread_offset[1] + c * {self.tmap.delta[1]};
                """)
            with self.tmap.tmap_loop(code, "s"):
                code.raw(
                    f"npq_offset_[s] = thread_offset[0] + s * {self.tmap.delta[0]};"
                )
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        if self.optimized:
            code.raw(f"""
            npq_offset_ +=  {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
            """)
            with self.tmap.tmap_loop(code, "s"):
                code.raw(f"""
                if (npq_offset_ + s * {self.tmap.delta[0]} >= params_.NPQ){{
                    uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1]});
                    mask_ = mask_ & (~mask);
                }}
                """)
            code.raw(f"pointer_ += params_.inc_k;")
        else:
            with self.tmap.tmap_loop(code, "s"):
                code.raw(f"""
                npq_offset_[s] += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
                """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = FunctionCode()
        if self.optimized:
            code.arg("stride, contig", f"int")
            code.raw(f"""
            return mask_ & (1u << (stride * {self.tmap.iterations[1]} + contig));
            """)
        else:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            hw_valid = [
                f"indexes[{i + 1}] >= 0 && indexes[{i + 1}] < problem_size_.output_dims[{i}]"
                for i in range(self.ndim)
            ]
            code.raw(f"""
            return indexes[0] < problem_size_.N && 
                {' && '.join(hw_valid)} &&
                indexes[{self.ndim + 1}] < problem_size_.K;
            """)
        return code.ret(f"bool")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        if self.optimized:
            code.arg("stride, contig", f"int")
            code.raw(f"""
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + 
                stride * params_.inc_strided + contig * params_.inc_contig);
            """)

        else:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            auto offset = params_.layout(indexes);
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def at(self):
        code = FunctionCode()
        if self.optimized:
            return code
        code.arg("stride, contig", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")
        npqs = codeops.unpack("npq", range(self.ndim + 1))
        code.raw(f"""
        auto npq = params_.layout_npq.inverse(npq_offset_[stride]);
        return {{{npqs}, k_offset_[contig]}};
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s", "c"):
            if self.optimized:
                code.raw(f"""
                int idx = s * {self.tmap.iterations[1]} + c;
                {self.access_t} const *access_ptr = get(s, c) + pointer_offset / {self.element_per_acc};
                tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                    frag_ptr[idx], access_ptr, valid(s, c));
                """)
            else:
                code.raw(f"""
                int idx = s * {self.tmap.iterations[1]} + c;
                auto indexes = at(s, c);
                {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                    frag_ptr[idx], access_ptr, valid(indexes));
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


class WeightIteratorDP4A(bases.ConvInputIterator):
    """
    fwd: NHWC -> NPQRSC @ KRSC, k = RSC
    dgrad: NPQK -> NHWRSK @ KRSC -> RSKC, k = RSK
    wgrad: NPQK @ NHWC -> NPQRSC, k = NPQ

    for weight, RSC or KRS

    TODO for dp4a, each dim of kernel must less than 8
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: layout.TensorGeneric,
                 optimized: bool = False,
                 transpose_load: bool = False,
                 increment_k_first: bool = False,
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
        self.optimized = optimized
        self.ndim = problem_size.ndim
        self.increment_k_first = increment_k_first
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        self.mask_sparse = problem_size.mask_sparse
        if op_type == ConvOpType.kForward:
            assert tmap.iterations[1] == 1
        self.gload = GlobalLoad(self.element_per_acc * self.dtype.itemsize(),
                                level="L2",
                                prefetch_size=128)
        self.add_param_class("gload", self.gload, "GlobalLoad")

        assert tmap.iterations.prod() * sub_tile_shape[0] < 32, "error"
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        if not optimized:
            self.params = AnalyticParams(problem_size, input_layout)
        else:
            self.params = WeightOptParams(dtype, tile_shape_mnk, problem_size,
                                          input_layout, tmap,
                                          increment_k_first)
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")

        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)
        self.reduce_channel_axis = 0 if self.op_type == ConvOpType.kBackwardInput else 1
        self.noreduce_channel_axis = 1 if self.op_type == ConvOpType.kBackwardInput else 0
        if problem_size.mask_sparse:
            assert optimized
        if optimized:
            if not self.increment_k_first:
                self.add_member("filter_kernel_idx_", f"int")
            self.add_member("reduce_channel_offset_", "int")
            if self.increment_k_first and self.problem_size.mask_sparse:
                self.add_member("reduce_channel_offset_backup_", "int")
                self.add_member(
                    "mask_backup_",
                    f"tv::array<uint32_t, {self.access_per_vector}>")
        else:
            self.add_member(
                "reduce_channel_offsets_",
                "int",
                array=f"[{tmap.iterations[self.reduce_channel_axis]}]")
            self.add_member(
                "noreduce_channel_offsets_",
                "int",
                array=f"[{tmap.iterations[self.noreduce_channel_axis]}]")
            self.add_member("filter_kernel_idxes_",
                            f"tv::array<int, {self.ndim}>")
        if optimized:
            self.add_member("mask_",
                            f"tv::array<uint32_t, {self.access_per_vector}>")
        # if problem_size.mask_sparse:

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")
        if self.optimized:
            if not self.increment_k_first:
                code.ctor_init("filter_kernel_idx_", "0")
        else:
            code.ctor_init("filter_kernel_idxes_",
                           f"{{{', '.join(['0'] * self.ndim)}}}")

        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        if self.optimized:
            code.raw(f"""
            mask_.clear();
            reduce_channel_offset_ = thread_offset[{self.reduce_channel_axis}];
            """)
            if self.increment_k_first and self.problem_size.mask_sparse:
                code.raw(f"""
                reduce_channel_offset_backup_ = thread_offset[{self.reduce_channel_axis}];
                """)
        else:
            with code.range_(
                    "i", str(self.tmap.iterations[self.reduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):

                code.raw(f"""
                reduce_channel_offsets_[i] = thread_offset[{self.reduce_channel_axis}] +
                    i * {self.tmap.delta[self.reduce_channel_axis]};
                """)
            with code.range_(
                    "i", str(self.tmap.iterations[self.noreduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):

                code.raw(f"""
                noreduce_channel_offsets_[i] = thread_offset[{self.noreduce_channel_axis}] +
                    i * {self.tmap.delta[self.noreduce_channel_axis]};
                """)

        if self.optimized:
            with self.tmap.tmap_loop(code, "s", "c"):
                with code.range_("ss", self.tmap.sub_tile_shape[0],
                                 "TV_PRAGMA_UNROLL"):
                    if self.tmap.iterations[1] == 1:
                        code.raw(f"""
                        uint32_t pred = thread_offset[0] + s * {self.tmap.delta[0]} + ss < problem_size.K;
                        """)
                        with code.range_("v", self.access_per_vector,
                                         "TV_PRAGMA_UNROLL"):

                            code.raw(f"""
                            mask_[v] |= (pred << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.tmap.sub_tile_shape[0]} + ss));
                            """)

                    else:
                        with code.range_("v", self.access_per_vector,
                                         "TV_PRAGMA_UNROLL"):
                            code.raw(f"""
                            uint32_t pred = (thread_offset[0] + s * {self.tmap.delta[0]} + ss < problem_size.K)
                                && (thread_offset[1] + c * {self.tmap.delta[1]} + v * {self.element_per_acc} < problem_size.C);
                            mask_[v] |= (pred << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.tmap.sub_tile_shape[0]} + ss));
                            """)

            if self.tmap.iterations[1] == 1:
                # if no iterations in C, we can just reset mask if c out of range.
                code.raw(f"""
                TV_PRAGMA_UNROLL
                for (int v = 0; v < {self.access_per_vector}; ++v){{
                    mask_[v] = thread_offset[1] + v * {self.element_per_acc} >= problem_size.C ? 0 : mask_[v];
                }}
                """)
            code.raw(
                f"pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * {self.dtype.bitsize()} / 8;"
            )
            if self.increment_k_first and self.mask_sparse:
                code.raw(f"""
                mask_backup_ = mask_;
                """)
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        if self.increment_k_first:
            return code
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if not self.optimized:
            for i in range(self.ndim - 1, -1, -1):
                code.raw(f"""
                if (++filter_kernel_idxes_[{i}] < problem_size_.ksize[{i}]){{
                    return;
                }}
                filter_kernel_idxes_[{i}] = 0;
                """)
            with code.range_(
                    "i", str(self.tmap.iterations[self.reduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                reduce_channel_offsets_[i] += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
                """)
        else:
            code.raw(f"""
            if (++filter_kernel_idx_ == params_.kernel_prod){{
                filter_kernel_idx_ = 0;
                // back to first c
                pointer_ += params_.inc_c;
                reduce_channel_offset_ += params_.filter_c_delta;
            }}else{{
                // back to first rs
                pointer_ += params_.inc_rs;
            }}
            """)
            if self.op_type == ConvOpType.kForward:
                # if self.tmap.iterations[1] == 1:
                # we assume tmap.iterations[1] always 1
                # so just set mask to zero if k out of range.
                code.raw(f"""
                TV_PRAGMA_UNROLL
                for (int v = 0; v < {self.access_per_vector}; ++v){{
                    mask_[v] = reduce_channel_offset_ + v * {self.element_per_acc} >= problem_size_.{C_or_K} ? 0 : mask_[v];
                }}
                """)
            else:
                # reduce_channel_offset_ is K, is stride
                with self.tmap.tmap_loop(code, "s"):
                    with code.range_("ss", self.tmap.sub_tile_shape[0],
                                     "TV_PRAGMA_UNROLL"):
                        code.raw(f"""
                        if (reduce_channel_offset_ + s * {self.tmap.delta[0]} + ss >= problem_size_.K){{
                            uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + ss);
                            TV_PRAGMA_UNROLL
                            for (int v = 0; v < {self.access_per_vector}; ++v){{
                                mask_[v] = mask_[v] & (~mask);
                            }}
                            // mask_ = mask_ & (~mask);
                        }}
                        """)

        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_k(self):
        code = FunctionCode()
        if not self.increment_k_first:
            return code
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        code.raw(f"""
        pointer_ += params_.inc_c;
        reduce_channel_offset_ += params_.filter_c_delta;
        """)
        if self.op_type == ConvOpType.kForward:
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int v = 0; v < {self.access_per_vector}; ++v){{
                clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_size_.{C_or_K}, v);
            }}
            """)
        else:
            with self.tmap.tmap_loop(code, "s"):
                with code.range_("ss", self.tmap.sub_tile_shape[0],
                                 "TV_PRAGMA_UNROLL"):
                    code.raw(f"""
                    if (reduce_channel_offset_ + s * {self.tmap.delta[0]} + ss >= problem_size_.K){{
                        uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + ss);
                        TV_PRAGMA_UNROLL
                        for (int v = 0; v < {self.access_per_vector}; ++v){{
                            mask_[v] = mask_[v] & (~mask);
                        }}
                    }}
                    """)

        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_filter(self):
        code = FunctionCode()
        if not self.increment_k_first:
            return code
        code.raw(f"""
        pointer_ += params_.inc_rs;
        """)
        return code

    @pccm.cuda.member_function(name="increment_filter",
                               device=True,
                               forceinline=True)
    def increment_filter_with_num(self):
        code = FunctionCode()
        code.arg("num", "int")
        if not self.increment_k_first:
            return code
        code.raw(f"""
        pointer_ += params_.inc_rs * num;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def reset_k(self):
        code = FunctionCode()
        if not self.increment_k_first:
            return code
        code.raw(f"""
        pointer_ += params_.inc_c_reset;
        reduce_channel_offset_ = reduce_channel_offset_backup_;
        mask_ = mask_backup_;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_if_not_pred(self):
        code = pccm.cuda.PTXCode()
        if not self.optimized:
            return code
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
        if not self.optimized:
            return code

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

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_byte_offset(self):
        code = FunctionCode()
        code.arg("byte_offset", str(self.long_index_t))
        code.raw(f"""
        pointer_ += byte_offset;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def at(self):
        code = FunctionCode()
        if self.optimized:
            return code
        code.arg("stride, contig, ss", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")
        rs = codeops.unpack("filter_kernel_idxes_", range(self.ndim))
        if self.op_type == ConvOpType.kForward:
            code.raw(f"""
            return {{noreduce_channel_offsets_[stride] + ss, {rs}, reduce_channel_offsets_[contig]}};
            """)
        elif self.op_type == ConvOpType.kBackwardInput:
            code.raw(f"""
            return {{reduce_channel_offsets_[stride], {rs}, noreduce_channel_offsets_[contig]}};
            """)
        else:
            raise NotImplementedError
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = FunctionCode()
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            return indexes[0] < problem_size_.K && 
                indexes[{self.ndim + 1}] < problem_size_.C;
            """)
        else:
            code.arg("s, c, ss, v", f"int")
            code.raw(f"""
            return mask_[v] & (1u << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss));
            """)
        return code.ret(f"bool")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            auto offset = params_.layout(indexes);
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
            """)
        else:
            code.arg("stride, contig, ss", "int")
            if self.sub_tile_shape[0] == 1:
                code.raw(
                    f"return reinterpret_cast<{self.const_access_pointer}>(pointer_ + contig * {self.tmap.delta[1]} * {self.dtype.bitsize()} / 8);"
                )
            else:
                code.raw(f"""
                    return reinterpret_cast<{self.const_access_pointer}>(pointer_ + 
                        contig * {self.tmap.delta[1] * self.dtype.bitsize()} / 8 + ss * params_.stride_rsc_bytes);
                """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s"):
            with code.range_("c", str(self.tmap.iterations[1]),
                             "TV_PRAGMA_UNROLL"):
                with code.range_("ss", str(self.sub_tile_shape[0]),
                                 "TV_PRAGMA_UNROLL"):
                    with code.range_("v", self.access_per_vector,
                                     "TV_PRAGMA_UNROLL"):
                        if not self.optimized:
                            assert self.access_per_vector == 1
                            code.raw(f"""
                            int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss; 
                            auto indexes = at(s, c, ss);
                            {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                            GlobalLoad::run(frag_ptr[idx], access_ptr, valid(indexes));

                            // tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            //     frag_ptr[idx], access_ptr, valid(indexes));
                            """)
                        else:
                            code.raw(f"""
                            int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0] * self.access_per_vector} + 
                                c * {self.sub_tile_shape[0] * self.access_per_vector} + ss * {self.access_per_vector} + v;
                            {self.access_t} const *access_ptr = get(s, c, ss) + v + pointer_offset / {self.element_per_acc};
                            // tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            //     frag_ptr[idx], access_ptr, valid(s, c, ss));
                            GlobalLoad::run(frag_ptr[idx], access_ptr, valid(s, c, ss, v));

                            """)
            if self.optimized:
                # weight only use one ptr, so we need to increment
                # in every stride iteration
                code.raw(f"""
                if (s != {self.tmap.iterations[0] - 1}){{
                    pointer_ += params_.inc_strided;
                }}
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
        code.ret(f"const {self.access_t}*")
        if not self.optimized:
            assert self.access_per_vector == 1
            code.raw(f"""
                auto indexes = at(s, c, 0);
                {self.access_t} const *access_ptr = get(indexes);
                valid_ref = valid(indexes);
            """)
        else:
            code.raw(f"""
                {self.access_t} const *access_ptr = get(s, c, 0) + 0;
                valid_ref = valid(s, c, 0, 0);
                if (c == {self.tmap.iterations[1] - 1} && s != {self.tmap.iterations[0] - 1})
                    pointer_ += params_.inc_strided;
            """)
        return code.raw("return access_ptr;")

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_invalid(self):
        code = pccm.code()
        if not self.optimized:
            return code
        with self.tmap.tmap_loop(code, "s"):
            code.raw(f"""
            if (s != {self.tmap.iterations[0] - 1}){{
                pointer_ += params_.inc_strided;
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask(self):
        code = pccm.code()
        return code


class WeightIteratorDP4AV2Mask(bases.ConvInputIterator):
    """
    fwd: NHWC -> NPQRSC @ KRSC, k = RSC
    dgrad: NPQK -> NHWRSK @ KRSC -> RSKC, k = RSK
    wgrad: NPQK @ NHWC -> NPQRSC, k = NPQ

    for weight, RSC or KRS

    TODO for dp4a, each dim of kernel must less than 8
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: layout.TensorGeneric,
                 optimized: bool = False,
                 transpose_load: bool = False,
                 increment_k_first: bool = False,
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
        self.optimized = optimized
        self.ndim = problem_size.ndim
        self.increment_k_first = increment_k_first
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        self.mask_sparse = problem_size.mask_sparse
        if op_type == ConvOpType.kForward:
            assert tmap.iterations[1] == 1
        self.gload = GlobalLoad(self.element_per_acc * self.dtype.itemsize(),
                                level="L2",
                                prefetch_size=128)
        self.add_param_class("gload", self.gload, "GlobalLoad")

        assert tmap.iterations.prod() * sub_tile_shape[0] < 32, "error"
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        if not optimized:
            self.params = AnalyticParams(problem_size, input_layout)
        else:
            self.params = WeightOptParams(dtype, tile_shape_mnk, problem_size,
                                          input_layout, tmap,
                                          increment_k_first)
        self.tmap = tmap
        self.mask_cls = Mask(
            seq(self.tmap.iterations[0], self.tmap.iterations[1],
                self.sub_tile_shape[0], self.access_per_vector))
        self.add_param_class("mask", self.mask_cls, "Mask")

        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")

        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)
        self.reduce_channel_axis = 0 if self.op_type == ConvOpType.kBackwardInput else 1
        self.noreduce_channel_axis = 1 if self.op_type == ConvOpType.kBackwardInput else 0
        if problem_size.mask_sparse:
            assert optimized
        if optimized:
            if not self.increment_k_first:
                self.add_member("filter_kernel_idx_", f"int")
            self.add_member("reduce_channel_offset_", "int")
            if self.increment_k_first and self.problem_size.mask_sparse:
                self.add_member("reduce_channel_offset_backup_", "int")
                self.add_member("mask_backup_", f"Mask")
        else:
            self.add_member(
                "reduce_channel_offsets_",
                "int",
                array=f"[{tmap.iterations[self.reduce_channel_axis]}]")
            self.add_member(
                "noreduce_channel_offsets_",
                "int",
                array=f"[{tmap.iterations[self.noreduce_channel_axis]}]")
            self.add_member("filter_kernel_idxes_",
                            f"tv::array<int, {self.ndim}>")
        if optimized:
            self.add_member("mask_", f"Mask")
        # if problem_size.mask_sparse:

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")
        if self.optimized:
            if not self.increment_k_first:
                code.ctor_init("filter_kernel_idx_", "0")
        else:
            code.ctor_init("filter_kernel_idxes_",
                           f"{{{', '.join(['0'] * self.ndim)}}}")

        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        if self.optimized:
            code.raw(f"""
            mask_.clear();
            reduce_channel_offset_ = thread_offset[{self.reduce_channel_axis}];
            """)
            if self.increment_k_first and self.problem_size.mask_sparse:
                code.raw(f"""
                reduce_channel_offset_backup_ = thread_offset[{self.reduce_channel_axis}];
                """)
        else:
            with code.range_(
                    "i", str(self.tmap.iterations[self.reduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):

                code.raw(f"""
                reduce_channel_offsets_[i] = thread_offset[{self.reduce_channel_axis}] +
                    i * {self.tmap.delta[self.reduce_channel_axis]};
                """)
            with code.range_(
                    "i", str(self.tmap.iterations[self.noreduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):

                code.raw(f"""
                noreduce_channel_offsets_[i] = thread_offset[{self.noreduce_channel_axis}] +
                    i * {self.tmap.delta[self.noreduce_channel_axis]};
                """)
        if self.optimized:
            for s in range(self.tmap.iterations[0]):
                for c in range(self.tmap.iterations[1]):
                    for ss in range(self.tmap.sub_tile_shape[0]):
                        for v in range(self.access_per_vector):
                            pred = (
                                f"(thread_offset[0] + {s * self.tmap.delta[0] + ss} < problem_size.K) && (thread_offset[1]"
                                f" + {c * self.tmap.delta[1] + v * self.element_per_acc} < problem_size.C)"
                            )
                            code.raw(
                                f"mask_.set_coord({pred}, {s}, {c}, {ss}, {v});"
                            )
            code.raw(
                f"pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * {self.dtype.bitsize()} / 8;"
            )
            if self.increment_k_first and self.mask_sparse:
                code.raw(f"""
                mask_backup_ = mask_;
                """)
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        if self.increment_k_first:
            return code
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if not self.optimized:
            for i in range(self.ndim - 1, -1, -1):
                code.raw(f"""
                if (++filter_kernel_idxes_[{i}] < problem_size_.ksize[{i}]){{
                    return;
                }}
                filter_kernel_idxes_[{i}] = 0;
                """)
            with code.range_(
                    "i", str(self.tmap.iterations[self.reduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                reduce_channel_offsets_[i] += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
                """)
        else:
            code.raw(f"""
            if (++filter_kernel_idx_ == params_.kernel_prod){{
                filter_kernel_idx_ = 0;
                // back to first c
                pointer_ += params_.inc_c;
                reduce_channel_offset_ += params_.filter_c_delta;
            }}else{{
                // back to first rs
                pointer_ += params_.inc_rs;
            }}
            """)
            if self.op_type == ConvOpType.kForward:
                # if self.tmap.iterations[1] == 1:
                # we assume tmap.iterations[1] always 1
                # so just set mask to zero if k out of range.
                for v in range(self.access_per_vector):
                    pred = f"reduce_channel_offset_ + {v * self.element_per_acc} >= problem_size_.{C_or_K}"
                    self.mask_cls.clear_mask_if_pred_template(
                        code, pred, (None, None, None, v))

                # code.raw(f"""
                # TV_PRAGMA_UNROLL
                # for (int v = 0; v < {self.access_per_vector}; ++v){{
                #     mask_[v] = reduce_channel_offset_ + v * {self.element_per_acc} >= problem_size.{C_or_K} ? 0 : mask_[v];
                # }}
                # """)
            else:
                # reduce_channel_offset_ is K, is stride
                for s in range(self.tmap.iterations[0]):
                    for ss in range(self.tmap.sub_tile_shape[0]):
                        pred = f"reduce_channel_offset_ + {s * self.tmap.delta[0] + ss} >= problem_size_.K"
                        self.mask_cls.clear_mask_if_pred_template(
                            code, pred, (s, None, ss, None))

                # with self.tmap.tmap_loop(code, "s"):
                #     with code.range_("ss", self.tmap.sub_tile_shape[0],
                #                      "TV_PRAGMA_UNROLL"):
                #         code.raw(f"""
                #         if (reduce_channel_offset_ + s * {self.tmap.delta[0]} + ss >= problem_size_.K){{
                #             uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + ss);
                #             TV_PRAGMA_UNROLL
                #             for (int v = 0; v < {self.access_per_vector}; ++v){{
                #                 mask_[v] = mask_[v] & (~mask);
                #             }}
                #             // mask_ = mask_ & (~mask);
                #         }}
                #         """)

        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_k(self):
        code = FunctionCode()
        if not self.increment_k_first:
            return code
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        code.raw(f"""
        pointer_ += params_.inc_c;
        reduce_channel_offset_ += params_.filter_c_delta;
        """)
        if self.op_type == ConvOpType.kForward:
            for v in range(self.access_per_vector):
                pred = f"reduce_channel_offset_ + {v * self.element_per_acc} >= problem_size_.{C_or_K}"
                self.mask_cls.clear_mask_if_pred_template(
                    code, pred, (None, None, None, v))

            # code.raw(f"""
            # TV_PRAGMA_UNROLL
            # for (int v = 0; v < {self.access_per_vector}; ++v){{
            #     clear_mask_if_pred(reduce_channel_offset_ + v * {self.element_per_acc} >= problem_size_.{C_or_K}, v);
            # }}
            # """)
        else:
            for s in range(self.tmap.iterations[0]):
                for ss in range(self.tmap.sub_tile_shape[0]):
                    pred = f"reduce_channel_offset_ + {s * self.tmap.delta[0] + ss} >= problem_size_.K"
                    self.mask_cls.clear_mask_if_pred_template(
                        code, pred, (s, None, ss, None))

            # with self.tmap.tmap_loop(code, "s"):
            #     with code.range_("ss", self.tmap.sub_tile_shape[0],
            #                         "TV_PRAGMA_UNROLL"):
            #         code.raw(f"""
            #         if (reduce_channel_offset_ + s * {self.tmap.delta[0]} + ss >= problem_size_.K){{
            #             uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + ss);
            #             TV_PRAGMA_UNROLL
            #             for (int v = 0; v < {self.access_per_vector}; ++v){{
            #                 mask_[v] = mask_[v] & (~mask);
            #             }}
            #         }}
            #         """)

        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_filter(self):
        code = FunctionCode()
        if not self.increment_k_first:
            return code
        code.raw(f"""
        pointer_ += params_.inc_rs;
        """)
        return code

    @pccm.cuda.member_function(name="increment_filter",
                               device=True,
                               forceinline=True)
    def increment_filter_with_num(self):
        code = FunctionCode()
        code.arg("num", "int")
        if not self.increment_k_first:
            return code
        code.raw(f"""
        pointer_ += params_.inc_rs * num;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def reset_k(self):
        code = FunctionCode()
        if not self.increment_k_first:
            return code
        code.raw(f"""
        pointer_ += params_.inc_c_reset;
        reduce_channel_offset_ = reduce_channel_offset_backup_;
        mask_ = mask_backup_;
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

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_byte_offset(self):
        code = FunctionCode()
        code.arg("byte_offset", str(self.long_index_t))
        code.raw(f"""
        pointer_ += byte_offset;
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def at(self):
        code = FunctionCode()
        if self.optimized:
            return code
        code.arg("stride, contig, ss", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")
        rs = codeops.unpack("filter_kernel_idxes_", range(self.ndim))
        if self.op_type == ConvOpType.kForward:
            code.raw(f"""
            return {{noreduce_channel_offsets_[stride] + ss, {rs}, reduce_channel_offsets_[contig]}};
            """)
        elif self.op_type == ConvOpType.kBackwardInput:
            code.raw(f"""
            return {{reduce_channel_offsets_[stride], {rs}, noreduce_channel_offsets_[contig]}};
            """)
        else:
            raise NotImplementedError
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = FunctionCode()
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            return indexes[0] < problem_size_.K && 
                indexes[{self.ndim + 1}] < problem_size_.C;
            """)
        else:
            code.arg("s, c, ss, v", f"int")
            code.raw(f"""
            return mask_.query_coord(s, c, ss, v);
            // return mask_[v] & (1u << (s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss));
            """)
        return code.ret(f"bool")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            auto offset = params_.layout(indexes);
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
            """)
        else:
            code.arg("stride, contig, ss", "int")
            if self.sub_tile_shape[0] == 1:
                code.raw(
                    f"return reinterpret_cast<{self.const_access_pointer}>(pointer_ + contig * {self.tmap.delta[1]} * {self.dtype.bitsize()} / 8);"
                )
            else:
                code.raw(f"""
                    return reinterpret_cast<{self.const_access_pointer}>(pointer_ + 
                        contig * {self.tmap.delta[1] * self.dtype.bitsize()} / 8 + ss * params_.stride_rsc_bytes);
                """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s"):
            with code.range_("c", str(self.tmap.iterations[1]),
                             "TV_PRAGMA_UNROLL"):
                with code.range_("ss", str(self.sub_tile_shape[0]),
                                 "TV_PRAGMA_UNROLL"):
                    if not self.optimized:
                        code.raw(f"""
                        auto indexes = at(s, c, ss);
                        {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                        """)
                    else:
                        code.raw(f"""
                        {self.access_t} const *access_ptr = get(s, c, ss) + pointer_offset / {self.element_per_acc};
                        """)
                    with code.range_("v", self.access_per_vector,
                                     "TV_PRAGMA_UNROLL"):
                        if not self.optimized:
                            assert self.access_per_vector == 1
                            code.raw(f"""
                            int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss; 
                            auto indexes = at(s, c, ss);
                            // {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                            GlobalLoad::run(frag_ptr[idx], access_ptr, valid(indexes));

                            // tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            //     frag_ptr[idx], access_ptr, valid(indexes));
                            """)
                        else:
                            code.raw(f"""
                            int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0] * self.access_per_vector} + 
                                c * {self.sub_tile_shape[0] * self.access_per_vector} + ss * {self.access_per_vector} + v;
                            // {self.access_t} const *access_ptr = get(s, c, ss) + v + pointer_offset / {self.element_per_acc};
                            // tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            //     frag_ptr[idx], access_ptr, valid(s, c, ss));
                            GlobalLoad::run(frag_ptr[idx], access_ptr + v, valid(s, c, ss, v));

                            """)
            if self.optimized:
                # weight only use one ptr, so we need to increment
                # in every stride iteration
                code.raw(f"""
                if (s != {self.tmap.iterations[0] - 1}){{
                    pointer_ += params_.inc_strided;
                }}
                """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_invalid(self):
        code = pccm.code()
        if not self.optimized:
            return code
        with self.tmap.tmap_loop(code, "s"):
            code.raw(f"""
            if (s != {self.tmap.iterations[0] - 1}){{
                pointer_ += params_.inc_strided;
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask(self):
        code = pccm.code()
        return code


class ForwardDgradIOIteratorDP4A(bases.ConvInputIterator):
    """
    fwd: NHWC -> NPQRSC @ KRSC, k = RSC
    dgrad: NPQK -> NHWRSK @ KRSC -> RSKC, k = RSK
    wgrad: NPQK @ NHWC -> NPQRSC, k = NPQ

    for forward and dgrad, the reduce axes is KRS or RSC, both contains RS
    so we merge them to one class.
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 optimized: bool = False,
                 transpose_load: bool = False):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype, element_count, sub_tile_shape[1])
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.optimized = optimized
        self.ndim = problem_size.ndim
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        assert tmap.iterations[1] == 1
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        is_output = op_type == ConvOpType.kBackwardInput
        if op_type != ConvOpType.kForward:
            assert sub_tile_shape[
                0] == 1, "only forward support sub tile shape (DP4A)"
        if not optimized:
            self.params = AnalyticParams(problem_size, input_layout, is_output)
        else:
            self.params = IOOptParams(dtype, tile_shape_mnk, problem_size,
                                      input_layout, tmap, is_output)
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")

        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        if optimized:
            self.add_member(
                "pointers_",
                self.const_byte_pointer,
                array=f"[{tmap.iterations[0] * sub_tile_shape[0]}]")
        else:
            self.add_member("pointer_", self.const_byte_pointer)
        self.add_member("reduce_channel_offset_", "int")
        self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")

        # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")

        if not optimized:
            self.add_member(
                "offset_npq_",
                "int",
                array=
                f"[{tmap.iterations[0] * sub_tile_shape[0]}][{self.ndim + 1}]")
        if optimized:
            mask_cnt = tmap.iterations[0] * self.problem_size.ndim
            self.add_member("mask_", str(self.index_t), array=f"[{mask_cnt}]")

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        if not self.optimized:
            code.ctor_init(
                "pointer_",
                f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")
        code.ctor_init("filter_kernel_idxes_",
                       f"{{{', '.join(['0'] * self.ndim)}}}")
        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        reduce_channel_offset_ = thread_offset[1];
        """)
        if self.optimized:
            code.raw(
                f"int offset_npq_[{self.tmap.iterations[0] * self.sub_tile_shape[0]}][{self.ndim + 1}];"
            )

        with code.range_("s", str(self.tmap.iterations[0]),
                         "TV_PRAGMA_UNROLL"):
            with code.range_("ss", str(self.sub_tile_shape[0]),
                             "TV_PRAGMA_UNROLL"):

                code.raw(f"""
                int offset_npq = thread_offset[0] + s * {self.tmap.delta[0]} + ss;
                params.layout_npq.inverse(offset_npq, offset_npq_[s * {self.sub_tile_shape[0]} + ss]);
                """)
                if self.optimized:
                    zero = ", ".join(["0"] * self.ndim)
                    if self.op_type == ConvOpType.kBackwardInput:
                        code.raw(f"""
                        auto coord = nhwrs_to_npqk</*NoStride=*/true>(offset_npq_[s], {{{zero}}});
                        """)
                    else:
                        code.raw(f"""
                        auto coord = npqrs_to_nhwc(offset_npq_[s * {self.sub_tile_shape[0]} + ss], {{{zero}}});
                        """)
                    code.raw(f"""
                    pointers_[s * {self.sub_tile_shape[0]} + ss] = 
                        reinterpret_cast<{self.const_byte_pointer}>(ptr) + 
                        params.layout(coord) * {self.dtype.bitsize()} / 8;
                    """)

        if self.optimized:
            C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
            code.raw(f"clear_mask_conv();")
            for dim in range(self.problem_size.ndim):
                if self.op_type == ConvOpType.kBackwardInput:
                    h_stmt = f"offset_npq_[stride * {self.sub_tile_shape[0]} + ss][{dim + 1}] + problem_size.padding[{dim}] - ksize_idx_ * problem_size.dilation[{dim}];"
                    inp_or_out = "output_dims"
                else:
                    h_stmt = f"offset_npq_[stride * {self.sub_tile_shape[0]} + ss][{dim + 1}] * problem_size.stride[{dim}] - problem_size.padding[{dim}] + ksize_idx_ * problem_size.dilation[{dim}];"
                    inp_or_out = "input_dims"
                code.raw(f"""
                for (int ksize_idx = 0; ksize_idx < problem_size.ksize[{dim}]; ++ksize_idx){{
                    TV_PRAGMA_UNROLL
                    for (int stride = 0; stride < {self.tmap.iterations[0]}; ++stride ){{
                        TV_PRAGMA_UNROLL
                        for (int ss = 0; ss < {self.tmap.sub_tile_shape[0]}; ++ss ){{
                            int ksize_idx_ = ksize_idx;
                            if (problem_size.mode == tv::gemm::ConvMode::kConvolution){{
                                ksize_idx_ = problem_size.ksize[{dim}] - 1 - ksize_idx;
                            }}
                            int h = {h_stmt};
                            {self.index_t} pred;
                            if ({dim} == 0) {{
                                pred = (offset_npq_[stride * {self.sub_tile_shape[0]} + ss][0] < problem_size.N) && h >= 0 && h < problem_size.{inp_or_out}[{dim}];
                            }} else {{
                                pred = h >= 0 && h < problem_size.{inp_or_out}[{dim}];
                            }}
                            mask_[stride * {self.ndim} + {dim}] |= pred << (ksize_idx_ + (ss << 3));
                        }}
                    }}
                }}
                """)
            # gemm_k_iterations / (C / {tileK})
            code.raw(f"""
            if (reduce_channel_offset_ >= problem_size.{C_or_K}){{
                clear_mask_conv();
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_conv(self):
        code = FunctionCode()
        if not self.optimized:
            return code

        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.problem_size.ndim * self.tmap.iterations[0]}; ++i){{
            mask_[i] = 0;
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_conv_cond(self):
        # TODO replace these asm code with PTXCode.asm_block
        code = FunctionCode()
        if not self.optimized:
            return code
        code.arg("enable", "bool")
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.problem_size.ndim * self.tmap.iterations[0]}; ++s) {{
            // We are using inline PTX assembly here to avoid an CUDA C++ compilation
            // artifact in which control flow instructions are generated. Instead, our
            // intent is to predicate the mov instructions.
            #if defined(__CUDA_ARCH__)
            asm volatile(
                "{{\\n"
                "  .reg .pred p;\\n"
                "  .reg .u32  m;"
                "  mov.u32 m, %2;"
                "  setp.ne.b32 p, %1, 0;\\n"
                "  @p mov.u32 m, 0;\\n"
                "  mov.u32 %0, m;\\n"
                "}}\\n" 
                :
                "=r"(mask_[s])
                : 
                "r"((int)enable),
                "r"(mask_[s])
            );
            #else
            if (enable) {{
                mask_[s] = 0;
            }}
            #endif
        }}
        """)
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if not self.optimized:
            for i in range(self.ndim - 1, -1, -1):
                code.raw(f"""
                if (++filter_kernel_idxes_[{i}] < problem_size_.ksize[{i}]){{
                    return;
                }}
                filter_kernel_idxes_[{i}] = 0;
                """)
            code.raw(f"""
            reduce_channel_offset_ += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
            """)
        else:
            with contextlib.ExitStack() as stack:
                code.raw("int next_dim = 0;")
                for i in range(self.ndim - 1, -1, -1):
                    if i != 0:
                        stack.enter_context(
                            code.if_(
                                f"++filter_kernel_idxes_[{i}] == problem_size_.ksize[{i}]"
                            ))
                        code.raw(f"filter_kernel_idxes_[{i}] = 0;")
                        if i > 1:
                            code.raw(f"next_dim = {self.ndim - 1 - i + 1};")
                    else:
                        with code.if_(
                                f"++filter_kernel_idxes_[{i}] == problem_size_.ksize[{i}]"
                        ):
                            code.raw(f"next_dim = {self.ndim};")
                            code.raw(f"filter_kernel_idxes_[{i}] = 0;")
                        with code.else_():
                            code.raw(f"next_dim = {self.ndim - 1};")
            code.raw(f"""
            add_byte_offset(params_.inc_next[next_dim]);
            if (next_dim == {self.ndim}) {{
                reduce_channel_offset_ += params_.filter_c_delta;
            }}
            clear_mask_conv_cond(reduce_channel_offset_ >= problem_size_.{C_or_K});
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_byte_offset(self):
        code = FunctionCode()
        code.arg("byte_offset", str(self.long_index_t))
        if self.optimized:
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.tmap.iterations[0] * self.sub_tile_shape[0]}; ++i){{
                pointers_[i] += byte_offset;
            }}
            """)
        else:
            code.raw(f"""
            pointer_ += byte_offset;
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def at(self):
        code = FunctionCode()
        if self.optimized:
            return code
        code.arg("stride, contig, ss", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")
        if self.op_type == bases.ConvOpType.kBackwardInput:
            strides = [
                f"res[{i + 1}] / problem_size_.stride[{i}]"
                for i in range(self.ndim)
            ]
            code.raw(f"""
            auto res = nhwrs_to_npqk<true>(offset_npq_[stride], filter_kernel_idxes_);
            return {{res[0], {', '.join(strides)}, res[{self.ndim + 1}]}};
            """)
            return code
        else:
            # forward
            code.raw(
                f"auto& npq = offset_npq_[stride * {self.sub_tile_shape[0]} + ss];"
            )
            for i in range(self.ndim):
                code.raw(f"""
                int r_{i} = filter_kernel_idxes_[{i}];
                if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                    r_{i} = (problem_size_.ksize[{i}] - 1 - filter_kernel_idxes_[{i}]);
                }}
                int h_{i} = npq[{i + 1}] * problem_size_.stride[{i}] - problem_size_.padding[{i}] + r_{i} * problem_size_.dilation[{i}];
                """)
            h0h1h2 = codeops.unpack_str("h", range(self.ndim))
            code.raw(f"""
            return {{npq[0], {h0h1h2}, reduce_channel_offset_}};
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def npqrs_to_nhwc(self):
        code = FunctionCode()
        code.arg("npq", "const int*")
        code.arg("rs", f"const tv::array<int, {self.ndim}>&")
        for i in range(self.ndim):
            code.raw(f"""
            int r_{i} = rs[{i}];
            if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                r_{i} = (problem_size_.ksize[{i}] - 1 - rs[{i}]);
            }}
            int h_{i} = npq[{i + 1}] * problem_size_.stride[{i}] - problem_size_.padding[{i}] + r_{i} * problem_size_.dilation[{i}];
            """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{npq[0], {h0h1h2}, reduce_channel_offset_}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 2}>")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def nhwrs_to_npqk(self):
        code = FunctionCode()
        code.arg("npq", "const int*")
        code.nontype_targ("NoStride", "bool")
        code.arg("rs", f"const tv::array<int, {self.ndim}>&")
        for i in range(self.ndim):
            code.raw(f"""
            int r_{i} = rs[{i}];
            if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                r_{i} = (problem_size_.ksize[{i}] - 1 - rs[{i}]);
            }}
            int h_{i} = (npq[{i + 1}] + problem_size_.padding[{i}] - r_{i} * problem_size_.dilation[{i}]) / (NoStride ? 1 : problem_size_.stride[{i}]);
            """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{npq[0], {h0h1h2}, reduce_channel_offset_}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 2}>")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = FunctionCode()
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            if self.op_type == bases.ConvOpType.kForward:
                hw_valid = [
                    f"indexes[{i + 1}] >= 0 && indexes[{i + 1}] < problem_size_.input_dims[{i}]"
                    for i in range(self.ndim)
                ]
                code.raw(f"""
                return indexes[0] < problem_size_.N && 
                    {' && '.join(hw_valid)} &&
                    indexes[{self.ndim + 1}] < problem_size_.C;
                """)
            elif self.op_type == bases.ConvOpType.kBackwardInput:
                hw_valid = []  # type: List[str]
                stride_valid = []  # type: List[str]
                for i in range(self.ndim):
                    hw_valid.append((
                        f"indexes[{i + 1}] / problem_size_.stride[{i}] >= 0 && "
                        f"indexes[{i + 1}] / problem_size_.stride[{i}] < problem_size_.output_dims[{i}]"
                    ))
                    stride_valid.append(
                        f"!(indexes[{i + 1}] % problem_size_.stride[{i}])")
                code.raw(f"""
                return indexes[0] < problem_size_.N && 
                    {' && '.join(hw_valid)} &&
                    {' && '.join(stride_valid)} &&
                    indexes[{self.ndim + 1}] < problem_size_.K;
                """)
            else:
                raise NotImplementedError
        else:
            code.arg("stride, contig, ss", f"int")
            mask_valids = [
                f"(mask_[stride * {self.ndim} + {i}] & ({self.index_t}(1) << (filter_kernel_idxes_[{i}] + (ss << 3))))"
                for i in range(self.ndim)
            ]
            code.raw(f"""
            return {' && '.join(mask_valids)};
            """)

        return code.ret(f"bool")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            auto offset = params_.layout(indexes);
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
            """)
        else:
            code.arg("stride, contig, ss", "int")
            code.raw(
                f"return reinterpret_cast<{self.const_access_pointer}>(pointers_[stride * {self.sub_tile_shape[0]} + ss]);"
            )
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s", "c"):
            if self.sub_tile_shape[0] > 1:
                with code.range_("ss", self.sub_tile_shape[0],
                                 "TV_PRAGMA_UNROLL"):
                    if not self.optimized:
                        if self.op_type == bases.ConvOpType.kBackwardInput:
                            code.raw(f"""
                            int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss;
                            auto indexes = at(s, c, ss);
                            auto indexes_no_stride = nhwrs_to_npqk<true>(offset_npq_[s], filter_kernel_idxes_);

                            {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                            tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                                frag_ptr[idx], access_ptr, valid(indexes_no_stride));
                            """)
                        else:
                            code.raw(f"""
                            int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss;
                            auto indexes = at(s, c, ss);
                            {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                            tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                                frag_ptr[idx], access_ptr, valid(indexes));
                            """)
                    else:
                        code.raw(f"""
                        int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss;
                        {self.access_t} const *access_ptr = get(s, c, ss) + pointer_offset / {self.element_per_acc};
                        tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            frag_ptr[idx], access_ptr, valid(s, c, ss));
                        """)
            else:
                if not self.optimized:
                    if self.op_type == bases.ConvOpType.kBackwardInput:
                        code.raw(f"""
                        int idx = s * {self.tmap.iterations[1]} + c;
                        auto indexes = at(s, c, ss);
                        auto indexes_no_stride = nhwrs_to_npqk<true>(offset_npq_[s], filter_kernel_idxes_);

                        {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                        tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            frag_ptr[idx], access_ptr, valid(indexes_no_stride));
                        """)
                    else:
                        code.raw(f"""
                        int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]} + ss;
                        auto indexes = at(s, c, ss);
                        {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                        tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                            frag_ptr[idx], access_ptr, valid(indexes));
                        """)
                else:
                    code.raw(f"""
                    int idx = s * {self.tmap.iterations[1] * self.sub_tile_shape[0]} + c * {self.sub_tile_shape[0]};
                    {self.access_t} const *access_ptr = get(s, c, 0) + pointer_offset / {self.element_per_acc};
                    tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                        frag_ptr[idx], access_ptr, valid(s, c, 0));
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

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask(self):
        code = pccm.code()
        return code


class ForwardDgradIOIterator(bases.ConvInputIterator):
    """
    fwd: NHWC -> NPQRSC @ KRSC, k = RSC
    dgrad: NPQK -> NHWRSK @ KRSC -> RSKC, k = RSK
    wgrad: NPQK @ NHWC -> NPQRSC, k = NPQ

    for forward and dgrad, the reduce axes is KRS or RSC, both contains RS
    so we merge them to one class.
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 optimized: bool = False,
                 transpose_load: bool = False):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta
        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype, element_count, sub_tile_shape[1])
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.optimized = optimized
        self.ndim = problem_size.ndim
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        assert tmap.iterations[1] == 1
        self.add_dependency(TensorViewNVRTC, GemmBasicKernel)
        is_output = op_type == ConvOpType.kBackwardInput
        if not optimized:
            self.params = AnalyticParams(problem_size, input_layout)
        else:
            self.params = IOOptParams(dtype, tile_shape_mnk, problem_size,
                                      input_layout, tmap, is_output)
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")

        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        if optimized:
            self.add_member("pointers_",
                            self.const_byte_pointer,
                            array=f"[{tmap.iterations[0]}]")
        else:
            self.add_member("pointer_", self.const_byte_pointer)

        self.add_member("reduce_channel_offset_", "int")
        self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")

        # self.add_member("filter_kernel_idxes_", f"tv::array<int, {self.ndim}>")

        if not optimized:
            self.add_member("offset_npq_",
                            "int",
                            array=f"[{tmap.iterations[0]}][{self.ndim + 1}]")
        if optimized:
            mask_cnt = tmap.iterations[0] * self.problem_size.ndim
            self.add_member("mask_", str(self.index_t), array=f"[{mask_cnt}]")

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        if not self.optimized:
            code.ctor_init(
                "pointer_",
                f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")
        code.ctor_init("filter_kernel_idxes_",
                       f"{{{', '.join(['0'] * self.ndim)}}}")
        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        reduce_channel_offset_ = thread_offset[1];
        """)
        if self.optimized:
            code.raw(
                f"int offset_npq_[{self.tmap.iterations[0]}][{self.ndim + 1}];"
            )

        with code.range_("s", str(self.tmap.iterations[0]),
                         "TV_PRAGMA_UNROLL"):
            code.raw(f"""
            int offset_npq = thread_offset[0] + s * {self.tmap.delta[0]};
            params.layout_npq.inverse(offset_npq, offset_npq_[s]);
            """)
            if self.optimized:
                zero = ", ".join(["0"] * self.ndim)
                if self.op_type == ConvOpType.kBackwardInput:
                    code.raw(f"""
                    auto coord = nhwrs_to_npqk</*NoStride=*/true>(offset_npq_[s], {{{zero}}});
                    """)
                else:
                    code.raw(f"""
                    auto coord = npqrs_to_nhwc(offset_npq_[s], {{{zero}}});
                    """)
                code.raw(f"""
                pointers_[s] = reinterpret_cast<{self.const_byte_pointer}>(ptr) + params.layout(coord) * {self.dtype.bitsize()} / 8;
                """)

        if self.optimized:
            C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
            code.raw(f"clear_mask_conv();")
            for dim in range(self.problem_size.ndim):
                if self.op_type == ConvOpType.kBackwardInput:
                    h_stmt = f"offset_npq_[stride][{dim + 1}] + problem_size.padding[{dim}] - ksize_idx_ * problem_size.dilation[{dim}];"
                    inp_or_out = "output_dims"
                else:
                    h_stmt = f"offset_npq_[stride][{dim + 1}] * problem_size.stride[{dim}] - problem_size.padding[{dim}] + ksize_idx_ * problem_size.dilation[{dim}];"
                    inp_or_out = "input_dims"
                code.raw(f"""
                for (int ksize_idx = 0; ksize_idx < problem_size.ksize[{dim}]; ++ksize_idx){{
                    TV_PRAGMA_UNROLL
                    for (int stride = 0; stride < {self.tmap.iterations[0]}; ++stride ){{
                        int ksize_idx_ = ksize_idx;
                        if (problem_size.mode == tv::gemm::ConvMode::kConvolution){{
                            ksize_idx_ = problem_size.ksize[{dim}] - 1 - ksize_idx;
                        }}
                        int h = {h_stmt};
                        {self.index_t} pred;
                        if ({dim} == 0){{
                            pred = (offset_npq_[stride][0] < problem_size.N) && h >= 0 && h < problem_size.{inp_or_out}[{dim}];
                        }}else{{
                            pred = h >= 0 && h < problem_size.{inp_or_out}[{dim}];
                        }}
                        mask_[stride * {self.ndim} + {dim}] |= pred << ksize_idx_;
                    }}
                }}
                """)
            code.raw(f"""
            if (reduce_channel_offset_ >= problem_size.{C_or_K}){{
                clear_mask_conv();
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_conv(self):
        code = FunctionCode()
        if not self.optimized:
            return code

        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.problem_size.ndim * self.tmap.iterations[0]}; ++i){{
            mask_[i] = 0;
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask_conv_cond(self):
        code = FunctionCode()
        if not self.optimized:
            return code
        code.arg("enable", "bool")
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.problem_size.ndim * self.tmap.iterations[0]}; ++s) {{
            // We are using inline PTX assembly here to avoid an CUDA C++ compilation
            // artifact in which control flow instructions are generated. Instead, our
            // intent is to predicate the mov instructions.
            #if defined(__CUDA_ARCH__)
            asm volatile(
                "{{\\n"
                "  .reg .pred p;\\n"
                "  .reg .u32  m;"
                "  mov.u32 m, %2;"
                "  setp.ne.b32 p, %1, 0;\\n"
                "  @p mov.u32 m, 0;\\n"
                "  mov.u32 %0, m;\\n"
                "}}\\n" 
                :
                "=r"(mask_[s])
                : 
                "r"((int)enable),
                "r"(mask_[s])
            );
            #else
            if (enable) {{
                mask_[s] = 0;
            }}
            #endif
        }}
        """)
        return code

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if not self.optimized:
            for i in range(self.ndim - 1, -1, -1):
                code.raw(f"""
                if (++filter_kernel_idxes_[{i}] < problem_size_.ksize[{i}]){{
                    return;
                }}
                filter_kernel_idxes_[{i}] = 0;
                """)
            code.raw(f"""
            reduce_channel_offset_ += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
            """)
        else:
            with contextlib.ExitStack() as stack:
                code.raw("int next_dim = 0;")
                for i in range(self.ndim - 1, -1, -1):
                    if i != 0:
                        stack.enter_context(
                            code.if_(
                                f"++filter_kernel_idxes_[{i}] == problem_size_.ksize[{i}]"
                            ))
                        code.raw(f"filter_kernel_idxes_[{i}] = 0;")
                        if i > 1:
                            code.raw(f"next_dim = {self.ndim - 1 - i + 1};")
                    else:
                        with code.if_(
                                f"++filter_kernel_idxes_[{i}] == problem_size_.ksize[{i}]"
                        ):
                            code.raw(f"next_dim = {self.ndim};")
                            code.raw(f"filter_kernel_idxes_[{i}] = 0;")
                        with code.else_():
                            code.raw(f"next_dim = {self.ndim - 1};")
            code.raw(f"""
            add_byte_offset(params_.inc_next[next_dim]);
            if (next_dim == {self.ndim}) {{
                reduce_channel_offset_ += params_.filter_c_delta;
            }}
            clear_mask_conv_cond(reduce_channel_offset_ >= problem_size_.{C_or_K});
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_byte_offset(self):
        code = FunctionCode()
        code.arg("byte_offset", str(self.long_index_t))
        if self.optimized:
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.tmap.iterations[0]}; ++i){{
                pointers_[i] += byte_offset;
            }}
            """)
        else:
            code.raw(f"""
            pointer_ += byte_offset;
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def at(self):
        code = FunctionCode()
        if self.optimized:
            return code
        code.arg("stride, contig", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")
        if self.op_type == bases.ConvOpType.kBackwardInput:
            strides = [
                f"res[{i + 1}] / problem_size_.stride[{i}]"
                for i in range(self.ndim)
            ]
            code.raw(f"""
            auto res = nhwrs_to_npqk<true>(offset_npq_[stride], filter_kernel_idxes_);
            return {{res[0], {', '.join(strides)}, res[{self.ndim + 1}]}};
            """)
            return code
        else:
            # forward
            code.raw("auto& npq = offset_npq_[stride];")
            for i in range(self.ndim):
                code.raw(f"""
                int r_{i} = filter_kernel_idxes_[{i}];
                if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                    r_{i} = (problem_size_.ksize[{i}] - 1 - filter_kernel_idxes_[{i}]);
                }}
                int h_{i} = npq[{i + 1}] * problem_size_.stride[{i}] - problem_size_.padding[{i}] + r_{i} * problem_size_.dilation[{i}];
                """)
            h0h1h2 = codeops.unpack_str("h", range(self.ndim))
            code.raw(f"""
            return {{npq[0], {h0h1h2}, filter_c_[contig]}};
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def npqrs_to_nhwc(self):
        code = FunctionCode()
        code.arg("npq", "const int*")
        code.arg("rs", f"const tv::array<int, {self.ndim}>&")
        for i in range(self.ndim):
            code.raw(f"""
            int r_{i} = rs[{i}];
            if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                r_{i} = (problem_size_.ksize[{i}] - 1 - rs[{i}]);
            }}
            int h_{i} = npq[{i + 1}] * problem_size_.stride[{i}] - problem_size_.padding[{i}] + r_{i} * problem_size_.dilation[{i}];
            """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{npq[0], {h0h1h2}, reduce_channel_offset_}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 2}>")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def nhwrs_to_npqk(self):
        code = FunctionCode()
        code.arg("npq", "const int*")
        code.nontype_targ("NoStride", "bool")
        code.arg("rs", f"const tv::array<int, {self.ndim}>&")
        for i in range(self.ndim):
            code.raw(f"""
            int r_{i} = rs[{i}];
            if (problem_size_.mode == tv::gemm::ConvMode::kConvolution) {{
                r_{i} = (problem_size_.ksize[{i}] - 1 - rs[{i}]);
            }}
            int h_{i} = (npq[{i + 1}] + problem_size_.padding[{i}] - r_{i} * problem_size_.dilation[{i}]) / (NoStride ? 1 : problem_size_.stride[{i}]);
            """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{npq[0], {h0h1h2}, reduce_channel_offset_}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 2}>")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = FunctionCode()
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            if self.op_type == bases.ConvOpType.kForward:
                hw_valid = [
                    f"indexes[{i + 1}] >= 0 && indexes[{i + 1}] < problem_size_.input_dims[{i}]"
                    for i in range(self.ndim)
                ]
                code.raw(f"""
                return indexes[0] < problem_size_.N && 
                    {' && '.join(hw_valid)} &&
                    indexes[{self.ndim + 1}] < problem_size_.C;
                """)
            elif self.op_type == bases.ConvOpType.kBackwardInput:
                hw_valid = []  # type: List[str]
                stride_valid = []  # type: List[str]
                for i in range(self.ndim):
                    hw_valid.append((
                        f"indexes[{i + 1}] / problem_size_.stride[{i}] >= 0 && "
                        f"indexes[{i + 1}] / problem_size_.stride[{i}] < problem_size_.output_dims[{i}]"
                    ))
                    stride_valid.append(
                        f"!(indexes[{i + 1}] % problem_size_.stride[{i}])")
                code.raw(f"""
                return indexes[0] < problem_size_.N && 
                    {' && '.join(hw_valid)} &&
                    {' && '.join(stride_valid)} &&
                    indexes[{self.ndim + 1}] < problem_size_.K;
                """)
            else:
                raise NotImplementedError
        else:
            code.arg("stride, contig", f"int")
            mask_valids = [
                f"(mask_[stride * {self.ndim} + {i}] & ({self.index_t}(1) << filter_kernel_idxes_[{i}]))"
                for i in range(self.ndim)
            ]
            code.raw(f"""
            return {' && '.join(mask_valids)};
            """)

        return code.ret(f"bool")

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            auto offset = params_.layout(indexes);
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
            """)
        else:
            code.arg("stride, contig", "int")
            code.raw(
                f"return reinterpret_cast<{self.const_access_pointer}>(pointers_[stride]);"
            )
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with self.tmap.tmap_loop(code, "s", "c"):
            if not self.optimized:
                if self.op_type == bases.ConvOpType.kBackwardInput:
                    code.raw(f"""
                    int idx = s * {self.tmap.iterations[1]} + c;
                    auto indexes = at(s, c);
                    auto indexes_no_stride = nhwrs_to_npqk<true>(offset_npq_[s], filter_kernel_idxes_);

                    {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                    tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                        frag_ptr[idx], access_ptr, valid(indexes_no_stride));
                    """)
                else:
                    code.raw(f"""
                    int idx = s * {self.tmap.iterations[1]} + c;
                    auto indexes = at(s, c);
                    {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                    tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                        frag_ptr[idx], access_ptr, valid(indexes));
                    """)
            else:
                code.raw(f"""
                int idx = s * {self.tmap.iterations[1]} + c;
                {self.access_t} const *access_ptr = get(s, c) + pointer_offset / {self.element_per_acc};
                tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                    frag_ptr[idx], access_ptr, valid(s, c));
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

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_mask(self):
        code = pccm.code()
        return code
