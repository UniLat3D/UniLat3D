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

from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pccm

from cumm import dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewNVRTC
from cumm.conv import bases
from cumm.gemm import codeops, constants, layout, thread_map
from cumm.gemm.core import MetaArray, array_type, metaseq, seq


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def conv_iwo_012_to_abc(op_type: bases.ConvOpType):
    """get a map that provide conv iwo index, give abc index.
    """
    if op_type == bases.ConvOpType.kForward:
        # i = a, w = b, o = c
        return [0, 1, 2]
    elif op_type == bases.ConvOpType.kBackwardInput:
        # i = c, w = b, o = a
        return [2, 1, 0]
    elif op_type == bases.ConvOpType.kBackwardWeight:
        # i = b, w = c, o = a
        return [1, 2, 0]
    else:
        raise NotImplementedError


def gemm_abc_012_to_iwo(op_type: bases.ConvOpType):
    if op_type == bases.ConvOpType.kForward:
        return [0, 1, 2]
    elif op_type == bases.ConvOpType.kBackwardInput:
        return [2, 1, 0]
    elif op_type == bases.ConvOpType.kBackwardWeight:
        # a = o, b = i, c = w
        return [2, 0, 1]
    else:
        raise NotImplementedError


def get_gemm_trans_abc(op_type: bases.ConvOpType):
    if op_type == bases.ConvOpType.kForward:
        return (False, True, False)
    elif op_type == bases.ConvOpType.kBackwardInput:
        return (False, False, False)
    elif op_type == bases.ConvOpType.kBackwardWeight:
        return (True, False, False)
    else:
        raise NotImplementedError


class ConvProblemCommon(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC, GemmBasic)

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def implicit_gemm_mnk(self):
        code = pccm.code()
        code.arg("op_type", "tv::gemm::ConvOpType")
        code.arg("N, C, K", "int")
        code.arg("kernel_volume, in_prod, out_prod", "int")
        code.arg("mask_sparse", "bool")
        code.raw(f"""
        if (mask_sparse){{
            switch (op_type) {{
                case tv::gemm::ConvOpType::kForward:
                    return {{N, K, C * kernel_volume}};
                case tv::gemm::ConvOpType::kBackwardInput:
                    return {{N, C, K * kernel_volume}};
                case tv::gemm::ConvOpType::kBackwardWeight:
                    return {{K, C * kernel_volume, N}};
                default:
                    return {{}};
            }}
            return {{}};
        }}else{{
            switch (op_type) {{
                case tv::gemm::ConvOpType::kForward:
                    return {{N * out_prod, K, C * kernel_volume}};
                case tv::gemm::ConvOpType::kBackwardInput:
                    return {{N * in_prod, C, K * kernel_volume}};
                case tv::gemm::ConvOpType::kBackwardWeight:
                    return {{K, C * kernel_volume, N * out_prod}};
                default:
                    return {{}};
            }}
            return {{}};
        }}
        """)
        code.ret(f"tv::array<int, 3>")
        return code

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def conv_iwo_012_to_abc(self):
        code = pccm.code()
        code.arg("op_type", "tv::gemm::ConvOpType")
        code.raw(f"""
        switch (op_type) {{
            case tv::gemm::ConvOpType::kForward:
                return {{0, 1, 2}};
            case tv::gemm::ConvOpType::kBackwardInput:
                return {{2, 1, 0}};
            case tv::gemm::ConvOpType::kBackwardWeight:
                return {{1, 2, 0}};
            default:
                return {{}};
        }}
        return {{}};
        """)
        code.ret(f"tv::array<int, 3>")
        return code

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def gemm_abc_012_to_iwo(self):
        code = pccm.code()
        code.arg("op_type", "tv::gemm::ConvOpType")
        code.raw(f"""
        switch (op_type) {{
            case tv::gemm::ConvOpType::kForward:
                return {{0, 1, 2}};
            case tv::gemm::ConvOpType::kBackwardInput:
                return {{2, 1, 0}};
            case tv::gemm::ConvOpType::kBackwardWeight:
                return {{2, 0, 1}};
            default:
                return {{}};
        }}
        return {{}};
        """)
        code.ret(f"tv::array<int, 3>")
        return code


class ConvProblem(pccm.ParameterizedClass):
    def __init__(self,
                 ndim: int,
                 op_type: bases.ConvOpType,
                 layout_desp_input: bases.ConvLayout,
                 layout_desp_weight: bases.ConvLayout,
                 layout_desp_output: bases.ConvLayout,
                 mask_sparse: bool = False):
        super().__init__()
        self.add_dependency(ConvProblemCommon)
        self.ndim = ndim
        self.op_type = op_type
        self.layout_desp_input = layout_desp_input
        self.layout_desp_weight = layout_desp_weight
        self.layout_desp_output = layout_desp_output
        self.mask_sparse = mask_sparse
        self.add_dependency(TensorViewNVRTC)
        # batch, input channel, output channel
        self.add_member("N, C, K", f"int")
        # HW
        if not mask_sparse:
            self.add_member(
                "input_dims, output_dims, ksize, padding, stride, dilation",
                f"tv::array<int, {ndim}>")
        else:
            self.add_member("kernel_volume", f"int")
        self.add_member("mode", "tv::gemm::ConvMode")
        self.add_member("split_k_slices, groups", "int")

        # cudasim
        self.N_ = 0
        self.C_ = 0
        self.K_ = 0
        self.input_dims_ = [0] * ndim
        self.output_dims_ = [0] * ndim
        self.ksize_ = [0] * ndim
        self.padding_ = [0] * ndim
        self.stride_ = [0] * ndim
        self.dilation_ = [0] * ndim
        self.mode_: bases.ConvMode = bases.ConvMode.kCrossCorrelation
        self.split_k_slices_ = -1
        self.groups_ = -1

    @pccm.constructor(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def defctor(self):
        return pccm.code()

    @pccm.constructor(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def ctor_without_out_calc(self):
        code = pccm.code()
        code.arg("N, C, K", "int")
        if self.mask_sparse:
            code.arg("kernel_volume", f"int")
        else:
            code.arg(
                "input_dims, output_dims, ksize, padding, stride, dilation",
                f"tv::array<int, {self.ndim}>")

        code.arg("mode", "tv::gemm::ConvMode", "tv::gemm::ConvMode::kCrossCorrelation")
        code.arg("split_k_slices", "int", "1")
        code.arg("groups", "int", "1")

        for arg in code.arguments:
            code.ctor_init(arg.name, arg.name)
        return code

    def python_ctor(self, N: int, C: int, K: int, input_dims: List[int],
                    output_dims: List[int], ksize: List[int],
                    padding: List[int], stride: List[int], dilation: List[int],
                    mode: bases.ConvMode, split_k_slices: int, groups: int):
        new_obj = ConvProblem(self.ndim, self.op_type, self.layout_desp_input,
                              self.layout_desp_weight, self.layout_desp_output)
        new_obj.N_ = N
        new_obj.C_ = C
        new_obj.K_ = K
        new_obj.input_dims_ = input_dims
        new_obj.output_dims_ = output_dims
        new_obj.ksize_ = ksize
        new_obj.padding_ = padding
        new_obj.stride_ = stride
        new_obj.dilation_ = dilation
        new_obj.mode_ = mode
        new_obj.split_k_slices_ = split_k_slices
        new_obj.groups_ = groups
        return new_obj

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"])
    def get_npq_shape(self):
        code = pccm.FunctionCode()
        if self.mask_sparse:
            code.raw(f"return {{N}};")
        else:
            pqs = codeops.unpack("output_dims", range(self.ndim))
            code.raw(f"return {{N, {pqs}}};")
        return code.ret(f"tv::array<int, {self.ndim + 1}>")

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def check_npq_not_overflow(self):
        code = pccm.code()
        code.raw(f"""
        auto shape = get_npq_shape();
        """)
        code.raw(f"return (")
        lines: List[str] = []
        for i in range(self.ndim + 1):
            lines.append(f"int64_t(shape[{i}])")
        code.raw("std::abs(" + " * ".join(lines) + ") <= std::numeric_limits<int>::max()")
        code.raw(");")
        code.ret("bool")
        return code

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def calc_output_dims(self):
        code = pccm.code()
        code.arg("input_dims, ksize, padding, stride, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.raw(f"""
        tv::array<int, {self.ndim}> out;
        for (int i = 0; i < {self.ndim}; ++i){{
            out[i] = ((input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) / stride[i]) + 1;
        }}
        return out;
        """)
        code.ret(f"tv::array<int, {self.ndim}>")
        return code

    @staticmethod
    def calc_output_dims_python(input_dims: List[int], ksize: List[int],
                                padding: List[int], stride: List[int],
                                dilation: List[int]):
        ndim = len(input_dims)
        out = [0] * ndim
        for i in range(ndim):
            out[i] = (
                (input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) //
                stride[i]) + 1
        return out

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def implicit_gemm_mnk(self):
        code = pccm.code()
        code.arg("op_type", "tv::gemm::ConvOpType")
        if self.mask_sparse:
            code.raw(f"""
            return ConvProblemCommon::implicit_gemm_mnk(op_type, N, C, K, kernel_volume, -1, -1, true);
            """)
        else:
            code.raw(f"""
            int ksize_prod = ksize.op<tv::arrayops::prod>();
            int in_prod = input_dims.op<tv::arrayops::prod>();
            int out_prod = output_dims.op<tv::arrayops::prod>();
            return ConvProblemCommon::implicit_gemm_mnk(op_type, N, C, K, ksize_prod, in_prod, out_prod, false);
            """)

        code.ret(f"tv::array<int, 3>")
        return code

    def implicit_gemm_mnk_python(self, conv_op_type: bases.ConvOpType):
        ksize_prod = int(np.prod(self.ksize_))
        out_prod = int(np.prod(self.output_dims_))
        in_prod = int(np.prod(self.input_dims_))

        if conv_op_type == bases.ConvOpType.kForward:
            return [self.N_ * out_prod, self.K_, self.C_ * ksize_prod]
        elif conv_op_type == bases.ConvOpType.kBackwardInput:
            return [self.N_ * in_prod, self.C_, self.K_ * ksize_prod]
        elif conv_op_type == bases.ConvOpType.kBackwardWeight:
            return [self.K_, self.C_ * ksize_prod, self.N_ * out_prod]
        else:
            raise NotImplementedError

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def implicit_gemm_k_iterations(self):
        code = pccm.code()
        code.arg("op_type", "tv::gemm::ConvOpType")
        code.arg("tile_shape_k", "int")
        if self.mask_sparse:
            code.raw(f"""
            switch (op_type) {{
                case tv::gemm::ConvOpType::kForward:
                    return kernel_volume * tv::div_up(tv::div_up(C, split_k_slices), tile_shape_k);
                case tv::gemm::ConvOpType::kBackwardInput:
                    return kernel_volume * tv::div_up(tv::div_up(K, split_k_slices), tile_shape_k);
                case tv::gemm::ConvOpType::kBackwardWeight:
                    return tv::div_up(tv::div_up(N, split_k_slices), tile_shape_k);
                default:
                    return 0;
            }}
            return 0;
            """)
        else:
            code.raw(f"""
            int ksize_prod = ksize.op<tv::arrayops::prod>();
            int in_prod = input_dims.op<tv::arrayops::prod>();
            int out_prod = output_dims.op<tv::arrayops::prod>();
            switch (op_type) {{
                case tv::gemm::ConvOpType::kForward:
                    return ksize_prod * tv::div_up(tv::div_up(C, split_k_slices), tile_shape_k);
                case tv::gemm::ConvOpType::kBackwardInput:
                    return ksize_prod * tv::div_up(tv::div_up(K, split_k_slices), tile_shape_k);
                case tv::gemm::ConvOpType::kBackwardWeight:
                    return tv::div_up(tv::div_up(N * out_prod, split_k_slices), tile_shape_k);
                default:
                    return 0;
            }}
            return 0;
            """)

        code.ret(f"int")
        return code

    def implicit_gemm_k_iterations_python(self, conv_op_type: bases.ConvOpType,
                                          tile_shape_k: int):
        ksize_prod = int(np.prod(self.ksize_))
        out_prod = int(np.prod(self.output_dims_))
        in_prod = int(np.prod(self.input_dims_))

        if conv_op_type == bases.ConvOpType.kForward:
            k_iters_in_C = codeops.div_up(self.C_, self.split_k_slices_)
            return ksize_prod * (codeops.div_up(k_iters_in_C, tile_shape_k))
        elif conv_op_type == bases.ConvOpType.kBackwardInput:
            k_iters_in_C = codeops.div_up(self.K_, self.split_k_slices_)
            return ksize_prod * (codeops.div_up(k_iters_in_C, tile_shape_k))
        elif conv_op_type == bases.ConvOpType.kBackwardWeight:
            k_iters_in_C = codeops.div_up(self.N_ * out_prod,
                                          self.split_k_slices_)
            return codeops.div_up(k_iters_in_C, tile_shape_k)
        else:
            raise NotImplementedError

    def get_input_shape_python(self):
        if self.layout_desp_input.is_channel_first():
            return [self.N_, self.C_, *self.input_dims_]
        else:
            return [self.N_, *self.input_dims_, self.C_]

    def get_weight_shape_python(self):
        if self.layout_desp_weight.is_channel_first():
            return [self.K_, self.C_, *self.ksize_]
        else:
            return [self.K_, *self.ksize_, self.C_]

    def get_output_shape_python(self):
        if self.layout_desp_output.is_channel_first():
            return [self.N_, self.K_, *self.output_dims_]
        else:
            return [self.N_, *self.output_dims_, self.K_]

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def get_input_shape(self):
        code = pccm.code()
        msg = ", ".join(f"input_dims[{i}]" for i in range(self.ndim))
        if self.mask_sparse:
            code.raw(f"return {{N, C}};")
            code.ret(f"tv::array<int, 2>")
        else:
            if self.layout_desp_input.is_channel_first():
                code.raw(f"return {{N, C, {msg}}};")
            else:
                code.raw(f"return {{N, {msg}, C}};")
            code.ret(f"tv::array<int, {self.ndim + 2}>")
        return code

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def get_weight_shape(self):
        code = pccm.code()
        if self.mask_sparse:
            msg = "kernel_volume"
        else:
            msg = ", ".join(f"ksize[{i}]" for i in range(self.ndim))

        if self.layout_desp_weight.is_channel_first():
            code.raw(f"return {{K, C, {msg}}};")
        else:
            code.raw(f"return {{K, {msg}, C}};")
        weight_ndim = 3 if self.mask_sparse else self.ndim + 2

        code.ret(f"tv::array<int, {weight_ndim}>")
        return code

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def get_output_shape(self):
        code = pccm.code()
        msg = ", ".join(f"output_dims[{i}]" for i in range(self.ndim))
        if self.mask_sparse:
            code.raw(f"return {{N, K}};")
            code.ret(f"tv::array<int, 2>")
        else:
            if self.layout_desp_input.is_channel_first():
                code.raw(f"return {{N, K, {msg}}};")
            else:
                code.raw(f"return {{N, {msg}, K}};")
            code.ret(f"tv::array<int, {self.ndim + 2}>")
        return code

    def get_a_b_layout_class(self):
        weight_ndim = 3 if self.mask_sparse else self.ndim + 2
        if self.op_type == bases.ConvOpType.kForward:
            return (self.layout_desp_input.get_layout_class(self.ndim + 2),
                    self.layout_desp_weight.get_layout_class(weight_ndim))
        elif self.op_type == bases.ConvOpType.kBackwardInput:
            return (self.layout_desp_output.get_layout_class(self.ndim + 2),
                    self.layout_desp_weight.get_layout_class(weight_ndim))
        elif self.op_type == bases.ConvOpType.kBackwardWeight:
            return (self.layout_desp_output.get_layout_class(self.ndim + 2),
                    self.layout_desp_input.get_layout_class(self.ndim + 2))
        else:
            raise NotImplementedError

    def get_c_layout_class(self):
        weight_ndim = 3 if self.mask_sparse else self.ndim + 2
        if self.op_type == bases.ConvOpType.kForward:
            return self.layout_desp_output.get_layout_class(self.ndim + 2)
        elif self.op_type == bases.ConvOpType.kBackwardInput:
            return self.layout_desp_input.get_layout_class(self.ndim + 2)
        elif self.op_type == bases.ConvOpType.kBackwardWeight:
            return self.layout_desp_weight.get_layout_class(weight_ndim)
        else:
            raise NotImplementedError

    def get_gemm_trans_abc(self):
        return get_gemm_trans_abc(self.op_type)
