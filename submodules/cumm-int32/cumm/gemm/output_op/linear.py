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

import pccm

from cumm import cudasim, dtypes
from cumm.constants import CUTLASS_MODE
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.dtypes import DType
from cumm.gemm import bases, core

if not CUTLASS_MODE:
    _DEFAULT_ROUND_STYLE = "tv::gemm::FloatRoundStyle::round_to_nearest"
else:
    _DEFAULT_ROUND_STYLE = "cutlass::FloatRoundStyle::round_to_nearest"


class LinearCombination(bases.GemmOutputOp):
    def __init__(self,
                 dtype_out: DType,
                 count: int,
                 dtype_acc: DType,
                 dtype_comp: DType,
                 unary_op,
                 float_round_style: str = _DEFAULT_ROUND_STYLE):
        super().__init__()
        self.add_param_class("unaryop", unary_op, "UnaryOp")
        self.add_include("tensorview/gemm/math/all.h")
        self.add_include("tensorview/gemm/core/constants.h")

        self.dtype_out = dtype_out
        self.count = count
        self.dtype_acc = dtype_acc
        self.dtype_comp = dtype_comp
        self.unary_op = unary_op
        self.round = float_round_style
        core.array_type(dtype_out, count)
        self.fragment_out_t = core.array_type(dtype_out, count)
        self.fragment_acc_t = core.array_type(dtype_acc, count)
        self.fragment_comp_t = core.array_type(dtype_comp, count)

        self.add_member("alpha, beta, act_alpha, act_beta", str(self.dtype_comp))
        self.add_member("act_type", "tv::gemm::Activation")

        # cudasim members
        self.alpha = -1
        self.beta = -1

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("alpha", str(self.dtype_comp), f"{self.dtype_comp}(1)")
        code.arg("beta", str(self.dtype_comp), f"{self.dtype_comp}(0)")
        code.arg("act_alpha", str(self.dtype_comp), f"{self.dtype_comp}(0)")
        code.arg("act_beta", str(self.dtype_comp), f"{self.dtype_comp}(0)")
        code.arg("type", "tv::gemm::Activation", "tv::gemm::Activation::kNone")

        code.ctor_init("alpha", "alpha")
        code.ctor_init("beta", "beta")
        code.ctor_init("act_alpha", "act_alpha")
        code.ctor_init("act_beta", "act_beta")
        code.ctor_init("act_type", "type")

        return code

    def python_ctor(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        return self

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def is_source_needed(self):
        return pccm.FunctionCode(f"return  beta != {self.dtype_comp}(0);").ret(
            "bool")

    def is_source_needed_python(self):
        return self.beta != 0

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_k_partition(self):
        code = pccm.code()
        code.arg("k_part, k_part_count", "int")
        code.raw(f"""
        if (k_part) {{
            beta = {self.dtype_comp}(1);
        }}
        """)
        return code

    def set_k_partition_python(self, k_part: int, k_part_count: int):
        if k_part:
            self.beta = 1.0

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               const=True,
                               name="operator()")
    def call_op_source(self):
        code = pccm.code()
        code.arg("accumulator", f"{self.fragment_acc_t} const&")
        code.arg("source", f"{self.fragment_out_t} const&")
        code.ret(f"{self.fragment_out_t}")
        if not CUTLASS_MODE:
            platform = "tv::gemm"
            platform_math = "tv::math"
        else:
            platform = "cutlass"
            platform_math = "cutlass"
        code.raw(f"""
        {platform}::NumericArrayConverter<{self.dtype_comp}, {self.dtype_out}, {self.count}, {self.round}>
            source_converter;
        {platform}::NumericArrayConverter<{self.dtype_comp}, {self.dtype_acc}, {self.count}, {self.round}>
            accumulator_converter;
        {self.fragment_comp_t} converted_source = source_converter(source);
        {self.fragment_comp_t} converted_accumulator = accumulator_converter(accumulator);

        {self.fragment_comp_t} intermediate;

        {platform_math}::multiplies<{self.fragment_comp_t}> mul_add_source;
        {platform_math}::multiply_add<{self.fragment_comp_t}> mul_add_accumulator;

        intermediate =
            mul_add_source(beta, converted_source); // X =  beta * C + uniform
        intermediate = mul_add_accumulator(alpha, converted_accumulator,
                                        intermediate); // D = alpha * Accum + X
        UnaryOp op;
        intermediate = op(intermediate, act_type, act_alpha, act_beta);
        // Convert to destination numeric type
        {platform}::NumericArrayConverter<{self.dtype_out}, {self.dtype_comp}, {self.count}, {self.round}>
            destination_converter;
        return destination_converter(intermediate);
        """)
        return code

    def call_op_source_python(self, accumulator: ArrayPtr, source: ArrayPtr):
        res = accumulator.copy()
        converted_source = source.astype(self.dtype_comp.tv_dtype)
        converted_acc = accumulator.astype(self.dtype_comp.tv_dtype)
        # if cudasim.debug_once():
        #     print("converted_source", converted_source.data.numpy_view())
        res.data.numpy_view()[:] = self.alpha * converted_acc.data.numpy_view(
        ) + self.beta * converted_source.data.numpy_view()
        res = self.unary_op(res)
        return res.astype(self.dtype_out.tv_dtype)

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               const=True,
                               name="operator()")
    def call_op_nosource(self):
        code = pccm.code()
        code.arg("accumulator", f"{self.fragment_acc_t} const&")
        code.ret(f"{self.fragment_out_t}")
        if not CUTLASS_MODE:
            platform = "tv::gemm"
            platform_math = "tv::math"
        else:
            platform = "cutlass"
            platform_math = "cutlass"

        code.raw(f"""
        {platform}::NumericArrayConverter<{self.dtype_comp}, {self.dtype_acc}, {self.count}, {self.round}>
            accumulator_converter;
        {self.fragment_comp_t} converted_accumulator = accumulator_converter(accumulator);
        {self.fragment_comp_t} intermediate;

        {platform_math}::multiplies<{self.fragment_comp_t}> mul_accumulator;

        intermediate = mul_accumulator(alpha, converted_accumulator); // D = alpha * Accum + X
        UnaryOp op;
        intermediate = op(intermediate, act_type, act_alpha, act_beta);
        // Convert to destination numeric type
        {platform}::NumericArrayConverter<{self.dtype_out}, {self.dtype_comp}, {self.count}, {self.round}>
            destination_converter;
        return destination_converter(intermediate);
        """)
        return code

    def call_op_nosource_python(self, accumulator: ArrayPtr):
        res = accumulator.copy()
        converted_acc = accumulator.astype(self.dtype_comp.tv_dtype)
        res.data.numpy_view()[:] = self.alpha * converted_acc.data.numpy_view()
        res = self.unary_op(res)
        return res.astype(self.dtype_out.tv_dtype)


class Int8Inference(bases.GemmOutputOp):
    def __init__(self,
                 dtype_out: DType,
                 count: int,
                 dtype_acc: DType,
                 dtype_comp: DType,
                 unary_op,
                 float_round_style: str = _DEFAULT_ROUND_STYLE,
                 scale_bias: bool = False):
        super().__init__()
        self.add_param_class("unaryop", unary_op, "UnaryOp")
        self.add_include("tensorview/gemm/math/all.h")
        self.add_include("tensorview/gemm/core/constants.h")

        self.dtype_out = dtype_out
        self.count = count
        self.dtype_acc = dtype_acc
        self.dtype_comp = dtype_comp
        self.unary_op = unary_op
        self.round = float_round_style
        self.scale_bias = scale_bias
        self.fragment_out_t = core.array_type(dtype_out, count)
        self.fragment_acc_t = core.array_type(dtype_acc, count)
        self.fragment_comp_t = core.array_type(dtype_comp, count)
        self.fragment_scale_t = core.array_type(dtype_comp, count)

        self.add_member("alpha, beta, act_alpha, act_beta", str(self.dtype_comp))
        self.add_member("act_type", "tv::gemm::Activation")

        # cudasim members
        self.alpha = -1
        self.beta = -1

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("alpha", str(self.dtype_comp), f"{self.dtype_comp}(1)")
        code.arg("beta", str(self.dtype_comp), f"{self.dtype_comp}(0)")
        code.arg("act_alpha", str(self.dtype_comp), f"{self.dtype_comp}(0)")
        code.arg("act_beta", str(self.dtype_comp), f"{self.dtype_comp}(0)")
        code.arg("type", "tv::gemm::Activation", "tv::gemm::Activation::kNone")

        code.ctor_init("alpha", "alpha")
        code.ctor_init("beta", "beta")
        code.ctor_init("act_alpha", "act_alpha")
        code.ctor_init("act_beta", "act_beta")
        code.ctor_init("act_type", "type")

        return code

    def python_ctor(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        return self

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def is_source_needed(self):
        return pccm.FunctionCode(f"return beta != {self.dtype_comp}(0);").ret(
            "bool")

    def is_source_needed_python(self):
        return self.beta != 0

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_k_partition(self):
        """TODO if part k, we must disable activation for all k_idx != 0 
        """
        code = pccm.code()
        code.arg("k_part, k_part_count", "int")
        code.raw(f"""
        if (k_part) {{
            // beta is for self reduce (part k from other wraps)
            beta = {self.dtype_comp}(1);
        }}
        """)
        return code

    def set_k_partition_python(self, k_part: int, k_part_count: int):
        if k_part:
            self.beta = 1.0

    def call_op_template(self, need_source: bool):
        code = pccm.code()
        code.arg("accumulator", f"{self.fragment_acc_t} const&")
        if need_source:
            code.arg("source", f"{self.fragment_out_t} const&")
        code.arg("bias", f"{self.fragment_scale_t} const&")
        code.arg("scale", f"{self.fragment_scale_t} const&")

        code.ret(f"{self.fragment_out_t}")
        if not CUTLASS_MODE:
            platform = "tv::gemm"
            platform_math = "tv::math"
        else:
            platform = "cutlass"
            platform_math = "cutlass"
        if need_source:
            code.raw(f"""
            {platform}::NumericArrayConverter<{self.dtype_comp}, {self.dtype_out}, {self.count}, {self.round}>
                source_converter;
            {self.fragment_comp_t} converted_source = source_converter(source);
            """)
        code.raw(f"""
        {platform}::NumericArrayConverter<{self.dtype_comp}, {self.dtype_acc}, {self.count}, {self.round}>
            accumulator_converter;
        {self.fragment_comp_t} converted_accumulator = accumulator_converter(accumulator);

        {self.fragment_comp_t} intermediate;

        {platform_math}::multiply_add<{self.fragment_comp_t}> mul_add_accumulator;
        // alpha = output scale, beta = output_add scale
        """)
        if self.scale_bias:
            code.raw(f"""
            {platform_math}::multiplies<{self.fragment_comp_t}> mul_add_source;
            auto bias_scaled = mul_add_source(alpha, bias); // bias_scaled = bias * alpha
            intermediate = mul_add_accumulator(scale, converted_accumulator, bias_scaled); // intermediate = scale * converted_accumulator + bias_scaled
            """)
        else:
            # assume bias scaled externally.
            code.raw(f"""
            intermediate = mul_add_accumulator(scale, converted_accumulator, bias); // intermediate = scale * converted_accumulator + bias_scaled
            """)
        if need_source:
            code.raw(f"""
            intermediate = mul_add_accumulator(beta, converted_source, intermediate); // res = converted_source * beta + intermediate
            """)
        code.raw(f"""
        UnaryOp op;
        // activation
        intermediate = op(intermediate, act_type, act_alpha, act_beta);
        """)
        with code.macro_if_("""defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \\
                        ((__CUDACC_VER_MAJOR__ > 10) ||                     \\
                        ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))"""):
            if self.dtype_out == dtypes.int8 and self.count == 1:
                # clamp int8 outputs.
                # if count > 1, the NumericArrayConverter already contains saturation.
                code.raw(f"""
                const {self.dtype_comp} kClamp = {self.dtype_comp}((1U << (sizeof({self.dtype_out}) * 8 - 1)) - 1);
                tv::math::minimum<{self.fragment_comp_t}> min_op;
                tv::math::maximum<{self.fragment_comp_t}> max_op;
                intermediate = max_op(intermediate, -kClamp - {self.dtype_comp}(1));
                intermediate = min_op(intermediate, kClamp);
                """)
        with code.macro_else_():
            if self.dtype_out == dtypes.int8:
                # clamp int8 outputs.
                code.raw(f"""
                const {self.dtype_comp} kClamp = {self.dtype_comp}((1U << (sizeof({self.dtype_out}) * 8 - 1)) - 1);
                tv::math::minimum<{self.fragment_comp_t}> min_op;
                tv::math::maximum<{self.fragment_comp_t}> max_op;
                intermediate = max_op(intermediate, -kClamp - {self.dtype_comp}(1));
                intermediate = min_op(intermediate, kClamp);
                """)
        code.macro_endif_()
        code.raw(f"""
        // Convert to destination numeric type
        {platform}::NumericArrayConverter<{self.dtype_out}, {self.dtype_comp}, {self.count}, {self.round}>
            destination_converter;
        return destination_converter(intermediate);
        """)
        return code

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               const=True,
                               name="operator()")
    def call_op_source(self):
        return self.call_op_template(True)

    def call_op_source_python(self, accumulator: ArrayPtr, source: ArrayPtr):
        res = accumulator.copy()
        converted_source = source.astype(self.dtype_comp.tv_dtype)
        converted_acc = accumulator.astype(self.dtype_comp.tv_dtype)
        # if cudasim.debug_once():
        #     print("converted_source", converted_source.data.numpy_view())
        res.data.numpy_view()[:] = self.alpha * converted_acc.data.numpy_view(
        ) + self.beta * converted_source.data.numpy_view()
        res = self.unary_op(res)
        return res.astype(self.dtype_out.tv_dtype)

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               const=True,
                               name="operator()")
    def call_op_nosource(self):
        return self.call_op_template(False)

    def call_op_nosource_python(self, accumulator: ArrayPtr):
        res = accumulator.copy()
        converted_acc = accumulator.astype(self.dtype_comp.tv_dtype)
        res.data.numpy_view()[:] = self.alpha * converted_acc.data.numpy_view()
        res = self.unary_op(res)
        return res.astype(self.dtype_out.tv_dtype)
