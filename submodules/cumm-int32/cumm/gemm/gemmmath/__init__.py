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

import numpy as np
import pccm

from cumm import dtypes
from cumm.common import TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import core


class Clamp(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, out_dtype: dtypes.DType,
                 num_element: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_include("tensorview/gemm/math/all.h")
        self.add_include("tensorview/gemm/core/constants.h")

        self.dtype = dtype

        self.out_dtype = out_dtype
        self.num_element = num_element


    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call_operator(self):
        code = pccm.code()
        argument_t = core.array_type(self.dtype, self.num_element)
        code.arg("src", f"const {argument_t} &")
        code.arg("type", f"tv::gemm::Activation")
        code.arg("alpha", f"float")
        code.arg("beta", f"float")

        code.raw(f"""
        constexpr {self.dtype} kClamp = {self.dtype}((1U << (sizeof({self.out_dtype}) * 8 - 1)) - 1);
        tv::math::minimum<{argument_t}> min_op;
        tv::math::maximum<{argument_t}> max_op;
        {argument_t} intermediate = max_op(src, -kClamp - {self.dtype}(1));
        intermediate = min_op(intermediate, kClamp);
        return intermediate;
        """)
        code.ret(argument_t)
        return code

    def python_ctor(self):
        return self

    def __call__(self, src: ArrayPtr):
        kClamp = int(((1 << (self.out_dtype.itemsize() * 8 - 1)) - 1))
        data = src.data.numpy_view()
        data = np.maximum(data, -kClamp - 1)
        intermediate = src.copy()
        intermediate_data = intermediate.data.numpy_view()
        intermediate_data[:] = np.minimum(data, kClamp)
        return intermediate


class UnaryIdentity(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, out_dtype: dtypes.DType,
                 num_element: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_include("tensorview/gemm/math/all.h")
        self.add_include("tensorview/gemm/core/constants.h")

        self.dtype = dtype
        self.out_dtype = out_dtype
        self.num_element = num_element

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call_operator(self):
        code = pccm.code()
        argument_t = core.array_type(self.dtype, self.num_element)
        code.arg("src", f"const {argument_t} &")
        code.arg("type", f"tv::gemm::Activation")
        code.arg("alpha", f"float")
        code.arg("beta", f"float")
        code.raw(f"""
        return src;
        """)
        code.ret(argument_t)
        return code

    def python_ctor(self):
        return self

    def __call__(self, src: ArrayPtr):
        return src.copy()

class UnaryActivationV1(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, out_dtype: dtypes.DType,
                 num_element: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_include("tensorview/gemm/math/all.h")
        self.add_include("tensorview/gemm/core/constants.h")

        self.dtype = dtype
        self.out_dtype = out_dtype
        self.num_element = num_element

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call_operator(self):
        code = pccm.code()
        argument_t = core.array_type(self.dtype, self.num_element)
        code.arg("src", f"const {argument_t} &")
        code.arg("type", f"tv::gemm::Activation")
        code.arg("alpha", f"float")
        code.arg("beta", f"float")
        code.raw(f"""
        namespace op = tv::arrayops;
        switch (type){{
            case tv::gemm::Activation::kNone:
                return src;
            case tv::gemm::Activation::kReLU:{{
                tv::math::maximum<{argument_t}> max_op;
                return max_op(src, {self.dtype}(0));
            }}
            default: return src;
        }}
        return src;
        """)
        code.ret(argument_t)
        return code

    def python_ctor(self):
        return self

    def __call__(self, src: ArrayPtr):
        return src.copy()


class UnaryActivation(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, out_dtype: dtypes.DType,
                 num_element: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_include("tensorview/gemm/math/all.h")
        self.add_include("tensorview/gemm/core/constants.h")

        self.dtype = dtype
        self.out_dtype = out_dtype
        self.num_element = num_element

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call_operator(self):
        code = pccm.code()
        argument_t = core.array_type(self.dtype, self.num_element)
        old_dtype = str(self.dtype)
        if self.dtype == dtypes.float16:
            old_dtype = "__half"

        nv_argument_t = core.array_type(f"tv::equivalent_data_type_t<{self.dtype}>", self.num_element)
        nv_argument_t2 = core.array_type(old_dtype, self.num_element)

        code.arg("src", f"const {argument_t} &")
        code.arg("type", f"tv::gemm::Activation")
        code.arg("alpha", f"float")
        code.arg("beta", f"float")
        code.raw(f"""
        namespace op = tv::arrayops;
        using scalar_nv_t = tv::equivalent_data_type_t<{self.dtype}>;
        auto& src_nv = reinterpret_cast<const {nv_argument_t}&>(src);
        using MathOp = op::MathScalarOp<tv::equivalent_data_type_t<{self.dtype}>>;
        switch (type){{
            case tv::gemm::Activation::kNone:
                return src;
            case tv::gemm::Activation::kReLU:{{
                tv::math::maximum<{argument_t}> max_op;
                return max_op(src, {self.dtype}(0));
            }}
            case tv::gemm::Activation::kLeakyReLU:{{
                {argument_t} res;
                TV_PRAGMA_UNROLL
                for (int i = 0; i < {self.num_element}; ++i){{
                    auto x = src[i];
                    res[i] = x >= {self.dtype}(0) ? x : x * {self.dtype}(alpha);
                }}
                return res;
            }}
            case tv::gemm::Activation::kSigmoid:{{
                {argument_t} res;
                TV_PRAGMA_UNROLL
                for (int i = 0; i < {self.num_element}; ++i){{
                    auto xx = MathOp::exp(MathOp::neg(src_nv[i]));
                    res[i] = {self.dtype}(1) / ({self.dtype}(1) + *reinterpret_cast<{self.dtype}*>( &xx ));
                }}
                return res;
            }}
            default: return src;
        }}
        return src;
        """)
        code.ret(argument_t)
        return code

    def python_ctor(self):
        return self

    def __call__(self, src: ArrayPtr):
        return src.copy()
