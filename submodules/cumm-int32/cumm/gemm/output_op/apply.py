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

from cumm import dtypes
from cumm.common import TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.dtypes import DType
from cumm.gemm import bases, core


class ApplyOutputOp(bases.GemmApply):
    def __init__(self, element_per_acc: int, output_op, out_frag_t: str,
                 inp_frag_t: str, scale_frag_t: str, int8_inference: bool = False):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.element_per_acc = element_per_acc
        self.output_op = output_op
        self.out_frag_t = out_frag_t
        self.inp_frag_t = inp_frag_t
        self.scale_frag_t = scale_frag_t
        self.int8_inference = int8_inference
        self.add_param_class("outop", output_op, "OutputOp")

    def python_ctor(self):
        return self

    @pccm.cuda.static_function(device=True, forceinline=True)
    def apply_output_operator(self):
        code = pccm.code()
        code.arg("output_fragment", f"{self.out_frag_t} &")
        code.arg("output_op", f"OutputOp const &")
        code.arg("aligned_accum_fragment", f"{self.inp_frag_t} const &")
        code.arg("source_fragment", f"{self.out_frag_t} const &")
        out_acc_type = core.array_type(
            f"typename {self.out_frag_t}::value_type", self.element_per_acc)
        inp_acc_type = core.array_type(
            f"typename {self.inp_frag_t}::value_type", self.element_per_acc)
        if self.int8_inference:
            return code.make_invalid()

        code.raw(f"""
        constexpr int kOutFragCount = tv::array_size_v<{self.out_frag_t}>;
        using OutAccessType = {out_acc_type};
        using InputAccessType = {inp_acc_type};
        OutAccessType *output_frag_ptr =
            reinterpret_cast<OutAccessType *>(&output_fragment);
        InputAccessType const *compute_frag_ptr =
            reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
        OutAccessType const *source_frag_ptr =
            reinterpret_cast<OutAccessType const *>(&source_fragment);
        constexpr int kOutOpIterations = kOutFragCount / {self.element_per_acc};
        TV_PRAGMA_UNROLL
        for (int i = 0; i < kOutOpIterations; ++i) {{
            output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
        }}
        """)
        return code

    @pccm.cuda.static_function(device=True, forceinline=True)
    def apply_output_with_int8_operator(self):
        code = pccm.code()
        code.arg("output_fragment", f"{self.out_frag_t} &")
        code.arg("output_op", f"OutputOp const &")
        code.arg("aligned_accum_fragment", f"{self.inp_frag_t} const &")
        code.arg("source_fragment", f"{self.out_frag_t} const &")
        code.arg("int8_bias_fragment", f"{self.scale_frag_t} const &")
        code.arg("int8_scale_fragment", f"{self.scale_frag_t} const &")
        if not self.int8_inference:
            return code.make_invalid()
        out_acc_type = core.array_type(
            f"typename {self.out_frag_t}::value_type", self.element_per_acc)
        inp_acc_type = core.array_type(
            f"typename {self.inp_frag_t}::value_type", self.element_per_acc)
        scale_acc_type = core.array_type(
            f"typename {self.scale_frag_t}::value_type", self.element_per_acc)
        code.raw(f"""
        constexpr int kOutFragCount = tv::array_size_v<{self.out_frag_t}>;
        using OutAccessType = {out_acc_type};
        using InputAccessType = {inp_acc_type};
        using ScaleAccessType = {scale_acc_type};

        OutAccessType *output_frag_ptr =
            reinterpret_cast<OutAccessType *>(&output_fragment);
        InputAccessType const *compute_frag_ptr =
            reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
        OutAccessType const *source_frag_ptr =
            reinterpret_cast<OutAccessType const *>(&source_fragment);
        ScaleAccessType const *bias_ptr =
            reinterpret_cast<ScaleAccessType const *>(&int8_bias_fragment);
        ScaleAccessType const *scale_ptr =
            reinterpret_cast<ScaleAccessType const *>(&int8_scale_fragment);

        constexpr int kOutOpIterations = kOutFragCount / {self.element_per_acc};
        TV_PRAGMA_UNROLL
        for (int i = 0; i < kOutOpIterations; ++i) {{
            output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i], bias_ptr[i], scale_ptr[i]);
        }}
        """)
        return code

    @pccm.cuda.static_function(device=True, forceinline=True)
    def apply_output_with_int8_operator_no_source(self):
        code = pccm.code()
        code.arg("output_fragment", f"{self.out_frag_t} &")
        code.arg("output_op", f"OutputOp const &")
        code.arg("aligned_accum_fragment", f"{self.inp_frag_t} const &")
        code.arg("int8_bias_fragment", f"{self.scale_frag_t} const &")
        code.arg("int8_scale_fragment", f"{self.scale_frag_t} const &")
        if not self.int8_inference:
            return code.make_invalid()
        out_acc_type = core.array_type(
            f"typename {self.out_frag_t}::value_type", self.element_per_acc)
        inp_acc_type = core.array_type(
            f"typename {self.inp_frag_t}::value_type", self.element_per_acc)
        scale_acc_type = core.array_type(
            f"typename {self.scale_frag_t}::value_type", self.element_per_acc)
        code.raw(f"""
        constexpr int kOutFragCount = tv::array_size_v<{self.out_frag_t}>;
        using OutAccessType = {out_acc_type};
        using InputAccessType = {inp_acc_type};
        using ScaleAccessType = {scale_acc_type};

        OutAccessType *output_frag_ptr =
            reinterpret_cast<OutAccessType *>(&output_fragment);
        InputAccessType const *compute_frag_ptr =
            reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
        ScaleAccessType const *bias_ptr =
            reinterpret_cast<ScaleAccessType const *>(&int8_bias_fragment);
        ScaleAccessType const *scale_ptr =
            reinterpret_cast<ScaleAccessType const *>(&int8_scale_fragment);

        constexpr int kOutOpIterations = kOutFragCount / {self.element_per_acc};
        TV_PRAGMA_UNROLL
        for (int i = 0; i < kOutOpIterations; ++i) {{
            output_frag_ptr[i] = output_op(compute_frag_ptr[i], bias_ptr[i], scale_ptr[i]);
        }}
        """)
        return code

    def apply_output_operator_python(self, output_fragment: ArrayPtr,
                                     output_op,
                                     aligned_accum_fragment: ArrayPtr,
                                     source_fragment: ArrayPtr):
        kOutFragCount = output_fragment.length
        kInputFragCount = aligned_accum_fragment.length
        kOutOpIterations = kOutFragCount // self.element_per_acc
        output_frag_ptr = output_fragment.change_access_size(
            self.element_per_acc)
        compute_frag_ptr = aligned_accum_fragment.change_access_size(
            self.element_per_acc)
        source_frag_ptr = source_fragment.change_access_size(
            self.element_per_acc)

        for i in range(kOutOpIterations):
            output_frag_ptr[i] = output_op.call_op_source_python(
                compute_frag_ptr[i], source_frag_ptr[i])

    @pccm.cuda.static_function(device=True, forceinline=True)
    def apply_output_operator_no_source(self):
        code = pccm.code()
        code.arg("output_fragment", f"{self.out_frag_t} &")
        code.arg("output_op", f"OutputOp const &")
        code.arg("aligned_accum_fragment", f"{self.inp_frag_t} const &")
        out_acc_type = core.array_type(
            f"typename {self.out_frag_t}::value_type", self.element_per_acc)
        inp_acc_type = core.array_type(
            f"typename {self.inp_frag_t}::value_type", self.element_per_acc)
        if self.int8_inference:
            return code.make_invalid()


        code.raw(f"""
        constexpr int kOutFragCount = tv::array_size_v<{self.out_frag_t}>;
        using OutAccessType = {out_acc_type};
        using InputAccessType = {inp_acc_type};
        OutAccessType *output_frag_ptr =
            reinterpret_cast<OutAccessType *>(&output_fragment);
        InputAccessType const *compute_frag_ptr =
            reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
        constexpr int kOutOpIterations = kOutFragCount / {self.element_per_acc};
        TV_PRAGMA_UNROLL
        for (int i = 0; i < kOutOpIterations; ++i) {{
            output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
        }}
        """)
        return code

    def apply_output_operator_no_source_python(
            self, output_fragment: ArrayPtr, output_op,
            aligned_accum_fragment: ArrayPtr):
        kOutFragCount = output_fragment.length
        kInputFragCount = aligned_accum_fragment.length
        kOutOpIterations = kOutFragCount // self.element_per_acc
        output_frag_ptr = output_fragment.change_access_size(
            self.element_per_acc)
        compute_frag_ptr = aligned_accum_fragment.change_access_size(
            self.element_per_acc)

        for i in range(kOutOpIterations):
            output_frag_ptr[i] = output_op.call_op_nosource_python(
                compute_frag_ptr[i])
