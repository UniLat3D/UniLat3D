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

from typing import List, Tuple

import numpy as np
import pccm

from cumm import dtypes
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.gemm import constants


class RowMajorInterleaved(pccm.ParameterizedClass):
    def __init__(self, interleave: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        # self.add_include("tensorview/gemm/core/layout.h")
        self.interleave = interleave
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        # self.type = f"tv::gemm::layout::RowMajorInterleaved<{interleave}>"
        shape = [0, 0]
        self.shape = shape
        self.stride = shape[1] * interleave

        self.static_stride = -1

    def python_factory(self, shape):
        self.shape = shape
        self.stride = shape[1] * self.interleave
        return self

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    def python_ctor(self, stride):
        new_obj = RowMajorInterleaved(self.interleave)
        new_obj.stride = stride
        return new_obj

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(
            f"return {self.class_name}(shape[1] * {self.interleave});")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    def from_shape_python(self, shape):
        l = RowMajorInterleaved(self.interleave)
        l.stride = shape[1] * self.interleave
        return l

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def call_operator(self):
        code = pccm.FunctionCode(f"""
        {self.index_t} row_major = x / {self.interleave};
        {self.index_t} row_minor = x % {self.interleave};
        return {self.long_index_t}(row_major) * {self.long_index_t}(stride) +
            {self.long_index_t}(y) * {self.interleave} + row_minor;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def __call__(self, x: int, y: int):
        row_major = x // self.interleave
        row_minor = x % self.interleave
        return row_major * self.stride + y * self.interleave + row_minor

    def inverse(self, axis: int):
        if axis == 0:
            code = pccm.FunctionCode(f"""
            {self.index_t} row_major = {self.index_t}(offset / stride);
            {self.index_t} residual = {self.index_t}(offset % stride);
            {self.index_t} row_minor = residual % {self.interleave};
            return row_major * {self.interleave} + row_minor;
            """)
        else:
            code = pccm.FunctionCode(f"""
            return (offset % stride) / {self.interleave};
            """)
        code.arg("offset", self.long_index_t)
        return code.ret(self.index_t)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_0(self):
        return self.inverse(0)

    def inverse_0_python(self, offset: int):
        row_major = offset // self.stride
        residual = offset % self.stride
        row_minor = residual % self.interleave
        return row_major * self.interleave + row_minor

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_1(self):
        return self.inverse(1)

    def inverse_1_python(self, offset: int):
        return (offset % self.stride) // self.interleave


class ColumnMajorInterleaved(pccm.ParameterizedClass):
    def __init__(self, interleave: int, shape: List[int]):
        super().__init__()
        # self.add_include("tensorview/gemm/core/layout.h")
        self.interleave = interleave
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        # self.type = f"tv::gemm::layout::RowMajorInterleaved<{interleave}>"
        shape = [0, 0]
        self.shape = shape
        self.stride = shape[0] * interleave
        self.static_stride = -1

    def python_ctor(self, stride):
        new_obj = ColumnMajorInterleaved(self.interleave, self.shape)
        new_obj.stride = stride
        return new_obj

    def __call__(self, x: int, y: int):
        column_major = y // self.interleave
        column_minor = y % self.interleave
        return column_major * self.stride + x * self.interleave + column_minor

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(
            f"return {self.class_name}(shape[0] * {self.interleave});")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    def from_shape_python(self, shape):
        l = ColumnMajorInterleaved(self.interleave, self.shape)
        l.stride = shape[0] * self.interleave
        return l

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def call_operator(self):
        code = pccm.FunctionCode(f"""
        {self.index_t} column_major = y / {self.interleave};
        {self.index_t} column_minor = y % {self.interleave};
        return {self.long_index_t}(column_major) * {self.long_index_t}(stride) +
            {self.long_index_t}(x) * {self.interleave} + column_minor;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def inverse(self, axis: int):
        if axis == 0:
            code = pccm.FunctionCode(f"""
            {self.index_t} column_major = {self.index_t}(offset / stride);
            {self.index_t} residual = {self.index_t}(offset % stride);
            {self.index_t} column_minor = residual % {self.interleave};
            return column_major * {self.interleave} + column_minor;
            """)
        else:
            code = pccm.FunctionCode(f"""
            return (offset % stride) / {self.interleave};
            """)
        code.arg("offset", self.long_index_t)
        return code.ret(self.index_t)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_0(self):
        return self.inverse(0)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_1(self):
        return self.inverse(1)

    def inverse_0_python(self, offset: int):
        column_major = offset // self.stride
        residual = (offset % self.stride)
        column_minor = residual % self.interleave
        return column_major * self.interleave + column_minor

    def inverse_1_python(self, offset: int):
        return (offset % self.stride) // self.interleave


class RowMajor(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        # self.add_include("tensorview/gemm/core/layout.h")
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        # self.type = f"tv::gemm::layout::RowMajorInterleaved<{interleave}>"
        shape = [0, 0]
        self.shape = shape
        self.stride = shape[1]
        self.static_stride = -1

    def python_ctor(self, stride):
        new_obj = RowMajor()
        new_obj.stride = stride
        return new_obj

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(f"return {self.class_name}(shape[1]);")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    def from_shape_python(self, shape):
        l = RowMajor()
        l.stride = shape[1]
        return l

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def call_operator(self):
        code = pccm.FunctionCode(f"""
        return {self.long_index_t}(x) * {self.long_index_t}(stride) + y;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def inverse(self, axis: int):
        if axis == 0:
            code = pccm.FunctionCode(
                f"return {self.index_t}(offset / stride);")
        else:
            code = pccm.FunctionCode(
                f"return {self.index_t}(offset % stride);")
        code.arg("offset", self.long_index_t)
        return code.ret(self.index_t)

    def inverse_0_python(self, offset: int):
        return (offset // self.stride)

    def inverse_1_python(self, offset: int):
        return (offset % self.stride)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_0(self):
        return self.inverse(0)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_1(self):
        return self.inverse(1)

    def __call__(self, x: int, y: int):
        return x * self.stride + y


class ColumnMajor(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        # self.add_include("tensorview/gemm/core/layout.h")
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        # self.type = f"tv::gemm::layout::RowMajorInterleaved<{interleave}>"
        shape = [0, 0]
        self.shape = shape
        self.stride = shape[0]
        self.static_stride = -1

    def python_ctor(self, stride):
        new_obj = ColumnMajor()
        new_obj.stride = stride
        return new_obj

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(f"return {self.class_name}(shape[0]);")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    def from_shape_python(self, shape):
        l = ColumnMajor()
        l.stride = shape[0]
        return l

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def call_operator(self):
        code = pccm.FunctionCode(f"""
        return {self.long_index_t}(y) * {self.long_index_t}(stride) + x;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def inverse(self, axis: int):
        if axis == 0:
            code = pccm.FunctionCode(
                f"return {self.index_t}(offset % stride);")
        else:
            code = pccm.FunctionCode(
                f"return {self.index_t}(offset / stride);")
        code.arg("offset", self.long_index_t)
        return code.ret(self.index_t)

    def inverse_0_python(self, offset: int):
        return (offset % self.stride)

    def inverse_1_python(self, offset: int):
        return (offset // self.stride)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_0(self):
        return self.inverse(0)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def inverse_1(self):
        return self.inverse(1)

    def __call__(self, x: int, y: int):
        return y * self.stride + x


def to_stride(shape: np.ndarray):
    stride = np.ones_like(shape)
    stride[:shape.shape[0] - 1] = np.cumprod(shape[::-1])[::-1][1:]
    return stride


class TensorGeneric(pccm.ParameterizedClass):
    """Generic Tensor Layout. 
    fast_divmod have faster inverse performance, 
    but need more registers.
    """
    def __init__(self, ndim: int, fast_divmod: bool = False, stride_dtype = dtypes.int32):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.fast_divmod = fast_divmod
        if stride_dtype == dtypes.int64:
            assert not fast_divmod, "don't support fast divmod for now"
        self.ndim = ndim
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.stride_dtype = stride_dtype
        # we only need contiguous stride here.
        if ndim > 1:
            self.add_member("strides", f"tv::array<{stride_dtype}, {ndim - 1}>")
            if fast_divmod:
                self.add_member("multipliers",
                                f"tv::array<unsigned int, {ndim - 1}>")
                self.add_member("shift_rights",
                                f"tv::array<unsigned int, {ndim - 1}>")
        if fast_divmod:
            self.add_include("tensorview/math/fastmath.h")

        # cudasim
        self.strides = [0] * ndim

    def python_ctor(self, stride: List[int]):
        assert self.ndim > 1
        new_obj = TensorGeneric(self.ndim)
        assert len(stride) == self.ndim - 1
        new_obj.strides = stride
        return new_obj

    def from_shape_python(self, shape: List[int]):
        assert len(shape) == self.ndim
        new_obj = TensorGeneric(self.ndim)
        new_obj.strides = to_stride(np.array(shape)).tolist()[:-1]
        return new_obj

    def __call__(self, indexes: List[int]):
        assert len(indexes) == self.ndim
        offset = indexes[-1]
        for i in range(self.ndim - 1):
            offset += self.strides[i] * indexes[i]
        return offset

    @pccm.constructor(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def ctor(self):
        code = pccm.code()
        if self.ndim > 1:
            code.arg("strides", f"tv::array<{self.stride_dtype}, {self.ndim - 1}> const&")
            code.ctor_init("strides", "strides")
        if self.fast_divmod:
            for i in range(self.ndim - 1):
                code.raw(f"""
                tv::math::find_divisor(multipliers[{i}], shift_rights[{i}], strides[{i}]);
                """)
        return code

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def from_shape(self):
        code = pccm.code()
        code.arg("shape", f"const tv::array<int, {self.ndim}> &")
        if self.ndim == 1:
            code.raw(f"""
            return {self.class_name}();
            """)
        else:
            code.raw(f"return {self.class_name}({{")
            lines: List[str] = []
            stmts: List[str] = []
            for i in range(self.ndim - 1):
                if self.stride_dtype != dtypes.int32:
                    lines.append(f"{self.stride_dtype}(shape[{i + 1}])")
                else:
                    lines.append(f"shape[{i + 1}]")
            for i in range(self.ndim - 1):
                stmts.append("  " + " * ".join(lines[self.ndim - 2 - i:]))
            code.raw(",\n".join(stmts[::-1]))
            code.raw("});")
        code.ret(self.class_name)
        return code

    @pccm.member_function(name="operator()",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def call_operator(self):
        code = pccm.code()
        code.arg("indexes", f"const tv::array<int, {self.ndim}> &")
        stmts = [f"indexes[{self.ndim - 1}]"]
        for i in range(self.ndim - 1):
            if self.stride_dtype != dtypes.int32:
                stmts.append(f"{self.long_index_t}(strides[{i}] * {self.stride_dtype}(indexes[{i}]))")
            else:
                stmts.append(f"{self.long_index_t}(strides[{i}] * indexes[{i}])")
        code.raw(f"return {' + '.join(stmts)};")
        return code.ret(self.long_index_t)

    @pccm.member_function(name="operator()",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def call_operator2(self):
        code = pccm.code()
        code.arg("indexes", f"const int*")
        stmts = [f"indexes[{self.ndim - 1}]"]
        for i in range(self.ndim - 1):
            if self.stride_dtype != dtypes.int32:
                stmts.append(f"{self.long_index_t}(strides[{i}] * {self.stride_dtype}(indexes[{i}]))")
            else:
                stmts.append(f"{self.long_index_t}(strides[{i}] * indexes[{i}])")
        code.raw(f"return {' + '.join(stmts)};")
        return code.ret(self.long_index_t)

    @pccm.member_function(name="operator()",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def call_operator3(self):
        code = pccm.code()
        code.arg("indexes", f"const tv::array<int64_t, {self.ndim}> &")
        stmts = [f"indexes[{self.ndim - 1}]"]
        for i in range(self.ndim - 1):
            stmts.append(f"{self.long_index_t}(strides[{i}] * indexes[{i}])")
        code.raw(f"return {' + '.join(stmts)};")
        return code.ret(self.long_index_t)

    @pccm.member_function(name="operator()",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def call_operator4(self):
        code = pccm.code()
        code.arg("indexes", f"const int64_t*")
        stmts = [f"indexes[{self.ndim - 1}]"]
        for i in range(self.ndim - 1):
            stmts.append(f"{self.long_index_t}(strides[{i}] * indexes[{i}])")
        code.raw(f"return {' + '.join(stmts)};")
        return code.ret(self.long_index_t)

    def inverse_python(self, offset: int):
        out = [0] * self.ndim
        for i in range(self.ndim - 1):
            out[i] = offset // self.strides[i]
            offset -= out[i] * self.strides[i]
        out[-1] = offset
        return out

    def inverse_template(self,
                         external_out: bool = False,
                         unpack: bool = False,
                         ptr_out: bool = False):
        if unpack:
            assert external_out is True, "packed must be external out"
        if ptr_out:
            assert not unpack and external_out
        code = pccm.code()
        code.arg("index", str(self.long_index_t))
        if unpack:
            for i in range(self.ndim):
                code.arg(f"idx_{i}", "int &")
        elif external_out:
            if ptr_out:
                code.arg("out", f"int*")
            else:
                code.arg("out", f"tv::array<int, {self.ndim}>&")
        output_names = [f"out[{i}]" for i in range(self.ndim)]
        if unpack:
            output_names = [f"idx_{i}" for i in range(self.ndim)]
        if not external_out:
            code.raw(f"""
            tv::array<int, {self.ndim}> out;
            """)

        if self.ndim == 1:
            if not external_out:
                code.raw(f"""
                out[0] = index;
                return out;
                """)
                code.ret(f"tv::array<int, {self.ndim}>")
            else:
                code.raw(f"""
                {output_names[0]} = index;
                """)
            return code
        assert self.ndim >= 2
        if self.fast_divmod:
            if self.ndim > 2:
                code.raw(f"""
                int residual;
                tv::math::fast_divmod({output_names[0]}, residual, index, strides[0], multipliers[0], shift_rights[0]);
                """)
            else:
                code.raw(f"""
                int residual = index;
                """)
            for i in range(1, self.ndim - 2):
                code.raw(f"""
                tv::math::fast_divmod({output_names[i]}, residual, residual, strides[{i}], multipliers[{i}], shift_rights[{i}]);
                """)
            idx = self.ndim - 1
            code.raw(f"""
            tv::math::fast_divmod({output_names[idx - 1]}, {output_names[idx]}, residual, strides[{idx - 1}], multipliers[{idx - 1}], shift_rights[{idx - 1}]);
            """)

        else:
            code.raw(f"""
            {self.long_index_t} residual = index;
            """)
            for i in range(self.ndim - 2):
                code.raw(f"""
                {output_names[i]} = int(residual / strides[{i}]);
                residual = residual % strides[{i}];
                """)
            code.raw(f"""
            {output_names[self.ndim - 2]} = int(residual / strides[{self.ndim - 2}]);
            {output_names[self.ndim - 1]} = int(residual % strides[{self.ndim - 2}]);
            """)
        if not external_out:
            code.raw(f"""
            return out;
            """)
            code.ret(f"tv::array<int, {self.ndim}>")
        return code

    @pccm.member_function(name="inverse",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def inverse_return_out(self):
        return self.inverse_template(False)

    @pccm.member_function(name="inverse",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def inverse_input_out(self):
        return self.inverse_template(True)

    @pccm.member_function(name="inverse",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def inverse_input_unpack(self):
        return self.inverse_template(True, True)

    @pccm.member_function(name="inverse",
                          header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def inverse_input_ptr_out(self):
        return self.inverse_template(True, False, True)


class TensorNCxHWx(pccm.ParameterizedClass):
    # TODO
    def __init__(self, ndim: int, interleave: int):
        super().__init__()
        assert ndim > 1
        self.ndim = ndim
        self.interleave = interleave


if __name__ == "__main__":
    T = TensorGeneric(4)
    from pccm.core.codegen import generate_code_list
    l = T.from_shape_python([1, 32, 64, 64])
    offset = l([0, 5, 33, 52])
    arr = np.random.uniform(-1, 1, size=[1, 32, 64, 64])
    print(offset, arr.reshape(-1)[offset], arr[0, 5, 33, 52])
