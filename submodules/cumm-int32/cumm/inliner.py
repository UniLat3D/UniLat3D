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

import enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pccm
from pccm import FunctionCode
from pccm.builder.inliner import InlineBuilder, InlineBuilderPlugin
from pccm.utils import get_qualname_of_type

from cumm import dtypes
from cumm import tensorview as tv
from cumm.common import TensorView

_TV_DTYPE_TO_CUMM_DTYPE = {d.tv_dtype: d for d in dtypes.ALL_DTYPES}


class CummTensorPlugin(InlineBuilderPlugin):
    def handle_captured_type(self,
                             name: str,
                             code: FunctionCode,
                             obj: Any,
                             user_arg: Optional[Any] = None) -> Optional[str]:
        return

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        if isinstance(obj, np.ndarray):
            return tv.from_numpy(obj)
        return obj

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        return "tv::Tensor"


class CummTensorKernelPlugin(InlineBuilderPlugin):
    def __init__(self) -> None:
        super().__init__()

    def handle_captured_type(self,
                             name: str,
                             code: FunctionCode,
                             obj: Any,
                             user_arg: Optional[Any] = None) -> Optional[str]:
        ten_name = name + "_ten_tensor"

        assert isinstance(obj, tv.Tensor)
        dtype = _TV_DTYPE_TO_CUMM_DTYPE[obj.dtype]
        code.raw(f"""
        {dtype}* {name} = {ten_name}.data_ptr<{dtype}>();
        """)
        return ten_name

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        if isinstance(obj, np.ndarray):
            return tv.from_numpy(obj)
        return obj

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        return "tv::Tensor"


_DEFAULT_PLUGINS: Dict[str, InlineBuilderPlugin] = {
    "numpy.ndarray": CummTensorKernelPlugin(),
    "cumm.core_cc.tensorview_bind.Tensor": CummTensorKernelPlugin(),
}

_CUMM_KERNEL_1D_NUM_NAME = "_cumm_pccm_inline_num"
_CUMM_KERNEL_1D_DEVICE_NAME = "_cumm_pccm_inline_device"


class CUDAMode(enum.Enum):
    Kernel1D = "Kernel1D"
    Kernel1DCUDA = "Kernel1DCUDA"


class TensorViewKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("tensorview/parallel/all.h")
        self.build_meta.add_cflags("nvcc", "--extended-lambda")


class CUDAInlineBuilder(InlineBuilder):
    def __init__(self,
                 deps: List[Type[pccm.Class]],
                 plugins: Optional[Dict[str, InlineBuilderPlugin]] = None,
                 index_name: str = "i") -> None:
        if plugins is None:
            plugins = _DEFAULT_PLUGINS
        deps.append(TensorViewKernel)
        super().__init__(deps=deps, plugins=plugins)
        self.index_name = index_name

    def handle_container_code(self, code_str: str, code: FunctionCode,
                              arg: Optional[CUDAMode]):
        if arg is None:
            return code.raw(code_str)
        name = "tv::kernel_1d_map"
        if arg == CUDAMode.Kernel1DCUDA:
            name = "tv::kernel_1d_map_cuda"

        prefix = f"{name}({_CUMM_KERNEL_1D_DEVICE_NAME}, {_CUMM_KERNEL_1D_NUM_NAME}, [=]TV_GPU_LAMBDA(size_t {self.index_name}){{"
        with code.block(prefix, start="", end="});"):
            code.raw(code_str)

    def kernel_1d(self, name: str, num: int, device: int,
                  code: Union[str, FunctionCode]):
        additional_args = {
            _CUMM_KERNEL_1D_NUM_NAME: num,
            _CUMM_KERNEL_1D_DEVICE_NAME: device,
        }
        return self.inline(name,
                           code,
                           ".cu",
                           additional_args,
                           _frame_cnt=2,
                           user_arg=CUDAMode.Kernel1D)

    def kernel_1d_cuda(self, name: str, num: int, device: int,
                       code: Union[str, FunctionCode]):
        additional_args = {
            _CUMM_KERNEL_1D_NUM_NAME: num,
            _CUMM_KERNEL_1D_DEVICE_NAME: device,
        }
        return self.inline(name,
                           code,
                           ".cu",
                           additional_args,
                           _frame_cnt=2,
                           user_arg=CUDAMode.Kernel1DCUDA)

    def kernel_1d_cpu(self, name: str, num: int, device: int,
                      code: Union[str, FunctionCode]):
        additional_args = {
            _CUMM_KERNEL_1D_NUM_NAME: num,
            _CUMM_KERNEL_1D_DEVICE_NAME: device,
        }
        return self.inline(name,
                           code,
                           ".cc",
                           additional_args,
                           _frame_cnt=2,
                           user_arg=CUDAMode.Kernel1D)


def main():
    import numpy as np

    # print(ast.dump(tree))
    aa = np.array([1], dtype=np.float32)
    a = [aa, aa]
    # print(nested_type_analysis(aa.shape))
    b = InlineBuilder(
        [TensorView], {
            "numpy.ndarray": CummTensorPlugin(),
            "cumm.core_cc.tensorview_bind.Tensor": CummTensorPlugin()
        })
    for i in range(10):
        b.inline(
            "just_a_name", f"""
        // pybind::array
        // tv::Tensor
        float* ptr = $a[0].data_ptr<float>();
        float* ptr2 = $a[1].data_ptr<float>();

        ptr[0] += 1;
        ptr2[0] += 1;
        """)
        print(aa[0])
    cub = CUDAInlineBuilder([])

    # c = tv.zeros([1000], tv.float32, 0)
    c = tv.from_numpy(aa)
    cub.kernel_1d("just_a_kernel", c.dim(0), c.device, f"""
    $c[i] = 1;
    """)
    print(aa)


if __name__ == "__main__":
    main()
