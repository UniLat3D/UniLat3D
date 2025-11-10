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
""" nvrtc inliner with capture.

don't forget to install a highlight extension to show captures in string!

my vscode highlight extension config:

    "highlight.regexes": {
        "(\\$)([_a-zA-Z][_a-zA-Z0-9]*)": {
            "filterFileRegex": ".*.py$",
            "decorations": [
                {
                    "color": "#c678dd",
                    "fontWeight": "bold"
                },
                {
                    "color": "#e06c75",
                    "fontWeight": "bold"
                }

            ]
        },
        "(\\$)(\\()(.*?)(\\))": {
            "filterFileRegex": ".*.py$",
            "decorations": [
                {
                    "color": "#c678dd",
                    "fontWeight": "bold"
                },
                {
                    "color": "#fbf755",
                    "fontWeight": "bold"
                },
                {
                    "color": "#e06c75",
                    "fontWeight": "bold"
                },
                {
                    "color": "#fbf755",
                    "fontWeight": "bold"
                },
            ]
        },

    },


"""

import abc
import contextlib
import contextvars
import re
import time

import pccm
from pccm.builder.inliner import InlineBuilder, InlineBuilderPlugin, PCCM_INLINE_NAMESPACE, PCCM_INLINE_FUNCTION_NAME, PCCM_INLINE_FUNCTION_NAME_FORMAT, get_base_type_string
from cumm.nvrtc import CummLLVMModule, CummMetalModule, CummNVRTCModule, CummNVRTCModuleBase, create_nvrtc_code
from pathlib import Path
from cumm.common import TensorViewKernel
import enum
from pccm.utils import get_qualname_of_type
from typing import Any, Callable, ContextManager, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
from cumm import dtypes, tensorview as tv
from cumm.common import TensorViewNVRTC, GemmBasic, TensorViewViewClass
from cumm.gemm.codeops import div_up
from cumm.tensorview import nullcontext
from ccimport import compat

_TORCH_MPS_SYNC: Optional[Callable[[], Any]] = None 

_TORCH_DTYPE_TO_TV: Dict[Any, int] = {}
_TORCH_DTYPE_TO_TV_STR: Dict[Any, str] = {}
TORCH_TENSOR_NAME = "torch.Tensor"
TORCH_PARAMETER_TENSOR_NAME = "torch.nn.parameter.Parameter"



class InlinerKernelLaunchContext:
    pass

INLINER_KERNEL_CTX: contextvars.ContextVar[Optional[InlinerKernelLaunchContext]] = contextvars.ContextVar("InlinerKernelLaunchContext", default=None)

@contextlib.contextmanager
def _enter_inliner_kernel_ctx(ctx: InlinerKernelLaunchContext):
    token = INLINER_KERNEL_CTX.set(ctx)
    try:
        yield ctx
    finally:
        INLINER_KERNEL_CTX.reset(token)

def _has_inliner_kernel_ctx():
    return INLINER_KERNEL_CTX.get() is not None

class LaunchParam:
    def __init__(self,
                 blocks: Tuple[int, ...],
                 threads: Tuple[int, ...],
                 smem: int = 0,
                 stream: int = 0) -> None:
        self.blocks = list(blocks)
        self.threads = list(threads)
        assert len(blocks) == 3
        assert len(threads) == 3
        self.smem = smem
        self.stream = stream


def _cached_get_torch_dtype_to_tv():
    import torch
    if not _TORCH_DTYPE_TO_TV:
        _TORCH_DTYPE_TO_TV.update({
            torch.float32: tv.float32,
            torch.float64: tv.float64,
            torch.float16: tv.float16,
            torch.int32: tv.int32,
            torch.int64: tv.int64,
            torch.int8: tv.int8,
            torch.int16: tv.int16,
            torch.uint8: tv.uint8,
            torch.bfloat16: tv.bfloat16,
        })
        torch_version = torch.__version__.split(".")
        major_version = int(torch_version[0])
        minor_version = int(torch_version[1])
        if (major_version, minor_version) >= (2, 3):
            _TORCH_DTYPE_TO_TV.update({
                torch.uint16: tv.uint16,
                torch.uint32: tv.uint32,
                torch.uint64: tv.uint64,
            })
    return _TORCH_DTYPE_TO_TV

def _cached_get_torch_dtype_to_tv_str():
    import torch
    if not _TORCH_DTYPE_TO_TV_STR:
        _TORCH_DTYPE_TO_TV_STR.update({
            torch.float32: "tv.float32",
            torch.float64: "tv.float64",
            torch.float16: "tv.float16",
            torch.int32: "tv.int32",
            torch.int64: "tv.int64",
            torch.int8: "tv.int8",
            torch.int16: "tv.int16",
            torch.uint8: "tv.uint8",
            torch.bfloat16: "tv.bfloat16",
        })
        torch_version = torch.__version__.split(".")
        major_version = int(torch_version[0])
        minor_version = int(torch_version[1])
        if (major_version, minor_version) >= (2, 3):
            _TORCH_DTYPE_TO_TV_STR.update({
                torch.uint16: "tv.uint16",
                torch.uint32: "tv.uint32",
                torch.uint64: "tv.uint64",
            })
    return _TORCH_DTYPE_TO_TV_STR

RESERVED_NAMES = set([
    "threadPositionInGrid",
    "threadGroupPositionInGrid",
])

class CUDAMode(enum.Enum):
    Kernel1D = "Kernel1D"
    Kernel2D = "Kernel2D"
    Kernel3D = "Kernel3D"

    KernelRaw = "KernelRaw"


def torch_tensor_to_tv(ten,
                       dtype: Optional[int] = None,
                       shape: Optional[List[int]] = None,
                       to_const: bool = False):
    # assert ten.is_contiguous(), "must be contiguous tensor"
    device = ten.device
    if device.type == "cpu":
        tv_device = -1
    elif device.type == "cuda":
        tv_device = 0
    elif device.type == "mps":
        tv_device = 0
    else:
        raise NotImplementedError
    if device.type == "mps":
        # mps data ptr is MTLBuffer, not real ptr
        # if we use ten.data_ptr(), the result will be 
        # MTLBuffer + data_offset, which will cause
        # segfault.
        ptr = ten.untyped_storage().data_ptr()
        offset = ten.storage_offset()
    else:
        ptr = ten.data_ptr()
        offset = 0
    if shape is None:
        shape = list(ten.shape)
    if dtype is None:
        dtype = _cached_get_torch_dtype_to_tv()[ten.dtype]
    if to_const:
        return tv.from_const_blob_strided(ptr, shape, list(ten.stride()), dtype,
                                    tv_device, offset)
    return tv.from_blob_strided(ptr, shape, list(ten.stride()), dtype,
                                tv_device, offset)

@contextlib.contextmanager
def measure_and_print_torch(name: str = "CUDATimer", *, stream: int = 0, out: Optional[List[float]] = None, enable: bool = True):
    if not enable:
        yield
    else:
        import torch
        if compat.IsAppleSiliconMacOs:
            torch.mps.synchronize()
            t = time.time()
            # start_ev = torch.mps.Event(enable_timing=True)
            # end_ev = torch.mps.Event(enable_timing=True)
            # start_ev.record()
            yield 
            # end_ev.record()
            torch.mps.synchronize()
            # TODO sync event will hang
            # start_ev.synchronize()
            # end_ev.synchronize()
            # duration = start_ev.elapsed_time(end_ev)
            duration = (time.time() - t) * 1000
            print(f"{name} duration: {duration} ms")
        else:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

            start_ev.record(torch.cuda.default_stream())
            yield 
            end_ev.record(torch.cuda.default_stream())
            start_ev.synchronize()
            end_ev.synchronize()
            duration = start_ev.elapsed_time(end_ev)
            print(f"{name} duration: {duration} ms")

def get_current_stream():
    import torch
    if compat.IsAppleSiliconMacOs:
        return 0
    return torch.cuda.current_stream().cuda_stream


class TVTensorPlugin(InlineBuilderPlugin):
    QualifiedName = get_qualname_of_type(tv.Tensor)

    def handle_captured_type(
            self,
            name: str,
            code: pccm.FunctionCode,
            obj: Any,
            user_arg: Optional[Any] = None) -> Optional[Tuple[str, str]]:
        return None

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        qname = get_qualname_of_type(type(obj))
        if qname == TORCH_TENSOR_NAME or qname == TORCH_PARAMETER_TENSOR_NAME:
            return torch_tensor_to_tv(obj)
        return obj

    def type_conversion_code(self, obj, src_name: str, tgt_name: str, user_arg: Optional[Any] = None) -> Optional[Tuple[str, List[str]]]:
        qname = get_qualname_of_type(type(obj))
        assert user_arg is not None and isinstance(user_arg, _NVRTCInlineParams)
        if user_arg.unchecked_mode:
            if qname == TORCH_TENSOR_NAME or qname == TORCH_PARAMETER_TENSOR_NAME:
                res: List[str] = []
                device = obj.device
                if device.type == "mps":
                    # mps data ptr is MTLBuffer, not real ptr
                    # if we use ten.data_ptr(), the result will be 
                    # MTLBuffer + data_offset, which will cause
                    # segfault.
                    tv_dtype_str = _cached_get_torch_dtype_to_tv_str()[obj.dtype]
                    itemsize = obj.itemsize
                    # storage offset isn't byte offset, so we need to multiply itemsize
                    # when use raw device pointer.
                    res.extend([
                        f"__{tgt_name}_tmp0 = {src_name}",
                        f"assert __{tgt_name}_tmp0.dtype == {obj.dtype}",
                        f"{tgt_name} = (EMPTY_TENSOR, kDevicePointer, ",
                        f"    __{tgt_name}_tmp0.untyped_storage().data_ptr(), __{tgt_name}_tmp0.storage_offset() * {itemsize})",
                    ])
                else:
                    res.extend([
                        f"__{tgt_name}_tmp0 = {src_name}",
                        f"assert __{tgt_name}_tmp0.dtype == {obj.dtype}",
                        f"{tgt_name} = (EMPTY_TENSOR, kDevicePointer, ",
                        f"    __{tgt_name}_tmp0.data_ptr(), 0)",
                    ])
                return "\n".join(res), [
                    "import torch",
                    "from cumm import tensorview as tv",
                    "kDevicePointer = tv._NVRTCModule.kDevicePointer",
                    # "kTensor = tv._NVRTCModule.kTensor",
                    "EMPTY_TENSOR = tv.Tensor()",
                ]
            else:
                assert isinstance(obj, tv.Tensor)
                if compat.IsAppleSiliconMacOs:
                    res = [
                        f"__{tgt_name}_tmp0 = {src_name}",
                        f"assert __{tgt_name}_tmp0.dtype == {obj.dtype}",
                        f"{tgt_name} = (__{tgt_name}_tmp0, kTensor, ",
                        f"    0, 0)",
                    ]
                    return "\n".join(res), [
                        "from cumm import tensorview as tv",
                        "kTensor = tv._NVRTCModule.kTensor",
                    ]
                else:
                    res = [
                        f"__{tgt_name}_tmp0 = {src_name}",
                        f"assert __{tgt_name}_tmp0.dtype == {obj.dtype}",
                        f"{tgt_name} = (EMPTY_TENSOR, kDevicePointer, ",
                        f"    __{tgt_name}_tmp0.byte_pointer(), 0)",
                    ]
                    return "\n".join(res), [
                        "from cumm import tensorview as tv",
                        "kDevicePointer = tv._NVRTCModule.kDevicePointer",
                        "EMPTY_TENSOR = tv.Tensor()",
                    ]
        else:
            if qname == TORCH_TENSOR_NAME or qname == TORCH_PARAMETER_TENSOR_NAME:
                res: List[str] = []
                device = obj.device
                tv_dtype_str = _cached_get_torch_dtype_to_tv_str()[obj.dtype]
                if device.type == "mps":
                    # mps data ptr is MTLBuffer, not real ptr
                    # if we use ten.data_ptr(), the result will be 
                    # MTLBuffer + data_offset, which will cause
                    # segfault.
                    res.extend([
                        f"__{tgt_name}_tmp0 = {src_name}",
                        f"assert __{tgt_name}_tmp0.dtype == {obj.dtype}",
                        f"{tgt_name} = tv.from_blob(__{tgt_name}_tmp0.untyped_storage().data_ptr(), ",
                        f"    list(__{tgt_name}_tmp0.shape), {tv_dtype_str}, 0, __{tgt_name}_tmp0.storage_offset())"
                    ])
                else:
                    res.extend([
                        f"__{tgt_name}_tmp0 = {src_name}",
                        f"assert __{tgt_name}_tmp0.dtype == {obj.dtype}",
                        f"{tgt_name} = tv.from_blob(__{tgt_name}_tmp0.data_ptr(), ",
                        f"    list(__{tgt_name}_tmp0.shape), {tv_dtype_str}, 0)"
                    ])
                return "\n".join(res), [
                    "import torch",
                    "from cumm import tensorview as tv",
                ]
            return f"{tgt_name} = {src_name}", []

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        qname = get_qualname_of_type(type(obj))
        if qname == TORCH_TENSOR_NAME or qname == TORCH_PARAMETER_TENSOR_NAME:
            # print("??????", obj.shape, obj.dtype)
            obj = torch_tensor_to_tv(obj)
        prefix = ""
        if obj.is_readonly():
            prefix = "const "
        if user_arg is not None and isinstance(user_arg, _NVRTCInlineParams):
            if user_arg.capture_tensor_as_tview:
                return f"tv::TensorView<{prefix}{dtypes.get_dtype_from_tvdtype(obj.dtype, True)}, {obj.ndim}>"
        return f"{prefix}{dtypes.get_dtype_from_tvdtype(obj.dtype, True)}*"



class BuiltinTypePlugin(InlineBuilderPlugin):
    QualifiedName = get_qualname_of_type(tv.Tensor)

    def handle_captured_type(
            self,
            name: str,
            code: pccm.FunctionCode,
            obj: Any,
            user_arg: Optional[Any] = None) -> Optional[Tuple[str, str]]:
        return None

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        return obj

    def type_conversion_code(self, obj, src_name: str, tgt_name: str, user_arg: Optional[Any] = None) -> Optional[Tuple[str, List[str]]]:
        res: List[str] = []
        assert isinstance(obj, (int, float, bool))
        obj_type_str = get_qualname_of_type(type(obj))
        if isinstance(obj, int):
            tv_dtype = tv.int64 
        elif isinstance(obj, float):
            tv_dtype = tv.float32 
        elif isinstance(obj, bool):
            tv_dtype = tv.uint8
        else:
            raise NotImplementedError
        assert user_arg is not None and isinstance(user_arg, _NVRTCInlineParams)
        if user_arg.unchecked_mode:
            if isinstance(obj, bool):
                # bools are func constants (only used in apple metal)
                res.extend([
                    f"__{tgt_name}_tmp0 = {src_name}",
                    f"assert isinstance(__{tgt_name}_tmp0, bool)",
                    f"{tgt_name} = (tv.full([1], __{tgt_name}_tmp0, tv.uint8), kConstant, ",
                    f"    0, 0)",
                ])
            else:
                res.extend([
                    f"__{tgt_name}_tmp0 = {src_name}",
                    # f"assert isinstance(__{tgt_name}_tmp0, {obj_type_str})",
                    f"{tgt_name} = (tv.full([1], __{tgt_name}_tmp0, {tv_dtype}), kScalar, ",
                    f"    0, 0)",
                ])
            return "\n".join(res), [
                "from cumm import tensorview as tv",
                "kScalar = tv._NVRTCModule.kScalar",
                "kConstant = tv._NVRTCModule.kConstant"
            ]
        else:
            return f"{tgt_name} = {src_name}", [] 

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        raise NotImplementedError("should not be called since std type is directly handled except compiled mode")

_NPDTYPE_TO_LIST_TYPE_STR: Dict[np.dtype, str] = {
    np.dtype(np.float16): "__half",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.int8): "int8_t",
    np.dtype(np.int16): "int16_t",
    np.dtype(np.int32): "int32_t",
    np.dtype(np.int64): "int64_t",
    np.dtype(np.uint8): "uint8_t",
    np.dtype(np.uint16): "uint16_t",
    np.dtype(np.uint32): "uint32_t",
    np.dtype(np.uint64): "uint64_t",
    np.dtype(np.bool_): "bool",
}

_NPDTYPE_TO_METAL_STR: Dict[np.dtype, str] = {
    np.dtype(np.float16): "half",
    np.dtype(np.float32): "float",
    np.dtype(np.int8): "char",
    np.dtype(np.int16): "short",
    np.dtype(np.int32): "int",
    np.dtype(np.int64): "long",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.uint32): "uint",
    np.dtype(np.uint64): "ulong",
    np.dtype(np.bool_): "bool",
}


class NumpyPlugin(InlineBuilderPlugin):
    """for nvrtc, we capture np.ndarray as tv::array, so the shape must fixed and small.
    """
    QualifiedName = get_qualname_of_type(np.ndarray)

    def handle_captured_type(self,
                             name: str,
                             code: pccm.FunctionCode,
                             obj: Any,
                             user_arg: Optional[Any] = None) -> Optional[str]:
        return

    def type_conversion(self, obj: np.ndarray, user_arg: Optional[Any] = None):
        # if isinstance(user_arg, _NVRTCInlineParams):
        #     if user_arg.is_cpu:
        #         return obj.reshape(-1).tolist()
        assert obj.nbytes <= 256, "we only support capture small numpy which will be passed by value to kernel"
        return obj

    def type_conversion_code(self, obj, src_name: str, tgt_name: str, user_arg: Optional[Any] = None) -> Optional[Tuple[str, List[str]]]:
        assert isinstance(obj, np.ndarray)
        assert user_arg is not None and isinstance(user_arg, _NVRTCInlineParams)
        tv_dtype = dtypes.get_dtype_from_npdtype(obj.dtype).tv_dtype
        if user_arg.unchecked_mode:
            res = [
                f"__{tgt_name}_tmp0 = {src_name}",
                f"assert __{tgt_name}_tmp0.dtype == numpy.{obj.dtype}",
                f"__{tgt_name}_tmp1 = tv.empty(__{tgt_name}_tmp0.shape, {tv_dtype})",
                f"__{tgt_name}_tmp1.numpy_view()[:] = __{tgt_name}_tmp0",

                f"{tgt_name} = (__{tgt_name}_tmp1, kScalar, ",
                f"    0, 0)",
            ]
            return "\n".join(res), [
                "from cumm import tensorview as tv",
                "import numpy",
                "kScalar = tv._NVRTCModule.kScalar",

            ]
        else:
            return f"{tgt_name} = {src_name}", [] 

        return f"{tgt_name} = {src_name}", []

    @staticmethod
    def get_cpp_type_static(
                     obj: np.ndarray,
                     user_arg: Optional[Any] = None) -> Union[str, Tuple[str, int]]:
        ndim = obj.ndim
        dtype = obj.dtype
        cpp_type = _NPDTYPE_TO_LIST_TYPE_STR[dtype]
        # if isinstance(user_arg, _NVRTCInlineParams):
        #     if user_arg.is_cpu:
        #         return cpp_type, obj.size
        array_type = ""
        array_type += "tv::array<" * ndim
        array_type += f"{cpp_type}, "
        shape_rev = obj.shape[::-1]
        array_type += ">, ".join(map(str, shape_rev))
        res = array_type + ">"
        return res

    def get_cpp_type(self,
                     obj: np.ndarray,
                     user_arg: Optional[Any] = None) -> Union[str, Tuple[str, int]]:
        
        return self.get_cpp_type_static(obj, user_arg)


class _NVRTCInlineParams:
    def __init__(self,
                 mode: CUDAMode,
                 launch: tv.LaunchParam,
                 verbose: bool = False,
                 verbose_path: str = "",
                 measure_time: bool = False,
                 is_cpu: bool = False,
                 capture_tensor_as_tview: bool = False,
                 perf_context: Optional[ContextManager] = None,
                 run_in_process: bool = False,
                 unchecked_mode: bool = True) -> None:
        self.mode = mode
        self.launch = launch
        self.verbose = verbose
        self.verbose_path = verbose_path
        self.measure_time = measure_time
        self.is_cpu = is_cpu
        self.capture_tensor_as_tview = capture_tensor_as_tview
        self.perf_context = perf_context
        self.run_in_process = run_in_process
        self.unchecked_mode = unchecked_mode


_NVRTC_FUNC_NAME = f"{PCCM_INLINE_NAMESPACE}::{PCCM_INLINE_FUNCTION_NAME}"
_NVRTC_FUNC_NAME_FORMAT = f"{PCCM_INLINE_NAMESPACE}::{PCCM_INLINE_FUNCTION_NAME_FORMAT}"

_DEFAULT_KERNEL_PLUGINS: Dict[str, InlineBuilderPlugin] = {
    "numpy.ndarray": NumpyPlugin(),
    TVTensorPlugin.QualifiedName: TVTensorPlugin(),
    TORCH_TENSOR_NAME: TVTensorPlugin(),
    TORCH_PARAMETER_TENSOR_NAME: TVTensorPlugin(),
    get_base_type_string(True)[0]: BuiltinTypePlugin(),
    get_base_type_string(1)[0]: BuiltinTypePlugin(),
    get_base_type_string(1.0)[0]: BuiltinTypePlugin(),
}

_CUMM_KERNEL_1D_SIZE_NAME = "_cumm_pccm_inline_size"

def identity():
    return

def _default_mps_sync_func():
    global _TORCH_MPS_SYNC
    if _TORCH_MPS_SYNC is None:
        try:
            import torch 
            _TORCH_MPS_SYNC = torch.mps.synchronize 
        except:
            _TORCH_MPS_SYNC = identity 
    _TORCH_MPS_SYNC()

class MPSContextBase(abc.ABC):
    @abc.abstractmethod
    def get_command_buffer(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dispatch_queue(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def commit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def flush_command_encoder(self):
        raise NotImplementedError


    # @abc.abstractmethod
    # def synchronize(self):
    #     raise NotImplementedError

class NVRTCInlineBuilder(InlineBuilder):
    def __init__(
            self,
            deps: List[Type[pccm.Class]],
            plugins: Optional[Dict[str, InlineBuilderPlugin]] = None,
            index_name: str = "i",
            root: Optional[Path] = None,
            build_root: Optional[Path] = None,
            build_kwargs: Optional[Dict[str, Any]] = None,
            param_deps: Optional[List[pccm.ParameterizedClass]] = None,
            measure_build_time: bool = False,
            reload_when_code_change: bool = False,
            remote_addr: str = "",
            default_deps: Optional[List[Type[pccm.Class]]] = None,
            std: str = "c++14",
            context: Optional[tv.Context] = None,
            mps_sync_func: Callable[[], None] = _default_mps_sync_func,
            mps_context: Optional[MPSContextBase] = None):
        if plugins is None:
            plugins = _DEFAULT_KERNEL_PLUGINS
        if default_deps is None:
            deps.extend([TensorViewNVRTC, ])
            if not compat.InMacOS:
                deps.append(GemmBasic)
        else:
            deps.extend(default_deps)
        if build_kwargs is None:
            build_kwargs = {}
        super().__init__(deps=deps,
                         plugins=plugins,
                         build_kwargs=build_kwargs,
                         root=root,
                         build_root=build_root,
                         param_deps=param_deps,
                         reload_when_code_change=reload_when_code_change)
        self.index_name = index_name
        self.cpp_std = std
        self.maximum_1d_threads = 512
        self.measure_build_time = measure_build_time
        self._remote_addr = remote_addr
        self._mps_sync_func = mps_sync_func
        self._mps_context = mps_context
        self._nvrtc_fn_name_cache: dict[str, str] = {}
        self.ctx = tv.Context() if context is None or mps_context is not None else context
        if compat.InMacOS and not self.ctx.has_apple_metal_context():
            if self._mps_context is None:
                self.ctx.create_apple_metal_context()
            else:
                self.ctx.create_or_update_metal_context_from_blob(self._mps_context.get_command_buffer(),
                                                                self._mps_context.get_dispatch_queue()) 
    
    def synchronize(self):
        self.ctx.synchronize()

    @contextlib.contextmanager
    def enter_inliner_scope(self):
        if compat.IsAppleSiliconMacOs and self._mps_context is not None:
            # if we use context from pytorch, we don't need to perform sync
            # because kernel will run on the same context.
            # the only problem is we must update the context after any pytorch operation.
            self.ctx.create_or_update_metal_context_from_blob(self._mps_context.get_command_buffer(),
                                                            self._mps_context.get_dispatch_queue()) 
            # pytorch mps kernels won't clear command encoding, so we need to flush it.
            self._mps_context.flush_command_encoder()
            yield
        elif not compat.IsAppleSiliconMacOs or _has_inliner_kernel_ctx():
            yield 
        else:
            # sync all mps operations from other lib
            self._mps_sync_func()
            # you shouldn't call any pytorch operation in this scope.
            try:
                with _enter_inliner_kernel_ctx(InlinerKernelLaunchContext()):
                    yield 
            finally:
                # sync all inliner operation
                self.synchronize()

    def synchronize_if_apple_metal(self):
        """This method is used for mixed torch and cumm inline cuda code.
        cuda have a default stream and context but metal not, at least we
        can't acquire that context in pytorch. this means
        we have two metal command queue in single process, metal ops exec 
        order on two queue is undefined. so if you mix torch and inline cuda
        code, run sync before/after any torch operation. you can use 
        `inliner.synchronize_if_apple_metal` to achieve this.
        """
        if compat.IsAppleSiliconMacOs:
            self.ctx.synchronize()

    def get_nvrtc_module(self, name: str) -> Optional[Union[CummNVRTCModule, CummMetalModule]]:
        for k, v in self.modules.items():
            if name == k[1]:
                return v.func
        return None
    
    def get_nvrtc_kernel_attrs(self, name: str) -> Dict[str, int]:
        nvrtc_mod = self.get_nvrtc_module(name)
        assert nvrtc_mod is not None
        return nvrtc_mod.get_kernel_attrs(nvrtc_mod.get_lowered_name(self._get_nvrtc_inline_func_name_for_debug(name)))

    def get_save_root(self,
                      path: Path,
                      root: Optional[Path] = None,
                      build_root: Optional[Path] = None):
        # override get_save_root because we don't need it for nvrtc.
        return Path()

    def get_base_class(self):
        return pccm.Class()

    def handle_container_code(self, code_str: str, code: pccm.FunctionCode,
                              arg: Optional[_NVRTCInlineParams]):
        is_cpu = False
        meta = pccm.cuda.CudaGlobalFunctionMeta(attrs=["__global__"])
        if arg is not None:
            is_cpu = arg.is_cpu
        if is_cpu:
            meta = pccm.cuda.ExternalFunctionMeta()
        if not is_cpu and compat.InMacOS:
            meta = pccm.cuda.CudaGlobalFunctionMeta(attrs=["kernel"])
        if arg is None:
            code.raw(code_str)
            return meta
        if not is_cpu and compat.InMacOS:
            # metal 
            capture_nparray_as_func_constant: bool = False
            func_const_cnt = 0
            new_args: List[pccm.Argument] = []
            func_constants: List[str] = []
            for prev_arg in code.arguments:
                if isinstance(prev_arg.userdata, np.ndarray):
                    if not capture_nparray_as_func_constant:
                        new_name = f"__cumm_metal_arg_{prev_arg.name}"
                        cpp_type_arr = NumpyPlugin.get_cpp_type_static(prev_arg.userdata)
                        prev_arg.type_str = f"constant {cpp_type_arr} &"
                        new_args.append(prev_arg)
                    else:
                        arr = prev_arg.userdata
                        if arr.ndim == 1:
                            arr = arr[np.newaxis]
                        vec_cnt = arr.shape[1]
                        names = []
                        cpp_type_scalar = _NPDTYPE_TO_LIST_TYPE_STR[prev_arg.userdata.dtype]
                        for i in range(arr.shape[0]):
                            new_name = f"__cumm_varg_{prev_arg.name}_{i}"
                            # names.append(new_name)
                            metal_type_base = _NPDTYPE_TO_METAL_STR[arr.dtype]
                            metal_type = f"{metal_type_base}{arr.shape[1]}"
                            cur_arg = prev_arg.copy()
                            cur_arg.name = new_name
                            cur_arg.type_str = metal_type
                            func_constants.append(f"constant {metal_type} {new_name} [[function_constant({func_const_cnt})]];")
                            names.append(f"tv::array<{cpp_type_scalar}, {vec_cnt}>{{" + ", ".join(f"{new_name}[{j}]" for j in range(arr.shape[1])) + "}")
                            func_const_cnt += 1
                        cpp_type_arr = NumpyPlugin.get_cpp_type_static(prev_arg.userdata)
                        arr_name = f", ".join(names)
                        code.raw(f"thread {cpp_type_arr} {prev_arg.name}{{{arr_name}}};")

                elif get_qualname_of_type(type(prev_arg.userdata)) == TORCH_TENSOR_NAME:
                    prev_arg.type_str = f"device {prev_arg.type_str}"
                    new_args.append(prev_arg)
                elif isinstance(prev_arg.userdata, tv.Tensor):
                    prev_arg.type_str = f"device {prev_arg.type_str}"
                    new_args.append(prev_arg)
                elif isinstance(prev_arg.userdata, bool):
                    func_constants.append(f"constant bool {prev_arg.name} [[function_constant({func_const_cnt})]];")
                    func_const_cnt += 1
                else:
                    assert isinstance(prev_arg.userdata, (int, float, np.floating, np.integer)), f"{prev_arg.name}, {type(prev_arg.userdata)}"
                    # dtype = "int64_t"
                    # if isinstance(prev_arg.userdata, (float, np.floating)):
                    #     dtype = "float"
                    prev_arg.type_str = f"constant {prev_arg.type_str}&"
                    new_args.append(prev_arg)

                    # func_constants.append(f"constant {dtype} {prev_arg.name} [[function_constant({func_const_cnt})]];")
                    # func_const_cnt += 1
            if arg.mode != CUDAMode.KernelRaw:
                code.arguments = new_args + [
                    pccm.Argument(self.index_name, "uint32_t", attributes=["thread_position_in_grid"])
                ]
            else:
                code.arguments = new_args + [
                    pccm.Argument("threadPositionInGrid", "uint3", attributes=["thread_position_in_grid"]),
                    pccm.Argument("threadgroupPositionInGrid", "uint3", attributes=["threadgroup_position_in_grid"]),
                    pccm.Argument("threadPositionInThreadgroup", "uint3", attributes=["thread_position_in_threadgroup"]),
                ]
            code.code_before_func_def = "\n".join(func_constants)
        trycatch_ctx = contextlib.nullcontext()
        if is_cpu:
            code.raw(f"""
            int __pccm_error_code = 0;
            """)
            trycatch_ctx = code.block("try", end=f"""
            }}catch (const std::exception& e){{
                tv::printf2("LLVM Inline Function Exception!. Error:", e.what());
                __pccm_error_code = 1;
            }}
            """)
        with trycatch_ctx:
            if arg.mode == CUDAMode.KernelRaw:
                code.raw(code_str)
            else:
                if not is_cpu:
                    if compat.InMacOS:
                        code.raw(code_str)
                    else:
                        with code.for_(
                                f"auto {self.index_name} : tv::KernelLoopX<int>({_CUMM_KERNEL_1D_SIZE_NAME})"
                        ):
                            code.raw(code_str)
                else:
                    with code.for_(
                            f"size_t i = 0; i <{_CUMM_KERNEL_1D_SIZE_NAME}; ++i"
                    ):
                        code.raw(code_str)
        if is_cpu:
            code.raw("return __pccm_error_code;")
            code.ret("int")
        return meta

    def build(self,
              pccm_cls: pccm.Class,
              mod_root: Path,
              name: str,
              timeout: float,
              user_arg: Optional[_NVRTCInlineParams] = None):
        verbose = False
        verbose_path = ""
        if user_arg is not None:
            verbose = user_arg.verbose
            verbose_path = user_arg.verbose_path
        ctx = nullcontext()
        if self.measure_build_time:
            ctx = tv.measure_and_print(f"{name} nvrtc build time")
        # with tv.measure_and_print("INLINE"):
        params = create_nvrtc_code([pccm_cls], std=self.cpp_std)
        is_cpu = False
        if user_arg is not None:
            is_cpu = user_arg.is_cpu
        if self._remote_addr != "":
            # this should be used only you want to debug
            # different cuda version.
            import tensorpc
            with tensorpc.RemoteManager(self._remote_addr) as robj:
                mod = robj.remote_call("NVRTCCompiler.compile_nvrtc", params)
        else:
            with ctx:
                if is_cpu:
                    if not compat.InLinux and not compat.InMacOS:
                        raise NotImplementedError("cpu jit only support linux")
                    mod = CummLLVMModule([pccm_cls],
                                         verbose=verbose,
                                         verbose_path=verbose_path,
                                         std=self.cpp_std)
                else:
                    if compat.InMacOS:
                        mod = CummMetalModule([pccm_cls],
                                            verbose=verbose,
                                            verbose_path=verbose_path,
                                            std=self.cpp_std)
                    else:
                        if verbose:
                            mod = CummNVRTCModule([pccm_cls],
                                                verbose=verbose,
                                                verbose_path=verbose_path,
                                                std=self.cpp_std)
                        else:
                            mod = CummNVRTCModuleBase.from_params(params)

        return mod

    def _get_nvrtc_inline_func_name_for_debug(self, name: str):
        return _NVRTC_FUNC_NAME_FORMAT.format(re.sub('[^0-9a-zA-Z]', '_', name))

    def run_func(self,
                name: str,
                 func: Union[CummNVRTCModuleBase, CummMetalModule],
                 *args,
                 user_args: Optional[_NVRTCInlineParams] = None):
        assert user_args is not None
        if name in self._nvrtc_fn_name_cache:
            real_name = self._nvrtc_fn_name_cache[name]
        else:
            real_name = self._get_nvrtc_inline_func_name_for_debug(name)
            self._nvrtc_fn_name_cache[name] = real_name
        if isinstance(func, CummMetalModule):
            launch = user_args.launch.copy()
            if launch.ctx is None or not launch.ctx.has_apple_metal_context():
                launch.ctx = self.ctx
            is_kernel_raw = user_args.mode == CUDAMode.KernelRaw
            with self.enter_inliner_scope():
                if user_args.unchecked_mode:
                    res = func.run_kernel_unchecked(real_name, launch, args, use_nonuniform_threadgroup=not is_kernel_raw)
                else:
                    res = func.run_kernel(real_name, launch, *args, perf_context=user_args.perf_context, use_nonuniform_threadgroup=not is_kernel_raw)

                if self._mps_context is not None:
                    self._mps_context.commit()
                return res
        else:
            if user_args.run_in_process:
                func.run_kernel_in_spawn_process(real_name, user_args.launch, *args)
            else:
                if user_args.unchecked_mode:
                    func.run_kernel_unchecked(real_name, user_args.launch, args)
                else:
                    func.run_kernel(real_name, user_args.launch, *args, perf_context=user_args.perf_context)
        return 

    def kernel_raw(self,
                   name: str,
                   param: tv.LaunchParam,
                   code: Union[str, pccm.FunctionCode],
                   verbose_path: str = "",
                   disable_cache: bool = False,
                   perf_context: Optional[ContextManager] = None,
                   run_in_process: bool = False,
                   additional_vars: Optional[Dict[str, Any]] = None,
                   *,
                   _frame_cnt: int = 2):
        verbose = verbose_path != ""
        user_arg = _NVRTCInlineParams(CUDAMode.KernelRaw, param, verbose,
                                      verbose_path,
                                      perf_context=perf_context, run_in_process=run_in_process)
        return self.inline(name,
                        code,
                        ".cu",
                        _frame_cnt=_frame_cnt,
                        user_arg=user_arg,
                        disable_cache=disable_cache,
                        additional_vars=additional_vars,
                        generate_non_nested_code=True)
    
    def kernel_1d(self,
                  name: str,
                  num: int,
                  stream: int,
                  code: Union[str, pccm.FunctionCode],
                  verbose_path: str = "",
                  disable_cache: bool = False,
                  perf_context: Optional[ContextManager] = None,
                  run_in_process: bool = False,
                  additional_vars: Optional[Dict[str, Any]] = None,
                  *,
                  _frame_cnt: int = 2,
                  maximum_1d_threads: Optional[int] = None,
                  num_1d_threads: Optional[int] = None):
        verbose = verbose_path != ""
        num = int(num)
        launch_param = self.get_1d_param(num, stream=stream, maximum_1d_threads=maximum_1d_threads, num_1d_threads=num_1d_threads)
        user_arg = _NVRTCInlineParams(CUDAMode.Kernel1D,
                                      launch_param,
                                      verbose, verbose_path,
                                      perf_context=perf_context, run_in_process=run_in_process)
        if additional_vars is not None:
            additional_args = {
                **additional_vars,
                _CUMM_KERNEL_1D_SIZE_NAME: num,
            }
        else:
            additional_args = {
                _CUMM_KERNEL_1D_SIZE_NAME: num,
            }
        if compat.InMacOS:
            additional_args.pop(_CUMM_KERNEL_1D_SIZE_NAME)
        return self.inline(name,
                           code,
                           ".cu",
                           additional_args,
                           _frame_cnt=_frame_cnt,
                           user_arg=user_arg,
                           disable_cache=disable_cache,
                           generate_non_nested_code=True)

    def cpu_kernel_1d(self,
                      name: str,
                      num: int,
                      code: Union[str, pccm.FunctionCode],
                      verbose_path: str = "",
                      disable_cache: bool = False):
        verbose = verbose_path != ""
        num = int(num)
        user_arg = _NVRTCInlineParams(CUDAMode.Kernel1D,
                                      self.get_1d_param(num, stream=0),
                                      verbose, verbose_path,
                                      is_cpu=True)
        additional_args = {
            _CUMM_KERNEL_1D_SIZE_NAME: num,
        }
        return self.inline(name,
                           code,
                           ".cu",
                           additional_args,
                           _frame_cnt=2,
                           user_arg=user_arg,
                           disable_cache=disable_cache)

    def cpu_kernel_raw(self,
                       name: str,
                       #    param: tv.LaunchParam,
                       code: Union[str, pccm.FunctionCode],
                       verbose_path: str = "",
                       disable_cache: bool = False):
        verbose = verbose_path != ""
        user_arg = _NVRTCInlineParams(CUDAMode.KernelRaw,
                                      tv.LaunchParam(
                                          (1, 1, 1), (1, 1, 1)), verbose,
                                      verbose_path, is_cpu=True)
        return self.inline(name,
                           code,
                           ".cu",
                           _frame_cnt=2,
                           user_arg=user_arg,
                           disable_cache=disable_cache)

    def cpu_prepare_libraries(self, *libs: str):
        import llvmlite.binding as llvm
        for l in libs:
            llvm.load_library_permanently(l)

    def get_1d_param(self, num: int, smem: int = 0, stream: int = 0, maximum_1d_threads: Optional[int] = None, num_1d_threads: Optional[int] = None):
        if num_1d_threads is not None:
            threads = num_1d_threads
        else:
            if maximum_1d_threads is None:
                maximum_1d_threads = self.maximum_1d_threads
            if num > maximum_1d_threads:
                threads = maximum_1d_threads
            else:
                threads = div_up(num, 32) * 32
        if compat.InMacOS:
            # for metal, we use nonuniform as default, 
            # so the grid size is the same as block size
            blocks = num 
        else:
            blocks = div_up(num, threads)
        return tv.LaunchParam((blocks, 1, 1), (threads, 1, 1), smem, stream)


def main():
    import torch
    from cumm import tensorview as tv
    from cumm.common import TensorViewArrayLinalg, TensorViewNVRTCHashKernel
    INLINE = NVRTCInlineBuilder(
        [TensorViewArrayLinalg, TensorViewNVRTCHashKernel], reload_when_code_change=True)

    print(1)
    a = tv.zeros([2], tv.float32, 0)
    print(2)
    INLINE.kernel_1d("hahaha", a.dim(0), 0, f"""
    $a[i] = 5;
    """)
    b = tv.zeros([1000, 3], tv.float32, 0)
    bbb = torch.rand(1000, 3).float().cuda()
    t = np.eye(4)
    for i in range(10):
        INLINE.kernel_1d(
            "hahaha2", a.dim(0), 0, f"""
        auto bb = $bbb + i * 3;
        auto x = bb[0] * $t[0][0] + bb[1] * $t[0][1] + bb[2] * $t[0][1] + $t[0][3];
        auto y = bb[0] * $t[1][0] + bb[1] * $t[1][1] + bb[2] * $t[1][1] + $t[1][3];
        auto z = bb[0] * $t[2][0] + bb[1] * $t[2][1] + bb[2] * $t[2][1] + $t[2][3];

        bb[0] = x;
        bb[1] = std::numeric_limits<float>::denorm_min();
        bb[2] = z;
        """)

    print(a.cpu().numpy())


if __name__ == "__main__":
    main()
