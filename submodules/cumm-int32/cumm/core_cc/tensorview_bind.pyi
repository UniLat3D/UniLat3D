from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue, enum
class AppleMetalImpl:
    @staticmethod
    def hello() -> None: ...
class TensorViewBind:
    @staticmethod
    def hello() -> None: ...
    @staticmethod
    def _compile_test() -> None: ...
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

import builtins
from typing import Dict, List, Optional, Tuple, Type, Union, overload
from enum import Enum

import numpy as np

class Context:
    def create_cuda_stream(self) -> None:
        ... 

    def has_cuda_stream(self) -> bool:
        ...

    def cuda_stream_int(self) -> int:
        ... 

    def synchronize_stream(self) -> None:
        ... 

    def set_cuda_stream(self, stream: int) -> "Context":
        ... 

    def create_apple_metal_context(self) -> None:
        ...

    def create_or_update_metal_context_from_blob(self, command_buffer_ptr: int, dispatch_queue_ptr: int) -> None:
        ...

    def has_apple_metal_context(self) -> bool:
        ...

    def synchronize(self) -> None:
        ...

class CUDAEvent:
    def __init__(self, name: str = "") -> None:
        ...

    def record(self, stream: int = 0) -> "CUDAEvent":
        ...

    def stream_wait_me(self, stream: int, flag: int = 0) -> "CUDAEvent":
        ...

    def sync(self) -> None:
        ...

    @staticmethod
    def duration(start: "CUDAEvent", stop: "CUDAEvent") -> float:
        ...

    @staticmethod
    def sync_and_duration(start: "CUDAEvent", stop: "CUDAEvent") -> float:
        ...

class CPUEvent:
    def __init__(self, name: str = "") -> None:
        ...

    def record(self, stream: int = 0) -> "CPUEvent":
        ...

    def stream_wait_me(self, stream: int, flag: int = 0) -> "CPUEvent":
        ...

    def sync(self) -> None:
        ...

    @staticmethod
    def duration(start: "CUDAEvent", stop: "CUDAEvent") -> float:
        ...

    @staticmethod
    def sync_and_duration(start: "CUDAEvent", stop: "CUDAEvent") -> float:
        ...

class CUDAKernelTimer:
    def __init__(self, enable: bool) -> None:
        ...

    def push(self, name: str) -> None:
        ...

    def pop(self) -> None:
        ...

    def record(self, name: str, stream: int = 0) -> None:
        ...

    def insert_pair(self, name: str, start: str, stop: str) -> str:
        ...

    def get_pair_duration(self, name: str) -> float:
        ...

    def has_pair(self, name: str) -> bool:
        ...

    def sync_all_event(self) -> None:
        ...

    def get_all_pair_duration(self) -> Dict[str, float]:
        ...

    @property
    def enable(self) -> bool:
        ...


class NVRTCProgram:
    kSource = 0
    kPTX = 1
    kCuBin = 2
    def __init__(self,
                 code: str,
                 headers: Dict[str, str] = {},
                 opts: List[str] = [],
                 program_name: str = "kernel") -> None:
        ...

    def ptx(self) -> str:
        ...

    def compile_log(self) -> str:
        ...

    def get_lowered_name(self, name: str) -> str:
        ...

    def to_string(self) -> str:
        ...

    def to_binary(self, serial_type: int) -> bytes:
        ...

    def get_predefined_lowered_name_map(self) -> Dict[str, str]: 
        ...

    @staticmethod
    def from_string(json_string: str) -> "NVRTCProgram":
        ...

    @staticmethod
    def from_binary(buffer: bytes) -> "NVRTCProgram":
        ...

class NVRTCModule:
    kTensor = 0
    kArray = 1
    kTensorView = 2
    kScalar = 3
    kConstant = 4
    kDevicePointer = 5

    @overload
    def __init__(self,
                 code: str,
                 headers: Dict[str, str] = {},
                 opts: List[str] = [],
                 program_name: str = "kernel",
                 name_exprs: List[str] = [],
                 cudadevrt_path: str = "") -> None:
        ...

    @overload
    def __init__(self, prog: NVRTCProgram, cudadevrt_path: str = "") -> None:
        ...

    def load(self) -> "NVRTCModule":
        ...

    def run_kernel(self, name: str, blocks: List[int], threads: List[int],
                   smem_size: int, stream: int, args: List[Tuple[Tensor,
                                                                 int, int, int]]):
        ...

    @property
    def program(self) -> NVRTCProgram:
        ...

    def get_lowered_name(self, name: str) -> str:
        ...

    def get_kernel_attributes(self, name: str) -> Dict[str, int]:
        ...

class MetalModule:
    @overload
    def __init__(self, binary: "Tensor"):
        ...

    @overload
    def __init__(self, code: str, preprocessorMacros: Dict[str, str], fastMathEnabled: bool = True):
        ...

    def run_kernel(self, name: str, blocks: List[int], threads: List[int],
                   smem_size: int, ctx: Context, args: List[Tuple[Tensor,
                                                                 int, int, int]],
                    use_nonuniform_threadgroup: bool = True):
        ...


class Tensor:
    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self,
                 shape: Union[List[int], Tuple[int]],
                 dtype: int = 0,
                 device: int = -1,
                 pinned: bool = False,
                 managed: bool = False):
        ...

    @property
    def shape(self) -> List[int]:
        ...

    @property
    def stride(self) -> List[int]:
        ...

    @property
    def dtype(self) -> int:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def itemsize(self) -> int:
        ...

    @property
    def ndim(self) -> int:
        ...

    @property
    def device(self) -> int:
        ...

    def pinned(self) -> bool:
        ...

    def is_contiguous(self) -> bool:
        ...

    def is_col_major_matrix(self) -> bool:
        ...

    def is_readonly(self) -> bool:
        ...

    def get_readonly(self) -> "Tensor":
        ...

    def byte_offset(self) -> int:
        ...
        
    def storage_bytesize(self) -> int:
        ...

    def bytesize(self) -> int:
        ...

    def empty(self) -> bool:
        ...

    def dim(self, axis: int) -> int:
        ...

    def slice_first_axis(self, start: int, end: int) -> "Tensor":
        ...

    def view(self, views: List[int]) -> "Tensor":
        ...

    def clone(self,
              pinned: bool = False,
              use_cpu_copy: bool = False) -> "Tensor":
        ...

    def clone_whole_storage(self) -> "Tensor":
        ...

    def zero_whole_storage_(self) -> None:
        ...

    def unsqueeze(self, axis: int) -> "Tensor":
        ...

    @overload
    def squeeze(self) -> "Tensor":
        ...

    @overload
    def squeeze(self, axis: int) -> "Tensor":
        ...

    @overload
    def __getitem__(self, idx: int) -> "Tensor":
        ...

    @overload
    def __getitem__(self, idx: slice) -> "Tensor":
        ...

    @overload
    def __getitem__(
        self, idx: Tuple[Union[int, None, slice, builtins.ellipsis],
                         ...]) -> "Tensor":
        ...

    def as_strided(self, shape: List[int], stride: List[int],
                   storage_byte_offset: int) -> "Tensor":
        ...

    def slice_axis(self,
                   dim: int,
                   start: Optional[int],
                   stop: Optional[int],
                   step: Optional[int] = None) -> "Tensor":
        ...

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        ...

    @property 
    def T(self) -> "Tensor":
        """get transposed matrix. tensor must be 2d.
        """
        ...

    def select(self, dim: int, index: int) -> "Tensor":
        ...

    def numpy(self) -> np.ndarray:
        ...

    def numpy_view(self) -> np.ndarray:
        ...

    @overload
    def cpu(self) -> "Tensor":
        ...

    @overload
    def cpu(self, ctx: Context) -> "Tensor":
        ...

    @overload
    def copy_(self, other: "Tensor") -> None:
        ...

    @overload
    def copy_(self, other: "Tensor", ctx: Context) -> None:
        ...

    @overload
    def copy_storage_(self, other: "Tensor") -> None:
        ...

    @overload
    def copy_storage_(self, other: "Tensor", ctx: Context) -> None:
        ...

    @overload
    def zero_(self) -> "Tensor":
        ...

    @overload
    def zero_(self, ctx: Context) -> "Tensor":
        ...

    @overload
    def cuda(self) -> "Tensor":
        ...

    @overload
    def cuda(self, ctx: Context) -> "Tensor":
        ...

    @overload
    def fill_int_(self, val: Union[int, float]) -> "Tensor":
        ...

    @overload
    def fill_int_(self, val: Union[int, float],
                  ctx: Context) -> "Tensor":
        ...

    @overload
    def fill_float_(self, val: Union[int, float]) -> "Tensor":
        ...

    @overload
    def fill_float_(self, val: Union[int, float],
                    ctx: Context) -> "Tensor":
        ...
    @overload
    def type_view(self, dtype: int) -> "Tensor":
        ...
    @overload
    def type_view(self, dtype: int, shape: List[int]) -> "Tensor":
        ...

    def byte_pointer(self) -> int:
        ...
        
    def gpu_address(self) -> int:
        ...



def zeros(shape: List[int],
          dtype: int,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    ...


@overload
def from_blob(ptr: int,
              shape: List[int],
              stride: List[int],
              dtype: int,
              device: int,
              storage_offset: int = 0) -> Tensor:
    ...


@overload
def from_const_blob(ptr: int,
                    shape: List[int],
                    stride: List[int],
                    dtype: int,
                    device: int,
                    storage_offset: int = 0) -> Tensor:
    ...


@overload
def from_blob(ptr: int,
              shape: List[int],
              dtype: int,
              device: int,
              storage_offset: int = 0) -> Tensor:
    ...


@overload
def from_const_blob(ptr: int,
                    shape: List[int],
                    dtype: int,
                    device: int,
                    storage_offset: int = 0) -> Tensor:
    ...


def empty(shape: List[int],
          dtype: int,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    ...


def full_float(shape: List[int],
               val: float,
               dtype: int,
               device: int = -1,
               pinned: bool = False,
               managed: bool = False) -> Tensor:
    ...


def full_int(shape: List[int],
             val: int,
             dtype: int,
             device: int = -1,
             pinned: bool = False,
             managed: bool = False) -> Tensor:
    ...


def zeros_managed(shape: List[int], dtype: int) -> Tensor:
    ...


def from_numpy(arr: np.ndarray) -> Tensor:
    ...


def get_compute_capability(index: int = -1) -> Tuple[int, int]:
    ...


def is_cpu_only() -> bool:
    ...


def cufilt(name: str) -> str:
    ...


def tvdtype_bitsize(dtype: int) -> int:
    ...

def tvdtype_itemsize(dtype: int) -> int:
    ...

class ConvOpType(Enum):
    Forward = 0
    BackwardInput = 1
    BackwardWeight = 2


class ConvIterAlgo(Enum):
    Analytic = 0
    Optimized = 1


class ConvMode(Enum):
    Convolution = 0
    CrossCorrelation = 1


class ConvLayoutType(Enum):
    ChannelFirst = 0
    ChannelLast = 1
    SpatialFirst = 2


class ShuffleStrideType(Enum):
    NoShuffle = 0
    ShuffleAC = 1
    ShuffleAB = 2

class Activation(Enum):
  None_ = 0
  ReLU = 1
  Sigmoid = 2
  LeakyReLU = 3

#   Tanh = 3
#   ELU = 5
#   SeLU = 6
#   Softsign = 7
#   Softplus = 8
#   Clip = 9
#   HardSigmoid = 10
#   ScaledTanh = 11
#   ThresholdedReLU = 12

class NVRTCParams:
    cumodule: NVRTCModule
    kernel_name: str
    init_kernel_name: str
    constant_name: str
    param_size: int
    param_storage: Tensor
    param_storage_cpu: Tensor
    num_threads: int
    smem_size: int
    mode: int


class GemmAlgoDesp:
    dtype_a: int
    dtype_b: int
    dtype_c: int
    tile_shape: Tuple[int, int, int]
    warp_tile_shape: Tuple[int, int, int]
    num_stage: int
    dacc: int
    dcomp: int
    algo: str
    tensorop: Tuple[int, int, int]
    split_k_serial_: int
    split_k_parallel_: int
    shuffle_type: ShuffleStrideType
    element_per_access_a: int
    element_per_access_b: int
    element_per_access_c: int
    access_per_vector: int
    is_nvrtc: bool
    min_arch: Tuple[int, int]

    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def copy(self) -> "GemmAlgoDesp":
        ...

    @property
    def split_k_serial(self) -> bool:
        ...

    @split_k_serial.setter
    def split_k_serial(self, val: bool) -> None:
        """
        Args:
            val: 
        """
        ...

    @property
    def split_k_parallel(self) -> bool:
        ...

    @split_k_parallel.setter
    def split_k_parallel(self, val: bool) -> None:
        """
        Args:
            val: 
        """
        ...

    def check_valid(self) -> None:
        ...

    @property
    def trans_a(self) -> bool:
        ...

    @trans_a.setter
    def trans_a(self, val: bool) -> None:
        """
        Args:
            val: 
        """
        ...

    @property
    def trans_b(self) -> bool:
        ...

    @trans_b.setter
    def trans_b(self, val: bool) -> None:
        """
        Args:
            val: 
        """
        ...

    @property
    def trans_c(self) -> bool:
        ...

    @trans_c.setter
    def trans_c(self, val: bool) -> None:
        """
        Args:
            val: 
        """
        ...

    def query_workspace_size(self, m: int, n: int, k: int,
                             split_k_slices: int) -> int:
        """
        Args:
            m: 
            n: 
            k: 
            split_k_slices: 
        """
        ...

    def supported(self, m: int, n: int, k: int) -> bool:
        """
        Args:
            m: 
            n: 
            k: 
        """
        ...

    def supported_ldx(self, lda: int, ldb: int, ldc: int) -> bool:
        """
        Args:
            lda: 
            ldb: 
            ldc: 
        """
        ...


class ConvAlgoDesp(GemmAlgoDesp):
    ndim: int
    op_type: ConvOpType
    iter_algo: ConvIterAlgo
    layout_i: ConvLayoutType
    layout_w: ConvLayoutType
    layout_o: ConvLayoutType
    interleave_i: int
    interleave_w: int
    interleave_o: int
    mask_sparse: bool
    increment_k_first: bool
    is_int8_inference: bool 
    dynamic_mask: bool

    def copy(self) -> "ConvAlgoDesp":
        ...

    def __init__(self, ndim: int, op_type: ConvOpType) -> None:
        """
        Args:
            ndim: 
            op_type: 
        """
        ...

    def __repr__(self) -> str:
        ...

    @staticmethod
    def conv_iwo_012_to_abc(op_type: ConvOpType) -> List[int]:
        """
        Args:
            op_type: 
        """
        ...

    @staticmethod
    def gemm_abc_012_to_iwo(op_type: ConvOpType) -> List[int]:
        """
        Args:
            op_type: 
        """
        ...

    @property
    def dtype_input(self) -> int:
        ...

    @property
    def dtype_weight(self) -> int:
        ...

    @property
    def dtype_output(self) -> int:
        ...

    def supported(self, m: int, n: int, k: int, C: int, K: int,
                  mask_width: int) -> bool:
        """
        Args:
            m: 
            n: 
            k: 
            C: 
            K: 
            mask_width: 
        """
        ...

    def query_conv_workspace_size(self, m: int, n: int, k: int,
                                  split_k_slices: int, kv: int) -> int:
        """
        Args:
            m: 
            n: 
            k: 
            split_k_slices: 
            kv: 
        """
        ...

    def supported_ldx_conv(self, ldi: int, ldw: int, ldo: int) -> bool:
        """
        Args:
            ldi: 
            ldw: 
            ldo: 
        """
        ...


class GemmParams:
    algo_desp: GemmAlgoDesp
    split_k_slices: int
    workspace: Tensor = Tensor()
    a_inds: Tensor = Tensor()
    b_inds: Tensor = Tensor()
    c_inds: Tensor = Tensor()
    alpha: float
    beta: float
    act_alpha: float
    act_beta: float
    act_type: Activation

    stream: int
    timer: CUDAKernelTimer
    nvrtc_params: NVRTCParams

    def __init__(
        self, timer: CUDAKernelTimer = CUDAKernelTimer(False)) -> None:
        """
        Args:
            timer: 
        """
        ...

    def check_valid(self) -> None:
        ...

    @property
    def a(self) -> Tensor:
        ...

    @a.setter
    def a(self, val: Tensor) -> None:
        """
        Args:
            val: 
        """
        ...

    @property
    def b(self) -> Tensor:
        ...

    @b.setter
    def b(self, val: Tensor) -> None:
        """
        Args:
            val: 
        """
        ...

    @property
    def c(self) -> Tensor:
        ...

    @c.setter
    def c(self, val: Tensor) -> None:
        """
        Args:
            val: 
        """
        ...
    @property
    def d(self) -> Tensor:
        ...

    @d.setter
    def d(self, val: Tensor) -> None:
        """
        Args:
            val: 
        """
        ...


class ConvParams:
    conv_algo_desp: ConvAlgoDesp
    input: Tensor
    weight: Tensor
    output: Tensor
    split_k_slices: int
    padding: List[int]
    stride: List[int]
    dilation: List[int]
    alpha: float
    beta: float
    act_alpha: float
    act_beta: float
    act_type: Activation

    mask_width: int
    mask_filter: int
    reverse_mask: bool
    verbose: bool
    timer: CUDAKernelTimer
    workspace: Tensor = Tensor()
    mask: Tensor = Tensor()
    mask_argsort: Tensor = Tensor()
    indices: Tensor = Tensor()
    mask_output: Tensor = Tensor()
    stream: int
    nvrtc_params: NVRTCParams
    bias: Tensor = Tensor()
    output_add: Tensor = Tensor()
    scale: Tensor = Tensor()

    def __init__(
        self,
        ndim: int,
        op_type: ConvOpType,
        timer: CUDAKernelTimer = CUDAKernelTimer(False)
    ) -> None:
        """
        Args:
            ndim: 
            op_type: 
            timer: 
        """
        ...


def run_nvrtc_gemm_kernel(params: GemmParams) -> None:
    ...


def run_nvrtc_conv_kernel(params: ConvParams) -> None:
    ...

def check_cuda_error() -> None:
    ...

def cat_first_axis(tens: List[Tensor]) -> Tensor:
    ...