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
from dataclasses import dataclass
from enum import Enum
import enum
from functools import partial
import io
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, ContextManager

import numpy as np
from pccm import Argument
from pccm.middlewares.pybind import (TemplateTypeStmt,
                                     _simple_template_type_parser)
import multiprocessing
from cumm.core_cc import tensorview_bind
from cumm.core_cc.tensorview_bind import CUDAKernelTimer, CUDAEvent, Context
from cumm.core_cc.tensorview_bind import NVRTCModule as _NVRTCModule
from cumm.core_cc.tensorview_bind import MetalModule as _MetalModule

from cumm.core_cc.tensorview_bind import NVRTCProgram, Tensor, check_cuda_error
from . import gemm, utils

from cumm.dtypes import get_npdtype_from_tvdtype

bool_ = 5
float16 = 7
float32 = 0
float64 = 4
int8 = 3
int16 = 2
int32 = 1
int64 = 8
uint8 = 6
uint16 = 9
uint32 = 10
uint64 = 11
bfloat16 = 12
tf32 = 13

custom16 = 100
custom32 = 101
custom48 = 102
custom64 = 103
custom80 = 104
custom96 = 105
custom128 = 106
unknown = -1

_SIMPLE_TYPES_TO_TV_DTYPE = {
    "int": int32,
    "int8_t": int8,
    "int16_t": int16,
    "int32_t": int32,
    "int64_t": int64,
    "uint8_t": uint8,
    "uint16_t": uint16,
    "uint32_t": uint32,
    "uint64_t": uint64,
    "std::intptr_t": int64,
    "std::uintptr_t": uint64,
    "size_t": uint64,
    "std::size_t": uint64,
    "unsigned": uint32,
    "long": int64,
    "short": int16,
    "float": float32,
    "double": float64,
    "unsigned long": uint64,
    "unsigned int": uint32,
    "__half": float16,
    "tv::half_t": float16,
    "tv::bfloat16_t": bfloat16,
    "at::Half": float16,
}  # type: Dict[str, int]

_VALID_CONTAINER_TYPES = {"tv::array", "std::array", "tv::TensorView"}

class NVRTCArgBaseType(enum.IntEnum):
    Scalar = 0
    Pointer = 1
    Array = 2
    TensorView = 3

def div_up(a, b):
    return (a + b - 1) // b 

@dataclass
class NVRTCArgMeta:
    base_type: NVRTCArgBaseType
    valid: bool
    simple_type: int
    shape: List[int]

    is_simple_ptr: bool = False
    is_scalar: bool = False
    count: Optional[int] = None


class NVRTCKernelMeta:
    def __init__(self, name: str, ns: str, args: List[Argument]):
        self.name = name
        self.ns = ns
        self.args = args
        self.arg_types = [
            _simple_template_type_parser(a.type_str, {}) for a in args
        ]
        self.simple_types: List[Optional[int]] = []
        self.arg_metas: List[NVRTCArgMeta] = []
        for arg, meta in zip(args, self.arg_types):
            simple_tv_type = -1
            is_simple_ptr = False
            valid = meta.name != ""
            shape: List[int] = []
            if valid and not meta.is_ptr:
                # determine scalar type
                cur_meta = meta
                while True:
                    if len(cur_meta.args) == 0:
                        if cur_meta.name in _SIMPLE_TYPES_TO_TV_DTYPE:
                            simple_tv_type = _SIMPLE_TYPES_TO_TV_DTYPE[
                                cur_meta.name]
                        break
                    is_valid_container = cur_meta.name in _VALID_CONTAINER_TYPES
                    if not is_valid_container and len(cur_meta.args) > 0:
                        valid = False
                        break
                    length = int(cur_meta.args[1].name)
                    shape.append(length)
                    cur_meta = cur_meta.args[0]
            elif valid and meta.is_ptr:
                if meta.name in _SIMPLE_TYPES_TO_TV_DTYPE:
                    is_simple_ptr = True
                    simple_tv_type = _SIMPLE_TYPES_TO_TV_DTYPE[meta.name]
            is_scalar = len(shape) == 0
            base_type = NVRTCArgBaseType.Scalar if is_scalar else NVRTCArgBaseType.Array
            if meta.name == "tv::TensorView":
                base_type = NVRTCArgBaseType.TensorView
                assert len(shape) == 1, "only support 1d tensorview (e.g. tv::Tensorview<float, N>)"
            if len(shape) == 0:
                shape = [1]
            # shape = shape[::-1]
            self.arg_metas.append(
                NVRTCArgMeta(base_type, valid, simple_tv_type, shape, is_simple_ptr, is_scalar))
            if isinstance(arg.array, int):
                self.arg_metas[-1].count = arg.array
            elif isinstance(arg.array, str):
                raise NotImplementedError("don't support string array", arg)

    def __repr__(self) -> str:
        return f"NVRTCKernelMeta[name={self.name},ns={self.ns},args={self.arg_metas}]"

class LaunchParam:
    def __init__(self,
                 blocks: Union[Tuple[int, ...], List[int]],
                 threads: Union[Tuple[int, ...], List[int]],
                 smem: int = 0,
                 stream: int = 0,
                 ctx: Optional[Context] = None) -> None:
        self.blocks = list(blocks)
        self.threads = list(threads)
        assert len(blocks) == 3
        assert len(threads) == 3
        self.smem = smem
        self.stream = stream
        self.ctx = ctx

    def copy(self):
        return LaunchParam(self.blocks, self.threads, self.smem, self.stream, self.ctx)


def _run_kernel(mod: Union[_NVRTCModule, _MetalModule], name: str, launch: LaunchParam,
                *args: Union[Tensor, int, float, List[int], List[float],
                            Tuple[float, ...], Tuple[int, ...]],
                perf_context: Optional[ContextManager] = None,
                name_to_meta: Optional[Dict[str, NVRTCKernelMeta]] = None,
                lower_name: Optional[str] = None,
                use_nonuniform_threadgroup: bool = True):
    # WARNING: use_nonuniform_threadgroup only used on apple silicon
    
    # t = time.time()
    metas: List[NVRTCArgMeta] = [NVRTCArgMeta(NVRTCArgBaseType.Scalar, False, -1, [])] * len(args)
    if name_to_meta:
        assert name in name_to_meta, f"can't find your kernel {name}, available: {name_to_meta.keys()}"
        assert len(args) == len(name_to_meta[name].args)
        metas = name_to_meta[name].arg_metas
    if lower_name is not None:
        name = lower_name
    kernel_args: List[Tuple[Tensor, int, int, int]] = []
    for arg, meta in zip(args, metas):
        if meta.valid:
            # print(meta.shape)
            if meta.is_simple_ptr:
                if not isinstance(arg, Tensor):
                    raise ValueError("your arg must be tensor")
                if not arg.dtype == meta.simple_type:
                    cur_dtype = get_npdtype_from_tvdtype(arg.dtype)
                    expected_dtype = get_npdtype_from_tvdtype(
                        meta.simple_type)
                    raise ValueError(
                        f"your tensor {arg.shape}|{cur_dtype}"
                        f" dtype not equal to {expected_dtype}")
                kernel_args.append((arg, _NVRTCModule.kTensor, 0, 0))
                continue
            elif meta.base_type == NVRTCArgBaseType.TensorView:
                if not isinstance(arg, Tensor):
                    raise ValueError("your arg must be tensor")
                assert arg.ndim == meta.shape[0],f"your tensor ndim {arg.ndim} must equal to f{meta.shape[0]}"
                if not arg.dtype == meta.simple_type:
                    cur_dtype = get_npdtype_from_tvdtype(arg.dtype)
                    expected_dtype = get_npdtype_from_tvdtype(
                        meta.simple_type)
                    raise ValueError(
                        f"your tensor {arg.shape}|{cur_dtype}"
                        f" dtype not equal to {expected_dtype}")
                kernel_args.append((arg, _NVRTCModule.kTensorView, 0, 0))
                continue
            else:
                # we can't ensure arg isn't tv::Tensor.
                if not isinstance(arg, Tensor):
                    assert not isinstance(arg, Tensor)
                    tv_dtype = meta.simple_type
                    if isinstance(arg, np.ndarray):
                        dtype = get_npdtype_from_tvdtype(meta.simple_type)
                        arg_array = arg.astype(dtype)
                        arg_type = _NVRTCModule.kArray
                    elif isinstance(arg, bool):
                        arg_type = _NVRTCModule.kConstant 
                        tv_dtype = uint8
                        arg_array = np.array(arg, dtype=np.uint8)
                    else:
                        dtype = get_npdtype_from_tvdtype(meta.simple_type)
                        assert isinstance(arg, (int, float, np.integer, np.floating))
                        arg_type = _NVRTCModule.kScalar
                        arg_array = np.array(arg, dtype=dtype)
                    if not arg_array.shape:
                        arg_array = arg_array.reshape(1)
                    assert list(arg_array.shape) == meta.shape, f"{arg_array.shape}, {meta.shape}"
                    # auto dtype cast
                    # TODO prevent floats assigned to ints
                    ten = empty(meta.shape, tv_dtype, -1)
                    ten.numpy_view()[:] = arg_array
                    kernel_args.append((ten, arg_type, 0, 0))
                    continue
        # meta isn't valid, use regular dtypes.
        if isinstance(arg, bool):
            ten = full([1], arg, uint8)
            kernel_args.append((ten, _NVRTCModule.kConstant, 0, 0))

        elif isinstance(arg, (int, float, np.integer, np.floating)):
            dtype = float32
            if isinstance(arg, (int, np.integer)):
                dtype = int64
            ten = full([1], arg, dtype)
            kernel_args.append((ten, _NVRTCModule.kArray, 0, 0))
        elif isinstance(arg, (list, tuple)):
            raise NotImplementedError("don't support list or tuple, you must use np.ndarray")
            dtype = np.float32
            if isinstance(arg[0], int):
                dtype = np.int64
            arg_np = np.array(arg, dtype=dtype)
            ten = from_numpy(arg_np).clone()
            kernel_args.append((ten, _NVRTCModule.kArray))
        elif isinstance(arg, np.ndarray):
            ten = from_numpy(arg).clone()
            kernel_args.append((ten, _NVRTCModule.kArray, 0, 0))
        else:
            assert isinstance(arg, Tensor)
            kernel_args.append((arg, _NVRTCModule.kTensor, 0, 0))
    if perf_context is not None:
        with perf_context:
            if isinstance(mod, _NVRTCModule):
                return mod.run_kernel(name, launch.blocks, launch.threads,
                                    launch.smem, launch.stream, kernel_args)
            else:
                assert launch.ctx is not None 
                return mod.run_kernel(name, launch.blocks, launch.threads,
                                    launch.smem, launch.ctx, kernel_args)
    if isinstance(mod, _NVRTCModule):
        # with measure_and_print(name):
        return mod.run_kernel(name, launch.blocks, launch.threads,
                                    launch.smem, launch.stream, kernel_args)
    else:
        assert launch.ctx is not None 
        # print("preprocess time", time.time() - t)
        # t = time.time()
        res = mod.run_kernel(name, launch.blocks, launch.threads,
                                    launch.smem, launch.ctx, kernel_args,
                                    use_nonuniform_threadgroup)
        # print(f"kernel {name} time: {time.time() - t}")
        return res 

class NVRTCModule:
    def __init__(self,
                 code: Union[str, NVRTCProgram],
                 headers: Optional[Dict[str, str]] = None,
                 opts: Optional[List[str]] = None,
                 program_name: str = "kernel.cu",
                 name_exprs: Optional[List[str]] = None,
                 name_to_meta: Optional[Dict[str, NVRTCKernelMeta]] = None,
                 cudadevrt_path: str = "") -> None:
        if headers is None:
            headers = {}
        if opts is None:
            opts = []
        if name_exprs is None:
            name_exprs = []
        if isinstance(code, str):
            self._mod = _NVRTCModule(code, headers, opts, program_name,
                                     name_exprs, cudadevrt_path)
        else:
            self._mod = _NVRTCModule(code, cudadevrt_path)
        self._name_exprs = name_exprs
        self.name_to_meta = name_to_meta
        self.cudadevrt_path = cudadevrt_path

    def load(self):
        return self._mod.load()

    def get_cpp_object(self):
        return self._mod

    def get_ptx(self):
        return self._mod.program.ptx()

    def to_string(self):
        return self._mod.program.to_string()

    def __getstate__(self):
        buf = self._mod.program.to_binary(NVRTCProgram.kCuBin)
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_mod']
        state['_mod'] = buf
        return state 

    def __setstate__(self, state):
        prog = NVRTCProgram.from_binary(state["_mod"])
        state["_mod"] = _NVRTCModule(prog, state["cudadevrt_path"])
        self.__dict__.update(state)

    def get_1d_launch_param(self, num: int, smem: int = 0, stream: int = 0):
        if num > 1024:
            threads = 1024
        else:
            threads = div_up(num, 32) * 32
        blocks = div_up(num, threads)
        return LaunchParam((blocks, 1, 1), (threads, 1, 1), smem, stream)

    def get_launch_param(self, blocks: Union[Tuple[int, ...], List[int]],
        threads: Union[Tuple[int, ...], List[int]], smem_size: int = 0, stream: int = 0):
        return LaunchParam(blocks, threads, smem_size, stream)

    def get_kernel_attrs(self, name: str):
        return self._mod.get_kernel_attributes(name)

    def run_kernel_unchecked(self, name: str, launch: LaunchParam, args: Sequence[Tuple[Tensor, int, int, int]]):
        if self.name_to_meta:
            assert name in self.name_to_meta, f"can't find your kernel {name}, available: {self.name_to_meta.keys()}"
            assert len(args) == len(self.name_to_meta[name].args)
        if self._name_exprs:
            name = self.get_lowered_name(name)
        return self._mod.run_kernel(name, launch.blocks, launch.threads,
                                    launch.smem, launch.stream, args)

    def run_kernel_in_spawn_process(self, name: str, launch: LaunchParam,
                   *args: Union[Tensor, int, float, List[int], List[float],
                                Tuple[float, ...], Tuple[int, ...]], timeout: Optional[float] = None):
        ctx = multiprocessing.get_context("spawn")
        arg_kernel_meta = []
        for arg in args:
            if isinstance(arg, Tensor):
                arg_kernel_meta.append((arg.cpu().numpy(), True))
            else:
                arg_kernel_meta.append((arg, False))
        ret_queue = ctx.Queue()
        proc = ctx.Process(target=self._run_kernel_in_spawn_process_func, args=(ret_queue, name, launch, arg_kernel_meta))
        proc.daemon = True 
        proc.start()
        # must get queue before proc join to avoid dead lock
        res = ret_queue.get()
        # TODO how to deal with timeout process (kernel deadlock)?
        proc.join(timeout)
        if isinstance(res, str):
            raise ValueError(f"Error, traceback: \n{res}")
        else:
            for arg, (arg_np, is_tensor) in zip(args, res):
                if isinstance(arg, Tensor):
                    # use copy_storage_ to avoid strided tensor copy which
                    # isn't supported
                    arg.copy_storage_(from_numpy(arg_np))
        

    def _run_kernel_in_spawn_process_func(self, q: multiprocessing.Queue, name: str, launch: LaunchParam,
                   arg_proc_metas: List[Tuple[Union[np.ndarray, int, float, List[int], List[float],
                                Tuple[float, ...], Tuple[int, ...]], bool]]):
        launch.stream = 0
        args = []
        arg_pairs = []
        for arg, is_tensor in arg_proc_metas:
            if is_tensor:
                assert isinstance(arg, np.ndarray)
                pair = (arg, from_numpy(arg).cuda())
                arg_pairs.append(pair)
                args.append(pair[-1])
            else:
                args.append(arg)
        try:
            self.run_kernel(name, launch, *args)
            for arg, ten in arg_pairs:
                arg[:] = ten.cpu().numpy()
            q.put(arg_proc_metas)
        except:
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            q.put(ss.getvalue())

    def run_kernel(self, name: str, launch: LaunchParam,
                   *args: Union[Tensor, int, float, List[int], List[float],
                                Tuple[float, ...], Tuple[int, ...]],
                   perf_context: Optional[ContextManager] = None):
        metas: List[NVRTCArgMeta] = [NVRTCArgMeta(NVRTCArgBaseType.Scalar, False, -1, [])] * len(args)
        if self.name_to_meta:
            assert name in self.name_to_meta, f"can't find your kernel {name}, available: {self.name_to_meta.keys()}"
            assert len(args) == len(self.name_to_meta[name].args)
            metas = self.name_to_meta[name].arg_metas
        lower_name = None 
        if self._name_exprs:
            lower_name = self.get_lowered_name(name)
        return _run_kernel(self._mod, name, launch, *args, perf_context=perf_context, name_to_meta=self.name_to_meta, lower_name=lower_name)

    @property
    def program(self):
        return self._mod.program

    def get_lowered_name(self, name: str) -> str:
        return self._mod.get_lowered_name(name)

    def tensor_scalar(self, val, dtype: int):
        return full([1], val, dtype)

    def arg_scalar(self, val, dtype: int):
        return (full([1], val, dtype), _NVRTCModule.kScalar)

    def arg_tensor(self, ten: Tensor):
        return (ten, _NVRTCModule.kTensor)

    def arg_array(self, arr, dtype: int):
        arg_array = np.array(arr)
        ten = empty(list(arg_array.shape), dtype, -1)
        ten.numpy_view()[:] = arg_array
        return (ten, _NVRTCModule.kArray)

class MetalModule:
    def __init__(self,
                 binary: bytes,
                 name_to_meta: Optional[Dict[str, NVRTCKernelMeta]] = None) -> None:
        binary_np = np.frombuffer(binary, dtype=np.uint8)
        binary_ten = from_numpy(binary_np)
        self.name_to_meta = name_to_meta

        self._mod = _MetalModule(binary_ten)

    def get_cpp_object(self):
        return self._mod

    def get_1d_launch_param(self, num: int, smem: int = 0, stream: int = 0):
        if num > 1024:
            threads = 1024
        else:
            threads = div_up(num, 32) * 32
        blocks = div_up(num, threads)
        return LaunchParam((blocks, 1, 1), (threads, 1, 1), smem, stream)

    def get_launch_param(self, blocks: Union[Tuple[int, ...], List[int]],
        threads: Union[Tuple[int, ...], List[int]], smem_size: int = 0, stream: int = 0):
        return LaunchParam(blocks, threads, smem_size, stream)


    def run_kernel_unchecked(self, name: str, launch: LaunchParam, args: List[Tuple[Tensor, int, int, int]], use_nonuniform_threadgroup: bool):
        assert launch.ctx is not None 
        if self.name_to_meta:
            assert name in self.name_to_meta, f"can't find your kernel {name}, available: {self.name_to_meta.keys()}"
            assert len(args) == len(self.name_to_meta[name].args)
        return self._mod.run_kernel(name, launch.blocks, launch.threads,
                                    launch.smem, launch.ctx, args, use_nonuniform_threadgroup)


    def run_kernel(self, name: str, launch: LaunchParam,
                   *args: Union[Tensor, int, float, List[int], List[float],
                                Tuple[float, ...], Tuple[int, ...]],
                   perf_context: Optional[ContextManager] = None,
                   use_nonuniform_threadgroup: bool = True):
        return _run_kernel(self._mod, name, launch, *args, perf_context=perf_context, name_to_meta=self.name_to_meta, 
            use_nonuniform_threadgroup=use_nonuniform_threadgroup)

    def tensor_scalar(self, val, dtype: int):
        return full([1], val, dtype)

    def arg_scalar(self, val, dtype: int):
        return (full([1], val, dtype), _NVRTCModule.kScalar)

    def arg_tensor(self, ten: Tensor):
        return (ten, _NVRTCModule.kTensor)

    def arg_array(self, arr, dtype: int):
        arg_array = np.array(arr)
        ten = empty(list(arg_array.shape), dtype, -1)
        ten.numpy_view()[:] = arg_array
        return (ten, _NVRTCModule.kArray)

class nullcontext(contextlib.AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass

class _TimerMeasure:
    def __init__(self, duration: float) -> None:
        self.duration = duration

class KernelTimer:
    def __init__(self, enable: bool = True, disable_if_cpu_only: bool = True) -> None:
        if disable_if_cpu_only:
            self.enable = enable and not tensorview_bind.is_cpu_only()
        else:
            self.enable = enable
        if self.enable:
            self._timer = CUDAKernelTimer(enable)
        else:
            self._timer = None

    def get_cpp_object(self):
        assert self._timer is not None 
        return self._timer

    @contextlib.contextmanager
    def _namespace(self, name: str):
        assert self._timer is not None
        self._timer.push(name)
        try:
            yield
        finally:
            self._timer.pop()

    @contextlib.contextmanager
    def _record(self, name: str, stream: int = 0, exit_handler: Optional[Callable[["CUDAKernelTimer", str], None]] = None, measure_time: bool = False):
        if not self.enable:
            yield _TimerMeasure(-1.0)
            return
        assert self._timer is not None
        self._timer.push(name)
        try:
            measure = _TimerMeasure(-1.0)
            pair_name = self._timer.insert_pair("", "start", "stop")
            self._timer.record("start", stream)
            yield measure
            self._timer.record("stop", stream)
            if measure_time:
                duration = self._timer.get_pair_duration(pair_name)
                measure.duration = duration
            if exit_handler is not None:
                exit_handler(self._timer, pair_name)
        finally:
            self._timer.pop()

    def namespace(self, name: str):
        if self.enable:
            return self._namespace(name)
        else:
            return nullcontext()

    def record(self, name: str, stream: int = 0):
        if self.enable:
            return self._record(name, stream)
        else:
            return nullcontext()

    def get_all_pair_time(self) -> Dict[str, float]:
        if self.enable:
            assert self._timer is not None
            return self._timer.get_all_pair_duration()
        else:
            return {}

    @staticmethod
    def collect_by_name(name: str, res: Dict[str, float]):
        filtered_res: Dict[str, float] = {}
        for k, v in res.items():
            k_split = k.split(".")
            if name in k_split:
                filtered_res[k] = v
        return filtered_res

def _print_exit_handler(tim: CUDAKernelTimer, name: str, out: Optional[List[float]] = None):
    duration = tim.get_pair_duration(name)
    print(f"{name}: {duration}")
    if out is not None:
        out[0] = duration

def measure_and_print(name: str = "CUDATimer", *, stream: int = 0, out: Optional[List[float]] = None, enable: bool = True):
    tim = KernelTimer(enable=enable)
    return tim._record(name, stream, partial(_print_exit_handler, out=out))

def measure_duration(name: str = "CUDATimer", *, stream: int = 0, enable: bool = True):
    tim = KernelTimer(enable=enable)
    return tim._record(name, stream, measure_time=True)

def measure_and_print_cpu(name: str = "CPUTimer", *, stream: int = 0, out: Optional[List[float]] = None, enable: bool = True):
    tim = KernelTimer(enable=enable, disable_if_cpu_only=False)
    return tim._record(name, stream, partial(_print_exit_handler, out=out))

def measure_duration_cpu(name: str = "CPUTimer", *, stream: int = 0, enable: bool = True):
    tim = KernelTimer(enable=enable, disable_if_cpu_only=False)
    return tim._record(name, stream, measure_time=True)

def get_numpy_view(ten: Tensor) -> np.ndarray:
    if not ten.is_contiguous():
        raise NotImplementedError(
            "numpy_view only support contiguous tv::Tensor")
    buf = ten.get_memoryview()
    return np.frombuffer(buf, dtype=TENSOR_TO_NPDTYPE_MAP[ten.dtype]).reshape(
        ten.shape)


def numpy_view(self):
    return get_numpy_view(self)


Tensor.numpy_view = numpy_view

NPDTYPE_TO_TENSOR_MAP = {
    np.dtype(np.float32): float32,
    np.dtype(np.int32): int32,
    np.dtype(np.int16): int16,
    np.dtype(np.int8): int8,
    np.dtype(np.float64): float64,
    np.dtype(np.bool_): bool_,
    np.dtype(np.uint8): uint8,
    np.dtype(np.float16): float16,
    np.dtype(np.int64): int64,
    np.dtype(np.uint16): uint16,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
}

ALL_TV_TENSOR_DTYPES = set([
    bool_, float16, float32, float64, int8, int16, int32, int64, uint8, uint16,
    uint32, uint64, tf32, bfloat16, custom16, custom32, custom48, custom64, custom80,
    custom96, custom128
])

TENSOR_TO_NPDTYPE_MAP = {v: k for k, v in NPDTYPE_TO_TENSOR_MAP.items()}
TENSOR_TO_NPDTYPE_MAP[tf32] = np.dtype(np.float32)

_SUPPORTED_FILL_INT = {int32, int16, int8, uint32, uint16, uint8, uint64, int64}
_SUPPORTED_FILL_INT_CUDA = {int32, int16, int8, uint32, uint16, uint8}


def zeros(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.zeros(shape, tv_dtype, device, pinned, managed)


def from_blob_strided(ptr: int, shape: List[int], stride: List[int],
                      dtype: Union[np.dtype, int], device: int, storage_offset: int = 0) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_blob(ptr, shape, stride, tv_dtype, device, storage_offset)


def from_const_blob_strided(ptr: int, shape: List[int], stride: List[int],
                            dtype: Union[np.dtype,
                                         int], device: int, storage_offset: int = 0) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_const_blob(ptr, shape, stride, tv_dtype,
                                           device, storage_offset)


def from_blob(ptr: int, shape: List[int], dtype: Union[np.dtype, int],
              device: int, storage_offset: int = 0) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_blob(ptr, shape, tv_dtype, device, storage_offset)


def from_const_blob(ptr: int, shape: List[int], dtype: Union[np.dtype, int],
                    device: int, storage_offset: int = 0) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_const_blob(ptr, shape, tv_dtype, device, storage_offset)


def empty(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.empty(shape, tv_dtype, device, pinned, managed)


def full(shape: List[int],
         val: Union[int, float],
         dtype: Union[np.dtype, int] = np.float32,
         device: int = -1,
         pinned: bool = False,
         managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    if tv_dtype == float32:
        return tensorview_bind.full_float(shape, val, tv_dtype, device, pinned,
                                          managed)
    elif device == -1 and tv_dtype in _SUPPORTED_FILL_INT:
        return tensorview_bind.full_int(shape, val, tv_dtype, device, pinned,
                                        managed)
    elif device != -1 and tv_dtype in _SUPPORTED_FILL_INT_CUDA:
        return tensorview_bind.full_int(shape, val, tv_dtype, device, pinned,
                                        managed)
    else:
        raise NotImplementedError


def zeros_managed(shape: List[int],
                  dtype: Union[np.dtype, int] = np.float32) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.zeros_managed(shape, tv_dtype)


def from_numpy(arr: np.ndarray) -> Tensor:
    return tensorview_bind.from_numpy(arr)


def get_compute_capability(index: int = -1):
    return tensorview_bind.get_compute_capability(index)


def is_cpu_only():
    return tensorview_bind.is_cpu_only()


def cufilt(name: str):
    return tensorview_bind.cufilt(name)

def tvdtype_bitsize(dtype: int) -> int:
    return tensorview_bind.tvdtype_bitsize(dtype)

def tvdtype_itemsize(dtype: int) -> int:
    return tensorview_bind.tvdtype_itemsize(dtype)

def cat_first_axis(tens: List[Tensor]):
    return tensorview_bind.cat_first_axis(tens)