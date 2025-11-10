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

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

from cumm.gemm.constants import NVRTCConstants

os.environ["CUMM_DEBUG"] = "1"
# _cudart = ctypes.CDLL('libcudart.so')

import pickle
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pccm
import torch
from pccm.core import CodeFormatter

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.gemm import kernel
from cumm.gemm.algospec.core import ShuffleStrideType
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.main import GemmMainUnitTest, NVRTCMode, gen_gemm_kernels

# def cu_prof_start():
#     ret = _cudart.cudaProfilerStart()
#     if ret != 0:
#         raise Exception('cudaProfilerStart() returned %d' % ret)

# def cu_prof_stop():
#     ret = _cudart.cudaProfilerStop()
#     if ret != 0:
#         raise Exception('cudaProfilerStop() returned %d' % ret)

_SPCONV_ROOT = Path(__file__).parent.parent.parent
_GEMM_ROOT = _SPCONV_ROOT / "src/spgemm/gemmdev"
_REFGEMM_ROOT = _SPCONV_ROOT / "src/spgemm/refops"


def build_gemm_lib(cus: List[pccm.Class]):
    lib = pccm.builder.build_pybind(cus,
                                    Path(__file__).parent / "mygemm_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_unittest",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True,
                                    global_header_only=False,
                                    std="c++17")

    return lib


def build_gather_lib(cus: List[pccm.Class]):
    lib = pccm.builder.build_pybind(cus,
                                    Path(__file__).parent / "gather_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_gather",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True)

    return lib


def build_scatter_lib(cus: List[pccm.Class]):
    lib = pccm.builder.build_pybind(cus,
                                    Path(__file__).parent / "scatter_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_scatter",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True)

    return lib


COUNTBIT_LOOKUP_TABLE = np.array([bin(i).count('1')
                                  for i in range(256)]).astype(np.int32)


def count_set_bits(v):
    'returns the count of set bits in each element of v'
    assert v.dtype in (np.uint8, np.uint16, np.uint32,
                       np.uint64), 'must be an unsigned int dtype'
    return COUNTBIT_LOOKUP_TABLE[np.reshape(v.view(np.uint8),
                                            v.shape + (-1, ))].sum(axis=-1)


from cumm.nvrtc import CummNVRTCModule


def gen_row_hilbert_mat(m, n, dtype):
    ret = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            ret[i, j] = 1./(i + j + 1)
    return ret.astype(dtype)

def gen_row_xhilbert_mat(m, n, dtype):
    ret = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            if j > i:
                ret[i, j] = 1. / (i + j + 1)
            elif j < i:
                ret[i, j] = -1./ (i + j + 1)
            else:
                ret[i, j] = 1
    return ret.astype(dtype)


def Try(m, n, k):
    a = gen_row_xhilbert_mat(m, k, np.float16)
    b = gen_row_xhilbert_mat(k, n, np.float16)
    return a@b


def gen_row_rand_mat(m, n, dtype):
    return np.random.uniform(-1, 1, size=[m, n]).astype(dtype)


def py_matmul(a, b, acc_type, tbk=8):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    ret = np.zeros((m, n), dtype=acc_type)
    for i in range(m):
        for j in range(n):
            acc = np.zeros([(k // tbk) + 1], dtype=acc_type)
            for nk in range(k):
                acc[nk // tbk] += a[i, nk] * b[nk, j]
            accacc = np.zeros([1], dtype=acc_type)
            for l in range(acc.shape[0]):
                accacc[0] += acc[l]
            ret[i, j] = accacc[0]
    return ret

preload_var = {}
def allocate(m, n, bTranspose, dtype):
    if bTranspose:
        m, n = n, m
    key = rf"{m}-{n}-{dtype}"
    if key not in preload_var.keys():
        ndarray = np.random.uniform(-1, 1, size=[m, n]).astype(dtype)
        preload_var[key + '-nd'] = ndarray
        preload_var[key] = tv.from_numpy(ndarray).cuda()
    return preload_var[key]


def _asdv_test_regular_gemm():
    np.random.seed(19260817)
    lib_object = None 
    use_nvrtc = True 
    with cudasim.enter_debug_context(True, 3):
        main_cu = GemmMainUnitTest()
        main_cu.namespace = "cumm.gemm.main"

    if not use_nvrtc:
        lib = build_gemm_lib([main_cu])
        lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = tv.gemm.GemmParams
    algo_cls = tv.gemm.GemmAlgoDesp
    nvrtc_mode = NVRTCMode.ConstantMemory
    params = main_cu.all_params[0]
    m = 1600000 
    n = 256
    k = 256
    # m = max(params.ts[0], m)
    # n = max(params.ts[1], n)
    # k = max(params.ts[2], k)
    '''
    a = gen_row_xhilbert_mat(m, k, dtypes.get_npdtype(params.dtype_a))
    b = gen_row_xhilbert_mat(k, n, dtypes.get_npdtype(params.dtype_b))
    '''

    print("PREPARE FINISH")
    os.system('clear')
    for params in main_cu.all_params:
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue
        ker = gen_gemm_kernels(params, nvrtc_mode=nvrtc_mode)
        ker.namespace = "wtf"
        t = time.time()
        custom_names = []
        if nvrtc_mode == NVRTCMode.ConstantMemory:
            custom_names = [f"&wtf::{NVRTCConstants.CONSTANT_PARAM_KEY}"]
        a_tv = allocate(m, k, params.trans_a, dtypes.get_npdtype(params.dtype_a))
        b_tv = allocate(k, n, params.trans_b, dtypes.get_npdtype(params.dtype_b))
        c_tv = allocate(m, n, params.trans_c, dtypes.get_npdtype(params.dtype_c))

        mod = CummNVRTCModule(
            [ker],
            # cudadevrt_path="/usr/local/cuda/lib64/libcudadevrt.a",
            verbose=False,
            verbose_path= "cop_out",
            custom_names=custom_names)
        # print(mod.get_ptx())

        mod.load()
        v = mod.get_kernel_attrs(mod.get_lowered_name("wtf::gemm_kernel"))
        print("registers: ", v['num_regs'])
        print(mod.kernels)
        print("RTC COMPILE TIME", time.time() - t)

        # print("DATA GEN FINISH")
        # print("WTF PREPARED")
        if params.splitk_serial:
            ksplit = 16
        else:
            ksplit = 1
        # print("CUDA PREPARED")
        algo = algo_cls()
        algo.tile_shape = params.ts
        algo.warp_tile_shape = params.wts
        algo.num_stage = params.num_stage
        algo.dacc = params.dtype_acc.tv_dtype
        algo.dcomp = params.dtype_comp.tv_dtype
        algo.algo = params.algo.value
        algo.trans_a = params.trans_a
        algo.trans_b = params.trans_b
        algo.trans_c = params.trans_c
        if params.tensorop is not None:
            algo.tensorop = params.tensorop.shape
        params_cpp = params_cls()
        params_cpp.algo_desp = algo
        params_cpp.split_k_slices = ksplit

        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv
        params_cpp.beta = 1.0
        params_cpp.alpha = 1.0

        nvrtc_params = tv.gemm.NVRTCParams()
        nvrtc_params.cumodule = mod.get_cpp_object()
        nvrtc_params.mode = nvrtc_mode.value
        nvrtc_params.num_threads = ker.num_threads
        nvrtc_params.smem_size = ker.smem_size
        if nvrtc_mode == NVRTCMode.DynamicParallism:
            nvrtc_params.kernel_name = mod.get_lowered_name(
                "wtf::nvrtc_kernel")

        elif nvrtc_mode == NVRTCMode.KernelAndCPU:
            nvrtc_params.kernel_name = mod.get_lowered_name("wtf::gemm_kernel")
            nvrtc_params.init_kernel_name = mod.get_lowered_name(
                "wtf::nvrtc_kernel_cpu_out")
            nvrtc_params.param_size = mod.const_values[
                f"wtf::{NVRTCConstants.SIZEOF_KEY}"]

            nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8, 0)
            nvrtc_params.param_storage_cpu = tv.empty(
                [nvrtc_params.param_size], tv.uint8, -1, pinned=True)

        elif nvrtc_mode == NVRTCMode.Direct:
            nvrtc_params.kernel_name = mod.get_lowered_name("wtf::gemm_kernel")
        elif nvrtc_mode == NVRTCMode.ConstantMemory:
            nvrtc_params.kernel_name = mod.get_lowered_name("wtf::gemm_kernel")
            nvrtc_params.init_kernel_name = mod.get_lowered_name(
                "wtf::nvrtc_kernel_cpu_out")
            nvrtc_params.param_size = mod.const_values[
                f"wtf::{NVRTCConstants.SIZEOF_KEY}"]
            nvrtc_params.constant_name = mod.get_lowered_name(
                f"&wtf::{NVRTCConstants.CONSTANT_PARAM_KEY}")
            nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8, 0)
        else:
            raise NotImplementedError
        for i in range(2):
            clock0 = time.time()
            if lib_object is not None:
                lib_object.matmul2(params_cpp)
            else:
                params_cpp.nvrtc_params = nvrtc_params
                with tv.measure_and_print():
                    tv.gemm.run_nvrtc_gemm_kernel(params_cpp)
            use_time = time.time() - clock0
            
            # c_cpu = c_tv.cpu().numpy()

            # print("MaxError:  ",np.abs(c_cpu / np.abs(c).max()).max())
            # if(np.abs(c_cpu / np.abs(c).max()).max() < 5e-3):
            print(params_cpp.algo_desp.__str__(), "PASSED", " Use time ", use_time)
        # else:
        #     print(params_cpp.algo_desp.__str__(), "NOT PASS")
        #     assert(0)
        # print(c_cpu.reshape(-1)[-16:])
        # print(c.reshape(-1)[-16:])
        
        # print(params_cpp.algo_desp, a.mean(), b.mean(), c.mean(),
         #      np.sum(np.abs(c_cpu - c)))



if __name__ == "__main__":
    # _asdv_test_simt_shuffle()
    # _asdv_test_simt_debug()
    _asdv_test_regular_gemm()
    # _test_gather()
    # _test_scatter()
