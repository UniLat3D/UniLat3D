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


def _asdv_test_simt():
    # /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --profile-from-start off --analysis-metrics -f --csv python /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --print-gpu-trace  --analysis-metrics -f --csv /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()

    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams

    for params in main_cu.simt_params:
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue
        ker = gen_gemm_kernels(params)
        if params.algo != kernel.GemmAlgo.SimtDP4A:
            m = 256 + 32
            n = 256 + 40
            k = 136
            m *= 2
            n *= 2
            k *= 2
            m = 316
            n = 235
            k = 658
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
        else:
            m = 256 + 32
            n = 256 + 40
            k = 136
            # m *= 2
            # n *= 2
            # k *= 2
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
        # print("DATA GEN FINISH")
        c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        # print("WTF PREPARED")
        # cu_prof_start()
        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        c_tv = tv.zeros(c.shape, params.dtype_c.tv_dtype, 0)
        params_cpp = params_cls()
        params_cpp.tile_shape = params.ts
        params_cpp.warp_tile_shape = params.wts
        params_cpp.num_stage = params.num_stage
        params_cpp.dacc = params.dtype_acc.tv_dtype
        params_cpp.dcomp = params.dtype_comp.tv_dtype
        params_cpp.algo = params.algo.value
        params_cpp.split_k_slices = 1
        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv
        params_cpp.trans_a = params.trans_a
        params_cpp.trans_b = params.trans_b
        params_cpp.trans_c = params.trans_c

        # print("CUDA PREPARED")
        lib_object.matmul2(params_cpp)
        c_cpu = c_tv.cpu().numpy()
        # cu_prof_stop()

        print(params.get_algo_name(), np.linalg.norm(c_cpu - c))


def _asdv_test_simt_shuffle():
    # /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --profile-from-start off --analysis-metrics -f --csv python /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --print-gpu-trace  --analysis-metrics -f --csv /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    main_cu.namespace = "cumm.gemm.main"
    # gather_cu = GatherKernel(dtypes.float32, seq(32, 128), 1, 512)
    # gather_cu.namespace = "rtx"
    lib = build_gemm_lib([main_cu])
    print("BUILD FINISHTED")

    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams
    algo_cls = lib.cumm.gemm.main.GemmAlgoDesp
    # gather_lib = lib_object.rtx.GatherKernel()

    for params, ker in zip(main_cu.all_params, main_cu.all_kernels):
        if params.shuffle_stride == ShuffleStrideType.NoShuffle:
            continue
        print("RTX", params.shuffle_stride, params.trans_a, params.trans_b,
              params.trans_c)
        for kk in range(1):
            m = 64000
            n = 128
            k = 128
            a_inds = np.arange(m, dtype=np.int32)
            c_inds = np.arange(m, dtype=np.int32)
            np.random.shuffle(a_inds)
            np.random.shuffle(c_inds)

            if params.dtype_a == dtypes.int8:
                a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
                if params.shuffle_stride != ShuffleStrideType.ShuffleAB:
                    b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
                    c_tmp = (a[a_inds].astype(np.float32) @ b.astype(
                        np.float32)).astype(dtypes.get_npdtype(params.dtype_c))
                    c = np.zeros((m, n), dtype=a.dtype)
                    c[c_inds] = c_tmp
                else:
                    b = np.random.randint(-2, 2, size=[m, n]).astype(np.int8)
                    c = (a[a_inds].T.astype(np.float32) @ b[c_inds].astype(
                        np.float32)).astype(dtypes.get_npdtype(params.dtype_c))

            else:
                a = np.random.uniform(-1, 1, size=[m, k]).astype(
                    dtypes.get_npdtype(params.dtype_a))
                if params.shuffle_stride != ShuffleStrideType.ShuffleAB:
                    b = np.random.uniform(-1, 1, size=[k, n]).astype(
                        dtypes.get_npdtype(params.dtype_b))
                    c_tmp = (a[a_inds].astype(np.float32) @ b.astype(
                        np.float32)).astype(dtypes.get_npdtype(params.dtype_c))
                    c = np.zeros((m, n), dtype=a.dtype)
                    c[c_inds] = c_tmp
                else:
                    b = np.random.uniform(-1, 1, size=[m, n]).astype(
                        dtypes.get_npdtype(params.dtype_b))
                    c = (a[a_inds].T.astype(np.float32) @ b[c_inds].astype(
                        np.float32)).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_b:
                b = np.ascontiguousarray(b.T)

            a_tv = tv.from_numpy(a).cuda()
            b_tv = tv.from_numpy(b).cuda()
            c_tv = tv.from_numpy(c).cuda()
            a_inds_tv = tv.from_numpy(a_inds).cuda()
            c_inds_tv = tv.from_numpy(c_inds).cuda()
            if params.splitk_serial:
                ksplit = 32
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
            algo.shuffle_type = params.shuffle_stride.value
            algo.trans_a = params.trans_a
            algo.trans_b = params.trans_b
            algo.trans_c = params.trans_c
            algo.element_per_access_a = ker.input_spec.input_sub_tile_shape_a[
                1]
            algo.element_per_access_b = ker.input_spec.input_sub_tile_shape_b[
                1]
            algo.element_per_access_c = ker.output_spec.out_iter.element_per_acc
            algo.split_k_serial = params.splitk_serial

            if params.tensorop is not None:
                algo.tensorop = params.tensorop.shape
            params_cpp = params_cls()
            params_cpp.algo_desp = algo
            params_cpp.split_k_slices = ksplit
            params_cpp.a = a_tv
            params_cpp.b = b_tv
            params_cpp.c = c_tv
            params_cpp.beta = 0.0
            for i in range(3):
                # gather_lib.gather(a_gout_tv, a_tv, a_inds_tv)
                # a_gout_tv_cpu = a_gout_tv.cpu().numpy()
                # print(np.linalg.norm(a_gout_tv_cpu - a[a_inds]))
                # c_tv.zero_()
                # lib_object.shuffle_matmul_ref(c_tv_f32, a_tv_f32, b_tv_transposed, a_inds_tv, c_inds_tv, a_inds_tv.dim(0))
                t = time.time()
                if params.shuffle_stride == ShuffleStrideType.ShuffleAB:
                    params_cpp.a_inds = a_inds_tv
                    params_cpp.b_inds = c_inds_tv
                    # c_tv.zero_()
                    lib_object.matmul2(params_cpp)
                else:
                    # c_tv.zero_()
                    params_cpp.a_inds = a_inds_tv
                    params_cpp.c_inds = c_inds_tv
                    lib_object.matmul2(params_cpp)
                assert params.get_algo_name() == ker.get_algo_name()
                # print(a_tv.shape, b_tv.shape, c_tv.shape)
                lib_object.device_synchronize()
                duration = time.time() - t
                c_cpu = c_tv.cpu().numpy()
                # cu_prof_stop()
                print(ksplit, params.dtype_c, params.get_algo_name(), duration,
                      np.linalg.norm(c_cpu - c))


def _asdv_test_simt_shuffle_debug():
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    main_cu.namespace = "cumm.gemm.main"
    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams
    algo_cls = lib.cumm.gemm.main.GemmAlgoDesp
    # gather_lib = lib_object.rtx.GatherKernel()
    with open("/home/yy/test_debug.pkl", "rb") as f:
        import pickle
        a, b, c, a_inds, b_inds = pickle.load(f)
        a_inds = np.arange(a_inds.shape[0]).astype(np.int32)
        b_inds = np.arange(b_inds.shape[0]).astype(np.int32)

        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        c_tv = tv.from_numpy(c).cuda()
        a_inds_tv = tv.from_numpy(a_inds).cuda()
        b_inds_tv = tv.from_numpy(b_inds).cuda()
        print(a_inds, b_inds)

    for params, ker in zip(main_cu.all_params, main_cu.all_kernels):
        if params.shuffle_stride == ShuffleStrideType.NoShuffle:
            continue
        print("RTX", params.shuffle_stride, params.trans_a, params.trans_b,
              params.trans_c)

        for kk in range(100):
            if params.splitk_serial:
                ksplit = 32
            else:
                continue
                ksplit = 1
            c_ = c_tv.clone()

            # print("CUDA PREPARED")
            algo = algo_cls()
            algo.tile_shape = params.ts
            algo.warp_tile_shape = params.wts
            algo.num_stage = params.num_stage
            algo.dacc = params.dtype_acc.tv_dtype
            algo.dcomp = params.dtype_comp.tv_dtype
            algo.algo = params.algo.value
            algo.shuffle_type = params.shuffle_stride.value
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
            params_cpp.c = c_
            params_cpp.beta = 0.0
            params_cpp.a_inds = a_inds_tv
            params_cpp.b_inds = b_inds_tv
            # c_tv.zero_()
            lib_object.matmul2(params_cpp)
            # print(a_tv.shape, b_tv.shape, c_tv.shape)
            c_cpu = c_tv.cpu().numpy()
            # cu_prof_stop()

            print(ksplit, params.dtype_c, params.get_algo_name(),
                  np.linalg.norm(c_cpu - c))


def _asdv_test_simt_debug():
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    main_cu.namespace = "cumm.gemm.main"
    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams
    algo_cls = lib.cumm.gemm.main.GemmAlgoDesp
    # gather_lib = lib_object.rtx.GatherKernel()
    with open("/home/yy/test_debug.pkl", "rb") as f:
        import pickle
        a, b, c, a_inds, b_inds = pickle.load(f)
        nhot = len(a_inds)
        a_inds = np.arange(a_inds.shape[0]).astype(np.int32)
        b_inds = np.arange(b_inds.shape[0]).astype(np.int32)
        a = a[:nhot]
        b = b[:nhot]
        print(a.shape, b.shape, c.shape)
        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        _asd = tv.zeros([5000], device=0)
        c_tv = tv.from_numpy(c).cuda()
        print(c.shape)
        raise NotImplementedError
        a_inds_tv = tv.from_numpy(a_inds).cuda()
        b_inds_tv = tv.from_numpy(b_inds).cuda()

        print(a_inds, b_inds)

    for params, ker in zip(main_cu.all_params, main_cu.all_kernels):
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue
        print("RTX", params.shuffle_stride, params.trans_a, params.trans_b,
              params.trans_c)

        for kk in range(100):
            if params.splitk_serial:
                ksplit = 32
            else:
                continue
                ksplit = 1
            # c_ = c_tv.clone()

            # print("CUDA PREPARED")
            algo = algo_cls()
            algo.tile_shape = params.ts
            algo.warp_tile_shape = params.wts
            algo.num_stage = params.num_stage
            algo.dacc = params.dtype_acc.tv_dtype
            algo.dcomp = params.dtype_comp.tv_dtype
            algo.algo = params.algo.value
            algo.shuffle_type = params.shuffle_stride.value
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
            params_cpp.beta = 0.0
            params_cpp.a_inds = a_inds_tv
            params_cpp.b_inds = b_inds_tv
            # c_tv.zero_()
            lib_object.matmul2(params_cpp)
            # print(a_tv.shape, b_tv.shape, c_tv.shape)
            c_cpu = c_tv.cpu().numpy()
            # cu_prof_stop()

            print(ksplit, params.dtype_c, params.get_algo_name(),
                  np.linalg.norm(c_cpu - c))


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
    a = tv.zeros([3], tv.int32, 0)
    for params in main_cu.all_params:
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue
        ker = gen_gemm_kernels(params, nvrtc_mode=nvrtc_mode)
        ker.namespace = "wtf"
        t = time.time()
        custom_names = []
        if nvrtc_mode == NVRTCMode.ConstantMemory:
            custom_names = [f"&wtf::{NVRTCConstants.CONSTANT_PARAM_KEY}"]

        mod = CummNVRTCModule(
            [ker],
            # cudadevrt_path="/usr/local/cuda/lib64/libcudadevrt.a",
            verbose=False,
            # verbose_path="/home/yy/Projects/cumm/build/dev_cumm",
            custom_names=custom_names)
        # with open("/home/yy/Projects/cumm/build/dev_cumm/kernel.ptx", "w") as f:
        #     f.write(mod.get_ptx())
        # print(mod.get_ptx())
        # breakpoint()
        mod.load()
        print(mod.kernels)
        print("RTC COMPILE TIME", time.time() - t)
        # print(mod.kernels)
        # breakpoint()
        m = 256 + 32
        n = 256 + 40
        k = 136
        m *= 2
        n *= 2
        k *= 2
        m = 64
        n = 64
        k = 32
        m = max(params.ts[0], m)
        n = max(params.ts[1], n)
        k = max(params.ts[2], k)
        if params.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
            dtype_c = params.dtype_c.npdtype()
            c = -(a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        else:
            '''
            a = gen_row_xhilbert_mat(m, k, dtypes.get_npdtype(params.dtype_a))
            b = gen_row_xhilbert_mat(k, n, dtypes.get_npdtype(params.dtype_b))
            '''
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            c = -py_matmul(a, b, dtypes.get_npdtype(params.dtype_c))
            raw_a = a
            raw_b = b
            raw_c = c
            '''
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            # a[:, 32:] = 0
            # b[32:] = 0
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
            '''
        # print("DATA GEN FINISH")
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
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
        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()

        d = c.copy()
        c_tv = tv.from_numpy(d).cuda()

        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv
        params_cpp.beta = 1.0
        params_cpp.alpha = 1.0
        params_cpp.act_type = tv.gemm.Activation.None_

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
        clock0 = time.time()
        if lib_object is not None:
            lib_object.matmul2(params_cpp)
        else:
            params_cpp.nvrtc_params = nvrtc_params
            with tv.measure_and_print():
                tv.gemm.run_nvrtc_gemm_kernel(params_cpp)
        use_time = time.time() - clock0

        c_cpu = c_tv.cpu().numpy()
    
        print("MaxError:  ",np.abs(c_cpu / np.abs(c).max()).max())
        if(np.abs(c_cpu / np.abs(c).max()).max() < 5e-3):
            print(params_cpp.algo_desp.__str__(), "PASSED")
        else:
            print(params_cpp.algo_desp.__str__(), "NOT PASS")
            assert(0)
        # print(c_cpu.reshape(-1)[-16:])
        # print(c.reshape(-1)[-16:])
        
        # print(params_cpp.algo_desp, a.mean(), b.mean(), c.mean(),
         #      np.sum(np.abs(c_cpu - c)))


def _asdv_test_turing():
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    for params in main_cu.turing_params:
        m = 256 + 32
        n = 256 + 40
        k = 136
        m *= 2
        n *= 2
        k *= 2
        m = 1024
        n = 1024
        k = 1024
        m = max(params.ts[0], m)
        n = max(params.ts[1], n)
        k = max(params.ts[2], k)
        # print("HAHAHA PREPARED")

        if params.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
            dtype_c = params.dtype_c.npdtype()
            c = (a.astype(dtype_c) @ b.astype(dtype_c)).astype(
                dtypes.get_npdtype(params.dtype_c))
        else:
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        # print("WTF PREPARED")

        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        c_tv = tv.zeros(c.shape, params.dtype_c.tv_dtype, 0)
        # print("CUDA PREPARED")
        lib_object.matmul(a_tv,
                          b_tv,
                          c_tv,
                          params.trans_a,
                          params.trans_b,
                          params.trans_c,
                          ts=params.ts,
                          wts=params.wts,
                          num_stage=params.num_stage,
                          dacc=params.dtype_acc.tv_dtype,
                          dcomp=params.dtype_comp.tv_dtype,
                          algo=params.algo.value,
                          tensorop=params.tensorop.shape)
        c_cpu = c_tv.cpu().numpy()
        print(params.get_algo_name(), np.sum(np.abs(c_cpu - c)))


def _test_gather1():
    import torch

    np.random.seed(12315)
    main_cu = Gather(seq(4, 128 * 4), 8, 256)
    main_cu.namespace = "cumm.gemm.gather"
    lib = build_gather_lib([main_cu])
    lib_object = lib.cumm.gemm.gather.Gather()
    m = 64000
    k = 126
    a = np.random.uniform(-2, 2, size=[m, k]).astype(np.float32)
    a_tv = tv.from_numpy(a).cuda()
    c_tv = a_tv.clone().zero_()
    inds = np.arange(m, dtype=np.int32)
    np.random.shuffle(inds)
    c_ref = a[inds]

    inds_tv = tv.from_numpy(inds).cuda()
    for i in range(5):
        t = time.time()
        lib_object.gather(c_tv, a_tv, inds_tv)
        # lib_object.stream_synchronize()
        torch.cuda.synchronize()
        print(time.time() - t)
    c_cpu = c_tv.cpu().numpy()
    print(np.linalg.norm(c_ref - c_cpu))


def _test_gather():
    from cumm.gemm.gather import Gather, GatherAll, GatherKernel, ScatterAll

    np.random.seed(12315)
    main_cu = GatherAll()
    # main_cu = GatherV2(seq(8, 128 * 4), 16, 256)
    # main_cu.namespace = "cumm.gemm.gather"
    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.gather.GatherAll()
    m = 64000
    k = 128
    a = np.random.uniform(-2, 2, size=[m, k]).astype(np.float32)
    a_tv = tv.from_numpy(a).cuda()
    c_tv = a_tv.clone().zero_()
    inds = np.arange(m, dtype=np.int32)
    np.random.shuffle(inds)
    c_ref = a[inds]

    inds_tv = tv.from_numpy(inds).cuda()
    times = []
    all_params = lib_object.get_all_gather_params()
    for p in all_params:
        if lib_object.supported(p[2], a_tv.dim(1), a_tv.dtype):
            t = time.time()
            lib_object.gather(c_tv, a_tv, inds_tv, *p)
            lib_object.stream_synchronize()
            times.append(time.time() - t)
    print(times)
    best_params = all_params[np.argmin(times)]
    print(best_params)
    for i in range(5):
        lib_object.stream_synchronize()

        t = time.time()
        # lib_object.gather2(c_tv, a_tv, inds_tv)
        lib_object.gather(c_tv, a_tv, inds_tv, *best_params)
        lib_object.stream_synchronize()

        print("G1", time.time() - t)
        lib_object.stream_synchronize()
        t = time.time()
        lib_object.gather2(c_tv, a_tv, inds_tv)
        lib_object.stream_synchronize()
        print("G2", time.time() - t)

    c_cpu = c_tv.cpu().numpy()
    print(np.linalg.norm(c_ref - c_cpu))


def _test_gather_pth():
    import torch
    np.random.seed(12315)
    m = 64000
    k = 128
    a = np.random.uniform(-2, 2, size=[m, k]).astype(np.float32)
    inds = np.arange(m, dtype=np.int64)
    np.random.shuffle(inds)

    a_th = torch.from_numpy(a).cuda()
    inds_th = torch.from_numpy(inds).cuda()

    for i in range(10):
        t = time.time()
        x = a_th[inds_th]
        torch.cuda.synchronize()

        print(time.time() - t)


def _test_scatter():
    np.random.seed(12315)
    main_cu = ScatterAll()
    # main_cu = GatherV2(seq(8, 128 * 4), 16, 256)
    # main_cu.namespace = "cumm.gemm.gather"
    lib = build_scatter_lib([main_cu])
    lib_object = lib.cumm.gemm.gather.ScatterAll()
    m = 64000
    k = 256
    a = np.random.uniform(-2, 2, size=[m, k]).astype(np.float32)
    a_tv = tv.from_numpy(a).cuda()
    c_tv = a_tv.clone().zero_()
    inds = np.arange(m, dtype=np.int32)
    np.random.shuffle(inds)
    c_ref = np.zeros_like(a)
    c_ref[inds] = a

    inds_tv = tv.from_numpy(inds).cuda()
    times = []
    all_params = lib_object.get_all_scatter_params()
    for p in all_params:
        if lib_object.supported_scatter(*p, a_tv.dim(1), a_tv.dtype):
            t = time.time()
            lib_object.scatter(c_tv, a_tv, inds_tv, *p)
            lib_object.stream_synchronize()
            times.append(time.time() - t)
    print(times)
    best_params = all_params[np.argmin(times)]
    print(best_params)
    for i in range(5):
        c_tv.zero_()
        lib_object.stream_synchronize()

        t = time.time()
        # lib_object.gather2(c_tv, a_tv, inds_tv)
        lib_object.scatter(c_tv, a_tv, inds_tv, *best_params)
        lib_object.stream_synchronize()
        print("G1", time.time() - t)
        c_tv.zero_()
        lib_object.stream_synchronize()
        t = time.time()
        lib_object.scatter2(c_tv, a_tv, inds_tv)
        lib_object.stream_synchronize()
        print("G2", time.time() - t)

    c_cpu = c_tv.cpu().numpy()
    # print(c_cpu)
    # print(c_ref)
    print(np.linalg.norm(c_ref - c_cpu))


if __name__ == "__main__":
    # _asdv_test_simt_shuffle()
    # _asdv_test_simt_debug()
    _asdv_test_regular_gemm()
    # _test_gather()
    # _test_scatter()
