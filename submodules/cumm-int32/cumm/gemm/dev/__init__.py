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

from cumm.gemm.constants import NVRTCConstants, NVRTCMode
from cumm.gemm.kernel import GemmKernel
from cumm import dtypes
import numpy as np 
from cumm import tensorview as tv
from cumm.nvrtc import CummNVRTCModule 

class GemmHelper:
    def __init__(self, ker: GemmKernel, m: int, n: int, k: int) -> None:
        m = max(ker.tile_shape[0], m)
        n = max(ker.tile_shape[1], n)
        k = max(ker.tile_shape[2], k)
        if ker.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
            dtype_c = ker.dtype_c.npdtype()
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtype_c)
        else:
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(ker.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(ker.dtype_b))
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(ker.dtype_c))
        if ker.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if ker.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if ker.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        self.m = m 
        self.n = n 
        self.k = k 
        self.a = a
        self.b = b 
        self.c = c 
        self.ker = ker

    def get_operands(self):
        a_tv = tv.from_numpy(self.a).cuda()
        b_tv = tv.from_numpy(self.b).cuda()

        c_tv = tv.zeros(list(self.c.shape), self.ker.dtype_c.tv_dtype, 0)

        return a_tv, b_tv, c_tv

    def get_params(self, ksplit: int = 1):
        a_tv = tv.from_numpy(self.a).cuda()
        b_tv = tv.from_numpy(self.b).cuda()

        c_tv = tv.zeros(list(self.c.shape), self.ker.dtype_c.tv_dtype, 0)

        algo = tv.gemm.GemmAlgoDesp()
        algo.tile_shape = self.ker.tile_shape
        algo.warp_tile_shape = self.ker.warp_tile_shape
        algo.num_stage = self.ker.num_stage
        algo.dacc = self.ker.dtype_acc.tv_dtype
        algo.dcomp = self.ker.dtype_comp.tv_dtype
        algo.algo = self.ker.algo.value
        algo.trans_a = self.ker.trans_a
        algo.trans_b = self.ker.trans_b
        algo.trans_c = self.ker.trans_c
        if self.ker.tensorop is not None:
            algo.tensorop = self.ker.tensorop.shape
        params_cpp = tv.gemm.GemmParams()
        params_cpp.algo_desp = algo
        params_cpp.split_k_slices = ksplit

        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv

        return params_cpp

    def get_nvrtc_params(self, nvrtc_mode: NVRTCMode, mod: CummNVRTCModule, gemm_ns: str):

        nvrtc_params = tv.gemm.NVRTCParams()
        nvrtc_params.cumodule = mod.get_cpp_object()
        nvrtc_params.mode = nvrtc_mode.value
        nvrtc_params.num_threads = self.ker.num_threads
        nvrtc_params.smem_size = self.ker.smem_size
        if nvrtc_mode == NVRTCMode.DynamicParallism:
            nvrtc_params.kernel_name = mod.get_lowered_name(
                f"{gemm_ns}::nvrtc_kernel")

        elif nvrtc_mode == NVRTCMode.KernelAndCPU:
            nvrtc_params.kernel_name = mod.get_lowered_name(f"{gemm_ns}::gemm_kernel")
            nvrtc_params.init_kernel_name = mod.get_lowered_name(
                "f{gemm_ns}::nvrtc_kernel_cpu_out")
            nvrtc_params.param_size = mod.const_values[
                f"{gemm_ns}::{NVRTCConstants.SIZEOF_KEY}"]

            nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8, 0)
            nvrtc_params.param_storage_cpu = tv.empty(
                [nvrtc_params.param_size], tv.uint8, -1, pinned=True)

        elif nvrtc_mode == NVRTCMode.Direct:
            nvrtc_params.kernel_name = mod.get_lowered_name(f"{gemm_ns}::gemm_kernel")
        elif nvrtc_mode == NVRTCMode.ConstantMemory:
            nvrtc_params.kernel_name = mod.get_lowered_name(f"{gemm_ns}::gemm_kernel")
            nvrtc_params.init_kernel_name = mod.get_lowered_name(
                f"{gemm_ns}::nvrtc_kernel_cpu_out")
            nvrtc_params.param_size = mod.const_values[
                f"{gemm_ns}::{NVRTCConstants.SIZEOF_KEY}"]
            nvrtc_params.constant_name = mod.get_lowered_name(
                f"&{gemm_ns}::{NVRTCConstants.CONSTANT_PARAM_KEY}")
            nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8, 0)
        else:
            raise NotImplementedError
        return nvrtc_params