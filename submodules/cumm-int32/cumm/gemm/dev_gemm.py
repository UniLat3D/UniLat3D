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

from cumm.gemm.algospec.core import TensorOp
from cumm.gemm.constants import NVRTCConstants, NVRTCMode
from cumm.gemm.dev import GemmHelper
from cumm.gemm.main import gen_gemm_params, gen_gemm_kernels
from cumm.gemm import kernel 
from cumm import cudasim
from cumm.nvrtc import CummNVRTCModule, get_cudadevrt_path
from cumm import tensorview as tv 
import numpy as np 
from cumm.gemm.cutlasstest import cutlass_test_gemm, GemmAlgoParams, CutlassGemm

def cutlass_test_tf32():
    params = GemmAlgoParams((128, 128, 16), (64, 64, 16), 2,
                            "f32,f32,f32,f32,f32", False, True, False,
                            kernel.GemmAlgo.Ampere, TensorOp((16, 8, 8)))
    main_cu = CutlassGemm(params, 128, "Sm80")
    cutlass_test_gemm(main_cu)


def dev_tf32():
    top = TensorOp((16, 8, 8), "tf32,tf32,f32")
    params = gen_gemm_params((128, 128, 16),
                        (64, 64, 16), 2, "f32,f32,f32,f32,f32",
                        kernel.GemmAlgo.Turing, top)[0]

    # top = TensorOp((16, 8, 8), "f16,f16,f16")
    # params = gen_gemm_params((128, 128, 32),
    #                     (64, 64, 32), 2, "f16,f16,f16,f16,f16",
    #                     kernel.GemmAlgo.Turing, top)[0]
    # ref: [4, 1], [8, 16]
    nvrtc_mode = NVRTCMode.ConstantMemory

    with cudasim.enter_debug_context(True, 0):
        cutlass_test_tf32()
        ker = gen_gemm_kernels(params, nvrtc_mode=nvrtc_mode)
    print("start")

    ker.namespace = "wtf"
    custom_names = []
    if nvrtc_mode == NVRTCMode.ConstantMemory:
        custom_names = [f"&{ker.namespace}::{NVRTCConstants.CONSTANT_PARAM_KEY}"]
    with cudasim.enter_debug_context(True, 0):
        
        with tv.measure_and_print("RTC Compile Time"):
            mod = CummNVRTCModule(
                [ker],
                cudadevrt_path=str(get_cudadevrt_path()),
                verbose=False,
                custom_names=custom_names)
            mod.load()
    np.random.seed(12315)
    helper = GemmHelper(ker, 128, 128, 8)

    params_cpp = helper.get_params()
    nvrtc_params = helper.get_nvrtc_params(nvrtc_mode, mod, ker.namespace)
    params_cpp.nvrtc_params = nvrtc_params
    with tv.measure_and_print("Gemm Time"):
        tv.gemm.run_nvrtc_gemm_kernel(params_cpp)
    c_cpu = params_cpp.c.cpu().numpy()
    print(ker.get_algo_name(), helper.a.mean(), helper.b.mean(), helper.c.mean(),
            np.linalg.norm(c_cpu - helper.c))

if __name__ == "__main__":
    dev_tf32()
