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


WARP_SIZE = 32
OPTIM_ACCESS = 16
OPTIM_ACCESS_BITS = OPTIM_ACCESS * 8
SMEM_BANK_SIZE_BITS = 128 * 8

class NVRTCConstants:
    SIZEOF_KEY = "kSizeOfParams"
    SMEM_KEY = "kSmemSize"
    NUM_THREADS_KEY = "kNumThreads"
    CONSTANT_PARAM_KEY = "params_raw"

class NVRTCMode(enum.Enum):
    """nvrtc mode for *NON-STATIC* gemm kernels.
    kernel params of gemm contains different init code that need jit
    if we doesn't provide a static init function.
    another solution is implement all param calculation in c++/python.
    this method requires additional time to maintain code, so I
    have no interest on it.
    """
    Disabled = 0
    # calc params directly in kernel. VERY SLOW.
    Direct = 1
    # launch a kernel, calculate params, then launch gemm kernel
    # in that kernel.
    # greatly slower than KernelAndCPU/ConstantMemory, I don't know why.
    DynamicParallism = 2
    # run init kernel first to generate params, copy params to cpu, then use
    # that param to launch kernel in host.
    KernelAndCPU = 3
    # similar to KernelAndCPU, don't need dev to cpu copy, copy result to 
    # Constant Memory instead.
    # fastest way, but DON'T SUPPORT MULTIPLE STREAM. the
    # constant memory is allocated once when nvrtc
    # module is created. so create new nvrtc module
    # for every stream.
    ConstantMemory = 4
    # static mode, only support implemented input iterators. NOT IMPLEMENTED
    Static = 5  

