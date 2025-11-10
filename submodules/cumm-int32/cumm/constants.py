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

import os
from pathlib import Path
from typing import List

from ccimport import compat
import subprocess
PACKAGE_NAME = "cumm"
PACKAGE_ROOT = Path(__file__).parent.resolve()

_TENSORVIEW_INCLUDE_PATHS: List[Path] = [
    PACKAGE_ROOT.parent / "include",  # pip dev install
    PACKAGE_ROOT / "include",  # pip package
]

TENSORVIEW_INCLUDE_PATH = _TENSORVIEW_INCLUDE_PATHS[0]
if not TENSORVIEW_INCLUDE_PATH.exists():
    for p in _TENSORVIEW_INCLUDE_PATHS[1:]:
        if p.exists():
            TENSORVIEW_INCLUDE_PATH = p

assert TENSORVIEW_INCLUDE_PATH.exists()

TENSORVIEW_INCLUDE_PATH = str(TENSORVIEW_INCLUDE_PATH)
TENSORVIEW_INCLUDE_PATH = os.getenv("CUMM_INCLUDE_PATH", TENSORVIEW_INCLUDE_PATH)
TENSORVIEW_LIBCUDACXX_PATH = PACKAGE_ROOT / "libcudacxx_include"
TENSORVIEW_IS_CUDA_PREBUILT = TENSORVIEW_LIBCUDACXX_PATH.exists()
CUTLASS_MODE = False
CUTLASS_INPUT_ITER = CUTLASS_MODE and True
CUTLASS_SMEM_WARP_ITER = CUTLASS_MODE and True
CUTLASS_OUTPUT_ITER = CUTLASS_MODE and True
CUTLASS_DEBUG = False

CUMM_CUDA_VERSION = os.getenv("CUMM_CUDA_VERSION", None)
if TENSORVIEW_IS_CUDA_PREBUILT:
    CUMM_CPU_ONLY_BUILD = False 
else:
    try:
        subprocess.check_output(["nvcc", "--version"
                                                ]).decode("utf-8").strip()
        CUMM_CPU_ONLY_BUILD = False 
    except:
        CUMM_CPU_ONLY_BUILD = not compat.IsAppleSiliconMacOs
    if CUMM_CUDA_VERSION is not None:
        CUMM_CPU_ONLY_BUILD = CUMM_CUDA_VERSION.strip() == ""

CUMM_DISABLE_JIT = os.getenv("CUMM_DISABLE_JIT", "0") == "1"

CUMM_MAXIMUM_NVRTC_CONV_NDIM = 3

CUMM_APPLE_METAL_CPP_ROOT = os.getenv("CUMM_APPLE_METAL_CPP_ROOT", None)