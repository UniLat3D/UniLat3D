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
import sys
from pathlib import Path

# _cudart = ctypes.CDLL('libcudart.so')
from codeai.distributed.example.file import upload_file

print(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import pccm
import spconv

from cumm import cudasim
from cumm import tensorview as tv
from cumm.gemm import kernel
from cumm.gemm.main import GemmMainUnitTest, gen_gemm_kernels

# def cu_prof_start():
#     ret = _cudart.cudaProfilerStart()
#     if ret != 0:
#         raise Exception('cudaProfilerStart() returned %d' % ret)

# def cu_prof_stop():
#     ret = _cudart.cudaProfilerStop()
#     if ret != 0:
#         raise Exception('cudaProfilerStop() returned %d' % ret)

print(spconv.__file__)
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from spconv.constants import SPCONV_ROOT

_SPCONV_ROOT = Path(__file__).parent.parent.parent
_GEMM_ROOT = _SPCONV_ROOT / "src/spgemm/gemmdev"
_REFGEMM_ROOT = _SPCONV_ROOT / "src/spgemm/refops"
from pccm.core import CodeFormatter


# from myclang import clangformat
class ClangFormat(CodeFormatter):
    def __call__(self, code: str):
        return clangformat.clang_format(code)


from cumm import dtypes


def spgemm_build_exec():
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    output = Path(__file__).parent / "spgemm_test_exec"

    pccm.builder.build_library([main_cu],
                               Path(__file__).parent / "spgemm_test_exec",
                               includes=[
                                   SPCONV_ROOT / "include",
                               ],
                               namespace_root=SPCONV_ROOT / "spconv",
                               build_dir=Path(__file__).parent / "build" /
                               "build_spgemm_exec",
                               verbose=False,
                               shared=False,
                               main_file_suffix=".cu",
                               std="c++14")
    return output


def spgemm_profile_win():
    out_file_name = spgemm_build_exec()
    ncu_sections = [
        "ComputeWorkloadAnalysis", "InstructionStats", "LaunchStats",
        "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Chart",
        "MemoryWorkloadAnalysis_Tables", "Occupancy", "SchedulerStats",
        "SourceCounters", "SpeedOfLight", "WarpStateStats"
    ]
    section_flags = sum([["--section", s] for s in ncu_sections], [])
    cmds = [
        "ncu", "-o",
        str((out_file_name).parent / "profile_spgemm"), *section_flags, "-f",
        str(out_file_name) + ".exe"
    ]
    subprocess.check_call(cmds, env=os.environ, shell=True)
    target = "/home/yy"
    upload_file("127.0.0.1:50073",
                Path(__file__).parent / "profile_spgemm.ncu-rep",
                target + "/profile_spgemm.ncu-rep",
                exist_ok=True)
    return


if __name__ == "__main__":
    # cutlass_test_simt()
    # cutlass_profile_win_simt()

    spgemm_profile_win()
