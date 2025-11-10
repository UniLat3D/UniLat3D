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

from typing import List, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import constants, core


class LdMatrix(pccm.ParameterizedClass):
    MaxNum = 4
    LineByteSize = 8 * 2
    NumLinePerMatrix = 8

    def __init__(self, is_rowmajor: bool, count: int):
        super().__init__()
        self.is_rowmajor = is_rowmajor
        self.count = count
        self.fragment_t = core.array_type("unsigned", self.count)
        self.add_include("tensorview/gemm/arch/memory_sm75.h")

    def python_ctor(self):
        return self

    @pccm.cuda.static_function(device=True, forceinline=True)
    def run(self):
        code = pccm.code()
        code.arg("D", f"{self.fragment_t} &")
        code.arg("ptr", "void const*")
        count_fmts = [f"%{i}" for i in range(self.count)]
        count_fmt = ", ".join(count_fmts)
        trans_str = ""
        xyzw = ["x", "y", "z", "w"]
        xyzw_r = ", ".join([f"\"=r\"({s})" for s in xyzw[:self.count]])
        if not self.is_rowmajor:
            trans_str = ".trans"
        asm_rowmajor = (
            f"asm volatile (\"ldmatrix.sync.aligned.x{self.count}{trans_str}"
            f".m8n8.shared.b16 {{{count_fmt}}}, [%{self.count}];\" : {xyzw_r} : \"r\"(addr));"
        )
        with code.macro_if_(
                "defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)"):
            code.raw("""
            unsigned addr = tv::gemm::get_smem_pointer(ptr);
            """)
            if self.count == 1:
                code.raw("int x;")
            elif self.count == 2:
                code.raw("int x, y;")
            elif self.count == 4:
                code.raw("int x, y, z, w;")
            else:
                raise NotImplementedError
            code.raw(asm_rowmajor)
            if self.count == 1:
                code.raw("reinterpret_cast<int &>(D) = x;")
            elif self.count == 2:
                code.raw("reinterpret_cast<int2 &>(D) = make_int2(x, y);")
            elif self.count == 4:
                code.raw(
                    "reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);")
        with code.macro_else_():
            code.raw("assert(0);")
        code.macro_endif_()
        return code

    async def __call__(self, D: ArrayPtr, ptr: ArrayPtr):
        lane_id = cudasim.get_lane_id()
        resource = cudasim.get_warp_resource()
        warp_data = await resource.gather(
            lane_id, (D, ptr), 0)  # type: List[Tuple[ArrayPtr, ArrayPtr]]
        if lane_id == 0:
            Ds = [x[0].change_access_byte_size(2) for x in warp_data]
            # print(Ds)
            smem_ptrs = [x[1] for x in warp_data]
            for i in range(self.count):
                mat_line_ptrs = smem_ptrs[i * 8:(i + 1) *
                                          8]  # type: List[ArrayPtr]
                mat_lines = [
                    p.change_access_byte_size(2) for p in mat_line_ptrs
                ]
                # print(mat_lines)
                for j in range(8):
                    for k in range(4):
                        if self.is_rowmajor:
                            # for tf32 simulation
                            mat_ptr = mat_lines[j].change_access_byte_size(
                                4)[k]
                            Ds[j * 4 +
                               k].change_access_byte_size(4)[i] = mat_ptr
                        else:
                            for l in range(2):
                                mat_ptr = mat_lines[
                                    k * 2 + l][j].change_access_byte_size(2)
                                Ds[j * 4 + k][i * 2 + l] = mat_ptr

        await resource.wait()
        return
