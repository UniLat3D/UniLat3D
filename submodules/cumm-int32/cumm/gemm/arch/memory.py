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

from typing import List, Optional, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.gemm import constants, core


class GlobalLoad(pccm.ParameterizedClass):
    def __init__(self,
                 load_bytes: int,
                 cache_op: Optional[pccm.cuda.CacheOpLd] = None,
                 prefetch: bool = False,
                 level: str = "",
                 prefetch_size: int = -1):
        super().__init__()
        self.load_bytes = load_bytes
        self.cache_op = cache_op
        self.load_dtype = dtypes.uint32
        self.prefetch = prefetch
        if prefetch_size != -1:
            assert prefetch_size in [64, 128, 256]
        if level:
            assert level == "L2"
        self.level = level
        self.prefetch_size = prefetch_size
        if load_bytes >= 4:
            self.count = self.load_bytes // 4
            assert self.load_bytes % 4 == 0
            self.fragment_t = core.array_type(str(self.load_dtype),
                                              self.load_bytes // 4)
        else:
            self.count = 1
            if load_bytes == 2:
                self.load_dtype = dtypes.uint16
            else:
                self.load_dtype = dtypes.uint8
            self.fragment_t = core.array_type(str(self.load_dtype), 1)

    def _run(self,
             code: pccm.cuda.PTXCode,
             level: str = "",
             prefetch_size: int = -1):
        with code.asm_block() as asm:
            ptr_addr = asm.global_ptr("ptr")
            frag_reg_type = pccm.cuda.RegDType.B32
            if self.load_dtype == dtypes.uint16:
                frag_reg_type = pccm.cuda.RegDType.U16
            frag = asm.reg_ptr("frag_ptr", frag_reg_type)
            pred = asm.ext_reg("(int)pred", pccm.cuda.RegDType.B32)
            for i in range(self.count):
                asm.mov(frag[i], frag[i])  # TODO WTF???

            with asm.pred_if("p", "ne", pred, 0) as reg:
                frag_unpack = frag.unpack(self.count)
                if self.count > 4:
                    num_vec_load = self.count // 4
                    for i in range(num_vec_load):
                        if self.prefetch:
                            asm.generic("prefetch.global.L2",
                                        [ptr_addr + i * 16])
                        asm.ld(ptr_addr + i * 16,
                               frag_unpack[i * 4:(i + 1) * 4], self.cache_op)
                else:
                    if self.prefetch:
                        asm.generic("prefetch.global.L2", [ptr_addr])
                    asm.ld(ptr_addr, frag_unpack, self.cache_op, level,
                           prefetch_size)

    @pccm.cuda.static_function(device=True, forceinline=True)
    def run(self):
        code = pccm.cuda.PTXCode()
        code.targ("Frag")
        code.arg("frag", f"Frag &")
        code.arg("ptr", "void const*")
        code.arg("pred", "bool")
        code.raw(
            f"{self.load_dtype}* frag_ptr = reinterpret_cast<{self.load_dtype}*>(&frag);"
        )
        if self.load_dtype == dtypes.uint8:
            code.raw(f"""
            if (pred){{
                frag_ptr[0] = reinterpret_cast<{self.load_dtype} const*>(ptr)[0];
            }}
            """)
        else:
            with code.macro_if_("CUDA_VERSION >= 11040 && (__CUDA_ARCH__ >= 750)"):
                self._run(code, self.level, self.prefetch_size)
            with code.macro_else_():
                self._run(code)
            code.macro_endif_()
        return code
