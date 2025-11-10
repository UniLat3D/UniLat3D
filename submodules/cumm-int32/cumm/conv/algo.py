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
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pccm

from cumm import dtypes
from cumm import tensorview as tv
from cumm.gemm.algospec.core import GemmAlgo, TensorOp
from cumm.gemm.core import MetaArray, metaseq, seq


class ConvAlgoParams(object):
    def __init__(self,
                 ts: Tuple[int, int, int],
                 wts: Tuple[int, int, int],
                 num_stage: int,
                 dtype_shorts: str,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 gemm_algo: GemmAlgo,
                 tensorop: Optional[TensorOp] = None,
                 splitk_serial: bool = False,
                 splitk_parallel: bool = False):
        self.ts = MetaArray(*ts)
        self.wts = MetaArray(*wts)
        self.num_stage = num_stage
        dtype_abcac = [
            dtypes.get_dtype_by_shortcut(s.strip())
            for s in dtype_shorts.split(",")
        ]
        assert len(dtype_abcac) == 5

        self.dtype_a = dtype_abcac[0]
        self.dtype_b = dtype_abcac[1]
        self.dtype_c = dtype_abcac[2]

        self.dtype_acc = dtype_abcac[3]
        self.dtype_comp = dtype_abcac[4]
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.algo = gemm_algo
        self.tensorop = tensorop
        self.splitk_serial = splitk_serial
        self.splitk_parallel = splitk_parallel

    def support_splitk(self):
        return self.splitk_serial or self.splitk_parallel

    def skipped(self):
        if self.dtype_a == dtypes.int8:
            if self.tensorop is not None:
                if (self.trans_a or not self.trans_b):
                    return True
        return False

    def get_algo_name(self):
        res = f"{self.algo.value}_{self.dtype_a.shortcut()}{self.dtype_b.shortcut()}{self.dtype_c.shortcut()}"
        res += f"{self.dtype_acc.shortcut()}{self.dtype_comp.shortcut()}"
        las = "n" if self.trans_a else "t"
        lbs = "n" if self.trans_b else "t"
        lcs = "n" if self.trans_c else "t"
        res += f"{las}{lbs}{lcs}"
        res += f"_m{self.ts[0]}n{self.ts[1]}k{self.ts[2]}"
        res += f"m{self.wts[0]}n{self.wts[1]}k{self.wts[2]}"
        if self.tensorop is not None:
            tss = self.tensorop.shape
            res += f"T{tss[0]}{tss[1]}{tss[2]}"
        res += f"_{self.num_stage}"
        return res
