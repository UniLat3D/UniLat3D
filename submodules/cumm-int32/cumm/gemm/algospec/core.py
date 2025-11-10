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
from typing import Dict, List, Optional, Tuple

from cumm.gemm.core import MetaArray, metaseq, seq
from cumm import dtypes

class GemmAlgo(enum.Enum):
    Simt = "Simt"
    SimtDP4A = "SimtDP4A"
    SimtDP2A = "SimtDP2A"
    Volta = "Volta"
    Turing = "Turing"
    Ampere = "Ampere"


_GEMM_MIN_ARCH_TO_ALGO: List[Tuple[Tuple[int, int], List[str]]] = [
    ((3, 5), [GemmAlgo.Simt.value]),
    ((6, 1), [GemmAlgo.SimtDP4A.value, GemmAlgo.SimtDP2A.value]),
    ((7, 0), [GemmAlgo.Volta.value]),
    ((7, 5), [GemmAlgo.Turing.value]),
    ((8, 0), [GemmAlgo.Ampere.value]),
]

_GEMM_ALGO_TO_MIN_ARCH: Dict[str, Tuple[int, int]] = {}

for min_arch, algos in _GEMM_MIN_ARCH_TO_ALGO:
    for algo in algos:
        _GEMM_ALGO_TO_MIN_ARCH[algo] = min_arch


def get_min_arch_of_algo(algo: GemmAlgo):
    return _GEMM_ALGO_TO_MIN_ARCH[algo.value]


def get_min_arch_of_algo_str(algo_str: str):
    return _GEMM_ALGO_TO_MIN_ARCH[algo_str]


def get_available_algo_str_from_arch(arch: Tuple[int, int]):
    res: List[str] = []
    for i in range(len(_GEMM_MIN_ARCH_TO_ALGO) - 1, -1, -1):
        arch_cur, algos = _GEMM_MIN_ARCH_TO_ALGO[i]
        if arch >= arch_cur:
            res.extend(algos)
    return res


class ShuffleStrideType(enum.Enum):
    NoShuffle = 0
    # A and C have indices, for spatial spconv forward and backward input
    ShuffleAC = 1
    # A and B have indices, for spatial spconv backward weight
    ShuffleAB = 2


class CacheOp(enum.Enum):
    Always = 0 # cache at all levels - accessed again
    Global = 1 # Cache at global level
    Streaming = 2 # Streaming - likely to be accessed once
    LastUse = 3 # Indicates the line will not be used again
    Volatile = 4 # Don't cache, and fetch again
    WriteBack = 5 # Write back at all coherent levels
    WriteThrough = 6 # Write through to system memory

class TensorOp(object):
    def __init__(self, shape: Tuple[int, int, int], top_dtypes: Optional[str] = None):
        self.shape = seq(*shape)
        self.top_dtypes = top_dtypes
        if top_dtypes is not None:
            dtype_abc = [
                dtypes.get_dtype_by_shortcut(s.strip())
                for s in top_dtypes.split(",")
            ]
            assert len(dtype_abc) == 3
            self.dtype_a = dtype_abc[0]
            self.dtype_b = dtype_abc[1]
            self.dtype_c = dtype_abc[2]

        else:
            self.dtype_a = None 
            self.dtype_b = None 
            self.dtype_c = None 

    def to_string(self):
        res= f"{self.shape[0]}{self.shape[1]}{self.shape[2]}"
        if self.top_dtypes is not None:
            res += self.top_dtypes

        return res 

    def __getitem__(self, val: int):
        return self.shape[val]
