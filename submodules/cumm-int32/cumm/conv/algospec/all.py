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

from typing import Dict, List, Optional, Tuple, Type, Union

from cumm.gemm.algospec import GemmAlgo

from .simt import AlgoSpecificSimt
from .turing import AlgoSpecificTuring
from .volta import AlgoSpecificVolta

_ALGOSPEC_TYPES = Union[Type[AlgoSpecificSimt], Type[AlgoSpecificVolta],
                        Type[AlgoSpecificTuring]]

ALGO_TO_SPEC: Dict[GemmAlgo, _ALGOSPEC_TYPES] = {
    GemmAlgo.Simt: AlgoSpecificSimt,
    GemmAlgo.SimtDP4A: AlgoSpecificSimt,
    GemmAlgo.Volta: AlgoSpecificVolta,
    GemmAlgo.Turing: AlgoSpecificTuring,
    GemmAlgo.Ampere: AlgoSpecificTuring
}


def get_algo_spec(algo: GemmAlgo):
    return ALGO_TO_SPEC[algo]
