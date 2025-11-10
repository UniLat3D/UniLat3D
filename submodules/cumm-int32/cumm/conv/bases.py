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

import abc
import enum
from typing import Optional, Union

import pccm

from cumm import dtypes
from cumm.constants import CUTLASS_MODE
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import bases, layout
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.thread_map import PitchLinear, PitchLinearWarpRaked

LAYOUT_TYPES = Union[layout.TensorGeneric]


class ConvMode(enum.Enum):
    kConvolution = 0
    kCrossCorrelation = 1


class ConvIterAlgo(enum.Enum):
    Analytic = 0
    Optimized = 1


class ConvLayoutType(enum.Enum):
    ChannelFirst = 0
    ChannelLast = 1
    SpatialFirst = 2  # RSKC layout for weight only.


class ConvOpType(enum.Enum):
    kForward = 0
    kBackwardInput = 1
    kBackwardWeight = 2


class ConvLayout:
    def __init__(self, layout_type: ConvLayoutType, interleave: int = 1):
        self.layout_type = layout_type
        self.interleave = interleave

    def __repr__(self):
        layout_str = "F" if self.layout_type == ConvLayoutType.ChannelFirst else "L"
        if self.layout_type == ConvLayoutType.ChannelLast:
            return layout_str
        return f"{layout_str}{self.interleave}"

    def is_channel_first(self):
        return self.layout_type == ConvLayoutType.ChannelFirst

    def get_cutlass(self):
        if self.layout_type == ConvLayoutType.ChannelFirst:
            if self.interleave == 1:
                return "cutlass::layout::TensorNCHW"
            else:
                return "cutlass::layout::TensorNCxHWx"
        else:
            return "cutlass::layout::TensorNHWC"

    def get_layout_class(self, ndim: int):
        if self.interleave == 1:
            return layout.TensorGeneric(ndim)
        raise NotImplementedError


NCHW = ConvLayout(ConvLayoutType.ChannelFirst)
NHWC = ConvLayout(ConvLayoutType.ChannelLast)

KRSC = ConvLayout(ConvLayoutType.ChannelLast)
RSKC = ConvLayout(ConvLayoutType.SpatialFirst)


class ConvTensor:
    def __init__(self, ndim: int, dtype: dtypes.DType, layout: ConvLayout):
        self.ndim = ndim
        self.dtype = dtype
        self.layout = layout


@pccm.skip_inherit
class ConvIterParams(pccm.ParameterizedClass):
    def python_ctor(self, conv_psize,
                    layout: LAYOUT_TYPES) -> "ConvIterParams":
        raise NotImplementedError


@pccm.skip_inherit
class ConvInputIterator(bases.GemmIterator):
    def python_ctor(self, params: ConvIterParams, problem_size, ptr: ArrayPtr,
                    thread_id: int, tb_offset: MetaArray[int]):
        raise NotImplementedError

    def get_params(self) -> pccm.ParameterizedClass:
        raise NotImplementedError

    def tile_increment_python(self, num_tile: int):
        raise NotImplementedError

    def clear_mask_python(self):
        return

    def increment_python(self):
        return self.tile_increment_python(1)

    def load_python(self, frag: ArrayPtr):
        raise NotImplementedError

