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
from typing import Optional, Tuple

import numpy as np
import pccm

from cumm import dtypes
from cumm.gemm import layout
from cumm.gemm.bases import (GemmApply, GemmInputIterator, GemmIterator,
                             GemmOutFragIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator, GemmSmemIterator,
                             GemmWarpIterator, WarpMma, GemmContainerBase)
from cumm.gemm.core import MetaArray, metaseq, seq
from .core import CacheOp

class Input(GemmContainerBase):
    """
    construct input iters
    Components:
        input iter A
        input iter B
    """
    @abc.abstractproperty
    def input_iter_a(self) -> GemmInputIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def input_iter_b(self) -> GemmInputIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def trans_a(self) -> bool:
        raise NotImplementedError

    @abc.abstractproperty
    def trans_b(self) -> bool:
        raise NotImplementedError

    @property
    def layout_a(self) -> pccm.Class:
        return layout.ColumnMajor() if self.trans_a else layout.RowMajor()

    @property
    def layout_b(self) -> pccm.Class:
        return layout.ColumnMajor() if self.trans_b else layout.RowMajor()

    @property
    def thread_map_a(self):
        return self.input_iter_a.tmap

    @property
    def thread_map_b(self):
        return self.input_iter_b.tmap

    @property
    def params_a(self):
        return self.input_iter_a.get_params()

    @property
    def params_b(self):
        return self.input_iter_b.get_params()

    @property
    def dtype_a(self):
        return self.input_iter_a.dtype

    @property
    def dtype_b(self):
        return self.input_iter_b.dtype

    @abc.abstractproperty
    def tile_shape(self) -> MetaArray[int]:
        raise NotImplementedError


class Mma(GemmContainerBase):
    @abc.abstractproperty
    def input_spec(self) -> Input:
        raise NotImplementedError

    @abc.abstractproperty
    def smem_iter_a(self) -> GemmSmemIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def smem_iter_b(self) -> GemmSmemIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def warp_iter_a(self) -> GemmWarpIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def warp_iter_b(self) -> GemmWarpIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def padding_mn(self) -> MetaArray[int]:
        raise NotImplementedError

    @abc.abstractproperty
    def warp_mma(self) -> WarpMma:
        raise NotImplementedError

    @abc.abstractproperty
    def num_warp_mma_iters(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def accumulator_size(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def partk(self) -> int:
        raise NotImplementedError

    # @abc.abstractproperty
    # def cache_op_a(self) -> CacheOp:
    #     raise NotImplementedError

    # @abc.abstractproperty
    # def cache_op_b(self) -> CacheOp:
    #     raise NotImplementedError

class Output(GemmContainerBase):
    @abc.abstractproperty
    def mma_spec(self) -> Mma:
        raise NotImplementedError

    @abc.abstractproperty
    def frag_iter(self) -> GemmOutFragIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def warp_store_iter(self) -> GemmOutWarpIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def smem_loader(self) -> GemmOutSmemLoader:
        raise NotImplementedError

    @abc.abstractproperty
    def out_iter(self) -> GemmOutputIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def const_out_iter(self) -> GemmOutputIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def int8_scalebias_out_iter(self) -> GemmOutputIterator:
        raise NotImplementedError

    @abc.abstractproperty
    def output_op(self) -> GemmOutputOp:
        raise NotImplementedError

    @abc.abstractproperty
    def apply_op(self) -> GemmApply:
        raise NotImplementedError

    @abc.abstractproperty
    def num_out_iters(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def frag_per_iter(self) -> int:
        raise NotImplementedError

    def get_accumulator_count(self) -> int:
        return self.frag_iter.element_count * self.num_out_iters

    def get_output_count(self) -> int:
        return self.out_iter.element_per_acc
