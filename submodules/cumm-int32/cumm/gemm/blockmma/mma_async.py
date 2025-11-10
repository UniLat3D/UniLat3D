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

import contextlib
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewKernel
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import (constants, layout, mask_iters, out_iters, thread_map,
                       volta_iters, volta_out_iters)
from cumm.gemm.algospec import bases
from cumm.gemm.arch.memory import GlobalLoad
from cumm.gemm.bases import (GemmInputIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator, GemmSmemIterator,
                             GemmWarpIterator)
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from .mma import BlockMmaStorage

class MmaAsync(pccm.ParameterizedClass):
    def __init__(self,
                 dtype_acc: dtypes.DType,
                 partk: int,
                 num_stage: int,
                 spec: bases.Mma,
                 smem_storage: BlockMmaStorage,
                 first_input_clear: bool = True,
                 clear_mask: bool = True,
                 mask_sparse: bool = False,
                 increment_k_first=False,
                 is_sparse_wgrad: bool = False):
        super().__init__()
        self.dtype_acc = dtype_acc
        self.add_param_class("mma_ns_wa", spec.warp_iter_a, "WarpIterA")
        self.add_param_class("mma_ns_wb", spec.warp_iter_b, "WarpIterB")
        self.add_param_class("mma_ns_sa", spec.smem_iter_a, "SmemIterA")
        self.add_param_class("mma_ns_sb", spec.smem_iter_b, "SmemIterB")
        self.smem_storage = smem_storage
        self.spec = spec
        self.num_stage = num_stage
        self.mask_sparse = mask_sparse
        self.increment_k_first = increment_k_first
        self.partk = partk
        self.first_input_clear = first_input_clear
        self.clear_mask = clear_mask
        self.input_spec = spec.input_spec
        self.is_sparse_wgrad = is_sparse_wgrad
        if is_sparse_wgrad:
            self.add_param_class("gl_wgrad", GlobalLoad(4), "GlobalLoad")
        self.add_param_class("mma_ns_gm", smem_storage, "GemmStorage")
        self.accumulator_fragment = array_type(dtype_acc,
                                               spec.accumulator_size)
        self.add_param_class("mma_ns_ia", self.input_spec.input_iter_a,
                             "InputIteratorA")
        self.add_param_class("mma_ns_ib", self.input_spec.input_iter_b,
                             "InputIteratorB")
        self.add_param_class("mma_ns_wmma", spec.warp_mma, "WarpMma")

        self.add_member("warp_iter_A", "WarpIterA")
        self.add_member("warp_iter_B", "WarpIterB")
        self.add_member("smem_iter_A", "SmemIterA")
        self.add_member("smem_iter_B", "SmemIterB")

        # cudasim
        self.warp_iter_A: Optional[GemmWarpIterator] = None
        self.warp_iter_B: Optional[GemmWarpIterator] = None
        self.smem_iter_A: Optional[GemmSmemIterator] = None
        self.smem_iter_B: Optional[GemmSmemIterator] = None

        self.smem_A_ptr: Optional[ArrayPtr] = None
        self.smem_B_ptr: Optional[ArrayPtr] = None

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("smem_storage", "GemmStorage*")
        code.arg("thread_idx,warp_idx_k,warp_m,warp_n,lane_idx", "int")
        code.ctor_init(
            "warp_iter_A",
            "smem_storage->smem_A.data(), warp_idx_k, warp_m, lane_idx")
        code.ctor_init(
            "warp_iter_B",
            "smem_storage->smem_B.data(), warp_idx_k, warp_n, lane_idx")
        code.ctor_init(
            "smem_iter_A",
            f"{self.smem_storage.smem_shape_a[1]}, smem_storage->smem_A.data(), thread_idx"
        )
        code.ctor_init(
            "smem_iter_B",
            f"{self.smem_storage.smem_shape_b[1]}, smem_storage->smem_B.data(), thread_idx"
        )

        return code
