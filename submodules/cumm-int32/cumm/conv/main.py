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

import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pccm
from ccimport import compat
from pccm.core import CodeFormatter

# from myclang import clangformat
from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import CummNVRTCLib, GemmBasic, GemmBasicHost, TensorView, TensorViewKernel
from cumm.constants import CUMM_MAXIMUM_NVRTC_CONV_NDIM, CUTLASS_MODE
from cumm.conv import kernel
from cumm.conv.bases import (NCHW, NHWC, ConvIterAlgo, ConvLayout,
                             ConvLayoutType, ConvMode, ConvOpType)
from cumm.conv.params import (ConvProblem,
                              conv_iwo_012_to_abc, gemm_abc_012_to_iwo,
                              get_gemm_trans_abc)
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import codeops
from cumm.gemm.algospec import GemmAlgo
from cumm.gemm.algospec.core import TensorOp
from cumm.gemm.core.metaarray import MetaArray
from cumm.gemm.kernel import GemmKernel
from cumm.gemm.main import (GemmAlgoParams, GemmMainUnitTest,
                            NVRTCMode)
from cumm.conv.nvrtc_code import nvrtc_conv_template
def seq(*vals):
    return np.array([*vals], dtype=np.int64)


class ConvAlgoParams(GemmAlgoParams):
    def __init__(self,
                 ndim: int,
                 op_type: ConvOpType,
                 iter_algo: ConvIterAlgo,
                 ts: Tuple[int, int, int],
                 wts: Tuple[int, int, int],
                 num_stage: int,
                 dtype_shorts: str,
                 layout_desp_input: ConvLayout,
                 layout_desp_weight: ConvLayout,
                 layout_desp_output: ConvLayout,
                 algo: GemmAlgo,
                 tensorop: Optional[TensorOp] = None,
                 splitk_serial: bool = False,
                 splitk_parallel: bool = False,
                 mask_sparse: bool = False,
                 increment_k_first: bool = False,
                 access_per_vector: int = 1,
                 is_nvrtc: bool = False,
                 int8_inference: bool = False,
                 dynamic_mask: bool = False):
        trans_a, trans_b, trans_c = get_gemm_trans_abc(op_type)
        super().__init__(ts,
                         wts,
                         num_stage,
                         dtype_shorts,
                         trans_a,
                         trans_b,
                         trans_c,
                         algo,
                         tensorop,
                         splitk_serial,
                         splitk_parallel,
                         access_per_vector=access_per_vector,
                         is_nvrtc=is_nvrtc)
        self.ndim = ndim
        self.op_type = op_type
        self.iter_algo = iter_algo
        self.mask_sparse = mask_sparse
        self.increment_k_first = increment_k_first
        self.int8_inference = int8_inference
        self.dynamic_mask = dynamic_mask

        indices = conv_iwo_012_to_abc(op_type)
        dtypes_abc = [self.dtype_a, self.dtype_b, self.dtype_c]
        self.dtype_input = dtypes_abc[indices[0]]
        self.dtype_weight = dtypes_abc[indices[1]]
        self.dtype_output = dtypes_abc[indices[2]]

        self.layout_desp_input = layout_desp_input
        self.layout_desp_weight = layout_desp_weight
        self.layout_desp_output = layout_desp_output

    def skipped(self):
        if self.op_type != ConvOpType.kForward and self.dtype_a.itemsize(
        ) == 1:
            return True

        return super().skipped()


def gen_gemm_params(op_types: List[ConvOpType],
                    ts,
                    wts,
                    ndim: int,
                    iter_algo: ConvIterAlgo,
                    stage: Union[int, List[int]],
                    dtypes_string: Union[str, List[str]],
                    li: ConvLayout,
                    lw: ConvLayout,
                    lo: ConvLayout,
                    algo: GemmAlgo,
                    tensorop: Optional[TensorOp],
                    splitk_serial: bool = False,
                    splitk_parallel: bool = False,
                    mask_sparse: bool = False,
                    increment_k_first: bool = False,
                    access_per_vector: int = 1,
                    is_nvrtc: bool = False,
                    int8_inference: bool = False,
                    dynamic_mask: Optional[bool] = None):
    res: List[ConvAlgoParams] = []
    if not isinstance(dtypes_string, list):
        dtypes_string = [dtypes_string]
    stages: List[int] = []
    if isinstance(stage, list):
        stages.extend(stage)
    else:
        stages.append(stage)
    dymasks = [False, True]
    if not mask_sparse:
        dymasks = [False]
    # dynamic_mask is used for debug
    if dynamic_mask is not None:
        dymasks = [dynamic_mask]
    for dts in dtypes_string:
        for op_type in op_types:
            for num_stage in stages:
                for dymask in dymasks:
                    if dymask and dynamic_mask is None:
                        cur_is_nvrtc = True
                    else:
                        cur_is_nvrtc = is_nvrtc
                    if op_type == ConvOpType.kBackwardWeight:
                        p = ConvAlgoParams(ndim, op_type, iter_algo, ts, wts, num_stage,
                                        dts, li, lw, lo, algo, tensorop, True,
                                        splitk_parallel, mask_sparse,
                                        increment_k_first, access_per_vector,
                                        is_nvrtc=cur_is_nvrtc, int8_inference=int8_inference,
                                        dynamic_mask=dymask)
                    else:
                        p = ConvAlgoParams(ndim, op_type, iter_algo, ts, wts, num_stage,
                                        dts, li, lw, lo, algo, tensorop,
                                        splitk_serial, splitk_parallel, mask_sparse,
                                        increment_k_first, access_per_vector,
                                        is_nvrtc=cur_is_nvrtc, int8_inference=int8_inference,
                                        dynamic_mask=dymask)

                    if not p.skipped():
                        res.append(p)
    return res


ConvFwdAndBwdInput = [ConvOpType.kBackwardInput, ConvOpType.kForward, ]
ConvFwd = [ConvOpType.kForward, ]

ConvBwdWeight = [ConvOpType.kBackwardWeight]
ConvAllOp = [
    ConvOpType.kForward, ConvOpType.kBackwardInput, ConvOpType.kBackwardWeight
]


def gen_spwgrad_params(ts,
                       wts,
                       ndim: int,
                       iter_algo: ConvIterAlgo,
                       stage: int,
                       dtypes_string: str,
                       li: ConvLayout,
                       lw: ConvLayout,
                       lo: ConvLayout,
                       algo: GemmAlgo,
                       tensorop: Optional[TensorOp],
                       splitk_serial: bool = False,
                       splitk_parallel: bool = False,
                       mask_sparse: bool = False,
                       increment_k_first: bool = False,
                       access_per_vector: int = 1):
    p = ConvAlgoParams(ndim, ConvOpType.kBackwardWeight, iter_algo, ts, wts,
                       stage, dtypes_string, li, lw, lo, algo, tensorop, True,
                       splitk_parallel, mask_sparse, increment_k_first,
                       access_per_vector)
    return [p]


def gen_gemm_kernels(params: ConvAlgoParams,
                     nvrtc_mode: NVRTCMode = NVRTCMode.Disabled):
    return kernel.ConvKernel(params.ndim,
                             params.op_type,
                             params.iter_algo,
                             params.ts,
                             params.wts,
                             params.num_stage,
                             dtype_a=params.dtype_a,
                             dtype_b=params.dtype_b,
                             dtype_c=params.dtype_c,
                             dtype_acc=params.dtype_acc,
                             dtype_comp=params.dtype_comp,
                             layout_desp_input=params.layout_desp_input,
                             layout_desp_output=params.layout_desp_output,
                             layout_desp_weight=params.layout_desp_weight,
                             algo=params.algo,
                             tensorop=params.tensorop,
                             splitk_serial=params.splitk_serial,
                             splitk_parallel=params.splitk_parallel,
                             mask_sparse=params.mask_sparse,
                             increment_k_first=params.increment_k_first,
                             access_per_vector=params.access_per_vector,
                             nvrtc_mode=nvrtc_mode,
                             int8_inference=params.int8_inference,
                             dynamic_mask=params.dynamic_mask,
                             nvrtc_compile=params.is_nvrtc)


SHUFFLE_SIMT_PARAMS = []

SHUFFLE_VOLTA_PARAMS = []
SHUFFLE_TURING_PARAMS = []


class ConvMainUnitTest(pccm.ParameterizedClass):
    def __init__(self, conv_params: Optional[List[ConvAlgoParams]] = None):
        super().__init__()
        self.add_dependency(TensorView, GemmBasic, GemmBasicHost, kernel.ConvNVRTCParams,
                            CummNVRTCLib)
        # unit test params: [ts, wts, stage, dtypes, trans, algo, tensorop]
        if conv_params is None:
            is_debug = os.getenv("CUMM_DEBUG", None)
            if is_debug is not None and is_debug == "1":
                simt_params: List[ConvAlgoParams] = [
                    # *gen_gemm_params(ConvFwd, (64, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "s8,s8,s8,s32,s32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.SimtDP4A, None),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 32),
                    #                  (32, 32, 32), 3, ConvIterAlgo.Optimized,
                    #                  2, "f16,f16,f16,f32,f32", NHWC, NHWC,
                    #                  NHWC, GemmAlgo.Turing,
                    #                  TensorOp((16, 8, 8))),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 16),
                    #                  (32, 32, 8), 3, ConvIterAlgo.Optimized, 2,
                    #                  "f32,f32,f32,f32,f32", NHWC, NHWC, NHWC,
                    #                  GemmAlgo.Simt, None),

                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
                    # # *gen_gemm_params(ConvBwdWeight, (128, 128, 8), (32, 64, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 64, 32), (32, 32, 16), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 32, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    # #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwd, (64, 32, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["s8,s8,s8,s32,s32"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((8, 8, 16))),
                    # *gen_gemm_params(ConvFwd, (64, 64, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["s8,s8,s8,s32,s32"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((8, 8, 16))),
                    # *gen_gemm_params(ConvFwd, (128, 128, 32), (64, 64, 32), 2, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8))),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 32, 32), (32, 32, 32),
                                        2,
                                        ConvIterAlgo.Optimized,
                                        2,
                                        ["s8,s8,s8,s32,s32"],
                                        NHWC,
                                        NHWC,
                                        NHWC,
                                        GemmAlgo.Turing,
                                        # None,
                                        TensorOp((16, 8, 16)),
                                        access_per_vector=1,
                                        dynamic_mask=False),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 32),
                    #                 2,
                    #                 ConvIterAlgo.Optimized,
                    #                 2,
                    #                 ["s8,s8,s8,s32,s32"],
                    #                 NHWC,
                    #                 NHWC,
                    #                 NHWC,
                    #                 GemmAlgo.Turing,
                    #                 TensorOp((16, 8, 32)),
                    #                 access_per_vector=0,
                    #                 dynamic_mask=False),
                    # *gen_gemm_params(ConvBwdWeight, (128, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (128, 64, 32), (64, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 64, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (128, 256, 32), (64, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),

                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 16), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 32, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 64, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOp((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),

                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Volta, TensorOp((8, 8, 4)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Volta, TensorOp((8, 8, 4)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Volta, TensorOp((8, 8, 4)), mask_sparse=True, increment_k_first=True, access_per_vector=1),

                    # *gen_spwgrad_params((128, 128, 8), (32, 64, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True, mask_width=32),
                    # *gen_gemm_params((32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True, mask_width=32),
                    # *gen_gemm_params(ConvFwd, 
                    #     (32, 128, 16), (32, 32, 8), 2, ConvIterAlgo.Optimized, 2,
                    #     "f32,f32,f32,f32,f32", NHWC, NHWC, NHWC, GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 16, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 64, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 16, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((64, 128, 8), (32, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f16,f16", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f16,f16,f16,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params((128, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", GemmAlgo.SimtDP4A, None),
                ]  # type: List[ConvAlgoParams]
                volta_params: List[ConvAlgoParams] = []
                turing_params: List[ConvAlgoParams] = []
            else:
                simt_params: List[ConvAlgoParams] = [
                    *SHUFFLE_SIMT_PARAMS,
                ]
                volta_params: List[ConvAlgoParams] = [
                    *SHUFFLE_VOLTA_PARAMS,
                ]
                turing_params: List[ConvAlgoParams] = [
                    *SHUFFLE_TURING_PARAMS,
                ]
            self.all_params = simt_params + volta_params + turing_params
            self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]

        else:
            assert len(conv_params) > 0
            self.all_params = conv_params
            self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]
        self.ker_names = [k.get_algo_name() for k in self.all_kernels]
        assert len(set(self.ker_names)) == len(
            self.ker_names), "kernel must unique"

    @staticmethod
    def _get_layout_types(ker: kernel.ConvKernel):
        p = ker.problem
        return (p.layout_desp_input.layout_type,
                p.layout_desp_weight.layout_type,
                p.layout_desp_output.layout_type)

    @staticmethod
    def _get_layout_interleaves(ker: kernel.ConvKernel):
        p = ker.problem
        return (p.layout_desp_input.interleave,
                p.layout_desp_weight.interleave,
                p.layout_desp_output.interleave)

    @staticmethod
    def _get_sparse_params(ker: kernel.ConvKernel):
        return (ker.mask_sparse, ker.increment_k_first, ker.int8_inference, ker.dynamic_mask)

    @staticmethod
    def conv_select_helper_stage1(kernels: List[kernel.ConvKernel],
                                  code: pccm.FunctionCode):
        ndim_op_iter_to_kers = codeops.group_by(
            lambda x: (x.problem.ndim, x.problem.op_type, x.iter_algo),
            kernels)
        for ndim_op_iter, ndim_op_iter_kers in ndim_op_iter_to_kers.items():
            if_tests = [
                f"algo_desp.ndim == {ndim_op_iter[0]}",
                f"static_cast<int>(algo_desp.op_type) == {ndim_op_iter[1].value}",
                f"static_cast<int>(algo_desp.iter_algo) == {ndim_op_iter[2].value}",
            ]
            with code.if_(" && ".join(if_tests)):
                li_lw_lo_to_kers = codeops.group_by(
                    ConvMainUnitTest._get_layout_types, ndim_op_iter_kers)
                for li_lw_lo, lilwlo_kers in li_lw_lo_to_kers.items():
                    if_tests = [
                        f"static_cast<int>(algo_desp.layout_i) == {li_lw_lo[0].value}",
                        f"static_cast<int>(algo_desp.layout_w) == {li_lw_lo[1].value}",
                        f"static_cast<int>(algo_desp.layout_o) == {li_lw_lo[2].value}",
                    ]
                    with code.if_(" && ".join(if_tests)):
                        lii_lwi_loi_to_kers = codeops.group_by(
                            ConvMainUnitTest._get_layout_interleaves,
                            lilwlo_kers)
                        for liilwiloi, liilwiloi_kers in lii_lwi_loi_to_kers.items(
                        ):
                            if_tests = [
                                f"algo_desp.interleave_i == {liilwiloi[0]}",
                                f"algo_desp.interleave_w == {liilwiloi[1]}",
                                f"algo_desp.interleave_o == {liilwiloi[2]}",
                            ]
                            with code.if_(" && ".join(if_tests)):
                                ms_ikf_mw_to_kers = codeops.group_by(
                                    ConvMainUnitTest._get_sparse_params,
                                    liilwiloi_kers)
                                for ms_ikf_mw, ms_ikf_mw_kers in ms_ikf_mw_to_kers.items(
                                ):
                                    assert len(
                                        ms_ikf_mw_kers
                                    ) == 1, "find multiple kernels for one configuration"
                                    if_tests = [
                                        f"algo_desp.mask_sparse == {pccm.boolean(ms_ikf_mw[0])}",
                                        f"algo_desp.increment_k_first == {pccm.boolean(ms_ikf_mw[1])}",
                                        f"algo_desp.is_int8_inference == {pccm.boolean(ms_ikf_mw[2])}",
                                        f"algo_desp.dynamic_mask == {pccm.boolean(ms_ikf_mw[3])}",
                                    ]
                                    with code.if_(" && ".join(if_tests)):
                                        yield ms_ikf_mw_kers

    @staticmethod
    def conv_select_helper(kernels: List[Union[kernel.ConvKernel]],
                           code: pccm.FunctionCode):
        for kers in GemmMainUnitTest.matmul_select_helper_stage2(
                kernels, code, False, False):
            yield from ConvMainUnitTest.conv_select_helper_stage1(kers, code)

    @pccm.pybind.mark
    @pccm.static_function
    def extract_mnk(self):
        code = pccm.code()
        code.arg("op_type", "int")
        code.arg("N, C, K", "int")
        code.arg("kernel_volume, in_prod, out_prod", "int")
        code.arg("mask_sparse", "bool")
        code.raw(f"""
        auto op_type_enum = static_cast<tv::gemm::ConvOpType>(op_type);
        auto res = tv::gemm::implicit_gemm_mnk(op_type_enum, N, C, K, 
            kernel_volume, in_prod, out_prod, mask_sparse);
        return {{res[0], res[1], res[2]}};
        """)
        return code.ret("std::array<int, 3>")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def implicit_gemm2(self):
        code_main = pccm.code()
        code_main.arg("params", "tv::gemm::ConvParams", pyanno="cumm.tensorview.gemm.ConvParams")
        code_main.add_dependency(TensorViewKernel)
        code_main.raw(f"auto& algo_desp = params.conv_algo_desp;")
        for kers, decl in GemmMainUnitTest.matmul_select_split(self.all_kernels, code_main):
            code = decl.code
            for ker in kers:
                if not ker.nvrtc_compile:
                    code.add_param_class("cp" + ker.get_algo_name(), ker.gemm_params,
                                        "ConvParams" + ker.get_algo_name())
                    code.add_param_class(ker.get_algo_name(), ker,
                                        "Conv" + ker.get_algo_name())
            nvrtc_conv_template(code)
            self.add_func_decl(decl)
            code.arg("params", "tv::gemm::ConvParams", pyanno="cumm.tensorview.gemm.ConvParams")

            for kers in self.conv_select_helper(kers, code):
                ker = kers[0]
                if ker.nvrtc_compile:
                    code.raw(f"""
                    TV_THROW_RT_ERR("you must add cumodule for nvrtc kernel.");
                    """)
                    continue
                p = ker.problem
                param_type_str = "ConvParams" + ker.get_algo_name()
                indices = conv_iwo_012_to_abc(ker.problem.op_type)
                inv_indices = gemm_abc_012_to_iwo(ker.problem.op_type)
                dtypes_abc = [ker.dtype_a, ker.dtype_b, ker.dtype_c]
                dtypes_iwo = [dtypes_abc[i] for i in indices]
                param_cls_name = "ConvParams" + ker.get_algo_name()
                param_cls_ns = "cp" + ker.get_algo_name()

                if not ker.support_splitk():
                    code.raw(f"""
                    TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                    """)
                # TODO if input is NCxHWx
                # TODO if input weight and output have different layout
                dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
                io_ndim = 2 if p.mask_sparse else p.ndim + 2
                weight_ndim = 3 if p.mask_sparse else p.ndim + 2
                abc_names = ["a", "b", "c"]
                abc_names = [abc_names[i] for i in indices]
                input_names = ["input", "weight", "output"]
                input_names = [input_names[i] for i in inv_indices]

                code.raw(f"""
                // {ker.get_algo_name()}
                found = true;
                bool d_is_bias = !bias.empty();
                """)
                if not ker.split_d_params:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(bias.empty(), "bias must be empty if split_d_params not enable");
                    """)
                if ker.int8_inference:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(!bias.empty(), "bias must not be empty if int8_inference enable");
                    TV_ASSERT_RT_ERR(!scale.empty(), "scale must not be empty if int8_inference enable");
                    """)
                if p.mask_sparse:
                    if p.op_type == ConvOpType.kBackwardWeight:
                        code.raw(f"""
                        TV_ASSERT_RT_ERR(mask_width > 0 && mask_width % {ker.tile_shape[2]} == 0, "error");
                        """)
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(!indices.empty(), "error");
                    TV_ASSERT_RT_ERR(!mask.empty(), "error");
                    TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                    int kernel_volume = weight.dim({dim_start});
                    tv::check_shape(indices, {{kernel_volume, -1}});
                    N = indices.dim(1);
                    {param_cls_ns}::ConvProblem problem(N, C, K, kernel_volume, 
                        tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                    """)
                    if p.op_type == ConvOpType.kBackwardWeight:
                        code.raw(f"""
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * {ker.dtype_b.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        """)
                    elif p.op_type == ConvOpType.kForward:
                        code.raw(f"""
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        """)
                    else:
                        code.raw(f"""
                        # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        TV_ASSERT_RT_ERR(N == input.dim(0), "error");
                        """)
                else:
                    code.raw(f"""
                    tv::array<int, {p.ndim}> input_dims, output_dims;
                    tv::array<int, {p.ndim}> ksize;
                    TV_ASSERT_RT_ERR({p.ndim} == padding.size() && {p.ndim} == stride.size() && {p.ndim} == dilation.size(), "error");
                    for (int i = {dim_start}; i < {dim_start + p.ndim}; ++i){{
                        ksize[i - {dim_start}] = weight.dim(i);
                        input_dims[i - {dim_start}] = input.dim(i);
                        output_dims[i - {dim_start}] = output.dim(i);
                    }}
                    int kernel_volume = ksize.op<tv::arrayops::prod>();
                    tv::array<int, {p.ndim}> padding_arr{{{code.unpack([f"padding[{i}]" for i in range(p.ndim)])}}};
                    tv::array<int, {p.ndim}> stride_arr{{{code.unpack([f"stride[{i}]" for i in range(p.ndim)])}}};
                    tv::array<int, {p.ndim}> dilation_arr{{{code.unpack([f"dilation[{i}]" for i in range(p.ndim)])}}};
                    auto output_dims_check_again = {param_cls_ns}::ConvProblem::calc_output_dims(input_dims, ksize, padding_arr, stride_arr, dilation_arr);
                    for (int i = 0; i < {p.ndim}; ++i){{
                        TV_ASSERT_RT_ERR(output_dims_check_again[i] == output_dims[i], "error");
                    }}
                    
                    {param_cls_ns}::ConvProblem problem(N, C, K, input_dims, output_dims, ksize, padding_arr, stride_arr, dilation_arr, 
                        tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                    """)
                code.raw(f"""
                if (!algo_desp.is_int8_inference){{
                    TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                }}
                auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const {ker.dtype_c}>() : (bias.empty() ? c_ten.data_ptr<const {ker.dtype_c}>() : bias.data_ptr<const {ker.dtype_c}>());
                if (!output_add.empty()){{
                    source_ptr = output_add.data_ptr<const {ker.dtype_c}>();
                }}
                """)
                params_str = [
                    "problem", 
                    f"a_ten.data_ptr<const {ker.dtype_a}>()",
                    f"b_ten.data_ptr<const {ker.dtype_b}>()",
                    f"c_ten.data_ptr<{ker.dtype_c}>()",
                    f"source_ptr",
                ]
                if ker.int8_inference:
                    params_str.append(f"bias.empty() ? nullptr : bias.data_ptr<const {ker.dtype_comp}>()",)
                    params_str.append(f"scale.empty() ? nullptr : scale.data_ptr<const {ker.dtype_comp}>()",)
                if p.mask_sparse:
                    # mask_out_ptr = "mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), "
                    # if p.op_type != ConvOpType.kForward:
                    #     mask_out_ptr = ""

                    # mask_width_str = "mask_width,"
                    # if p.op_type != ConvOpType.kBackwardWeight:
                    #     mask_width_str = ""
                    params_str.extend([
                        "mask.data_ptr<const uint32_t>()",
                        "mask_argsort.data_ptr<const int32_t>()",
                        "indices.data_ptr<const int32_t>()",
                        # "params.mask_int_count",
                    ])
                    if p.op_type == ConvOpType.kForward:
                        params_str.append(f"mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>()")
                    params_str.extend([
                        "params.mask_filter",
                        "params.reverse_mask",
                    ])
                    if p.op_type == ConvOpType.kBackwardWeight:
                        params_str.append("mask_width")
                    
                params_str.extend([
                    f"{ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta)",
                    f"{ker.dtype_comp}(params.act_alpha), {ker.dtype_comp}(params.act_beta)",
                    "params.act_type",
                ])
                if ker.support_splitk():
                    params_str.append(f"split_k_slices")
                    if ker.have_workspace:
                        params_str.append(f"workspace.raw_data()")
                else:
                    params_str.append(f"1")
                    if ker.have_workspace:
                        params_str.append(f"nullptr")
                params_str.append("d_is_bias")
                param_str = ", ".join(params_str)
                code.raw(f"""
                {param_type_str} ker_params({param_str});
                """)
                    # code.raw(f"""
                    # {param_type_str} ker_params(
                    #     problem, a_ten.data_ptr<const {ker.dtype_a}>(), b_ten.data_ptr<const {ker.dtype_b}>(),
                    #     c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(),
                    #     mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(),
                    #     indices.data_ptr<const int32_t>(), {mask_out_ptr} params.mask_filter, 
                    #     params.reverse_mask, {mask_width_str} 
                    #     {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta){", split_k_slices, workspace.raw_data()" if ker.support_splitk() else ""});
                    # """)
                # else:
                #     params_str.extend([
                #         f"{ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta)",
                #     ])

                #     code.raw(f"""
                #     {param_type_str} ker_params(
                #         problem, a_ten.data_ptr<const {ker.dtype_a}>(), b_ten.data_ptr<const {ker.dtype_b}>(),
                #         c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                #         {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta){", split_k_slices, workspace.raw_data()" if ker.support_splitk() else ""});
                #     """)
                if p.op_type == ConvOpType.kBackwardWeight:
                    code.raw(f"""
                    int num_reduced_mask = tv::div_up(ker_params.problem.N, ker_params.mask_width);
                    TV_ASSERT_RT_ERR(mask.dim(0) >= num_reduced_mask, "error");
                    """)
                code.raw(f"""
                tv::cuda::Launch launcher(ker_params.grid_dims, dim3({ker.num_threads}),
                                            {ker.smem_size}, reinterpret_cast<cudaStream_t>(params.stream));
                cudaError_t result;
                if ({ker.smem_size} >= (48 << 10)) {{
                    result = cudaFuncSetAttribute({ker.get_algo_name()}::conv_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    {ker.smem_size});
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                    result = cudaFuncSetAttribute(
                        {ker.get_algo_name()}::conv_kernel,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                }}
                """)
                # if cudasim.enable_debug():
                code.raw(f"""
                auto timer = tv::CUDATimer(params.verbose);
                // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                {{
                    tv::CUDAKernelTimerGuard timerguard(\"{ker.get_algo_name()}\", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                    launcher({ker.get_algo_name()}::conv_kernel, ker_params);
                }}
                """)
                if p.mask_sparse:
                    code.raw(f"""
                    TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", input.shape(), weight.shape(), output.shape(), 
                        indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                    """)
                else:
                    code.raw(f"""
                    TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", input.shape(), weight.shape(), output.shape());
                    """)

                # if cudasim.enable_debug():
                code.raw(f"""
                if (params.verbose){{
                    cudaFuncAttributes attr;
                    checkCudaErrors(
                        cudaFuncGetAttributes(&attr, {ker.get_algo_name()}::conv_kernel));
                    tv::ssprint("{ker.get_algo_name()} kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                }}

                """)
                code.raw(f"return;")
            code.raw("""
            if (!found){
                TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.__repr__(), tv::dtype_str(input.dtype()), 
                    tv::dtype_str(weight.dtype()), tv::dtype_str(output.dtype()), tv::dtype_str(dacc), 
                    tv::dtype_str(dcomp));
            }
            """)
        code_main.raw("""
        TV_THROW_RT_ERR("can't find any suitable algo for your parameters.", 
            algo_desp.algo, algo_desp.dtype_a, algo_desp.dtype_b, algo_desp.dtype_c,
            algo_desp.__repr__());
        """)
        return code_main


    # @pccm.pybind.mark
    # @pccm.cuda.static_function
    def implicit_gemm2_deprecated(self):
        code = pccm.code()
        for p, ker in zip(self.all_params, self.all_kernels):
            code.add_param_class("cp" + ker.get_algo_name(), ker.gemm_params,
                                 "ConvParams" + ker.get_algo_name())
            code.add_param_class(ker.get_algo_name(), ker,
                                 "Conv" + ker.get_algo_name())
        code.arg("params", "tv::gemm::ConvParams", pyanno="cumm.tensorview.gemm.ConvParams")
        code.add_dependency(TensorViewKernel)
        ch_first = ConvLayoutType.ChannelFirst.value
        nvrtc_conv_template(code)
        for kers in self.conv_select_helper(self.all_kernels, code):
            ker = kers[0]
            p = ker.problem
            param_type_str = "ConvParams" + ker.get_algo_name()
            indices = conv_iwo_012_to_abc(ker.problem.op_type)
            inv_indices = gemm_abc_012_to_iwo(ker.problem.op_type)
            dtypes_abc = [ker.dtype_a, ker.dtype_b, ker.dtype_c]
            dtypes_iwo = [dtypes_abc[i] for i in indices]
            param_cls_name = "ConvParams" + ker.get_algo_name()
            param_cls_ns = "cp" + ker.get_algo_name()

            if not ker.support_splitk():
                code.raw(f"""
                TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                """)
            # TODO if input is NCxHWx
            # TODO if input weight and output have different layout
            dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
            io_ndim = 2 if p.mask_sparse else p.ndim + 2
            weight_ndim = 3 if p.mask_sparse else p.ndim + 2
            abc_names = ["a", "b", "c"]
            abc_names = [abc_names[i] for i in indices]
            input_names = ["input", "weight", "output"]
            input_names = [input_names[i] for i in inv_indices]

            code.raw(f"""
            // {ker.get_algo_name()}
            found = true;
            bool d_is_bias = !bias.empty();
            """)
            if not ker.split_d_params:
                code.raw(f"""
                TV_ASSERT_RT_ERR(bias.empty(), "bias must be empty if split_d_params not enable");
                """)
                if ker.int8_inference:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(!bias.empty(), "bias must not be empty if int8_inference enable");
                    TV_ASSERT_RT_ERR(!scale.empty(), "scale must not be empty if int8_inference enable");
                    """)

            if p.mask_sparse:
                if p.op_type == ConvOpType.kBackwardWeight:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(mask_width > 0 && mask_width % {ker.tile_shape[2]} == 0, "error");
                    """)
                code.raw(f"""
                TV_ASSERT_RT_ERR(!indices.empty(), "error");
                TV_ASSERT_RT_ERR(!mask.empty(), "error");
                TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                int kernel_volume = weight.dim({dim_start});
                tv::check_shape(indices, {{kernel_volume, -1}});
                N = indices.dim(1);
                {param_cls_ns}::ConvProblem problem(N, C, K, kernel_volume, 
                    tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                """)
                if p.op_type == ConvOpType.kBackwardWeight:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                    # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * {ker.dtype_b.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    """)
                elif p.op_type == ConvOpType.kForward:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                    # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    """)
                else:
                    code.raw(f"""
                    # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    TV_ASSERT_RT_ERR(N == input.dim(0), "error");
                    """)
            else:
                code.raw(f"""
                tv::array<int, {p.ndim}> input_dims, output_dims;
                tv::array<int, {p.ndim}> ksize;
                TV_ASSERT_RT_ERR({p.ndim} == padding.size() && {p.ndim} == stride.size() && {p.ndim} == dilation.size(), "error");
                for (int i = {dim_start}; i < {dim_start + p.ndim}; ++i){{
                    ksize[i - {dim_start}] = weight.dim(i);
                    input_dims[i - {dim_start}] = input.dim(i);
                    output_dims[i - {dim_start}] = output.dim(i);
                }}
                int kernel_volume = ksize.op<tv::arrayops::prod>();
                tv::array<int, {p.ndim}> padding_arr{{{code.unpack([f"padding[{i}]" for i in range(p.ndim)])}}};
                tv::array<int, {p.ndim}> stride_arr{{{code.unpack([f"stride[{i}]" for i in range(p.ndim)])}}};
                tv::array<int, {p.ndim}> dilation_arr{{{code.unpack([f"dilation[{i}]" for i in range(p.ndim)])}}};
                auto output_dims_check_again = {param_cls_ns}::ConvProblem::calc_output_dims(input_dims, ksize, padding_arr, stride_arr, dilation_arr);
                for (int i = 0; i < {p.ndim}; ++i){{
                    TV_ASSERT_RT_ERR(output_dims_check_again[i] == output_dims[i], "error");
                }}
                {param_cls_ns}::ConvProblem problem(N, C, K, input_dims, output_dims, ksize, padding_arr, stride_arr, dilation_arr, 
                    tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                """)
            params_str = [
                "problem", 
                f"a_ten.data_ptr<const {ker.dtype_a}>()",
                f"b_ten.data_ptr<const {ker.dtype_b}>()",
                f"c_ten.data_ptr<{ker.dtype_c}>()",
                f"bias.empty() ? c_ten.data_ptr<const {ker.dtype_c}>() : bias.data_ptr<const {ker.dtype_c}>()",
            ]
            if p.mask_sparse:
                # mask_out_ptr = "mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), "
                # if p.op_type != ConvOpType.kForward:
                #     mask_out_ptr = ""

                # mask_width_str = "mask_width,"
                # if p.op_type != ConvOpType.kBackwardWeight:
                #     mask_width_str = ""
                params_str.extend([
                    "mask.data_ptr<const uint32_t>()",
                    "mask_argsort.data_ptr<const int32_t>()",
                    "indices.data_ptr<const int32_t>()",
                ])
                if p.op_type == ConvOpType.kForward:
                    params_str.append(f"mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>()")
                params_str.extend([
                    "params.mask_filter",
                    "params.reverse_mask",
                ])
                if p.op_type == ConvOpType.kBackwardWeight:
                    params_str.append("mask_width")
                
            params_str.extend([
                f"{ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta)",
                f"{ker.dtype_comp}(params.act_alpha), {ker.dtype_comp}(params.act_beta)",
                "params.act_type",
            ])
            if ker.support_splitk():
                params_str.append(f"split_k_slices")
                if ker.have_workspace:
                    params_str.append(f"workspace.raw_data()")
            else:
                params_str.append(f"1")
                if ker.have_workspace:
                    params_str.append(f"nullptr")
            params_str.append("d_is_bias")
            param_str = ", ".join(params_str)
            code.raw(f"""
            {param_type_str} ker_params({param_str});
            """)
                # code.raw(f"""
                # {param_type_str} ker_params(
                #     problem, a_ten.data_ptr<const {ker.dtype_a}>(), b_ten.data_ptr<const {ker.dtype_b}>(),
                #     c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(),
                #     mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(),
                #     indices.data_ptr<const int32_t>(), {mask_out_ptr} params.mask_filter, 
                #     params.reverse_mask, {mask_width_str} 
                #     {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta){", split_k_slices, workspace.raw_data()" if ker.support_splitk() else ""});
                # """)
            # else:
            #     params_str.extend([
            #         f"{ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta)",
            #     ])

            #     code.raw(f"""
            #     {param_type_str} ker_params(
            #         problem, a_ten.data_ptr<const {ker.dtype_a}>(), b_ten.data_ptr<const {ker.dtype_b}>(),
            #         c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
            #         {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta){", split_k_slices, workspace.raw_data()" if ker.support_splitk() else ""});
            #     """)
            if p.op_type == ConvOpType.kBackwardWeight:
                code.raw(f"""
                int num_reduced_mask = tv::div_up(ker_params.problem.N, ker_params.mask_width);
                TV_ASSERT_RT_ERR(mask.dim(0) >= num_reduced_mask, "error");
                """)
            code.raw(f"""
            tv::cuda::Launch launcher(ker_params.grid_dims, dim3({ker.num_threads}),
                                        {ker.smem_size}, reinterpret_cast<cudaStream_t>(params.stream));
            cudaError_t result;
            if ({ker.smem_size} >= (48 << 10)) {{
                result = cudaFuncSetAttribute({ker.get_algo_name()}::conv_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                {ker.smem_size});
                TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                result = cudaFuncSetAttribute(
                    {ker.get_algo_name()}::conv_kernel,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
            }}
            """)
            # if cudasim.enable_debug():
            code.raw(f"""
            auto timer = tv::CUDATimer(params.verbose);
            // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
            {{
                tv::CUDAKernelTimerGuard timerguard(\"{ker.get_algo_name()}\", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                launcher({ker.get_algo_name()}::conv_kernel, ker_params);
            }}
            """)
            if p.mask_sparse:
                code.raw(f"""
                TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", input.shape(), weight.shape(), output.shape(), 
                    indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                """)
            else:
                code.raw(f"""
                TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", input.shape(), weight.shape(), output.shape());
                """)

            # if cudasim.enable_debug():
            code.raw(f"""
            if (params.verbose){{
                cudaFuncAttributes attr;
                checkCudaErrors(
                    cudaFuncGetAttributes(&attr, {ker.get_algo_name()}::conv_kernel));
                tv::ssprint("{ker.get_algo_name()} kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
            }}

            """)
            code.raw(f"return;")
        code.raw("""
        if (!found){
            TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.__repr__(), tv::dtype_str(input.dtype()), 
                tv::dtype_str(weight.dtype()), tv::dtype_str(output.dtype()), tv::dtype_str(dacc), 
                tv::dtype_str(dcomp));
        }
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def get_all_conv_algo_desp(self):
        code = pccm.code()
        code.raw(f"""
        std::vector<tv::gemm::ConvAlgoDesp> desps;
        """)
        for ker in self.all_kernels:
            min_arch = (0, 0)
            ker_min_arch = ker.min_arch()
            if ker_min_arch is not None:
                min_arch = ker_min_arch
            code.raw("{")
            code.raw(f"""
            tv::gemm::ConvAlgoDesp desp({ker.problem.ndim}, tv::gemm::ConvOpType({ker.problem.op_type.value}));
            desp.dtype_a = {ker.dtype_a.tv_dtype};
            desp.dtype_b = {ker.dtype_b.tv_dtype};
            desp.dtype_c = {ker.dtype_c.tv_dtype};
            desp.dacc = {ker.dtype_acc.tv_dtype};
            desp.dcomp = {ker.dtype_comp.tv_dtype};

            desp.trans_a_set({pccm.boolean(ker.trans_a)});
            desp.trans_b_set({pccm.boolean(ker.trans_b)});
            desp.trans_c_set({pccm.boolean(ker.trans_c)});
            desp.tile_shape = {{{ker.tile_shape[0]}, {ker.tile_shape[1]}, {ker.tile_shape[2]}}};
            desp.warp_tile_shape = {{{ker.warp_tile_shape[0]}, {ker.warp_tile_shape[1]}, {ker.warp_tile_shape[2]}}};
            """)
            if ker.tensorop is not None:
                code.raw(
                    f"desp.tensorop = {{{ker.tensorop[0]}, {ker.tensorop[1]}, {ker.tensorop[2]}}};"
                )
            else:
                code.raw(f"desp.tensorop = {{-1, -1, -1}};")
            code.raw(f"""
            desp.num_stage = {ker.num_stage};
            desp.algo = "{ker.algo.value}";
            desp.split_k_serial_set({pccm.boolean(ker.splitk_serial)});
            desp.split_k_parallel_set({pccm.boolean(ker.splitk_parallel)});
            desp.shuffle_type = static_cast<tv::gemm::ShuffleStrideType>({ker.shuffle_stride.value});
            desp.element_per_access_a = {ker.input_spec.input_iter_a.element_per_acc};
            desp.element_per_access_b = {ker.input_spec.input_iter_b.element_per_acc};
            desp.element_per_access_c = {ker.output_spec.out_iter.element_per_acc};
            desp.access_per_vector = {ker.access_per_vector};
            desp.min_arch = std::make_tuple({min_arch[0]}, {min_arch[1]});

            // Conv attrs
            desp.ndim = {ker.problem.ndim};
            desp.op_type = static_cast<tv::gemm::ConvOpType>({ker.problem.op_type.value});
            desp.iter_algo = static_cast<tv::gemm::ConvIterAlgo>({ker.iter_algo.value});
            desp.layout_i = static_cast<tv::gemm::ConvLayoutType>({ker.problem.layout_desp_input.layout_type.value});
            desp.layout_w = static_cast<tv::gemm::ConvLayoutType>({ker.problem.layout_desp_weight.layout_type.value});
            desp.layout_o = static_cast<tv::gemm::ConvLayoutType>({ker.problem.layout_desp_output.layout_type.value});
            desp.interleave_i = {ker.problem.layout_desp_input.interleave};
            desp.interleave_w = {ker.problem.layout_desp_weight.interleave};
            desp.interleave_o = {ker.problem.layout_desp_output.interleave};
            desp.mask_sparse = {pccm.boolean(ker.mask_sparse)};
            desp.increment_k_first = {pccm.boolean(ker.increment_k_first)};
            desp.is_int8_inference = {pccm.boolean(ker.int8_inference)};
            desp.dynamic_mask = {pccm.boolean(ker.dynamic_mask)};
            TV_ASSERT_RT_ERR(desp.__repr__() == \"{ker.get_algo_name()}\", "error", desp.__repr__(), \"{ker.get_algo_name()}\");
            desps.push_back(desp);
            """)
            code.raw("}")
        code.raw(f"""
        return desps;
        """)
        return code.ret("std::vector<tv::gemm::ConvAlgoDesp>",
                        pyanno="List[ConvAlgoDesp]")

    # @lineprof.lineprof_wrapper_cpp
    def implicit_gemm_python(self,
                             input_: np.ndarray,
                             weight: np.ndarray,
                             output: np.ndarray,
                             input_meta: np.ndarray,
                             weight_meta: np.ndarray,
                             output_meta: np.ndarray,
                             padding: List[int],
                             stride: List[int],
                             dilation: List[int],
                             ndim: int,
                             iter_algo: ConvIterAlgo,
                             op_type: ConvOpType,
                             i_ltype: ConvLayoutType,
                             w_ltype: ConvLayoutType,
                             o_ltype: ConvLayoutType,
                             ts: np.ndarray,
                             wts: np.ndarray,
                             num_stage: int,
                             dacc: dtypes.DType,
                             dcomp: dtypes.DType,
                             algo: str,
                             tensorop: np.ndarray,
                             i_interleave: int = 1,
                             w_interleave: int = 1,
                             o_interleave: int = 1):
        found = False
        for p, ker in zip(self.all_params, self.all_kernels):
            indices = conv_iwo_012_to_abc(p.op_type)
            inv_indices = gemm_abc_012_to_iwo(p.op_type)
            dtypes_abc = [p.dtype_a, p.dtype_b, p.dtype_c]
            dtypes_iwo = [dtypes_abc[i] for i in indices]

            if_tests = [
                dtypes_iwo[0].npdtype() == input_.dtype,
                dtypes_iwo[1].npdtype() == weight.dtype,
                dtypes_iwo[2].npdtype() == output.dtype,
                p.layout_desp_input.layout_type == i_ltype,
                p.layout_desp_weight.layout_type == w_ltype,
                p.layout_desp_output.layout_type == o_ltype,
                p.layout_desp_input.interleave == i_interleave,
                p.layout_desp_weight.interleave == w_interleave,
                p.layout_desp_output.interleave == o_interleave,
                p.ts[0] == ts[0] and p.ts[1] == ts[1] and p.ts[2] == ts[2],
                p.wts[0] == wts[0] and p.wts[1] == wts[1]
                and p.wts[2] == wts[2],
                p.num_stage == num_stage,
                p.dtype_acc == dacc,
                p.dtype_comp == dcomp,
                algo == p.algo.value,
            ]
            if all(if_tests):
                found = True
                assert input_.ndim == p.ndim + 2
                assert weight.ndim == p.ndim + 2
                assert output.ndim == p.ndim + 2
                N = input_.shape[0]
                if p.layout_desp_input.is_channel_first():
                    C = input_.shape[1]
                else:
                    C = input_.shape[p.ndim + 1]
                K = weight.shape[0]
                if p.layout_desp_output.is_channel_first():
                    K2 = output.shape[1]
                else:
                    K2 = output.shape[p.ndim + 1]
                assert K == K2
                ksize = [0] * p.ndim
                input_dims = [0] * p.ndim
                output_dims = [0] * p.ndim
                dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
                for i in range(dim_start, dim_start + p.ndim):
                    ksize[i - dim_start] = weight.shape[i]
                    input_dims[i - dim_start] = input_.shape[i]
                    output_dims[i - dim_start] = output.shape[i]

                output_dims_check_again = ConvProblem.calc_output_dims_python(
                    input_dims, ksize, padding, stride, dilation)
                assert output_dims_check_again == output_dims
                problem = ker.problem.python_ctor(N, C, K, input_dims,
                                                  output_dims, ksize, padding,
                                                  stride, dilation,
                                                  ConvMode.kCrossCorrelation,
                                                  1, 1)
                print(problem.N_, problem.C_, problem.K_, problem.output_dims_)
                inputs = [input_, weight, output]
                input_metas = [input_meta, weight_meta, output_meta]
                input_abcs = [inputs[i] for i in inv_indices]
                input_meta_abcs = [input_metas[i] for i in inv_indices]

                a_ten = input_abcs[0]
                b_ten = input_abcs[1]
                c_ten = input_abcs[2]
                a_meta_ten = input_meta_abcs[0]
                b_meta_ten = input_meta_abcs[1]

                if cudasim.enable_debug():
                    a_ptr = ArrayPtr(p.dtype_a.tv_dtype,
                                     a_ten.size,
                                     external_data=tv.from_numpy(a_ten),
                                     meta_data=tv.from_numpy(a_meta_ten))
                    b_ptr = ArrayPtr(p.dtype_b.tv_dtype,
                                     b_ten.size,
                                     external_data=tv.from_numpy(b_ten),
                                     meta_data=tv.from_numpy(b_meta_ten))
                else:
                    a_ptr = ArrayPtr(p.dtype_a.tv_dtype,
                                     a_ten.size,
                                     external_data=tv.from_numpy(a_ten),
                                     meta_data=tv.Tensor())
                    b_ptr = ArrayPtr(p.dtype_b.tv_dtype,
                                     b_ten.size,
                                     external_data=tv.from_numpy(b_ten),
                                     meta_data=tv.Tensor())

                c_ptr = ArrayPtr(p.dtype_c.tv_dtype,
                                 c_ten.size,
                                 external_data=tv.from_numpy(c_ten))
                params = ker.gemm_params.python_ctor(problem, a_ptr, b_ptr,
                                                     c_ptr, c_ptr, 1.0, 0.0)
                func = partial(ker.conv_kernel_python, params=params)
                blocks = params.grid_dims
                threads = cudasim.Dim3(ker.num_threads, 1, 1)
                return asyncio.run(
                    cudasim.kernel_launch(func, blocks, threads,
                                          ker.smem_size)), blocks, threads
        raise NotImplementedError
