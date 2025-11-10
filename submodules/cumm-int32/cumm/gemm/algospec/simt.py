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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pccm

from cumm import dtypes
from cumm.common import (GemmBasic, GemmBasicKernel, TensorView,
                         TensorViewKernel)
from cumm.gemm import (constants, gemmmath, layout, mask_iters, out_iters,
                       output_op, thread_map, volta_iters, volta_out_iters,
                       wmma)
from cumm.gemm.algospec import bases
from cumm.gemm.bases import (GemmApply, GemmInputIterator, GemmIterator,
                             GemmOutFragIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator)
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.wmma.simt import WarpMmaSimt

from .core import GemmAlgo, ShuffleStrideType, TensorOp


def simt_transpose_padding(threads: int, crosswise: int, size_in_bits: int):
    if (size_in_bits >= 32):
        return threads // crosswise // (size_in_bits // 32)
    else:
        return threads // crosswise * (32 // size_in_bits)


class InputSimt(bases.Input):
    def __init__(
            self,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            dtype_a: dtypes.DType,
            dtype_b: dtypes.DType,
            trans_a: bool,
            trans_b: bool,
            algo: GemmAlgo = GemmAlgo.Simt,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            tensor_op: Optional[TensorOp] = None):
        is_dp4a = algo == GemmAlgo.SimtDP4A
        self._trans_a = trans_a
        self._trans_b = trans_b

        if not is_dp4a:
            self.input_trans_load_a = False
            self.input_trans_load_b = False
            self.input_last_residual = True
        else:
            # for DP4A, we need to transpose matrix to MK/NK
            self.input_trans_load_a = trans_a
            self.input_trans_load_b = not trans_b
            self.input_last_residual = False
        m = tile_shape[0]
        n = tile_shape[1]
        k = tile_shape[2]
        self._tile_shape = tile_shape
        self.input_tile_shape_a = seq(m, k)
        if trans_a:
            self.input_tile_shape_a = seq(k, m)
        self.input_tile_shape_b = seq(k, n)
        if trans_b:
            self.input_tile_shape_b = seq(n, k)
        self.advance_axis_a = 0 if trans_a else 1
        self.advance_axis_b = 1 if trans_b else 0
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE
        # Simt
        if is_dp4a:
            # Why dp4a load a 4x4 sub matrix?
            # 1. dp4a inst shape is 1x1x4, so we need to load 4 element in k axis
            # 2. if transposed layout, we need to load 4x4 and transpose it.
            # 3. we need vector load from smem to register.
            self.input_sub_tile_shape_a = seq(4, 4)
            self.input_sub_tile_shape_b = seq(4, 4)
        # elif dtype_a == dtypes.float16 and dtype_b == dtypes.float16:
        #     self.input_sub_tile_shape_a = seq(2, 2)
        #     self.input_sub_tile_shape_b = seq(2, 2)
        else:
            self.input_sub_tile_shape_a = seq(1, 1)
            self.input_sub_tile_shape_b = seq(1, 1)

        self.alignment_a = self.input_sub_tile_shape_a[1]
        self.alignment_b = self.input_sub_tile_shape_b[1]
        self.tmap_a = thread_map.PitchLinear(self.input_tile_shape_a,
                                             self.input_sub_tile_shape_a,
                                             self.num_threads)
        self.tmap_b = thread_map.PitchLinear(self.input_tile_shape_b,
                                             self.input_sub_tile_shape_b,
                                             self.num_threads)
        shuffle_a = shuffle_stride != ShuffleStrideType.NoShuffle
        shuffle_b = shuffle_stride == ShuffleStrideType.ShuffleAB

        inp_iter_a_param = mask_iters.MaskTileIteratorParams(
            dtype_a, self.input_tile_shape_a, self.input_sub_tile_shape_a,
            self.tmap_a, self.advance_axis_a, shuffle_a)

        inp_iter_b_param = mask_iters.MaskTileIteratorParams(
            dtype_b, self.input_tile_shape_b, self.input_sub_tile_shape_b,
            self.tmap_b, self.advance_axis_b, shuffle_b)

        self.inp_iter_a = mask_iters.MaskTileIterator(
            dtype_a, self.input_tile_shape_a, self.input_sub_tile_shape_a,
            self.tmap_a, inp_iter_a_param, self.advance_axis_a,
            self.alignment_a, self.input_trans_load_a,
            self.input_last_residual, shuffle_a)

        self.inp_iter_b = mask_iters.MaskTileIterator(
            dtype_b, self.input_tile_shape_b, self.input_sub_tile_shape_b,
            self.tmap_b, inp_iter_b_param, self.advance_axis_b,
            self.alignment_b, self.input_trans_load_b,
            self.input_last_residual, shuffle_b)

    @property
    def input_iter_a(self) -> GemmInputIterator:
        return self.inp_iter_a

    @property
    def input_iter_b(self) -> GemmInputIterator:
        return self.inp_iter_b

    @property
    def trans_a(self) -> bool:
        return self._trans_a

    @property
    def trans_b(self) -> bool:
        return self._trans_b

    @property
    def tile_shape(self) -> MetaArray[int]:
        return self._tile_shape


class MmaSimt(bases.Mma):
    def __init__(self,
                 input_spec: bases.Input,
                 tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int],
                 num_stage: int,
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_acc: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 tensorop: Optional[TensorOp] = None,
                 algo: GemmAlgo = GemmAlgo.Simt):
        self._input_spec = input_spec
        is_dp4a = algo == GemmAlgo.SimtDP4A
        # input_sub_tile_shape_a = input_spec.
        if is_dp4a:
            self.input_sub_tile_shape_a = seq(4, 4)
            self.input_sub_tile_shape_b = seq(4, 4)
        # elif dtype_a == dtypes.float16 and dtype_b == dtypes.float16:
        #     self.input_sub_tile_shape_a = seq(2, 2)
        #     self.input_sub_tile_shape_b = seq(2, 2)
        else:
            self.input_sub_tile_shape_a = seq(1, 1)
            self.input_sub_tile_shape_b = seq(1, 1)

        if not trans_a:
            padding_m = simt_transpose_padding(constants.WARP_SIZE,
                                               tile_shape[2],
                                               dtype_a.bitsize())
        else:
            padding_m = 0

        if trans_b:
            padding_n = simt_transpose_padding(constants.WARP_SIZE,
                                               tile_shape[2],
                                               dtype_b.bitsize())
        else:
            padding_n = 0

        self._padding_mn = seq(padding_m, padding_n)
        self._accumulator_size = warp_tile_shape[0] * warp_tile_shape[
            1] // constants.WARP_SIZE
        warp_shape = seq(8, 4)
        if warp_tile_shape[0] <= warp_tile_shape[1]:
            warp_shape = seq(4, 8)
        self.warp_shape = warp_shape
        thread_mma_shape = seq(warp_tile_shape[0] // warp_shape[0],
                               warp_tile_shape[1] // warp_shape[1])
        lane_vec_load_shape = seq(constants.OPTIM_ACCESS // dtype_a.itemsize(),
                                  constants.OPTIM_ACCESS // dtype_b.itemsize())
        self.thread_mma_shape = thread_mma_shape
        if is_dp4a:
            lane_mma_shape = seq(min(lane_vec_load_shape[0], 4),
                                 min(lane_vec_load_shape[1], 4), 4)
        # elif dtype_a == dtypes.float16 and dtype_b == dtypes.float16:
        #     lane_mma_shape = seq(
        #         min(lane_vec_load_shape[0], 4),
        #         min(lane_vec_load_shape[1], 4), 2)
        else:
            lane_mma_shape = seq(
                min(lane_vec_load_shape[0], thread_mma_shape[0]),
                min(lane_vec_load_shape[1], thread_mma_shape[1]), 1)
        self.lane_mma_shape = lane_mma_shape
        lane_interleave = 2 if thread_mma_shape[0] > 4 and thread_mma_shape[
            1] > 4 else 1
        if is_dp4a:
            self.lane_layout = layout.RowMajorInterleaved(lane_interleave)
        else:
            self.lane_layout = layout.RowMajorInterleaved(lane_interleave)
        self.warp_gemm_iters = warp_tile_shape[2] // lane_mma_shape[
            2]  # type: int
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE
        self._partk = self.warp_count_shape[2]
        self._smem_iter_a = mask_iters.SmemTileIteratorV2(
            dtype_a,
            input_spec.thread_map_a,
            seq(tile_shape[2], tile_shape[0]),
            self.input_sub_tile_shape_a,
            0,
            self.num_threads,
            smem_shape=seq(0, padding_m) +
            seq(tile_shape[2] * num_stage, tile_shape[0]),
            transposed_input=not trans_a)

        self._smem_iter_b = mask_iters.SmemTileIteratorV2(
            dtype_b,
            input_spec.thread_map_b,
            seq(tile_shape[2], tile_shape[1]),
            self.input_sub_tile_shape_b,
            0,
            self.num_threads,
            smem_shape=seq(0, padding_n) +
            seq(tile_shape[2] * num_stage, tile_shape[1]),
            transposed_input=trans_b)
        self._warp_iter_a = mask_iters.WarpTileIterator(
            dtype_a,
            seq(tile_shape[2], tile_shape[0] + padding_m),
            seq(warp_tile_shape[2], warp_tile_shape[0]),
            warp_shape,
            seq(lane_mma_shape[2], lane_mma_shape[0]),
            layout.RowMajorInterleaved(self.input_sub_tile_shape_a[0]),
            self.lane_layout,
            padding_m,
            True,
            partk=self.partk)

        self._warp_iter_b = mask_iters.WarpTileIterator(
            dtype_b,
            seq(tile_shape[2], tile_shape[1] + padding_n),
            seq(warp_tile_shape[2], warp_tile_shape[1]),
            warp_shape,
            seq(lane_mma_shape[2], lane_mma_shape[1]),
            layout.RowMajorInterleaved(self.input_sub_tile_shape_b[0]),
            self.lane_layout,
            padding_n,
            False,
            partk=self.partk)
        # if dtype_a == dtypes.float16 and dtype_b == dtypes.float16:
        #     self._warp_mma = WarpMmaSimt(
        #         (thread_mma_shape[0], thread_mma_shape[1], lane_mma_shape[2]),
        #         dtype_a, dtype_b, dtype_acc, False, True, False)
        # else:
        self._warp_mma = WarpMmaSimt(
            (thread_mma_shape[0], thread_mma_shape[1], lane_mma_shape[2]),
            dtype_a, dtype_b, dtype_acc, True, False, False)

    @property
    def input_spec(self) -> bases.Input:
        return self._input_spec

    @property
    def padding_mn(self):
        return self._padding_mn

    @property
    def partk(self):
        return self._partk

    @property
    def smem_iter_a(self):
        return self._smem_iter_a

    @property
    def smem_iter_b(self):
        return self._smem_iter_b

    @property
    def warp_iter_a(self):
        return self._warp_iter_a

    @property
    def warp_iter_b(self):
        return self._warp_iter_b

    @property
    def warp_mma(self):
        return self._warp_mma

    @property
    def num_warp_mma_iters(self):
        return self.warp_gemm_iters

    @property
    def accumulator_size(self) -> int:
        return self._accumulator_size


class OutputSimt(bases.Output):
    def __init__(
            self,
            mma_spec: MmaSimt,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            num_stage: int,
            dtype_c: dtypes.DType,
            dtype_acc: dtypes.DType,
            dtype_comp: dtypes.DType,
            trans_c: bool,
            tensorop: Optional[TensorOp] = None,
            algo: GemmAlgo = GemmAlgo.Simt,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            access_per_vector: int = 1,
            int8_inference: bool = False,
            with_source: bool = False):
        assert algo == GemmAlgo.Simt or algo == GemmAlgo.SimtDP4A
        self._mma_spec = mma_spec
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.acc_frag_iter = out_iters.OutFragIter(
            dtype_acc, mma_spec.lane_mma_shape[1],
            mma_spec.thread_mma_shape[1] // mma_spec.lane_mma_shape[1])
        self.out_warp_tile_iter = out_iters.OutWarpTileIterator(
            dtype_acc,
            warp_tile_shape,
            mma_spec.warp_shape,
            mma_spec.lane_mma_shape,
            mma_spec.lane_layout,
            stride=tile_shape[1],
            scalar_store=True)
        self.out_smem_padding = self.out_warp_tile_iter.padding
        # 1, 4, 4, 1, 128
        self.part_shape = seq(1, mma_spec.warp_count_shape[0],
                              mma_spec.warp_shape[0], 1, tile_shape[1])
        group_count = warp_tile_shape[0] // (mma_spec.warp_shape[0] *
                                             mma_spec.lane_mma_shape[0])
        self.part_dilation = seq(warp_tile_shape[0] // mma_spec.warp_shape[0],
                                 1, group_count, mma_spec.lane_mma_shape[0], 1)
        self.smem_part_dilation = seq(1, 1, 1, 1, 1)
        output_op_count = 1
        self.output_op_count = output_op_count
        self._num_out_iters = warp_tile_shape[0] // mma_spec.warp_shape[0]

        self.out_tmap = thread_map.Out5DLinear(dtype_acc, self.part_shape,
                                               self.part_dilation,
                                               self.warp_count,
                                               output_op_count)
        self.out_smem_tmap = thread_map.Out5DLinear(dtype_acc, self.part_shape,
                                                    self.smem_part_dilation,
                                                    self.warp_count,
                                                    output_op_count)
        self.shared_mem_alignment = dtype_acc.bitsize(
        ) * self.out_tmap.element_per_acc // 8
        self._frag_per_iter = 1
        self.out_smem_loader = out_iters.OutSmemLoader(
            dtype_acc, self.out_smem_tmap, output_op_count,
            tile_shape[1] + self.out_smem_padding[1],
            self.shared_mem_alignment)
        shuffle = shuffle_stride == ShuffleStrideType.ShuffleAC
        out_iter_params = out_iters.OutIteratorParams(self.out_tmap, shuffle)
        bias_out_iter_params = out_iters.OutIteratorParams(self.out_tmap, False, int8_inference)
        self._out_iter = out_iters.OutIterator(
            dtype_c,
            self.out_tmap,
            out_iter_params,
            self.part_shape,
            self.part_dilation,
            output_op_count,
            shuffle_in_stride=shuffle,
            access_per_vector=access_per_vector,
            with_source=with_source)
        self._const_out_iter = out_iters.OutIterator(
            dtype_c,
            self.out_tmap,
            out_iter_params,
            self.part_shape,
            self.part_dilation,
            output_op_count,
            read_only=True,
            shuffle_in_stride=shuffle,
            access_per_vector=access_per_vector)
        rate = min(max(dtype_comp.bitsize() // dtype_c.bitsize(), 1), output_op_count)
        self.int8_scalebias_iterator = out_iters.OutIterator(
            dtype_comp,
            self.out_tmap,
            bias_out_iter_params,
            self.part_shape,
            self.part_dilation,
            output_op_count,
            read_only=True,
            shuffle_in_stride=False,
            access_per_vector=rate)
        self.out_unary_op_fp_t = f"tv::math::UnaryIdentity<{dtype_comp}, {output_op_count}>"
        self.out_unary_op_i8_t = f"tv::math::Clamp<{dtype_comp}, {dtype_c}, {output_op_count}>"
        out_unary_op_fp = gemmmath.UnaryActivation(dtype_comp, dtype_c,
                                                 output_op_count)
        out_unary_op_i8 = gemmmath.Clamp(dtype_comp, dtype_c, output_op_count)

        if dtype_c == dtypes.int8 and not int8_inference:
            self.out_unary_op_t = self.out_unary_op_i8_t
            self.out_unary_op = out_unary_op_i8
        else:
            self.out_unary_op_t = self.out_unary_op_fp_t
            self.out_unary_op = out_unary_op_fp
        if int8_inference:
            self._output_op = output_op.linear.Int8Inference(
                dtype_c, output_op_count, dtype_acc, dtype_comp, self.out_unary_op)
        else:
            self._output_op = output_op.linear.LinearCombination(
                dtype_c, output_op_count, dtype_acc, dtype_comp, self.out_unary_op)
        self._apply_op = output_op.apply.ApplyOutputOp(
            output_op_count, self.output_op, self.out_iter.fragment_t,
            self.out_smem_loader.fragment_t, self.int8_scalebias_iterator.fragment_t,
            int8_inference)

    @property
    def mma_spec(self) -> bases.Mma:
        return self._mma_spec

    @property
    def smem_padding(self) -> MetaArray[int]:
        return self.out_smem_padding

    @property
    def frag_iter(self) -> GemmOutFragIterator:
        return self.acc_frag_iter

    @property
    def warp_store_iter(self) -> GemmOutWarpIterator:
        return self.out_warp_tile_iter

    @property
    def smem_loader(self) -> GemmOutSmemLoader:
        return self.out_smem_loader

    @property
    def out_iter(self) -> GemmOutputIterator:
        return self._out_iter

    @property
    def const_out_iter(self) -> GemmOutputIterator:
        return self._const_out_iter
    
    @property
    def int8_scalebias_out_iter(self) -> GemmOutputIterator:
        return self.int8_scalebias_iterator

    @property
    def num_out_iters(self) -> int:
        return self._num_out_iters

    @property
    def frag_per_iter(self) -> int:
        return self._frag_per_iter

    @property
    def output_op(self) -> GemmOutputOp:
        return self._output_op

    @property
    def apply_op(self) -> GemmApply:
        return self._apply_op


class AlgoSpecificSimt(object):
    def __init__(
            self,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            num_stage: int,
            dtype_a: dtypes.DType,
            dtype_b: dtypes.DType,
            dtype_c: dtypes.DType,
            dtype_acc: dtypes.DType,
            dtype_comp: dtypes.DType,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            tensorop: Optional[TensorOp] = None,
            algo: GemmAlgo = GemmAlgo.Simt,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle):
        assert algo == GemmAlgo.Simt or algo == GemmAlgo.SimtDP4A
        self.input_spec = InputSimt(tile_shape, warp_tile_shape, dtype_a,
                                    dtype_b, trans_a, trans_b, algo,
                                    shuffle_stride)
        self.mma_spec = MmaSimt(self.input_spec, tile_shape, warp_tile_shape,
                                num_stage, dtype_a, dtype_b, dtype_acc,
                                trans_a, trans_b, tensorop, algo)
        self.output_spec = OutputSimt(self.mma_spec, tile_shape,
                                      warp_tile_shape, num_stage, dtype_c,
                                      dtype_acc, dtype_comp, trans_c, tensorop,
                                      algo, shuffle_stride)
