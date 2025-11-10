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
from cumm.gemm import (constants, gemmmath, layout, layout_tensorop,
                       mask_iters, out_iters, output_op, thread_map,
                       turing_iters, turing_my_iters, turing_out_iters, wmma)
from cumm.gemm.algospec import bases
from cumm.gemm.bases import (GemmApply, GemmInputIterator, GemmIterator,
                             GemmOutFragIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator)
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.wmma.simt import WarpMmaSimt

from .core import GemmAlgo, ShuffleStrideType, TensorOp


class InputTuring(bases.Input):
    def __init__(
            self,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            dtype_a: dtypes.DType,
            dtype_b: dtypes.DType,
            trans_a: bool,
            trans_b: bool,
            algo: GemmAlgo = GemmAlgo.Turing,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            tensor_op: Optional[TensorOp] = None,
            async_kernel: bool = False):
        m = tile_shape[0]
        n = tile_shape[1]
        k = tile_shape[2]
        self._trans_a = trans_a
        self._trans_b = trans_b
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

        access_size_bits = 128
                
        self.alignment_a = access_size_bits // dtype_a.bitsize()
        self.alignment_b = access_size_bits // dtype_b.bitsize()
        self.input_sub_tile_shape_a = seq(1, self.alignment_a)
        self.input_sub_tile_shape_b = seq(1, self.alignment_b)
        if not trans_a:
            warp_shape_raked_a = seq(
                1, tile_shape[2] // (access_size_bits // dtype_a.bitsize()))
            warp_shape_raked_a[
                0] = constants.WARP_SIZE // warp_shape_raked_a[1]
        else:
            warp_shape_raked_a = seq(4, 8)
            # rearrange the warp if block is extremely small
            if warp_shape_raked_a[1] * (access_size_bits // dtype_a.bitsize()) > tile_shape[0]:             # for low channel m
                warp_shape_raked_a[1] = tile_shape[0] // (access_size_bits // dtype_a.bitsize())
                warp_shape_raked_a[0] = constants.WARP_SIZE // warp_shape_raked_a[1]
                assert(warp_shape_raked_a.prod() == constants.WARP_SIZE)
        if trans_b:
            warp_shape_raked_b = seq(
                1, tile_shape[2] // (access_size_bits // dtype_b.bitsize()))
            warp_shape_raked_b[
                0] = constants.WARP_SIZE // warp_shape_raked_b[1]
        else:
            warp_shape_raked_b = seq(4, 8)
            if warp_shape_raked_b[1] * (access_size_bits // dtype_b.bitsize()) > tile_shape[1]:             #for low chanel n
                warp_shape_raked_b[1] = tile_shape[1] // (access_size_bits // dtype_b.bitsize())
                warp_shape_raked_b[0] = constants.WARP_SIZE // warp_shape_raked_b[1]
                assert(warp_shape_raked_b.prod() == constants.WARP_SIZE)
        self.tmap_a = thread_map.PitchLinearWarpRaked(
            self.input_tile_shape_a, self.input_sub_tile_shape_a,
            warp_shape_raked_a, self.num_threads)
        self.tmap_b = thread_map.PitchLinearWarpRaked(
            self.input_tile_shape_b, self.input_sub_tile_shape_b,
            warp_shape_raked_b, self.num_threads)
        self.padding_mn = seq(0, 0)

        self.input_trans_load_a = False
        self.input_trans_load_b = False
        self.input_last_residual = True
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


class MmaTuring(bases.Mma):
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
                 tensorop: TensorOp,
                 algo: GemmAlgo = GemmAlgo.Turing,
                 async_kernel: bool = False):
        self._input_spec = input_spec
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE

        tile_shape_km = seq(tile_shape[2], tile_shape[0])
        tile_shape_kn = seq(tile_shape[2], tile_shape[1])
        warp_tile_shape_km = seq(warp_tile_shape[2], warp_tile_shape[0])
        warp_tile_shape_kn = seq(warp_tile_shape[2], warp_tile_shape[1])
        self._partk = self.warp_count_shape[2]
        # if trans_a:
        #     warp_shape_raked_a = warp_shape_raked_a[::-1]  # type: np.ndarray
        # if trans_b:
        #     warp_shape_raked_b = warp_shape_raked_b[::-1]  # type: np.ndarray
        # print(warp_shape_raked_a, warp_shape_raked_b)
        # raise NotImplementedError
        self.tmap_a = input_spec.input_iter_a.tmap
        self.tmap_b = input_spec.input_iter_b.tmap
        self.warp_gemm_iters = warp_tile_shape[2] // tensorop.shape[2]
        self._accumulator_size = warp_tile_shape[0] * warp_tile_shape[
            1] // constants.WARP_SIZE

        self.input_trans_load_a = False
        self.input_trans_load_b = False
        self.input_last_residual = True
        tensor_op_num_threads = 32
        inst_shape_km = seq(tensorop.shape[2], tensorop.shape[0])
        inst_shape_kn = seq(tensorop.shape[2], tensorop.shape[1])
        crosswise_a = 128 // dtype_a.itemsize()
        if trans_a:
            is_crosswise_a = False
            crosswise_a = min(crosswise_a, tile_shape[0])  # if a thradBlock shape is small
        else:
            is_crosswise_a = True
            crosswise_a = tile_shape[2]

        crosswise_b = 128 // dtype_b.itemsize()
        if not trans_b:
            is_crosswise_b = False
            crosswise_b = min(crosswise_b, tile_shape[1])
        else:
            is_crosswise_b = True
            crosswise_b = tile_shape[2]

        smem_layout_a = layout_tensorop.TensorOpMultiplicand(
            dtype_a.bitsize(), crosswise_a)
        smem_layout_b = layout_tensorop.TensorOpMultiplicand(
            dtype_b.bitsize(), crosswise_b)
        if trans_a:  # permute m
            smem_layout_a.static_stride = tile_shape_km[1]
            my_smem_layout_a = turing_my_iters.MyTensorOpLayout(
                dtype_a, self.tmap_a.warp_shape, tile_shape_km[1], 0,
                seq(num_stage, 1), True)
        else:  # permute k
            smem_layout_a.static_stride = num_stage * tile_shape_km[0]
            my_smem_layout_a = turing_my_iters.MyTensorOpLayout(
                dtype_a, self.tmap_a.warp_shape, tile_shape_km[0], 1,
                seq(1, num_stage), False)

        if not trans_b:  # permute m
            smem_layout_b.static_stride = tile_shape_kn[1]
            my_smem_layout_b = turing_my_iters.MyTensorOpLayout(
                dtype_b, self.tmap_b.warp_shape, tile_shape_kn[1], 0,
                seq(num_stage, 1), True)
        else:  # permute k
            smem_layout_b.static_stride = num_stage * tile_shape_kn[0]
            my_smem_layout_b = turing_my_iters.MyTensorOpLayout(
                dtype_b, self.tmap_b.warp_shape, tile_shape_kn[0], 1,
                seq(1, num_stage), False)

        if trans_a:
            self._smem_iter_a = turing_my_iters.SmemTileIterator(
                is_crosswise_a,
                dtype_a,
                tile_shape_km,
                my_smem_layout_a,
                smem_layout_a,
                self.tmap_a,
                num_stage=num_stage)
            self._warp_iter_a = turing_my_iters.WarpIteratorCongruous(
                dtype_a, tile_shape_km, my_smem_layout_a, smem_layout_a,
                warp_tile_shape_km, True, inst_shape_km, 1, self.partk)
        else:
            self._smem_iter_a = turing_my_iters.SmemTileIterator(
                is_crosswise_a,
                dtype_a,
                tile_shape_km,
                my_smem_layout_a,
                smem_layout_a,
                self.tmap_a,
                crosswise=tile_shape[2],
                num_stage=num_stage)
            self._warp_iter_a = turing_my_iters.WarpIteratorCrosswise(
                dtype_a, tile_shape_km, my_smem_layout_a, smem_layout_a,
                warp_tile_shape_km, True, inst_shape_km, 1, self.partk)

        if trans_b:
            self._smem_iter_b = turing_my_iters.SmemTileIterator(
                is_crosswise_b,
                dtype_b,
                tile_shape_kn,
                my_smem_layout_b,
                smem_layout_b,
                self.tmap_b,
                crosswise=tile_shape[2],
                num_stage=num_stage)
            self._warp_iter_b = turing_my_iters.WarpIteratorCrosswise(
                dtype_b, tile_shape_kn, my_smem_layout_b, smem_layout_b,
                warp_tile_shape_kn, False, inst_shape_kn, 1, self.partk)
        else:
            self._smem_iter_b = turing_my_iters.SmemTileIterator(
                is_crosswise_b,
                dtype_b,
                tile_shape_kn,
                my_smem_layout_b,
                smem_layout_b,
                self.tmap_b,
                num_stage=num_stage)
            self._warp_iter_b = turing_my_iters.WarpIteratorCongruous(
                dtype_b, tile_shape_kn, my_smem_layout_b, smem_layout_b,
                warp_tile_shape_kn, False, inst_shape_kn, 1, self.partk)
        # print(self.warp_iter_b)
        # raise NotImplementedError
        # turing only support tnt mma.sync layout
        self._warp_mma = wmma.turing.WarpMmaTuring(
            (warp_tile_shape[0], warp_tile_shape[1], tensorop.shape[2]),
            tensorop.shape, dtype_a, dtype_b, dtype_acc, False, True, False,
            tensorop.dtype_a, tensorop.dtype_b, tensorop.dtype_c)

    @property
    def input_spec(self) -> bases.Input:
        return self._input_spec

    @property
    def padding_mn(self):
        return seq(0, 0)

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


class OutputTuring(bases.Output):
    def __init__(
            self,
            mma_spec: bases.Mma,
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
        self._mma_spec = mma_spec
        output_op_count = constants.OPTIM_ACCESS_BITS // dtype_c.bitsize()
        # TODO support more mixed tile shape for int8
        if dtype_c == dtypes.int8 and dtype_acc == dtypes.int32 and tile_shape[:2] == seq(
                128, 128) and warp_tile_shape[:2] == seq(64, 64):
            output_op_count = 8
        if dtype_c == dtypes.int8 and dtype_acc == dtypes.int32 and tile_shape[:2] == seq(
                128, 64) and warp_tile_shape[:2] == seq(64, 32):
            output_op_count = 8
        if dtype_c == dtypes.int8 and dtype_acc == dtypes.int32 and tile_shape[:2] == seq(
                64, 64) and warp_tile_shape[:2] == seq(32, 32):
            output_op_count = 8

        # 32 * 16 * (128 * 4096 + 256 * 4096) * 2
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()

        # print(self.out_smem_padding)
        tensorop_rows = 8
        # here no different between cluster = self.warp_count_shape[0] and
        # group = self.warp_count_shape[0]
        self.part_shape = seq(1, 1, self.warp_count_shape[0], tensorop_rows,
                              tile_shape[1])
        remain = thread_map.warp_partition_remain(self.part_shape[1:4],
                                                  self.warp_count)
        output_op_count = min(
            output_op_count,
            self.part_shape[3:].prod() // remain[2] // constants.WARP_SIZE)
        self.output_op_count = output_op_count
        self.acc_frag_iter = turing_out_iters.OutFragIterTensorOp(
            dtype_acc, warp_tile_shape, seq(*tensorop.shape))
        mixed_enable = dtype_c.itemsize() < dtype_acc.itemsize()
        mixed_enable_dtypes = (dtype_c == dtypes.float16
                               and dtype_acc == dtypes.float32)
        mixed_enable_dtypes |= (dtype_c == dtypes.int8
                                and dtype_acc == dtypes.int32
                                and tile_shape[:2] == seq(128, 128)
                                and warp_tile_shape[:2] == seq(64, 64))
        mixed_enable_dtypes |= (dtype_c == dtypes.int8
                                and dtype_acc == dtypes.int32
                                and tile_shape[:2] == seq(128, 64)
                                and warp_tile_shape[:2] == seq(64, 32))
        mixed_enable &= mixed_enable_dtypes
        # mixed_enable = False
        if mixed_enable:
            self.out_warp_tile_iter = turing_out_iters.OutWarpTileIteratorTensorOpMixed(
                dtype_acc, tile_shape, warp_tile_shape, seq(*tensorop.shape),
                output_op_count, 8)
        else:
            self.out_warp_tile_iter = turing_out_iters.OutWarpTileIteratorTensorOp(
                dtype_acc, tile_shape, warp_tile_shape, seq(*tensorop.shape))
        self._out_smem_padding = self.out_warp_tile_iter.padding
        # TODO handle constant 16
        self.part_dilation = seq(warp_tile_shape[0] // tensorop_rows, 1, 1,
                                 warp_tile_shape[0] // tensorop_rows, 1)
        self.smem_part_dilation = seq(1, 1, 1, 1, 1)
        self._num_out_iters = warp_tile_shape[0] // tensorop_rows

        self.out_tmap = thread_map.Out5DLinear(dtype_acc, self.part_shape,
                                               self.part_dilation,
                                               self.warp_count,
                                               output_op_count)
        self.out_smem_tmap = thread_map.Out5DLinear(dtype_acc, self.part_shape,
                                                    self.smem_part_dilation,
                                                    self.warp_count,
                                                    output_op_count)
        # print(self.part_shape, self.part_dilation, self.out_tmap.iterations, self.out_tmap.delta)
        # raise NotImplementedError

        self.shared_mem_alignment = dtype_acc.bitsize(
        ) * self.out_tmap.element_per_acc // 8
        if mixed_enable:
            self.out_smem_loader = turing_out_iters.OutSmemLoaderMixed(
                dtype_acc, tile_shape, self.out_smem_tmap, output_op_count,
                tile_shape[1] + self._out_smem_padding[1],
                self.shared_mem_alignment, 8)
        else:
            self.out_smem_loader = out_iters.OutSmemLoader(
                dtype_acc, self.out_smem_tmap, output_op_count,
                tile_shape[1] + self._out_smem_padding[1],
                self.shared_mem_alignment)
        # do = f32, dacc = f32, epa=4
        # do = f16, dacc = f32, epa=8
        # do = s8, dacc = s32, epa=16
        # do = s8, dacc = s32, epa=8

        self._frag_per_iter = 1
        if dtype_acc == dtypes.float32 and dtype_c == dtypes.float16 and mixed_enable and self.mma_spec.partk == 1:
            self._frag_per_iter = 2
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
        return self._out_smem_padding

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


class AlgoSpecificTuring(object):
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
            tensorop: TensorOp,
            algo: GemmAlgo = GemmAlgo.Turing,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle):
        assert algo == GemmAlgo.Turing or algo == GemmAlgo.Ampere
        self.input_spec = InputTuring(tile_shape, warp_tile_shape, dtype_a,
                                      dtype_b, trans_a, trans_b, algo,
                                      shuffle_stride)
        self.mma_spec = MmaTuring(self.input_spec, tile_shape, warp_tile_shape,
                                  num_stage, dtype_a, dtype_b, dtype_acc,
                                  trans_a, trans_b, tensorop, algo)
        self.output_spec = OutputTuring(self.mma_spec, tile_shape,
                                        warp_tile_shape, num_stage, dtype_c,
                                        dtype_acc, dtype_comp, trans_c,
                                        tensorop, algo, shuffle_stride)
