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

from typing import Optional

from cumm import dtypes
from cumm.conv import input_iters, sparse_iters
from cumm.conv.bases import (LAYOUT_TYPES, ConvInputIterator, ConvIterAlgo,
                             ConvLayout, ConvLayoutType, ConvOpType,
                             ConvTensor)
from cumm.conv.params import ConvProblem
from cumm.gemm import constants, layout, mask_iters, thread_map
from cumm.gemm.algospec import bases
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType, TensorOp
from cumm.gemm.algospec.turing import MmaTuring, OutputTuring
from cumm.gemm.core import MetaArray, metaseq, seq


class InputTuring(bases.Input):

    def __init__(self,
                 problem: ConvProblem,
                 iter_algo: ConvIterAlgo,
                 tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int],
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 algo: GemmAlgo = GemmAlgo.Turing,
                 mask_sparse: bool = False,
                 increment_k_first: bool = False,
                 access_per_vector: int = 1):
        ndim = problem.ndim
        self.problem = problem
        self.iter_algo = iter_algo
        self.access_per_vector = access_per_vector
        trans_a, trans_b, _ = problem.get_gemm_trans_abc()
        self._trans_a = trans_a
        self._trans_b = trans_b
        self._layout_a, self._layout_b = problem.get_a_b_layout_class()
        self.input_trans_load_a = False
        self.input_trans_load_b = False
        self.input_last_residual = True
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

        access_size_bits = 128
        self.alignment_a = access_size_bits // dtype_a.bitsize()
        self.alignment_b = access_size_bits // dtype_b.bitsize()
        self.input_sub_tile_shape_a = seq(1, self.alignment_a)
        self.input_sub_tile_shape_b = seq(1, self.alignment_b)
        access_size_bits = 128
        if not trans_a:
            warp_shape_raked_a = seq(
                1, tile_shape[2] // (access_size_bits // dtype_a.bitsize()))
            warp_shape_raked_a[
                0] = constants.WARP_SIZE // warp_shape_raked_a[1]
        else:
            warp_shape_raked_a = seq(4, 8)
            if warp_shape_raked_a[1] * (access_size_bits // dtype_a.bitsize()
                                        ) > tile_shape[0]:  # for low channel m
                warp_shape_raked_a[1] = tile_shape[0] // (access_size_bits //
                                                          dtype_a.bitsize())
                warp_shape_raked_a[
                    0] = constants.WARP_SIZE // warp_shape_raked_a[1]
                assert (warp_shape_raked_a.prod() == constants.WARP_SIZE)

        if trans_b:
            warp_shape_raked_b = seq(
                1, tile_shape[2] // (access_size_bits // dtype_b.bitsize()))
            warp_shape_raked_b[
                0] = constants.WARP_SIZE // warp_shape_raked_b[1]
        else:
            warp_shape_raked_b = seq(4, 8)
            if warp_shape_raked_b[1] * (access_size_bits // dtype_b.bitsize()
                                        ) > tile_shape[1]:  #for low chanel n
                warp_shape_raked_b[1] = tile_shape[1] // (access_size_bits //
                                                          dtype_b.bitsize())
                warp_shape_raked_b[
                    0] = constants.WARP_SIZE // warp_shape_raked_b[1]
                assert (warp_shape_raked_b.prod() == constants.WARP_SIZE)
        self.tmap_a = thread_map.PitchLinearWarpRaked(
            self.input_tile_shape_a, self.input_sub_tile_shape_a,
            warp_shape_raked_a, self.num_threads)
        self.tmap_b = thread_map.PitchLinearWarpRaked(
            self.input_tile_shape_b, self.input_sub_tile_shape_b,
            warp_shape_raked_b, self.num_threads)
        self.padding_mn = seq(0, 0)
        if self.problem.op_type == ConvOpType.kForward or self.problem.op_type == ConvOpType.kBackwardInput:
            if mask_sparse:
                if self.access_per_vector == 0:
                    inp_iter_cls = sparse_iters.ForwardDgradSparseIOIteratorV2Mask
                    w_iter_cls = input_iters.WeightIteratorDP4AV2Mask
                else:
                    inp_iter_cls = sparse_iters.ForwardDgradSparseIOIterator
                    w_iter_cls = input_iters.WeightIteratorDP4A
                self.inp_iter_a = inp_iter_cls(
                    dtype_a,
                    problem.op_type,
                    tile_shape,
                    self.input_sub_tile_shape_a,
                    self.tmap_a,
                    self.problem,
                    increment_k_first,
                    access_per_vector=access_per_vector)
            else:
                assert self.access_per_vector != 0
                inp_iter_cls = input_iters.ForwardDgradIOIterator
                w_iter_cls = input_iters.WeightIteratorDP4A
                self.inp_iter_a = inp_iter_cls(
                    dtype_a,
                    problem.op_type,
                    tile_shape,
                    self.input_sub_tile_shape_a,
                    self.tmap_a,
                    self.problem,
                    self.layout_a,
                    optimized=iter_algo == ConvIterAlgo.Optimized)

            self.inp_iter_b = w_iter_cls(
                dtype_b,
                problem.op_type,
                tile_shape,
                self.input_sub_tile_shape_b,
                self.tmap_b,
                self.problem,
                self.layout_b,
                optimized=iter_algo == ConvIterAlgo.Optimized,
                increment_k_first=increment_k_first,
                access_per_vector=access_per_vector)
        else:
            if mask_sparse:
                if self.access_per_vector == 0:
                    inp_iter_cls = sparse_iters.ForwardDgradSparseIOIteratorV2Mask

                else:
                    inp_iter_cls = sparse_iters.ForwardDgradSparseIOIterator  # V2Mask

                self.inp_iter_a = inp_iter_cls(
                    dtype_a,
                    problem.op_type,
                    tile_shape,
                    self.input_sub_tile_shape_a,
                    self.tmap_a,
                    self.problem,
                    increment_k_first,
                    is_wgrad_out=True,
                    is_wgrad_input=False,
                    access_per_vector=access_per_vector)
                self.inp_iter_b = inp_iter_cls(
                    dtype_b,
                    problem.op_type,
                    tile_shape,
                    self.input_sub_tile_shape_b,
                    self.tmap_b,
                    self.problem,
                    increment_k_first,
                    is_wgrad_out=False,
                    is_wgrad_input=True,
                    access_per_vector=access_per_vector)

            else:
                self.inp_iter_a = input_iters.OutputNPQIterator(
                    dtype_a,
                    problem.op_type,
                    tile_shape,
                    self.input_sub_tile_shape_a,
                    self.tmap_a,
                    self.problem,
                    self.layout_a,
                    optimized=iter_algo == ConvIterAlgo.Optimized)

                self.inp_iter_b = input_iters.InputNPQIterator(
                    dtype_b,
                    problem.op_type,
                    tile_shape,
                    self.input_sub_tile_shape_b,
                    self.tmap_b,
                    self.problem,
                    self.layout_b,
                    optimized=iter_algo == ConvIterAlgo.Optimized)

    @property
    def layout_a(self) -> LAYOUT_TYPES:
        return self._layout_a

    @property
    def layout_b(self) -> LAYOUT_TYPES:
        return self._layout_b

    @property
    def input_iter_a(self) -> ConvInputIterator:
        return self.inp_iter_a

    @property
    def input_iter_b(self) -> ConvInputIterator:
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


class AlgoSpecificTuring(object):

    def __init__(self,
                 problem: ConvProblem,
                 tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int],
                 num_stage: int,
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_c: dtypes.DType,
                 dtype_acc: dtypes.DType,
                 dtype_comp: dtypes.DType,
                 iter_algo: ConvIterAlgo,
                 tensorop: Optional[TensorOp] = None,
                 algo: GemmAlgo = GemmAlgo.Turing,
                 mask_sparse: bool = False,
                 increment_k_first: bool = False,
                 access_per_vector: int = 1,
                 int8_inference: bool = False):
        assert algo == GemmAlgo.Turing or algo == GemmAlgo.Ampere
        trans_a, trans_b, trans_c = problem.get_gemm_trans_abc()
        self.input_spec = InputTuring(problem, iter_algo, tile_shape,
                                      warp_tile_shape, dtype_a, dtype_b, algo,
                                      mask_sparse, increment_k_first,
                                      access_per_vector)
        self.mma_spec = MmaTuring(self.input_spec, tile_shape, warp_tile_shape,
                                  num_stage, dtype_a, dtype_b, dtype_acc,
                                  trans_a, trans_b, tensorop, algo)
        shuffle_stride = ShuffleStrideType.NoShuffle
        if mask_sparse and not problem.op_type == ConvOpType.kBackwardWeight:
            shuffle_stride = ShuffleStrideType.ShuffleAC
        enable_both_io = int8_inference 
        # enable_both_io = False
        self.output_spec = OutputTuring(self.mma_spec,
                                        tile_shape,
                                        warp_tile_shape,
                                        num_stage,
                                        dtype_c,
                                        dtype_acc,
                                        dtype_comp,
                                        trans_c,
                                        tensorop,
                                        algo,
                                        shuffle_stride=shuffle_stride,
                                        access_per_vector=access_per_vector,
                                        int8_inference=int8_inference,
                                        with_source=enable_both_io)
