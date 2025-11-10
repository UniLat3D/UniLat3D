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

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import (GemmBasic, GemmBasicKernel, TensorView,
                         TensorViewKernel, TensorViewNVRTC,
                         TensorViewNVRTCKernel, GemmKernelFlags)
from cumm.constants import (CUTLASS_DEBUG, CUTLASS_INPUT_ITER, CUTLASS_MODE,
                            CUTLASS_OUTPUT_ITER, CUTLASS_SMEM_WARP_ITER)
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import (constants, layout, mask_iters, out_iters, thread_map,
                       volta_iters, volta_out_iters, wmma)
from cumm.gemm.algospec import GemmAlgo, TensorOp, bases, get_algo_spec
from cumm.gemm.algospec.core import ShuffleStrideType, get_min_arch_of_algo
from cumm.gemm.blockmma import BlockMmaStorage, Mma, MmaMultiStage
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.constants import NVRTCMode
from cumm.gemm.outputs import Output, OutputSmemStorage
from cumm.gemm.utils import GemmUtils, GemmUtilsCPU
from cumm.gemm.wmma.simt import WarpMmaSimt
from cumm.gemm.constants import NVRTCConstants
from cumm.gemm.bases import GemmComponentBase

def div_up(a, b):
    return (a + b - 1) // b


# class GemmNVRTCParams(pccm.Class):
#     """used for nvrtc kernels. all ptrs are void*, alpha and beta are float.
#     """
#     def __init__(self):
#         super().__init__()
#         self.add_member("m, n, k", "int")
#         self.add_member("ptr_A", f"const void*")
#         self.add_member("ptr_B", f"const void*")
#         self.add_member("ptr_C", f"void*")
#         self.add_member("ptr_D", f"const void*")

#         self.add_member("stride_A", f"int64_t")
#         self.add_member("stride_B", f"int64_t")
#         self.add_member("stride_C", f"int64_t")
#         self.add_member("stride_D", f"int64_t")

#         self.add_member("IndiceA", f"const int*")
#         self.add_member("IndiceBorC", f"const int*")

#         self.add_member("alpha, beta", f"float")
#         self.add_member("split_k_slices", f"int")
#         self.add_member("workspace", "void*")


class GemmParams(pccm.ParameterizedClass):
    def __init__(
            self,
            tile_shape: MetaArray[int],
            dtype_a: dtypes.DType,
            dtype_b: dtypes.DType,
            dtype_c: dtypes.DType,
            dtype_comp: dtypes.DType,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            itera_params: mask_iters.MaskTileIteratorParams,
            iterb_params: mask_iters.MaskTileIteratorParams,
            out_params: out_iters.OutIteratorParams,
            have_workspace: bool = False,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            split_d_params: bool = False):
        super().__init__()

        self.add_dependency(TensorViewNVRTC, GemmBasic, GemmUtilsCPU)
        self.add_param_class("gemmutils", GemmUtils(tile_shape), "GemmUtils")
        self.itera_params = itera_params
        self.iterb_params = iterb_params
        self.out_params = out_params
        self.shuffle_stride = shuffle_stride
        self.cutlass_a_type = ("Mma::IteratorA")
        self.cutlass_b_type = ("Mma::IteratorB")
        self.cutlass_a_param_type = self.cutlass_a_type + "::Params"
        self.cutlass_b_param_type = self.cutlass_b_type + "::Params"

        self.add_param_class("itera_p", itera_params, "IterAParams")
        self.add_param_class("iterb_p", iterb_params, "IterBParams")

        self.add_param_class("out_params_ns", out_params, "OutIterParams")

        self.tile_shape = tile_shape
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.dtype_comp = dtype_comp
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c

        self.add_member("gemm_k_size_per_split, m, n, k", "int")
        self.add_member("ptr_A", f"const {dtype_a}*")
        self.add_member("ptr_B", f"const {dtype_b}*")
        self.add_member("ptr_C", f"{dtype_c}*")
        self.add_member("ptr_D", f"const {dtype_c}*")

        self.add_member("stride_A", f"int64_t")
        self.add_member("stride_B", f"int64_t")
        self.add_member("stride_C", f"int64_t")
        self.add_member("stride_D", f"int64_t")

        self.add_member("alpha, beta, act_alpha, act_beta", f"{dtype_comp}")

        self.add_member("act_type", f"tv::gemm::Activation")

        self.add_member("grid_dims", f"dim3")
        self.have_workspace = have_workspace
        if have_workspace:
            self.add_member("workspace", "void*")
        if CUTLASS_INPUT_ITER:
            self.add_member("itera_params_", self.cutlass_a_param_type)
            self.add_member("iterb_params_", self.cutlass_b_param_type)
        else:
            self.add_member("itera_params_", f"IterAParams")
            self.add_member("iterb_params_", f"IterBParams")
        if CUTLASS_OUTPUT_ITER:
            self.add_member("params_C",
                            f"Epilogue::OutputTileIterator::Params")
            self.add_member("params_D",
                            f"Epilogue::OutputTileIterator::Params")
        else:
            # TODO reduce output register usage
            self.add_member("out_params_", f"OutIterParams")
            if split_d_params:
                self.add_member("out_params_scalebias_", f"OutIterParams")
        self.split_d_params = split_d_params
        # cudasim members
        self.m = 0
        self.n = 0
        self.k = 0
        self.gemm_k_size = 0
        self.ptr_A: Optional[ArrayPtr] = None
        self.ptr_B: Optional[ArrayPtr] = None
        self.ptr_C: Optional[ArrayPtr] = None
        self.ptr_D: Optional[ArrayPtr] = None
        self.alpha = 0
        self.beta = 0
        self.grid_dims = cudasim.Dim3(0, 0, 0)
        self.itera_params_: Optional[mask_iters.MaskTileIteratorParams] = None
        self.iterb_params_: Optional[mask_iters.MaskTileIteratorParams] = None
        self.out_params_: Optional[out_iters.OutIteratorParams] = None

    def python_ctor(self,
                    m: int,
                    n: int,
                    k: int,
                    A: ArrayPtr,
                    B: ArrayPtr,
                    C: ArrayPtr,
                    D: ArrayPtr,
                    alpha: float,
                    beta: float,
                    split_k_slices: int = 1):
        new_obj = GemmParams(self.tile_shape, self.dtype_a, self.dtype_b,
                             self.dtype_c, self.dtype_comp, self.trans_a,
                             self.trans_b, self.trans_c, self.itera_params,
                             self.iterb_params, self.out_params)
        new_obj.grid_dims.x = mask_iters.div_up(m, new_obj.tile_shape[0])
        new_obj.grid_dims.y = mask_iters.div_up(n, new_obj.tile_shape[1])
        new_obj.grid_dims.z = split_k_slices

        total_gemm_k_iterations = mask_iters.div_up(k, new_obj.tile_shape[2])
        gemm_k_iterations = mask_iters.div_up(total_gemm_k_iterations,
                                              new_obj.grid_dims.z)
        new_obj.gemm_k_size = gemm_k_iterations * new_obj.tile_shape[2]

        new_obj.ptr_A = A
        new_obj.ptr_B = B
        new_obj.ptr_C = C
        new_obj.ptr_D = D
        new_obj.m = m
        new_obj.n = n
        new_obj.k = k
        new_obj.alpha = alpha
        new_obj.beta = beta
        a_stride = k
        if self.trans_a:
            a_stride = m
        b_stride = n
        if self.trans_b:
            b_stride = k

        new_obj.itera_params_ = self.itera_params.python_ctor(a_stride)
        new_obj.iterb_params_ = self.iterb_params.python_ctor(b_stride)
        new_obj.out_params_ = self.out_params.python_ctor(n)
        return new_obj

    @pccm.cuda.constructor(host=True, device=True, inline=True)
    def ctor(self):
        code = pccm.code()
        code.raw(f"""

        """)
        if CUTLASS_MODE:
            code.raw(f"""
            CutlassGemm::ThreadblockSwizzle threadblock_swizzle;

            cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
                {{m, n, k}}, 
                {{{self.tile_shape[0]}, {self.tile_shape[1]}, {self.tile_shape[2]}}},
                1);
            grid_dims.x = grid_shape.m();
            grid_dims.y = grid_shape.n();
            grid_dims.z = grid_shape.k();
            """)
        else:
            code.raw(f"""
            auto grid_dims_arr = GemmUtilsCPU::get_logical_tile_count(m, n, k, {self.tile_shape[0]}, {self.tile_shape[1]}, split_k_slices);
            // tv::printf2_once(m, n, k, {self.tile_shape[0]}, {self.tile_shape[1]}, split_k_slices, grid_dims_arr[0], grid_dims_arr[1], grid_dims_arr[2]);
            grid_dims.x = grid_dims_arr[0];
            grid_dims.y = grid_dims_arr[1];
            grid_dims.z = grid_dims_arr[2];
            """)
        code.raw(f"""
        // int total_gemm_k_iterations = tv::div_up(k, {self.tile_shape[2]}); // 160, 16 = 10
        // int gemm_k_iterations_per_split =
        //     tv::div_up(total_gemm_k_iterations, int(grid_dims.z)); // 10, 4 = 3
        // gemm_k_size_per_split = gemm_k_iterations_per_split * {self.tile_shape[2]}; // 3 * 16 = 48, 0-48, 48-96, 96-144, 144-192
        gemm_k_size_per_split = GemmUtils::get_gemm_k_size_per_split(k, split_k_slices);

        // tv::ssprint("gemm_k_size_per_split", m, n, k, gemm_k_size_per_split, grid_dims.x, grid_dims.y, grid_dims.z);
        """)
        # if CUTLASS_INPUT_ITER:
        #     code.raw(f"""
        #     cutlass::layout::RowMajor layoutA({pccm.boolean(self.trans_a)} ? m : k);
        #     cutlass::layout::RowMajor layoutB({pccm.boolean(self.trans_b)} ? k : n);

        #     itera_params_ = {self.cutlass_a_param_type}(layoutA);
        #     iterb_params_ = {self.cutlass_b_param_type}(layoutB);
        #     """)
        # else:
        #     if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
        #         code.raw(f"""
        #         itera_params_ = IterAParams({pccm.boolean(self.trans_a)} ? m : k, IndiceA);
        #         iterb_params_ = IterBParams({pccm.boolean(self.trans_b)} ? k : n);
        #         """)
        #     elif self.shuffle_stride == ShuffleStrideType.ShuffleAB:
        #         code.raw(f"""
        #         itera_params_ = IterAParams({pccm.boolean(self.trans_a)} ? m : k, IndiceA);
        #         iterb_params_ = IterBParams({pccm.boolean(self.trans_b)} ? k : n, IndiceB);
        #         """)
        #     else:
        #         code.raw(f"""
        #         itera_params_ = IterAParams({pccm.boolean(self.trans_a)} ? m : k);
        #         iterb_params_ = IterBParams({pccm.boolean(self.trans_b)} ? k : n);
        #         """)

        # if CUTLASS_OUTPUT_ITER:
        #     code.raw(f"""
        #     cutlass::layout::RowMajor layoutC(n);
        #     cutlass::layout::RowMajor layoutD(n);
        #     params_C = Epilogue::OutputTileIterator::Params(layoutC);
        #     params_D = Epilogue::OutputTileIterator::Params(layoutD);
        #     """)
        # else:
        #     if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
        #         code.raw("out_params_ = OutIterParams(n, IndiceC);")
        #     else:
        #         code.raw("out_params_ = OutIterParams(n);")
        if CUTLASS_INPUT_ITER:
            code.raw(f"""
            cutlass::layout::RowMajor layoutA(stride_A);
            cutlass::layout::RowMajor layoutB(stride_B);

            itera_params_ = {self.cutlass_a_param_type}(layoutA);
            iterb_params_ = {self.cutlass_b_param_type}(layoutB);
            """)
        else:
            if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
                code.raw(f"""
                itera_params_ = IterAParams(stride_A, IndiceA);
                iterb_params_ = IterBParams(stride_B);
                """)
            elif self.shuffle_stride == ShuffleStrideType.ShuffleAB:
                code.raw(f"""
                itera_params_ = IterAParams(stride_A, IndiceA);
                iterb_params_ = IterBParams(stride_B, IndiceB);
                """)
            else:
                code.raw(f"""
                itera_params_ = IterAParams(stride_A);
                iterb_params_ = IterBParams(stride_B);
                """)

        if CUTLASS_OUTPUT_ITER:
            code.raw(f"""
            cutlass::layout::RowMajor layoutC(stride_C);
            cutlass::layout::RowMajor layoutD(stride_D);
            params_C = Epilogue::OutputTileIterator::Params(layoutC);
            params_D = Epilogue::OutputTileIterator::Params(layoutD);
            """)
        else:
            # TODO find a way to specify D strides for bias Add
            if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
                code.raw("out_params_ = OutIterParams(stride_C, IndiceC);")
            else:
                code.raw("out_params_ = OutIterParams(stride_C);")
            if self.split_d_params:
                # D don't support 
                if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
                    code.raw("out_params_scalebias_ = OutIterParams(stride_D, IndiceD);")
                else:
                    code.raw("out_params_scalebias_ = OutIterParams(stride_D);")

            
        code.arg("m, n, k", "int")
        code.arg("A", f"const {self.dtype_a}*")
        code.arg("B", f"const {self.dtype_b}*")
        code.arg("C", f"{self.dtype_c}*")
        code.arg("D", f"const {self.dtype_c}*")
        code.arg("stride_A, stride_B, stride_C, stride_D", f"int64_t")

        if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
            code.arg("IndiceA", f"const int*")
            code.arg("IndiceC", f"const int*")
            code.arg("IndiceD", f"const int*")
        elif self.shuffle_stride == ShuffleStrideType.ShuffleAB:
            code.arg("IndiceA", f"const int*")
            code.arg("IndiceB", f"const int*")

        code.arg("alpha", f"{self.dtype_comp}", f"{self.dtype_comp}(1)")
        code.arg("beta", f"{self.dtype_comp}", f"{self.dtype_comp}(0)")
        code.arg("act_alpha", f"{self.dtype_comp}", f"{self.dtype_comp}(0)")
        code.arg("act_beta", f"{self.dtype_comp}", f"{self.dtype_comp}(0)")
        code.arg("act_type", f"tv::gemm::Activation", f"tv::gemm::Activation::kNone")

        code.arg("split_k_slices", "int", "1")
        if self.have_workspace:
            code.arg("workspace", "void*", "nullptr")
        code.ctor_init("m", "m")
        code.ctor_init("n", "n")
        code.ctor_init("k", "k")
        code.ctor_init("ptr_A", "A")
        code.ctor_init("ptr_B", "B")
        code.ctor_init("ptr_C", "C")
        code.ctor_init("ptr_D", "D")
        code.ctor_init("stride_A", "stride_A")
        code.ctor_init("stride_B", "stride_B")
        code.ctor_init("stride_C", "stride_C")
        code.ctor_init("stride_D", "stride_D")

        code.ctor_init("alpha", "alpha")
        code.ctor_init("beta", "beta")
        code.ctor_init("act_alpha", "act_alpha")
        code.ctor_init("act_beta", "act_beta")
        code.ctor_init("act_type", "act_type")

        if self.have_workspace:
            code.ctor_init("workspace", "workspace")
        return code


class GemmKernel(GemmComponentBase):
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
            splitk_serial: bool = False,
            splitk_parallel: bool = False,
            need_source: bool = True,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            access_per_vector: int = 1,
            nvrtc_mode: NVRTCMode = NVRTCMode.Disabled,
            async_kernel: bool = False,
            split_d_params: bool = True,
            nvrtc_compile: bool = False):
        """
        splitK and sliceK:
        https://github.com/NVIDIA/cutlass/issues/211#issuecomment-801992218
        split K: multiple block in k axis
        slice K: multiple warp in k axis

        nvrtc_compile: only affact behavior in library compile.
        """
        super().__init__()
        self.add_dependency(TensorViewNVRTCKernel, layout.RowMajor,
                            layout.ColumnMajor, GemmBasicKernel, GemmUtilsCPU)
        if nvrtc_mode == NVRTCMode.Disabled:
            self.add_dependency(GemmKernelFlags)
        self.add_param_class("gemmutils", GemmUtils(tile_shape), "GemmUtils")
        self.tile_shape = tile_shape
        self.shuffle_stride = shuffle_stride
        self.warp_tile_shape = warp_tile_shape
        self.num_stage = num_stage
        self.tensorop = tensorop
        self.nvrtc_compile = nvrtc_compile
        self.splitk_serial = splitk_serial
        self.splitk_parallel = splitk_parallel
        self.need_source = need_source
        self.access_per_vector = access_per_vector
        self.is_nvrtc = nvrtc_mode != NVRTCMode.Disabled
        self.nvrtc_mode = nvrtc_mode
        self.async_kernel = async_kernel
        self.split_d_params = split_d_params
        transpose_gemm = trans_c
        if transpose_gemm:
            self.dtype_a = dtype_b
            self.dtype_b = dtype_a
            trans_a = not trans_a
            trans_b = not trans_b
            tmp = trans_a
            trans_a = trans_b
            trans_b = tmp
            trans_c = not trans_c
        else:
            self.dtype_a = dtype_a
            self.dtype_b = dtype_b
        dtype_a = self.dtype_a
        dtype_b = self.dtype_b

        self.dtype_c = dtype_c
        self.dtype_acc = dtype_acc
        self.dtype_comp = dtype_comp
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.algo = algo
        algo_spec = get_algo_spec(self.algo)(tile_shape, warp_tile_shape,
                                             num_stage, dtype_a, dtype_b,
                                             dtype_c, dtype_acc, dtype_comp,
                                             trans_a, trans_b, trans_c,
                                             tensorop, algo, shuffle_stride)
        self.algo_spec = algo_spec
        self.input_spec = algo_spec.input_spec
        self.mma_spec = algo_spec.mma_spec
        self.output_spec = algo_spec.output_spec

        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE
        self.partk = self.warp_count_shape[2]
        self.add_param_class("inpitera", self.input_spec.input_iter_a,
                             "InputIteratorA")
        self.add_param_class("inpiterb", self.input_spec.input_iter_b,
                             "InputIteratorB")

        padding_mn = self.mma_spec.padding_mn

        self.acc_frag_iter = self.output_spec.acc_frag_iter
        self.out_warp_tile_iter = self.output_spec.out_warp_tile_iter
        out_smem_padding = self.output_spec.smem_padding
        self.fragment_c_t = array_type(
            dtype_acc, self.output_spec.get_accumulator_count())
        self.gemm_smem_storage = BlockMmaStorage(tile_shape,
                                                 seq(0, padding_mn[0]),
                                                 seq(0, padding_mn[1]),
                                                 num_stage, dtype_a, dtype_b)
        self.out_smem_storage = OutputSmemStorage(
            seq(
                tile_shape[0] // self.output_spec.num_out_iters *
                self.warp_count_shape[2], tile_shape[1]), out_smem_padding,
            dtype_acc, self.output_spec.frag_per_iter)
        # if partk > 1, we need more smem tile to save each k result.
        # self.frag_per_iter = self.output_spec.frag_per_iter
        # self.out_num_tile = self.output_spec.frag_per_iter if self.output_spec.frag_per_iter > 1 else self.partk
        # self.out_tile_size = self.out_smem_storage.smem_size // dtype_acc.itemsize() // self.out_num_tile
        if cudasim.enable_debug():
            print(self.out_smem_storage.smem_size,
                  self.gemm_smem_storage.smem_size)
        self.smem_size = max(self.out_smem_storage.smem_size,
                             self.gemm_smem_storage.smem_size)
        self.add_param_class("gemm_smem_storage", self.gemm_smem_storage,
                             "BlockMmaStorage")
        self.add_param_class("out_smem_storage", self.out_smem_storage,
                             "OutputSmemStorage")
        inp_iter_a_param = self.input_spec.input_iter_a.get_params()
        inp_iter_b_param = self.input_spec.input_iter_b.get_params()
        have_workspace = splitk_serial or splitk_parallel
        self.have_workspace = have_workspace
        self.gemm_params = GemmParams(tile_shape, dtype_a, dtype_b, dtype_c,
                                      dtype_comp, trans_a, trans_b, trans_c,
                                      inp_iter_a_param, inp_iter_b_param,
                                      self.output_spec.out_iter.get_params(),
                                      have_workspace, shuffle_stride,
                                      split_d_params)
        # self.add_dependency(GemmNVRTCParams)
        self.add_param_class("gemm_params", self.gemm_params, "GemmParams")
        if self.is_nvrtc:
            self.add_code_before_class(f"""
            __constant__ int {NVRTCConstants.SIZEOF_KEY} = sizeof(GemmParams);
            __constant__ int {NVRTCConstants.NUM_THREADS_KEY} = {self.num_threads};
            __constant__ int {NVRTCConstants.SMEM_KEY} = {self.smem_size};
            """)
        if self.nvrtc_mode == NVRTCMode.ConstantMemory:
            self.add_code_before_class(f"""
            __constant__ tv::array<uint8_t, sizeof(GemmParams)> {NVRTCConstants.CONSTANT_PARAM_KEY};
            """)

        self.cutlass_smem_a_type = ("Mma::SmemIteratorA")
        self.cutlass_smem_b_type = ("Mma::SmemIteratorB")
        self.cutlass_warp_a_type = "Mma::Operator::IteratorA"
        self.cutlass_warp_b_type = "Mma::Operator::IteratorB"

        if self.num_stage <= 2:
            self.mma_container = Mma(dtype_acc, self.partk, num_stage,
                                     self.mma_spec, self.gemm_smem_storage)
        else:
            self.mma_container = MmaMultiStage(dtype_acc, self.partk, num_stage,
                                     self.mma_spec, self.gemm_smem_storage)
        self.output = Output(dtype_acc, self.warp_count_shape, self.partk,
                             self.output_spec, self.out_smem_storage)
        self.add_param_class("out_iter", self.output_spec.out_iter, "OutIter")
        self.add_param_class("out_iter_const", self.output_spec.const_out_iter,
                             "ConstOutIter")

        self.add_param_class("out_op", self.output_spec.output_op, "OutputOp")

        self.add_param_class("mma", self.mma_container, "Mma")
        self.add_param_class("output", self.output, "Output")

    def min_arch(self) -> Tuple[int, int]:
        arch = self.mma_container.min_arch()
        if arch is None:
            return (0, 0)
        else:
            return arch

    def get_algo_name(self):
        res = f"{self.algo.value}_{self.dtype_a.shortcut()}{self.dtype_b.shortcut()}{self.dtype_c.shortcut()}"
        res += f"{self.dtype_acc.shortcut()}{self.dtype_comp.shortcut()}"
        las = "n" if self.trans_a else "t"
        lbs = "n" if self.trans_b else "t"
        lcs = "n" if self.trans_c else "t"
        res += f"{las}{lbs}{lcs}"
        res += f"_m{self.tile_shape[0]}n{self.tile_shape[1]}k{self.tile_shape[2]}"
        res += f"m{self.warp_tile_shape[0]}n{self.warp_tile_shape[1]}k{self.warp_tile_shape[2]}"
        res += f"A{self.access_per_vector}"
        if self.tensorop is not None:
            tes = self.tensorop.shape
            res += f"T{tes[0]}{tes[1]}{tes[2]}"
            if self.tensorop.top_dtypes is not None:
                t = self.tensorop
                res += f"{t.dtype_a.shortcut()}{t.dtype_b.shortcut()}{t.dtype_c.shortcut()}"
        res += f"_{self.num_stage}"
        if self.splitk_serial:
            res += "1"
        else:
            res += "0"
        if self.splitk_parallel:
            res += "1"
        else:
            res += "0"
        if self.shuffle_stride != ShuffleStrideType.NoShuffle:
            res += f"_S{self.shuffle_stride.value}"
        return res

    def support_splitk(self):
        return self.splitk_serial or self.splitk_parallel

    @pccm.cuda.cuda_global_function  # (inline=True)
    def gemm_kernel(self):
        code = pccm.code()
        if self.is_nvrtc:
            if self.nvrtc_mode != NVRTCMode.ConstantMemory:
                if self.nvrtc_mode == NVRTCMode.Direct:
                    code.arg("gemm_nvrtc_params", "tv::gemm::GemmNVRTCParams")
                    self._nvrtc_params_template(code, "gemm_nvrtc_params")
                else:
                    code.arg("params", "GemmParams")
            else:
                code.raw(f"""
                GemmParams params = *(reinterpret_cast<GemmParams*>(
                    {NVRTCConstants.CONSTANT_PARAM_KEY}.data()));
                """)
        else:
            code.arg("params", "GemmParams")
        min_arch = self.min_arch()
        arch_num = min_arch[0] * 100 + min_arch[1] * 10
        # use __CUDA_ARCH__ macro to reduce binary size
        # TODO this macro can't reduce compile time
        with code.macro_if_(
                f"defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= {arch_num})"):
            if CUTLASS_MODE:
                code.raw(f"""
                CutlassGemm::ThreadblockSwizzle threadblock_swizzle;
                extern __shared__ uint8_t SharedStorage[];
                typename CutlassGemm::GemmKernel::SharedStorage *shared_storage =
                    reinterpret_cast<typename CutlassGemm::GemmKernel::SharedStorage *>(SharedStorage);

                constexpr bool kSplitKSerial = {pccm.boolean(self.splitk_serial)};

                cutlass::gemm::GemmCoord threadblock_tile_offset =
                    threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

                // Early exit if CTA is out of range
                if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
                params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {{
                    return;
                }}
                // Compute initial location in logical coordinates
                cutlass::MatrixCoord tb_offset_A{{
                    threadblock_tile_offset.m() * {self.tile_shape[0]},
                    threadblock_tile_offset.k() * params.gemm_k_size,
                }};

                cutlass::MatrixCoord tb_offset_B{{
                    threadblock_tile_offset.k() * params.gemm_k_size,
                    threadblock_tile_offset.n() * {self.tile_shape[1]}
                }};
                int problem_size_k = min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
                int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + {self.tile_shape[2]} - 1) / {self.tile_shape[2]};

                """)
            else:
                code.raw(f"""
                // tv::printf2_once("?????222", params.grid_dims.x, params.grid_dims.y);

                constexpr bool kSplitKSerial = {pccm.boolean(self.splitk_serial)};
                extern __shared__ uint8_t SharedStorage[];
                auto gemm_shared_mem =
                    reinterpret_cast<BlockMmaStorage *>(SharedStorage);
                auto out_shared_mem =
                    reinterpret_cast<OutputSmemStorage *>(SharedStorage);

                int tile_offset_m = blockIdx.x;
                int tile_offset_n = blockIdx.y;
                int tile_offset_k = blockIdx.z;
                """)
                with code.if_(
                        "tile_offset_m >= params.grid_dims.x || tile_offset_n >= params.grid_dims.y"
                ):
                    code.raw(f"return;")
                code.raw(f"""
                tv::array<int, 2> block_offset_A{{tile_offset_m * {self.tile_shape[0]},
                                                tile_offset_k * params.gemm_k_size_per_split}};
                tv::array<int, 2> block_offset_B{{tile_offset_k * params.gemm_k_size_per_split,
                                                tile_offset_n * {self.tile_shape[1]}}};
                // Gemm::InputIteratorA::Params params_A(params.k);
                // Gemm::InputIteratorB::Params params_B(params.n);
                // refine gemm iteration for split-k
                auto problem_size_k = GemmUtils::get_gemm_k_bound(params.k, params.gemm_k_size_per_split, tile_offset_k);
                auto gemm_k_iterations = GemmUtils::get_gemm_iterations(problem_size_k, params.gemm_k_size_per_split, tile_offset_k);
                // int problem_size_k = min(params.k, (tile_offset_k + 1) * params.gemm_k_size_per_split);
                // int gemm_k_iterations =
                //     tv::div_up(problem_size_k - block_offset_A[1], {self.tile_shape[2]});
                """)
            code.raw(f"""
            int thread_idx = threadIdx.x;
            """)
            if CUTLASS_INPUT_ITER:
                code.raw(f"""
                {self.gemm_params.cutlass_a_type} input_iter_A(
                    params.params_A,
                    params.ref_A.data(),
                    {{params.problem_size.m(), problem_size_k}},
                    thread_idx,
                    tb_offset_A);
                {self.gemm_params.cutlass_b_type} input_iter_B(
                    params.params_B,
                    params.ref_B.data(),
                    {{problem_size_k, params.problem_size.n()}},
                    thread_idx,
                    tb_offset_B);
                """)
            else:
                if self.trans_a:
                    a_extent = "tv::array<int, 2>{problem_size_k, params.m}"
                    a_offset = "tv::array<int, 2>{block_offset_A[1], block_offset_A[0]}"
                else:
                    a_extent = "tv::array<int, 2>{params.m, problem_size_k}"
                    a_offset = "block_offset_A"

                if not self.trans_b:
                    b_extent = "tv::array<int, 2>{problem_size_k, params.n}"
                    b_offset = "block_offset_B"
                else:
                    b_extent = "tv::array<int, 2>{params.n, problem_size_k}"
                    b_offset = "tv::array<int, 2>{block_offset_B[1], block_offset_B[0]}"

                code.raw(f"""
                InputIteratorA input_iter_A(
                    params.itera_params_, params.ptr_A,
                    {a_extent},
                    thread_idx,
                    {a_offset});
                InputIteratorB input_iter_B(
                    params.iterb_params_, params.ptr_B,
                    {b_extent},
                    thread_idx,
                    {b_offset});
                """)
            code.raw(f"""
            int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
            int lane_idx = threadIdx.x % 32;
            """)
            if CUTLASS_SMEM_WARP_ITER:
                if not CUTLASS_DEBUG:
                    # code.raw(f"""
                    # DebugMma mma(shared_storage->main_loop, thread_idx, warp_idx, lane_idx);

                    # {self.cutlass_warp_a_type} warp_iter_A(shared_storage->main_loop.operand_A_ref(), lane_idx);
                    # {self.cutlass_warp_b_type} warp_iter_B(shared_storage->main_loop.operand_B_ref(), lane_idx);
                    # {self.cutlass_smem_a_type} smem_iter_A(shared_storage->main_loop.operand_A_ref(), thread_idx);
                    # {self.cutlass_smem_b_type} smem_iter_B(shared_storage->main_loop.operand_B_ref(), thread_idx);
                    # int warp_mn =
                    #     warp_idx % ({self.warp_count_shape[0]} * {self.warp_count_shape[1]});
                    # int warp_idx_k =
                    #     warp_idx / ({self.warp_count_shape[0]} * {self.warp_count_shape[1]});
                    # int warp_m = warp_mn % {self.warp_count_shape[0]};
                    # int warp_n = warp_mn / {self.warp_count_shape[0]};

                    # warp_iter_A.add_tile_offset({{warp_m, {self.warp_gemm_iters} * warp_idx_k}});
                    # warp_iter_B.add_tile_offset({{{self.warp_gemm_iters} * warp_idx_k, warp_n}});
                    # """)
                    code.raw(f"""
                    DebugMma mma(shared_storage->main_loop, thread_idx, warp_idx, lane_idx);
                    auto& smem_iter_A = mma.smem_iter_A;
                    auto& smem_iter_B = mma.smem_iter_B;

                    auto& warp_iter_A = mma.warp_iter_A;
                    auto& warp_iter_B = mma.warp_iter_B;
                    Mma mma(shared_storage->main_loop, thread_idx, warp_idx, lane_idx);

                    """)

            else:
                code.raw(f"""
                int warp_mn =
                    warp_idx % ({self.warp_count_shape[0]} * {self.warp_count_shape[1]});
                int warp_idx_k =
                    warp_idx / ({self.warp_count_shape[0]} * {self.warp_count_shape[1]});
                int warp_m = warp_mn % {self.warp_count_shape[0]};
                int warp_n = warp_mn / {self.warp_count_shape[0]};
                Mma mma(gemm_shared_mem, thread_idx, warp_idx_k, warp_m, warp_n, lane_idx);
                """)
            code.raw(f"""
            {self.fragment_c_t} accumulators;
            accumulators.clear();
            """)
            with code.if_("!kSplitKSerial || gemm_k_iterations > 0"):
                if CUTLASS_DEBUG:
                    code.raw(f"""
                    mma(gemm_k_iterations, accumulators, input_iter_A, input_iter_B, accumulators);
                    """)
                else:
                    code.raw(f"""
                    mma(gemm_k_iterations, accumulators, input_iter_A, input_iter_B, accumulators);
                    """)
            if cudasim.enable_debug():
                code.raw(f"""
                tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(accumulators, "accumulator");
                // tv::print_fragment_once<int, 0, 16, {cudasim.debug_tx()}>(accumulators);

                """)
            # code.raw(f"""
            
            # tv::print_fragment_meta_once<float, 0>(accumulators, "accumulator");
            # """)
            if CUTLASS_OUTPUT_ITER:
                code.raw(f"""

                // // C = alpha * A@B + beta * D, D can be C
                Epilogue::OutputOp oop(params.output_op);
                cutlass::MatrixCoord threadblock_offset(
                    threadblock_tile_offset.m() * {self.tile_shape[0]},
                    threadblock_tile_offset.n() * {self.tile_shape[1]}
                );
                int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

                // Construct the semaphore.
                cutlass::Semaphore semaphore(params.semaphore + block_idx, thread_idx);

                typename Epilogue::OutputTileIterator iterator_C(
                    params.params_C,
                    params.ref_C.data(),
                    params.problem_size.mn(),
                    thread_idx,
                    threadblock_offset
                );

                // Tile iterator writing to destination tensor.
                typename Epilogue::OutputTileIterator iterator_D(
                    params.params_D,
                    params.ref_D.data(),
                    params.problem_size.mn(),
                    thread_idx,
                    threadblock_offset
                );

                Epilogue epilogue(
                    shared_storage->epilogue, 
                    thread_idx, 
                    warp_idx, 
                    lane_idx);

                
                epilogue(oop, iterator_D, accumulators, iterator_C); 

                """)

            else:
                code.raw(f"""
                // tv::printf2_once("HERE 0", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z);

                // // C = alpha * A@B + beta * D, D can be C
                OutputOp output_op(params.alpha, params.beta, params.act_alpha, params.act_beta, params.act_type);
                """)
                if self.splitk_serial:
                    code.raw(f"""
                    int block_idx = tile_offset_m + tile_offset_n * params.grid_dims.x;
                    tv::Semaphore semaphore(reinterpret_cast<int*>(params.workspace) + block_idx, thread_idx);
                    if (params.grid_dims.z > 1){{
                        semaphore.fetch();
                        output_op.set_k_partition(tile_offset_k, params.grid_dims.z);
                    }}
                    """)
                code.raw(f"""
                tv::array<int, 2> block_offset_C{{tile_offset_m * {self.tile_shape[0]},
                                                tile_offset_n * {self.tile_shape[1]}}};

                """)
                out_iter_both_io = self.output_spec.out_iter.support_both_source_and_target()
                if out_iter_both_io:
                    code.raw(f"""
                    OutIter out_iter_C(params.out_params_, params.ptr_C, params.ptr_D, {{params.m, params.n}},
                                            {{block_offset_C[0], block_offset_C[1]}},
                                            thread_idx);
                    """)
                else:
                    code.raw(f"""
                    OutIter out_iter_C(params.out_params_, params.ptr_C, {{params.m, params.n}},
                                            {{block_offset_C[0], block_offset_C[1]}},
                                            thread_idx);
                    """)
                if self.splitk_serial:
                    # we reuse iter_c for splitk_serial to save some time.
                    code.raw(f"""
                    bool need_self_reduce = false;
                    if (params.grid_dims.z > 1){{
                        if (tile_offset_k){{
                            need_self_reduce = true;
                        }}
                        semaphore.wait(tile_offset_k);
                        __threadfence();
                    }}
                    """)
                if self.need_source and not out_iter_both_io:
                    if self.split_d_params:
                        code.raw(f"""
                        ConstOutIter out_iter_D(params.out_params_scalebias_, params.ptr_D, {{params.m, params.n}},
                                            {{block_offset_C[0], block_offset_C[1]}},
                                            thread_idx);
                        """)
                    else:
                        code.raw(f"""
                        ConstOutIter out_iter_D(params.out_params_, params.ptr_D, {{params.m, params.n}},
                                            {{block_offset_C[0], block_offset_C[1]}},
                                            thread_idx);
                        """)
                code.raw(
                    f"Output out(out_shared_mem, thread_idx, warp_idx_k, warp_m, warp_n, lane_idx);"
                )
                if self.splitk_serial:
                    with code.if_("need_self_reduce"):
                        code.raw(
                            f"out.run_self_reduce(output_op, accumulators, out_iter_C);"
                        )
                    with code.else_():
                        if self.need_source:
                            if out_iter_both_io:
                                code.raw(f"out.run(output_op, accumulators, out_iter_C);")
                            else:
                                code.raw(f"out.run(output_op, accumulators, out_iter_C, out_iter_D);")
                        else:
                            code.raw(
                                f"out.run_no_source(output_op, accumulators, out_iter_C);"
                            )
                else:
                    if self.need_source:
                        if out_iter_both_io:
                            code.raw(
                                f"out.run(output_op, accumulators, out_iter_C);"
                            )
                        else:
                            code.raw(
                                f"out.run(output_op, accumulators, out_iter_C, out_iter_D);"
                            )
                    else:
                        code.raw(
                            f"out.run_no_source(output_op, accumulators, out_iter_C);")

                if self.splitk_serial:
                    code.raw(f"""
                    if (params.grid_dims.z > 1){{
                        int lock = 0;
                        if (params.grid_dims.z == tile_offset_k + 1) {{
                            // The final threadblock resets the semaphore for subsequent grids.
                            lock = 0;
                        }}
                        else {{
                            // Otherwise, the semaphore is incremented
                            lock = tile_offset_k + 1;
                        }}
                        __threadfence();
                        semaphore.release(lock);
                    }}
                    """)
        with code.macro_else_():
            code.raw(f"""
            tv::printf2_once("this arch isn't supported!");
            assert(0);
            """)
        code.macro_endif_()
        return code

    def _nvrtc_params_template(self,
                               code: pccm.FunctionCode,
                               param_name: str = "p"):
        p = param_name
        args = [
            f"{p}.m, {p}.n, {p}.k",
            f"reinterpret_cast<const {self.dtype_a}*>({p}.ptr_A)",
            f"reinterpret_cast<const {self.dtype_b}*>({p}.ptr_B)",
            f"reinterpret_cast<{self.dtype_c}*>({p}.ptr_C)",
            f"reinterpret_cast<const {self.dtype_c}*>({p}.ptr_D)",
            f"{p}.stride_A, {p}.stride_B, {p}.stride_C, {p}.stride_D", 
        ]
        if self.shuffle_stride != ShuffleStrideType.NoShuffle:
            if self.shuffle_stride == ShuffleStrideType.ShuffleAC:
                args.append(f"{p}.indiceA, {p}.indiceBorC, {p}.indiceD")
            else:
                args.append(f"{p}.indiceA, {p}.indiceBorC")

        args.extend([ 
            f"{self.dtype_comp}({p}.alpha), {self.dtype_comp}({p}.beta)",
            f"{self.dtype_comp}({p}.act_alpha), {self.dtype_comp}({p}.act_beta)",
            f"static_cast<tv::gemm::Activation>({p}.act_type)",
            f"{p}.split_k_slices"
        ])
        if self.have_workspace:
            args.append(f"{p}.workspace")
        code.raw(f"""

        GemmParams params({", ".join(args)});
        """)

    def nvrtc_kernel_template(self, dynamic_parallism: bool = True):
        """we must use dynamic parallism first to calculate
        some kernel meta data, then launch regular kernel.

        """
        code = pccm.code()
        code.arg("p", "tv::gemm::GemmNVRTCParams")
        if not dynamic_parallism:
            code.arg("out", "GemmParams*")
        else:
            code.arg("blocks", "dim3")
            code.arg("stream", "cudaStream_t")

        self._nvrtc_params_template(code)
        if dynamic_parallism:
            code.raw(f"""
            // tv::printf2_once(params.m, params.n, params.k, params.gemm_k_size_per_split);
            gemm_kernel<<<blocks, dim3({self.num_threads}), {self.smem_size}, stream>>>(params);
            """)
        else:
            # TODO Strange nvrtc bug
            code.raw(f"""
            auto out_ptr = reinterpret_cast<uint8_t*>(out);
            auto param_ptr = reinterpret_cast<const uint8_t*>(&params);
            for (auto i : tv::KernelLoopX<int>(sizeof(GemmParams))){{
                out_ptr[i] = param_ptr[i];
            }}
            // out[0] = params;
            """)

        return code

    @pccm.cuda.cuda_global_function
    def nvrtc_kernel(self):
        """we use dynamic parallism first to calculate
        some kernel meta data, then launch regular kernel.
        """
        res = self.nvrtc_kernel_template(True)
        if self.nvrtc_mode != NVRTCMode.DynamicParallism:
            return res.make_invalid()
        return res

    @pccm.cuda.cuda_global_function
    def nvrtc_kernel_cpu_out(self):
        """we use a kernel to calculate metadata, then copy to
        cpu, finally use that parameter to launch gemm kernel.
        """
        res = self.nvrtc_kernel_template(False)
        if self.nvrtc_mode != NVRTCMode.ConstantMemory and self.nvrtc_mode != NVRTCMode.KernelAndCPU:
            return res.make_invalid()
        return res

    # @lineprof.lineprof_wrapper
    async def gemm_kernel_python(self, params: GemmParams):
        smem = cudasim.get_smem()
        gemm_storage = self.gemm_smem_storage
        smem_A = smem[:gemm_storage.smem_size_a].view(
            dtypes.get_npdtype(self.dtype_a))
        assert smem_A.nbytes == gemm_storage.smem_size_a
        smem_B = smem[gemm_storage.smem_size_a:gemm_storage.smem_size].view(
            dtypes.get_npdtype(self.dtype_b))
        assert smem_B.nbytes == gemm_storage.smem_size_b
        out_storage = self.out_smem_storage
        smem_out = smem[:out_storage.smem_size].view(
            dtypes.get_npdtype(self.dtype_acc))
        if cudasim.enable_debug():
            smem_A_ptr = ArrayPtr(self.dtype_a.tv_dtype,
                                  smem_A.nbytes // self.dtype_a.itemsize(),
                                  external_data=tv.from_numpy(smem_A))
            smem_B_ptr = ArrayPtr(self.dtype_b.tv_dtype,
                                  smem_B.nbytes // self.dtype_b.itemsize(),
                                  external_data=tv.from_numpy(smem_B))
        else:
            smem_A_ptr = ArrayPtr(self.dtype_a.tv_dtype,
                                  smem_A.nbytes // self.dtype_a.itemsize(),
                                  external_data=tv.from_numpy(smem_A),
                                  meta_data=tv.Tensor())
            smem_B_ptr = ArrayPtr(self.dtype_b.tv_dtype,
                                  smem_B.nbytes // self.dtype_b.itemsize(),
                                  external_data=tv.from_numpy(smem_B),
                                  meta_data=tv.Tensor())

        thread_idx = cudasim.threadIdx().x

        # share smem metadata in block
        smem_A_ptr = await cudasim.block_broadcast(smem_A_ptr)
        smem_B_ptr = await cudasim.block_broadcast(smem_B_ptr)

        smem_out_ptr = ArrayPtr(self.dtype_acc.tv_dtype,
                                smem_out.nbytes // self.dtype_acc.itemsize(),
                                external_data=tv.from_numpy(smem_out))

        tile_offset_m = cudasim.blockIdx().x
        tile_offset_n = cudasim.blockIdx().y
        tile_offset_k = cudasim.blockIdx().z
        if (tile_offset_m >= params.grid_dims.x
                or tile_offset_n >= params.grid_dims.y):
            return

        block_offset_A = seq(tile_offset_m * self.tile_shape[0],
                             tile_offset_k * params.gemm_k_size)
        block_offset_B = seq(tile_offset_k * params.gemm_k_size,
                             tile_offset_n * self.tile_shape[1])
        block_offset_C = seq(tile_offset_m * self.tile_shape[0],
                             tile_offset_n * self.tile_shape[1])
        block_offset_D = seq(tile_offset_m * self.tile_shape[0],
                             tile_offset_n * self.tile_shape[1])
        problem_size_k = min(params.k,
                             (tile_offset_k + 1) * params.gemm_k_size)
        gemm_k_iterations = mask_iters.div_up(
            problem_size_k - block_offset_A[1], self.tile_shape[2])
        if self.trans_a:
            extent_A = seq(problem_size_k, params.m)
            tb_offset_A = seq(block_offset_A[1], block_offset_A[0])
        else:
            extent_A = seq(params.m, problem_size_k)
            tb_offset_A = seq(block_offset_A[0], block_offset_A[1])

        input_iter_A = self.input_spec.input_iter_a.python_ctor(
            params.itera_params_,
            params.ptr_A,
            extent_A,
            thread_idx,
            tb_offset_A,
            is_left=True)
        if self.trans_b:
            extent_B = seq(params.n, problem_size_k)
            tb_offset_B = seq(block_offset_B[1], block_offset_B[0])
        else:
            extent_B = seq(problem_size_k, params.n)
            tb_offset_B = seq(block_offset_B[0], block_offset_B[1])
        input_iter_B = self.input_spec.input_iter_b.python_ctor(
            params.iterb_params_,
            params.ptr_B,
            extent_B,
            thread_idx,
            tb_offset_B,
            is_left=False)

        warp_idx = cudasim.get_warp_id()
        lane_idx = thread_idx % 32
        await cudasim.syncthreads()
        warp_idx_k = warp_idx // (self.warp_count_shape[0] *
                                  self.warp_count_shape[1])
        warp_mn = warp_idx % (self.warp_count_shape[0] *
                              self.warp_count_shape[1])
        warp_m = warp_mn % self.warp_count_shape[0]
        warp_n = warp_mn // self.warp_count_shape[0]
        mma = await self.mma_container.python_ctor(smem_A_ptr, smem_B_ptr,
                                                   thread_idx, warp_idx_k,
                                                   warp_m, warp_n, lane_idx)
        accumulators = ArrayPtr(self.dtype_acc.tv_dtype,
                                self.mma_spec.accumulator_size)
        accumulators.clear()
        res_inputs = await mma(gemm_k_iterations, accumulators, input_iter_A,
                               input_iter_B, accumulators)

        await cudasim.syncthreads()
        # cudasim.debug_print(acc[:16])

        output_op = self.output_spec.output_op.python_ctor(
            params.alpha, params.beta)
        if self.splitk_serial:
            if cudasim.gridDim().z > 1:
                output_op.set_k_partition_python(tile_offset_k,
                                                 cudasim.gridDim().z)
        out_iter_C = self.output_spec.out_iter.python_ctor(
            params.out_params_, params.ptr_C, seq(params.m, params.n),
            seq(block_offset_C[0], block_offset_C[1]), thread_idx)
        out_iter_D = self.output_spec.out_iter.python_ctor(
            params.out_params_, params.ptr_D, seq(params.m, params.n),
            seq(block_offset_C[0], block_offset_C[1]), thread_idx)
        if self.splitk_serial and cudasim.gridDim().z > 1:
            if tile_offset_k > 0:
                out_iter_C = out_iter_D
        output = self.output.python_ctor(smem_out_ptr, thread_idx, warp_idx_k,
                                         warp_m, warp_n, lane_idx)
        need_self_reduce = False
        if self.splitk_serial:
            if cudasim.gridDim().z > 1:
                if tile_offset_k > 0:
                    need_self_reduce = True
        if self.splitk_serial:
            if need_self_reduce:
                res_output = await output(output_op,
                                          accumulators,
                                          out_iter_C,
                                          self_reduce=True)
            else:
                if self.need_source:
                    res_output = await output(output_op, accumulators,
                                              out_iter_C, out_iter_D)
                else:
                    res_output = await output(output_op, accumulators,
                                              out_iter_C)
        else:
            if self.need_source:
                res_output = await output(output_op, accumulators, out_iter_C,
                                          out_iter_D)
            else:
                res_output = await output(output_op, accumulators, out_iter_C)
        if not cudasim.enable_debug():
            return

        res = {
            **res_output,
            **res_inputs,
        }
        return res
