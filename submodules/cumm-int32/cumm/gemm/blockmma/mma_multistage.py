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

from ast import arg
import contextlib
from mimetypes import knownfiles
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
from cumm.gemm.arch.cpasync import CpAsyncCopy, CpAsyncGroup, AsyncCopyConfig
from cumm.gemm.bases import (GemmInputIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator, GemmSmemIterator,
                             GemmWarpIterator)
from cumm.gemm.mask_iters import MaskTileIterator
from cumm.gemm.turing_my_iters import SmemTileIterator
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.blockmma.mma import MaskIGemmIterator, div_up, BlockMmaStorage, MaskIGemmIteratorMaskLoaderDynamic
from enum import Enum
from cumm.conv.bases import ConvOpType


class SharedMemoryClearOption:
    kNone = 0
    kZfill = 1
    kClearLastStage = 2


class AsyncCopyIteration(pccm.ParameterizedClass):
    def __init__(self, InputIter: MaskTileIterator, SmemIter: SmemTileIterator, num_warp_mma_iters, nIdx=0) -> None:
        super().__init__()
        assert InputIter.can_multistage_load() and SmemIter.can_multistage_load()
        self.input_iter = InputIter
        self.smem_iter = SmemIter
        self.input_args = list(InputIter.enumurate_get_param(python=False))
        self.smem_args = list(SmemIter.enumurate_get_param(python=False))
        assert len(self.input_args) == len(self.smem_args)
        self.num_warp_mma_iters = num_warp_mma_iters
        self.nIdx = nIdx
        len_args = len(self.input_args)
        startIdx = len_args * nIdx // num_warp_mma_iters
        endIdx = len_args * (nIdx + 1) // num_warp_mma_iters
        self.used_input_args = self.input_args[startIdx: endIdx]
        self.used_smem_args = self.smem_args[startIdx: endIdx]
        self.used_input_args_python = list(InputIter.enumurate_get_param(python=True))[startIdx: endIdx]
        self.used_smem_args_python = list(SmemIter.enumurate_get_param(python=True))[startIdx: endIdx]
        
        self.cp_async_copy = CpAsyncCopy(SmemIter.tmap, SmemIter.dtype)
        self.add_param_class("cp_async_copy", self.cp_async_copy, "CpAsyncCp")
        assert len(self.used_input_args) == len(self.used_smem_args)

    async def do_copy_python(self, InputIter: MaskTileIterator, SmemIter: SmemTileIterator, cp_async_group):
        for iarg, sarg in zip(self.used_input_args_python, self.used_smem_args_python):
            input_pack = InputIter.load_ptr_with_param_python(*iarg)
            input_ptr, input_valid, input_addr = input_pack
            smem_pack = await SmemIter.store_ptr_with_param_python(*sarg)
            smem_ptr, smem_valid, smem_addr = smem_pack
            self.cp_async_copy.copy_python(smem_ptr, input_ptr, input_valid and smem_valid, cp_async_group)

    @pccm.cuda.static_function(device=True, forceinline=True)
    def do_copy(self):
        code = pccm.FunctionCode()
        code.targ("InputIter").targ("SmemIter")
        code.arg("input_iter", "InputIter&").arg("smem_iter", "SmemIter&")
        if len(list(zip(self.used_input_args, self.used_smem_args))) == 0:
            return code.raw("///// nothing to do here /////")                       # forbid warning in compile
        code.raw("""
            bool valid;
            const void* src_ptr;
            void* dest_ptr;
        """)
        for iarg, sarg in zip(self.used_input_args, self.used_smem_args):
            code.raw(f"""
                valid = true;
                src_ptr = input_iter.load_ptr_with_param({iarg}, valid);
                dest_ptr = smem_iter.store_ptr_with_param({sarg}, valid);
                                                
                CpAsyncCp::copy(dest_ptr, src_ptr, valid);
            """)
                    # we can not excape any of load:
                    # smem_iter valid is same in a warp, but we need input_iter itself to change its state;
                    # input_iter valid is diff through a warp.

                    # but input_iter will fill the valid,  smem_iter only make valid false(if true: no do)
        return code

    async def do_copy_zfill_python(self, InputIter: MaskTileIterator, SmemIter: SmemTileIterator, cp_async_group):
        for iarg, sarg in zip(self.used_input_args_python, self.used_smem_args_python):
            input_pack = InputIter.load_ptr_with_param_python(*iarg)
            input_ptr, input_valid, input_addr = input_pack
            smem_pack = await SmemIter.store_ptr_with_param_python(*sarg)
            smem_ptr, smem_valid, smem_addr = smem_pack
            self.cp_async_copy.copy_zfill_python(smem_ptr, input_ptr, input_valid and smem_valid, cp_async_group)

    @pccm.cuda.static_function(device=True, forceinline=True)
    def do_copy_zfill(self):
        code = pccm.FunctionCode()
        code.targ("InputIter").targ("SmemIter")
        code.arg("input_iter", "InputIter&").arg("smem_iter", "SmemIter&")
        if len(list(zip(self.used_input_args, self.used_smem_args))) == 0:
            return code.raw("///// nothing to do here /////")
        code.raw("""
            bool valid;
            const void* src_ptr;
            void* dest_ptr;
        """)
        for iarg, sarg in zip(self.used_input_args, self.used_smem_args):
            code.raw(f"""
                valid = true;
                src_ptr = input_iter.load_ptr_with_param({iarg}, valid);
                dest_ptr = smem_iter.store_ptr_with_param({sarg}, valid);
                                                
                CpAsyncCp::copy_zfill(dest_ptr, src_ptr, valid);
            """)
                    # we can not excape any of load:
                    # smem_iter valid is same in a warp, but we need input_iter itself to change some value
                    # input_iter valid is diff through a warp.
        return code


class MmaMultiStage(pccm.ParameterizedClass):
    """
        only make sense in Ampere arch.
        it requires cp.async.... to accelerate io mma.
    """
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
                 is_sparse_wgrad: bool = False,
                 smem_clear_opt = SharedMemoryClearOption.kNone,
                 op_type = ConvOpType.kForward,
                 dynamic_mask: bool = False):
        super().__init__()
        self.dynamic_mask = dynamic_mask
        assert smem_clear_opt in [SharedMemoryClearOption.kZfill, SharedMemoryClearOption.kNone], "Not Implemented"
        if mask_sparse:
            assert increment_k_first, "not impl"
            assert smem_clear_opt == SharedMemoryClearOption.kZfill, "I think it is"
        self.smem_clear_opt = smem_clear_opt
        self.dtype_acc = dtype_acc
        miter = MaskIGemmIterator(increment_k_first)
        self.add_param_class("mma_ns_miter", miter, "MaskIGemmIterator")
        self.miterd = MaskIGemmIteratorMaskLoaderDynamic(spec.input_spec.tile_shape, increment_k_first, op_type)
        self.add_param_class("mma_ns_miterD", self.miterd, "MaskIGemmIteratorDynamic")

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
        self.wmma = spec.warp_mma

        # test stage=2
        # self.num_stage = 2

        self.cpasync_group_ = CpAsyncGroup(self.num_stage)
        self.add_param_class("cpasync_group", self.cpasync_group_, "CpAsyncGroup")
        
        self.global_async_cp_a = AsyncCopyIteration(self.input_spec.input_iter_a, spec.smem_iter_a, 1)
        self.global_async_cp_b = AsyncCopyIteration(self.input_spec.input_iter_b, spec.smem_iter_b, 1)
        self.add_param_class("async_cp_iter_global_A", self.global_async_cp_a, "GlobalAsyncCopyIter_A")
        self.add_param_class("async_cp_iter_global_B", self.global_async_cp_b, "GlobalAsyncCopyIter_B")
        self.async_cp_a_type = []
        self.async_cp_b_type = []
        for iter_idx in range(self.spec.num_warp_mma_iters):
            async_cp_iter_a = AsyncCopyIteration(self.input_spec.input_iter_a, spec.smem_iter_a, self.spec.num_warp_mma_iters, iter_idx)
            async_cp_iter_b = AsyncCopyIteration(self.input_spec.input_iter_b, spec.smem_iter_b, self.spec.num_warp_mma_iters, iter_idx)
            self.add_param_class(f"async_cp_iter_{iter_idx}_A", async_cp_iter_a, f"AsyncCopyIter_{iter_idx}_A")
            self.add_param_class(f"async_cp_iter_{iter_idx}_B", async_cp_iter_b, f"AsyncCopyIter_{iter_idx}_B")
            self.async_cp_a_type.append(async_cp_iter_a)
            self.async_cp_b_type.append(async_cp_iter_b)        
        
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
        self.smem_iter_A: Optional[SmemTileIterator] = None
        self.smem_iter_B: Optional[SmemTileIterator] = None

        self.smem_A_ptr: Optional[ArrayPtr] = None
        self.smem_B_ptr: Optional[ArrayPtr] = None

        self.cpasync_group: Optional[CpAsyncGroup] = None

        ############ for debug ########
        #  not use sm80 async load but frag (for A or B)
        # self.add_code_before_class("#define DEBUG_MMA_MS_DOWNFALL_A")
        # self.add_code_before_class("#define DEBUG_MMA_MS_DOWNFALL_B")

        # not write smem!  (A or B)
        # self.add_code_before_class("#define DEBUG_MMA_MS_NOT_WRITE_SMEM_A")
        # self.add_code_before_class("#define DEBUG_MMA_MS_NOT_WRITE_SMEM_B")

        # not read input!  (A or B)
        # self.add_code_before_class("#define DEBUG_MMA_MS_NOT_READ_INPUT_A")
        # self.add_code_before_class("#define DEBUG_MMA_MS_NOT_READ_INPUT_B")

        # not inc iters(A and B)!
        # self.add_code_before_class("#define DEBUG_MMA_MA_NOT_INC_FILTER")
        # self.add_code_before_class("#define DEBUG_MMA_MA_NOT_UPDATE_INDICES_A")

        # ----------------------------optimize or not -------------------------------
        # self.add_code_before_class("#define MMA_MA_OPTIMIZE_LESS_LOAD")  # do it really optimized?  ## no

        
    def min_arch(self) -> Optional[Tuple[int, int]]:
        min_arch = (8, 0)
        wmma_arch = self.wmma.min_arch()
        if wmma_arch is not None:
            if wmma_arch < min_arch:
                return min_arch
            else:
                return wmma_arch
        else:
            return min_arch

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

    async def python_ctor(self, smem_A_ptr: ArrayPtr, smem_B_ptr: ArrayPtr,
                          thread_idx: int, warp_idx_k: int, warp_m: int,
                          warp_n: int, lane_idx: int):
        new_obj = MmaMultiStage(self.dtype_acc, self.partk, self.num_stage, self.spec,
                      self.smem_storage, self.first_input_clear,
                      self.clear_mask)
        new_obj.warp_iter_A = await self.spec.warp_iter_a.python_ctor(
            smem_A_ptr, warp_idx_k, warp_m, lane_idx)
        new_obj.warp_iter_B = await self.spec.warp_iter_b.python_ctor(
            smem_B_ptr, warp_idx_k, warp_n, lane_idx)
        new_obj.smem_iter_A = self.spec.smem_iter_a.python_ctor(
            self.smem_storage.smem_shape_a[1], smem_A_ptr, thread_idx)
        new_obj.smem_iter_B = self.spec.smem_iter_b.python_ctor(
            self.smem_storage.smem_shape_b[1], smem_B_ptr, thread_idx)
        new_obj.smem_A_ptr = smem_A_ptr
        new_obj.smem_B_ptr = smem_B_ptr
        
        new_obj.cpasync_group = self.cpasync_group_.python_ctor()

        return new_obj

    async def copy_tiles_and_advance_python(self, input_iter_A: MaskTileIterator,
                                            input_iter_B: MaskTileIterator,
                                            group_idx):
        assert group_idx < self.spec.num_warp_mma_iters
        await self.async_cp_a_type[group_idx].do_copy_python(input_iter_A, self.smem_iter_A, self.cpasync_group)
        await self.async_cp_b_type[group_idx].do_copy_python(input_iter_B, self.smem_iter_B, self.cpasync_group)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def copy_tiles_and_advance(self):
        code = pccm.FunctionCode()
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("group_idx", "const int &")
        with code.macro_if_("defined(DEBUG_MMA_MS_DOWNFALL_A) || defined(DEBUG_MMA_MS_DOWNFALL_B)"):
            code.raw(f"""
                if (group_idx == {self.spec.num_warp_mma_iters - 1}){{
#ifdef DEBUG_MMA_MS_DOWNFALL_A
                {self.input_spec.input_iter_a.fragment_t} input_frag_A;
#ifndef DEBUG_MMA_MS_NOT_READ_INPUT_A
                    input_iter_A.load(input_frag_A);
#endif
#ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_A
                    smem_iter_A.store(input_frag_A);
#endif
#endif
#ifdef DEBUG_MMA_MS_DOWNFALL_B
                {self.input_spec.input_iter_b.fragment_t} input_frag_B;
#ifndef DEBUG_MMA_MS_NOT_READ_INPUT_B
                    input_iter_B.load(input_frag_B);
#endif
#ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_B
                    smem_iter_B.store(input_frag_B);
#endif
#endif
                }}
            """)
        code.macro_endif_()

        for ind in range(self.spec.num_warp_mma_iters):
            if self.smem_clear_opt == SharedMemoryClearOption.kNone:
                code.raw(f"""
                    if(group_idx == {ind}){{
#if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                        AsyncCopyIter_{ind}_A::do_copy(input_iter_A, smem_iter_A);
#endif
#if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                        AsyncCopyIter_{ind}_B::do_copy(input_iter_B, smem_iter_B);
#endif
                        return;
                    }}
                """)
            else:
                code.raw(f"""
                    if(group_idx == {ind}){{
#if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                        AsyncCopyIter_{ind}_A::do_copy_zfill(input_iter_A, smem_iter_A);
#endif
#if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                        AsyncCopyIter_{ind}_B::do_copy_zfill(input_iter_B, smem_iter_B);
#endif
                        return;
                    }}
                """)
        return code

    def call_mask_dummy(self):
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        WarpMma warp_mma;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        {self.spec.warp_iter_a.fragment_t} warp_frag_A;
        {self.spec.warp_iter_b.fragment_t} warp_frag_B;
        while(!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        int smem_write_stage_idx = 0;
        while(!mask_iter.end){{
            input_iter_A.update_indices();
            for(int i=0; i<gemm_k_iterations; ++i){{
                input_iter_A.load(input_frag_A);
                input_iter_B.load(input_frag_B);
                smem_iter_A.store(input_frag_A);
                smem_iter_B.store(input_frag_B);
                __syncthreads();
                ++smem_iter_A;
                ++smem_iter_B;
                for (int j=0; j<{self.spec.num_warp_mma_iters}; ++j){{
                    warp_iter_A.load(warp_frag_A);
                    warp_iter_B.load(warp_frag_B);
                    ++warp_iter_A;
                    ++warp_iter_B;
                    warp_mma(accumulators, warp_frag_A, warp_frag_B, accumulators);
                }}
                if (smem_write_stage_idx == {self.num_stage - 1}){{
                    smem_iter_A.tile_increment({-self.num_stage});
                    smem_iter_B.tile_increment({-self.num_stage});
                    warp_iter_A.tile_increment({-self.num_stage * self.spec.num_warp_mma_iters});
                    warp_iter_B.tile_increment({-self.num_stage * self.spec.num_warp_mma_iters});
                    
                    smem_write_stage_idx = 0;
                }} else
                    ++smem_write_stage_idx;
                input_iter_A.increment_k();
                input_iter_B.increment_k();
            }}
            
            input_iter_A.reset_k();
            input_iter_B.reset_k();
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
            while(!mask_iter.valid() && !mask_iter.end){{
                ++mask_iter;
                input_iter_A.increment_filter();
                input_iter_B.increment_filter();
            }}
        }}
        """)
        return code
    
    def call_mask_sparse_wgrad(self):
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("reduced_mask_ptr", f"const uint32_t*")
        code.arg("num_reduced_mask", f"const int&")
        code.arg("tile_offset_k", f"const int&")
        code.arg("split_k_slices", f"const int&")

        code.arg("filter_offset", f"const int&")
        code.arg("mask_width", f"const int&")
        if self.dynamic_mask:
            code.arg("mask_int_count", f"const int&")

        code.raw(f"""
        int mask_width_rate = mask_width / {self.input_spec.tile_shape[2]};
        """)
        if self.dynamic_mask:
            code.raw(f"""
            uint32_t filter_offset_mask = 1u << (filter_offset % 32);
            int mask_offset = filter_offset / 32;
            """)
            mask_idx_str = "mask_idx * mask_int_count + mask_offset"
        else:
            code.raw(f"""
            uint32_t filter_offset_mask = 1u << filter_offset;
            """)
            mask_idx_str = "mask_idx"
        code.raw(f"""
        accumulators = src_accumulators;


        input_iter_B.increment_filter(filter_offset);
        int k_idx = tile_offset_k;
        int mask_idx = k_idx / mask_width_rate;
        uint32_t mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;

        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;

        TV_PRAGMA_UNROLL
        for (int stage=0; stage < {self.num_stage - 1}; ++stage, --gemm_k_iterations){{
            while (!(mask & filter_offset_mask) && (gemm_k_iterations > 0)){{
                k_idx += split_k_slices;
#ifndef MMA_MA_OPTIMIZE_LESS_LOAD
                mask_idx = k_idx / mask_width_rate;
                mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
#else
                if ((1 + mask_idx) * mask_width_rate <= k_idx){{
                    mask_idx = k_idx / mask_width_rate;
                    mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
                }}
#endif
                --gemm_k_iterations;
                input_iter_A.increment_no_clear_mask();
                input_iter_B.increment_no_clear_mask();
            }}
            input_iter_A.clear_mask_if_batch_unbound();
            input_iter_B.clear_mask_if_batch_unbound();
            input_iter_A.update_indices();
            input_iter_B.update_indices();
            GlobalAsyncCopyIter_A::do_copy_zfill(input_iter_A, smem_iter_A);
            GlobalAsyncCopyIter_B::do_copy_zfill(input_iter_B, smem_iter_B);
            CpAsyncGroup::make_fence();
            ++input_iter_A;
            ++input_iter_B;
            ++smem_iter_A;
            ++smem_iter_B;
            k_idx += split_k_slices;
#ifndef MMA_MA_OPTIMIZE_LESS_LOAD
            mask_idx = k_idx / mask_width_rate;
            mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
#else
            if ((1 + mask_idx) * mask_width_rate <= k_idx){{
                mask_idx = k_idx / mask_width_rate;
                mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
            }}
#endif
        }}
        int smem_write_stage_idx = {self.num_stage - 1};
        int smem_read_stage_idx = 0;

        while (!(mask & filter_offset_mask) && (gemm_k_iterations > 0)){{
            k_idx += split_k_slices;
#ifndef MMA_MA_OPTIMIZE_LESS_LOAD
            mask_idx = k_idx / mask_width_rate;
            mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
#else
            if ((1 + mask_idx) * mask_width_rate <= k_idx){{
                mask_idx = k_idx / mask_width_rate;
                mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
            }}
#endif
            --gemm_k_iterations;
            input_iter_A.increment_no_clear_mask();
            input_iter_B.increment_no_clear_mask();
        }}
        input_iter_A.clear_mask_if_batch_unbound();
        input_iter_B.clear_mask_if_batch_unbound();
        input_iter_A.update_indices();
        input_iter_B.update_indices();

        CpAsyncGroup::wait_final_group();
        __syncthreads();

        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);
        
        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);
        ++warp_iter_A;
        ++warp_iter_B;
        

        for (; gemm_k_iterations > {-self.num_stage + 1}; ){{
            TV_PRAGMA_UNROLL
            for (int warp_mma_k=0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                
                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                ++warp_iter_A;
                ++warp_iter_B;

                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                    warp_frag_B[warp_mma_k % 2], accumulators);

                if (warp_mma_k < {self.spec.num_warp_mma_iters - 1})
                    copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);

                if (warp_mma_k + 2 == {self.spec.num_warp_mma_iters}) {{
                    copy_tiles_and_advance(input_iter_A, input_iter_B, {self.spec.num_warp_mma_iters - 1});

                    CpAsyncGroup::make_fence();
                    ++smem_iter_A;
                    ++smem_iter_B;
                    ++input_iter_A;
                    ++input_iter_B;

                    CpAsyncGroup::wait_final_group();
                    __syncthreads();

                    //do some chores before async wait

                    k_idx += split_k_slices;
#ifndef MMA_MA_OPTIMIZE_LESS_LOAD
                    mask_idx = k_idx / mask_width_rate;
                    mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
#else
                    if ((1 + mask_idx) * mask_width_rate <= k_idx){{
                        mask_idx = k_idx / mask_width_rate;
                        mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
                    }}
#endif

                    --gemm_k_iterations;
                    while (!(mask & filter_offset_mask) && (gemm_k_iterations > 0)){{
                        k_idx += split_k_slices;
#ifndef MMA_MA_OPTIMIZE_LESS_LOAD
                        mask_idx = k_idx / mask_width_rate;
                        mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
#else
                        if ((1 + mask_idx) * mask_width_rate <= k_idx){{
                            mask_idx = k_idx / mask_width_rate;
                            mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
                        }}
#endif
                        --gemm_k_iterations;
                        input_iter_A.increment_no_clear_mask();
                        input_iter_B.increment_no_clear_mask();
                    }}
                    input_iter_A.clear_mask_if_batch_unbound();
                    input_iter_B.clear_mask_if_batch_unbound();
                    input_iter_A.update_indices();
                    input_iter_B.update_indices();

                    if (smem_write_stage_idx == {self.num_stage - 1}) {{
                        smem_iter_A.tile_increment(-{self.num_stage});
                        smem_iter_B.tile_increment(-{self.num_stage});
                        smem_write_stage_idx = 0;
                    }} else
                        ++smem_write_stage_idx;
                    
                    if (smem_read_stage_idx == {self.num_stage - 1}) {{
                        warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        smem_read_stage_idx = 0;
                    }} else
                        ++smem_read_stage_idx;

                }}
            }}
        }}
        """)
        return code

    def call_mask_sparse_increase_k_V2(self):
        code = pccm.code()
        code.arg("gemm_k_iterations", f"const int&")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        if self.dynamic_mask:
            code.arg("mask_iter", "MaskIGemmIteratorDynamic&")
        else:
            code.arg("mask", f"uint32_t")
        code.arg("RS", f"const int&")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = {self.num_stage - 1};
        int smem_read_stage_idx = 0;
        """)
        if not self.dynamic_mask:
            code.raw(f"""
            MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
            """)
        code.raw(f"""
        int local_gemm_k_iterations = gemm_k_iterations;
        while(!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        input_iter_A.update_indices();

        TV_PRAGMA_UNROLL
        for (int stage=0; stage < {self.num_stage - 1}; ++stage){{
            GlobalAsyncCopyIter_A::do_copy_zfill(input_iter_A, smem_iter_A);
            GlobalAsyncCopyIter_B::do_copy_zfill(input_iter_B, smem_iter_B);
            CpAsyncGroup::make_fence();
            input_iter_A.increment_k();
            input_iter_B.increment_k();
            ++smem_iter_A;
            ++smem_iter_B;
            --local_gemm_k_iterations;
            if (!mask_iter.end && local_gemm_k_iterations == 0){{
                ++mask_iter;
                input_iter_A.reset_k();
                input_iter_B.reset_k();
                input_iter_A.increment_filter();
                input_iter_B.increment_filter();
                while (!mask_iter.valid() && !mask_iter.end){{
                    ++mask_iter;
                    input_iter_A.increment_filter();
                    input_iter_B.increment_filter();
                }}
                input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                input_iter_A.update_indices();
                local_gemm_k_iterations = gemm_k_iterations;
            }}
        }}
        CpAsyncGroup::wait_final_group();
        __syncthreads();
        
        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);
        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);
        ++warp_iter_A;
        ++warp_iter_B;

        while (local_gemm_k_iterations != {-self.num_stage + 1}){{

            TV_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});

                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                ++warp_iter_A;
                ++warp_iter_B;

                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                    warp_frag_B[warp_mma_k % 2], accumulators);
                
                if (warp_mma_k < {self.spec.num_warp_mma_iters - 1})
                    copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);

                if (warp_mma_k + 2 == {self.spec.num_warp_mma_iters}){{
                    copy_tiles_and_advance(input_iter_A, input_iter_B, {self.spec.num_warp_mma_iters - 1});
                    CpAsyncGroup::make_fence();

                    // do chores before wait
                    ++smem_iter_A;
                    ++smem_iter_B;
                    input_iter_A.increment_k();
                    input_iter_B.increment_k();
                    --local_gemm_k_iterations;
                    if (!mask_iter.end && local_gemm_k_iterations == 0){{
                        ++mask_iter;
                        input_iter_A.reset_k();
                        input_iter_B.reset_k();
                        input_iter_A.increment_filter();
                        input_iter_B.increment_filter();
                        while (!mask_iter.valid() && !mask_iter.end){{
                            ++mask_iter;
                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();
                        }}
                        input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                        input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                        input_iter_A.update_indices();
                        local_gemm_k_iterations = gemm_k_iterations;
                    }}
                    if (smem_write_stage_idx == {self.num_stage - 1}) {{
                        smem_iter_A.tile_increment(-{self.num_stage});
                        smem_iter_B.tile_increment(-{self.num_stage});
                        smem_write_stage_idx = 0;
                    }} else
                        ++smem_write_stage_idx;
                    
                    if (smem_read_stage_idx == {self.num_stage - 1}) {{
                        warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        smem_read_stage_idx = 0;
                    }} else
                        ++smem_read_stage_idx;
                    
                    // finish chores
                    CpAsyncGroup::wait_final_group();
                    __syncthreads();
                }}
            }}
        }}
        """)
        return code

    def call_mask_sparse_increase_k_RS_MSTAGE(self):
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 0;
        int smem_read_stage_idx = 0;
        // mask = 0xffffffff;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        while(!mask_iter.valid()){{
            ++mask_iter;
#ifndef DEBUG_MMA_MA_NOT_INC_FILTER
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
#endif
        }}
        int back_gemm_k_iterations = gemm_k_iterations;
        while(!mask_iter.end){{
#ifndef DEBUG_MMA_MA_NOT_UPDATE_INDICES_A
            input_iter_A.update_indices();
#endif
            gemm_k_iterations = back_gemm_k_iterations;
            TV_PRAGMA_UNROLL
            for(int i=0; i < {self.num_stage - 1}; ++i, --gemm_k_iterations){{
                if (gemm_k_iterations > 0){{
#ifndef DEBUG_MMA_MS_DOWNFALL_A
                    GlobalAsyncCopyIter_A::do_copy_zfill(input_iter_A, smem_iter_A);
#else
                    {self.input_spec.input_iter_a.fragment_t} input_frag_A;
#ifndef DEBUG_MMA_MS_NOT_READ_INPUT_A
                    input_iter_A.load(input_frag_A);
#endif
#ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_A
                    smem_iter_A.store(input_frag_A);
#endif
#endif
#ifndef DEBUG_MMA_MS_DOWNFALL_B
                    GlobalAsyncCopyIter_B::do_copy_zfill(input_iter_B, smem_iter_B);
#else
                    {self.input_spec.input_iter_b.fragment_t} input_frag_B;
#ifndef DEBUG_MMA_MS_NOT_READ_INPUT_B
                    input_iter_B.load(input_frag_B);
#endif
#ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_B
                    smem_iter_B.store(input_frag_B);
#endif
#endif
                    input_iter_A.increment_k();
                    input_iter_B.increment_k();
                }}
                CpAsyncGroup::make_fence();
                ++smem_iter_A;
                ++smem_iter_B;
                if (smem_write_stage_idx == {self.num_stage - 1}){{
                    smem_iter_A.tile_increment({-self.num_stage});
                    smem_iter_B.tile_increment({-self.num_stage});
                    smem_write_stage_idx = 0;
                }} else
                    ++smem_write_stage_idx;
                if (gemm_k_iterations == 1){{
                    input_iter_A.reset_k();
                    input_iter_B.reset_k();
#ifndef DEBUG_MMA_MA_NOT_INC_FILTER
                    input_iter_A.increment_filter();
                    input_iter_B.increment_filter();
#endif
                    ++mask_iter;
                    while (!mask_iter.valid() && !mask_iter.end){{
                        input_iter_A.increment_filter();
                        input_iter_B.increment_filter();
                        ++mask_iter;
                    }}
                }}
            }}
            CpAsyncGroup::wait_final_group();
            __syncthreads();
            warp_iter_A.set_kgroup_index(0);
            warp_iter_B.set_kgroup_index(0);

            warp_iter_A.load(warp_frag_A[0]);
            warp_iter_B.load(warp_frag_B[0]);

            ++warp_iter_A;
            ++warp_iter_B;

            for (; gemm_k_iterations > {-self.num_stage + 1}; ){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                    warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});

                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++warp_iter_A;
                    ++warp_iter_B;

                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                        warp_frag_B[warp_mma_k % 2], accumulators);

                    if (warp_mma_k < {self.spec.num_warp_mma_iters - 1})
                        if (gemm_k_iterations > 0)
                            copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);

                    if (warp_mma_k + 2 == {self.spec.num_warp_mma_iters}) {{
                        if (gemm_k_iterations > 0)
                            copy_tiles_and_advance(input_iter_A, input_iter_B, {self.spec.num_warp_mma_iters - 1});

                        CpAsyncGroup::make_fence();
                        CpAsyncGroup::wait_final_group();

                        __syncthreads();
                        ++smem_iter_A;
                        ++smem_iter_B;
                        if (gemm_k_iterations > 0){{
                            input_iter_A.increment_k();
                            input_iter_B.increment_k();
                        }}
                        
                        if (smem_write_stage_idx == {self.num_stage - 1}) {{
                            smem_iter_A.tile_increment(-{self.num_stage});
                            smem_iter_B.tile_increment(-{self.num_stage});
                            smem_write_stage_idx = 0;
                        }} else
                            ++smem_write_stage_idx;
                        
                        if (smem_read_stage_idx == {self.num_stage - 1}) {{
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                            warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                            smem_read_stage_idx = 0;
                        }} else
                            ++smem_read_stage_idx;
                        --gemm_k_iterations;
                        if (gemm_k_iterations == 0){{
                            input_iter_A.reset_k();
                            input_iter_B.reset_k();
#ifndef DEBUG_MMA_MA_NOT_INC_FILTER
                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();
#endif
                            ++mask_iter;
                            while (!mask_iter.valid() && !mask_iter.end){{
#ifndef DEBUG_MMA_MA_NOT_INC_FILTER
                                input_iter_A.increment_filter();
                                input_iter_B.increment_filter();
#endif
                                ++mask_iter;
                            }}
                        }}
                    }}
                }}

            }}
            CpAsyncGroup::wait_all();
            
            TV_PRAGMA_UNROLL
            for(int i=0; i <{self.spec.num_warp_mma_iters - 1}; ++i){{
                ++warp_iter_A;
                ++warp_iter_B;
            }}
            if (smem_read_stage_idx == {self.num_stage - 1}) {{
                warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                        {self.spec.num_warp_mma_iters});
                warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                        {self.spec.num_warp_mma_iters});
                smem_read_stage_idx = 0;
            }} else
                ++smem_read_stage_idx;

            if (smem_read_stage_idx != smem_write_stage_idx){{
                smem_iter_A.tile_increment(smem_read_stage_idx - smem_write_stage_idx);
                smem_iter_B.tile_increment(smem_read_stage_idx - smem_write_stage_idx);
                smem_write_stage_idx = smem_read_stage_idx;
            }}
            
            __syncthreads();
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call(self):
        """ Multi Stage MMA
        Smem have nStage plots.
        firstly ASYNC transport nStage-1 times from global InputIter to SmemIter
        then run gemm_k_iterations times: (current from 0 to gemm_k_iter -1)
            wait (current % nStage) plot finish transport( cp.async.wait_group nStage - 2)
            WMMA this, mean while,  transport (current-1 % nStage) plot from next Iter
        final  cp.async.wait_all;
        """
        if self.mask_sparse:
            if self.is_sparse_wgrad:
                return self.call_mask_sparse_wgrad()
            if self.increment_k_first:
                return self.call_mask_sparse_increase_k_V2()
            raise NotImplementedError

        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        assert not self.mask_sparse

        code.raw(f"""
        accumulators = src_accumulators;
        """)
        code.raw(f"""
        TV_PRAGMA_UNROLL
        for(int stage = 0; stage< {self.num_stage - 1}; ++stage, --gemm_k_iterations){{
            if(gemm_k_iterations == 0){{
                input_iter_A.clear_mask();
                input_iter_B.clear_mask();
            }}
            GlobalAsyncCopyIter_A::do_copy_zfill(input_iter_A, smem_iter_A);
            GlobalAsyncCopyIter_B::do_copy_zfill(input_iter_B, smem_iter_B);
            ++input_iter_A;
            ++input_iter_B;
            ++smem_iter_A;
            ++smem_iter_B;
            CpAsyncGroup::make_fence();
        }}
        CpAsyncGroup::wait_final_group();
        __syncthreads();

        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];

        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        WarpMma warp_mma;

        int smem_write_stage_idx = {self.num_stage - 1};
        int smem_read_stage_idx = 0;
        """)
        if self.clear_mask:
            code.raw(f"""
            if (gemm_k_iterations == 0) {{
                input_iter_A.clear_mask();
                input_iter_B.clear_mask();
            }}
            """)
        with code.for_(f"; gemm_k_iterations > {- self.num_stage + 1}; "):
            with code.for_(
                    f"int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k",
                    prefix="TV_PRAGMA_UNROLL"):
                code.raw(f"""
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                // if warp_mma_k is last, smem load next, warp load next too.

                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                ++warp_iter_A;
                ++warp_iter_B;

                // if (warp_mma_k > 0)
                //     warp_mma.transform(...)

                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                    warp_frag_B[warp_mma_k % 2], accumulators);

                if (warp_mma_k < {self.spec.num_warp_mma_iters - 1})
                    copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);

                if (warp_mma_k + 2 == {self.spec.num_warp_mma_iters}) {{
                    copy_tiles_and_advance(input_iter_A, input_iter_B, {self.spec.num_warp_mma_iters - 1});

                    CpAsyncGroup::make_fence();
                    CpAsyncGroup::wait_final_group();

                    __syncthreads();
                    ++smem_iter_A;
                    ++smem_iter_B;
                    ++input_iter_A;
                    ++input_iter_B;

                    if (smem_write_stage_idx == {self.num_stage - 1}) {{
                        smem_iter_A.tile_increment(-{self.num_stage});
                        smem_iter_B.tile_increment(-{self.num_stage});
                        smem_write_stage_idx = 0;
                    }} else
                        ++smem_write_stage_idx;
                    
                    if (smem_read_stage_idx == {self.num_stage - 1}) {{
                        warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        smem_read_stage_idx = 0;
                    }} else
                        ++smem_read_stage_idx;
                    --gemm_k_iterations;
                    if (gemm_k_iterations == 0){{
                        input_iter_A.clear_mask();
                        input_iter_B.clear_mask();
                    }}
                }}
                """)
                # with code.if_(f"warp_mma_k + 1 == {self.spec.num_warp_mma_iters}"):
                #     code.raw(f"""
                #         warp_mma.transform(...)
                #     """)
        return code

    async def __call__(self, gemm_k_iterations: int, accumulators: ArrayPtr,
                       input_iter_A: GemmInputIterator,
                       input_iter_B: GemmInputIterator,
                       src_accumulators: ArrayPtr):
        assert self.warp_iter_A is not None
        assert self.warp_iter_B is not None
        assert self.smem_iter_A is not None
        assert self.smem_iter_B is not None
        smem_iter_A = self.smem_iter_A
        smem_iter_B = self.smem_iter_B
        warp_iter_A = self.warp_iter_A
        warp_iter_B = self.warp_iter_B

        
        inp_coords_A_list = []
        inp_coords_B_list = []
        smem_coords_A_list = []
        smem_coords_B_list = []
        warp_coords_A_list = []
        warp_coords_B_list = []
        warp_frag_A_list = []
        warp_frag_B_list = []


        gemm_k_iterations_bkp = gemm_k_iterations
        for stage in range(self.num_stage - 1):
            if gemm_k_iterations == 0:
                input_iter_A.clear_mask_python()
                input_iter_B.clear_mask_python()
            await self.global_async_cp_a.do_copy_zfill_python(input_iter_A, smem_iter_A, self.cpasync_group)
            await self.global_async_cp_b.do_copy_zfill_python(input_iter_B, smem_iter_B, self.cpasync_group)
            input_iter_A.increment_python()
            input_iter_B.increment_python()
            smem_iter_A.increment_python()
            smem_iter_B.increment_python()
            self.cpasync_group.make_fence_python()
            gemm_k_iterations -= 1

        await self.cpasync_group.wait_final_group_python()
        await cudasim.syncthreads()
        # if cudasim.threadIdx().x == 0:
        #     smem_A = self.smem_A_ptr.data.numpy()
        #     print(smem_A.mean(), smem_A.min(), smem_A.max())
        #     print(input_frag_A.meta_data.numpy_view())
        #     print(smem_A_ptr.meta_data.numpy_view().astype(np.int32).reshape(-1, 128 + self.algo_spec.padding_mn[0])[:8, :10])


        warp_frag_A = [
            ArrayPtr(warp_iter_A.dtype.tv_dtype,
                     self.spec.warp_iter_a.element_count) for _ in range(2)
        ]
        warp_frag_B = [
            ArrayPtr(warp_iter_B.dtype.tv_dtype,
                     self.spec.warp_iter_b.element_count) for _ in range(2)
        ]


        warp_iter_A.set_wmma_k_index_python(0)
        warp_iter_B.set_wmma_k_index_python(0)

        warp_coords_A = await warp_iter_A.load_python(warp_frag_A[0])
        warp_coords_B = await warp_iter_B.load_python(warp_frag_B[0])

        warp_coords_A_list.append(warp_coords_A)
        warp_coords_B_list.append(warp_coords_B)
        warp_frag_A_list.append(warp_frag_A[0].meta_data.numpy_view().copy())
        warp_frag_B_list.append(warp_frag_B[0].meta_data.numpy_view().copy())

        warp_iter_A.increment_python()
        warp_iter_B.increment_python()
        if cudasim.debug_once():
            inpd = warp_frag_A[0].data.numpy_view()
            print("FirstWarpA",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())
            print(inpd)
            inpd = warp_frag_B[0].data.numpy_view()
            print("FirstWarpB",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())

        warp_mma = self.spec.warp_mma.python_ctor()
        smem_write_stage_idx = self.num_stage - 1
        smem_read_stage_idx = 0

        if cudasim.debug_once():
            print("gemm_k_iterations", gemm_k_iterations)
        while gemm_k_iterations > (-self.num_stage + 1):
            for warp_mma_k in range(self.spec.num_warp_mma_iters):
                warp_iter_A.set_wmma_k_index_python(
                    (warp_mma_k + 1) % self.spec.num_warp_mma_iters)
                warp_iter_B.set_wmma_k_index_python(
                    (warp_mma_k + 1) % self.spec.num_warp_mma_iters)
                
                warp_coords_A = await warp_iter_A.load_python(
                    warp_frag_A[(warp_mma_k + 1) % 2])
                warp_coords_B = await warp_iter_B.load_python(
                    warp_frag_B[(warp_mma_k + 1) % 2])
                
                warp_iter_A.increment_python()
                warp_iter_B.increment_python()

                # if len(
                #         warp_frag_A_list
                # ) != self.spec.num_warp_mma_iters * gemm_k_iterations_bkp:
                #     warp_frag_A_list.append(
                #         warp_frag_A[(warp_mma_k + 1) %
                #                     2].meta_data.numpy_view().copy())
                #     warp_frag_B_list.append(
                #         warp_frag_B[(warp_mma_k + 1) %
                #                     2].meta_data.numpy_view().copy())
                #     warp_coords_A_list.append(warp_coords_A)
                #     warp_coords_B_list.append(warp_coords_B)

                if warp_mma_k > 0:
                    pass        # there should be a transform

                await warp_mma(accumulators, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accumulators)

                if warp_mma_k < self.spec.num_warp_mma_iters - 1:
                    await self.copy_tiles_and_advance_python(input_iter_A, input_iter_B, warp_mma_k)
                
                if warp_mma_k + 2 == self.spec.num_warp_mma_iters:
                    await self.copy_tiles_and_advance_python(input_iter_A, input_iter_B, self.spec.num_warp_mma_iters - 1)
                    self.cpasync_group.make_fence_python()

                    await self.cpasync_group.wait_final_group_python()
                    await cudasim.syncthreads()
                    input_iter_A.increment_python()
                    input_iter_B.increment_python()
                    smem_iter_A.increment_python()
                    smem_iter_B.increment_python()
                    if smem_write_stage_idx == self.num_stage - 1:
                        smem_iter_A.tile_increment_python(-self.num_stage)
                        smem_iter_B.tile_increment_python(-self.num_stage)
                        smem_write_stage_idx = 0
                    else:
                        smem_write_stage_idx += 1
                    
                    if smem_read_stage_idx == self.num_stage - 1:
                        warp_iter_A.tile_increment_python(
                            -self.num_stage * self.partk *
                            self.spec.num_warp_mma_iters)
                        warp_iter_B.tile_increment_python(
                            -self.num_stage * self.partk *
                            self.spec.num_warp_mma_iters)
                        smem_read_stage_idx = 0
                    else:
                        smem_read_stage_idx += 1
                    gemm_k_iterations -= 1
                    if gemm_k_iterations == 0:
                        input_iter_A.clear_mask_python()
                        input_iter_B.clear_mask_python()
        if cudasim.debug_once():
            acc = accumulators.data.numpy_view()
            cudasim.debug_print(
                f"accumulator {acc.mean()} , max: {acc.max()} , min: {acc.min()}"
            )

        res = {
            "InputA": {
                "input_coords": inp_coords_A_list,
                "smem_coords": smem_coords_A_list,
                "warp_coords": warp_coords_A_list,
                "smem_shape": smem_iter_A.get_smem_vis_shape(),
                "warp_frags": warp_frag_A_list,
                "input_epa": input_iter_A.element_per_acc,
                "smem_epa": smem_iter_A.element_per_acc,
                "warp_epa": warp_iter_A.element_per_acc,
            },
            "InputB": {
                "input_coords": inp_coords_B_list,
                "smem_coords": smem_coords_B_list,
                "warp_coords": warp_coords_B_list,
                "warp_frags": warp_frag_B_list,
                "smem_shape": smem_iter_B.get_smem_vis_shape(),
                "input_epa": input_iter_B.element_per_acc,
                "smem_epa": smem_iter_B.element_per_acc,
                "warp_epa": warp_iter_B.element_per_acc,
            },
        }
        return res
