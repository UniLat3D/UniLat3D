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
                             GemmWarpIterator, GemmComponentBase)
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.conv.bases import ConvOpType


def div_up(a, b):
    return (a + b - 1) // b


class BlockMmaStorage(pccm.ParameterizedClass):
    def __init__(self, tile_shape: MetaArray[int],
                 smem_padding_a: MetaArray[int],
                 smem_padding_b: MetaArray[int], num_stage: int,
                 dtype_a: dtypes.DType, dtype_b: dtypes.DType):
        super().__init__()
        self.tile_shape = tile_shape
        self.smem_padding_a = smem_padding_a
        self.smem_padding_b = smem_padding_b

        self.num_stage = num_stage
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b

        self.smem_shape_a = seq(tile_shape[2] * num_stage + smem_padding_a[0],
                                smem_padding_a[1] + tile_shape[0])
        self.smem_shape_b = seq(tile_shape[2] * num_stage + smem_padding_b[0],
                                smem_padding_b[1] + tile_shape[1])
        self.smem_alignment = 16
        self.smem_size_a = self.smem_shape_a.prod() * dtype_a.itemsize()
        self.smem_size_b = self.smem_shape_b.prod() * dtype_b.itemsize()
        self.smem_size_a = div_up(self.smem_size_a,
                                  self.smem_alignment) * self.smem_alignment
        self.smem_size_b = div_up(self.smem_size_b,
                                  self.smem_alignment) * self.smem_alignment

        self.smem_size = self.smem_size_a + self.smem_size_b

        self.add_member(
            "smem_A",
            f"tv::alignedarray<{dtype_a}, {self.smem_shape_a.prod()}, {self.smem_alignment}>"
        )
        self.add_member(
            "smem_B",
            f"tv::alignedarray<{dtype_b}, {self.smem_shape_b.prod()}, {self.smem_alignment}>"
        )

class MaskIGemmIteratorMaskLoaderDynamic(pccm.ParameterizedClass):
    def __init__(self, tile_shape, increment_k_first, op_type):
        super().__init__()
        self.add_dependency(GemmBasicKernel, GemmBasic, TensorViewKernel)
        self.num_masks_per_thread = tile_shape[0] // constants.WARP_SIZE
        self.add_member("current_mask", "int")
        self.add_member("mask_int_count", "const int&")
        self.add_member("mask_load_idx", "int")
        self.add_member("RS_pos", "int")
        self.add_member("RS_offset", "int")
        if not increment_k_first:
            self.add_member("k_idx", "int")

        self.add_member("mask_filter", "const uint32_t &")
        self.add_member("tile_offset_m", "const int&")
        self.add_member("mask_ptr", "const uint32_t* const&")
        self.add_member("mask_out_ptr", "uint32_t* const&")
        self.add_member("RS", "const int&")
        self.add_member("reverse", "const bool&")
        self.add_member("gemm_k_iteration", "const int&")
        self.add_member("lane_idx", "const int&")
        self.add_member("problem_m", "const int&")

        self.add_member("end", "bool")
        self.add_member("mask_or_sum", "int")

        self.increment_k_first = increment_k_first
        self.op_type = op_type
        self.tile_shape = tile_shape


    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("mask_ptr", "const uint32_t* const&")
        code.arg("mask_out_ptr", "uint32_t* const&")
        code.arg("mask_int_count", "const int&")
        code.arg("tile_offset_m", "const int&")
        code.arg("gemm_k_iteration", "const int&")
        code.arg("RS", "const int&")
        code.arg("mask_filter", "const uint32_t&")
        code.arg("reverse", "const bool&")
        code.arg("lane_idx", "const int&")
        code.arg("problem_m", "const int&")
        code.ctor_init("mask_int_count", "mask_int_count")
        code.ctor_init("mask_ptr", "mask_ptr")
        code.ctor_init("mask_out_ptr", "mask_out_ptr")
        code.ctor_init("tile_offset_m", "tile_offset_m")
        code.ctor_init("RS", "RS")
        code.ctor_init("gemm_k_iteration", "gemm_k_iteration")
        code.ctor_init("end", "false")
        code.ctor_init("reverse", "reverse")
        code.ctor_init("mask_filter", "mask_filter")

        code.ctor_init("lane_idx", "lane_idx")
        code.ctor_init("problem_m", "problem_m")

        if not self.increment_k_first:
            code.ctor_init("k_idx", "0")
        code.raw(f"""
        mask_or_sum = 0;
        int used_rs = tv::div_up(RS, 32);
        """)
        with code.if_("reverse"):
            code.raw(f"""
                for (mask_load_idx = 0; mask_load_idx < used_rs - 1; ++mask_load_idx){{
                    load_mask(true);
                    mask_or_sum |= __brev(current_mask);
                }}
            """)
        with code.else_():
            code.raw(f"""
                for (mask_load_idx = 1; mask_load_idx < used_rs; ++mask_load_idx){{
                    load_mask(true);
                    mask_or_sum |= current_mask;
                }}
            """)


        code.raw(f"""
            init_mask_iter(true);
            mask_or_sum |= current_mask;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
            //     printf("OP {str(self.op_type)} into Dynamic Mask Loader  {"is" if self.increment_k_first else "not"} increment k,   mask_int_count:%d, reverse: %d\\n", mask_int_count, (int)reverse);
            // __syncwarp();

        """)

        return code

    @pccm.cuda.member_function(inline=True, device=True)
    def load_mask(self):
        code = pccm.code()
        code.arg("save", "const bool&", "false")
        code.raw(f"""
            current_mask = 0;
            tv::array<uint32_t, {self.num_masks_per_thread}> masks;
            masks.clear();
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.num_masks_per_thread}; ++i){{
                if (tile_offset_m * {self.tile_shape[0]} + i * {constants.WARP_SIZE} + lane_idx < problem_m){{
                    masks[i] = mask_ptr[mask_int_count * (tile_offset_m * {self.tile_shape[0]} + i * {constants.WARP_SIZE} + lane_idx) + mask_load_idx];
                }}
            }}
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.num_masks_per_thread}; ++i){{
                current_mask |= masks[i];
            }}
            // perform a warp reduce to get block mask
            TV_PRAGMA_UNROLL
            for (int mask = {constants.WARP_SIZE // 2}; mask > 0; mask /= 2) {{
                current_mask |= __shfl_xor_sync(0xffffffff, current_mask, mask, 32);
            }}
            current_mask &= mask_filter;
        """)
        if self.op_type == ConvOpType.kForward:
            with code.if_("save"):
                code.raw(f"""
                if (mask_out_ptr != nullptr)
                    mask_out_ptr[tile_offset_m * mask_int_count + mask_load_idx] = current_mask;
                """)

        return code
    @pccm.cuda.member_function(inline=True, device=True)
    def init_mask_iter(self):
        code = pccm.code()
        code.arg("save", "const bool&", "false")
        if self.op_type == ConvOpType.kBackwardInput:
            code.raw(f"""
                if (reverse){{
                    int used_mask = tv::div_up(RS, 32);
                    mask_load_idx = used_mask - 1;
                    RS_pos = 0;
                    RS_offset = used_mask * 32 - RS;
                    load_mask(save);
                    current_mask = __brev(current_mask);
                    return;
                }}
            """)
        code.raw(f"""
            mask_load_idx = 0;
            RS_pos = RS_offset = 0;
            load_mask(save);
        """)
        return code
    
    @pccm.cuda.member_function(device=True, inline=True, name="operator++")
    def increment_filter(self):
        code = pccm.code()
        with code.if_("++RS_pos >= RS"):
            if self.increment_k_first:
                code.raw("""
                    end = true;
                    return;
                """)
            else:
                code.raw("""
                    if (++k_idx >= gemm_k_iteration){{
                        end = true;
                        return;
                    }}
                    init_mask_iter();
                    return;
                """)
        with code.if_("(RS_pos + RS_offset) % 32 == 0"):
            if self.op_type == ConvOpType.kBackwardInput:
                with code.if_("reverse"):
                    code.raw("--mask_load_idx;")
                with code.else_():
                    code.raw("++mask_load_idx;")
            else:
                code.raw("++mask_load_idx;")
            code.raw("load_mask();")
            if self.op_type == ConvOpType.kBackwardInput:
                with code.if_("reverse"):
                    code.raw("current_mask = __brev(current_mask);")
        return code
    
    @pccm.cuda.member_function(device=True, inline=True)
    def valid(self):
        code = pccm.code()
        code.ret("bool")
        code.raw("""
            return !!(
                    current_mask & (1u << ((RS_pos + RS_offset) % 32))
                    );
        """)
        return code
    
    @pccm.cuda.member_function(device=True, inline=True)
    def empty(self):
        return pccm.code().ret("bool").raw("return !mask_or_sum;")


class MaskIGemmIterator(pccm.ParameterizedClass):
    def __init__(self, increment_k_first: bool = False):
        super().__init__()
        if not increment_k_first:
            self.add_member("k_idx, filter_idx", "int")
        else:
            self.add_member("filter_idx", "int")

        self.add_member("gemm_k_iterations, RS", "const int&")
        self.add_member("mask", "const uint32_t&")

        self.add_member("end", "bool")
        self.increment_k_first = increment_k_first

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("gemm_k_iterations, RS", "const int&")
        code.arg("mask", "const uint32_t&")
        if not self.increment_k_first:
            code.ctor_init("k_idx", "0")

        code.ctor_init("filter_idx", "0")
        code.ctor_init("gemm_k_iterations", "gemm_k_iterations")
        code.ctor_init("RS", "RS")

        code.ctor_init("mask", "mask")
        code.ctor_init("end", "false")

        return code

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator++")
    def increment(self):
        code = pccm.code()
        if self.increment_k_first:
            code.raw(f"""
            if (++filter_idx < RS){{
                return;
            }}
            end = true;
            """)
        else:
            code.raw(f"""
            if (++filter_idx < RS){{
                return;
            }}
            filter_idx = 0;
            if (++k_idx < gemm_k_iterations){{
                return;
            }}
            end = true;
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def valid(self):
        code = pccm.code()
        code.raw(f"""
        return mask & (1u << filter_idx);
        """)
        return code.ret("bool")


class Mma(GemmComponentBase):
    """a strange behavier exists in ptx compiler
    if we construct iterators in main kernel, compiled code is different from 
    construct them inside a class.
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
                 op_type: ConvOpType = ConvOpType.kForward,
                 dynamic_mask: bool = False):
        super().__init__()
        self.dtype_acc = dtype_acc
        miter = MaskIGemmIterator(increment_k_first)
        self.add_param_class("mma_ns_miter", miter, "MaskIGemmIterator")
        self.miterd = MaskIGemmIteratorMaskLoaderDynamic(spec.input_spec.tile_shape, increment_k_first, op_type)
        self.add_param_class("mma_ns_miterD", self.miterd, "MaskIGemmIteratorDynamic")
        self.dynamic_mask = dynamic_mask
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
        self.wmma = spec.warp_mma
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

    def min_arch(self) -> Optional[Tuple[int, int]]:
        return self.wmma.min_arch()

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
        new_obj = Mma(self.dtype_acc, self.partk, self.num_stage, self.spec,
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

        return new_obj

    def call_mask_sparse_k_first(self):
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        if self.dynamic_mask:
            code.arg("mask_iter", "MaskIGemmIteratorDynamic&")
        else:
            code.arg("mask", f"uint32_t")

        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        # for inc filter first, 0123 0123 0123 0123
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        // mask = 0xffffffff;
        """)
        if not self.dynamic_mask:
            code.raw(f"""
            MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
            """)
        code.raw(f"""
        // find initial gemm index
        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        // move to next valid location.
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        // TODO we should increment mask here to hidden increment compute time.
        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
            for (int i = 0; i < gemm_k_iterations; ++i){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                    if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                        // save to S1
                        smem_iter_A.store(input_frag_A);
                        smem_iter_B.store(input_frag_B);
                        __syncthreads();
                        ++smem_iter_A;
                        ++smem_iter_B;
                        // SMEM double buffer
                        if (smem_write_stage_idx == 1) {{
                            // back to S0
                            smem_iter_A.tile_increment(-{self.num_stage});
                            smem_iter_B.tile_increment(-{self.num_stage});
                        }} else {{
                            // 
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                            warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                        }}
                        smem_write_stage_idx ^= 1;
                    }}
                    warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++warp_iter_A;
                    ++warp_iter_B;
                    // load next input frag
                    // to hide long input latency
                    // by a whole wmma operation
                    if (warp_mma_k == 0){{
                        // 01 001
                        // here input iter point to next location of current
                        // mask iter (may be invalid), we need to increment to
                        // find a valid location.
                        if (i == gemm_k_iterations - 1){{

                            input_iter_A.reset_k();
                            input_iter_B.reset_k();
                            ++mask_iter;

                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();

                            while (!mask_iter.valid() && !mask_iter.end){{
                                ++mask_iter;
                                input_iter_A.increment_filter();
                                input_iter_B.increment_filter();
                            }}
                            // load next indices
                            // TODO why do we need 20 more registers when use if?
                            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_A.update_indices();
                        }}
                        input_iter_A.load(input_frag_A);
                        input_iter_B.load(input_frag_B);
                        input_iter_A.increment_k();
                        input_iter_B.increment_k();
                    }}
                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                            warp_frag_B[warp_mma_k % 2], accumulators);

                }}
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
        code.arg("num_reduced_mask", f"int")
        code.arg("tile_offset_k", f"int")
        code.arg("split_k_slices", f"int")

        code.arg("filter_offset", f"int")
        code.arg("mask_width", f"int")
        if self.dynamic_mask:
            code.arg("mask_int_count", f"const int&")

        # assert self.mask_width > 0
        # mask_width_rate = self.mask_width // self.input_spec.tile_shape[2]
        # assert self.mask_width % self.input_spec.tile_shape[2] == 0
        # assert mask_width_rate > 0
        # mask_width % tile_shape[2] == 0
        # mask_width % (tile_shape[2] * splitk) == 0 OR (tile_shape[2] * splitk) % mask_width == 0
        # mask_width_rate = mask_width // tile_shape[2]
        # unified_iterations delta = (tile_shape[2] * splitk)
        #
        # so mask_idx = unified_iterations // mask_width_rate
        code.raw(f"""
        int mask_width_rate = mask_width / {self.input_spec.tile_shape[2]};
        // tv::printf2_once("WTF num_reduced_mask",gemm_k_iterations, num_reduced_mask, split_k_slices, mask_width, mask_width_rate, filter_offset);
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
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        input_iter_B.increment_filter(filter_offset);
        int k_idx = tile_offset_k;
        int mask_idx = k_idx / mask_width_rate;
        uint32_t mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
        // mask = 0xffffffff;
        // find first valid mask
        // int total_skip_count = 0;
        int skip_cnt = 0; 
        // int gemm_k_iterations_bkp = gemm_k_iterations;
        while (!(mask & filter_offset_mask) && (gemm_k_iterations)){{
            skip_cnt += 1;
            k_idx += split_k_slices;
            mask_idx = k_idx / mask_width_rate;
            mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
            // mask = 0xffffffff;
            --gemm_k_iterations;
            input_iter_A.increment_no_clear_mask();
            input_iter_B.increment_no_clear_mask();
        }}
        // here current mask is loaded, k_idx and mask_idx point to current location.
        if (!gemm_k_iterations){{
            return;
        }}
        input_iter_A.clear_mask_if_batch_unbound();
        input_iter_B.clear_mask_if_batch_unbound();
        // input_iter_A += skip_cnt;
        // input_iter_B += skip_cnt;
        // total_skip_count += skip_cnt;
        // now input iter point to a valid location
        input_iter_A.update_indices();
        input_iter_B.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        ++input_iter_A; // point to next location, may be invalid
        ++input_iter_B;

        // now we have first valid location.

        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (gemm_k_iterations){{
            TV_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                    // save to S1
                    smem_iter_A.store(input_frag_A);
                    smem_iter_B.store(input_frag_B);
                    __syncthreads();
                    ++smem_iter_A;
                    ++smem_iter_B;
                    // SMEM double buffer
                    if (smem_write_stage_idx == 1) {{
                        // back to S0
                        smem_iter_A.tile_increment(-{self.num_stage});
                        smem_iter_B.tile_increment(-{self.num_stage});
                    }} else {{
                        // 
                        warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                    }}
                    smem_write_stage_idx ^= 1;
                }}
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                ++warp_iter_A;
                ++warp_iter_B;
                // load next input frag
                // to hide long input latency
                // by a whole wmma operation
                if (warp_mma_k == 0) {{
                    // 01 001
                    // here input iter point to next location of current
                    // mask iter (may be invalid), we need to increment to
                    // find a valid location.
                    --gemm_k_iterations;
                    skip_cnt = 0;
                    k_idx += split_k_slices;
                    mask_idx = k_idx / mask_width_rate;
                    // load current mask
                    mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
                    // mask = 0xffffffff;
                    while (!(mask & filter_offset_mask) && (gemm_k_iterations)){{
                        skip_cnt += 1;
                        k_idx += split_k_slices;
                        mask_idx = k_idx / mask_width_rate;
                        mask = (mask_idx < num_reduced_mask) ? reduced_mask_ptr[{mask_idx_str}] : 0;
                        // mask = 0xffffffff;
                        --gemm_k_iterations;
                        input_iter_A.increment_no_clear_mask();
                        input_iter_B.increment_no_clear_mask();
                        // ++input_iter_A;
                        // ++input_iter_B;
                    }}
                    input_iter_A.clear_mask_if_batch_unbound();
                    input_iter_B.clear_mask_if_batch_unbound();
                    // input_iter_A.clear_mask_if_pred(!gemm_k_iterations);
                    // input_iter_B.clear_mask_if_pred(!gemm_k_iterations);
                    // input_iter_A += skip_cnt;
                    // input_iter_B += skip_cnt;
                    // total_skip_count += skip_cnt;

                    // tv::printf2_once("RTX", skip_cnt, gemm_k_iterations);
                    input_iter_A.update_indices();
                    input_iter_B.update_indices();
                    input_iter_A.load(input_frag_A);
                    input_iter_B.load(input_frag_B);
                    ++input_iter_A;
                    ++input_iter_B;
                }}
                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                        warp_frag_B[warp_mma_k % 2], accumulators);
            }}

        }}
        // tv::printf2_once("FINAL SKIP", total_skip_count, "PREV", gemm_k_iterations_bkp);

        """)
        return code

    def call_mask_sparse_k_first_ffs(self):
        # slower than while loop. may due to slow ffs/mul.
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
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        // mask = 0xffffffff;
        // MaskIGemmIteratorV2 mask_iter(mask);
        // find initial gemm index
        int dist = __ffs(mask);
        mask >>= (dist - 1); // move to first valid location
        input_iter_A.increment_filter(dist - 1); // move to first valid location
        input_iter_B.increment_filter(dist - 1); // move to first valid location
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        input_iter_A.increment_k();
        input_iter_B.increment_k();

        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (mask){{
            for (int i = 0; i < gemm_k_iterations; ++i){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                    if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                        // save to S1
                        smem_iter_A.store(input_frag_A);
                        smem_iter_B.store(input_frag_B);
                        __syncthreads();
                        ++smem_iter_A;
                        ++smem_iter_B;
                        // SMEM double buffer
                        // TODO add a option to use single smem buffer
                        // to save smem
                        if (smem_write_stage_idx == 1) {{
                            // back to S0
                            smem_iter_A.tile_increment(-{self.num_stage});
                            smem_iter_B.tile_increment(-{self.num_stage});
                        }} else {{
                            // 
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                            warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                        }}
                        smem_write_stage_idx ^= 1;
                    }}
                    warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});

                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++warp_iter_A;
                    ++warp_iter_B;
                    // load next input frag
                    // to hide long input latency
                    // by a whole wmma operation
                    if (warp_mma_k == 0){{
                        // here input iter point to next location of current
                        // mask iter (may be invalid), we need to increment to
                        // find a valid location.
                        if (i == gemm_k_iterations - 1){{
                            input_iter_A.reset_k();
                            input_iter_B.reset_k();
                            mask >>= 1; // point to first invalid location
                            int dist = __ffs(mask); // find dist to next valid location
                            mask >>= (dist - 1); // move to that location
                            input_iter_A.increment_filter(dist);
                            input_iter_B.increment_filter(dist);
                            // TODO why do we need 20 more registers when use if?
                            // if (!mask_iter.end){{
                            input_iter_A.clear_all_mask_if_not_pred(mask);
                            input_iter_B.clear_all_mask_if_not_pred(mask);
                            input_iter_A.update_indices();
                            // }}
                        }}
                        input_iter_A.load(input_frag_A);
                        input_iter_B.load(input_frag_B);
                        input_iter_A.increment_k();
                        input_iter_B.increment_k();
                    }}
                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                            warp_frag_B[warp_mma_k % 2], accumulators);
                }}
            }}
        }}
        """)
        return code

    def call_mask_sparse_filter_first(self):
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        if self.dynamic_mask:
            code.arg("mask_iter", "MaskIGemmIteratorDynamic&")
        else:
            code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        # for inc filter first, 0123 0123 0123 0123
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        // mask = 0xffffffff;
        """)
        if not self.dynamic_mask:
            code.raw(f"""
            MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
            """)
        code.raw(f"""
        // find initial gemm index
        while (!mask_iter.valid()){{
            ++mask_iter;
            ++input_iter_A;
            ++input_iter_B;
            input_iter_B.load_invalid();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        ++input_iter_A; // point to next location, may be invalid
        ++input_iter_B;

        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
            TV_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                    // save to S1
                    smem_iter_A.store(input_frag_A);
                    smem_iter_B.store(input_frag_B);
                    __syncthreads();
                    ++smem_iter_A;
                    ++smem_iter_B;
                    // SMEM double buffer
                    if (smem_write_stage_idx == 1) {{
                        // back to S0
                        smem_iter_A.tile_increment(-{self.num_stage});
                        smem_iter_B.tile_increment(-{self.num_stage});
                    }} else {{
                        // 
                        warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                    }}
                    smem_write_stage_idx ^= 1;
                }}
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                ++warp_iter_A;
                ++warp_iter_B;
                // load next input frag
                // to hide long input latency
                // by a whole wmma operation
                if (warp_mma_k == 0){{
                    // 01 001
                    // here input iter point to next location of current
                    // mask iter (may be invalid), we need to increment to
                    // find a valid location.
                    ++mask_iter;
                    while (!mask_iter.valid() && !mask_iter.end){{
                        ++mask_iter;
                        ++input_iter_A;
                        ++input_iter_B;
                        input_iter_B.load_invalid();
                    }}
                    // now mask iter is valid, input_iter_A and input_iter_B
                    // point to a valid location.
                    input_iter_A.update_indices();
                    input_iter_A.load(input_frag_A);
                    input_iter_B.load(input_frag_B);
                    ++input_iter_A;
                    ++input_iter_B;
                }}
                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                        warp_frag_B[warp_mma_k % 2], accumulators);
            }}
        }}
        """)
        return code

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call(self):
        """ pipelined mma:
        let I is input read, Si is smem store, W0 is warp load 0, W1 is warp load 1.
        the mma math operation is M0 and M1 for W0 and W1
        the init progress is I S0 W0
        every gemm iteration is I+ W1 M0 [W0 M1 W1 M0] * (gemm_k - 2) ... S+ W0+ M1
        double buffer [W0 M1 W1 M0] is used to hidden smem read (Wi) latency
        and mma operation (Mi)
        we load next input buffer in start of current wmma to hidden input latency,
        the last S+ and W0+ is hidden by last M1.

        for smem double buffer, TODO
        """
        if self.mask_sparse:
            if self.is_sparse_wgrad:
                return self.call_mask_sparse_wgrad()
            if self.increment_k_first:
                return self.call_mask_sparse_k_first()
            else:
                return self.call_mask_sparse_filter_first()
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")

        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        if self.first_input_clear:
            code.raw(f"""
            input_frag_A.clear();
            input_frag_B.clear();
            """)
        code.raw(f"""
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        ++input_iter_A;
        ++input_iter_B;
        """)
        if cudasim.enable_debug():
            code.raw(f"""
            tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(input_frag_A, "FirstInputA");
            tv::print_fragment_once<float, 0, 16, {cudasim.debug_tx()}>(input_frag_A);

            tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(input_frag_B, "FirstInputB");
            """)

        code.raw(f"""
        // tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(input_frag_A, "FirstInputA", blockIdx.z, gemm_k_iterations);
        // tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(input_frag_B, "FirstInputB", blockIdx.z);
        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();

        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];

        ++smem_iter_A;
        ++smem_iter_B;
        warp_iter_A.set_kgroup_index(0);
        warp_iter_B.set_kgroup_index(0);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        // tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(warp_frag_A[0], "FirstWarpA", blockIdx.z, warp_frag_A[0].size());
        // tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(warp_frag_B[0], "FirstWarpB", blockIdx.z);
        // if (blockIdx.z == 0){{
        //     tv::print_fragment_once<float, 0, 8, {cudasim.debug_tx()}>(warp_frag_A[0]);
        // }}

        ++warp_iter_A;
        ++warp_iter_B;

        WarpMma warp_mma;

        int smem_write_stage_idx = 1;
        """)
        if self.clear_mask:
            code.raw(f"""
            if (gemm_k_iterations <= 1) {{
                input_iter_A.clear_mask();
                input_iter_B.clear_mask();
            }}
            """)
        with code.for_("; gemm_k_iterations > 0; --gemm_k_iterations"):
            with code.for_(
                    f"int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k",
                    prefix="TV_PRAGMA_UNROLL"):
                code.raw(f"""
                if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                    // TODO
                    // tv::printf2_once(gemm_k_iterations);
                    smem_iter_A.store(input_frag_A);
                    smem_iter_B.store(input_frag_B);
                    __syncthreads();
                    ++smem_iter_A;
                    ++smem_iter_B;
                    if (smem_write_stage_idx == 1) {{
                        smem_iter_A.tile_increment(-{self.num_stage});
                        smem_iter_B.tile_increment(-{self.num_stage});
                    }} else {{
                        warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                {self.spec.num_warp_mma_iters});
                    }}
                    smem_write_stage_idx ^= 1;
                }}
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                // if warp_mma_k is last, smem load next, warp load next too.

                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                ++warp_iter_A;
                ++warp_iter_B;
                """)
                with code.if_("warp_mma_k == 0"):
                    code.raw(f"""
                    input_iter_A.load(input_frag_A);
                    input_iter_B.load(input_frag_B);
                    // tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(input_frag_A, "InputA", blockIdx.z);
                    // tv::print_fragment_meta_once<float, {cudasim.debug_tx()}>(input_frag_B, "InputB", blockIdx.z);

                    ++input_iter_A;
                    ++input_iter_B;
                    """)
                    if self.clear_mask:
                        code.raw(f"""
                        if (gemm_k_iterations <= 2) {{
                            input_iter_A.clear_mask();
                            input_iter_B.clear_mask();
                        }}
                    """)
                code.raw(f"""
                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                        warp_frag_B[warp_mma_k % 2], accumulators);
                """)
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

        input_frag_A = ArrayPtr(input_iter_A.dtype.tv_dtype,
                                input_iter_A.element_count)
        input_frag_B = ArrayPtr(input_iter_B.dtype.tv_dtype,
                                input_iter_B.element_count)
        input_frag_A.clear()
        input_frag_B.clear()
        inp_coords_A_list = []
        inp_coords_B_list = []
        smem_coords_A_list = []
        smem_coords_B_list = []
        warp_coords_A_list = []
        warp_coords_B_list = []
        warp_frag_A_list = []
        warp_frag_B_list = []
        warp_frag_A = [
            ArrayPtr(warp_iter_A.dtype.tv_dtype,
                     self.spec.warp_iter_a.element_count) for _ in range(2)
        ]
        warp_frag_B = [
            ArrayPtr(warp_iter_B.dtype.tv_dtype,
                     self.spec.warp_iter_b.element_count) for _ in range(2)
        ]

        inp_coors_A = input_iter_A.load_python(input_frag_A)
        # print(inp_coors_A)
        inp_coords_A_list.append(inp_coors_A)
        # if cudasim.threadIdx().x < 32:
        #     print(cudasim.threadIdx().x, inp_coors_A)
        inp_coors_B = input_iter_B.load_python(input_frag_B)
        inp_coords_B_list.append(inp_coors_B)
        if cudasim.debug_once():
            print("GEMM ITERATIONS", gemm_k_iterations)
            inpd = input_frag_A.data.numpy_view()
            print("FirstInputA",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())
            inpd = input_frag_B.data.numpy_view()
            print("FirstInputB",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())

        input_iter_A.increment_python()
        input_iter_B.increment_python()
        smem_coords_A = await smem_iter_A.store_python(input_frag_A)
        smem_coords_B = await smem_iter_B.store_python(input_frag_B)
        smem_coords_A_list.append(smem_coords_A)
        smem_coords_B_list.append(smem_coords_B)

        await cudasim.syncthreads()
        # if cudasim.threadIdx().x == 0:
        #     smem_A = self.smem_A_ptr.data.numpy()
        #     print(smem_A.mean(), smem_A.min(), smem_A.max())
        #     print(input_frag_A.meta_data.numpy_view())
        #     print(smem_A_ptr.meta_data.numpy_view().astype(np.int32).reshape(-1, 128 + self.algo_spec.padding_mn[0])[:8, :10])

        smem_iter_A.increment_python()
        smem_iter_B.increment_python()

        warp_iter_A.set_wmma_k_index_python(0)
        warp_iter_B.set_wmma_k_index_python(0)

        warp_coords_A = await warp_iter_A.load_python(warp_frag_A[0])
        warp_coords_B = await warp_iter_B.load_python(warp_frag_B[0])
        # if (cudasim.threadIdx().x == 0):

        warp_coords_A_list.append(warp_coords_A)
        warp_coords_B_list.append(warp_coords_B)
        warp_frag_A_list.append(warp_frag_A[0].meta_data.numpy_view().copy())
        warp_frag_B_list.append(warp_frag_B[0].meta_data.numpy_view().copy())

        # if cudasim.threadIdx().x == 0:
        #     print(warp_frag_A[0].data.numpy_view().astype(np.int32))
        #     print(warp_frag_A[0].data.numpy_view().mean(), "WARP_A_FIRST")
        #     print(warp_frag_B[0].data.numpy_view().mean(), "WARP_B_FIRST")
        # if cudasim.threadIdx().x == 0:

        # print(cudasim.threadIdx().x, input_frag_A.data.mean(), "input_frag_A FIRST")
        # print(input_frag_B.data.mean(), "input_frag_B FIRST")
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
        smem_write_stage_idx = 1
        gemm_k_iterations_bkp = gemm_k_iterations
        if gemm_k_iterations <= 1:
            input_iter_A.clear_mask_python()
            input_iter_B.clear_mask_python()
        if cudasim.debug_once():
            print("gemm_k_iterations", gemm_k_iterations)
        while gemm_k_iterations > 0:
            for warp_mma_k in range(self.spec.num_warp_mma_iters):
                if (warp_mma_k == self.spec.num_warp_mma_iters - 1):
                    smem_coords_A = await smem_iter_A.store_python(input_frag_A
                                                                   )
                    smem_coords_B = await smem_iter_B.store_python(input_frag_B
                                                                   )
                    if len(smem_coords_A_list) < gemm_k_iterations_bkp:
                        smem_coords_A_list.append(smem_coords_A)
                        smem_coords_B_list.append(smem_coords_B)
                    await cudasim.syncthreads()
                    smem_iter_A.increment_python()
                    smem_iter_B.increment_python()

                    if (smem_write_stage_idx == 1):
                        smem_iter_A.tile_increment_python(-self.num_stage)
                        smem_iter_B.tile_increment_python(-self.num_stage)
                    else:
                        warp_iter_A.tile_increment_python(
                            -self.num_stage * self.partk *
                            self.spec.num_warp_mma_iters)
                        warp_iter_B.tile_increment_python(
                            -self.num_stage * self.partk *
                            self.spec.num_warp_mma_iters)

                    smem_write_stage_idx ^= 1
                # if cudasim.threadIdx().x == 255:
                #     print(warp_mma_k)
                warp_iter_A.set_wmma_k_index_python(
                    (warp_mma_k + 1) % self.spec.num_warp_mma_iters)
                warp_iter_B.set_wmma_k_index_python(
                    (warp_mma_k + 1) % self.spec.num_warp_mma_iters)

                warp_coords_A = await warp_iter_A.load_python(
                    warp_frag_A[(warp_mma_k + 1) % 2])
                warp_coords_B = await warp_iter_B.load_python(
                    warp_frag_B[(warp_mma_k + 1) % 2])
                if len(
                        warp_frag_A_list
                ) != self.spec.num_warp_mma_iters * gemm_k_iterations_bkp:
                    warp_frag_A_list.append(
                        warp_frag_A[(warp_mma_k + 1) %
                                    2].meta_data.numpy_view().copy())
                    warp_frag_B_list.append(
                        warp_frag_B[(warp_mma_k + 1) %
                                    2].meta_data.numpy_view().copy())
                    warp_coords_A_list.append(warp_coords_A)
                    warp_coords_B_list.append(warp_coords_B)

                # if cudasim.threadIdx().x == 0:
                #     wa = warp_frag_A[(warp_mma_k + 1) % 2].data.numpy_view()
                #     print(wa.astype(np.int32))
                warp_iter_A.increment_python()
                warp_iter_B.increment_python()
                if (warp_mma_k == 0):
                    inp_coors_A = input_iter_A.load_python(input_frag_A)
                    inp_coors_B = input_iter_B.load_python(input_frag_B)
                    inp_coords_A_list.append(inp_coors_A)
                    inp_coords_B_list.append(inp_coors_B)
                    if cudasim.debug_once():
                        inpd = input_frag_A.data.numpy_view()
                        print("InputA",
                              cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                              inpd.min())
                        inpd = input_frag_B.data.numpy_view()
                        print("InputB",
                              cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                              inpd.min())

                    # if cudasim.threadIdx().x == 200:
                    #     print(input_frag_A.data.mean(), "INPUT A")
                    input_iter_A.increment_python()
                    input_iter_B.increment_python()
                    if (gemm_k_iterations <= 2):
                        input_iter_A.clear_mask_python()
                        input_iter_B.clear_mask_python()
                if cudasim.debug_once():
                    inpd = warp_frag_A[warp_mma_k % 2].data.numpy_view()
                    print(f"WarpA", warp_mma_k,
                          cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                          inpd.min())
                    inpd = warp_frag_B[warp_mma_k % 2].data.numpy_view()
                    print(f"WarpB", warp_mma_k,
                          cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                          inpd.min())
                await warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                               warp_frag_B[warp_mma_k % 2], accumulators)

            gemm_k_iterations -= 1
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
