import contextlib
from typing import List, Optional, Union

import numpy as np
import pccm
from pccm.core import FunctionCode
from pccm.targets.cuda_ptx import RegDType

from cumm import dtypes
from cumm.common import GemmBasic, GemmBasicKernel, TensorView, TensorViewMath
from cumm.conv import bases, params
from cumm.conv.bases import LAYOUT_TYPES, ConvMode, ConvOpType
from cumm.gemm import codeops, constants, layout, thread_map
from cumm.gemm.arch.memory import GlobalLoad
from cumm.gemm.core import MetaArray, array_type, metaseq, seq


class WeightIterator(bases.ConvInputIterator):
    """
    fwd: NHWC -> NPQRSC @ KRSC, k = RSC
    dgrad: NPQK -> NHWRSK @ KRSC -> RSKC, k = RSK
    wgrad: NPQK @ NHWC -> NPQRSC, k = NPQ

    for weight, RSC or RSK
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 op_type: bases.ConvOpType,
                 tile_shape_mnk: MetaArray[int],
                 sub_tile_shape: MetaArray[int],
                 tmap: Union[thread_map.PitchLinear,
                             thread_map.PitchLinearWarpRaked],
                 problem_size: params.ConvProblem,
                 input_layout: Union[layout.TensorGeneric],
                 optimized: bool = False,
                 transpose_load: bool = False,
                 increment_k_first: bool = False):
        self.thread_access_shape = tmap.iterations
        self.iteration_delta = tmap.delta

        element_count = self.thread_access_shape[0] * self.thread_access_shape[
            1] * sub_tile_shape.prod()
        self.tile_shape_mnk = tile_shape_mnk
        super().__init__(dtype, element_count, sub_tile_shape[1])
        self.sub_tile_shape = sub_tile_shape
        self.op_type = op_type
        self.optimized = optimized
        self.ndim = problem_size.ndim
        self.increment_k_first = increment_k_first
        self.mask_sparse = problem_size.mask_sparse
        # for RR input (dgrad weight), it's possible to have tmap.iterations[1] > 1
        if op_type == ConvOpType.kForward:
            assert tmap.iterations[1] == 1
        assert tmap.iterations.prod() < 32, "error"
        self.add_dependency(TensorView, GemmBasicKernel)
        if not optimized:
            self.params = AnalyticParams(problem_size, input_layout)
        else:
            self.params = WeightOptParams(dtype, tile_shape_mnk, problem_size,
                                          input_layout, tmap,
                                          increment_k_first)
        self.tmap = tmap
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.problem_size = problem_size
        self.add_param_class("problem", problem_size, "ConvProblem")
        self.add_param_class("params", self.params, "Params")

        self.layout = input_layout

        self.add_param_class("input_layout", input_layout, "Layout")

        self.add_member("params_", "Params const&")
        self.add_member("problem_size_", "ConvProblem const&")
        self.add_member("pointer_", self.const_byte_pointer)
        self.reduce_channel_axis = 0 if self.op_type == ConvOpType.kBackwardInput else 1
        self.noreduce_channel_axis = 1 if self.op_type == ConvOpType.kBackwardInput else 0

        if optimized:
            self.add_member("filter_kernel_idx_", f"int")
            self.add_member("reduce_channel_offset_", "int")
        else:
            self.add_member(
                "reduce_channel_offsets_",
                "int",
                array=f"[{tmap.iterations[self.reduce_channel_axis]}]")
            self.add_member(
                "noreduce_channel_offsets_",
                "int",
                array=f"[{tmap.iterations[self.noreduce_channel_axis]}]")
            self.add_member("filter_kernel_idxes_",
                            f"tv::array<int, {self.ndim}>")

        if optimized:
            self.add_member("mask_", str(self.index_t))

    def get_params(self) -> pccm.ParameterizedClass:
        return self.params

    @pccm.cuda.constructor(host=True, device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("params", "Params const&")
        code.arg("problem_size", "ConvProblem const&")
        code.arg("ptr", self.const_pointer)
        code.arg("thread_id", "int")
        code.arg("threadblock_offset", "const tv::array<int, 2>&")
        code.ctor_init("params_", "params")
        code.ctor_init("problem_size_", "problem_size")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_byte_pointer}>(ptr)")
        if self.optimized:
            code.ctor_init("filter_kernel_idx_", "0")
            code.ctor_init("mask_", "0")

        else:
            code.ctor_init("filter_kernel_idxes_",
                           f"{{{', '.join(['0'] * self.ndim)}}}")

        code.raw(f"""
        auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
        """)
        if self.optimized:
            code.raw(f"""
            reduce_channel_offset_ = thread_offset[{self.reduce_channel_axis}];
            """)
        else:
            with code.range_(
                    "i", str(self.tmap.iterations[self.reduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                reduce_channel_offsets_[i] = thread_offset[{self.reduce_channel_axis}] +
                     i * {self.tmap.delta[self.reduce_channel_axis]};
                """)
            with code.range_(
                    "i", str(self.tmap.iterations[self.noreduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                noreduce_channel_offsets_[i] = thread_offset[{self.noreduce_channel_axis}] +
                     i * {self.tmap.delta[self.noreduce_channel_axis]};
                """)

        if self.optimized:
            with self.tmap.tmap_loop(code, "s", "c"):
                if self.tmap.iterations[1] == 1:
                    code.raw(f"""
                    uint32_t pred = thread_offset[0] + s * {self.tmap.delta[0]} < problem_size.K;
                    """)
                else:
                    code.raw(f"""
                    uint32_t pred = (thread_offset[0] + s * {self.tmap.delta[0]} < problem_size.K)
                         && (thread_offset[1] + c * {self.tmap.delta[1]} < problem_size.C);
                    """)
                code.raw(f"""
                mask_ |= (pred << (s * {self.tmap.iterations[1]} + c));
                """)
            if self.tmap.iterations[1] == 1:
                code.raw(f"""
                if (thread_offset[1] >= problem_size.C){{
                    mask_ = 0;
                }}
                """)
            code.raw(
                f"pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * {self.dtype.bitsize()} / 8;"
            )
        return code

    @pccm.cuda.member_function(name="operator++",
                               host=True,
                               device=True,
                               forceinline=True)
    def increment(self):
        code = FunctionCode()
        C_or_K = "C" if self.op_type == ConvOpType.kForward else "K"
        if not self.optimized:
            for i in range(self.ndim - 1, -1, -1):
                code.raw(f"""
                if (++filter_kernel_idxes_[{i}] < problem_size_.ksize[{i}]){{
                    return;
                }}
                filter_kernel_idxes_[{i}] = 0;
                """)
            with code.range_(
                    "i", str(self.tmap.iterations[self.reduce_channel_axis]),
                    "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                reduce_channel_offsets_[i] += {self.tile_shape_mnk[2]} * problem_size_.split_k_slices;
                """)
        else:
            code.raw(f"""
            if (++filter_kernel_idx_ == params_.kernel_prod){{
                filter_kernel_idx_ = 0;
                // back to first c
                pointer_ += params_.inc_c;
                reduce_channel_offset_ += params_.filter_c_delta;
            }}else{{
                // back to first rs
                pointer_ += params_.inc_rs;
            }}
            """)
            if self.op_type == ConvOpType.kForward:
                code.raw(f"""
                if (reduce_channel_offset_ >= problem_size_.{C_or_K}) {{
                    mask_ = 0;
                }}
                """)
            else:
                # reduce_channel_offset_ is K
                with self.tmap.tmap_loop(code, "s"):
                    code.raw(f"""
                    if (reduce_channel_offset_ + s * {self.tmap.delta[0]} >= problem_size_.{C_or_K}){{
                        uint32_t mask = ((1u << {self.tmap.iterations[1]}) - 1) << (s * {self.tmap.iterations[1]});
                        mask_ = mask_ & (~mask);
                    }}
                    """)
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def add_byte_offset(self):
        code = FunctionCode()
        code.arg("byte_offset", str(self.long_index_t))
        code.raw(f"""
        pointer_ += byte_offset;
        """)
        return code

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               const=True)
    def at(self):
        code = FunctionCode()
        if self.optimized:
            return code
        code.arg("stride, contig", "int")
        code.ret(f"tv::array<int, {self.ndim + 2}>")
        rs = codeops.unpack("filter_kernel_idxes_", range(self.ndim))
        if self.op_type == ConvOpType.kForward:
            code.raw(f"""
            return {{noreduce_channel_offsets_[stride], {rs}, reduce_channel_offsets_[contig]}};
            """)
        elif self.op_type == ConvOpType.kBackwardInput:
            code.raw(f"""
            return {{reduce_channel_offsets_[stride], {rs}, noreduce_channel_offsets_[contig]}};
            """)
        else:
            raise NotImplementedError
        return code

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               const=True)
    def valid(self):
        code = FunctionCode()
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            return indexes[0] < problem_size_.K && 
                indexes[{self.ndim + 1}] < problem_size_.C;
            """)
        else:
            code.arg("stride, contig", f"int")
            code.raw(f"""
            return mask_ & (1u << (stride * {self.tmap.iterations[1]} + contig));
            """)
        return code.ret(f"bool")

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               const=True)
    def get(self):
        code = FunctionCode()
        code.ret(self.const_access_pointer)
        if not self.optimized:
            code.arg("indexes", f"const tv::array<int, {self.ndim + 2}>&")
            code.raw(f"""
            auto offset = params_.layout(indexes);
            return reinterpret_cast<{self.const_access_pointer}>(pointer_ + offset * {self.dtype.bitsize()} / 8);
            """)
        else:
            code.arg("stride, contig", "int")
            code.raw(
                f"return reinterpret_cast<{self.const_access_pointer}>(pointer_ + contig * {self.tmap.delta[1]} * {self.dtype.bitsize()} / 8);"
            )
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.code()
        code.raw(f"""
        frag.clear();
        {self.access_t} *frag_ptr = reinterpret_cast<{self.access_t} *>(&frag);
        """)
        with code.range_("s", self.tmap.iterations[0], "TV_PRAGMA_UNROLL"):
            with code.range_("c", self.tmap.iterations[1], "TV_PRAGMA_UNROLL"):
                if not self.optimized:
                    code.raw(f"""
                    int idx = s * {self.tmap.iterations[1]} + c;
                    auto indexes = at(s, c);
                    {self.access_t} const *access_ptr = get(indexes) + pointer_offset / {self.element_per_acc};
                    tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                        frag_ptr[idx], access_ptr, valid(indexes));
                    """)
                else:
                    code.raw(f"""
                    int idx = s * {self.tmap.iterations[1]} + c;
                    {self.access_t} const *access_ptr = get(s, c) + pointer_offset / {self.element_per_acc};
                    tv::gemm::global_load<{self.access_t}, sizeof({self.access_t})>(
                        frag_ptr[idx], access_ptr, valid(s, c));
                    """)
            if self.optimized:
                # weight only use one ptr, so we need to increment
                # in every stride iteration
                code.raw(f"""
                if (s != {self.tmap.iterations[0] - 1}){{
                    pointer_ += params_.inc_strided;
                }}
                """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def load_invalid(self):
        code = pccm.code()
        if not self.optimized:
            return code
        with self.tmap.tmap_loop(code, "s"):
            code.raw(f"""
            if (s != {self.tmap.iterations[0] - 1}){{
                pointer_ += params_.inc_strided;
            }}
            """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def clear_mask(self):
        code = pccm.code()
        return code
