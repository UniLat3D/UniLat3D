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

from operator import truediv
from os import access
from typing import List

import numpy as np
import pccm

from cumm import dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, GemmBasicKernel, TensorView
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.cudasim import checkers, warp_broadcast
from cumm.gemm import arch, bases, constants, layout_tensorop, thread_map
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.thread_map import PitchLinearWarpRaked

# def seq(*vals) -> np.ndarray:
#     return np.array([*vals], dtype=np.int64)


def div_up(a, b):
    return (a + b - 1) // b


def count_set_bits(n):
    count = 0
    while (n):
        n &= (n - 1)
        count += 1
    return count


def swizzle_increment(index: int, width: int):
    assert count_set_bits(width) == 1, "width must be power of 2"
    if index & 0b1 == 0:
        # bit 0 advance
        return 1 * width
    elif index == 0b1:
        # bit 1 advance
        return 0b11 * width
    elif index == 0b11:
        # bit 2 advance
        return 0b111 * width
    elif index == 0b111:
        # bit 3 advance
        return 0b1111 * width
    else:
        raise NotImplementedError


def swizzle_increment_code(code: pccm.FunctionCode, offset_var: str,
                           index_var: str, width: int):
    assert count_set_bits(width) == 1, "width must be power of 2"
    with code.if_(f"(({index_var}) & 0b1) == 0"):
        code.raw(f"{offset_var} ^= 1 * {width};")
    with code.else_if_(f"({index_var}) == 0b1"):
        code.raw(f"{offset_var} ^= 0b11 * {width};")
    with code.else_if_(f"({index_var}) == 0b11"):
        code.raw(f"{offset_var} ^= 0b111 * {width};")
    with code.else_if_(f"({index_var}) == 0b111"):
        code.raw(f"{offset_var} ^= 0b1111 * {width};")


class MyTensorOpLayout(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, warp_shape: MetaArray[int],
                 base_stride: int, stage_axis: int,
                 stage_count: MetaArray[int], is_permute_m: bool):
        """sw subsw
        Swizzle: multi-level switch system.
        2: 01 -> 10
        4: 01 -> 10
        8: 01 -> 10
        stride = base_stride * interleave * stage_count[1]
        """

        super().__init__()
        self.dtype = dtype
        self.is_permute_m = is_permute_m
        self.is_spec_32 = dtype.bitsize() == 32 and is_permute_m
        self.warp_shape = warp_shape
        interleave = 1
        smem_bank_size = 128
        self.base_stride = base_stride
        self.stage_axis = stage_axis
        self.stage_count = stage_count

        smem_access_size_bits = 128
        smem_access_size = 128 // 8
        store_access_size = dtype.bitsize()
        warp_access_size = dtype.bitsize()
        self.element_per_acc = smem_access_size // dtype.itemsize()
        store_num_access_bf = smem_bank_size // smem_access_size
        warp_num_access_bf = smem_bank_size // smem_access_size
        assert store_num_access_bf == 8
        warp_bank_free_length = 8
        if not self.is_spec_32:
            warp_bfa_access_shape = metaseq(warp_num_access_bf, 1)
            if warp_shape[1] < store_num_access_bf:
                # if a QP access a shape with stride > 1
                # we must use interleaved layout to ensure
                # bank free store
                # interleave: every kInterleave lines put in contiguous line in smem
                # interleave == 1, warp_bfa_access_shape=[8, 1]:
                # 0
                #  1
                #   2
                #    3
                #     4
                #      5
                #       6
                #        7
                # interleave == 2 (input stride=0-7, warp_bfa_access_shape=[4, 2]):
                # 0   1
                #  2   3
                #   4   5
                #    6   7
                # interleave == 4 (input stride=0-7, warp_bfa_access_shape=[2, 4]):
                # 0 1 2 3
                #  4 5 6 7
                # interleave == 8 (input stride=0-7, warp_bfa_access_shape=[1, 8]):
                # 01234567

                interleave = store_num_access_bf // warp_shape[1]
                warp_bfa_access_shape = metaseq(
                    warp_num_access_bf // interleave, interleave)

            self.interleave = interleave
            self.warp_bfa_access_shape = warp_bfa_access_shape
            # num pointers:
            # when a swizzle part can't be handled in ONE iteration for ONE warp,
            # we need multiple pointers to handle different part of a swizzle part.

            # 1. SmemStoreIter
            # during store, if warp shape is [4, 8] and swizzle part is [8, 8], then we
            # need multiple pointers.

            # 2. WarpReadIter
            # for PermuteK iters (Crosswise in cutlass), the permute is done in k axis,
            # the min stride of tensor op is 8, which is the maximum size of swizzle part stride
            # so PermuteK iters don't need multiple pointers. (LDSM count is always 1 in k axis)
            #
            # for PermuteM iters (Congruous in cutlass), the permute is done in m|n axis,
            # ldm shape contig may smaller than swizzle part contig.
            # so one swizzle part may contains multiple iterations.

            # for PermuteK iters, if the tensor op stride is smaller than maximum swizzle part stride,
            # we still need multiple pointers.
            self.num_smem_pointers = max(
                warp_bfa_access_shape[0] // warp_shape[0], 1)

            self.subsw_length = 4
            if interleave != 1:
                self.subsw_length = min(warp_bfa_access_shape[0], 4)
            self.subsw_shape = metaseq(self.subsw_length, self.subsw_length)

            self.sw_shape = metaseq(warp_bfa_access_shape[0],
                                    store_num_access_bf)
            self.tile_shape = self.sw_shape
            self.part_shape = self.subsw_shape
            self.subsw_count = self.sw_shape // self.subsw_shape

            self.num_swizzle_part = warp_bfa_access_shape[
                0] // self.subsw_length
        else:
            self.sw_shape = metaseq(4, 8)
            self.subsw_shape = metaseq(4, 8)
            self.subsw_count = self.sw_shape // self.subsw_shape
            self.tile_shape = self.sw_shape
            self.part_shape = self.subsw_shape
            self.interleave = 1
            self.num_smem_pointers = 1
            self.warp_bfa_access_shape = metaseq(8, 1)

        self.static_stride = base_stride * self.interleave * stage_count[1]
        self.static_stride_vec = self.static_stride // self.element_per_acc

        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        # cudasim mmebers
        self.stride = 0

    def __repr__(self):
        return f"TensorOpLayout[sw={self.sw_shape}|il={self.interleave}]"

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        return code

    def python_ctor(self, stride):
        new_obj = MyTensorOpLayout(self.dtype, self.warp_shape,
                                   self.base_stride, self.stage_axis,
                                   self.stage_count, self.is_permute_m)
        new_obj.stride = stride
        return new_obj

    def from_shape_python(self, shape):
        l = MyTensorOpLayout(self.dtype, self.warp_shape, self.base_stride,
                             self.stage_axis, self.stage_count,
                             self.is_permute_m)
        l.stride = shape[0]
        return l

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(f"return {self.class_name}();")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True,
                               const=True)
    def call_operator(self):
        code = pccm.code()
        shape_before_interleave = self.sw_shape[1] // self.interleave
        assert self.subsw_count[0] == 1 or self.subsw_count[0] == 2
        if self.is_spec_32:
            code.raw(f"""
                int tc = ec / 32;
                int ts = s / 4;
                int c = (ec % 32) / {self.element_per_acc};
                s = s % 4;
                return ((c ^ (2 * s)) * {self.element_per_acc} +
                        s * {self.static_stride} + tc * 32 +
                        ts * {self.static_stride} * 4 + (ec % 4));
            """)
        else:
            code.raw(f"""
            int vc = ec / {self.element_per_acc};
            int interleaved_s = s / {self.interleave};
            int idx_in_interleave_s = s % {self.interleave};
            // shape_before_interleave = {self.sw_shape[1]} // {self.interleave}
            // int sw_idx_s = interleaved_s / {self.sw_shape[0]};
            int sw_idx_c = vc / {shape_before_interleave};
            int idx_in_sw_c = vc % {shape_before_interleave} + idx_in_interleave_s * {shape_before_interleave};
            int idx_in_sw_s = interleaved_s % {self.sw_shape[0]};

            int subsw_idx_s = idx_in_sw_s / {self.subsw_shape[0]};
            int subsw_idx_c = idx_in_sw_c / {self.subsw_shape[1]};
            int idx_in_subsw_s = idx_in_sw_s % {self.subsw_shape[0]};
            int idx_in_subsw_c = idx_in_sw_c % {self.subsw_shape[1]};
            
            // if subsw_idx_s == 0, permuted_subsw_idx_c = 0/1 = subsw_idx_c
            // else permuted_subsw_idx_c = 1/0
            int permuted_subsw_idx_c = subsw_idx_c;
            if ({self.subsw_count[0]} > 1){{
                permuted_subsw_idx_c = subsw_idx_c ^ (subsw_idx_s % 2);
            }}
            int premuted_idx_in_subsw_c = idx_in_subsw_c ^ (idx_in_subsw_s % {self.subsw_length});
            int final_c = sw_idx_c * {self.sw_shape[1]} + permuted_subsw_idx_c * {self.subsw_shape[1]} + premuted_idx_in_subsw_c;
            // if ec % epa != 0
            int final_ec = final_c * {self.element_per_acc} + ec % {self.element_per_acc};
            int final_s = interleaved_s * {self.static_stride};
            return final_ec + final_s;
            """)
        code.arg("s,ec", self.index_t)
        return code.ret(self.long_index_t)

    def __call__(self, s: int, ec: int):
        if self.is_spec_32:
            tc = ec // 32
            ts = s // 4

            c = (ec % 32) // self.element_per_acc
            s = s % 4
            offset = ((c ^ (2 * s)) * self.element_per_acc +
                      s * self.static_stride + tc * 32 +
                      ts * self.static_stride * 4 + ec % 4)
            return offset

        vc = ec // self.element_per_acc
        interleaved_s = s // self.interleave
        idx_in_interleave_s = s % self.interleave
        # shape_before_interleave = self.warp_shape[1]
        shape_before_interleave = self.sw_shape[1] // self.interleave
        sw_idx_s = interleaved_s // self.sw_shape[0]
        sw_idx_c = vc // shape_before_interleave
        idx_in_sw_c = vc % shape_before_interleave + idx_in_interleave_s * shape_before_interleave
        idx_in_sw_s = interleaved_s % self.sw_shape[0]
        subsw_idx_s = idx_in_sw_s // self.subsw_shape[0]
        subsw_idx_c = idx_in_sw_c // self.subsw_shape[1]
        idx_in_subsw_s = idx_in_sw_s % self.subsw_shape[0]
        idx_in_subsw_c = idx_in_sw_c % self.subsw_shape[1]
        permuted_subsw_idx_c = subsw_idx_c
        # if subsw_idx_s == 0, permuted_subsw_idx_c = 0/1 = subsw_idx_c
        # else permuted_subsw_idx_c = 1/0
        if self.subsw_count[0] == 2:
            permuted_subsw_idx_c = subsw_idx_c ^ (subsw_idx_s % 2)
        elif self.subsw_count[0] != 1:
            raise NotImplementedError
        if self.subsw_length == 4:
            premuted_idx_in_subsw_c = idx_in_subsw_c ^ (idx_in_subsw_s % 4)
        elif self.subsw_length == 2:
            premuted_idx_in_subsw_c = idx_in_subsw_c ^ (idx_in_subsw_s % 2)
        else:
            raise NotImplementedError
        # print("self.subsw_shape", self.subsw_shape, self.sw_shape, self.interleave, self.subsw_count, self.element_per_acc)
        final_c = sw_idx_c * self.sw_shape[
            1] + permuted_subsw_idx_c * self.subsw_shape[1]
        final_c += premuted_idx_in_subsw_c
        # if ec % epa != 0
        final_ec = final_c * self.element_per_acc + ec % self.element_per_acc
        final_s = interleaved_s * self.static_stride
        return final_ec + final_s

    def get_ldm_initial_offset_ref(self,
                                   lane_idx: int,
                                   ldm_count: MetaArray[int],
                                   transpose: bool = False,
                                   permute_m_pointer_idx: int = 0):
        """ transpose (not operend A)
        if transpose:
            Q0 Q1
            Q2 Q3
        else:
            Q0 Q2
            Q1 Q3
        """
        if ldm_count[0] == 1:
            # Q0 Q1 Q2 Q3
            # stride: 01234567 01234567 ....
            # contig: 00000000 11111111 22222222 ....
            stride = lane_idx & 0b111
            contig_vec = lane_idx >> 3
        elif ldm_count[1] == 1:
            # Q0
            # Q1
            # Q2
            # Q3
            # stride: lane_id
            # contig: 0
            stride = lane_idx
            contig_vec = 0
        elif ldm_count == (2, 2):
            if transpose:  # operand B
                # Q0 Q1
                # Q2 Q3
                # stride: 01234567 01234567 89ABCDEF 89ABCDEF
                # contig: 00000000 11111111 00000000 11111111
                stride = (lane_idx & 0b111) + ((lane_idx >> 4) << 3)
                contig_vec = (lane_idx & 0b1111) >> 3
            else:  # operand A
                # Q0 Q2
                # Q1 Q3
                # stride: 01234567 89ABCDEF 01234567 89ABCDEF
                # contig: 00000000 00000000 11111111 11111111
                stride = lane_idx & 0b1111
                contig_vec = lane_idx >> 4
        else:
            raise NotImplementedError
        return self(
            stride, contig_vec * self.element_per_acc +
            permute_m_pointer_idx * ldm_count[1] * self.element_per_acc)

    def get_ldm_initial_index_fast(self,
                                   lane_idx: int,
                                   ldm_count: MetaArray[int],
                                   transpose: bool = False,
                                   permute_m_pointer_idx: int = 0):
        """ transpose (not operend A)
        if transpose:
            Q0 Q1
            Q2 Q3
        else:
            Q0 Q2
            Q1 Q3

        common bit seq rules:
        add is slower than xor
        x & 0b100 == 0000 1111 0000 1111
        x & 0b11  == 0123 0123 0123 0123
        x + 8 == x ^ 0b1000 when x < 8
        0000 1111 0000 1111 == (x & 0b111) >> 2 == (x & 0b100) >> 2.
        """
        # if permute_m_pointer_idx > 0:
        #     assert self.interleave == 1   
        #  we support NOW!
        if self.interleave == 4:
            # 0 1 2 3    0 1 2 3
            #  4 5 6 7  4 5 6 7
            if ldm_count[1] == 1:
                assert(permute_m_pointer_idx < 2)
                # Q0
                # Q1
                # Q2
                # Q3
                # stride: 0000 1111 2222 3333 ....
                # contig_vec: 0246 1357 0246 1357 ... = 0246[...] ^ (stride & 1)
                # 0246[...] = (lane_idx & 3) << 1)
                stride = lane_idx >> 2
                contig_vec = ((lane_idx & 3) << 1) ^ (stride & 1) ^ permute_m_pointer_idx           # LINK  modify here
            elif ldm_count == (2, 2):
                if transpose:  # operand B
                    assert(permute_m_pointer_idx == 0)
                    # Q0 Q1
                    # Q2 Q3
                    # stride: 00001111 00001111 22223333 22223333 = 00001111[...] + 0[16] 1[16]
                    # = (lane & 0b111) >> 2
                    # contig: 02461357 13570246  02461357 13570246
                    #  = 0246... + 0000 1111 1111 0000
                    # = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 8) >> 3))
                    # 0000 1111 0000 1111 ^ 0000 0000 1111 1111 = 0000 1111 1111 0000
                    # = 02461357[] + 00000000 88888888 ...
                    # also 0246.... ^ 0000 1111 8888 9999 0000 1111 8888 9999
                    stride = ((lane_idx & 0b111) >> 2) + (lane_idx >> 4 << 1)
                    contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ (
                        (lane_idx & 0b1000) >> 3)
                else:  # operand A
                    # Q0 Q2
                    # Q1 Q3
                    # stride: 0000 1111 2222 3333 0000 1111 2222 3333
                    # contig: 02461357 02461357 13570246 13570246, 0101 1010
                    # also 0246.... ^ 0000 1111 0000 1111 8888 9999 8888 9999

                    stride = (lane_idx & 0b1111) >> 2
                    contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ (
                        (lane_idx & 0b10000) >> 4)
            else:
                assert(permute_m_pointer_idx < 1 and ldm_count == (1, 2))       # no other possible
                # Q0 Q1 Q2 Q3
                # stride: 0000 1111 0000 1111 ...
                # contig: 02461357 13570246 +8 +8
                stride = (lane_idx & 0b111) >> 2
                contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ (
                    (lane_idx & 0b1000) >> 3) + (lane_idx >> 4 << 3)
        elif self.interleave == 2:
            # 0   1
            #  2   3
            #   4   5
            #    6   7
            #  0   1
            # 2   3
            #    4   5
            #   6   7
            if ldm_count[1] == 1:
                assert(permute_m_pointer_idx < 4)
                # Q0
                # Q1
                # Q2
                # Q3
                # stride: 0011 2233 4455 6677 ....  = lane_idx >> 1
                # contig_vec: 04152637 ... = 00112233 + 04040404 = (lane_idx >> 1) & 0b11 + (lane_idx & 1) << 2
                # 0246[...] = (lane_idx & 3) << 1)
                stride = lane_idx >> 1
                contig_vec = ((lane_idx >> 1) & 0b11) ^ ((lane_idx & 1) << 2) ^ permute_m_pointer_idx           # LINK moified here
            elif ldm_count == (2, 2):
                assert(permute_m_pointer_idx < 2)
                if transpose:  # operand B
                    # Q0 Q1
                    # Q2 Q3
                    # stride: 00112233 00112233 44556677 44556677 = ((lane_idx >> 1) & 0b11) + (lane_idx >> 4 << 2)
                    # contig: 04152637 15043726 04152637 15043726 = 04152637[] ^ 00000000 11111111 00000000 11111111 ((lane_idx >> 3) & 1)
                    _00112233 = ((lane_idx >> 1) & 0b11)
                    stride = _00112233 + (lane_idx >> 4 << 2)
                    contig_vec = (_00112233 + ((lane_idx & 1) << 2)) ^ (
                        (lane_idx >> 3) & 1) ^ (permute_m_pointer_idx << 1)                     # LINK modify here
                else:  # operand A
                    # Q0 Q2
                    # Q1 Q3
                    # stride: 00112233 44556677 00112233 44556677 = (lane_idx & 0b1111) >> 1
                    # contig: 04152637 04152637 15043726 15043726
                    # 0[8] 1[8] = lane_idx >> 4
                    stride = (lane_idx & 0b1111) >> 1
                    contig_vec = ((((lane_idx >> 1) & 0b11) +
                                  ((lane_idx & 1) << 2)) ^ (lane_idx >> 4)) ^ (permute_m_pointer_idx << 1)   # LINK modify here
            elif ldm_count == (1, 2):
                assert(permute_m_pointer_idx < 2 and lane_idx < 16)
                stride = (lane_idx & 0b111) >> 1
                contig_vec = ((((lane_idx >> 1) & 0b11) +
                              ((lane_idx & 1) << 2)) ^ (lane_idx >> 3)) ^ (permute_m_pointer_idx << 1)          # LINK add for this
            else:
                assert(permute_m_pointer_idx < 1 and ldm_count == (1, 4))
                # Q0 Q1 Q2 Q3
                # stride: 00112233 00112233 00112233 00112233
                # contig: 04152637 15043726 26370415 37261504 = 04152637 ^ 00000000 11111111 22222222 33333333
                stride = (lane_idx & 0b111) >> 1
                contig_vec = (((lane_idx >> 1) & 0b11) +
                              ((lane_idx & 1) << 2)) ^ (lane_idx >> 3)
        elif self.interleave == 1:
            # 0
            #  1
            #   2
            #    3
            #     4
            #      5
            #       6
            #        7
            if ldm_count[1] == 1:
                # Q0
                # Q1
                # Q2
                # Q3
                # stride: 01234567 89... = lane_idx & 0b111
                # contig_vec: 01234567 01... = stride
                # if permute_m_pointer_idx > 0, contig_vec = 01234567 ^ permute_m_pointer_idx
                # 0246[...] = (lane_idx & 3) << 1)
                stride = lane_idx
                contig_vec = (lane_idx & 0b111) ^ permute_m_pointer_idx
                assert permute_m_pointer_idx < 8
            elif ldm_count == (2, 2):
                if transpose:  # operand B
                    # Q0 Q1
                    # Q2 Q3
                    # stride: 01234567 01234567 89ABCDEF 89ABCDEF = lane_idx & 0b111 + lane_idx >> 4 << 3
                    # contig: 01234567 10325476 01234567 10325476 = stride ^ (0[8] 1[8] 0[8] 1[8])
                    # if permute_m_pointer_idx > 0, contig_vec = stride ^ (permute_m_pointer_idx * 2[8], permute_m_pointer_idx * 2 + 1[8])
                    _01234567 = (lane_idx & 0b111)
                    stride = _01234567 + (lane_idx >> 4 << 3)
                    contig_vec = _01234567 ^ (((lane_idx >> 3) & 1) +
                                              (permute_m_pointer_idx << 1))
                else:  # operand A
                    # Q0 Q2
                    # Q1 Q3
                    # stride: 01234567 89ABCDEF 01234567 89ABCDEF = lane_idx & 0b1111
                    # contig: 01234567 01234567 10325476 10325476 = _01234567 ^ (lane_idx >> 4)
                    # 0[8] 1[8] = lane_idx >> 4
                    _01234567 = (lane_idx & 0b111)
                    stride = lane_idx & 0b1111
                    contig_vec = _01234567 ^ ((lane_idx >> 4) +
                                              (permute_m_pointer_idx << 1))
                assert permute_m_pointer_idx < 4
            elif ldm_count == (1, 2):
                assert(lane_idx < 16 and permute_m_pointer_idx < 4)
                stride = lane_idx & 0b111
                contig_vec = stride ^ ((lane_idx >> 3) ^
                                       (permute_m_pointer_idx << 1))
            else:
                # Q0 Q1 Q2 Q3
                # stride: 01234567 01234567 01234567 01234567
                # contig: 01234567 10325476 23016745 32107654 = 01234567 ^ 00000000 11111111 22222222 33333333
                stride = lane_idx & 0b111
                contig_vec = stride ^ ((lane_idx >> 3) +
                                       (permute_m_pointer_idx << 2))
                assert permute_m_pointer_idx < 2
        else:
            raise NotImplementedError
        return stride, contig_vec

    def get_ldm_initial_offset_fast(self,
                                    lane_idx: int,
                                    ldm_count: MetaArray[int],
                                    transpose: bool = False,
                                    permute_m_pointer_idx: int = 0):
        stride, contig_vec = self.get_ldm_initial_index_fast(
            lane_idx, ldm_count, transpose, permute_m_pointer_idx)
        res = stride * self.static_stride + contig_vec * self.element_per_acc
        ref = self.get_ldm_initial_offset_ref(lane_idx, ldm_count, transpose,
                                              permute_m_pointer_idx)
        if res != ref:
            print(self.sw_shape, self.subsw_shape)
            print(
                f"{lane_idx, ldm_count, transpose, permute_m_pointer_idx, self.interleave, self.is_spec_32}, {res, ref}"
            )
            assert res == ref
        return res

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def get_ldm_initial_offset(self):
        code = pccm.code()
        code.arg("lane_idx,permute_m_pointer_idx", self.index_t)
        code.arg("transpose", "bool")
        # is hard to get ldm count here even if it's a static constexpr variable,
        # so we use template instead.
        code.nontype_targ("LdmCountStride,LdmCountContig", "int")
        code.raw(f"""
        int stride = -1;
        int contig_vec = -1;
        """)
        if self.interleave == 4:
            code.raw(f"""
            if (LdmCountContig == 1){{
                stride = lane_idx >> 2;
                contig_vec = ((lane_idx & 3) << 1) ^ (stride & 1) ^ permute_m_pointer_idx;
            }} else if (LdmCountContig == 2 && LdmCountStride == 2){{
                if (transpose){{
                    stride = ((lane_idx & 0b111) >> 2) + (lane_idx >> 4 << 1);
                    contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 0b1000) >> 3);
                }}else{{
                    stride = (lane_idx & 0b1111) >> 2;
                    contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 0b10000) >> 4);
                }}
            }}else{{
                stride = (lane_idx & 0b111) >> 2;
                contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 0b1000) >> 3) + (lane_idx >> 4 << 3);
            }}
            """)
        elif self.interleave == 2:
            code.raw(f"""
            if (LdmCountContig == 1){{
                stride = lane_idx >> 1;
                contig_vec = ((lane_idx >> 1) & 0b11) ^ ((lane_idx & 1) << 2) ^ permute_m_pointer_idx;
            }} else if (LdmCountContig == 2 && LdmCountStride == 2){{
                if (transpose){{
                    int _00112233 = ((lane_idx >> 1) & 0b11);
                    stride = _00112233 + (lane_idx >> 4 << 2);
                    contig_vec = (_00112233 + ((lane_idx & 1) << 2)) ^ ((lane_idx >> 3) & 1) ^ (permute_m_pointer_idx << 1);
                }}else{{
                    stride = (lane_idx & 0b1111) >> 1;
                    contig_vec = ((((lane_idx >> 1) & 0b11) + ((lane_idx & 1) << 2)) ^ (lane_idx >> 4)) ^ (permute_m_pointer_idx << 1);
                }}
            }}else if (LdmCountContig == 2 && LdmCountStride == 1){{  //incomplete ldm block
                stride = (lane_idx & 0b111) >> 1;
                contig_vec = (((lane_idx >> 1) & 0b11) ^ ((lane_idx & 1) << 2)) ^ (lane_idx >> 3) ^ (permute_m_pointer_idx << 1);
            }} else{{
                stride = (lane_idx & 0b111) >> 1;
                contig_vec = (((lane_idx >> 1) & 0b11) + ((lane_idx & 1) << 2)) ^ (lane_idx >> 3);
            }}
            """)
        elif self.interleave == 1:
            code.raw(f"""
            if (LdmCountContig == 1){{
                stride = lane_idx;
                contig_vec = (lane_idx & 0b111) ^ permute_m_pointer_idx;
            }} else if (LdmCountContig == 2 && LdmCountStride == 2){{
                if (transpose){{
                    int _01234567 = (lane_idx & 0b111);
                    stride = _01234567 + (lane_idx >> 4 << 3);
                    contig_vec = _01234567 ^ (((lane_idx >> 3) & 1) + (permute_m_pointer_idx << 1));
                }}else{{
                    int _01234567 = (lane_idx & 0b111);
                    stride = lane_idx & 0b1111;
                    contig_vec = _01234567 ^ ((lane_idx >> 4) + (permute_m_pointer_idx << 1));
                }}
            }}else if (LdmCountContig == 2 && LdmCountStride == 1){{    // incomplete ldm block
                stride = lane_idx & 0b111;
                contig_vec = stride ^ ((lane_idx >> 3) ^ (permute_m_pointer_idx << 1));
            }}else {{
                stride = lane_idx & 0b111;
                contig_vec = stride ^ ((lane_idx >> 3) + (permute_m_pointer_idx << 2));
            }}
            """)
        else:
            raise NotImplementedError
        code.raw(f"""
        return stride * {self.static_stride} + contig_vec * {self.element_per_acc};
        """)
        return code.ret(self.long_index_t)

    @pccm.cuda.member_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def get_ldm_initial_offset_ref_cpp(self):
        code = pccm.code()
        code.arg("lane_idx,permute_m_pointer_idx", self.index_t)
        code.arg("transpose", "bool")
        # is hard to get ldm count here even if it's a static constexpr variable,
        # so we use template instead.
        code.nontype_targ("LdmCountStride,LdmCountContig", "int")
        code.raw(f"""
        int stride = -1;
        int contig_vec = -1;
        """)
        code.raw(f"""
        if (LdmCountStride == 1){{
            stride = lane_idx & 0b111;
            contig_vec = lane_idx >> 3;
        }} else if (LdmCountContig == 2 && LdmCountStride == 2){{
            if (transpose){{
                stride = (lane_idx & 0b111) + ((lane_idx >> 4) << 3);
                contig_vec = (lane_idx & 0b1111) >> 3;
            }}else{{
                stride = lane_idx & 0b1111;
                contig_vec = lane_idx >> 4;
            }}
        }}else{{
            stride = lane_idx;
            contig_vec = 0;
        }}
        """)
        code.raw(f"""
        return (*this)(stride, contig_vec * {self.element_per_acc} + LdmCountContig * permute_m_pointer_idx * {self.element_per_acc});
        """)
        return code.ret(self.long_index_t)


def _ci_dev_layout_dev():
    msg = ""
    msg_contig = ""
    permute_m_pointer_idx = 0
    for lane_idx in range(32):
        stride = (lane_idx & 0b111) + ((lane_idx >> 4) << 3)
        contig_vec = (lane_idx & 0b1111) >> 3
        msg += f"{stride:01x}"
        msg_contig += f"{contig_vec:01x}"

        if (lane_idx + 1) % 8 == 0:
            msg += " "
            msg_contig += " "
    print(msg)
    print(msg_contig)


class SmemTileIterator(bases.GemmSmemIterator):
    def __init__(self,
                 is_crosswise: bool,
                 dtype: dtypes.DType,
                 tile_shape_km: MetaArray[int],
                 my_layout: MyTensorOpLayout,
                 smem_layout: layout_tensorop.TensorOpMultiplicand,
                 tmap: PitchLinearWarpRaked,
                 alignment: int = -1,
                 crosswise: int = 0,
                 num_stage: int = 2):
        super().__init__(dtype,
                         tmap.iterations.prod() * my_layout.element_per_acc,
                         my_layout.element_per_acc)
        # cultass shape: mk
        # our shape: km
        # A crosswise: [64, 32], congruous: [32, 64]
        # , congruous: [32, 128]
        self.is_sparse_input_distribute = hasattr(tmap, "is_sparse") and tmap.is_sparse   
        if(self.is_sparse_input_distribute):
            self.add_member('is_skipped', 'bool')
        self.transposed_smem = is_crosswise
        self.ref_layout = smem_layout
        # if crosswise, A is mk, B is nk, so k_axis == 1
        # else A is km, B is kn, so k_axis == 0
        self.k_axis = 0 if not is_crosswise else 1
        self.is_crosswise = is_crosswise
        self.smem_layout = my_layout
        self.tile_shape_km = tile_shape_km
        self.num_stage = num_stage
        self.smem_vis_shape = [tile_shape_km[0] * num_stage, tile_shape_km[1]]
        self.layout = my_layout
        ss = tile_shape_km[1] * self.smem_layout.interleave
        self.smem_vis_shape = [
            tile_shape_km[0] * num_stage * tile_shape_km[1] // ss, ss
        ]
        self.stride_with_stage = self.smem_layout.base_stride * self.smem_layout.stage_count[
            1]
        self.static_stride_vec = self.smem_layout.static_stride_vec

        self.tmap = tmap
        if alignment == -1:
            alignment = dtype.bitsize() * tmap.element_per_acc // 8
        self.alignment = alignment
        self.add_param_class("tmap", tmap, "ThreadMap")
        self.crosswise = crosswise

        self.add_param_class("layout", self.layout, "Layout")
        # if is_crosswise:
        #     print(self.pointer_count, self.my_layout)
        self.iterations = tmap.iterations  # same as tmap iterations
        if self.smem_layout.interleave != 1:
            self.interleaved_delta = metaseq(
                tmap.delta[0] // self.smem_layout.interleave,
                tmap.delta[1] * self.smem_layout.interleave)
        else:
            self.interleaved_delta = tmap.delta

        # num pointers:
        self.pointer_count = self.smem_layout.num_smem_pointers
        # print(self.smem_layout.interleave, self.static_stride, self.pointer_count, self.stage_axis)

        # self.add_member("stride_", self.index_t)
        self.add_member("pointer_",
                        self.access_pointer,
                        array=f"[{self.pointer_count}]")

        self.add_member("byte_offset_", self.index_t)
        # self.add_member("iteration_contiguous_, iteration_strided_", "int")
        # cudasim members
        self.pointer_: List[ArrayPtr] = [None] * self.pointer_count
        self.byte_offset_ = 0

    def get_smem_vis_shape(self) -> MetaArray[int]:
        return seq(self.smem_vis_shape[0], self.smem_vis_shape[1])

    def __repr__(self):
        return f"SmemIter[id={self.interleaved_delta}|it={self.iterations}]"

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        # TODO remove this argument
        code.arg("stride", "int")
        code.arg("ptr", self.pointer)
        code.arg("thread_id", "int")
        code.ctor_init("byte_offset_", "0")

        if self.is_sparse_input_distribute:
            code.raw('''
                is_skipped = ThreadMap::is_skipped(thread_id);
                if(is_skipped)
                    return;
            ''')

        code.raw(f"""
        auto thread_offset_base = ThreadMap::initial_offset(thread_id);
        auto layout = Layout();
        TV_PRAGMA_UNROLL
        for (int i = 0; i < {self.pointer_count}; ++i) {{
            pointer_[i] = reinterpret_cast<{self.access_pointer}>(
                ptr + layout(thread_offset_base[0] + i * {self.tmap.warp_shape[0]},
                            thread_offset_base[1]));
        }}
        """)
        return code

    def python_ctor(self, stride: int, ptr: ArrayPtr, thread_id: int):
        new_obj = SmemTileIterator(self.is_crosswise, self.dtype,
                                   self.tile_shape_km, self.smem_layout,
                                   self.ref_layout, self.tmap, self.alignment,
                                   self.crosswise, self.num_stage)
        l = new_obj.layout.python_ctor(self.stride_with_stage)
        thread_offset_base = new_obj.tmap.initial_offset_python(thread_id)
        refl = new_obj.ref_layout.python_ctor(new_obj.stride_with_stage)

        # for smem store iters, only stride axis may insufficient
        for i in range(new_obj.pointer_count):
            off = l(thread_offset_base[0] + i * new_obj.tmap.warp_shape[0],
                    thread_offset_base[1])
            # refoff = refl(
            #     thread_offset_base[0] + i * new_obj.tmap.warp_shape[0],
            #     thread_offset_base[1])
            # if refoff != off:
            #     print(off, refoff)
            new_obj.pointer_[i] = (ptr + off).change_access_size(
                new_obj.element_per_acc)
        if self.is_sparse_input_distribute:
            assert(new_obj.is_sparse_input_distribute)
            new_obj.is_skipped = new_obj.tmap.is_skipped_python(thread_id)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True, const=True)
    def get(self):
        code = pccm.FunctionCode(f"""
        {self.access_pointer} access_ptr = pointer_[s & {self.pointer_count - 1}];
        int external_stride_idx = (s & ~{self.pointer_count - 1});
        int access_offset = (external_stride_idx * {self.interleaved_delta[0]} *
                         {self.static_stride_vec} + c * {self.interleaved_delta[1] // self.element_per_acc});
        
        char *access_byte_ptr =
            reinterpret_cast<char *>(access_ptr + access_offset);
        return reinterpret_cast<{self.access_pointer}>(access_byte_ptr + byte_offset_);
        """).arg("s, c", "int")
        return code.ret(f"{self.access_pointer}")

    def get_python(self, s: int, c: int):
        access_ptr = self.pointer_[s & (self.pointer_count - 1)]
        # for other ptrs in same swizzle part,
        # we have already added the stride offset, so we remove
        # stride offset for all subsequence ptrs.
        external_stride_idx = s & (~((self.pointer_count - 1)))
        access_offset = (external_stride_idx * self.interleaved_delta[0] *
                         self.static_stride_vec +
                         c * self.interleaved_delta[1] // self.element_per_acc)
        ptr = ((access_ptr + access_offset).change_access_byte_size(1) +
               self.byte_offset_)
        return ptr.change_access_size(self.element_per_acc)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(
            f"byte_offset_ += offset * sizeof({self.dtype});")
        code.arg("offset", self.long_index_t)
        return code

    def add_pointer_offset_python(self, offset: int):
        self.byte_offset_ += offset * self.dtype.itemsize()

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_tile_offset(self):
        code = pccm.FunctionCode(f"""
        add_pointer_offset(c * {self.layout.static_stride // self.layout.stage_count[1]} +
            s * {self.tile_shape_km[0] * self.tile_shape_km[1] *
            self.layout.stage_count[1]});
        """).arg("s, c", "int")
        return code

    def add_tile_offset_python(self, s: int, c: int):
        # print(s, c, self.stage_count[1], self.tile_shape_km[0], self.tile_shape_km[1])
        self.add_pointer_offset_python(
            c * self.layout.static_stride // self.layout.stage_count[1] +
            s * self.tile_shape_km[0] * self.tile_shape_km[1] *
            self.layout.stage_count[1])

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        if self.layout.stage_axis == 1:
            code = pccm.FunctionCode(f"""
            add_tile_offset(0, num_tile);
            """)
        else:
            code = pccm.FunctionCode(f"""
            add_tile_offset(num_tile, 0);
            """)

        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        if self.layout.stage_axis == 1:
            self.add_tile_offset_python(0, num)
        else:
            self.add_tile_offset_python(num, 0)

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        store_with_byte_offset(frag, pointer_offset * {self.dtype.bitsize()} / 8);
        """)
        code.arg("frag",
                 f"{self.fragment_t} const&").arg("pointer_offset",
                                                  str(self.index_t))
        return code

    async def store_with_pointer_offset_python(self, frag: ArrayPtr,
                                               offset: int):
        return await self.store_with_byte_offset_python(
            frag, offset * self.dtype.itemsize())

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_with_byte_offset(self):
        code = pccm.FunctionCode()
        if self.is_sparse_input_distribute:
            code.raw(f"""
                if(is_skipped)
                    return;
            """)
        code.raw(f"""
        {self.const_access_pointer} frag_ptr = reinterpret_cast<{self.const_access_pointer}>(&frag);

        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.tmap.iterations[0]}; ++s) {{
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.tmap.iterations[1]}; ++c) {{
                int access_idx = c + s * {self.tmap.iterations[1]};
                char *byte_ptr = reinterpret_cast<char *>(get(s, c)) + byte_offset;
                {self.access_pointer} access_ptr = reinterpret_cast<{self.access_pointer}>(byte_ptr);
                *access_ptr = frag_ptr[access_idx];
            }}
        }}
        """)

        code.arg("frag",
                 f"{self.fragment_t} const&").arg("byte_offset",
                                                  str(self.index_t))
        return code
    

    def can_multistage_load(self):
        return True
    
    def enumurate_get_param(self, python = False):
        contig = 1
        strided = 0
        for s in range(self.tmap.iterations[strided]):
            for c in range(self.tmap.iterations[contig]):
                if python:
                    yield (s, c)
                else:
                    yield f"{s}, {c}"
    
    async def store_ptr_with_param_python(self, s, c): # ret (ArrayPtr, bool: valid, ptr_addrs)
        ptr_addrs = np.zeros((self.tmap.element_per_acc,), dtype=np.int32)
        if self.is_sparse_input_distribute and self.is_skipped:
            return (self.get_python(0, 0), False, ptr_addrs)
        access_ptr = self.get_python(s, c).change_access_size(self.element_per_acc)
        ptr_addrs[:] = np.arange(access_ptr.offset, access_ptr.offset + self.element_per_acc)
        valid = True
        await checkers.smem_bank_conflicit_check(access_ptr, 0)
        assert access_ptr.length > 0
        return (access_ptr, valid, ptr_addrs)
    
    @pccm.cuda.member_function(device=True, forceinline=True)
    def store_ptr_with_param(self):
        code = pccm.FunctionCode()
        code.arg("s, c", "int")
        code.arg("valid_ref", "bool&")
        code.ret(f"{self.access_pointer}")
        if self.is_sparse_input_distribute:
            code.raw("""
                if(is_skipped){
                    valid_ref = false;
                    return nullptr;
                }
            """)
        code.raw(f"""
            return reinterpret_cast<{self.access_pointer}>(get(s, c));
        """)
        return code

    async def store_with_byte_offset_python(self, frag: ArrayPtr,
                                            byte_offset: int):
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)
        if self.is_sparse_input_distribute and self.is_skipped:
            return ptr_addrs
        frag_ptr = frag.change_access_size(self.element_per_acc)
        for s in range(self.tmap.iterations[0]):
            for c in range(self.tmap.iterations[1]):
                access_idx = c + s * self.tmap.iterations[1]
                byte_ptr = self.get_python(
                    s, c).change_access_byte_size(1) + byte_offset
                access_ptr = byte_ptr.change_access_size(self.element_per_acc)
                await checkers.smem_bank_conflicit_check(access_ptr, 0)
                if access_ptr.length <= 0:
                    continue

                access_ptr[0] = frag_ptr[access_idx]
                ptr_addrs[access_idx * frag_ptr.access_size:(access_idx + 1) *
                          frag_ptr.access_size] = np.arange(
                              access_ptr.offset,
                              access_ptr.offset + frag_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def store(self):
        code = pccm.FunctionCode(f"""
        store_with_pointer_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t} const&")
        return code

    async def store_python(self, frag: ArrayPtr):
        return await self.store_with_pointer_offset_python(frag, 0)

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        if self.layout.stage_axis == 1:
            code = pccm.FunctionCode(f"""
            add_tile_offset(0, 1);
            return *this;
            """)
        else:
            code = pccm.FunctionCode(f"""
            add_tile_offset(1, 0);
            return *this;
            """)

        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        if self.layout.stage_axis == 1:
            self.add_tile_offset_python(0, 1)
            return self
        else:
            self.add_tile_offset_python(1, 0)
            return self


class WarpIteratorCrosswise(bases.GemmWarpIterator):
    """PermuteK: the smem permute is done in k axis,
    which means when warp gemm increment (k increment), next location isn't trivial, 
    we need to handle it carefully. 

    Layout: LDM group -> WarpTileK
    """
    def __init__(self, dtype: dtypes.DType, tile_shape_km: MetaArray[int],
                 my_layout: MyTensorOpLayout,
                 smem_layout: layout_tensorop.TensorOpMultiplicand,
                 warp_tile_shape_km: MetaArray[int], operand_a: bool,
                 inst_shape_km: MetaArray[int], mma_inst_delta: int,
                 partk: int):
        self.threads = 32
        tile_shape_mk = tile_shape_km[::-1]
        inst_shape_mk = inst_shape_km[::-1]
        warp_tile_shape_mk = warp_tile_shape_km[::-1]
        element_count = warp_tile_shape_mk[0] * inst_shape_mk[1] // self.threads

        super().__init__(dtype, element_count, my_layout.element_per_acc)
        self.tile_shape_km = tile_shape_km
        self.inst_shape_km = inst_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km
        self.warp_tile_shape_mk = warp_tile_shape_mk
        self.inst_shape_mk = inst_shape_mk
        self.tile_shape_mk = tile_shape_mk

        self.ref_layout = smem_layout
        self.layout = my_layout
        self.num_warp_gemm_iters = warp_tile_shape_mk[1] // inst_shape_mk[1]
        self.stride_with_stage = self.layout.base_stride * self.layout.stage_count[
            1]
        self.add_param_class("layout", self.layout, "TensorOpLayout")

        self.operand_a = operand_a
        self.mma_inst_delta = mma_inst_delta
        self.partk = partk
        self.crosswise = smem_layout.crosswise
        self.static_stride_vec = my_layout.static_stride // my_layout.element_per_acc
        assert self.layout.element_per_acc * self.dtype.itemsize(
        ) == arch.ldmatrix.LdMatrix.LineByteSize
        self.lds_op_outer = self.layout.element_per_acc
        self.ldm_line_size = self.layout.element_per_acc
        self.ldm_num_line = arch.ldmatrix.LdMatrix.NumLinePerMatrix
        self.lds_op_inner = arch.ldmatrix.LdMatrix.NumLinePerMatrix
        assert warp_tile_shape_mk[0] % self.lds_op_outer == 0
        assert warp_tile_shape_mk[1] % self.lds_op_inner == 0
        # ldm_count: number of sub matrix in ldmatrix inst
        # inst shape is mk ([16, 8]), so we need to load [16, 8]
        # so self.lds_op_outer * sizeof(T) must be 8 x f16?
        self.ldm_count = seq(1, inst_shape_mk[1] // self.ldm_line_size)
        self.ldm_count[0] = arch.ldmatrix.LdMatrix.MaxNum // self.ldm_count[1]
        if (self.ldm_count[0] * self.ldm_num_line) > warp_tile_shape_mk[0]:
            # ldm too many lines (at most 32 line), just use warp_tile_shape_km[0] // self.ldm_num_line
            self.ldm_count[0] = warp_tile_shape_mk[0] // self.ldm_num_line

        self.ldm_iters = seq(
            warp_tile_shape_mk[0] // self.ldm_num_line // self.ldm_count[0], 1)
        # number of k per tile?
        self.k_groups_per_tile = self.layout.sw_shape[
            1] // self.layout.interleave // self.ldm_count[1]
        self.k_group_inc_mask = 0
        if self.k_groups_per_tile // self.partk > 1:
            # k_group_inc_mask: for maximum swizzle length limit
            self.k_group_inc_mask = (1 << int(
                np.log2(self.k_groups_per_tile // self.partk)) - 1) - 1
        # k inc width: when warp gemm increment, the length of byte increment (when no swizzle)
        self.k_inc_byte_width = self.ldm_count[1] * self.dtype.bitsize(
        ) * self.layout.element_per_acc // 8
        # self.add_member("sections_", "int")
        # self.add_member("stride_", self.index_t)
        self.add_member("pointer_", self.const_access_pointer)
        # self.add_member("pointer_bkp_", self.const_access_pointer)

        self.add_member("byte_offset_", self.index_t)
        self.add_member("wmma_k_index_", "int")

        self.ldmatrix = arch.ldmatrix.LdMatrix(True, self.ldm_count.prod())
        self.add_param_class("ldsm", self.ldmatrix, "LdMatrix")
        # cudasim members
        self.pointer_: ArrayPtr = None
        self.byte_offset_ = -1
        self.wmma_k_index_ = -1

    def __repr__(self):
        return (f"WarpIterPK[ldss={self.ldm_count}|"
                f"ldsi={self.ldm_iters}|g={self.k_groups_per_tile}]")

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_idx_k, warp_idx_mn, lane_idx", "int")
        code.ctor_init("pointer_",
                       f"reinterpret_cast<{self.const_access_pointer}>(ptr)")
        # code.ctor_init("pointer_bkp_",
        #                f"reinterpret_cast<{self.const_access_pointer}>(ptr)")
        code.ctor_init("wmma_k_index_", "0")
        code.ctor_init("byte_offset_", "0")
        code.raw(f"""
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
            lane_idx = lane_idx % ({self.ldm_count.prod()} * {self.ldm_num_line});
        #endif
        int offset_e = TensorOpLayout::get_ldm_initial_offset<{self.ldm_count[0]}, {self.ldm_count[1]}>(
            lane_idx, 0, {pccm.boolean(not self.operand_a)});
        byte_offset_ = offset_e * {self.dtype.bitsize()} / 8;
        add_tile_offset({self.num_warp_gemm_iters} * warp_idx_k, warp_idx_mn);
        """)
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_mn: int, lane_idx: int):
        new_obj = WarpIteratorCrosswise(self.dtype, self.tile_shape_km,
                                        self.layout, self.ref_layout,
                                        self.warp_tile_shape_km,
                                        self.operand_a, self.inst_shape_km,
                                        self.mma_inst_delta, self.partk)
        new_obj.pointer_ = ptr.change_access_size(new_obj.element_per_acc)
        new_obj.wmma_k_index_ = 0
        new_obj.byte_offset_ = 0
        layout = new_obj.layout.python_ctor(new_obj.layout.static_stride //
                                            new_obj.layout.stage_count[1])
        # turing ldmatrix don't support invalid addr.
        # the number of lane needed is ldmatrix.count * 8
        lane_idx = lane_idx % (new_obj.ldm_count.prod() * new_obj.lds_op_inner)
        ref_offset = layout.get_ldm_initial_offset_fast(
            lane_idx, self.ldm_count, not self.operand_a)
        new_obj.byte_offset_ = (ref_offset * new_obj.dtype.bitsize() // 8)
        new_obj.add_tile_offset_python(
            new_obj.num_warp_gemm_iters * warp_idx_k, warp_idx_mn)
        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_tile_offset(self):
        interleaved_warp_tile_stride = self.warp_tile_shape_mk[
            0] // self.layout.interleave
        code = pccm.FunctionCode(f"""
        int mn_offset = warp_idx_mn;
        int k_offset = warp_idx_k;
        int sw_part_idx = k_offset / {self.k_groups_per_tile};
        int idx_in_sw_part = k_offset % {self.k_groups_per_tile};

        byte_offset_ ^= (idx_in_sw_part * {self.k_inc_byte_width});
        // tv::printf2_block_once(threadIdx.x, "premuteK", byte_offset_);

        pointer_ +=
            mn_offset * {interleaved_warp_tile_stride * self.static_stride_vec} +
            sw_part_idx * {self.layout.sw_shape[1]};
        """)
        return code.arg("warp_idx_k, warp_idx_mn", "int")

    def add_tile_offset_python(self, warp_idx_k: int, warp_idx_mn: int):
        mn_offset = warp_idx_mn
        k_offset = warp_idx_k
        sw_part_idx = k_offset // self.k_groups_per_tile
        idx_in_sw_part = k_offset % self.k_groups_per_tile
        # swizzle inside a swizzle part
        self.byte_offset_ ^= int(idx_in_sw_part * self.k_inc_byte_width)
        interleaved_warp_tile_stride = self.warp_tile_shape_mk[
            0] // self.layout.interleave
        self.pointer_ += (
            mn_offset * interleaved_warp_tile_stride * self.static_stride_vec +
            sw_part_idx * self.layout.sw_shape[1])

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        add_tile_offset(num_tile, 0);
        """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        return self.add_tile_offset_python(num, 0)

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        num_k_inc = (self.k_groups_per_tile // self.partk)
        code = pccm.code()

        if num_k_inc > 1:
            assert count_set_bits(
                self.k_inc_byte_width) == 1, "width must be power of 2"
            # precedence of c++ bit operators is lower than logical operators.
            # so we need brackets here, python code don't need them.
            code.raw(f"""
            if (((wmma_k_index_ & {self.k_group_inc_mask}) & 1) == 0){{
                // bit 0 advance
                byte_offset_ ^= 0b1 * {self.k_inc_byte_width};
            }}
            else if ((wmma_k_index_ & {self.k_group_inc_mask}) == 0b1){{
                // bit 1 advance
                byte_offset_ ^= 0b11 * {self.k_inc_byte_width};
            }}
            else if ((wmma_k_index_ & {self.k_group_inc_mask}) == 0b11){{
                // bit 2 advance
                byte_offset_ ^= 0b111 * {self.k_inc_byte_width};
            }}
            """)
        code.raw(f"""
        wmma_k_index_++;
        if (wmma_k_index_ == {num_k_inc}) {{
            wmma_k_index_ = 0;
            // k group increment
            add_tile_offset({self.k_groups_per_tile}, 0);
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        """warps in k handle divided swizzle tile, then jump to next swizzle tile if exists.
        --swizzle0-- --swizzle1--
        w0 w1 w2 w3  w0 w1 w2 w3
        """
        num_k_inc = (self.k_groups_per_tile // self.partk)
        if (num_k_inc > 1):
            # mask: largest number of bit during increment
            mask = self.k_group_inc_mask

            self.byte_offset_ ^= layout_tensorop.swizzle_increment(
                self.wmma_k_index_ & mask, self.k_inc_byte_width)
        self.wmma_k_index_ += 1
        if (self.wmma_k_index_ == num_k_inc):
            self.wmma_k_index_ = 0
            # increase sw part
            self.add_tile_offset_python(self.k_groups_per_tile, 0)
        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_byte_offset(self):
        code = pccm.FunctionCode(f"""
        tv::array<unsigned, {self.ldm_count.prod()}> *fetch_ptr =
            reinterpret_cast<tv::array<unsigned, {self.ldm_count.prod()}> *>(&frag);
        TV_PRAGMA_UNROLL
        for (int s = 0; s < {self.ldm_iters[0]}; ++s) {{
            TV_PRAGMA_UNROLL
            for (int c = 0; c < {self.ldm_iters[1]}; ++c) {{
                int access_idx = c + s * {self.ldm_iters[1]};
                {self.const_access_pointer} source_ptr =
                    pointer_ + {self.ldm_count[1]} * c +
                    {self.ldm_num_line} * {self.ldm_count[0]} * s * 
                    {self.static_stride_vec // self.layout.interleave};
                char const *source_byte_ptr =
                    reinterpret_cast<char const *>(source_ptr) + byte_offset +
                    byte_offset_;
                LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
            }}
        }}
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("byte_offset",
                                                    str(self.index_t))
        return code

    async def load_with_byte_offset_python(self, frag: ArrayPtr,
                                           byte_offset: int):
        fetch_ptr = frag.change_access_byte_size(self.ldm_count.prod() * 4)
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)

        for s in range(self.ldm_iters[0]):
            for c in range(self.ldm_iters[1]):
                access_idx = c + s * self.ldm_iters[1]

                source_ptr = (self.pointer_ + self.ldm_count[1] * c +
                              self.ldm_num_line // self.layout.interleave *
                              self.ldm_count[0] * s * self.static_stride_vec
                              ).change_access_size(self.element_per_acc)

                source_byte_ptr = source_ptr.change_access_byte_size(
                    1) + byte_offset + self.byte_offset_
                await checkers.smem_bank_conflicit_check(fetch_ptr, access_idx)
                await self.ldmatrix(fetch_ptr[access_idx], source_byte_ptr)
                ptr_addrs[access_idx * fetch_ptr.access_size:(access_idx + 1) *
                          fetch_ptr.access_size] = np.arange(
                              source_byte_ptr.offset,
                              source_byte_ptr.offset + fetch_ptr.access_size)
        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, pointer_offset * sizeof({self.dtype}));
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_kgroup_index(self):
        code = pccm.FunctionCode(
            f"wmma_k_index_ = wmma_k % ({self.k_groups_per_tile // self.partk});"
        )
        code.arg("wmma_k", "int")
        return code

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_byte_offset_python(frag, 0)

    def set_wmma_k_index_python(self, wmma_k):
        self.wmma_k_index_ = wmma_k % (self.k_groups_per_tile // self.partk)


class WarpIteratorCongruous(bases.GemmWarpIterator):
    def __init__(self, dtype: dtypes.DType, tile_shape_km: MetaArray[int],
                 my_layout: MyTensorOpLayout,
                 smem_layout: layout_tensorop.TensorOpMultiplicand,
                 warp_tile_shape_km: MetaArray[int], operand_a: bool,
                 inst_shape_km: MetaArray[int], mma_inst_delta: int,
                 partk: int):
        self.threads = 32

        self.is_spec_32 = dtype.bitsize() == 32
        element_count = warp_tile_shape_km[1] * inst_shape_km[0] // self.threads

        if self.is_spec_32:
            super().__init__(dtype, element_count, 1)
        else:
            super().__init__(dtype, element_count, my_layout.element_per_acc)
        # cultass shape: mk
        # our shape: km
        self.ref_layout = smem_layout
        self.my_layout = my_layout
        self.num_warp_gemm_iters = warp_tile_shape_km[0] // inst_shape_km[0]
        self.tile_shape_km = tile_shape_km
        self.warp_tile_shape_km = warp_tile_shape_km
        self.operand_a = operand_a
        self.inst_shape_km = inst_shape_km

        self.mma_inst_delta = mma_inst_delta
        self.partk = partk

        self.layout = my_layout
        self.static_stride_vec = my_layout.static_stride // my_layout.element_per_acc
        self.static_stride = my_layout.static_stride
        self.ldm_line_size = self.layout.element_per_acc
        self.ldm_num_line = arch.ldmatrix.LdMatrix.NumLinePerMatrix
        self.add_param_class("layout", self.layout, "TensorOpLayout")

        if self.is_spec_32:
            # 32bit mma (tf32) don't support ldmatrix.trans.

            self.ldm_num_line = self.layout.sw_shape[0]
            self.ldm_line_size = self.threads // self.ldm_num_line
            self.ldm_count = metaseq(inst_shape_km[0] // self.ldm_num_line,
                                     inst_shape_km[1] // self.ldm_line_size)
            # tf32 iter can't use element per acc > 1 because it need transposed load.
            # so we need more pointers here.
            self.pointer_count = self.layout.sw_shape[
                1] * self.layout.element_per_acc // self.ldm_line_size
            self.ldm_iters = metaseq(
                1, warp_tile_shape_km[1] // self.ldm_line_size //
                self.ldm_count[1])
        else:

            self.lds_op_outer = self.layout.element_per_acc
            self.lds_op_inner = 8
            self.ldm_count = metaseq(inst_shape_km[0] // self.ldm_num_line, 1)
            self.ldm_count[
                1] = min(arch.ldmatrix.LdMatrix.MaxNum // self.ldm_count[0], 
                warp_tile_shape_km[1] // self.lds_op_outer)
            self.ldm_iters = metaseq(
                1, warp_tile_shape_km[1] // self.layout.element_per_acc //
                self.ldm_count[1])
            self.pointer_count = self.layout.sw_shape[1] // self.layout.interleave // self.ldm_count[1]
        assert warp_tile_shape_km[1] % self.ldm_line_size == 0
        assert warp_tile_shape_km[0] % self.ldm_num_line == 0

        self.k_groups_per_tile = self.warp_tile_shape_km[0] // inst_shape_km[
            0]  # type: int
        # print(self.layout.tile_shape)
        # print(self.class_name, self.ldm_count, self.ldm_iters,
        #       self.k_groups_per_tile)

        self.real_pointer_count = self.pointer_count
        self.minimize_regisiter_optimize = False
        if self.is_spec_32 and self.pointer_count > self.ldm_iters[1] * self.ldm_count[1]:
            self.pointer_count = self.ldm_iters[1] * self.ldm_count[1]
            self.minimize_regisiter_optimize = True
            self.add_member("ref_pointer_", self.pointer + "&")
            self.add_member("lane_id_", "const int&", "0")
        
        if self.pointer_count > self.ldm_iters[1] and not self.is_spec_32:       # to much pointer,  we need to do something to reduce register cnt
            self.pointer_count = min(self.pointer_count, self.ldm_iters[1])
            self.minimize_regisiter_optimize = True
            self.add_member("ref_pointer_", self.pointer + "&")
            self.add_member("lane_id_", "const int&", "0")

        self.add_member("wmma_k_index_", "int")
        self.add_member("pointer_",
                        self.const_access_pointer,
                        array=f"[{self.pointer_count}]")
        self.add_member("byte_offset_", self.index_t)
        self.ldmatrix = arch.ldmatrix.LdMatrix(False, self.ldm_count.prod())

        self.add_param_class("ldsm", self.ldmatrix, "LdMatrix")

        # cudasim members
        self.pointer_: List[ArrayPtr] = [None] * self.pointer_count
        self.byte_offset_ = -1
        self.wmma_k_index_ = -1

    def __repr__(self):
        return (f"WarpIterPM[ldss={self.ldm_count}|"
                f"ldsi={self.ldm_iters}|g={self.k_groups_per_tile}]")

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.code()
        code.arg("ptr", self.pointer)
        code.arg("warp_idx_k, warp_idx_mn, lane_idx", "int")
        code.ctor_init("wmma_k_index_", "0")
        code.ctor_init("byte_offset_", "0")
        if not self.is_spec_32:
            code.raw(f"lane_idx %= {self.ldm_count.prod() * 8};")
        if self.minimize_regisiter_optimize:
            code.ctor_init("ref_pointer_", "ptr")
            code.ctor_init("lane_id_", "lane_idx")
        if self.minimize_regisiter_optimize:
            code.raw(f"""
                add_tile_offset({self.num_warp_gemm_iters} * warp_idx_k, warp_idx_mn, true);
            """)
            return code
        if not self.is_spec_32:
            code.raw(f"""
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.pointer_count}; ++i) {{
                int offset = TensorOpLayout::get_ldm_initial_offset<{self.ldm_count[0]}, {self.ldm_count[1]}>(
                    lane_idx, i, {pccm.boolean(not self.operand_a)});
                pointer_[i] = reinterpret_cast<{self.const_access_pointer} >(ptr + offset);
            }}
            add_tile_offset({self.num_warp_gemm_iters} * warp_idx_k, warp_idx_mn);
            """)
        else:
            code.raw(f"""
            for (int i = 0; i < {self.pointer_count}; ++i) {{
                int access_strided = lane_idx % {self.ldm_num_line};
                int access_contiguous = (lane_idx / {self.ldm_num_line}) +
                                        (access_strided ^ i) * {self.ldm_line_size};
                pointer_[i] = reinterpret_cast<{self.const_access_pointer} >(ptr) +
                                access_contiguous + access_strided * {self.static_stride};
            }}
            add_tile_offset({self.num_warp_gemm_iters} * warp_idx_k, warp_idx_mn);
            """)
        return code

    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_mn: int, lane_idx: int):
        new_obj = WarpIteratorCongruous(self.dtype, self.tile_shape_km,
                                        self.my_layout, self.ref_layout,
                                        self.warp_tile_shape_km,
                                        self.operand_a, self.inst_shape_km,
                                        self.mma_inst_delta, self.partk)
        new_obj.wmma_k_index_ = 0
        new_obj.byte_offset_ = 0
        if not self.is_spec_32:
            lane_idx %= self.ldm_count.prod() * 8
        if self.minimize_regisiter_optimize:
            new_obj.store_ptr = ptr
            new_obj.lane_id = lane_idx
        else:
            if not self.is_spec_32:
                layout = new_obj.layout.python_ctor(new_obj.layout.static_stride)
                for i in range(self.pointer_count):
                    ref_offset = layout.get_ldm_initial_offset_fast(
                        lane_idx,
                        self.ldm_count,
                        self.operand_a,
                        permute_m_pointer_idx=i)
                    new_obj.pointer_[i] = (
                        ptr.change_access_size(self.element_per_acc) +
                        ref_offset // self.element_per_acc)
            else:
                for i in range(self.pointer_count):
                    access_strided = lane_idx % self.ldm_num_line
                    access_contiguous = lane_idx // self.ldm_num_line + (
                        access_strided ^ i) * self.ldm_line_size
                    new_obj.pointer_[i] = (
                        ptr.change_access_size(self.element_per_acc) +
                        access_contiguous +
                        access_strided * self.static_stride)

        new_obj.add_tile_offset_python(self.num_warp_gemm_iters * warp_idx_k,
                                       warp_idx_mn, self.minimize_regisiter_optimize)

        return new_obj

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_pointer_offset(self):
        code = pccm.FunctionCode(
            f"byte_offset_ += offset * sizeof({self.dtype});")
        code.arg("offset", self.long_index_t)
        return code

    def add_pointer_offset_python(self, offset: int):
        self.byte_offset_ += offset * self.dtype.itemsize()

    @pccm.cuda.member_function(device=True, forceinline=True)
    def add_tile_offset(self):
        code = pccm.code()
        code.raw(f"""
            constexpr int kContigEqual = {self.layout.warp_bfa_access_shape[0] * (16 // self.dtype.itemsize())};

        int mn_offset = warp_idx_mn;
        int k_offset = warp_idx_k;

        if ({self.warp_tile_shape_km[1]} < kContigEqual || force_update) {{
          constexpr int kwarp_per_crosswise = {self.layout.warp_bfa_access_shape[0] * (16 // self.dtype.itemsize()) // self.warp_tile_shape_km[1]};
          int warp_offset = warp_idx_mn & (kwarp_per_crosswise - 1);
          mn_offset = warp_idx_mn ^ warp_offset;            // kwarp_per_crosswise is 2^a
          warp_offset *= {self.real_pointer_count} / kwarp_per_crosswise;
          if (warp_offset || force_update) {{
        """)
        if self.minimize_regisiter_optimize:
            if not self.is_spec_32:
                code.raw(f"""
                TV_PRAGMA_UNROLL
                for (int i = 0; i < {self.pointer_count}; ++i) {{
                    int offset = TensorOpLayout::get_ldm_initial_offset<{self.ldm_count[0]}, {self.ldm_count[1]}>(
                        lane_id_ % {self.ldm_count.prod() * 8}, (i + warp_offset) % {self.real_pointer_count}, {pccm.boolean(not self.operand_a)});
                    pointer_[i] = reinterpret_cast<{self.const_access_pointer} >(ref_pointer_ + offset);
                }}
                """)
            else:
                code.raw(f"""
                TV_PRAGMA_UNROLL
                for (int i = 0; i < {self.pointer_count}; ++i) {{
                    int access_strided = lane_id_ % {self.ldm_num_line};
                    int access_contiguous = (lane_id_ / {self.ldm_num_line}) +
                                            (access_strided ^ ((i + warp_offset) % {self.real_pointer_count})) * {self.ldm_line_size};
                    pointer_[i] = reinterpret_cast<{self.const_access_pointer} >(ref_pointer_) +
                                    access_contiguous + access_strided * {self.static_stride};
                }}""")
        else:
            code.raw(f"""
            {self.const_access_pointer} buffer[{self.pointer_count}];
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.pointer_count}; ++i) 
              buffer[i] = pointer_[(i + warp_offset) % {self.pointer_count}];
            
            TV_PRAGMA_UNROLL
            for (int i = 0; i < {self.pointer_count}; ++i) 
              pointer_[i] = buffer[i];""")
        
        code.raw(f"""
          }}
        }}
        
        int offset = (k_offset * {self.ldm_count[0] * self.ldm_num_line *
                                    self.static_stride_vec *
                                    self.layout.element_per_acc // self.layout.interleave} +
                    mn_offset * {self.warp_tile_shape_km[1]});
        add_pointer_offset(offset);
        """)
        return code.arg("warp_idx_k, warp_idx_mn", "int").arg("force_update", "bool", "false")

    def add_tile_offset_python(self, warp_idx_k: int, warp_idx_mn: int, force_update: bool = False):
        mn_offset = warp_idx_mn
        k_offset = warp_idx_k
        # two warp handle one swizzle part.
        # for second warp, we need to swap pointers (swizzle part for second warp)
        if (self.warp_tile_shape_km[1] < self.layout.warp_bfa_access_shape[0] * (16 // self.dtype.itemsize()) or force_update):        # warp_shape < crosswise
            kwarp_per_crosswise = self.layout.warp_bfa_access_shape[0] * (16 // self.dtype.itemsize()) // self.warp_tile_shape_km[1]
            warp_offset = warp_idx_mn % kwarp_per_crosswise
            mn_offset = warp_idx_mn - warp_offset

            pointer_offset = warp_offset * (self.real_pointer_count // kwarp_per_crosswise) #  NOTE: fix once
            if self.minimize_regisiter_optimize and (pointer_offset or force_update):
                if self.is_spec_32:
                    for i in range(self.pointer_count):
                        access_strided = self.lane_id % self.ldm_num_line
                        access_contiguous = self.lane_id // self.ldm_num_line + (
                            access_strided ^ ((i + pointer_offset) % self.real_pointer_count)) * self.ldm_line_size
                        self.pointer_[i] = (
                            self.store_ptr.change_access_size(self.element_per_acc) +
                            access_contiguous +
                            access_strided * self.static_stride)
                else:
                    layout = self.layout.python_ctor(self.layout.static_stride)
                    for i in range(self.pointer_count):
                        ref_offset = layout.get_ldm_initial_offset_fast(
                            self.lane_id,
                            self.ldm_count,
                            self.operand_a,
                            permute_m_pointer_idx= (i + pointer_offset) % self.real_pointer_count)
                        self.pointer_[i] = (
                            self.store_ptr.change_access_size(self.element_per_acc) +
                            ref_offset // self.element_per_acc)
            else:
                self.pointer_ = self.pointer_[pointer_offset:] + self.pointer_[:pointer_offset]
            
            # if minimize_register_optimize. nPointer > self.ldm_iters[1], so we must recalc
            # it force_update is set, we also re calc
            # else we only need to repermute by python slice.
            # note:  add_tile_offset(, !=0) can only use once
            # mn_offset = (warp_idx_mn >> 1) << 1
        self.add_pointer_offset_python(
            (k_offset * self.ldm_count[0] * self.ldm_num_line) *
            self.static_stride_vec * self.layout.element_per_acc // self.layout.interleave +
            mn_offset * self.warp_tile_shape_km[1])

    @pccm.cuda.member_function(device=True, forceinline=True)
    def tile_increment(self):
        code = pccm.FunctionCode(f"""
        add_tile_offset(num_tile, 0);
        """)
        return code.arg("num_tile", "int")

    def tile_increment_python(self, num: int):
        return self.add_tile_offset_python(num, 0)

    @pccm.cuda.member_function(name="operator++",
                               device=True,
                               forceinline=True)
    def operator_pp(self):
        code = pccm.FunctionCode(f"""
        add_tile_offset(1, 0); // strided, contig
        // tv::printf2_block_once(threadIdx.x, "byte_offset_=", byte_offset_);
        if ({self.partk} > 1) {{
            ++wmma_k_index_;
            // Jump to next stage
            if (wmma_k_index_ == {self.k_groups_per_tile}) {{
                wmma_k_index_ = 0;
                add_tile_offset((({self.partk} - 1) * {self.k_groups_per_tile}), 0);
            }}
        }}
        return *this;
        """)
        return code.ret(f"{self.class_name} &")

    def increment_python(self):
        self.add_tile_offset_python(1, 0)
        if (self.partk > 1):
            self.wmma_k_index_ += 1
            # Jump to next stage
            if (self.wmma_k_index_ == self.k_groups_per_tile):
                self.wmma_k_index_ = 0
                self.add_tile_offset_python(
                    ((self.partk - 1) * self.k_groups_per_tile), 0)
        return self

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_byte_offset(self):
        if not self.is_spec_32:
            code = pccm.FunctionCode(f"""
            tv::array<unsigned, {self.ldm_count.prod()}> *fetch_ptr = 
            reinterpret_cast<tv::array<unsigned, {self.ldm_count.prod()}> *>(&frag);

            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.ldm_iters[0]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.ldm_iters[1]}; ++c) {{

                    int access_idx = c + s * {self.ldm_iters[1]};

                    {self.const_access_pointer} source_ptr =
                        pointer_[c % {self.pointer_count}] +
                        {self.layout.tile_shape[1]} * (c / {self.pointer_count}) +
                        {self.ldm_count[0]} * s * {self.static_stride_vec};

                    char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

                    LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
                }}
            }}
            """)
        else:
            code = pccm.FunctionCode(f"""
            {self.dtype} *fetch_ptr = reinterpret_cast<{self.dtype} *>(&frag);

            TV_PRAGMA_UNROLL
            for (int s = 0; s < {self.ldm_iters[0]}; ++s) {{
                TV_PRAGMA_UNROLL
                for (int c = 0; c < {self.ldm_iters[1]}; ++c) {{
                    TV_PRAGMA_UNROLL
                    for (int ss = 0; ss < {self.ldm_count[0]}; ++ss) {{
                        TV_PRAGMA_UNROLL
                        for (int cc = 0; cc < {self.ldm_count[1]}; ++cc) {{
                            int access_idx =
                                cc + (ss + (c + s * {self.ldm_iters[1]}) *
                                            {self.ldm_count[0]}) *
                                        {self.ldm_count[1]};
                            int access_idx_contiguous = cc + c * {self.ldm_count[1]};
                            int access_idx_strided =
                                (ss + s * {self.ldm_count[0]}) * {self.ldm_num_line};

                            {self.const_access_pointer} source_ptr =
                                pointer_[access_idx_contiguous % {self.pointer_count}] +
                                {self.layout.tile_shape[1]} * {self.layout.element_per_acc} *
                                    (access_idx_contiguous / {self.pointer_count}) +
                                access_idx_strided * {self.static_stride};

                            char const *source_byte_ptr =
                                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                                byte_offset_;

                            fetch_ptr[access_idx] =
                                *reinterpret_cast<{self.dtype} const *>(source_byte_ptr);
                        }}
                    }}
                }}
            }}
            """)

        code.arg("frag", f"{self.fragment_t}&").arg("byte_offset",
                                                    str(self.index_t))
        return code

    async def load_with_byte_offset_python(self, frag: ArrayPtr,
                                           byte_offset: int):
        ptr_addrs = np.zeros((frag.length, ), dtype=np.int32)
        if not self.is_spec_32:
            fetch_ptr = frag.change_access_byte_size(self.ldm_count.prod() * 4)
            for s in range(self.ldm_iters[0]):
                for c in range(self.ldm_iters[1]):
                    access_idx = c + s * self.ldm_iters[1]
                    sw_part_idx = c // self.pointer_count
                    pointer_part = self.pointer_[c % self.pointer_count]
                    source_ptr = (pointer_part +
                                  self.layout.sw_shape[1] * sw_part_idx +
                                  self.ldm_count[0] * self.ldm_num_line * s *
                                  self.static_stride_vec)
                    source_byte_ptr = source_ptr.change_access_byte_size(
                        1) + byte_offset + self.byte_offset_
                    await checkers.smem_bank_conflicit_check(
                        fetch_ptr, access_idx)
                    await self.ldmatrix(fetch_ptr[access_idx], source_byte_ptr)
                    ptr_addrs[access_idx *
                              fetch_ptr.access_size:(access_idx + 1) *
                              fetch_ptr.access_size] = np.arange(
                                  source_byte_ptr.offset,
                                  source_byte_ptr.offset +
                                  fetch_ptr.access_size)
        else:
            fetch_ptr = frag.change_access_size(1)
            for s in range(self.ldm_iters[0]):
                for c in range(self.ldm_iters[1]):
                    for ss in range(self.ldm_count[0]):
                        for cc in range(self.ldm_count[1]):
                            access_idx = cc + (
                                ss + (c + s * self.ldm_iters[1]) *
                                self.ldm_count[0]) * self.ldm_count[1]
                            access_contig = cc + c * self.ldm_count[1]
                            access_stride = (
                                ss + s * self.ldm_count[0]) * self.ldm_num_line
                            pointer_part = self.pointer_[access_contig %
                                                         self.pointer_count]

                            source_ptr = (
                                pointer_part + self.layout.sw_shape[1] *
                                self.layout.element_per_acc * (access_contig //
                                self.real_pointer_count) +
                                access_stride * self.static_stride)
                            source_byte_ptr = source_ptr.change_access_byte_size(
                                1) + byte_offset + self.byte_offset_
                            fetch_ptr[
                                access_idx] = source_byte_ptr.change_access_size(
                                    1)[0]

        return ptr_addrs

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load_with_pointer_offset(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, pointer_offset * sizeof({self.dtype}));
        """)
        code.arg("frag", f"{self.fragment_t}&").arg("pointer_offset",
                                                    str(self.index_t))
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def load(self):
        code = pccm.FunctionCode(f"""
        load_with_byte_offset(frag, 0);
        """)
        code.arg("frag", f"{self.fragment_t}&")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def set_kgroup_index(self):
        code = pccm.code()
        code.arg("wmma_k", "int")
        return code

    async def load_python(self, frag: ArrayPtr):
        return await self.load_with_byte_offset_python(frag, 0)

    def set_wmma_k_index_python(self, wmma_k):
        return
