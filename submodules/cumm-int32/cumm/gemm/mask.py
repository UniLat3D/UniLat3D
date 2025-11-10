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

from typing import List, Optional, Tuple

import numpy as np
import pccm

from cumm.gemm.codeops import div_up
from cumm.gemm.core.metaarray import MetaArray, seq
from cumm.gemm.layout_tensorop import rowmajor_inverse_list, to_stride_list


class Mask(pccm.ParameterizedClass):
    def __init__(self, shape: MetaArray[int]):
        super().__init__()
        self.shape = shape
        for s in shape:
            assert (s & (s - 1) == 0) and s != 0, f"{s} must be power of 2"

        self.num_mask_32 = div_up(shape.prod(), 32)
        self.stride = to_stride_list(shape)
        self.num_32_for_each = seq(div_up(s, 32) for s in self.stride)
        self.add_member("mask_", f"tv::array<uint32_t, {self.num_mask_32}>")

    def get_masks(self, *idxes: Optional[int]):
        assert len(idxes) == len(self.shape)
        masks = [0] * self.num_mask_32
        refined_idxes: List[List[int]] = []
        refined_shape: MetaArray[int] = MetaArray(*[0] * len(self.shape))
        prod: int = 1
        for i, idx in enumerate(idxes):
            if idx is None:
                refined_idxes.append(list(range(self.shape[i])))
            else:
                refined_idxes.append([idx])
            refined_shape[i] = len(refined_idxes[-1])
            prod *= refined_shape[i]
        for i in range(prod):
            coords = rowmajor_inverse_list(i, refined_shape)
            m = 0
            for j in range(len(self.shape)):
                m += refined_idxes[j][coords[j]] * self.stride[j]
            mask_idx = m >> 5
            mask_bin_idx = m & 0b11111
            masks[mask_idx] |= (1 << mask_bin_idx)
        return masks

    @pccm.cuda.constructor(host=True, device=True, forceinline=True)
    def ctor(self):
        return pccm.code()

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def clear(self):
        code = pccm.code()
        code.raw(f"mask_.clear();")
        return code

    @pccm.cuda.member_function(const=True,
                               host=True,
                               device=True,
                               forceinline=True)
    def query(self):
        code = pccm.code()
        code.arg("idx", "int")
        if self.num_mask_32 == 1:
            code.raw(f"""
            return mask_[0] & (1 << idx);
            """)
            return code.ret("uint32_t")
        code.raw(f"return mask_[idx >> 5] & (1u << (idx & 0b11111))")
        return code.ret("uint32_t")

    @pccm.cuda.member_function(const=True,
                               host=True,
                               device=True,
                               forceinline=True)
    def query_coord(self):
        code = pccm.code()
        strs: List[str] = []
        for i in range(len(self.shape)):
            code.arg(f"idx{i}", "int")
            strs.append(f"idx{i} * {self.stride[i]}")
        code.raw(f"""
        return query({" + ".join(strs)});
        """)
        return code.ret("uint32_t")

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def set(self):
        code = pccm.code()
        code.arg("pred", "uint32_t")
        code.arg("idx", "int")

        if self.num_mask_32 == 1:
            code.raw(f"""
            mask_[0] |= pred << idx;
            """)
            return code
        code.raw(f"""
        mask_[idx >> 5] |= pred << (idx & 0b11111);
        """)
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def set_coord(self):
        code = pccm.code()
        code.arg("pred", "uint32_t")
        strs: List[str] = []
        for i in range(len(self.shape)):
            code.arg(f"idx{i}", "int")
            strs.append(f"idx{i} * {self.stride[i]}")
        code.raw(f"""
        return set(pred, {" + ".join(strs)});
        """)
        return code

    @pccm.cuda.member_function(host=True, device=True, forceinline=True)
    def clear_if_pred(self):
        code = pccm.code()
        code.arg("pred", "uint32_t")
        strs: List[str] = []
        for i in range(len(self.shape)):
            code.arg(f"idx{i}", "int")
            if i != len(self.shape) - 1:
                strs.append(f"idx{i} * {self.stride[i]}")
        code.raw(f"""
        return set(pred, {" + ".join(strs)});
        """)
        return code

    def clear_mask_if_pred_template(self, code: pccm.FunctionCode, pred: str,
                                    idxes: Tuple[Optional[int], ...]):
        # TODO use asm may faster?
        clear_masks = self.get_masks(*idxes)
        for j in range(len(clear_masks)):
            mask_j = np.array(clear_masks[j], dtype=np.uint32)
            if clear_masks[j] == 0:
                continue
            mask = ~mask_j
            if mask != 0:
                code.raw(f"""
                mask_.mask_[{j}] = {pred} ? 
                    mask_.mask_[{j}] & {mask}u : mask_.mask_[{j}];
                """)
            else:
                code.raw(f"""
                mask_.mask_[{j}] = {pred} ? 0u : mask_.mask_[{j}];
                """)
        return code

    def clear_mask_if_not_pred_template(self, code: pccm.FunctionCode,
                                        pred: str, idxes: Tuple[Optional[int],
                                                                ...]):
        # TODO use asm may faster?
        clear_masks = self.get_masks(*idxes)
        for j in range(len(clear_masks)):
            mask_j = np.array(clear_masks[j], dtype=np.uint32)
            if clear_masks[j] == 0:
                continue
            mask = ~mask_j
            if mask != 0:
                code.raw(f"""
                mask_.mask_[{j}] = {pred} ? 
                     mask_.mask_[{j}]: mask_.mask_[{j}] & {mask}u;
                """)
            else:
                code.raw(f"""
                mask_.mask_[{j}] = {pred}u ? mask_.mask_[{j}]: 0;
                """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_all_mask_if_not_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        for x in range(self.num_mask_32):
            code.raw(f"mask_[{x}] = pred ? mask_[{x}] : 0u;")
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def clear_all_mask_if_pred(self):
        code = pccm.cuda.PTXCode()
        code.arg("pred", "bool")
        for x in range(self.num_mask_32):
            code.raw(f"mask_[{x}] = pred ? 0u : mask_[{x}];")
        return code


if __name__ == "__main__":
    mask = Mask(seq(1, 1, 1, 8))
    for i in range(8):
        print(mask.get_masks(None, None, None, i))
