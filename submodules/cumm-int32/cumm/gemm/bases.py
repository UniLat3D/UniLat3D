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

import abc
from typing import Optional, Tuple, Union

import pccm

from cumm import dtypes
from cumm.constants import CUTLASS_MODE
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.thread_map import PitchLinear, PitchLinearWarpRaked


class GemmContainerBase(abc.ABC):
    def min_arch(self) -> Optional[Tuple[int, int]]:
        return None


@pccm.skip_inherit
class GemmComponentBase(pccm.ParameterizedClass):
    def min_arch(self) -> Optional[Tuple[int, int]]:
        return None

@pccm.skip_inherit
class DTypeBase(GemmComponentBase):
    def __init__(self,
                 dtype: dtypes.DType,
                 element_per_acc: int,
                 alignment: int = 0,
                 access_per_vector: int = 1):
        super().__init__()
        self.dtype = dtype
        self.pointer = f"{dtype} *"
        self.const_pointer = f"const {dtype} *"
        self.byte_pointer = f"char *"
        self.const_byte_pointer = f"const char *"
        self.long_index_t = dtypes.int64
        self.index_t = dtypes.int32
        assert access_per_vector >= 0
        if access_per_vector == 0:
            access_per_vector = element_per_acc
        self.access_per_vector = access_per_vector
        assert element_per_acc % access_per_vector == 0
        element_per_acc = element_per_acc // access_per_vector
        self.element_per_acc = element_per_acc

        if alignment == -1:
            alignment = element_per_acc * dtype.bitsize() // 8
        alignment = min(alignment, element_per_acc * dtype.bitsize() // 8)
        if CUTLASS_MODE:
            if alignment > 0:
                fmt = "cutlass::AlignedArray<{}, {}, {}>"
            else:
                fmt = "cutlass::Array<{}, {}>"
            self.access_t = fmt.format(dtype, element_per_acc, alignment)
            self.access_pointer = f"{self.access_t} *"
            self.const_access_pointer = f"const {self.access_t} *"
        else:
            if alignment > 0:
                fmt = "tv::alignedarray<{}, {}, {}>"
            else:
                fmt = "tv::array<{}, {}>"
            # here we must use int4/2/1 instead of standard float16 array
            # to make sure expected ptx is generated.
            # if we use float16[8] as a 128bit access type,
            # ptx may contains st.shared.u16, u32, etc
            # if use int4/2/1, correct st.shared.v4.u32 is generated.
            access_size_bits = element_per_acc * dtype.bitsize()
            if alignment * 8 < access_size_bits:
                # c++ don't support alignment smaller than size of a type.
                # so we reduce type size and increment count.
                access_size_bits_refine = min(alignment * 8, access_size_bits)
                if alignment != 0:
                    refine_count = access_size_bits // access_size_bits_refine
                else:
                    refine_count = 1
            else:
                access_size_bits_refine = access_size_bits
                refine_count = 1
            if access_size_bits_refine == 128:
                self.access_t = fmt.format("int4", refine_count, alignment)
            elif access_size_bits_refine == 64:
                self.access_t = fmt.format("int2", refine_count, alignment)
            elif access_size_bits_refine == 32:
                self.access_t = fmt.format("int", refine_count, alignment)
            else:
                self.access_t = fmt.format(dtype, element_per_acc, alignment)
            self.access_pointer = f"{self.access_t} *"
            self.const_access_pointer = f"const {self.access_t} *"


@pccm.skip_inherit
class GemmIterator(DTypeBase):
    """
    Iterator Base
    
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 fragment_length: int,
                 element_per_acc: int,
                 alignment: int = 0,
                 access_per_vector: int = 1):
        super().__init__(dtype, element_per_acc, alignment, access_per_vector)
        self.element_count = fragment_length
        assert fragment_length > 0, "zero size frag length isn't allowed"
        self.fragment_t = array_type(dtype, self.element_count)


@pccm.skip_inherit
class GemmInputIterator(GemmIterator):
    """
    Iterator Base
    
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 tmap: Union[PitchLinearWarpRaked, PitchLinear],
                 sub_tile_shape: MetaArray[int],
                 fragment_length: int,
                 element_per_acc: int,
                 alignment: int = 0,
                 access_per_vector: int = 1):
        super().__init__(dtype, fragment_length, element_per_acc, alignment,
                         access_per_vector)
        self.tmap = tmap
        self.sub_tile_shape = sub_tile_shape

    def get_params(self) -> pccm.ParameterizedClass:
        raise NotImplementedError

    def python_ctor(self,
                    params,
                    ptr: ArrayPtr,
                    extent: MetaArray[int],
                    thread_id: int,
                    tb_offset: MetaArray[int],
                    is_left: bool = True) -> "GemmInputIterator":
        raise NotImplementedError

    def tile_increment_python(self, num_tile: int):
        raise NotImplementedError

    def clear_mask_python(self):
        raise NotImplementedError

    def increment_python(self):
        return self.tile_increment_python(1)

    def load_python(self, frag: ArrayPtr):
        raise NotImplementedError


@pccm.skip_inherit
class GemmSmemIterator(GemmIterator):
    def python_ctor(self, stride: int, ptr: ArrayPtr,
                    thread_id: int) -> "GemmSmemIterator":
        raise NotImplementedError

    def tile_increment_python(self, num_tile: int):
        raise NotImplementedError

    def increment_python(self):
        return self.tile_increment_python(1)

    async def store_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def get_smem_vis_shape(self) -> MetaArray[int]:
        raise NotImplementedError


@pccm.skip_inherit
class GemmWarpIterator(GemmIterator):
    async def python_ctor(self, ptr: ArrayPtr, warp_idx_k: int,
                          warp_idx_residual: int,
                          lane_idx: int) -> "GemmWarpIterator":
        raise NotImplementedError

    def tile_increment_python(self, num_tile: int):
        raise NotImplementedError

    def increment_python(self):
        return self.tile_increment_python(1)

    async def load_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def set_wmma_k_index_python(self, wmma_k: int):
        raise NotImplementedError


@pccm.skip_inherit
class GemmOutputIterator(GemmIterator):
    """
    Iterator Base
    
    """
    def get_params(self) -> pccm.ParameterizedClass:
        raise NotImplementedError

    def support_both_source_and_target(self) -> bool:
        raise NotImplementedError

    def python_ctor(self, params, ptr: ArrayPtr, extent: MetaArray[int],
                    offset_2d: MetaArray[int],
                    thread_idx: int) -> "GemmOutputIterator":
        raise NotImplementedError

    def store_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def load_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def increment_python(self):
        raise NotImplementedError


@pccm.skip_inherit
class GemmOutWarpIterator(GemmIterator):
    def python_ctor(self, ptr: ArrayPtr, warp_offset_m: int,
                    warp_offset_n: int,
                    lane_idx: int) -> "GemmOutWarpIterator":
        raise NotImplementedError

    async def store_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def add_pointer_offset_python(self, pointer_offset: int):
        raise NotImplementedError


@pccm.skip_inherit
class GemmOutSmemLoader(GemmIterator):
    def python_ctor(self, ptr: ArrayPtr,
                    thread_idx: int) -> "GemmOutSmemLoader":
        raise NotImplementedError

    async def load_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def add_pointer_offset_python(self, pointer_offset: int):
        raise NotImplementedError


@pccm.skip_inherit
class GemmOutFragIterator(GemmIterator):
    def python_ctor(self, src_ptr: ArrayPtr) -> "GemmOutFragIterator":
        raise NotImplementedError

    def load_python(self, frag: ArrayPtr):
        raise NotImplementedError

    def increment_python(self):
        raise NotImplementedError


@pccm.skip_inherit
class WarpMma(GemmComponentBase):
    def python_ctor(self) -> "WarpMma":
        raise NotImplementedError

    async def __call__(self, D: ArrayPtr, A: ArrayPtr, B: ArrayPtr,
                       C: ArrayPtr):
        raise NotImplementedError



@pccm.skip_inherit
class GemmOutputOp(GemmComponentBase):
    def python_ctor(self, alpha: float, beta: float) -> "GemmOutputOp":
        raise NotImplementedError

    def call_op_source_python(self, accumulator: ArrayPtr, source: ArrayPtr):
        raise NotImplementedError

    def call_op_nosource_python(self, accumulator: ArrayPtr):
        raise NotImplementedError

    def set_k_partition_python(self, k_part: int, k_part_count: int):
        raise NotImplementedError


@pccm.skip_inherit
class GemmApply(GemmComponentBase):
    def python_ctor(self) -> "GemmApply":
        raise NotImplementedError

    def apply_output_operator_python(self, output_fragment: ArrayPtr,
                                     output_op: GemmOutputOp,
                                     aligned_accum_fragment: ArrayPtr,
                                     source_fragment: ArrayPtr):
        raise NotImplementedError

    def apply_output_operator_no_source_python(
            self, output_fragment: ArrayPtr, output_op: GemmOutputOp,
            aligned_accum_fragment: ArrayPtr):
        raise NotImplementedError
