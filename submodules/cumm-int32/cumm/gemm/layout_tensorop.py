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

from typing import List, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.gemm import constants
from cumm.gemm.core import MetaArray, metaseq, seq


def to_stride(shape: np.ndarray):
    return np.cumprod(shape[::-1])[::-1]


def to_stride_list(shape: MetaArray[int]):
    shape = shape[::-1]
    res = MetaArray(*shape)
    m = 1
    for i, s in enumerate(shape):
        res[i] = m
        m *= s
    return res[::-1]


def rowmajor_inverse(index: int, shape: np.ndarray) -> np.ndarray:
    res = np.zeros_like(shape)
    ndim = len(shape)
    for i in range(ndim - 1, -1, -1):
        res[i] = index % shape[i]
        if i > 0:
            index -= res[i]
            index /= shape[i]
    return res


def rowmajor_inverse_list(index: int, shape: MetaArray[int]) -> MetaArray[int]:
    res = shape.copy()
    ndim = len(shape)
    for i in range(ndim - 1, -1, -1):
        res[i] = index % shape[i]
        if i > 0:
            index -= res[i]
            index //= shape[i]
    return res


class VoltaTensorOpCrosswise(pccm.ParameterizedClass):
    def __init__(self, element_size: int, kblock: int = 32):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.element_size = element_size
        self.tile_shape = metaseq(8, 4)
        self.part_shape = metaseq(4, 4)
        self.access_size = 64
        self.element_per_acc = self.access_size // element_size
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        self.static_stride = -1

        # cudasim mmebers
        self.stride = 0

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    def python_ctor(self, stride):
        new_obj = VoltaTensorOpCrosswise(self.element_size)
        new_obj.stride = stride
        return new_obj

    def from_shape_python(self, shape):
        l = VoltaTensorOpCrosswise(self.element_size)
        l.stride = shape[1]
        return l

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(f"return {self.class_name}(shape[1]);")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               const=True)
    def call_operator(self):
        code = pccm.FunctionCode(f"""
        int vec_contiguous_idx = y / {self.element_per_acc};
        int vec_strided_idx = x;
        int vec_strided_within_tile = vec_contiguous_idx & 0x7;
        // 0: tile: 4x64, a smem bank
        // 1. map to tile offset. assume we have 4x128, so tile offset
        // is 0, 64, 128, 192, ...
        int permuted_vec_contiguous  =  vec_strided_idx & (~0xF);
        // 2. inside a tile, map to each permuted sub tile 4x16
        // (0,4,8,12)[], (0,16,32,48)[]
        permuted_vec_contiguous += (vec_strided_idx & 0x3) * 4;
        permuted_vec_contiguous += (((vec_strided_idx >> 2) ^ ((vec_strided_idx & 0x10) >> 3)) & 0x3);
        // 3. generate permuted offset
        permuted_vec_contiguous ^= ((vec_strided_within_tile >> 1) & 0x3);

        int permuted_vec_strided = vec_contiguous_idx;
        int element_contiguous = permuted_vec_contiguous *  {self.element_per_acc} + 
                                (y % {self.element_per_acc});
    
        return element_contiguous + permuted_vec_strided * (stride * {self.element_per_acc});
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def __call__(self, x: int, y: int):
        vec_contiguous_idx = y // self.element_per_acc
        vec_strided_idx = x
        vec_strided_within_tile = vec_contiguous_idx & 0x7
        # // 0: tile: 4x64, a smem bank
        # // 1. map to tile offset. assume we have 4x128, so tile offset
        # // is 0, 64, 128, 192, ...
        permuted_vec_contiguous = vec_strided_idx & (~0xF)
        # // 2. inside a tile, map to each permuted sub tile 4x16
        # // (0,4,8,12)[], (0,16,32,48)[]
        permuted_vec_contiguous += (vec_strided_idx & 0x3) * 4
        permuted_vec_contiguous += (((vec_strided_idx >> 2) ^
                                     ((vec_strided_idx & 0x10) >> 3)) & 0x3)
        # // 3. generate permuted offset
        permuted_vec_contiguous ^= ((vec_strided_within_tile >> 1) & 0x3)

        permuted_vec_strided = vec_contiguous_idx
        element_contiguous = permuted_vec_contiguous * self.element_per_acc + (
            y % self.element_per_acc)

        return element_contiguous + permuted_vec_strided * (
            self.stride * self.element_per_acc)


class VoltaTensorOpCongruous(pccm.ParameterizedClass):
    def __init__(self, operand_a: bool, element_size: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.tile_shape = metaseq(4, 8)
        if operand_a:
            self.part_shape = metaseq(4, 4)
        else:
            self.part_shape = metaseq(2, 8)
        self.access_size = 128
        self.element_size = element_size
        self.operand_a = operand_a
        self.element_per_acc = self.access_size // element_size
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        self.static_stride = -1

        # cudasim mmebers
        self.stride = 0

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    def python_ctor(self, stride):
        new_obj = VoltaTensorOpCongruous(self.operand_a, self.element_size)
        new_obj.stride = stride
        return new_obj

    def from_shape_python(self, shape):
        l = VoltaTensorOpCongruous(self.operand_a, self.element_size)
        l.stride = shape[1]
        return l

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(f"return {self.class_name}(shape[1]);")
        code.arg("shape", "const tv::array<int, 2> &")
        code.ret(self.class_name)
        return code

    @pccm.cuda.member_function(name="operator()",
                               host=True,
                               device=True,
                               forceinline=True,
                               const=True)
    def call_operator(self):
        code = pccm.FunctionCode(f"""
        int vec_contiguous_idx = y / {self.element_per_acc};
        int vec_strided_idx = x;

        // Compute the fundamental tile being accessed
        int tile_contiguous_idx = vec_contiguous_idx / {self.tile_shape[1]};
        int tile_strided_idx = vec_strided_idx / {self.tile_shape[0]};

        int tile_contiguous_residual = vec_contiguous_idx % {self.tile_shape[1]};
        int tile_strided_residual = vec_strided_idx % {self.tile_shape[0]};

        int permuted_strided_within_tile;
        int permuted_contiguous_within_tile;
        """)
        if self.operand_a:
            code.raw(f"""
            permuted_strided_within_tile = (tile_contiguous_residual >> 1);
            permuted_contiguous_within_tile =
                (tile_strided_residual ^ permuted_strided_within_tile) |
                ((tile_contiguous_residual & 1) << 2);
            """)
        else:
            code.raw(f"""
            permuted_strided_within_tile = (tile_contiguous_residual & 0x3);
            permuted_contiguous_within_tile =
                (tile_strided_residual ^ permuted_strided_within_tile) |
                (tile_contiguous_residual & 0x4);
            """)
        code.raw(f"""
        // Compute final element location
        int element_contiguous = (tile_contiguous_idx * {self.tile_shape[1]} +
                                permuted_contiguous_within_tile) *
                                    {self.element_per_acc} +
                                (y % {self.element_per_acc});

        int element_strided =
            tile_strided_idx * {self.tile_shape[0]} + permuted_strided_within_tile;

        auto res = element_contiguous + element_strided * stride;
        // tv::printf2_block_once(threadIdx.x, stride_,
        // "VoltaTensorOpMultiplicandBCongruous", res, coord.strided(),
        // coord.contiguous());
        return res;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def __call__(self, x: int, y: int) -> int:
        vec_contiguous_idx = y // self.element_per_acc
        vec_strided_idx = x

        #  Compute the fundamental tile being accessed
        tile_contiguous_idx = vec_contiguous_idx // self.tile_shape[
            1]  # type: int
        tile_strided_idx = vec_strided_idx // self.tile_shape[0]  # type: int

        tile_contiguous_residual = vec_contiguous_idx % self.tile_shape[
            1]  # type: int
        tile_strided_residual = vec_strided_idx % self.tile_shape[
            0]  # type: int

        if self.operand_a:
            permuted_strided_within_tile = (tile_contiguous_residual >> 1)
            permuted_contiguous_within_tile = (
                tile_strided_residual ^ permuted_strided_within_tile) | (
                    (tile_contiguous_residual & 1) << 2)
        else:
            permuted_strided_within_tile = (tile_contiguous_residual & 0x3)
            permuted_contiguous_within_tile = (
                tile_strided_residual ^ permuted_strided_within_tile) | (
                    tile_contiguous_residual & 0x4)

        # Compute final element location
        element_contiguous = (
            (tile_contiguous_idx * self.tile_shape[1] +
             permuted_contiguous_within_tile) * self.element_per_acc +
            (y % self.element_per_acc))

        element_strided = tile_strided_idx * self.tile_shape[
            0] + permuted_strided_within_tile

        res = element_contiguous + element_strided * self.stride
        # tv::printf2_block_once(threadIdx.x, stride_,
        # "VoltaTensorOpMultiplicandBCongruous", res, coord.strided(),
        # coord.contiguous())
        return res


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
    with code.if_(f"{index_var} & 0b1 == 0"):
        code.raw(f"{offset_var} ^= 1 * {width};")
    with code.else_if_(f"{index_var} == 0b1"):
        code.raw(f"{offset_var} ^= 0b11 * {width};")
    with code.else_if_(f"{index_var} == 0b11"):
        code.raw(f"{offset_var} ^= 0b111 * {width};")
    with code.else_if_(f"{index_var} == 0b111"):
        code.raw(f"{offset_var} ^= 0b1111 * {width};")


class TensorOpMultiplicand(pccm.ParameterizedClass):
    def __init__(self, element_size: int, crosswise: int):
        super().__init__()
        # crosswise == 128 / (element_size / 8) for congruous
        self.add_dependency(TensorViewNVRTC)
        self.access_size = 128
        # kCrosswise elements in the contiguous dimension would span to a
        # shared memory cache line.
        self.crosswise = crosswise
        self.element_size = element_size
        self.spec_32 = element_size == 32
        self.element_per_acc = self.access_size // element_size
        # Contiguous dimension of the tile shape matches one shared memory cache
        # line - 128B.  For 128bit access size, it equals to 8 accesses.
        tile_shape_contig = 128 // (self.access_size // 8)
        tile_shape_contig_in_element = tile_shape_contig * self.element_per_acc
        # Number of kblocks to store PartitionShape::kContiguous Elements
        # when store input tile to smem, if the input tile stride is small (32)
        # the thread 0-7 access two line of input tile,
        # to avoid bank conflict, two lines must be saved to one line in smem.
        # so the factor is smem_bank_length // input_tile_stride
        # i.e. tile_shape_contig_in_element // crosswise

        # for congruous (no need to transpose smem), factor always 1 because TODO
        self.factor = tile_shape_contig_in_element // crosswise
        # factor == 128 // (self.access_size // 8) * self.access_size // element_size // (128 / (element_size / 8))
        # == 128 // 8 // element_size // (128 / (element_size / 8)) == 1
        # tile_shape_contig // self.factor == tile_shape_contig // (tile_shape_contig * self.element_per_acc // crosswise)
        # == crosswise // self.element_per_acc
        # The strided dimension needs to be at least (WarpSize(32) /
        # kTileShapeContiguous) for a warp to access.  To ensure conflict free
        # access, it also needs to be at least (kTileShapeContiguous / kFactor).
        # (crosswise // self.element_per_acc)
        # See comments below
        # assert tile_shape_contig // self.factor == crosswise // self.element_per_acc
        if (tile_shape_contig // self.factor) > (32 // tile_shape_contig):
            tile_shape_stride = tile_shape_contig // self.factor
        else:
            tile_shape_stride = (32 // tile_shape_contig)
        if not self.spec_32:
            self.tile_shape = metaseq(tile_shape_stride, tile_shape_contig)
            self.part_shape = metaseq(4, 4)
        else:
            self.tile_shape = metaseq(4, 8)
            self.part_shape = metaseq(4, 8)

        self.part_count = self.tile_shape // self.part_shape
        # self.access_count = self.part_shape  # TODO
        self.access_tensor_contig = metaseq(1, self.part_count[1],
                                            self.part_shape[1])
        self.access_tensor_strided = metaseq(1, self.part_count[0],
                                             self.part_shape[0])
        # self.access_stride_contig = to_stride(self.access_tensor_contig)
        # self.access_stride_strided = to_stride(self.access_tensor_strided)

        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        self.static_stride = -1
        # cudasim mmebers
        self.stride = 0

    def __repr__(self):
        return (f"TensorOpMultiplicand[ts={self.tile_shape}|"
                f"ps={self.part_shape}|f={self.factor}|c={self.crosswise}]")

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    def python_ctor(self, stride):
        new_obj = TensorOpMultiplicand(self.element_size, self.crosswise)
        new_obj.stride = stride
        return new_obj

    def from_shape_python(self, shape):
        l = TensorOpMultiplicand(self.element_size, self.crosswise)
        l.stride = shape[0]
        return l

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(f"return {self.class_name}(shape[0]);")
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
        if self.spec_32:
            code.raw(f"""
            int tc = y / 32;
            int ts = x / 4;

            int c = (y % 32) / {self.element_per_acc};
            int s = x % 4;
            {self.long_index_t} offset = (c ^ (2 * s)) * {self.element_per_acc} + s * stride +
                            tc * 32 + ts * stride * 4 + y % 4;
            return offset;
            """)
        else:
            code.raw(f"""
            // block tensor contig: [num_tile_x, num_part_x, num_x]
            // block tensor strided: [num_tile_y, num_part_y, num_y]

            int vec_contiguous_idx = y / {self.element_per_acc};
            int vec_strided_idx = x / {self.factor};
            // Compute the fundamental tile being accessed
            int tile_contiguous_idx =
                vec_contiguous_idx / ({self.tile_shape[1]} / {self.factor});

            int tile_contiguous_residual =
                vec_contiguous_idx % ({self.tile_shape[1]} / {self.factor}) +
                ((x % {self.factor}) * ({self.tile_shape[1]} / {self.factor}));
            int tile_strided_residual = vec_strided_idx % {self.tile_shape[0]};

            // Compute the 'partition' within the fundamental tile
            int partition_contiguous_idx =
                tile_contiguous_residual / {self.part_shape[1]};
            int partition_strided_idx =
                tile_strided_residual / {self.part_shape[0]};

            int partition_contiguous_residual =
                tile_contiguous_residual % {self.part_shape[1]};
            int partition_strided_residual =
                tile_strided_residual % {self.part_shape[0]};
            // Then swizzle
            int permuted_vec_contiguous_within_partition =
                partition_contiguous_residual ^ (partition_strided_residual % 4);

            int permuted_partition_contiguous_within_tile =
                partition_contiguous_idx ^ (partition_strided_idx % 2);
            // Compute final element location
            int element_contiguous = (tile_contiguous_idx * {self.tile_shape[1]} +
                                    permuted_partition_contiguous_within_tile *
                                        {self.part_shape[1]} +
                                    permuted_vec_contiguous_within_partition) *
                                        {self.element_per_acc} +
                                    (y % {self.element_per_acc});

            int element_strided = vec_strided_idx;
            return element_contiguous + element_strided * stride * {self.factor};
            """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)

    def __call__(self, x: int, y: int):
        if self.spec_32:
            tc = y // 32
            ts = x // 4

            c = (y % 32) // self.element_per_acc
            s = x % 4
            offset = ((c ^ (2 * s)) * self.element_per_acc + s * self.stride +
                      tc * 32 + ts * self.stride * 4 + y % 4)
            return offset
        else:
            vec_contiguous_idx = y // self.element_per_acc
            # factor: interleave stride with factor to contiguous.
            # for example, if factor == 4, for x=0,1,2,3, tile is put in contiguous.
            vec_strided_idx = x // self.factor
            # Compute the fundamental tile being accessed
            # equivalent to tile_contiguous_idx = vec_contiguous_idx // crosswise_size
            tile_contiguous_idx = vec_contiguous_idx // (self.tile_shape[1] //
                                                         self.factor)

            tile_contiguous_residual = vec_contiguous_idx % (
                self.tile_shape[1] // self.factor)
            # if x % factor != 0, tile_contiguous_residual move to right location.
            tile_contiguous_residual += (x % self.factor) * (
                self.tile_shape[1] // self.factor)

            tile_strided_residual = vec_strided_idx % self.tile_shape[0]

            # Compute the 'partition' within the fundamental tile
            partition_contiguous_idx = tile_contiguous_residual // self.part_shape[
                1]
            partition_strided_idx = tile_strided_residual // self.part_shape[0]

            partition_contiguous_residual = tile_contiguous_residual % self.part_shape[
                1]
            partition_strided_residual = tile_strided_residual % self.part_shape[
                0]

            # Then swizzle
            # indexes_contig = rowmajor_inverse(vec_contiguous_idx, self.access_tensor_contig)
            # indexes_strided = rowmajor_inverse(vec_strided_idx, self.access_tensor_strided)
            # if cudasim.threadIdx().x == 0:
            #     print(x, y, indexes_contig, tile_contiguous_idx, partition_contiguous_idx, partition_contiguous_residual, "CONTIG FACTOR", self.factor, self.stride)
            #     print(indexes_strided, partition_strided_idx, partition_strided_residual, "FACTOR", self.factor)

            permuted_vec_contiguous_within_partition = (
                partition_contiguous_residual ^
                (partition_strided_residual % 4))
            permuted_partition_contiguous_within_tile = (
                partition_contiguous_idx ^ (partition_strided_idx % 2))
            # permuted_partition_contiguous_within_tile = partition_contiguous_idx

            # Compute final element location
            element_contiguous = (
                (tile_contiguous_idx * self.tile_shape[1] +
                 permuted_partition_contiguous_within_tile * self.part_shape[1]
                 + permuted_vec_contiguous_within_partition) *
                self.element_per_acc + (y % self.element_per_acc))

            element_strided = vec_strided_idx

            return element_contiguous + element_strided * self.factor * self.stride

    def get_ldm_initial_offset_ref(self,
                                   lane_idx: int,
                                   ldm_count: MetaArray[int],
                                   transpose: bool = False,
                                   contig_offset: int = 0):
        """ transpose (not operend A)
        if not transpose:
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
                stride = lane_idx & 0b111 + ((lane_idx >> 4) << 3)
                contig_vec = (lane_idx >> 4) & 1
            else:  # operand A
                # Q0 Q2
                # Q1 Q3
                # stride: 01234567 89ABCDEF 01234567 89ABCDEF
                # contig: 00000000 00000000 11111111 11111111
                stride = lane_idx & 0b1111
                contig_vec = lane_idx >> 4
        else:
            raise NotImplementedError
        return self(stride, contig_vec * self.element_per_acc + contig_offset)


class TensorOpMultiplicandColumnMajorInterleaved(pccm.ParameterizedClass):
    def __init__(self, element_size: int, interleave: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.access_size = 128
        self.interleave = interleave
        self.element_size = element_size
        self.element_per_acc = self.access_size // element_size
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)
        self.static_stride = -1
        # cudasim mmebers
        self.stride = 0

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    def python_ctor(self, stride):
        new_obj = TensorOpMultiplicandColumnMajorInterleaved(
            self.element_size, self.interleave)
        new_obj.stride = stride
        return new_obj

    def from_shape_python(self, shape):
        l = TensorOpMultiplicandColumnMajorInterleaved(self.element_size,
                                                       self.interleave)
        l.stride = shape[0] * self.interleave
        return l

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(
            f"return {self.class_name}(shape[0] * {self.interleave});")
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
        code = pccm.FunctionCode(f"""
        constexpr int rows_per_smem_cache_line = 128 / {self.interleave};
        int row_id = x / rows_per_smem_cache_line;
        int col_id = (x % rows_per_smem_cache_line) * {self.interleave} + y;
        int access_block_id = col_id >> 4;
        int swizzle_access_block_id = access_block_id ^ (row_id & 1);
        int swizzle_col_id = swizzle_access_block_id << 4;
        return row_id * 128 + swizzle_col_id;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)


class TensorOpMultiplicandRowMajorInterleaved(pccm.ParameterizedClass):
    def __init__(self, element_size: int, interleave: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.access_size = 128
        self.interleave = interleave
        self.element_size = element_size
        self.element_per_acc = self.access_size // element_size
        self.index_t = str(dtypes.int32)
        self.long_index_t = str(dtypes.int64)
        self.add_member("stride", self.index_t)

    @pccm.cuda.constructor(host=True,
                           device=True,
                           forceinline=True,
                           constexpr=True)
    def ctor(self):
        code = pccm.code()
        code.arg("stride_", self.index_t)
        code.ctor_init("stride", "stride_")
        return code

    @pccm.cuda.static_function(host=True,
                               device=True,
                               forceinline=True,
                               constexpr=True)
    def from_shape(self):
        code = pccm.FunctionCode(
            f"return {self.class_name}(shape[1] * {self.interleave});")
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
        code = pccm.FunctionCode(f"""
        int const rows_per_smem_cache_line = 128 / {self.interleave};
        int row_id = x / rows_per_smem_cache_line;
        int col_id = (x % rows_per_smem_cache_line) * {self.interleave} + y;
        int access_block_id = col_id >> 4;
        int swizzle_access_block_id = access_block_id ^ (row_id & 1);
        int swizzle_col_id = swizzle_access_block_id << 4;
        return row_id * 128 + swizzle_col_id;
        """)
        code.arg("x,y", self.index_t)
        return code.ret(self.long_index_t)


if __name__ == "__main__":
    l = TensorOpMultiplicand(16, 32).from_shape_python([32, 128])
    for i in range(4):
        for j in range(4):

            print(i, j, l(i, j * 8))
    print(l.factor)
    pass
