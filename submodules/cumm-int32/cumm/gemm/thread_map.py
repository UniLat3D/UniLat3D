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
from typing import List

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import TensorViewNVRTC
from cumm.gemm import constants
from cumm.gemm.core import MetaArray, metaseq, seq

DONT_CARE = 1


def calc_thread_access_shape(tile: MetaArray[int], num_thread: int,
                             sub_tile: MetaArray[int]):
    thread_access_tile = tile // sub_tile
    if (num_thread >= thread_access_tile[1]):
        return metaseq(
            thread_access_tile[0] // (num_thread // thread_access_tile[1]), 1)
    else:
        return metaseq(thread_access_tile[0],
                       thread_access_tile[1] // num_thread)


def calc_thread_access_delta(tile: MetaArray[int], num_thread: int,
                             sub_tile: MetaArray[int]):
    thread_access_tile = tile // sub_tile
    if (num_thread >= thread_access_tile[1]):
        return metaseq(num_thread * sub_tile[0] // thread_access_tile[1],
                       DONT_CARE)
    else:
        return metaseq(1, num_thread * sub_tile[1])


def warp_partition_remain(shape: MetaArray[int], warp_count: int):
    res = metaseq(*[0] * len(shape))
    b = warp_count
    for i in range(len(shape)):
        res[i] = b
        b = 1 if shape[i] > b else b // shape[i]
    return res


def warp_partition(shape: MetaArray[int], warp_count: int):
    res = metaseq(*[0] * len(shape))
    b = warp_count
    for i in range(len(shape)):
        res[i] = 1 if shape[i] > b else b // shape[i]
        b = res[i]
    return res


def partition_iteration(shape: MetaArray[int], warp_count: int):
    remain = warp_partition_remain(shape, warp_count)
    res = metaseq(*[0] * len(shape))
    for i in range(len(shape)):
        if shape[i] > remain[i]:
            res[i] = shape[i] // remain[i]
        else:
            res[i] = 1
    return res


def partition_delta(shape: MetaArray[int], dilation: MetaArray[int],
                    warp_count: int):
    remain = warp_partition_remain(shape, warp_count)
    iteration = partition_iteration(shape, warp_count)

    res = metaseq(*[0] * len(shape))
    m = 1
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] > remain[i]:
            res[i] = m * shape[i] * dilation[i] // iteration[i]
        else:
            res[i] = 1
        m *= shape[i] * dilation[i]
    return res


def tight_partition_delta(shape: MetaArray[int], dilation: MetaArray[int],
                          warp_count: int):
    remain = warp_partition_remain(shape, warp_count)
    iteration = partition_iteration(shape, warp_count)

    res = metaseq(*[0] * len(shape))
    m = 1
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] > remain[i]:
            res[i] = m * shape[i] // iteration[i]
        else:
            res[i] = 1
        m *= shape[i] * dilation[i]
    return res


@pccm.skip_inherit
class InputThreadMapBase(pccm.ParameterizedClass):
    @property
    def iterations(self) -> MetaArray[int]:
        raise NotImplementedError

    @property
    def delta(self) -> MetaArray[int]:
        raise NotImplementedError

    @contextlib.contextmanager
    def tmap_loop(self,
                  code: pccm.FunctionCode,
                  stride_var: str,
                  contig_var: str = ""):
        iters = self.iterations
        with code.range_(stride_var, str(iters[0]), "TV_PRAGMA_UNROLL"):
            if contig_var:
                with code.range_(contig_var, str(iters[1]),
                                 "TV_PRAGMA_UNROLL"):
                    yield
            else:
                yield

    def tmap_loop_unroll_sc(self):
        for s in range(self.iterations[0]):
            for c in range(self.iterations[1]):
                yield s, c

    def tmap_loop_unroll_s(self):
        for s in range(self.iterations[0]):
            yield s

    def tmap_loop_unroll_c(self):
        for c in range(self.iterations[1]):
            yield c


class PitchLinear(InputThreadMapBase):
    def __init__(self, tile_shape: MetaArray[int],
                 sub_tile_shape: MetaArray[int], num_threads: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.tile_shape = tile_shape
        self.sub_tile_shape = sub_tile_shape
        self.num_threads = num_threads
        self.element_per_acc = sub_tile_shape[1]  # type: int
        self.tile_access_shape = tile_shape // sub_tile_shape

        self._iterations = calc_thread_access_shape(tile_shape, num_threads,
                                                    sub_tile_shape)
        self._delta = calc_thread_access_delta(
            metaseq(tile_shape[0], tile_shape[1]), num_threads, sub_tile_shape)

    @property
    def iterations(self) -> MetaArray[int]:
        return self._iterations

    @property
    def delta(self) -> MetaArray[int]:
        return self._delta

    def __repr__(self):
        return f"PitchLinear[{self.iterations}|{self.delta}]"

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def initial_offset(self):
        code = pccm.FunctionCode(f"""
            return {{(thread_id / {self.tile_access_shape[1]}) * {self.sub_tile_shape[0]},
                    (thread_id % {self.tile_access_shape[1]}) *  {self.sub_tile_shape[1]}}};
        """)
        return code.arg("thread_id", "int").ret("tv::array<int, 2>")

    def initial_offset_python(self, thread_id: int):
        return seq(
            (thread_id // self.tile_access_shape[1]) * self.sub_tile_shape[0],
            (thread_id % self.tile_access_shape[1]) * self.sub_tile_shape[1])


class PitchLinearWarpRaked(InputThreadMapBase):
    def __init__(self, tile_shape: MetaArray[int],
                 sub_tile_shape: MetaArray[int], warp_shape: MetaArray[int],
                 num_threads: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.tile_shape = tile_shape
        self.sub_tile_shape = sub_tile_shape
        self.all_threads = num_threads              # all thread we have
        self.warp_shape = warp_shape
        self.element_per_acc = sub_tile_shape[1]  # type: int
        self.warp_size = warp_shape[0] * warp_shape[1]  # type: int

        self.maximum_threads = tile_shape[0] * tile_shape[1] // self.element_per_acc     # max thread that all work
        self.num_threads = min(self.maximum_threads, self.all_threads)     # num of string we use to read                   # thread we use
        self.is_sparse = (self.num_threads != self.all_threads)
        
        self.num_warp = self.num_threads // self.warp_size
        self.thread_access_shape = metaseq(1, sub_tile_shape[1])
        self.tile_access_shape = tile_shape // self.thread_access_shape
        self.warp_access_count = self.tile_access_shape // self.warp_shape
        if self.warp_access_count[0] >= self.num_warp:
            warp_count_strided = self.num_warp
            warp_count_contig = 1
        else:
            warp_count_strided = self.warp_access_count[0]
            warp_count_contig = self.num_warp // warp_count_strided

        self.warp_count = metaseq(warp_count_strided, warp_count_contig)
        self._iterations = self.warp_access_count // self.warp_count  # type: MetaArray[int]
        self._delta = metaseq(self.warp_shape[0],
                              self.warp_shape[1] * self.element_per_acc)


    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def is_skipped(self):
        code = pccm.FunctionCode()
        code.arg('thread_id', 'int').ret('bool')
        code.raw(rf'''
            if(!{int(self.is_sparse)})
                return false;
            return thread_id >= {self.num_threads};
        ''')
        return code
    
    def is_skipped_python(self, thread_id: int) -> bool:
        if(not self.is_sparse):
            return False
        return thread_id >= self.num_threads

    @property
    def iterations(self) -> MetaArray[int]:
        return self._iterations

    @property
    def delta(self) -> MetaArray[int]:
        return self._delta

    def __repr__(self):
        warp_dilation = self.warp_shape * self.iterations

        data = {
            "it": self.iterations,
            "d": self.delta,
            "warp": self.warp_shape,
            "wc": self.warp_count,
            "wd": warp_dilation,
            "epa": self.element_per_acc,
        }
        msg = "|".join(f"{n}={v}" for n, v in data.items())
        return f"PitchLinearWarpRaked[{msg}]"

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def initial_offset(self):
        warp_dilation = self.warp_shape * self.iterations
        code = pccm.FunctionCode(f"""
        if(is_skipped(thread_id))
            return {{0, 0}};        // to InputIter: inefficient but convenience dummy offset.
        int warp_id = (thread_id / {self.warp_size});
        int lane_id = (thread_id % {self.warp_size});
        tv::array<int, 2> warp_offset{{warp_id / {self.warp_count[1]},
                                    warp_id % {self.warp_count[1]}}};
        constexpr tv::array<int, 2> kWarpDilation{{{warp_dilation[0]}, {warp_dilation[1]}}};
        tv::array<int, 2> thread_offset_in_warp{{lane_id / {self.warp_shape[1]},
                                                lane_id % {self.warp_shape[1]}}};
        tv::array<int, 2> offset_in_tile =
            kWarpDilation * warp_offset + thread_offset_in_warp;
        return {{offset_in_tile[0], offset_in_tile[1] * {self.element_per_acc}}};
        """)
        return code.arg("thread_id", "int").ret("tv::array<int, 2>")

    def initial_offset_python(self, thread_id: int):
        if self.is_skipped_python(thread_id):
            return seq(0, 0)
        warp_id = thread_id // self.warp_size
        lane_id = thread_id % self.warp_size
        warp_offset = seq(warp_id // self.warp_count[1],
                          warp_id % self.warp_count[1])
        warp_dilation = self.warp_shape * self.iterations
        thread_offset_in_warp = seq(lane_id // self.warp_shape[1],
                                    lane_id % self.warp_shape[1])
        offset_in_tile = warp_dilation * warp_offset + thread_offset_in_warp
        return seq(offset_in_tile[0], offset_in_tile[1] * self.element_per_acc)

    def initial_offset_nosubtile_python(self, thread_id: int):
        return self.initial_offset_python(thread_id)


def get_2d_access_iter_delta(shape: MetaArray[int], warp_remain: int, epa: int,
                             element_size_bits: int):
    iters = metaseq(0, 0)
    deltas = metaseq(0, 0)
    acc_shape = metaseq(0, 0)
    warp_part = metaseq(1, 1)
    memory_access_size = 128
    warp_size = constants.WARP_SIZE
    # multiple warp handle rows
    # for T1688, every warp of two warps handle 4 rows
    shape_row = shape[3] // warp_remain
    shape_width = shape[4] // epa
    # 128byte smem bank size
    # warp access must not cause bank conflict
    warp_bank_free_access_width = memory_access_size // (
        epa * element_size_bits // 8)
    # print(shape, warp_remain, epa, warp_bank_free_access_width)

    warp_bank_free_access_rows = warp_size // warp_bank_free_access_width
    if warp_bank_free_access_rows > shape_row:
        acc_shape[0] = shape_row
        acc_shape[1] = warp_size // shape_row
    else:
        acc_shape[1] = min(shape_width,
                           min(warp_size, warp_bank_free_access_width))
        acc_shape[0] = min(shape[3], warp_size // acc_shape[1])
    msg = (f"epa {epa} too large. "
           f"{acc_shape[1]} * {epa}(epa) must <= {shape[-1]}(part column) "
           "for bank free access.")
    assert acc_shape[1] * epa <= shape[-1], msg
    iters[0] = shape_row // acc_shape[0]
    iters[1] = shape_width // acc_shape[1]
    # print(warp_remain, acc_shape, shape_row, shape_width, epa)
    deltas[0] = acc_shape[0]
    deltas[1] = acc_shape[1] * epa
    return iters, deltas, acc_shape, warp_part


def get_1d_access_iter_delta_v1(shape: MetaArray[int], warp_remain: int,
                                epa: int, element_size_bits: int):
    warp_size = constants.WARP_SIZE
    deltas = metaseq(1, warp_size * epa)
    iters = metaseq(1, shape[4] // deltas[1])
    acc_shape = metaseq(1, warp_size)
    # divide warp to
    warp_part = metaseq(1, warp_remain)
    return iters, deltas, acc_shape, warp_part


def get_1d_access_iter_delta(shape: MetaArray[int], warp_remain: int, epa: int,
                             element_size_bits: int):
    warp_size = constants.WARP_SIZE
    warp_remain_row = shape[3]
    warp_remain_col = warp_remain // warp_remain_row
    deltas = metaseq(1, warp_size * epa)
    iters = metaseq(1, shape[4] // deltas[1] // warp_remain_col)
    # iters = metaseq(1, shape[4] // constants.WARP_SIZE // epa)
    acc_shape = metaseq(1, warp_size)
    # divide warp to
    warp_part_stride = metaseq(warp_remain // shape[3], 1)
    return iters, deltas, acc_shape, warp_part_stride


class Out5DLinear(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, part_shape: MetaArray[int],
                 part_dilation: MetaArray[int], warp_count: int,
                 element_per_acc: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.part_shape = part_shape
        self.part_dilation = part_dilation
        iterations = partition_iteration(part_shape[1:4], warp_count)
        self.remain = warp_partition_remain(part_shape[1:4], warp_count)
        self.part = warp_partition(part_shape[1:4], warp_count)
        delta = tight_partition_delta(part_shape[1:4], part_dilation[1:4],
                                      warp_count)
        warp_remain_for_row_col = self.remain[2]
        use_2d_access = part_shape[3] > warp_remain_for_row_col
        self.element_per_acc = element_per_acc
        if use_2d_access:
            # iters, deltas, acc_shape, warp_part
            res = get_2d_access_iter_delta(part_shape, warp_remain_for_row_col,
                                           element_per_acc, dtype.bitsize())
        else:
            res = get_1d_access_iter_delta(part_shape, warp_remain_for_row_col,
                                           element_per_acc, dtype.bitsize())
        iters2d, delta2d, acc_shape, warp_part = res
        self.iters2d = iters2d
        # TODO warp_part[1] isn't correct even if we don't use it.
        self.warp_parts = metaseq(1, self.part[0], self.part[1], warp_part[0],
                                  warp_part[1])
        self.iterations = metaseq(1, iterations[0], iterations[1], iters2d[0],
                                  iters2d[1])
        if cudasim.enable_debug():
            print("OUT TMAP", self.remain, warp_count, self.warp_parts, part_shape,
                  iterations, self.iterations, iters2d, acc_shape)

        self.delta = metaseq(1, delta[0], delta[1], delta2d[0], delta2d[1])
        self.acc_shape_2d = acc_shape
        self.long_index_t = dtypes.int64
        self.index_t = dtypes.int32

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def initial_offset(self):
        code = pccm.FunctionCode(f"""
        int warp_idx = thread_idx / {constants.WARP_SIZE};
        int lane_idx = thread_idx %  {constants.WARP_SIZE};

        // Compute warp location
        int cluster_idx = warp_idx / {self.warp_parts[1]};
        int residual_cluster = warp_idx % {self.warp_parts[1]};

        int group_idx = residual_cluster / {self.warp_parts[2]};
        int residual_group = residual_cluster % {self.warp_parts[2]};

        int row_idx = residual_group / {self.warp_parts[3]};
        int col_idx = residual_group % {self.warp_parts[3]};

        // Compute per-lane offset
        // in 1d warp, row offset always 0
        int lane_row_offset = lane_idx / {self.acc_shape_2d[1]};
        int lane_col_offset = lane_idx % {self.acc_shape_2d[1]};

        // Compute coordinate in output space
        //
        // kPartShape: [Tile, Cluster, Group, Row, Col]
        // in out iter, x * 4 * 2 * 4 * 1 = x * 32

        // in smem loader, x * 4 * 1 * 1 * 1 = x * 4
        // 0, 4, 8, 12, full parallel,
        // in 0~3, warp 0 handle (0, 2), warp 1 handle (1, 3)
        // in 4~7, warp 2 handle (4, 6), warp 3 handle (5, 7)
        // so for thread 0, it handle [0, 0,32,64,96] and [2, 0,32,64,96]
        int cluster_offset = cluster_idx * {self.part_shape[2]} * {self.part_dilation[2]} *
                            {self.part_shape[3]} * {self.part_dilation[3]};
        // 0, 1, 0, 1 * 8// warp 0 handle [0, 2], warp 1 handle [1, 3]
        int group_offset = group_idx * {self.part_shape[3]} * {self.part_dilation[3]};
        // 0
        int row_offset = row_idx * {self.iterations[3]} * {self.acc_shape_2d[0]}; // 1d
        // we mul kElementsPerAccess here because unit of kAccessShape2D[1] isn't element.
        int column_offset =
            col_idx * {self.iters2d[1]} * {self.acc_shape_2d[1]} * {self.element_per_acc};
        return {{cluster_offset + group_offset + row_offset + lane_row_offset, 
            (column_offset + lane_col_offset) * {self.element_per_acc}}};

        """)
        return code.arg("thread_idx", "int").ret("tv::array<int, 2>")

    def initial_offset_python(self, thread_idx: int):

        warp_idx = thread_idx // constants.WARP_SIZE
        lane_idx = thread_idx % constants.WARP_SIZE

        cluster_idx = warp_idx // self.warp_parts[1]
        residual_cluster = warp_idx % self.warp_parts[1]

        group_idx = residual_cluster // self.warp_parts[2]
        residual_group = residual_cluster % self.warp_parts[2]

        row_idx = residual_group // self.warp_parts[3]
        col_idx = residual_group % self.warp_parts[3]
        lane_row_offset = lane_idx // self.acc_shape_2d[1]
        lane_col_offset = lane_idx % self.acc_shape_2d[1]
        cluster_offset = cluster_idx * self.part_shape[2] * self.part_dilation[
            2] * self.part_shape[3] * self.part_dilation[3]
        group_offset = group_idx * self.part_shape[3] * self.part_dilation[3]
        row_offset = row_idx * self.iterations[3] * self.acc_shape_2d[0]
        column_offset = col_idx * self.iters2d[1] * self.acc_shape_2d[1]

        return seq(
            cluster_offset + group_offset + row_offset + lane_row_offset,
            (column_offset + lane_col_offset) * self.element_per_acc)

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def iteration_inc_params(self):
        code = pccm.FunctionCode(f"""
        tv::array<{self.long_index_t}, 3> increments{{}};
        increments[0] = stride * {self.delta[1]} -
                        stride * {self.delta[2]} * ({self.iterations[2]} - 1) -
                        stride * {self.delta[3]} * ({self.iterations[3]} - 1);

        increments[1] =
            stride * {self.delta[2]} - stride * {self.delta[3]} * ({self.iterations[3]} - 1);
        increments[2] = stride * {self.delta[3]};
        return increments;

        """)
        return code.arg("stride",
                        "int").ret(f"tv::array<{self.long_index_t}, 3>")

    def iteration_inc_params_python(self, stride: int):
        increments = seq(0, 0, 0)
        increments[0] = (stride * self.delta[1] - stride * self.delta[2] *
                         (self.iterations[2] - 1) - stride * self.delta[3] *
                         (self.iterations[3] - 1))

        increments[1] = stride * self.delta[2] - stride * self.delta[3] * (
            self.iterations[3] - 1)
        increments[2] = stride * self.delta[3]
        return increments

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def iteration_advance_params(self):
        code = pccm.FunctionCode(f"""
        tv::array<{self.long_index_t}, 4> advances{{}};
        // so advances[0] == 
        advances[0] =
            stride * {self.part_shape[0]} * {self.part_shape[1]} * {self.part_shape[2]} * {self.part_shape[3]};
        // TODO for cluster, advance_cluster should be wrong but the dilation of
        // cluster is always 1 in all cutlass configs. review this later.

        advances[1] = stride * {self.part_shape[2]} * {self.part_dilation[2]} * {self.part_shape[3]} *
                    {self.part_dilation[3]};
        // move to next 'dilation'
        // row dilation: LaneMmaShape::kM
        // for standard strided access,
        // first dilation, we need handle 0, 2, 4, 6 group, so warp0 = 0, 4; warp1 =
        // 2, 6, delta = 16 0, 2, 4, 6 rows in smem mapped to 0, 2, 4, 6 rows in
        // output memory (with different dilation). second dilation, we need handle
        // 1, 3, 5, 7 group, so warp0 = 1, 5; warp1 = 3, 7 1, 3, 5, 7 rows in smem
        // mapped to 1, 3, 5, 7 rows in output memory (with different dilation).

        // and advance_group should be part_shape[3] * part_dilation[3].

        // but the layout of lane mma result isn't standard.
        // they are strided too. columns are fixed in OutWarpTileIter, but rows
        // aren't. if lane mma count row is 2, we need to move to bottom part of mma
        // block instead of move forward.

        // first dilation, we need handle 0, 1, 2, 3 group, so warp0 = 0, 2; warp1 =
        // 1, 3 0, 2, 4, 6 rows in smem mapped to 0, 1, 2, 3 rows in output memory
        // second dilation, we need handle 4, 5, 6, 7 group, so warp0 = 4, 6; warp1
        // = 5, 7 1, 3, 5, 7 rows in smem mapped to 4, 5, 6, 7 rows in output memory
        advances[2] =
            stride * ({self.part_shape[2]} - 1) * {self.part_shape[3]} * {self.part_dilation[3]};
        advances[3] = stride * {self.part_shape[3]};
        return advances;
        """)
        return code.arg("stride",
                        "int").ret(f"tv::array<{self.long_index_t}, 4>")

    def iteration_advance_params_python(self, stride: int):
        advances = seq(0, 0, 0, 0)
        advances[0] = stride * self.part_shape[0] * self.part_shape[
            1] * self.part_shape[2] * self.part_shape[3]

        advances[1] = stride * self.part_shape[2] * self.part_dilation[
            2] * self.part_shape[3] * self.part_dilation[3]
        advances[2] = stride * (self.part_shape[2] -
                                1) * self.part_shape[3] * self.part_dilation[3]
        advances[3] = stride * self.part_shape[3]
        return advances
