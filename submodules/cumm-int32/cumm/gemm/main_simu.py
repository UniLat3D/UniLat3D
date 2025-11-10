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

import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from tensorpc.apps import vis
import numpy as np

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.gemm import kernel
from cumm.gemm.main import GemmMainUnitTest, gen_gemm_kernels

VIS_IP = "127.0.0.1:51051"

GEMM_VIS_GLOBAL_SCALE = 10
import os

os.environ["CUMM_DEBUG"] = "1"


def vis_in_relay(figs):
    vis.vis_figures(VIS_IP, figs)


class GridPlane2d(vis.objects.Lines2d):
    def __init__(self,
                 minmax: Tuple[float, float, float, float],
                 vsize: Tuple[float, float],
                 color: str,
                 width: float = 1,
                 opacity: float = 1,
                 last_residual: bool = True):
        num = (int((minmax[2] - minmax[0]) / vsize[0]),
               int((minmax[3] - minmax[1]) / vsize[1]))
        xs = np.linspace(minmax[0], minmax[2], num[0] + 1, endpoint=True)
        ys = np.linspace(minmax[1], minmax[3], num[1] + 1, endpoint=True)
        horizontal_lines = np.zeros((num[1] + 1, 2, 2), dtype=np.float32)
        vertical_lines = np.zeros((num[0] + 1, 2, 2), dtype=np.float32)
        # in 3D, x is vertical and y is horizontal.
        horizontal_lines[:, 0, 0] = minmax[0]
        horizontal_lines[:, 0, 1] = ys
        horizontal_lines[:, 1, 0] = minmax[2]
        horizontal_lines[:, 1, 1] = ys

        vertical_lines[:, 0, 0] = xs
        vertical_lines[:, 0, 1] = minmax[1]
        vertical_lines[:, 1, 0] = xs
        vertical_lines[:, 1, 1] = minmax[3]
        lines = np.concatenate([horizontal_lines, vertical_lines])
        lines = lines[:, :, ::-1]
        return super().__init__(lines, color, width, opacity)


class Coords2d(vis.objects.Lines2d):
    def __init__(self,
                 coords: np.ndarray,
                 vsize: Tuple[float, float],
                 epa: int,
                 color: str,
                 width: float = 1,
                 opacity: float = 1):
        # coords: [N, 2]
        assert coords.shape[0] % epa == 0
        coords = coords.reshape(coords.shape[0] // epa, epa, 2)[:, 0] * vsize
        vsize = (vsize[0], vsize[1] * epa)
        coords_x1y0 = coords.copy()
        coords_x1y0[:, 0] += vsize[0]
        coords_x0y1 = coords.copy()
        coords_x0y1[:, 1] += vsize[1]
        coords_x1y1 = coords.copy()
        coords_x1y1[:, 0] += vsize[0]
        coords_x1y1[:, 1] += vsize[1]
        lines = np.stack([
            coords, coords_x1y0, coords_x1y0, coords_x1y1, coords_x1y1,
            coords_x0y1, coords_x0y1, coords
        ],
                         axis=1)
        lines = lines.reshape(-1, 2, 2)[:, :, ::-1]
        return super().__init__(lines, color, width, opacity)


def transpose(l):
    return list(map(list, zip(*l)))


def offset_to_coord(offset: np.ndarray, stride: int) -> np.ndarray:
    return np.stack([offset // stride, offset % stride], axis=1)


def vis_area_coords(fig: vis.figure.ImageFigure,
                    offset_list,
                    group_size,
                    shape,
                    offset: Tuple[float, float],
                    name: str,
                    epa: int = 1):
    scale = GEMM_VIS_GLOBAL_SCALE
    num_it = len(offset_list[0])
    offsets_raw: np.ndarray = np.stack(offset_list).reshape(
        group_size, num_it, -1)
    offsets = offsets_raw.reshape(-1)
    offsets = offsets[offsets != -1]
    coords = offset_to_coord(offsets, shape[1])
    with fig.layer(name) as layer:
        grid = GridPlane2d([0, 0, shape[0] * scale, shape[1] * scale],
                           [scale, scale], "green")
        coor = Coords2d(coords, [scale, scale], epa, 'red')
        layer.add_object(grid)
        layer.add_object(coor)
        for it in range(num_it):
            offsets_per_it = offsets_raw[:, it].reshape(-1)
            offsets_per_it = offsets_per_it[offsets_per_it != -1]
            if offsets_per_it.size == 0:
                continue
            coords_per_it = offset_to_coord(offsets_per_it, shape[1])[:, ::-1]
            epa = 2
            coords_per_it_epa = coords_per_it.reshape(
                coords_per_it.shape[0] // epa, epa, 2)[:, 0]
            epa = 8
            texts = vis.objects.Texts2d(coords_per_it_epa * scale,
                                        [f"{it}"] * len(coords_per_it_epa),
                                        "black", 10)
            layer.add_object(texts)
        layer.move(offset[0], offset[1])
        return layer.bound()


def vis_gemm_input_2d(inp: np.ndarray,
                      blocks: cudasim.Dim3,
                      threads: cudasim.Dim3,
                      fig_per_group: Dict[int, vis.figure.ImageFigure],
                      res,
                      name: str,
                      offset: List[float],
                      coord_input: bool = False):
    grouped_res = {}
    smem_shape = [0, 0]
    input_epa = 0
    smem_epa = 0
    warp_epa = 0
    scale = GEMM_VIS_GLOBAL_SCALE

    tile_shape = None
    part_shape = None
    for (bx, by, bz, tx, ty, tz), val in res.items():
        warp_id = (tx // 32)
        lane_id = (tx % 32)
        input_coords = val["input_coords"]
        smem_coords = val["smem_coords"]
        warp_coords = val["warp_coords"]
        warp_frags = val["warp_frags"]

        smem_shape = val["smem_shape"]
        input_epa = val["input_epa"]
        smem_epa = val["smem_epa"]
        warp_epa = val["warp_epa"]
        if "tile_shape" in val:
            tile_shape = val["tile_shape"]
        if "part_shape" in val:
            part_shape = val["part_shape"]

        group_id = warp_id * 32 + tx // 8  # 4 phase 128bit load
        group_id = threads.calc_offset(
            tx, ty, tz) + blocks.calc_offset(bx, by, bz) * threads.count()
        group_id = int(group_id)
        # group_id = (tx // 32) * 32 + (tx % 4) * 8 + (tx // 4)
        # group_id //= 8
        # group_id = warp_id
        # group_id //= 4
        # group_id = tx // 4
        # if tx % 2 != 0:
        #     continue
        if group_id not in grouped_res:
            grouped_res[group_id] = []
        grouped_res[group_id].append(
            (input_coords, smem_coords, warp_coords, warp_frags, tx))           # TODO  not good smem shape
    # smem shape to bank view
    # bank_size_element_view = 128 // inp.itemsize
    # smem_shape = [kernel.div_up(smem_shape[0] * smem_shape[1], bank_size_element_view), bank_size_element_view]
    if name == 'A':
        smem_shape = [64, 64]
    elif name == 'B':
        smem_shape = [64, 64]
    else:
        assert(0)
    # print(smem_shape)
    # if smem_shape == [32, 256]:
    #     smem_shape = [64, 128]
    # breakpoint()
    grouped_res = OrderedDict(sorted(grouped_res.items(), key=lambda x: x[0]))
    fake_img = None
    bounds: List[List[float]] = []
    for group_id, v in grouped_res.items():
        v = transpose(v)
        input_offsets_list = v[0]
        smem_offsets_list = v[1]
        warp_offsets_list = v[2]
        warp_frags_list = v[3]

        group_size = len(v[0])
        if group_id not in fig_per_group:
            fig_per_group[group_id] = vis.figure.ImageFigure(0,
                                                             fake_img,
                                                             thread_id=v[-1],
                                                             group_id=group_id)
        fig = fig_per_group[group_id]

        bound_inp = vis_area_coords(fig,
                                    input_offsets_list,
                                    group_size,
                                    inp.shape,
                                    offset,
                                    f"{name}_inp",
                                    epa=input_epa)
        if coord_input:
            # convert warp frag data to input coords, then draw them in input.
            num_it = len(warp_frags_list[0])
            warp_data = np.stack(warp_frags_list)
            offsets_raw: np.ndarray = warp_data.reshape(
                group_size, num_it, -1).astype(np.int32)
            offsets = offsets_raw.reshape(-1)
            coords = offset_to_coord(offsets, inp.shape[1])
            with fig.layer(f"{name}_warp_data") as layer:
                coor = Coords2d(coords, [scale, scale], 1, 'aqua')
                layer.add_object(coor)
                for it in range(num_it):
                    offsets_per_it = offsets_raw[:, it].reshape(-1)
                    coords_per_it = offset_to_coord(offsets_per_it,
                                                    inp.shape[1])[:, ::-1]
                    texts = vis.objects.Texts2d(coords_per_it * scale,
                                                [f"{it}"] * len(coords_per_it),
                                                "blue", 10)
                    layer.add_object(texts)
                layer.move(offset[0], offset[1])
        bounds.append(bound_inp)

        bound_smem = vis_area_coords(fig,
                                     smem_offsets_list,
                                     group_size,
                                     smem_shape,
                                     [bound_inp[2] + 10, bound_inp[1]],
                                     f"{name}_smem",
                                     epa=smem_epa)
        bounds.append(bound_smem)
        bound_warp = vis_area_coords(fig,
                                     warp_offsets_list,
                                     group_size,
                                     smem_shape,
                                     [bound_smem[0], bound_smem[3] + 10],
                                     f"{name}_warp",
                                     epa=warp_epa)
        bounds.append(bound_warp)
        if tile_shape is not None:
            grid = GridPlane2d(
                [0, 0, smem_shape[0] * scale, smem_shape[1] * scale],
                [scale * tile_shape[0], scale * tile_shape[1]],
                "black",
                width=2)
            grid2 = GridPlane2d(
                [0, 0, smem_shape[0] * scale, smem_shape[1] * scale],
                [scale * part_shape[0], scale * part_shape[1]],
                "magenta",
                width=1)

            with fig.layer(f"{name}_tile") as layer:
                layer.add_object(grid)
                layer.add_object(grid2)

                layer.move(bound_inp[2] + 10, bound_inp[1])

    bounds_arr = np.array(bounds)
    return np.array(
        [*bounds_arr[:, :2].min(axis=0), *bounds_arr[:, 2:].max(axis=0)])


def vis_gemm_output_2d(dtype_acc: dtypes.DType,
                       blocks: cudasim.Dim3,
                       threads: cudasim.Dim3,
                       fig_per_group: Dict[int, vis.figure.ImageFigure],
                       res,
                       name: str,
                       offset: List[float],
                       coord_input: bool = False):
    grouped_res = {}
    smem_shape = [0, 0]
    smem_save_epa = 0
    smem_load_epa = 0
    scale = GEMM_VIS_GLOBAL_SCALE
    for (bx, by, bz, tx, ty, tz), val in res.items():
        warp_id = (tx // 32)
        lane_id = (tx % 32)
        smem_save_coords = val["smem_save_coords"]
        smem_load_coords = val["smem_load_coords"]
        smem_shape = val["smem_shape"]
        smem_save_epa = val["smem_save_epa"]
        smem_load_epa = val["smem_load_epa"]
        group_id = warp_id * 32 + tx // 8  # 4 phase 128bit load
        group_id = threads.calc_offset(
            tx, ty, tz) + blocks.calc_offset(bx, by, bz) * threads.count()
        group_id = int(group_id)
        # group_id = (tx // 32) * 32 + (tx % 4) * 8 + (tx // 4)
        # group_id //= 8
        # group_id = warp_id
        # group_id //= 4
        # group_id = tx // 4
        # if tx % 2 != 0:
        #     continue
        if group_id not in grouped_res:
            grouped_res[group_id] = []
        grouped_res[group_id].append((smem_save_coords, smem_load_coords, tx))
    # smem shape to bank view
    bank_size_element_view = 128 // dtype_acc.itemsize()
    # smem_shape = [kernel.div_up(smem_shape[0] * smem_shape[1], bank_size_element_view), bank_size_element_view]
    grouped_res = OrderedDict(sorted(grouped_res.items(), key=lambda x: x[0]))
    fake_img = None
    bounds: List[List[float]] = []
    for group_id, v in grouped_res.items():
        v = transpose(v)
        smem_save_coords_list = v[0]
        smem_load_coords_list = v[1]
        group_size = len(v[0])
        if group_id not in fig_per_group:
            fig_per_group[group_id] = vis.figure.ImageFigure(0,
                                                             fake_img,
                                                             thread_id=v[-1],
                                                             group_id=group_id)
        fig = fig_per_group[group_id]
        bound_save = vis_area_coords(fig,
                                     smem_save_coords_list,
                                     group_size,
                                     smem_shape,
                                     offset,
                                     f"{name}_smem_save",
                                     epa=smem_save_epa)
        bounds.append(bound_save)
        bound_load = vis_area_coords(fig,
                                     smem_load_coords_list,
                                     group_size,
                                     smem_shape,
                                     [bound_save[2] + 10, bound_save[1]],
                                     f"{name}_smem_load",
                                     epa=smem_load_epa)
        bounds.append(bound_load)
    bounds_arr = np.array(bounds)
    return np.array(
        [*bounds_arr[:, :2].min(axis=0), *bounds_arr[:, 2:].max(axis=0)])


def _asdv_test_simt_python(coord_input: bool = False):
    with cudasim.enter_debug_context(True):

        main_cu = GemmMainUnitTest()
        for params in main_cu.all_params[:]:
            np.random.seed(12315)

            ker = gen_gemm_kernels(params)
            if params.algo != kernel.GemmAlgo.SimtDP4A:
                m = 256 + 32
                n = 256 + 40
                k = 24
                m = 32
                n = 16
                k = 16
                # m = max(params.ts[0], m)
                # n = max(params.ts[1], n)
                # k = max(params.ts[2], k)

                a = np.random.uniform(-1, 1, size=[m, k]).astype(
                    dtypes.get_npdtype(params.dtype_a))
                b = np.random.uniform(-1, 1, size=[k, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))

                c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
                c_dev_0 = a[:4] @ b[:, :4]
                c_dev_1 = a[16:20] @ b[:, :4]
                c_dev_2 = a[4:8] @ b[:, :4]

                acc_dev = np.concatenate([c_dev_0, c_dev_1])
                print(acc_dev.shape, acc_dev.mean(), acc_dev.max(),
                      acc_dev.min())
                print(np.concatenate([a[:4, :2], a[16:20, :2]]))
                c_dev_0 = a[:4, :2] @ b[:2, :4]
                c_dev_1 = a[16:20, :2] @ b[:2, :4]

                acc_dev = np.concatenate([c_dev_0, c_dev_1])
                print(acc_dev.shape, acc_dev.mean(), acc_dev.max(),
                      acc_dev.min())

            else:
                m = 256 + 32
                n = 256 + 40
                k = 56
                m = 64
                n = 128
                k = 32
                m = max(params.ts[0], m)
                n = max(params.ts[1], n)
                k = max(params.ts[2], k)
                print(m, n, k)
                a = np.random.randint(-5, 5, size=[m, k]).astype(np.int8)
                b = np.random.randint(-5, 5, size=[k, n]).astype(np.int8)
                # print("DATA GEN FINISH")
                dtype_np_c = dtypes.get_npdtype(params.dtype_c)
                c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                    dtypes.get_npdtype(params.dtype_c))
                c_dev_0 = a[:4] @ b[:, :4]
                c_dev_1 = a[16:20] @ b[:, :4]
                c_dev_2 = a[4:8] @ b[:, :4]
                print(params.trans_a, params.trans_b, params.trans_c)
                print(c_dev_0)
                print(a.T[:, :4])
                print(b[:, :8])

            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))
            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)

            if coord_input:
                if params.trans_a:
                    for i in range(k):
                        for j in range(m):
                            a_meta[i, j] = (i * m + j)
                else:
                    for i in range(m):
                        for j in range(k):
                            a_meta[i, j] = (i * k + j)
                if params.trans_b:
                    for i in range(n):
                        for j in range(k):
                            b_meta[i, j] = (i * k + j)
                else:
                    for i in range(k):
                        for j in range(n):
                            b_meta[i, j] = (i * n + j)

            a_tv = a.copy()
            b_tv = b.copy()
            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0])
            duration = time.time() - t

            vis_res = {}
            for k, v in vis_res_per_thread.items():
                for k2, v2 in v.items():
                    if k2 not in vis_res:
                        vis_res[k2] = {}
                    vis_res[k2][k] = v2

            fig_per_group: Dict[int, vis.figure.ImageFigure] = {}
            A_bound = vis_gemm_input_2d(a_tv,
                                        blocks,
                                        threads,
                                        fig_per_group,
                                        vis_res["InputA"],
                                        "A", [0, 0],
                                        coord_input=coord_input)
            B_bound = vis_gemm_input_2d(b_tv,
                                        blocks,
                                        threads,
                                        fig_per_group,
                                        vis_res["InputB"],
                                        "B", [0, A_bound[3] + 10],
                                        coord_input=coord_input)
            O_bound = vis_gemm_output_2d(params.dtype_acc, blocks, threads,
                                         fig_per_group, vis_res["Output"], "O",
                                         [0, B_bound[3] + 10])
            data = pickle.dumps(fig_per_group)
            vis_in_relay(list(fig_per_group.values()))
            # print(TestCase().assertAllClose(c_tv, c))
            print(c_tv.reshape(-1)[:10], c.reshape(-1)[:10])
            print(c_tv.reshape(-1)[-10:], c.reshape(-1)[-10:])
            print(a.mean(), b.mean(),
                  np.linalg.norm(c_tv - c), "Time=", duration)


def _asdv_test_volta_python(coord_input: bool):

    np.random.seed(12315)
    with cudasim.enter_debug_context(True):
        main_cu = GemmMainUnitTest()
        for params in main_cu.volta_params[:1]:
            ker = gen_gemm_kernels(params)
            m = 256 + 32
            n = 256 + 40
            k = 32
            m = 32
            n = 16
            k = 16
            m = max(params.ts[0], m)
            n = max(params.ts[1], n)
            k = max(params.ts[2], k)

            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))

            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)

            if coord_input:
                if params.trans_a:
                    for i in range(k):
                        for j in range(m):
                            a_meta[i, j] = (i * m + j)
                else:
                    for i in range(m):
                        for j in range(k):
                            a_meta[i, j] = (i * k + j)
                if params.trans_b:
                    for i in range(n):
                        for j in range(k):
                            b_meta[i, j] = (i * k + j)
                else:
                    for i in range(k):
                        for j in range(n):
                            b_meta[i, j] = (i * n + j)
            a_tv = a.copy()
            b_tv = b.copy()
            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0])
            duration = time.time() - t

            vis_res = {}
            for k, v in vis_res_per_thread.items():
                for k2, v2 in v.items():
                    if k2 not in vis_res:
                        vis_res[k2] = {}
                    vis_res[k2][k] = v2

            fig_per_group: Dict[int, vis.figure.ImageFigure] = {}
            A_bound = vis_gemm_input_2d(a_tv,
                                        blocks,
                                        threads,
                                        fig_per_group,
                                        vis_res["InputA"],
                                        "A", [0, 0],
                                        coord_input=coord_input)
            B_bound = vis_gemm_input_2d(b_tv,
                                        blocks,
                                        threads,
                                        fig_per_group,
                                        vis_res["InputB"],
                                        "B", [0, A_bound[3] + 10],
                                        coord_input=coord_input)
            O_bound = vis_gemm_output_2d(params.dtype_acc, blocks, threads,
                                         fig_per_group, vis_res["Output"], "O",
                                         [0, B_bound[3] + 10])

            # print(TestCase().assertAllClose(c_tv, c))
            # print(c_tv.reshape(-1)[:10] -  c.reshape(-1)[:10])
            # print(c_tv.reshape(-1)[-10:] -  c.reshape(-1)[-10:])

            print(params.get_algo_name(), a.mean(), np.linalg.norm(c_tv - c),
                  "Time=", duration)
            vis_in_relay(list(fig_per_group.values()))


def unittest_python():
    np.random.seed(12315)
    with cudasim.enter_debug_context(False):
        main_cu = GemmMainUnitTest()
        for params in main_cu.all_params:
            t = time.time()
            ker = gen_gemm_kernels(params)
            m = params.ts[0]
            n = params.ts[1]
            k = params.ts[2]

            if params.dtype_a == dtypes.int8:
                a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
                b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
                dtype_c = params.dtype_c.npdtype()
                c = (a.astype(dtype_c) @ b.astype(dtype_c)).astype(
                    dtypes.get_npdtype(params.dtype_c))

            else:
                a = np.random.uniform(-1, 1, size=[m, k]).astype(
                    dtypes.get_npdtype(params.dtype_a))
                b = np.random.uniform(-1, 1, size=[k, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))
                c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))
            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)
            a_tv = a.copy()
            b_tv = b.copy()
            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0])
            duration = time.time() - t
            print(params.get_algo_name(), a.mean(), np.linalg.norm(c_tv - c),
                  "Time=", duration)


def _asdv_test_turing_python(coord_input: bool = False):
    np.random.seed(12315)
    with cudasim.enter_debug_context(True, 3):
        main_cu = GemmMainUnitTest()
        print(len(main_cu.all_params))

        for params in main_cu.all_params[:]:
            ker = gen_gemm_kernels(params)
            print(ker.get_algo_name())

            # print("START", params.get_algo_name())
            m = 256 + 32
            n = 256 + 40
            k = 32
            m, n, k = 0, 0, 0
            # m = 80
            # n = 80
            # k = 160
            m = max(params.ts[0], m)
            n = max(params.ts[1], n)
            k = max(params.ts[2], k)
            print(m, n, k)
            if params.dtype_a == dtypes.int8:
                a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
                b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
                dtype_c = params.dtype_c.npdtype()
                c = -(a.astype(np.float16) @ b.astype(np.float16)).astype(
                    dtypes.get_npdtype(params.dtype_c))

            else:
                a = np.random.uniform(-1, 1, size=[m, k]).astype(
                    dtypes.get_npdtype(params.dtype_a))
                b = np.random.uniform(-1, 1, size=[k, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))
                # a[:, 32:] = 0
                # b[32:] = 0

                c = -(a @ b).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))
            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)

            if coord_input:
                if params.trans_a:
                    for i in range(k):
                        for j in range(m):
                            a_meta[i, j] = (i * m + j)
                else:
                    for i in range(m):
                        for j in range(k):
                            a_meta[i, j] = (i * k + j)
                if params.trans_b:
                    for i in range(n):
                        for j in range(k):
                            b_meta[i, j] = (i * k + j)
                else:
                    for i in range(k):
                        for j in range(n):
                            b_meta[i, j] = (i * n + j)

            a_tv = a.copy()
            b_tv = b.copy()
            cc_tv = np.zeros_like(c)

            c_tv = c.copy()
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0],
                split_k_slices=1)
            duration = time.time() - t
            vis_res = {}
            for k, v in vis_res_per_thread.items():
                for k2, v2 in v.items():
                    if k2 not in vis_res:
                        vis_res[k2] = {}
                    vis_res[k2][k] = v2

            fig_per_group: Dict[int, vis.figure.ImageFigure] = {}
            A_bound = vis_gemm_input_2d(a_tv,
                                        blocks,
                                        threads,
                                        fig_per_group,
                                        vis_res["InputA"],
                                        "A", [0, 0],
                                        coord_input=coord_input)
            B_bound = vis_gemm_input_2d(b_tv,
                                        blocks,
                                        threads,
                                        fig_per_group,
                                        vis_res["InputB"],
                                        "B", [0, A_bound[3] + 10],
                                        coord_input=coord_input)
            O_bound = vis_gemm_output_2d(params.dtype_acc, blocks, threads,
                                         fig_per_group, vis_res["Output"], "O",
                                         [0, B_bound[3] + 10])

            # print(TestCase().assertAllClose(c_tv, c)) 
            # print(c_tv.reshape(-1)[:10], c.reshape(-1)[:10])
            # print(c_tv.reshape(-1)[-10:] -  c.reshape(-1)[-10:])

            print(ker.get_algo_name(), a.mean(), b.mean(), c.mean(),
                  np.linalg.norm(c_tv - c), "Time=", duration)
            
            print("max error is ", np.abs(c_tv).max())
            if(np.abs(c_tv).max() < 1e-2):
                print("PASSED")
            else:
                print("NOT PASS")
                assert(0)

            vis_in_relay(list(fig_per_group.values()))


if __name__ == "__main__":
    # fig = vis.figure.PointCloudFigure(0, np.zeros((1, 3)))
    # with fig.layer("WTF") as layer:
    #     coords = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=np.float32)

    #     layer.add_object(GridPlane([0, 0, 128, 8], [0.5, 0.5], 'green'))
    #     layer.add_object(Coords(coords, [0.5, 0.5], 4, 'red'))

    # vis_in_relay([fig])
    # unittest_python()
    # _asdv_test_simt_python(True)
    _asdv_test_turing_python(True)
    # _asdv_test_volta_python(True)
