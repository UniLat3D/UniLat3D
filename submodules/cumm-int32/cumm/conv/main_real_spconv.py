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

import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pccm
import torch
import torch.nn.functional as F
from spconv.core_cc.csrc.sparse.all import SpconvOps

from cumm import dtypes
from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT
from cumm.conv.bases import NCHW, NHWC, ConvIterAlgo, ConvOpType
from cumm.conv.main import ConvMainUnitTest, gen_gemm_kernels
from cumm.conv.params import ConvProblem
from cumm.gemm import kernel
from cumm.gemm.codeops import div_up

os.environ["CUMM_DEBUG"] = "1"


def _asdv_test_spconv():
    limit_input_n = 16384
    limit_input_n = None
    with (Path.home() /
          Path("Projects/spconv-release/spconv/test/data/test_spconv.pkl")
          ).open("rb") as f:

        # with (Path.home() / Path("OneDrive/dev/spconv-release/spconv/test/data/test_spconv.pkl")).open("rb") as f:
        voxels_np, indices_np, spatial_shape = pickle.load(f)
        voxels_np = voxels_np[:limit_input_n]
        indices_np = indices_np[:limit_input_n]
        voxels = tv.from_numpy(voxels_np).cuda()
        indices = tv.from_numpy(indices_np).cuda()
    print(spatial_shape, indices_np.shape)
    ndim = 3
    ksize = [3, 3, 3]
    kv = int(np.prod(ksize))
    indices_with_bs = indices_np
    indices = tv.from_numpy(indices_with_bs).cuda()
    out_indices = tv.zeros([indices.dim(0) * kv, 4], tv.int32, 0)
    indice_num_per_loc = tv.zeros([kv], tv.int32, 0)
    padding = [0] * ndim
    stride = [1] * ndim
    dilation = [1] * ndim

    points = voxels.view([-1, 3])
    hashdata_subm = tv.zeros([points.dim(0) * 2], tv.custom64, 0)
    indice_pairs_np = np.full([2, kv, indices.dim(0)], -1, np.int32)
    indice_pairs = tv.from_numpy(indice_pairs_np).cuda()
    indice_pairs_mask_np = np.full([2, indices.dim(0)], (1 << (kv // 2)),
                                   np.uint32)

    indice_pairs_mask = tv.from_numpy(indice_pairs_mask_np).cuda()

    indice_pairs_mask2 = tv.empty([1, indices.dim(0)], tv.uint32, 0)

    indice_pairs_igemm = indice_pairs.clone()
    out_act = SpconvOps.generate_subm_conv_inds(indices, hashdata_subm,
                                                indice_pairs, out_indices,
                                                indice_num_per_loc, 1,
                                                spatial_shape, ksize,
                                                [1, 1, 1])
    indice_pairs_np = indice_pairs.cpu().numpy()
    indice_num_per_loc_np = indice_num_per_loc.cpu().numpy()
    out_act = SpconvOps.generate_subm_conv_inds(
        indices, hashdata_subm, indice_pairs_igemm, out_indices,
        indice_num_per_loc, 1, spatial_shape, ksize, [1, 1, 1],
        indice_pairs_mask, False)
    out_act = SpconvOps.generate_subm_conv_inds(
        indices, hashdata_subm, indice_pairs_igemm, out_indices,
        indice_num_per_loc, 1, spatial_shape, ksize, [1, 1, 1],
        indice_pairs_mask2, False)

    indice_pairs_mask_np = indice_pairs_mask.cpu().numpy()
    indice_pairs_igemm_np = indice_pairs_igemm.cpu().numpy()

    # print(indice_pairs_mask_np.max(), indice_pairs_mask_np.min(), indice_pairs_mask_np.mean())
    # with (Path.home() / "inds_debug.pkl").open("rb") as f:
    #     indices_with_bs_ref = pickle.load(f)
    # print(np.linalg.norm(indices_with_bs_ref - indice_pairs_igemm_np))

    # return

    indice_num_per_loc_np = indice_num_per_loc.cpu().numpy()
    print(indice_num_per_loc_np)
    # return
    mask_splits = []
    mask_split_cpus = []

    mask_split_indss = []
    indice_pairs_mask_cpu = indice_pairs_mask.cpu().numpy()
    # indice_pairs_mask.fill_int_((1 << 27) - 1)
    indice_pairs_mask2_2 = indice_pairs_mask2.clone()
    masks_np = np.array([0b11111111111111, (0b1111111111111) << 14],
                        dtype=np.uint32)
    masks_tv = tv.from_numpy(masks_np)
    # for x in range(10):
    #     print("-----------")
    #     for j in range(2):
    #         mask_split = indice_pairs_mask[j].clone()
    #         mask_split_inds = SpconvOps.sort_1d_by_key(mask_split)
    #         mask_splits.append(mask_split)
    #         mask_split_indss.append(mask_split_inds)
    indice_pairs_mask2s = [indice_pairs_mask2, indice_pairs_mask2_2]
    for x in range(2):
        print("-----------")
        for j in range(2):
            mask_split = indice_pairs_mask2s[j].clone()
            mask_split_inds = SpconvOps.sort_1d_by_key_split(
                mask_split, masks_tv.slice_first_axis(j, j + 1))
            mask_splits.append(mask_split)
            mask_split_indss.append(mask_split_inds)

    mask_output = tv.empty([2, div_up(indices.dim(0), 32)], tv.uint32, 0)
    # return
    # from spconv.core_cc import arrayref
    np.random.seed(12315)
    main_cu = ConvMainUnitTest()
    lib = pccm.builder.build_pybind([main_cu],
                                    Path(__file__).parent / "imgemm_test",
                                    includes=[
                                        PACKAGE_ROOT / "include",
                                    ],
                                    namespace_root=PACKAGE_ROOT / "cumm",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_unittest_conv",
                                    pybind_file_suffix=".cc",
                                    verbose=True,
                                    disable_anno=True)

    lib_object = lib.cumm.conv.main.ConvMainUnitTest()
    algo_cls = lib.cumm.conv.main.ConvAlgoDesp
    params_cls = lib.cumm.conv.main.ConvParams
    # NKRS @ KRSC
    mask_width = -1
    for params in main_cu.all_params:
        print("START", params.get_algo_name())
        if not params.mask_sparse:
            continue
        # if not params.op_type == ConvOpType.kForward:
        #     continue
        ker = gen_gemm_kernels(params)
        if params.op_type == ConvOpType.kForward:
            mask_width = params.ts[0]
        # NCHW -> KCRS @ NCRSPQ = NKPQ
        C = 128
        K = 128
        if params.dtype_a == dtypes.int8:
            inp = np.random.randint(-1, 1, size=[points.shape[0],
                                                 C]).astype(np.int8)
            weight = np.random.randint(-1, 1, size=[K, *ksize,
                                                    C]).astype(np.int8)
            output = np.random.randint(-1, 1, size=[
                points.shape[0], K
            ]).astype(dtypes.get_npdtype(params.dtype_output))
        else:
            inp = np.random.uniform(-1, 1, size=[points.shape[0], C]).astype(
                dtypes.get_npdtype(params.dtype_input))
            weight = np.random.uniform(-1, 1, size=[K, *ksize, C]).astype(
                dtypes.get_npdtype(params.dtype_weight))
            output = np.random.uniform(-1, 1, size=[
                points.shape[0], K
            ]).astype(dtypes.get_npdtype(params.dtype_output))
        weight_ref = weight.transpose(1, 2, 3, 0, 4)
        weight_ref = np.ascontiguousarray(weight_ref).reshape(-1, K, C)

        if params.op_type == ConvOpType.kBackwardInput:
            inp_tv = tv.zeros(inp.shape, params.dtype_input.tv_dtype, 0)
        else:
            inp_tv = tv.from_numpy(inp).cuda()
        if params.op_type == ConvOpType.kBackwardWeight:
            weight_tv = tv.zeros(weight.shape, params.dtype_weight.tv_dtype, 0)
        else:
            weight_tv = tv.from_numpy(weight).cuda()
        # _ = tv.zeros([5000, 10], tv.float32, 0)
        if params.op_type == ConvOpType.kForward:
            output_tv = tv.zeros(output.shape, params.dtype_output.tv_dtype, 0)
        else:
            output_tv = tv.from_numpy(output).cuda()
        if params.op_type == ConvOpType.kForward:
            indice_pairs = indice_pairs_igemm[0]
        elif params.op_type == ConvOpType.kBackwardInput:
            indice_pairs = indice_pairs_igemm[1]
        else:
            indice_pairs = indice_pairs_igemm[0]
        # with (Path.home() / "debug_spconv.pkl").open("rb") as f:
        #     indice_pairs_np_x, mask_split_indss_np = pickle.load(f)
        #     indice_pairs = tv.from_numpy(indice_pairs_np_x).cuda()
        #     mask_split_indss = [tv.from_numpy(x).cuda() for x in mask_split_indss_np]

        spk = 1
        if params.op_type == ConvOpType.kBackwardWeight:
            # TODO support splitk parallel
            spk = 32
        algo = algo_cls(ker.problem.ndim, ker.problem.op_type.value)
        algo.tile_shape = params.ts
        algo.warp_tile_shape = params.wts
        algo.num_stage = params.num_stage
        algo.dacc = params.dtype_acc.tv_dtype
        algo.dcomp = params.dtype_comp.tv_dtype
        algo.algo = params.algo.value
        algo.trans_a = params.trans_a
        algo.trans_b = params.trans_b
        algo.trans_c = params.trans_c
        algo.element_per_access_a = ker.input_spec.input_iter_a.element_per_acc
        algo.element_per_access_b = ker.input_spec.input_iter_b.element_per_acc
        algo.element_per_access_c = ker.output_spec.out_iter.element_per_acc
        algo.split_k_serial = params.splitk_serial
        algo.dtype_a = params.dtype_a.tv_dtype
        algo.dtype_b = params.dtype_b.tv_dtype
        algo.dtype_c = params.dtype_c.tv_dtype
        algo.mask_sparse = params.mask_sparse
        algo.increment_k_first = params.increment_k_first
        algo.access_per_vector = params.access_per_vector

        if params.tensorop is not None:
            algo.tensorop = params.tensorop.shape
        params_cpp = params_cls(ker.problem.ndim, ker.problem.op_type.value)
        params_cpp.conv_algo_desp = algo
        params_cpp.split_k_slices = spk
        params_cpp.input = inp_tv
        params_cpp.weight = weight_tv.view([K, -1, C])
        params_cpp.output = output_tv
        params_cpp.padding = padding
        params_cpp.stride = stride
        params_cpp.dilation = dilation
        params_cpp.mask_width = mask_width

        params_cpp.beta = 0.0
        tv.zeros([1, 2], device=0)
        # print("START")

        for i in range(1):
            # output_tv.zero_()
            torch.cuda.synchronize()
            t = time.time()
            cnt = 2
            # if params.op_type == ConvOpType.kBackwardWeight:
            #     cnt = 1
            for j in range(cnt):

                beta = 1 if j == 1 else 0
                if params.op_type == ConvOpType.kBackwardWeight:
                    if j == 0:
                        params_cpp.mask_filter = (0b11111111111111)
                    else:
                        params_cpp.mask_filter = ((0b1111111111111) << 14)

                    mask_op = mask_output[j]
                else:
                    params_cpp.mask_filter = 0xffffffff

                    mask_op = mask_splits[j]
                if params.op_type == ConvOpType.kBackwardInput:
                    params_cpp.reverse_mask = True
                params_cpp.mask = mask_op
                params_cpp.mask_argsort = mask_split_indss[j]
                params_cpp.indices = indice_pairs
                params_cpp.mask_output = mask_output[j]
                params_cpp.beta = beta
                lib_object.implicit_gemm2(params_cpp)

                # lib_object.implicit_gemm(inp_tv, weight_tv, output_tv, padding, stride, dilation,
                #     ndim=ndim, iter_algo_=params.iter_algo.value, op_type_=params.op_type.value,
                #     i_ltype_=params.layout_desp_input.layout_type.value,
                #     w_ltype_=params.layout_desp_weight.layout_type.value,
                #     o_ltype_=params.layout_desp_output.layout_type.value,
                #     ts=params.ts, wts=params.wts, num_stage=params.num_stage, dacc=params.dtype_acc.tv_dtype,
                #     dcomp=params.dtype_comp.tv_dtype, algo=params.algo.value, tensorop=[0, 0, 0], split_k_slices=spk,
                #     mask_sparse=True, increment_k_first=params.increment_k_first,
                #     mask=mask_op, mask_argsort=mask_split_indss[j], indices=indice_pairs,
                #     beta=beta, mask_output=mask_output[j])  # type: tv.Tensor
            torch.cuda.synchronize()
            # print(time.time() - t, params.op_type)
        op_duration = 0
        if params.op_type == ConvOpType.kForward:
            output_ref = np.zeros_like(output, dtype=np.float32)
            # ref algorithm
            for filter_offset in range(kv):
                if filter_offset > kv // 2:
                    nhot = indice_num_per_loc_np[kv - 1 - filter_offset]
                elif filter_offset == kv // 2:
                    nhot = points.shape[0]
                else:
                    nhot = indice_num_per_loc_np[filter_offset]
                a_inds = indice_pairs_np[0][filter_offset][:nhot]
                c_inds = indice_pairs_np[1][filter_offset][:nhot]
                # print(a_inds_cpu[:10])
                a = inp[a_inds]
                cc = a.astype(np.float32) @ weight_ref[filter_offset].T.astype(
                    np.float32)
                output_ref[c_inds] += cc

            output_cpu = output_tv.cpu().numpy()
            if params.dtype_a.itemsize() == 1:
                output_cpu = output_cpu.astype(np.float32)
            duration = time.time() - t
            # RS_str = "_".join(map(str, RS))
            # CK_str = f"{C}_{K}"
            # with (Path.home() / f"temp_res_{RS_str}_{CK_str}.pkl").open("rb") as f:
            #     out_feature = pickle.load(f).reshape(-1)
            my = output_cpu.reshape(-1)
            # print(np.linalg.norm(out_feature - my))
            print("ERROR", np.linalg.norm(output_ref.reshape(-1) - my))

            # print(params.get_algo_name(), output_ref.mean(), output_ref.max(), output_ref.min(),
            #     "Time=", op_duration)

            # print(params.get_algo_name(), output_cpu.mean(), output_cpu.max(), output_cpu.min(),
            #     "Time=", op_duration)
            # print(params.get_algo_name(), np.linalg.norm(out_feature - my),
            #     "Time=", op_duration)

        elif params.op_type == ConvOpType.kBackwardInput:
            dinput_ref = np.zeros_like(inp, dtype=np.float32)
            # ref algorithm
            for filter_offset in range(kv):
                if filter_offset > kv // 2:
                    nhot = indice_num_per_loc_np[kv - 1 - filter_offset]
                elif filter_offset == kv // 2:
                    nhot = points.shape[0]
                else:
                    nhot = indice_num_per_loc_np[filter_offset]
                a_inds = indice_pairs_np[1][filter_offset][:nhot]
                c_inds = indice_pairs_np[0][filter_offset][:nhot]

                # print(a_inds_cpu[:10])
                a = output[a_inds]
                # NK @ KC
                cc = a.astype(np.float32) @ weight_ref[filter_offset].astype(
                    np.float32)
                dinput_ref[c_inds] += cc

            din_cpu = inp_tv.cpu().numpy()
            print("ERROR",
                  np.linalg.norm(din_cpu.reshape(-1) - dinput_ref.reshape(-1)))
            # print(din_cpu.reshape(-1))
            # print(dinput_ref.reshape(-1))
            duration = time.time() - t
            # print(params.get_algo_name(), din_cpu.mean(), din_cpu.max(), din_cpu.min(),
            #     "Time=", op_duration)
        else:
            dw_ref = np.zeros_like(weight_ref, dtype=np.float32)  # KV, K, C
            for filter_offset in range(kv):
                if filter_offset > kv // 2:
                    nhot = indice_num_per_loc_np[kv - 1 - filter_offset]
                elif filter_offset == kv // 2:
                    nhot = points.shape[0]
                else:
                    nhot = indice_num_per_loc_np[filter_offset]
                o_inds = indice_pairs_np[1][filter_offset][:nhot]
                i_inds = indice_pairs_np[0][filter_offset][:nhot]
                # print(a_inds_cpu[:10])
                out_gather = output[o_inds]  # [N, K]
                inp_gather = inp[i_inds]  # [N, C]
                # KN @ NC
                dw_res = out_gather.astype(np.float32).T @ inp_gather.astype(
                    np.float32)
                # if filter_offset == 13:
                #     print(dw_res.mean(), dw_res.min(), dw_res.max())
                #     ref_res = output.T @ inp
                #     print(ref_res.mean(), ref_res.min(), ref_res.max())

                dw_ref[filter_offset] = dw_res
            indice_pairs_np_test = indice_pairs.cpu().numpy()
            # print(indice_pairs_np_test[0])
            dw_ref_kcrs = dw_ref.transpose(1, 0, 2)
            dw_cpu = weight_tv.cpu().numpy().reshape(K, np.prod(ksize), C)
            print("ERROR",
                  np.linalg.norm(dw_cpu.reshape(-1) - dw_ref_kcrs.reshape(-1)))
            # print("ERROR2", np.linalg.norm(dw_cpu.reshape(-1)[:448] - dw_ref_kcrs.reshape(-1)[:448]))
            # print("RTX", indices_np.shape)
            # print(dw_cpu[:, 13].reshape(-1)[:50] - dw_ref_kcrs[:, 13].reshape(-1)[:50])
            # print(dw_cpu.reshape(-1)[:10],dw_cpu.reshape(-1)[448:450])
            # print(dw_ref_kcrs.reshape(-1)[:10],dw_ref_kcrs.reshape(-1)[448:450])
            duration = time.time() - t
            # print(params.get_algo_name(), dw_cpu.mean(), dw_cpu.max(), dw_cpu.min(),
            #     "Time=", op_duration)


if __name__ == "__main__":
    _asdv_test_spconv()
