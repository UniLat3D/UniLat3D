// Copyright 2024 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "nvrtc_bases.h"
#include "utils.h"
#include <tensorview/profile/all.h>
#include <tensorview/tensor.h>
#include "constants.h"
namespace tv {
namespace gemm {

struct GemmAlgoDesp {
  int dtype_a;
  int dtype_b;
  int dtype_c;
  int trans_a_;
  int trans_b_;
  int trans_c_;
  std::array<int, 3> tile_shape;
  std::array<int, 3> warp_tile_shape;
  int num_stage;
  int dacc;
  int dcomp;
  std::string algo;
  std::array<int, 3> tensorop = std::array<int, 3>{};
  int split_k_serial_ = 0;
  int split_k_parallel_ = 0;
  ShuffleStrideType shuffle_type = ShuffleStrideType::kNoShuffle;
  int element_per_access_a = -1;
  int element_per_access_b = -1;
  int element_per_access_c = -1;
  int access_per_vector = 1;
  bool is_nvrtc = false;
  std::tuple<int, int> min_arch = std::tuple<int, int>{0, 0};
  GemmAlgoDesp()
      : dtype_a(int(tv::unknown)), dtype_b(int(tv::unknown)),
        dtype_c(int(tv::unknown)), trans_a_(-1), trans_b_(-1), trans_c_(-1),
        tile_shape({-1, -1, -1}), warp_tile_shape({-1, -1, -1}), num_stage(-1),
        dacc(int(tv::unknown)), dcomp(int(tv::unknown)), algo(""),
        tensorop({-1, -1, -1}), shuffle_type(ShuffleStrideType::kNoShuffle),
        split_k_serial_(0), split_k_parallel_(0), element_per_access_a(-1),
        element_per_access_b(-1), element_per_access_c(-1),
        access_per_vector(1), min_arch({0, 0}) {}
  std::string __repr__() {

    check_valid();
    std::stringstream ss;
    ss << algo << "_" << tv::dtype_short_str(dtype_a)
       << tv::dtype_short_str(dtype_b) << tv::dtype_short_str(dtype_c)
       << tv::dtype_short_str(dacc) << tv::dtype_short_str(dcomp);
    ss << (trans_a() ? "n" : "t") << (trans_b() ? "n" : "t")
       << (trans_c() ? "n" : "t");
    ss << "_m" << tile_shape[0] << "n" << tile_shape[1] << "k" << tile_shape[2];
    ss << "m" << warp_tile_shape[0] << "n" << warp_tile_shape[1] << "k"
       << warp_tile_shape[2];
    ss << "A" << access_per_vector;
    if (tensorop[0] != -1) {
      ss << "T" << tensorop[0] << tensorop[1] << tensorop[2];
    }
    ss << "_" << num_stage;
    ss << (split_k_serial() ? 1 : 0) << (split_k_parallel() ? 1 : 0);
    if (shuffle_type != ShuffleStrideType::kNoShuffle) {
      ss << "_S" << static_cast<int>(shuffle_type);
    }
    return ss.str();
  }
  bool split_k_serial() { return split_k_serial_ == 1; }
  void split_k_serial_set(bool val) { split_k_serial_ = val ? 1 : 0; }
  bool split_k_parallel() { return split_k_parallel_ == 1; }
  void split_k_parallel_set(bool val) { split_k_parallel_ = val ? 1 : 0; }
  void check_valid() {

    TV_ASSERT_RT_ERR(trans_a_ != -1 && trans_b_ != -1 && trans_c_ != -1 &&
                         !algo.empty(),
                     "trans_a, trans_b, trans_c and algo must be set");
    for (int i = 0; i < 3; ++i) {
      TV_ASSERT_RT_ERR(
          tile_shape[i] > 0 && warp_tile_shape[i] > 0,
          "tile_shape and warp_tile_shape must be set, but they are",
          tile_shape, warp_tile_shape);
    }
    if (algo != "Simt" && algo != "SimtDP4A" && algo != "SimtDP2A") {
      // tensor op must not empty
      for (int i = 0; i < 3; ++i) {
        TV_ASSERT_RT_ERR(tensorop[i] > 0, "tensorop must be set, but they are",
                         tensorop);
      }
    }
    TV_ASSERT_RT_ERR(dtype_a != int(tv::unknown) &&
                         dtype_b != int(tv::unknown) &&
                         dtype_c != int(tv::unknown),
                     "dacc and dcomp must be set to valid value");
    TV_ASSERT_RT_ERR(dacc != int(tv::unknown) && dcomp != int(tv::unknown),
                     "dacc and dcomp must be set to valid value");
    TV_ASSERT_RT_ERR(num_stage > 0, "num_stage must larger than zero");
  }
  bool trans_a() { return trans_a_ == 1; }
  void trans_a_set(bool val) { trans_a_ = val ? 1 : 0; }
  bool trans_b() { return trans_b_ == 1; }
  void trans_b_set(bool val) { trans_b_ = val ? 1 : 0; }
  bool trans_c() { return trans_c_ == 1; }
  void trans_c_set(bool val) { trans_c_ = val ? 1 : 0; }
  int query_workspace_size(int m, int n, int k, int split_k_slices) {

    auto logical_tile_count = get_logical_tile_count(
        m, n, k, tile_shape[0], tile_shape[1], split_k_slices);
    int workspace_size = 0;
    if (split_k_slices > 1) {
      if (split_k_serial()) {
        workspace_size =
            sizeof(int) * logical_tile_count[0] * logical_tile_count[1];
      } else if (split_k_parallel()) {
        workspace_size = tv::detail::sizeof_dtype(tv::DType(dacc)) * m * n *
                         logical_tile_count[2];
      } else {
        TV_THROW_INVALID_ARG("not impemented");
      }
    }
    return workspace_size;
  }
  bool supported(int m, int n, int k) {

    bool res = true;
    auto lda = trans_a() ? m : k;
    auto ldb = trans_b() ? k : n;
    auto ldc = trans_c() ? m : n;
    if (element_per_access_a > 0) {
      res &= lda % element_per_access_a == 0;
    }
    if (element_per_access_b > 0) {
      res &= ldb % element_per_access_b == 0;
    }
    if (element_per_access_c > 0) {
      res &= ldc % element_per_access_c == 0;
    }
    return res;
  }
  bool supported_ldx(int lda, int ldb, int ldc) {

    bool res = true;
    if (element_per_access_a > 0) {
      res &= lda % element_per_access_a == 0;
    }
    if (element_per_access_b > 0) {
      res &= ldb % element_per_access_b == 0;
    }
    if (element_per_access_c > 0) {
      res &= ldc % element_per_access_c == 0;
    }
    return res;
  }
};

struct ConvAlgoDesp : public GemmAlgoDesp {
  int ndim;
  ConvOpType op_type;
  ConvIterAlgo iter_algo;
  ConvLayoutType layout_i;
  ConvLayoutType layout_w;
  ConvLayoutType layout_o;
  int interleave_i;
  int interleave_w;
  int interleave_o;
  bool mask_sparse = false;
  bool increment_k_first = false;
  std::array<int, 3> conv2gemm_inds;
  std::array<int, 3> gemm2conv_inds;
  bool dynamic_mask = false;
  bool is_int8_inference = false;
  ConvAlgoDesp(int ndim, ConvOpType op_type)
      : GemmAlgoDesp(), ndim(ndim), op_type(op_type),
        iter_algo(ConvIterAlgo::kOptimized),
        layout_i(ConvLayoutType::kChannelLast),
        layout_w(ConvLayoutType::kChannelLast),
        layout_o(ConvLayoutType::kChannelLast), interleave_i(1),
        interleave_w(1), interleave_o(1),
        conv2gemm_inds(conv_iwo_012_to_abc(op_type)),
        gemm2conv_inds(gemm_abc_012_to_iwo(op_type)) {}
  std::string __repr__() {

    check_valid();
    std::stringstream ss;
    ss << GemmAlgoDesp::__repr__();
    ss << "_C" << ndim << static_cast<int>(op_type)
       << static_cast<int>(iter_algo);
    std::string layout_i_str =
        layout_i == ConvLayoutType::kChannelFirst ? "F" : "L";
    std::string layout_w_str =
        layout_w == ConvLayoutType::kChannelFirst ? "F" : "L";
    std::string layout_o_str =
        layout_o == ConvLayoutType::kChannelFirst ? "F" : "L";
    if (interleave_i > 1) {
      layout_i_str += std::to_string(interleave_i);
    }
    if (interleave_w > 1) {
      layout_w_str += std::to_string(interleave_w);
    }
    if (interleave_o > 1) {
      layout_o_str += std::to_string(interleave_o);
    }
    ss << layout_i_str << layout_w_str << layout_o_str;
    if (mask_sparse) {
      ss << "_" << (increment_k_first ? "SK" : "SF");
    }
    if (dynamic_mask){
      ss << "D";
    }
    if (is_int8_inference){
      ss << "_S8";
    }
    return ss.str();
  }
  static std::array<int, 3> conv_iwo_012_to_abc(ConvOpType op_type) {

    if (op_type == ConvOpType::kForward) {
      return {0, 1, 2};
    }
    if (op_type == ConvOpType::kBackwardInput) {
      return {2, 1, 0};
    }
    if (op_type == ConvOpType::kBackwardWeight) {
      return {1, 2, 0};
    }
    TV_THROW_RT_ERR("unknown op type", static_cast<int>(op_type));
  }
  static std::array<int, 3> gemm_abc_012_to_iwo(ConvOpType op_type) {

    if (op_type == ConvOpType::kForward) {
      return {0, 1, 2};
    }
    if (op_type == ConvOpType::kBackwardInput) {
      return {2, 1, 0};
    }
    if (op_type == ConvOpType::kBackwardWeight) {
      return {2, 0, 1};
    }
    TV_THROW_RT_ERR("unknown op type", static_cast<int>(op_type));
  }
  int dtype_input() {

    std::array<int, 3> dtypes{dtype_a, dtype_b, dtype_c};
    return dtypes[conv2gemm_inds[0]];
  }
  int dtype_weight() {

    std::array<int, 3> dtypes{dtype_a, dtype_b, dtype_c};
    return dtypes[conv2gemm_inds[1]];
  }
  int dtype_output() {

    std::array<int, 3> dtypes{dtype_a, dtype_b, dtype_c};
    return dtypes[conv2gemm_inds[2]];
  }
  bool supported(int m, int n, int k, int C, int K, int mask_width) {

    bool res = GemmAlgoDesp::supported(m, n, k);
    if (mask_sparse) {
      if (op_type == ConvOpType::kForward) {
        // NC -> NRSC @ KRSC
        res &= C % element_per_access_a == 0;
      } else if (op_type == ConvOpType::kBackwardInput) {
        // NK -> NRSK @ KRSC -> RSKC
        res &= K % element_per_access_a == 0;
      } else {
        // NK @ NC -> NRSC
        // we must ensure every k iteration only have one mask (from forward),
        res &= mask_width % tile_shape[2] == 0;
        res &= K % element_per_access_a == 0;
        res &= C % element_per_access_b == 0;
      }
    }
    return res;
  }
  int query_conv_workspace_size(int m, int n, int k, int split_k_slices,
                                int kv) {

    if (!mask_sparse) {
      return query_workspace_size(m, n, k, split_k_slices);
    }
    auto logical_tile_count = get_spconv_logical_tile_count(
        m, n, k, tile_shape[0], tile_shape[1], split_k_slices, kv, op_type);
    int workspace_size = 0;
    if (split_k_slices > 1) {
      if (split_k_serial()) {
        workspace_size =
            sizeof(int) * logical_tile_count[0] * logical_tile_count[1];
      } else if (split_k_parallel()) {
        workspace_size = tv::detail::sizeof_dtype(tv::DType(dacc)) * m * n *
                         logical_tile_count[2];
      } else {
        TV_THROW_INVALID_ARG("not impemented");
      }
    }
    return workspace_size;
  }
  bool supported_ldx_conv(int ldi, int ldw, int ldo) {

    bool res = true;
    std::array<int, 3> epas{element_per_access_a, element_per_access_b,
                            element_per_access_c};
    int epa_i = epas[conv2gemm_inds[0]];
    int epa_w = epas[conv2gemm_inds[1]];
    int epa_o = epas[conv2gemm_inds[2]];
    if (epa_i > 0) {
      res &= ldi % epa_i == 0;
    }
    if (epa_w > 0) {
      res &= ldw % epa_w == 0;
    }
    if (epa_o > 0) {
      res &= ldo % epa_o == 0;
    }
    return res;
  }
};

struct GemmParams {
  GemmAlgoDesp algo_desp;
  tv::Tensor a;
  tv::Tensor b;
  tv::Tensor c;
  tv::Tensor d = tv::Tensor();

  int split_k_slices = 1;
  tv::Tensor workspace = tv::Tensor();
  tv::Tensor a_inds = tv::Tensor();
  tv::Tensor b_inds = tv::Tensor();
  tv::Tensor c_inds = tv::Tensor();
  float alpha;
  float beta;
  float act_alpha;
  float act_beta;
  Activation act_type = Activation::kNone;
  std::uintptr_t stream;
  tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false);
  tv::gemm::NVRTCParams nvrtc_params = tv::gemm::NVRTCParams();
  GemmParams(tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false))
      : a(tv::Tensor()), b(tv::Tensor()), c(tv::Tensor()), split_k_slices(1),
        workspace(tv::Tensor()), a_inds(tv::Tensor()), b_inds(tv::Tensor()),
        c_inds(tv::Tensor()), alpha(1.0), beta(0.0), stream(0), timer(timer) {}
  void check_valid() {

    algo_desp.check_valid();
    TV_ASSERT_RT_ERR(!a.empty() && !b.empty() && !c.empty(),
                     "a,b,c must not empty");
    if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAC) {
      TV_ASSERT_RT_ERR(!c_inds.empty(), "a_inds,c_inds tensor must not empty");
    } else if (algo_desp.shuffle_type ==
               tv::gemm::ShuffleStrideType::kShuffleAB) {
      TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(),
                       "a_inds,b_inds tensor must not empty");
    }
  }
  tv::Tensor a_get() { return a; }
  void a_set(tv::Tensor val) {

    a = val;
    algo_desp.dtype_a = int(a.dtype());
  }
  tv::Tensor b_get() { return b; }
  void b_set(tv::Tensor val) {

    b = val;
    algo_desp.dtype_b = int(b.dtype());
  }
  tv::Tensor c_get() { return c; }
  void c_set(tv::Tensor val) {

    c = val;
    algo_desp.dtype_c = int(c.dtype());
  }
  tv::Tensor d_get() { return d; }
  void d_set(tv::Tensor val) {
    TV_ASSERT_RT_ERR(c.dtype() == val.dtype(), "d dtype must equal to c");
    d = val;
  }

};

struct ConvParams {
  ConvAlgoDesp conv_algo_desp;
  tv::Tensor input;
  tv::Tensor weight;
  tv::Tensor output;
  tv::Tensor output_add = tv::Tensor();
  tv::Tensor bias = tv::Tensor();
  tv::Tensor scale = tv::Tensor();
  int split_k_slices = 1;
  std::vector<int> padding;
  std::vector<int> stride;
  std::vector<int> dilation;
  float alpha;
  float beta;
  float act_alpha;
  float act_beta;
  Activation act_type = Activation::kNone;
  int mask_width;
  uint32_t mask_filter;
  bool reverse_mask;
  bool verbose;
  int mask_int_count;
  tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false);
  tv::Tensor workspace = tv::Tensor();
  tv::Tensor mask = tv::Tensor();
  tv::Tensor mask_argsort = tv::Tensor();
  tv::Tensor indices = tv::Tensor();
  tv::Tensor mask_output = tv::Tensor();
  std::uintptr_t stream = 0;
  tv::gemm::NVRTCParams nvrtc_params = tv::gemm::NVRTCParams();
  ConvParams(int ndim, ConvOpType op_type, tv::CUDAKernelTimer timer)
      : conv_algo_desp(ndim, op_type), input(tv::Tensor()),
        weight(tv::Tensor()), output(tv::Tensor()), padding(std::vector<int>()),
        stride(std::vector<int>()), dilation(std::vector<int>()), alpha(1.0),
        beta(0.0), mask_width(-1), mask_filter(0xffffffff), reverse_mask(false),
        verbose(false), timer(timer), mask_int_count(1) {}
};

} // namespace gemm
} // namespace tv