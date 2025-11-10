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

#include <tensorview/core/all.h>
#include "constants.h"
#define CUMM_MAXIMUM_NVRTC_CONV_NDIM 3

namespace tv {
namespace gemm {

struct GemmNVRTCParams {
public:
  int m, n, k;
  const void *ptr_A, *ptr_B, *ptr_D;
  void *ptr_C;
  int64_t stride_A, stride_B, stride_C, stride_D;
  const int *indiceA, *indiceBorC, *indiceD;
  float alpha, beta;
  float act_alpha;
  float act_beta;
  int act_type;
  int split_k_slices;
  void *workspace;
};

struct SparseConvNVRTCParams {
public:
  const void *ptr_A, *ptr_B, *ptr_D;
  void *ptr_C;
  float alpha, beta;
  float act_alpha;
  float act_beta;
  int act_type;
  int split_k_slices;
  void *workspace;

  const uint32_t *mask_ptr;
  uint32_t *mask_out_ptr;
  const int32_t *indice_ptr;
  const int32_t *mask_argsort_ptr;

  uint32_t mask_filter;
  bool reverse_mask;
  int mask_width;

  int ndim, N, C, K;
  int kernel_volume;
  int mode;
  int groups;
  bool d_is_bias;
  const void* bias_pointer;
  const void* scale_pointer;
};

struct ConvNVRTCParams {
public:
  const void *ptr_A, *ptr_B, *ptr_D;
  void *ptr_C;
  float alpha, beta;
  float act_alpha;
  float act_beta;
  int act_type;
  int split_k_slices;
  void *workspace;

  int ndim, N, C, K;
  tv::array<int, CUMM_MAXIMUM_NVRTC_CONV_NDIM> input_dims, output_dims, ksize,
      padding, stride, dilation;
  int kernel_volume;
  int mode;
  int groups;
  bool d_is_bias;
  const void* bias_pointer;
  const void* scale_pointer;
};

} // namespace gemm
} // namespace tv