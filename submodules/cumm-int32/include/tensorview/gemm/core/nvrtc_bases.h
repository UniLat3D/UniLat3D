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

#include <tensorview/cuda/nvrtc.h>
#include <tensorview/tensor.h>

namespace tv {
namespace gemm {
struct NVRTCParams {

public:
  std::shared_ptr<tv::NVRTCModule> cumodule; // module is a keyword in c++ 20
  std::string kernel_name, init_kernel_name, constant_name;
  int param_size = -1;
  tv::Tensor param_storage, param_storage_cpu;

  int num_threads = -1;
  int smem_size = -1;
  int mode = -1;
};

} // namespace gemm
} // namespace tv