// Copyright 2019-2021 Yan Yan
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
#include <chrono>
#include <tensorview/core/defs.h>
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuda_runtime_api.h>
#endif
#include <iostream>
#include <tensorview/tensorview.h>

namespace tv {

#if defined(TV_HARDWARE_ACC_CUDA)
template <typename TimeT = std::chrono::microseconds> struct CudaContextTimer {
  CudaContextTimer(bool enable = true) : enable_(enable) {
    if (enable_) {
      {
        checkCudaErrors(cudaDeviceSynchronize());
        mCurTime = std::chrono::steady_clock::now();
      }
    }
  }
  typename TimeT::rep report() {
    typename TimeT::rep res;
    if (enable_) {
      checkCudaErrors(cudaDeviceSynchronize());
      auto duration = std::chrono::duration_cast<TimeT>(
          std::chrono::steady_clock::now() - mCurTime);
      res = duration.count();
      mCurTime = std::chrono::steady_clock::now();
    }
    return res;
  }
  CudaContextTimer<TimeT> &enable() {
    enable_ = true;
    return *this;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> mCurTime;
  bool enable_;
};

using CUDATimer = CudaContextTimer<>;

#endif

template <typename TimeT = std::chrono::microseconds> struct CPUTimer {
  CPUTimer() { mCurTime = std::chrono::steady_clock::now(); }
  typename TimeT::rep report() {
    auto duration = std::chrono::duration_cast<TimeT>(
        std::chrono::steady_clock::now() - mCurTime);
    auto res = duration.count();
    mCurTime = std::chrono::steady_clock::now();
    return res;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> mCurTime;
};

} // namespace tv
