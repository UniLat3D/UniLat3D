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
// from pytorch.aten
#include <tensorview/core/common.h>
#include <type_traits>
#include <cuda.h>
namespace tv {
namespace cuda {

template <typename T1, typename T2> inline int DivUp(const T1 a, const T2 b) {
  return (a + b - 1) / b;
}

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 512;
constexpr int CUDA_MAX_GRID = 65535;

// CUDA: number of blocks for threads.

inline int64_t getNumThreads(const int64_t N) {
  if (N > CUDA_NUM_THREADS) {
    return CUDA_NUM_THREADS;
  }
  return DivUp(N, 32) * 32;
}

template <size_t MaxNumThreads> inline int64_t getNumThreadsEx(const int64_t N) {
  if (N > MaxNumThreads) {
    return MaxNumThreads;
  }
  return DivUp(N, 32) * 32;
}

inline int64_t getBlocks(const int64_t N) {
  TV_ASSERT_RT_ERR(N > 0,
                   "CUDA kernel launch blocks must be positive, but got N=", N);
  return DivUp(N, getNumThreads(N));
}

template <size_t MaxNumThreads> inline int64_t getBlocksEx(const int64_t N) {
  TV_ASSERT_RT_ERR(N > 0,
                   "CUDA kernel launch blocks must be positive, but got N=", N);
  return DivUp(N, getNumThreadsEx<MaxNumThreads>(N));
}

template <size_t NumThreads = 1024> struct LaunchEx {
  LaunchEx(dim3 blocks_, dim3 threads_, cudaStream_t stream_ = 0)
      : blocks(blocks_), threads(threads_), smem_size(0), stream(stream_) {}
  LaunchEx(dim3 blocks_, dim3 threads_, int64_t smem_size,
           cudaStream_t stream_ = 0)
      : blocks(blocks_), smem_size(smem_size), threads(threads_),
        stream(stream_) {}

  LaunchEx(int64_t size, cudaStream_t stream_ = 0)
      : blocks(getBlocksEx<NumThreads>(size)),
        threads(getNumThreadsEx<NumThreads>(size)), smem_size(0),
        stream(stream_) {}

  template <typename F, class... Args> void operator()(F &&f, Args &&...args) {
    std::forward<F>(f)<<<blocks, threads, smem_size, stream>>>(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  void run_launch_api(CUfunction kernel, Args &&...args) {
    void *args_vec[] = {&args...};
    TV_CUDART_RESULT_CHECK(
        cudaLaunchKernel(kernel, blocks.x, blocks.y, blocks.z, // grid dim
                         threads.x, threads.y, threads.z,      // block dim
                         smem_size, stream, // shared mem and stream
                         args_vec, 0));     // arguments
  }

  template <class... Args>
  void run_launch_driver_api(CUfunction kernel, Args &&...args) {
    void *args_vec[] = {&args...};
    TV_CUDA_RESULT_CHECK(
        cuLaunchKernel(kernel, blocks.x, blocks.y, blocks.z, // grid dim
                       threads.x, threads.y, threads.z,      // block dim
                       smem_size, stream, // shared mem and stream
                       args_vec, 0));     // arguments
  }

  dim3 blocks;
  dim3 threads;
  int64_t smem_size;
  cudaStream_t stream;
};

using Launch = LaunchEx<1024>;

} // namespace cuda

} // namespace tv