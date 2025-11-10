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
#include <exception>
#include <tensorview/context.h>
#include <tensorview/core/all.h>
#if defined(TV_CUDA_CC)
#include <tensorview/cuda/launch.h>
#include <tensorview/kernel_utils.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tv {
namespace detail {

template <bool V> constexpr bool delay_bool_eval_v = V;

#if defined(TV_CUDA_CC)
template <typename F, class... Args>
__global__ void simple_kernel_1d(Args... args, size_t size, F f) {
  for (auto i : tv::KernelLoopX<size_t>(size)) {
    f(i, args...);
  }
}

template <typename F, class... Args>
__global__ void simple_kernel_1d_beginendstep(Args... args, size_t size, F f) {
  f(blockIdx.x * blockDim.x + threadIdx.x, size, gridDim.x * blockDim.x,
    args...);
}

#endif

template <size_t MaxNumThreads = 512, size_t GrainSize = 0,
          bool MapMode = false, typename F, class... Args>
void kernel_1d_impl(Context ctx, int device, size_t size, F &&f,
                    Args &&...args) {
  // if extend-lambda isn't added, TV_IS_EXTEND_LAMBDA always return false and
  // cause compile problem even if we don't use this function, so we delay
  // evaulation of this constexpr by a simple template.
#if defined(TV_CUDA_CC)
  static constexpr bool is_nv_host_device_func =
      delay_bool_eval_v<TV_IS_EXTEND_LAMBDA(F)>;
  static_assert(is_nv_host_device_func,
                "only support extend lambda with __host__ __device__");
#endif
  // static_assert(argument_size_v<F> == sizeof...(args) + (MapMode ? 1 : 3),
  //               "your lambda must have N + 1(3 if not MapMode) arg (begin, "
  //               "[end, step])");
  if (device == -1) {
#ifdef _OPENMP
    int64_t begin = 0;
    int64_t end = size;
    // from pytorch/aten/src/ATen/ParallelOpenMP.h
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
#pragma omp parallel if (MaxNumThreads > 1 && omp_get_max_threads() > 1 &&     \
                         !omp_in_parallel() && ((end - begin) > GrainSize))
    {
      int64_t num_threads = omp_get_num_threads();
      if (MaxNumThreads > 0) {
        num_threads = std::min(num_threads, int64_t(MaxNumThreads));
      }
      if (GrainSize > 0) {
        num_threads =
            std::min(num_threads, div_up((end - begin), int64_t(GrainSize)));
      }

      int64_t tid = omp_get_thread_num();
      int64_t chunk_size = div_up((end - begin), num_threads);
      int64_t begin_tid = begin + tid * chunk_size;
      if (begin_tid < end) {
        try {
          // TODO remove c++14 if constexpr.
          if_constexpr<MapMode>(
              [&](auto _) {
                for (size_t i = begin_tid;
                     i < std::min(end, chunk_size + begin_tid); ++i) {
                  std::forward<F>(_(f))(i, std::forward<Args>(args)...);
                }
              },
              [&](auto _) {
                std::forward<F>(_(f))(begin_tid,
                                      std::min(end, chunk_size + begin_tid), 1,
                                      std::forward<Args>(args)...);
              });
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
        }
      }
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }
#else
    if_constexpr<MapMode>(
        [&](auto _) {
          for (size_t i = 0; i < size; ++i) {
            std::forward<F>(_(f))(i, std::forward<Args>(args)...);
          }
        },
        [&](auto _) {
          std::forward<F>(_(f))(0, size, 1, std::forward<Args>(args)...);
        });
#endif
  } else {
#if defined(TV_CUDA_CC)
    tv::cuda::LaunchEx<MaxNumThreads> launcher(
        size, reinterpret_cast<cudaStream_t>(ctx.cuda_stream()));
    if_constexpr<MapMode>(
        [&](auto _) {
          launcher(simple_kernel_1d<F, Args...>, std::forward<Args>(args)...,
                   size, _(f));
        },
        [&](auto _) {
          launcher(simple_kernel_1d_beginendstep<F, Args...>,
                   std::forward<Args>(args)..., size, _(f));
        });
    TV_CHECK_CUDA_ERR_V2("launch failed.");
#else
    TV_THROW_INVALID_ARG("your code doesn't compile with TV_ENABLE_HARDWARE_ACC, or your code "
                         "isn't compiled by nvcc.")
#endif
  }
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0,
          bool MapMode = false, typename F, class... Args>
void kernel_1d_impl_cuda(Context ctx, int device, size_t size, F &&f,
                         Args &&...args) {
  // if extend-lambda isn't added, TV_IS_EXTEND_DEVICE_LAMBDA always return
  // false and cause compile problem even if we don't use this function, so we
  // delay evaulation of this constexpr by a simple template.
#if defined(TV_CUDA_CC)
  static constexpr bool is_nv_host_device_func =
      delay_bool_eval_v<TV_IS_EXTEND_DEVICE_LAMBDA(F)>;
  static_assert(is_nv_host_device_func,
                "only support extend lambda with __host__ __device__");
#endif
  // static_assert(argument_size_v<F> == sizeof...(args) + (MapMode ? 1 : 3),
  //               "your lambda must have N + 1(3 if not MapMode) arg (begin, "
  //               "[end, step])");
  if (device == -1) {
    TV_THROW_INVALID_ARG("this function only support cuda tensors.")
  } else {
#if defined(TV_CUDA_CC)
    tv::cuda::LaunchEx<MaxNumThreads> launcher(
        size, reinterpret_cast<cudaStream_t>(ctx.cuda_stream()));
    if_constexpr<MapMode>(
        [&](auto _) {
          launcher(simple_kernel_1d<F, Args...>, std::forward<Args>(args)...,
                   size, _(f));
        },
        [&](auto _) {
          launcher(simple_kernel_1d_beginendstep<F, Args...>,
                   std::forward<Args>(args)..., size, _(f));
        });
    TV_CHECK_CUDA_ERR_V2("launch failed.");
#else
    TV_THROW_INVALID_ARG("your code doesn't compile with TV_ENABLE_HARDWARE_ACC, or your code "
                         "isn't compiled by nvcc.")
#endif
  }
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0,
          bool MapMode = false, typename F, class... Args>
void kernel_1d_impl_cpu(int device, size_t size, F &&f, Args &&...args) {
  static_assert(argument_size_v<F> == sizeof...(args) + (MapMode ? 1 : 3),
                "your lambda must have N + 1(or 3 if not MapMode) args (begin, "
                "[end, step])");
  if (device == -1) {
#ifdef _OPENMP
    int64_t begin = 0;
    int64_t end = size;
    // from pytorch/aten/src/ATen/ParallelOpenMP.h
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
#pragma omp parallel if (MaxNumThreads > 1 && omp_get_max_threads() > 1 &&     \
                         !omp_in_parallel() && ((end - begin) > GrainSize))
    {
      int64_t num_threads = omp_get_num_threads();
      if (MaxNumThreads > 0) {
        num_threads = std::min(num_threads, int64_t(MaxNumThreads));
      }
      if (GrainSize > 0) {
        num_threads =
            std::min(num_threads, div_up((end - begin), int64_t(GrainSize)));
      }

      int64_t tid = omp_get_thread_num();
      int64_t chunk_size = div_up((end - begin), num_threads);
      int64_t begin_tid = begin + tid * chunk_size;
      if (begin_tid < end) {
        try {
          // TODO remove c++14 if constexpr.
          if_constexpr<MapMode>(
              [&](auto _) {
                for (size_t i = begin_tid;
                     i < std::min(end, chunk_size + begin_tid); ++i) {
                  std::forward<F>(_(f))(i, std::forward<Args>(args)...);
                }
              },
              [&](auto _) {
                std::forward<F>(_(f))(begin_tid,
                                      std::min(end, chunk_size + begin_tid), 1,
                                      std::forward<Args>(args)...);
              });
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
        }
      }
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }
#else
    if_constexpr<MapMode>(
        [&](auto _) {
          for (size_t i = 0; i < size; ++i) {
            std::forward<F>(_(f))(i, std::forward<Args>(args)...);
          }
        },
        [&](auto _) {
          std::forward<F>(_(f))(0, size, 1, std::forward<Args>(args)...);
        });
#endif
  } else {
    TV_THROW_INVALID_ARG("you use cpu-only kernel_1d with GPU tensor.")
  }
}

} // namespace detail

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_map(int device, size_t size, F &&f, Args &&...args) {
  return detail::kernel_1d_impl<MaxNumThreads, GrainSize, true>(
      Context(), device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_map(Context ctx, int device, size_t size, F &&f,
                          Args &&...args) {
  return detail::kernel_1d_impl<MaxNumThreads, GrainSize, true>(
      ctx, device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d(int device, size_t size, F &&f, Args &&...args) {
  return detail::kernel_1d_impl<MaxNumThreads, GrainSize, false>(
      Context(), device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d(Context ctx, int device, size_t size, F &&f,
                      Args &&...args) {
  return detail::kernel_1d_impl<MaxNumThreads, GrainSize, false>(
      ctx, device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_map_cuda(int device, size_t size, F &&f, Args &&...args) {
  return detail::kernel_1d_impl_cuda<MaxNumThreads, GrainSize, true>(
      Context(), device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_map_cuda(Context ctx, int device, size_t size, F &&f,
                               Args &&...args) {
  return detail::kernel_1d_impl_cuda<MaxNumThreads, GrainSize, true>(
      ctx, device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_cuda(int device, size_t size, F &&f, Args &&...args) {
  return detail::kernel_1d_impl_cuda<MaxNumThreads, GrainSize, false>(
      Context(), device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_cuda(Context ctx, int device, size_t size, F &&f,
                           Args &&...args) {
  return detail::kernel_1d_impl_cuda<MaxNumThreads, GrainSize, false>(
      ctx, device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_map_cpu(int device, size_t size, F &&f, Args &&...args) {
  return detail::kernel_1d_impl_cpu<MaxNumThreads, GrainSize, true>(
      device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

template <size_t MaxNumThreads = 512, size_t GrainSize = 0, typename F,
          class... Args>
inline void kernel_1d_cpu(int device, size_t size, F &&f, Args &&...args) {
  return detail::kernel_1d_impl_cpu<MaxNumThreads, GrainSize, false>(
      device, size, std::forward<F>(f), std::forward<Args>(args)...);
}

} // namespace tv