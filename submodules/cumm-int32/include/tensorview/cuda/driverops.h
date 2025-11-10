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
#include <tensorview/tensorview.h>

namespace tv {

#if defined(TV_HARDWARE_ACC_CUDA)

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

// template <typename T>
// void check(T result, char const *const func, const char *const file,
//            int const line) {
//   if (result) {
//     fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
//             static_cast<unsigned int>(result), func);
//     DEVICE_RESET
//     // Make sure we call CUDA Device Reset before exiting
//     exit(EXIT_FAILURE);
//   }
// }

#define checkCudaErrors(val) TV_CUDART_RESULT_CHECK(val)

template <typename T>
void host2dev(T *dst, const T *src, size_t size, cudaStream_t s) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyHostToDevice, s));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s) {
  host2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s) {
  host2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T> void host2dev(T *dst, const T *src, size_t size) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<const T, Rank, PtrTraits2, Tindex2> src) {
  host2dev(dst.data(), src.data(), std::min(dst.size(), src.size()));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<T, Rank, PtrTraits2, Tindex2> src) {
  host2dev(dst.data(), src.data(), std::min(dst.size(), src.size()));
}

template <typename T>
void dev2host(T *dst, const T *src, size_t size, cudaStream_t s) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost, s));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s) {
  dev2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s) {
  dev2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T> void dev2host(T *dst, const T *src, size_t size) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<const T, Rank, PtrTraits2, Tindex2> src) {
  dev2host(dst.data(), src.data(), std::min(dst.size(), src.size()));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<T, Rank, PtrTraits2, Tindex2> src) {
  dev2host(dst.data(), src.data(), std::min(dst.size(), src.size()));
}

template <typename T>
void dev2dev(T *dst, const T *src, size_t size, cudaStream_t s) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice, s));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
             const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
             cudaStream_t s) {
  dev2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
             const TensorView<T, Rank, PtrTraits2, Tindex2> src,
             cudaStream_t s) {
  dev2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T> void dev2dev(T *dst, const T *src, size_t size) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
             const TensorView<const T, Rank, PtrTraits2, Tindex2> src) {
  dev2dev(dst.data(), src.data(), std::min(dst.size(), src.size()));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
             const TensorView<T, Rank, PtrTraits2, Tindex2> src) {
  dev2dev(dst.data(), src.data(), std::min(dst.size(), src.size()));
}

template <typename T>
void host2host(T *dst, const T *src, size_t size, cudaStream_t s) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyHostToHost, s));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
               const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
               cudaStream_t s) {
  host2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
               const TensorView<T, Rank, PtrTraits2, Tindex2> src,
               cudaStream_t s) {
  host2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T> void host2host(T *dst, const T *src, size_t size) {
  TV_CUDART_RESULT_CHECK(
      cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToHost));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
               const TensorView<const T, Rank, PtrTraits2, Tindex2> src) {
  host2host(dst.data(), src.data(), std::min(dst.size(), src.size()));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
               const TensorView<T, Rank, PtrTraits2, Tindex2> src) {
  host2host(dst.data(), src.data(), std::min(dst.size(), src.size()));
}

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
void zero_dev(TensorView<T, Rank, PtrTraits, Tindex> tensor) {
  TV_CUDART_RESULT_CHECK(
      cudaMemset(tensor.data(), 0, tensor.size() * sizeof(T)));
}

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
void zero_dev(TensorView<T, Rank, PtrTraits, Tindex> tensor, cudaStream_t s) {
  TV_CUDART_RESULT_CHECK(
      cudaMemsetAsync(tensor.data(), 0, tensor.size() * sizeof(T), s));
}
template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
void zero_host(TensorView<T, Rank, PtrTraits, Tindex> tensor) {
  std::fill(tensor.data(), tensor.data() + tensor.size(), 0);
}

namespace detail {
typedef union {
  float src;
  uint32_t dst;
} FloatBits;

typedef union {
  __half src;
  uint16_t dst;
} Float16Bits;

typedef union {
  int32_t src;
  uint32_t dst;
} IntBits;

typedef union {
  int16_t src;
  uint16_t dst;
} Int16Bits;

typedef union {
  int8_t src;
  uint8_t dst;
} Int8Bits;

} // namespace detail

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
struct FillDev {
  static void run_async(TensorView<T, Rank, PtrTraits, Tindex> tensor, T val,
                        cudaStream_t s) {
    TV_THROW_INVALID_ARG(
        "not implemented, only 32bit/16bit/8bit is supported.");
  }
  static void run(TensorView<T, Rank, PtrTraits, Tindex> tensor, T val) {
    TV_THROW_INVALID_ARG(
        "not implemented, only 32bit/16bit/8bit is supported.");
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<float, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<float, Rank, PtrTraits, Tindex> tensor,
                        float val, cudaStream_t s) {
    detail::FloatBits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(
        cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst,
                         tensor.size(), s));
  }
  static void run(TensorView<float, Rank, PtrTraits, Tindex> tensor,
                  float val) {
    detail::FloatBits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(cuMemsetD32(
        reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<__half, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<__half, Rank, PtrTraits, Tindex> tensor,
                        __half val, cudaStream_t s) {
    detail::Float16Bits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(
        cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst,
                         tensor.size(), s));
  }
  static void run(TensorView<__half, Rank, PtrTraits, Tindex> tensor,
                  __half val) {
    detail::Float16Bits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(cuMemsetD16(
        reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<int32_t, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<int32_t, Rank, PtrTraits, Tindex> tensor,
                        int32_t val, cudaStream_t s) {
    detail::IntBits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(
        cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst,
                         tensor.size(), s));
  }
  static void run(TensorView<int32_t, Rank, PtrTraits, Tindex> tensor,
                  int32_t val) {
    detail::IntBits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(cuMemsetD32(
        reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<uint32_t, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<uint32_t, Rank, PtrTraits, Tindex> tensor,
                        uint32_t val, cudaStream_t s) {
    TV_CUDADRV_RESULT_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(tensor.data()), val, tensor.size(), s));
  }
  static void run(TensorView<uint32_t, Rank, PtrTraits, Tindex> tensor,
                  uint32_t val) {
    TV_CUDADRV_RESULT_CHECK(cuMemsetD32(
        reinterpret_cast<CUdeviceptr>(tensor.data()), val, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<uint16_t, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<uint16_t, Rank, PtrTraits, Tindex> tensor,
                        uint16_t val, cudaStream_t s) {
    TV_CUDADRV_RESULT_CHECK(cuMemsetD16Async(
        reinterpret_cast<CUdeviceptr>(tensor.data()), val, tensor.size(), s));
  }
  static void run(TensorView<uint16_t, Rank, PtrTraits, Tindex> tensor,
                  uint16_t val) {
    TV_CUDADRV_RESULT_CHECK(cuMemsetD16(
        reinterpret_cast<CUdeviceptr>(tensor.data()), val, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<uint8_t, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<uint8_t, Rank, PtrTraits, Tindex> tensor,
                        uint8_t val, cudaStream_t s) {
    TV_CUDADRV_RESULT_CHECK(cuMemsetD8Async(
        reinterpret_cast<CUdeviceptr>(tensor.data()), val, tensor.size(), s));
  }
  static void run(TensorView<uint8_t, Rank, PtrTraits, Tindex> tensor,
                  uint8_t val) {
    TV_CUDADRV_RESULT_CHECK(cuMemsetD8(
        reinterpret_cast<CUdeviceptr>(tensor.data()), val, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<int16_t, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<int16_t, Rank, PtrTraits, Tindex> tensor,
                        int16_t val, cudaStream_t s) {
    detail::Int16Bits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(
        cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst,
                         tensor.size(), s));
  }
  static void run(TensorView<int16_t, Rank, PtrTraits, Tindex> tensor,
                  int16_t val) {
    detail::Int16Bits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(cuMemsetD16(
        reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst, tensor.size()));
  }
};

template <int Rank, template <class> class PtrTraits, typename Tindex>
struct FillDev<int8_t, Rank, PtrTraits, Tindex> {
  static void run_async(TensorView<int8_t, Rank, PtrTraits, Tindex> tensor,
                        int8_t val, cudaStream_t s) {
    detail::Int8Bits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(
        cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst,
                        tensor.size(), s));
  }
  static void run(TensorView<int8_t, Rank, PtrTraits, Tindex> tensor,
                  int8_t val) {
    detail::Int8Bits fb;
    fb.src = val;
    TV_CUDADRV_RESULT_CHECK(cuMemsetD8(
        reinterpret_cast<CUdeviceptr>(tensor.data()), fb.dst, tensor.size()));
  }
};
#else
#define checkCudaErrors(val)
#endif

} // namespace tv