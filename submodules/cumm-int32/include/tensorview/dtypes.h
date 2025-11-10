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
#include <tensorview/core/defs.h>
#if defined(TV_HARDWARE_ACC_CUDA)
// #include <cuda.h>
#include <cuda_fp16.h>
#endif
#if (CUDA_VERSION >= 11000 && defined(TV_HARDWARE_ACC_CUDA))
#include <cuda_bf16.h>
#endif
#if (CUDA_VERSION >= 11080 && defined(TV_HARDWARE_ACC_CUDA))
#include <cuda_fp8.h>
#endif

#include <tensorview/core/defs.h>
#ifdef TV_PARALLEL_RTC
#include <tensorview/core/nvrtc_std.h>
#else
#include <algorithm>
#include <type_traits>
#include <cstdint>
#endif
#define TV_UNKNOWN_DTYPE_STRING "unknown"

namespace tv {
enum DType {
  bool_ = 5,
  float16 = 7,
  float32 = 0,
  float64 = 4,
  int8 = 3,
  int16 = 2,
  int32 = 1,
  int64 = 8,
  uint8 = 6,
  uint16 = 9,
  uint32 = 10,
  uint64 = 11,
  bfloat16 = 12,
  tf32 = 13,
  float_e4m3 = 14,
  float_e5m2 = 15,
  custom16 = 100,
  custom32 = 101,
  custom48 = 102,
  custom64 = 103,
  custom80 = 104,
  custom96 = 105,
  custom128 = 106,
  unknown = -1
};

namespace detail {

TV_HOST_DEVICE_INLINE constexpr bool strings_equal(char const *a, char const *b) {
  return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr DType get_custom_dtype() {
  if (sizeof(T) == 2) {
    return custom16;
  } else if (sizeof(T) == 4) {
    return custom32;
  } else if (sizeof(T) == 6) {
    return custom48;
  } else if (sizeof(T) == 8) {
    return custom64;
  } else if (sizeof(T) == 10) {
    return custom80;
  } else if (sizeof(T) == 12) {
    return custom96;
  } else if (sizeof(T) == 16) {
    return custom128;
  } else {
    return unknown;
  }
}

template <typename T> struct TypeToString {
  static constexpr const char *value = TV_UNKNOWN_DTYPE_STRING;
};

template <typename T> struct TypeToDtype {
  static constexpr DType dtype = get_custom_dtype<T>();
};

template <> struct TypeToDtype<int32_t> {
  static constexpr DType dtype = int32;
};
#if defined(TV_HARDWARE_ACC_CUDA)
template <> struct TypeToDtype<__half> {
  static constexpr DType dtype = float16;
};
#endif
#if (CUDA_VERSION >= 11000 && defined(TV_HARDWARE_ACC_CUDA))
template <> struct TypeToDtype<__nv_bfloat16> {
  static constexpr DType dtype = bfloat16;
};
#endif
#if (CUDA_VERSION >= 11080 && defined(TV_HARDWARE_ACC_CUDA))
template <> struct TypeToDtype<__nv_fp8_e5m2> {
  static constexpr DType dtype = float_e5m2;
};
template <> struct TypeToDtype<__nv_fp8_e4m3> {
  static constexpr DType dtype = float_e4m3;
};

#endif

template <> struct TypeToDtype<float> {
  static constexpr DType dtype = float32;
};
template <> struct TypeToDtype<double> {
  static constexpr DType dtype = float64;
};
template <> struct TypeToDtype<int16_t> {
  static constexpr DType dtype = int16;
};
template <> struct TypeToDtype<int8_t> { static constexpr DType dtype = int8; };
template <> struct TypeToDtype<int64_t> {
  static constexpr DType dtype = int64;
};
template <> struct TypeToDtype<uint8_t> {
  static constexpr DType dtype = uint8;
};
template <> struct TypeToDtype<uint16_t> {
  static constexpr DType dtype = uint16;
};
template <> struct TypeToDtype<uint32_t> {
  static constexpr DType dtype = uint32;
};
template <> struct TypeToDtype<uint64_t> {
  static constexpr DType dtype = uint64;
};
template <> struct TypeToDtype<bool> { static constexpr DType dtype = bool_; };

// in nvcc unsigned long long may not equivalent to uint64_t
struct __place_holder_t;

template <>
struct TypeToDtype<
    std::conditional<std::is_same<uint64_t, unsigned long long>::value,
                     __place_holder_t, unsigned long long>::type> {
  static constexpr DType dtype = uint64;
};

} // namespace detail

template <class T>
constexpr DType type_v = detail::TypeToDtype<std::decay_t<T>>::dtype;

template <typename T> TV_HOST_DEVICE_INLINE constexpr const char *dtype_str(T t) {
  switch (t) {
  case DType::bool_:
    return "bool";
  case DType::float32:
    return "float32";
  case DType::int8:
    return "int8";
  case DType::int16:
    return "int16";
  case DType::int32:
    return "int32";
  case DType::float64:
    return "float64";
  case DType::int64:
    return "int64";
  case DType::uint8:
    return "uint8";
  case DType::uint16:
    return "uint16";
  case DType::uint32:
    return "uint32";
  case DType::uint64:
    return "uint64";
  case DType::float16:
    return "half";
  case DType::float_e4m3:
    return "float_e4m3";
  case DType::float_e5m2:
    return "float_e5m2";
  case DType::custom16:
    return "custom16";
  case DType::custom32:
    return "custom32";
  case DType::custom48:
    return "custom48";
  case DType::custom64:
    return "custom64";
  case DType::custom80:
    return "custom80";
  case DType::custom96:
    return "custom96";
  case DType::custom128:
    return "custom128";
  case DType::tf32:
    return "tf32";
  case DType::bfloat16:
    return "bfloat16";
  default:
    return TV_UNKNOWN_DTYPE_STRING;
  }
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr const char *dtype_short_str(T t) {
  switch (t) {
  case DType::bool_:
    return "b1";
  case DType::float32:
    return "f32";
  case DType::int8:
    return "s8";
  case DType::int16:
    return "s16";
  case DType::int32:
    return "s32";
  case DType::float64:
    return "f64";
  case DType::int64:
    return "s64";
  case DType::uint8:
    return "u8";
  case DType::uint16:
    return "u16";
  case DType::uint32:
    return "u32";
  case DType::uint64:
    return "u64";
  case DType::float16:
    return "f16";
  case DType::float_e4m3:
    return "e4m3";
  case DType::float_e5m2:
    return "e5m2";
  case DType::custom16:
    return "x16";
  case DType::custom32:
    return "x32";
  case DType::custom48:
    return "x48";
  case DType::custom64:
    return "x64";
  case DType::custom80:
    return "x80";
  case DType::custom96:
    return "x96";
  case DType::custom128:
    return "x128";
  case DType::tf32:
    return "tf32";
  case DType::bfloat16:
    return "bf16";
  default:
    return TV_UNKNOWN_DTYPE_STRING;
  }
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr int bit_size(T t) {
  switch (t) {
  case DType::bool_:
    return 1;
  case DType::float32:
    return 32;
  case DType::int8:
    return 8;
  case DType::int16:
    return 16;
  case DType::int32:
    return 32;
  case DType::float64:
    return 64;
  case DType::int64:
    return 64;
  case DType::uint8:
    return 8;
  case DType::uint16:
    return 16;
  case DType::uint32:
    return 32;
  case DType::uint64:
    return 64;
  case DType::float16:
    return 16;
  case DType::float_e4m3:
    return 8;
  case DType::float_e5m2:
    return 8;
  case DType::custom16:
    return 16;
  case DType::custom32:
    return 32;
  case DType::custom48:
    return 48;
  case DType::custom64:
    return 64;
  case DType::custom80:
    return 80;
  case DType::custom96:
    return 96;
  case DType::custom128:
    return 128;
  case DType::tf32:
    return 32;
  case DType::bfloat16:
    return 16;
  default:
    return -1;
  }
}

// we can define template specs to extend detail::TypeToString.
template <typename T>
constexpr const char *type_s =
    detail::strings_equal(dtype_str(type_v<T>), TV_UNKNOWN_DTYPE_STRING)
        ? detail::TypeToString<std::decay_t<T>>::value
        : dtype_str(type_v<T>);

} // namespace tv

// from pytorch aten/src/ATen/Dispatch.h.
// these macros should only be used with cuda extend lambdas
// because they don't support generic lambda,
// otherwise you should always use tv::dispatch functions.
#define TV_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)      \
  case enum_type: {                                                            \
    using HINT = type;                                                         \
    return __VA_ARGS__();                                                      \
  }

#define TV_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                       \
  TV_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, scalar_t, __VA_ARGS__)

#define TV_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                            \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
      TV_PRIVATE_CASE_TYPE(NAME, tv::float64, double, __VA_ARGS__)             \
      TV_PRIVATE_CASE_TYPE(NAME, tv::float32, float, __VA_ARGS__)              \
    default:                                                                   \
      TV_THROW_INVALID_ARG(#NAME, " not implemented for '",                    \
                           tv::dtype_str(TYPE), "'");                          \
    }                                                                          \
  }()

#define TV_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                   \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
      TV_PRIVATE_CASE_TYPE(NAME, tv::float64, double, __VA_ARGS__)             \
      TV_PRIVATE_CASE_TYPE(NAME, tv::float32, float, __VA_ARGS__)              \
      TV_PRIVATE_CASE_TYPE(NAME, tv::float16, __half, __VA_ARGS__)             \
    default:                                                                   \
      TV_THROW_INVALID_ARG(#NAME, " not implemented for '",                    \
                           tv::dtype_str(TYPE), "'");                          \
    }                                                                          \
  }()

#define TV_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                            \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
      TV_PRIVATE_CASE_TYPE(NAME, tv::uint8, uint8_t, __VA_ARGS__)              \
      TV_PRIVATE_CASE_TYPE(NAME, tv::int8, int8_t, __VA_ARGS__)                \
      TV_PRIVATE_CASE_TYPE(NAME, tv::int32, int32_t, __VA_ARGS__)              \
      TV_PRIVATE_CASE_TYPE(NAME, tv::int64, int64_t, __VA_ARGS__)              \
      TV_PRIVATE_CASE_TYPE(NAME, tv::int16, __VA_ARGS__)                       \
    default:                                                                   \
      TV_THROW_INVALID_ARG(#NAME, " not implemented for '",                    \
                           tv::dtype_str(TYPE), "'");                          \
    }                                                                          \
  }()


#define TV_DISPATCH_I32_I64(TYPE, NAME, ...)                                   \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
      TV_PRIVATE_CASE_TYPE(NAME, tv::int32, int32_t, __VA_ARGS__)              \
      TV_PRIVATE_CASE_TYPE(NAME, tv::int64, int64_t, __VA_ARGS__)              \
    default:                                                                   \
      TV_THROW_INVALID_ARG(#NAME, " not implemented for '",                    \
                           tv::dtype_str(TYPE), "'");                          \
    }                                                                          \
  }()
