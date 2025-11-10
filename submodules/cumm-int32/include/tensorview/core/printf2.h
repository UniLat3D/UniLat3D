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
#include "array.h"
#include "defs.h"
#ifndef TV_PARALLEL_RTC
#include <cstdio>
#include <cstdint>
#endif
#include "const_string.h"
namespace tv {

namespace detail {

template <typename T> struct type_to_format;

template <> struct type_to_format<int> {
  static constexpr auto value = make_const_string("%d");
};

template <> struct type_to_format<bool> {
  static constexpr auto value = make_const_string("%d");
};

template <> struct type_to_format<unsigned int> {
  static constexpr auto value = make_const_string("%u");
};

template <> struct type_to_format<long> {
  static constexpr auto value = make_const_string("%ld");
};

template <> struct type_to_format<unsigned long> {
  static constexpr auto value = make_const_string("%lu");
};

template <> struct type_to_format<long long> {
  static constexpr auto value = make_const_string("%lld");
};

template <> struct type_to_format<unsigned long long> {
  static constexpr auto value = make_const_string("%llu");
};

template <> struct type_to_format<float> {
  static constexpr auto value = make_const_string("%f");
};

template <> struct type_to_format<double> {
  static constexpr auto value = make_const_string("%lf");
};

template <> struct type_to_format<char *> {
  static constexpr auto value = make_const_string("%s");
};

template <> struct type_to_format<const char *> {
  static constexpr auto value = make_const_string("%s");
};

template <> struct type_to_format<char> {
  static constexpr auto value = make_const_string("%c");
};

template <> struct type_to_format<int8_t> {
  static constexpr auto value = make_const_string("%hd");
};

template <> struct type_to_format<int16_t> {
  static constexpr auto value = make_const_string("%hd");
};

template <> struct type_to_format<uint8_t> {
  static constexpr auto value = make_const_string("%hu");
};

template <> struct type_to_format<uint16_t> {
  static constexpr auto value = make_const_string("%hu");
};

template <class T, size_t N>
TV_HOST_DEVICE_INLINE constexpr auto generate_array_format() {
  // [%d, %d, %d, %d]
  auto elem_fmt = type_to_format<T>::value;
  int elem_fmt_size = elem_fmt.size();
  tv::const_string<2 + (type_to_format<T>::value).size() * N + 2 * (N - 1)>
      res{};
  res[0] = '[';
  res[res.size() - 1] = ']';
  for (int i = 0; i < N; ++i) {
    for (int j = 1 + i * (elem_fmt_size + 2);
         j < 1 + i * (elem_fmt_size + 2) + elem_fmt_size; ++j) {
      res[j] = elem_fmt[j - (1 + i * (elem_fmt_size + 2))];
    }
    if (i != N - 1) {
      res[1 + i * (elem_fmt_size + 2) + elem_fmt_size] = ',';
      res[1 + i * (elem_fmt_size + 2) + elem_fmt_size + 1] = ' ';
    }
  }
  return res;
}

template <class T, size_t N> struct type_to_format<array<T, N>> {
  static constexpr auto value = generate_array_format<T, N>();
};

template <class T, size_t N> struct type_to_format<T[N]> {
  static constexpr auto value = generate_array_format<T, N>();
};

template <size_t N> struct type_to_format<const char[N]> {
  static constexpr auto value = type_to_format<const char *>::value;
};

template <char Sep, class... Ts> struct types_to_format;

template <char Sep, class T> struct types_to_format<Sep, T> {
  static constexpr auto value = type_to_format<std::decay_t<T>>::value + "\n";
};

template <char Sep, class T, class... Ts>
struct types_to_format<Sep, T, Ts...> {
  static constexpr auto value = type_to_format<std::decay_t<T>>::value +
                                const_string<1>(Sep) +
                                types_to_format<Sep, Ts...>::value;
};

} // namespace detail

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2(Ts... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, Ts...>::value;
  printf(fmt.c_str(), args...);
}

template <char Sep = ' ', unsigned Tx = 0, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_once(Ts... args) {
#if defined(__CUDA_ARCH__)
  if ((threadIdx.x == Tx && threadIdx.y == 0 && threadIdx.z == 0 &&
       blockIdx.x == 0 && blockIdx.y == 0))
    printf2<Sep>(args...);
#else
  printf2<Sep>(args...);
#endif
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_block_once(Ts... args) {
#if defined(__CUDA_ARCH__)
  if ((blockIdx.x == 0 && blockIdx.y == 0))
    printf2<Sep>(args...);
#else
  printf2<Sep>(args...);
#endif
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_thread_once(Ts... args) {
#if defined(__CUDA_ARCH__)
  if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0))
    printf2<Sep>(args...);
#else
  printf2<Sep>(args...);
#endif
}

namespace detail {

template <char Sep = ' ', typename T, size_t N, int... Indexes, class... Ts>
TV_HOST_DEVICE_INLINE void
printf2_array_impl(array<T, N> arg, mp_list_int<Indexes...>, Ts &&...args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt =
      types_to_format<Sep, array<T, N>, Ts...>::value;
  printf(fmt.c_str(), arg[Indexes]..., args...);
}

template <char Sep = ' ', typename T, size_t N, int... Indexes, class... Ts>
TV_HOST_DEVICE_INLINE void
printf2_array_impl(T const (&arg)[N], mp_list_int<Indexes...>, Ts &&...args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt =
      types_to_format<Sep, array<T, N>, Ts...>::value;
  printf(fmt.c_str(), arg[Indexes]..., args...);
}

} // namespace detail

template <char Sep = ' ', typename T, size_t N, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array(array<T, N> arg, Ts &&...args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  return detail::printf2_array_impl<Sep>(arg, mp_make_list_c_sequence<int, N>{},
                                         std::forward<Ts>(args)...);
}

template <char Sep = ' ', typename T, size_t N, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array(T const (&arg)[N], Ts &&...args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  return detail::printf2_array_impl<Sep>(arg, mp_make_list_c_sequence<int, N>{},
                                         std::forward<Ts>(args)...);
}

template <char Sep = ' ', unsigned Tx = 0, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array_once(Ts &&...args) {
#if defined(__CUDA_ARCH__)
  if ((threadIdx.x == Tx && threadIdx.y == 0 && threadIdx.z == 0 &&
       blockIdx.x == 0 && blockIdx.y == 0))
    printf2_array(args...);
#else
  printf2_array(std::forward<Ts>(args)...);
#endif
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array_block_once(Ts &&...args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
#if defined(__CUDA_ARCH__)
  if ((blockIdx.x == 0 && blockIdx.y == 0))
    printf2_array(args...);
#else
  printf2_array(std::forward<Ts>(args)...);
#endif
}

} // namespace tv

#define TV_DEVICE_ASSERT(x)                                            \
  if (!(x)) {                                                                   \
    tv::printf2("Device Assert:", #x);                                                       \
    assert(x);                                                                 \
  }

#define TV_DEVICE_ASSERT_WITH_PRINT(x, ...)                                            \
  if (!(x)) {                                                                   \
    tv::printf2("Device Assert:", #x, __VA_ARGS__);                                                       \
    assert(x);                                                                 \
  }
