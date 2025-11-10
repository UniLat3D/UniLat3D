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
#if !defined(TV_CUDA_CC) && !defined(__CUDACC_RTC__)
#include <string>
#endif 
namespace tv {

template <size_t N> class const_string : public array<char, N + 1> {
public:
  template <typename... Chars>
  TV_HOST_DEVICE_INLINE constexpr const_string(Chars... chars)
      : array<char, N + 1>{chars..., '\0'} {}

  // Copy constructor
  template <int... Indexes>
  TV_HOST_DEVICE_INLINE constexpr const_string(
      const const_string<N> &rhs,
      mp_list_int<Indexes...> dummy = mp_list_int_range<0, N>{})
      : array<char, N + 1>{rhs[Indexes]..., '\0'} {}

  template <size_t N1, int... Indexes>
  TV_HOST_DEVICE_INLINE constexpr const_string(const const_string<N1> &rhs,
                                               mp_list_int<Indexes...>)
      : array<char, N + 1>{rhs[Indexes]..., '\0'} {}

  template <int... Indexes>
  TV_HOST_DEVICE_INLINE constexpr const_string(const char TV_METAL_THREAD (&value)[N + 1],
                                               mp_list_int<Indexes...>)
      : const_string(value[Indexes]...) {}

  TV_HOST_DEVICE_INLINE constexpr const_string(const char TV_METAL_THREAD (&value)[N + 1])
      : const_string(value, std::make_index_sequence<N>{}) {}

  TV_HOST_DEVICE_INLINE constexpr size_t size() const { return N; }

  TV_HOST_DEVICE_INLINE constexpr TV_METAL_THREAD const char *c_str() const {
    return this->array_;
  }
#ifndef TV_CUDA_CC
  operator std::string() const { return std::string(c_str()); }
#endif
};

namespace detail {
template <class TArray1, class TArray2, int... IndsL, int... IndsR>
TV_HOST_DEVICE_INLINE constexpr const_string<sizeof...(IndsL) +
                                             sizeof...(IndsR)>
string_concat_impl(TV_METAL_THREAD const TArray1 &a, TV_METAL_THREAD const TArray2 &b, mp_list_int<IndsL...>,
                   mp_list_int<IndsR...>) noexcept {
  return const_string<sizeof...(IndsL) + sizeof...(IndsR)>(a[IndsL]...,
                                                           b[IndsR]...);
}

template <std::size_t N, int... Indexes>
TV_HOST_DEVICE_INLINE constexpr auto
make_const_string_impl(TV_METAL_THREAD const char (&value)[N], mp_list_int<Indexes...>) {
  return const_string<N - 1>(value[Indexes]...);
}

} // namespace detail

template <std::size_t N>
TV_HOST_DEVICE_INLINE constexpr auto make_const_string(TV_METAL_THREAD const char (&value)[N]) {
  return detail::make_const_string_impl(value, mp_list_int_range<0, N - 1>{});
}

template <size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr const_string<N1 + N2>
operator+(TV_METAL_THREAD const const_string<N1> &lfs, TV_METAL_THREAD const const_string<N2> &rfs) {
  return detail::string_concat_impl(lfs, rfs, mp_list_int_range<0, N1>{},
                                    mp_list_int_range<0, N2>{});
}

template <size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr const_string<N1 + N2 - 1>
operator+(TV_METAL_THREAD const const_string<N1> &lfs, TV_METAL_THREAD const char (&rfs)[N2]) {
  return detail::string_concat_impl(lfs, rfs, mp_list_int_range<0, N1>{},
                                    mp_list_int_range<0, N2 - 1>{});
}

template <size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr const_string<N1 - 1 + N2>
operator+(TV_METAL_THREAD const char (&lfs)[N1], TV_METAL_THREAD const const_string<N2> &rfs) {
  return detail::string_concat_impl(lfs, rfs, mp_list_int_range<0, N1 - 1>{},
                                    mp_list_int_range<0, N2>{});
}

} // namespace tv
