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
#ifdef TV_PARALLEL_RTC
#include <tensorview/core/nvrtc_std.h>
#else
#include <type_traits>
#include <limits>
#endif

namespace tv {
namespace hash {

namespace detail {
struct __place_holder_t;

template <int N, bool Signed> struct SizeToInt;

template <bool Signed> struct SizeToInt<1, Signed> {
  using type = std::conditional_t<Signed, int8_t, uint8_t>;
};

template <bool Signed> struct SizeToInt<2, Signed> {
  using type = std::conditional_t<Signed, int16_t, uint16_t>;
};

template <bool Signed> struct SizeToInt<4, Signed> {
  using type = std::conditional_t<Signed, int32_t, uint32_t>;
};

#if defined(TV_METAL_CC)
template <bool Signed> struct SizeToInt<8, Signed> {
  using type = std::conditional_t<Signed, int64_t, uint64_t>;
};
#elif defined(TV_CUDA_CC) 
template <bool Signed> struct SizeToInt<8, Signed> {
  using type = std::conditional_t<Signed, int64_t, std::conditional_t<std::is_same<uint64_t, unsigned long long>::value, uint64_t, unsigned long long>>;
};
#else 
template <bool Signed> struct SizeToInt<8, Signed> {
  using type = std::conditional_t<Signed, int64_t, uint64_t>;
};

#endif

template <typename T> struct MapTypeToInt {
  using type = typename SizeToInt<sizeof(T), true>::type;
};

template <typename T> struct MapTypeToUnsignedInt {
  using type = typename SizeToInt<sizeof(T), false>::type;
};

template <typename K>
struct platform_not_support_atomic_64_metas {
#ifdef TV_METAL_CC
  static_assert(!std::is_same_v<K, int64_t>, "int64 key not supported when 64bit CAS not supported, use uint64 instead.");
#endif
  TV_HOST_DEVICE_INLINE static constexpr auto map_user_key(K key) {
    return key;
  }
  TV_HOST_DEVICE_INLINE static constexpr auto unmap_user_key(K key) {
    return key;
  }

};

#ifdef TV_METAL_CC
// template <>
// struct platform_not_support_atomic_64_metas<int64_t> {
  
//   // in platform without 64bit atomic cas, user can only use 62bit data.
//   // drop 2nd and 3rd bit, keep sign and rest
//   static constexpr TV_METAL_CONSTANT uint64_t kMask = 0x9fff'ffff'7fff'ffff;

//   TV_HOST_DEVICE_INLINE static constexpr auto map_user_key(uint64_t key) {
//     auto res = key & kMask;
//     // move 1st bit in last 32 bit to 2nd bit in first 32 bit
//     return res | ((key & 0x8000'0000) << 31);
//   }
//   TV_HOST_DEVICE_INLINE static constexpr auto unmap_user_key(uint64_t key) {
//    // user key format) 0b.*0....... (first 32 bit) 0b0....... (last 32 bit)
//    // move * to first 0 in last 32 bit
//     auto res = key & kMask;
//     return res | ((key & 0x4000'0000'0000'0000) >> 31);
//   }
// };
template <>
struct platform_not_support_atomic_64_metas<uint64_t> {
  // drop 1st and 2nd bit
  static constexpr TV_METAL_CONSTANT uint64_t kMask = 0x3fff'ffff'7fff'ffff;

  TV_HOST_DEVICE_INLINE static constexpr auto map_user_key(uint64_t key) {
    auto res = key & kMask;
    // move 1st bit in last 32 bit to 1st bit in first 32 bit
    return res | ((key & 0x8000'0000) << 32);
  }
  TV_HOST_DEVICE_INLINE static constexpr auto unmap_user_key(uint64_t key) {
    auto res = key & kMask;
    return res | ((key & 0x8000'0000'0000'0000) >> 32);
  }
};
#endif


template <typename K>
struct default_empty_key;

template <>
struct default_empty_key<int32_t>{
  static constexpr TV_METAL_CONSTANT int32_t value = std::numeric_limits<int32_t>::max();
};

template <>
struct default_empty_key<int64_t>{
#ifdef TV_METAL_CC
  // when using metal with 64bit key, custom empty key is not supported
  // all empty key for u64 and i64 are set to 0xFFFFFFFFFFFFFFFF
  static constexpr TV_METAL_CONSTANT int64_t value = -1;
#else 
  static constexpr TV_METAL_CONSTANT int64_t value = std::numeric_limits<int64_t>::max();
#endif 
};
template <>
struct default_empty_key<uint32_t>{
  static constexpr TV_METAL_CONSTANT uint32_t value = std::numeric_limits<uint32_t>::max();
};

template <>
struct default_empty_key<uint64_t>{
#ifdef TV_METAL_CC
  static constexpr TV_METAL_CONSTANT uint64_t value = std::numeric_limits<uint64_t>::max();
#else 
  static constexpr TV_METAL_CONSTANT uint64_t value = std::numeric_limits<uint64_t>::max();
#endif 
};
#ifdef TV_CUDA_CC

template <>
struct default_empty_key<
    std::conditional<std::is_same<uint64_t, unsigned long long>::value,
                     __place_holder_t, unsigned long long>::type> {
  static constexpr unsigned long long value = std::numeric_limits<unsigned long long>::max();
};
#endif

} // namespace detail

template <int K>
using itemsize_to_unsigned_t = typename detail::SizeToInt<K, false>::type;

template <typename K>
using to_unsigned_t = typename detail::MapTypeToUnsignedInt<K>::type;

template <typename K>
constexpr TV_METAL_CONSTANT K default_empty_key_v = detail::default_empty_key<K>::value;

template <typename K, typename V, K EmptyKey = default_empty_key_v<K>>
struct pair {
  K first;
  V second;
  using key_type_uint = to_unsigned_t<K>;
  TV_HOST_DEVICE_INLINE bool empty() { return first == EmptyKey; }
};

template <typename T> TV_HOST_DEVICE size_t align_to_power2(T size) {
  size_t r = 0;
  size_t num_1_bit = size & 1 ? 1 : 0;
  while (size >>= 1) {
    r++;
    if (size & 1) {
      ++num_1_bit;
    }
  }
  if (num_1_bit == 1) {
    return 1 << r;
  } else {
    return 1 << (r + 1);
  }
}

} // namespace hash

} // namespace tv
