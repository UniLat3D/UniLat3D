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
#include "hash_core.h"
#include <tensorview/core/all.h>
#ifdef __METAL_VERSION__ 
#pragma METAL internals : enable

#endif

namespace tv {
namespace hash {

namespace detail {
template <size_t Nbits> struct MortonCore;
template <> struct MortonCore<32> {
  TV_HOST_DEVICE_INLINE static uint32_t split_by_3bits(uint32_t a) {
    uint32_t x = a & 0x000003ff; // we only look at the first 10 bits
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x0300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
  }
  TV_HOST_DEVICE_INLINE static uint32_t get_third_bits(uint32_t a) {
    uint32_t x = a & 0x9249249; // we only look at the first 10 bits
    x = (x ^ (x >> 2)) & 0x30c30c3;
    x = (x ^ (x >> 4)) & 0x0300f00f;
    x = (x ^ (x >> 8)) & 0x30000ff;
    x = (x ^ (x >> 16)) & 0x000003ff;
    return x;
  }
};

template <> struct MortonCore<64> {
  TV_HOST_DEVICE_INLINE static uint64_t split_by_3bits(uint32_t a) {
    uint64_t x = a & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
  }
  TV_HOST_DEVICE_INLINE static uint32_t get_third_bits(uint64_t a) {
    uint64_t x = a & 0x1249249249249249;
    x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
    x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x1f00000000ffff;
    x = (x ^ (x >> 32)) & 0x1fffff;
    return static_cast<uint32_t>(x);
  }
};

} // namespace detail

template <typename T> struct Morton;

template <> struct Morton<uint32_t> : public detail::MortonCore<32> {
  static constexpr TV_METAL_CONSTANT int kNumBits = 32;
  TV_HOST_DEVICE_INLINE static uint32_t encode(uint32_t x, uint32_t y,
                                               uint32_t z) {
    return (split_by_3bits(z) << 2) | (split_by_3bits(y) << 1) |
           (split_by_3bits(x) << 0);
  }
  TV_HOST_DEVICE_INLINE static tv::array<uint32_t, 3> decode(uint32_t d) {
    return {get_third_bits(d), get_third_bits(d >> 1), get_third_bits(d >> 2)};
  }
  template <uint32_t Axis>
  TV_HOST_DEVICE_INLINE static uint32_t decode_axis(uint32_t d) {
    return get_third_bits(d >> Axis);
  }
};

// template <> struct Morton<int64_t> : public detail::MortonCore<64> {
//   static constexpr TV_METAL_CONSTANT int kNumBits = 64;
//   TV_HOST_DEVICE_INLINE static uint64_t encode(int32_t x, int32_t y,
//                                                int32_t z) {
//     auto abs_x = x >= 0 ? x : -x;
//     auto abs_y = y >= 0 ? y : -y;
//     auto abs_z = z >= 0 ? z : -z;
//     return (split_by_3bits(abs_z | ((z & 0x80000000u) >> 11)) << 2) |
//            (split_by_3bits(abs_y | ((y & 0x80000000u) >> 11)) << 1) |
//            (split_by_3bits(abs_x | ((x & 0x80000000u) >> 11)) << 0);
//   }
//   TV_HOST_DEVICE_INLINE static tv::array<int32_t, 3> decode(uint64_t d) {
//     auto x_with_sign = get_third_bits(d);
//     auto y_with_sign = get_third_bits(d >> 1);
//     auto z_with_sign = get_third_bits(d >> 2);
//     auto x_sign = x_with_sign & 0x100000u;
//     auto y_sign = y_with_sign & 0x100000u;
//     auto z_sign = z_with_sign & 0x100000u;
//     int32_t x_abs = static_cast<int32_t>(x_with_sign & 0xfffffu);
//     int32_t y_abs = static_cast<int32_t>(y_with_sign & 0xfffffu);
//     int32_t z_abs = static_cast<int32_t>(z_with_sign & 0xfffffu);
//     return {x_sign ? -x_abs : x_abs,
//             y_sign ? -y_abs : y_abs,
//             z_sign ? -z_abs : z_abs};
//   }
//   template <uint32_t Axis>
//   TV_HOST_DEVICE_INLINE static uint32_t decode_axis(uint64_t d) {
//     auto x_with_sign = get_third_bits(d >> Axis);
//     auto x_sign = x_with_sign & 0x100000;
//     auto x_abs = x_with_sign & 0xfffff;
//     return x_sign ? -x_abs : x_abs;
//   }
// };

template <> struct Morton<uint64_t> : public detail::MortonCore<64> {
  static constexpr TV_METAL_CONSTANT int kNumBits = 64;
  TV_HOST_DEVICE_INLINE static uint64_t encode(uint32_t x, uint32_t y,
                                               uint32_t z) {
    return (split_by_3bits(z) << 2) | (split_by_3bits(y) << 1) |
           (split_by_3bits(x) << 0);
  }

  TV_HOST_DEVICE_INLINE static tv::array<uint32_t, 3> decode(uint64_t d) {
    return {get_third_bits(d), get_third_bits(d >> 1), get_third_bits(d >> 2)};
  }
  template <uint32_t Axis>
  TV_HOST_DEVICE_INLINE static uint32_t decode_axis(uint64_t d) {
    return get_third_bits(d >> Axis);
  }
};
#ifndef TV_METAL_CC
template <> struct Morton<std::conditional_t<std::is_same<uint64_t, unsigned long long>::value, detail::__place_holder_t, unsigned long long>> : public detail::MortonCore<64> {
  static constexpr TV_METAL_CONSTANT int kNumBits = 64;
  TV_HOST_DEVICE_INLINE static uint64_t encode(uint32_t x, uint32_t y,
                                               uint32_t z) {
    return (split_by_3bits(z) << 2) | (split_by_3bits(y) << 1) |
           (split_by_3bits(x) << 0);
  }

  TV_HOST_DEVICE_INLINE static tv::array<uint32_t, 3> decode(uint64_t d) {
    return {get_third_bits(d), get_third_bits(d >> 1), get_third_bits(d >> 2)};
  }
  template <uint32_t Axis>
  TV_HOST_DEVICE_INLINE static uint32_t decode_axis(uint64_t d) {
    return get_third_bits(d >> Axis);
  }
};
#endif

template <typename K> struct Murmur3Hash {
  using key_type = tv::hash::to_unsigned_t<K>;

  TV_HOST_DEVICE_INLINE static key_type hash(key_type k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
  }
  TV_HOST_DEVICE_INLINE static key_type encode(key_type x) { 
    return detail::platform_not_support_atomic_64_metas<K>::map_user_key(x);
  }
  TV_HOST_DEVICE_INLINE static key_type hash_scalar(key_type key) {
    key = detail::platform_not_support_atomic_64_metas<K>::unmap_user_key(key);
    return hash(key);
  }
};
template <typename K> struct SpatialHash {
  TV_HOST_DEVICE_INLINE static constexpr K direct_hash_offset(){
    return std::conditional_t<sizeof_bits_v<K> == 32,
                         std::integral_constant<K, 0x100>,
                         std::integral_constant<K, 0x100000>>::value;
  }
  using key_type = tv::hash::to_unsigned_t<K>;
  TV_HOST_DEVICE_INLINE static key_type hash(uint32_t x, uint32_t y,
                                             uint32_t z) {
    return x * 73856096u ^ y * 193649663u ^ z * 83492791u;
  }
  TV_HOST_DEVICE_INLINE static key_type encode(uint32_t x, uint32_t y,
                                               uint32_t z) {
    auto res = Morton<key_type>::encode(x, y, z);
    return detail::platform_not_support_atomic_64_metas<K>::map_user_key(res);
  }
  // this function hash a single number by decode and hash from existed key
  TV_HOST_DEVICE_INLINE static key_type hash_scalar(key_type key) {
    key = detail::platform_not_support_atomic_64_metas<K>::unmap_user_key(key);
    auto decoded = Morton<key_type>::decode(key);
    return hash(decoded[0], decoded[1], decoded[2]);
  }
};

template <typename K> struct IdentityHash {
  using key_type = tv::hash::to_unsigned_t<K>;

  TV_HOST_DEVICE_INLINE static key_type hash(key_type k) { return k; }
  TV_HOST_DEVICE_INLINE static key_type encode(key_type x) { 
    return detail::platform_not_support_atomic_64_metas<K>::map_user_key(x);
  }
  TV_HOST_DEVICE_INLINE static key_type hash_scalar(key_type key) {
    key = detail::platform_not_support_atomic_64_metas<K>::unmap_user_key(key);
    return hash(key);
  }
};

namespace detail {

template <typename T> struct FNVInternal;
template <> struct FNVInternal<uint32_t> {
  constexpr static TV_METAL_CONSTANT uint32_t defaultOffsetBasis = 0x811C9DC5;
  constexpr static TV_METAL_CONSTANT uint32_t prime = 0x01000193;
};

template <> struct FNVInternal<uint64_t> {
  constexpr static TV_METAL_CONSTANT uint64_t defaultOffsetBasis = 0xcbf29ce484222325;
  constexpr static TV_METAL_CONSTANT uint64_t prime = 0x100000001b3;
};

#ifdef TV_CUDA_CC
static constexpr bool kIsUint64SameAsULL =
    std::is_same<uint64_t, unsigned long long>::value;

template <>
struct FNVInternal<
    std::conditional<kIsUint64SameAsULL, int64_t, unsigned long long>::type> {
  constexpr static unsigned long long defaultOffsetBasis = 0xcbf29ce484222325;
  constexpr static unsigned long long prime = 0x100000001b3;
};
#endif
} // namespace detail

template <typename K>
struct FNV1aHash : detail::FNVInternal<tv::hash::to_unsigned_t<K>> {
  using key_type = tv::hash::to_unsigned_t<K>;
  TV_HOST_DEVICE_INLINE static key_type hash(key_type key) {
    key_type ret = detail::FNVInternal<key_type>::defaultOffsetBasis;
    key_type key_u = *(reinterpret_cast<TV_METAL_THREAD key_type *>(&key));
    // const char* key_ptr = reinterpret_cast<const char*>(&key);
    TV_PRAGMA_UNROLL
    for (size_t i = 0; i < sizeof(K); ++i) {
      ret ^= (key_u >> (i * 8)) & 0xff;
      // ret ^= key_ptr[i];
      ret *= detail::FNVInternal<key_type>::prime;
    }
    return ret;
  }
  TV_HOST_DEVICE_INLINE static key_type encode(key_type x) { 
    return detail::platform_not_support_atomic_64_metas<K>::map_user_key(x);
  }
  TV_HOST_DEVICE_INLINE static key_type hash_scalar(key_type key) {
    key = detail::platform_not_support_atomic_64_metas<K>::unmap_user_key(key);
    return hash(key);
  }
};

} // namespace hash
} // namespace tv

#ifdef __METAL_VERSION__ 
#pragma METAL internals : disable

#endif
