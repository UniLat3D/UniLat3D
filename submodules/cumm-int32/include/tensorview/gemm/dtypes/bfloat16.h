/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Defines a proxy class for storing non-standard 16-bit floating point
   values with 8 bits of exponent and 7 bit of mantissa.
*/
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdint>
#include <limits>
#define CUDA_NAMESPACE_STD std
#else 
#include "fp_nvrtc.h"
#include <tensorview/core/nvrtc_std.h>
#define CUDA_NAMESPACE_STD cuda::std

#endif
#include "half.h"
#include <tensorview/core/all.h>

#ifdef __CUDACC__
#if (__CUDACC_VER_MAJOR__ >= 11)
#include <cuda_bf16.h>
#endif
#endif

namespace tv {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point type with 8 bits of exponent and 7 bits of mantissa.
struct alignas(2) bfloat16_t {

  //
  // Data members
  //

  /// Storage type
  uint16_t storage;

  //
  // Methods
  //

  /// Constructs from an unsigned short
  TV_HOST_DEVICE_INLINE
  static bfloat16_t bitcast(uint16_t x) {
    bfloat16_t h;
    h.storage = x;
    return h;
  }

  /// Default constructor
  TV_HOST_DEVICE_INLINE
  bfloat16_t() : storage(0) {}

  /// Floating-point conversion - round toward nearest
  TV_HOST_DEVICE_INLINE
  explicit bfloat16_t(float x) {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) &&                        \
    (__CUDACC_VER_MAJOR__ >= 11)

    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(storage) : "f"(x));

#else
    uint32_t bits = reinterpret_cast<uint32_t &>(x);

    if ((bits & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((bits & (1 << 16)) != 0);
      bool round_bit = ((bits & (1 << 15)) != 0);
      bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        bits += uint32_t(1 << 16);
      }
    } else if (bits & ~0xff800000) {
      bits = 0x7fffffff;
    }

    storage = uint16_t((bits >> 16) & 0xffff);
#endif
  }

  /// Floating-point conversion - round toward nearest
  TV_HOST_DEVICE_INLINE
  explicit bfloat16_t(double x) : bfloat16_t(float(x)) {}

  /// Integer conversion - round toward nearest
  TV_HOST_DEVICE_INLINE
  explicit bfloat16_t(int x) {
    float flt = static_cast<float>(x);
    storage = uint16_t(reinterpret_cast<uint32_t const &>(flt) >> 16);
  }

  /// Converts to float
  TV_HOST_DEVICE_INLINE
  operator float() const {
    unsigned bits = (unsigned(storage) << 16);
    return reinterpret_cast<float const &>(bits);
  }

  /// Converts to float
  TV_HOST_DEVICE_INLINE
  operator double() const { return double(float(*this)); }

  /// Converts to int
  TV_HOST_DEVICE_INLINE
  explicit operator int() const { return int(float(*this)); }

  /// Casts to bool
  TV_HOST_DEVICE_INLINE
  operator bool() const { return (float(*this) != 0.0f); }

  /// Obtains raw bits
  TV_HOST_DEVICE_INLINE
  uint16_t raw() const { return storage; }
  /// Returns the sign bit
  TV_HOST_DEVICE_INLINE
  bool signbit() const { return ((raw() & 0x8000) != 0); }

  /// Returns the biased exponent
  TV_HOST_DEVICE_INLINE
  int exponent_biased() const { return int((raw() >> 7) & 0x0ff); }

  /// Returns the unbiased exponent
  TV_HOST_DEVICE_INLINE
  int exponent() const { return exponent_biased() - 127; }

  /// Returns the mantissa
  TV_HOST_DEVICE_INLINE
  int mantissa() const { return int(raw() & 0x7f); }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

TV_HOST_DEVICE_INLINE
bool signbit(tv::bfloat16_t const &h) { return h.signbit(); }

TV_HOST_DEVICE_INLINE
tv::bfloat16_t abs(tv::bfloat16_t const &h) {
  return tv::bfloat16_t::bitcast(h.raw() & 0x7fffffff);
}

TV_HOST_DEVICE_INLINE
bool isnan(tv::bfloat16_t const &h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isfinite(tv::bfloat16_t const &h) {
  return (h.exponent_biased() != 0x0ff);
}

TV_HOST_DEVICE_INLINE
tv::bfloat16_t nan_bf16(const char *) {
  // NVIDIA canonical NaN
  return tv::bfloat16_t::bitcast(0x7fff);
}

TV_HOST_DEVICE_INLINE
bool isinf(tv::bfloat16_t const &h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isnormal(tv::bfloat16_t const &h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

TV_HOST_DEVICE_INLINE
int fpclassify(tv::bfloat16_t const &h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x0ff) {
    if (mantissa) {
      return FP_NAN;
    } else {
      return FP_INFINITE;
    }
  } else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    } else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

TV_HOST_DEVICE_INLINE
tv::bfloat16_t sqrt(tv::bfloat16_t const &h) {
#if defined(__CUDACC_RTC__)
  return tv::bfloat16_t(sqrtf(float(h)));
#else
  return tv::bfloat16_t(std::sqrt(float(h)));
#endif
}

TV_HOST_DEVICE_INLINE
bfloat16_t copysign(bfloat16_t const &a, bfloat16_t const &b) {

  uint16_t a_mag = (reinterpret_cast<uint16_t const &>(a) & 0x7fff);
  uint16_t b_sign = (reinterpret_cast<uint16_t const &>(b) & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return reinterpret_cast<bfloat16_t const &>(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tv

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace CUDA_NAMESPACE_STD {

#if !defined(__CUDACC_RTC__)
/// Numeric limits
template <> struct numeric_limits<tv::bfloat16_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 7;

  /// Least positive value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t min() { return tv::bfloat16_t::bitcast(0x01); }

  /// Minimum finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t lowest() { return tv::bfloat16_t::bitcast(0xff7f); }

  /// Maximum finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t max() { return tv::bfloat16_t::bitcast(0x7f7f); }

  /// Returns smallest finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t epsilon() { return tv::bfloat16_t::bitcast(0x1000); }

  /// Returns smallest finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t round_error() { return tv::bfloat16_t(0.5f); }

  /// Returns smallest finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t infinity() { return tv::bfloat16_t::bitcast(0x7f80); }

  /// Returns smallest finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t quiet_NaN() { return tv::bfloat16_t::bitcast(0x7fff); }

  /// Returns smallest finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t signaling_NaN() {
    return tv::bfloat16_t::bitcast(0x7fff);
  }

  /// Returns smallest finite value
  TV_HOST_DEVICE_INLINE
  static tv::bfloat16_t denorm_min() { return tv::bfloat16_t::bitcast(0x1); }
};
#endif

} // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace tv {

///////////////////////////////////////////////////////////////////////////////////////////////////

TV_HOST_DEVICE_INLINE
bool operator==(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return float(lhs) == float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator!=(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return float(lhs) != float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator<(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return float(lhs) < float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator<=(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return float(lhs) <= float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator>(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return float(lhs) > float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator>=(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return float(lhs) >= float(rhs);
}

TV_HOST_DEVICE_INLINE
bfloat16_t operator+(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return bfloat16_t(float(lhs) + float(rhs));
}

TV_HOST_DEVICE_INLINE
bfloat16_t operator-(bfloat16_t const &lhs) { return bfloat16_t(-float(lhs)); }

TV_HOST_DEVICE_INLINE
bfloat16_t operator-(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return bfloat16_t(float(lhs) - float(rhs));
}

TV_HOST_DEVICE_INLINE
bfloat16_t operator*(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return bfloat16_t(float(lhs) * float(rhs));
}

TV_HOST_DEVICE_INLINE
bfloat16_t operator/(bfloat16_t const &lhs, bfloat16_t const &rhs) {
  return bfloat16_t(float(lhs) / float(rhs));
}

TV_HOST_DEVICE_INLINE
bfloat16_t &operator+=(bfloat16_t &lhs, bfloat16_t const &rhs) {
  lhs = bfloat16_t(float(lhs) + float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
bfloat16_t &operator-=(bfloat16_t &lhs, bfloat16_t const &rhs) {
  lhs = bfloat16_t(float(lhs) - float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
bfloat16_t &operator*=(bfloat16_t &lhs, bfloat16_t const &rhs) {
  lhs = bfloat16_t(float(lhs) * float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
bfloat16_t &operator/=(bfloat16_t &lhs, bfloat16_t const &rhs) {
  lhs = bfloat16_t(float(lhs) / float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
bfloat16_t &operator++(bfloat16_t &lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = bfloat16_t(tmp);
  return lhs;
}

TV_HOST_DEVICE_INLINE
bfloat16_t &operator--(bfloat16_t &lhs) {
  float tmp(lhs);
  --tmp;
  lhs = bfloat16_t(tmp);
  return lhs;
}

TV_HOST_DEVICE_INLINE
bfloat16_t operator++(bfloat16_t &lhs, int) {
  bfloat16_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = bfloat16_t(tmp);
  return ret;
}

TV_HOST_DEVICE_INLINE
bfloat16_t operator--(bfloat16_t &lhs, int) {
  bfloat16_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = bfloat16_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// namespace detail {

// template <> struct simple_type_convert<bfloat16_t> {
//   TV_HOST_DEVICE_INLINE auto operator()(bfloat16_t val){
//     return float(val);
//   }
// };

// template <> struct type_to_format<bfloat16_t> {
//   static constexpr tv::array<char, 2> value{'%', 'f'};
// };

// }

namespace detail {

template <> struct TypeToDtype<bfloat16_t> {
  static constexpr DType dtype = bfloat16;
};

#ifdef __CUDACC__
#if (__CUDACC_VER_MAJOR__ >= 11)
template <>
struct equivalent_data_type<bfloat16_t> {
  using type = __nv_bfloat16;
};
#endif
#endif

} // namespace detail

} // namespace tv

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

// TV_HOST_DEVICE_INLINE
// tv::bfloat16_t operator "" _bf16(long double x) {
//   return tv::bfloat16_t(float(x));
// }

// TV_HOST_DEVICE_INLINE
// tv::bfloat16_t operator "" _bf16(unsigned long long int x) {
//   return tv::bfloat16_t(int(x));
// }

/////////////////////////////////////////////////////////////////////////////////////////////////
