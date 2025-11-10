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
    \brief Defines a proxy class for storing Tensor Float 32 data type.
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

namespace tv {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Tensor Float 32 data type
/// 1bit sign, 8bit exp, 10bit fraction.
struct alignas(4) tfloat32_t {

  //
  // Data members
  //

  /// Storage type
  uint32_t storage;

  //
  // Methods
  //

  /// Constructs from an unsigned int
  TV_HOST_DEVICE_INLINE
  static tfloat32_t bitcast(uint32_t x) {
    tfloat32_t h;
    h.storage = x;
    return h;
  }

  /// Emulated rounding is fast in device code
  TV_HOST_DEVICE_INLINE
  static tfloat32_t round_half_ulp_truncate(float const &s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);

#if defined(__CUDA_ARCH__)
    if (::isfinite(s)) {
      x += 0x1000u;
    }
#else
    if (std::isfinite(s)) {
      x += 0x1000u;
    }
#endif

    return tfloat32_t::bitcast(x);
  }

  /// Default constructor
  TV_HOST_DEVICE_INLINE
  tfloat32_t() : storage(0) {}

  /// Floating-point conversion - round toward nearest even
  TV_HOST_DEVICE_INLINE
  explicit tfloat32_t(float x) : storage(round_half_ulp_truncate(x).storage) {}

  /// Floating-point conversion - round toward nearest even
  TV_HOST_DEVICE_INLINE
  explicit tfloat32_t(double x) : tfloat32_t(float(x)) {}

  /// Integer conversion - round toward zero
  TV_HOST_DEVICE_INLINE
  explicit tfloat32_t(int x) {
    float flt = static_cast<float>(x);
    storage = reinterpret_cast<uint32_t const &>(flt);
  }

  /// Converts to float
  TV_HOST_DEVICE_INLINE
  operator float() const {

    // Conversions to IEEE single-precision requires clearing dont-care bits
    // of the mantissa.
    unsigned bits = (storage & ~0x1fffu);

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
  uint32_t raw() const { return storage; }

  /// Returns the sign bit
  TV_HOST_DEVICE_INLINE
  bool signbit() const { return ((raw() & 0x80000000) != 0); }

  /// Returns the biased exponent
  TV_HOST_DEVICE_INLINE
  int exponent_biased() const { return int((raw() >> 23) & 0x0ff); }

  /// Returns the unbiased exponent
  TV_HOST_DEVICE_INLINE
  int exponent() const { return exponent_biased() - 127; }

  /// Returns the mantissa
  TV_HOST_DEVICE_INLINE
  int mantissa() const { return int(raw() & 0x7fffff); }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

TV_HOST_DEVICE_INLINE
bool signbit(tv::tfloat32_t const &h) { return h.signbit(); }

TV_HOST_DEVICE_INLINE
tv::tfloat32_t abs(tv::tfloat32_t const &h) {
  return tv::tfloat32_t::bitcast(h.raw() & 0x7fffffff);
}

TV_HOST_DEVICE_INLINE
bool isnan(tv::tfloat32_t const &h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isfinite(tv::tfloat32_t const &h) {
  return (h.exponent_biased() != 0x0ff);
}

TV_HOST_DEVICE_INLINE
tv::tfloat32_t nan_tf32(const char *) {
  // NVIDIA canonical NaN
  return tv::tfloat32_t::bitcast(0x7fffffff);
}

TV_HOST_DEVICE_INLINE
bool isinf(tv::tfloat32_t const &h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isnormal(tv::tfloat32_t const &h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

TV_HOST_DEVICE_INLINE
int fpclassify(tv::tfloat32_t const &h) {
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
tv::tfloat32_t sqrt(tv::tfloat32_t const &h) {
#if defined(__CUDACC_RTC__)
  return tv::tfloat32_t(sqrtf(float(h)));
#else
  return tv::tfloat32_t(std::sqrt(float(h)));
#endif
}

TV_HOST_DEVICE_INLINE
tfloat32_t copysign(tfloat32_t const &a, tfloat32_t const &b) {

  uint32_t a_mag = (reinterpret_cast<uint32_t const &>(a) & 0x7fffffff);
  uint32_t b_sign = (reinterpret_cast<uint32_t const &>(b) & 0x80000000);
  uint32_t result = (a_mag | b_sign);

  return reinterpret_cast<tfloat32_t const &>(result);
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
template <> struct numeric_limits<tv::tfloat32_t> {
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
  static int const digits = 19;

  /// Least positive value
  static tv::tfloat32_t min() { return tv::tfloat32_t::bitcast(0x01); }

  /// Minimum finite value
  static tv::tfloat32_t lowest() { return tv::tfloat32_t::bitcast(0xff7fffff); }

  /// Maximum finite value
  static tv::tfloat32_t max() { return tv::tfloat32_t::bitcast(0x7f7fffff); }

  /// Returns smallest finite value
  static tv::tfloat32_t epsilon() { return tv::tfloat32_t::bitcast(0x1000); }

  /// Returns smallest finite value
  static tv::tfloat32_t round_error() { return tv::tfloat32_t(0.5f); }

  /// Returns smallest finite value
  static tv::tfloat32_t infinity() {
    return tv::tfloat32_t::bitcast(0x7f800000);
  }

  /// Returns smallest finite value
  static tv::tfloat32_t quiet_NaN() {
    return tv::tfloat32_t::bitcast(0x7fffffff);
  }

  /// Returns smallest finite value
  static tv::tfloat32_t signaling_NaN() {
    return tv::tfloat32_t::bitcast(0x7fffffff);
  }

  /// Returns smallest finite value
  static tv::tfloat32_t denorm_min() { return tv::tfloat32_t::bitcast(0x1); }
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace tv {

///////////////////////////////////////////////////////////////////////////////////////////////////

TV_HOST_DEVICE_INLINE
bool operator==(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return float(lhs) == float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator!=(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return float(lhs) != float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator<(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return float(lhs) < float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator<=(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return float(lhs) <= float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator>(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return float(lhs) > float(rhs);
}

TV_HOST_DEVICE_INLINE
bool operator>=(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return float(lhs) >= float(rhs);
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator+(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return tfloat32_t(float(lhs) + float(rhs));
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator-(tfloat32_t const &lhs) {
  float x = -reinterpret_cast<float const &>(lhs);
  return reinterpret_cast<tfloat32_t const &>(x);
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator-(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return tfloat32_t(float(lhs) - float(rhs));
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator*(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return tfloat32_t(float(lhs) * float(rhs));
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator/(tfloat32_t const &lhs, tfloat32_t const &rhs) {
  return tfloat32_t(float(lhs) / float(rhs));
}

TV_HOST_DEVICE_INLINE
tfloat32_t &operator+=(tfloat32_t &lhs, tfloat32_t const &rhs) {
  lhs = tfloat32_t(float(lhs) + float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
tfloat32_t &operator-=(tfloat32_t &lhs, tfloat32_t const &rhs) {
  lhs = tfloat32_t(float(lhs) - float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
tfloat32_t &operator*=(tfloat32_t &lhs, tfloat32_t const &rhs) {
  lhs = tfloat32_t(float(lhs) * float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
tfloat32_t &operator/=(tfloat32_t &lhs, tfloat32_t const &rhs) {
  lhs = tfloat32_t(float(lhs) / float(rhs));
  return lhs;
}

TV_HOST_DEVICE_INLINE
tfloat32_t &operator++(tfloat32_t &lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = tfloat32_t(tmp);
  return lhs;
}

TV_HOST_DEVICE_INLINE
tfloat32_t &operator--(tfloat32_t &lhs) {
  float tmp(lhs);
  --tmp;
  lhs = tfloat32_t(tmp);
  return lhs;
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator++(tfloat32_t &lhs, int) {
  tfloat32_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = tfloat32_t(tmp);
  return ret;
}

TV_HOST_DEVICE_INLINE
tfloat32_t operator--(tfloat32_t &lhs, int) {
  tfloat32_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = tfloat32_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// namespace detail {

// template <> struct simple_type_convert<tfloat32_t> {
//   TV_HOST_DEVICE_INLINE auto operator()(tfloat32_t val){
//     return float(val);
//   }
// };

// template <> struct type_to_format<tfloat32_t> {
//   static constexpr tv::array<char, 2> value{'%', 'f'};
// };

// }
namespace detail {

template <> struct TypeToDtype<tfloat32_t> {
  static constexpr DType dtype = tf32;
};

} // namespace detail

} // namespace tv

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

// TV_HOST_DEVICE_INLINE
// tv::tfloat32_t operator "" _tf32(long double x) {
//   return tv::tfloat32_t(float(x));
// }

// TV_HOST_DEVICE_INLINE
// tv::tfloat32_t operator "" _tf32(unsigned long long int x) {
//   return tv::tfloat32_t(int(x));
// }

/////////////////////////////////////////////////////////////////////////////////////////////////
