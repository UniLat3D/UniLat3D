// Copyright (C) 2008-2021 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

#pragma once

#include "core.h"
#include <tensorview/core/defs.h>

#ifdef __CUDACC_RTC__
#include <cuda/std/cfloat>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/cassert>

#define _GLIBCXX_USE_CONSTEXPR constexpr
#define _GLIBCXX_CONSTEXPR TV_HOST_DEVICE_INLINE constexpr
#define _GLIBCXX_USE_NOEXCEPT noexcept

#define __glibcxx_integral_traps false

#define __glibcxx_signed_b(T, B) ((T)(-1) < 0)

#define __glibcxx_min_b(T, B)                                                  \
  (__glibcxx_signed_b(T, B) ? -__glibcxx_max_b(T, B) - 1 : (T)0)

#define __glibcxx_max_b(T, B)                                                  \
  (__glibcxx_signed_b(T, B)                                                    \
       ? (((((T)1 << (__glibcxx_digits_b(T, B) - 1)) - 1) << 1) + 1)           \
       : ~(T)0)

#define __glibcxx_digits_b(T, B) (B - __glibcxx_signed_b(T, B))

// The fraction 643/2136 approximates log10(2) to 7 significant digits.
#define __glibcxx_digits10_b(T, B) (__glibcxx_digits_b(T, B) * 643L / 2136)

#define __glibcxx_signed(T) __glibcxx_signed_b(T, sizeof(T) * __CHAR_BIT__)
#define __glibcxx_min(T) __glibcxx_min_b(T, sizeof(T) * __CHAR_BIT__)
#define __glibcxx_max(T) __glibcxx_max_b(T, sizeof(T) * __CHAR_BIT__)
#define __glibcxx_digits(T) __glibcxx_digits_b(T, sizeof(T) * __CHAR_BIT__)
#define __glibcxx_digits10(T) __glibcxx_digits10_b(T, sizeof(T) * __CHAR_BIT__)

#define __glibcxx_max_digits10(T) (2 + (T)*643L / 2136)

namespace std {

enum float_round_style {
  round_indeterminate = -1,     /// Intermediate.
  round_toward_zero = 0,        /// To zero.
  round_to_nearest = 1,         /// To the nearest representable value.
  round_toward_infinity = 2,    /// To infinity.
  round_toward_neg_infinity = 3 /// To negative infinity.
};

/**
 *  @brief Describes the denormalization for floating-point types.
 *
 *  These values represent the presence or absence of a variable number
 *  of exponent bits.  This type is used in the std::numeric_limits class.
 */
enum float_denorm_style {
  /// Indeterminate at compile time whether denormalized values are allowed.
  denorm_indeterminate = -1,
  /// The type does not allow denormalized values.
  denorm_absent = 0,
  /// The type allows denormalized values.
  denorm_present = 1
};

/**
 *  @brief Part of std::numeric_limits.
 *
 *  The @c static @c const members are usable as integral constant
 *  expressions.
 *
 *  @note This is a separate class for purposes of efficiency; you
 *        should only access these members as part of an instantiation
 *        of the std::numeric_limits class.
 */
struct __numeric_limits_base {
  /** This will be true for all fundamental types (which have
      specializations), and false for everything else.  */
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = false;

  /** The number of @c radix digits that be represented without change:  for
      integer types, the number of non-sign bits in the mantissa; for
      floating types, the number of @c radix digits in the mantissa.  */
  static _GLIBCXX_USE_CONSTEXPR int digits = 0;

  /** The number of base 10 digits that can be represented without change. */
  static _GLIBCXX_USE_CONSTEXPR int digits10 = 0;

#if __cplusplus >= 201103L
  /** The number of base 10 digits required to ensure that values which
      differ are always differentiated.  */
  static constexpr int max_digits10 = 0;
#endif

  /** True if the type is signed.  */
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = false;

  /** True if the type is integer.  */
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = false;

  /** True if the type uses an exact representation. All integer types are
      exact, but not all exact types are integer.  For example, rational and
      fixed-exponent representations are exact but not integer. */
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = false;

  /** For integer types, specifies the base of the representation.  For
      floating types, specifies the base of the exponent representation.  */
  static _GLIBCXX_USE_CONSTEXPR int radix = 0;

  /** The minimum negative integer such that @c radix raised to the power of
      (one less than that integer) is a normalized floating point number.  */
  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;

  /** The minimum negative integer such that 10 raised to that power is in
      the range of normalized floating point numbers.  */
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;

  /** The maximum positive integer such that @c radix raised to the power of
      (one less than that integer) is a representable finite floating point
      number.  */
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;

  /** The maximum positive integer such that 10 raised to that power is in
      the range of representable finite floating point numbers.  */
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  /** True if the type has a representation for positive infinity.  */
  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;

  /** True if the type has a representation for a quiet (non-signaling)
      Not a Number.  */
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;

  /** True if the type has a representation for a signaling
      Not a Number.  */
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;

  /** See std::float_denorm_style for more information.  */
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;

  /** True if loss of accuracy is detected as a denormalization loss,
      rather than as an inexact result. */
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  /** True if-and-only-if the type adheres to the IEC 559 standard, also
      known as IEEE 754.  (Only makes sense for floating point types.)  */
  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;

  /** True if the set of values representable by the type is
      finite.  All built-in types are bounded, this member would be
      false for arbitrary precision types. */
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = false;

  /** True if the type is @e modulo. A type is modulo if, for any
      operation involving +, -, or * on values of that type whose
      result would fall outside the range [min(),max()], the value
      returned differs from the true value by an integer multiple of
      max() - min() + 1. On most machines, this is false for floating
      types, true for unsigned integers, and true for signed integers.
      See PR22200 about signed integers.  */
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  /** True if trapping is implemented for this type.  */
  static _GLIBCXX_USE_CONSTEXPR bool traps = false;

  /** True if tininess is detected before rounding.  (see IEC 559)  */
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;

  /** See std::float_round_style for more information.  This is only
      meaningful for floating types; integer types will all be
      round_toward_zero.  */
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <typename _Tp> struct numeric_limits : public __numeric_limits_base {
  /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
  static _GLIBCXX_CONSTEXPR _Tp min() _GLIBCXX_USE_NOEXCEPT { return _Tp(); }

  /** The maximum finite value.  */
  static _GLIBCXX_CONSTEXPR _Tp max() _GLIBCXX_USE_NOEXCEPT { return _Tp(); }

#if __cplusplus >= 201103L
  /** A finite value x such that there is no other finite value y
   *  where y < x.  */
  TV_HOST_DEVICE_INLINE static constexpr _Tp lowest() noexcept { return _Tp(); }
#endif

  /** The @e machine @e epsilon:  the difference between 1 and the least
      value greater than 1 that is representable.  */
  static _GLIBCXX_CONSTEXPR _Tp epsilon() _GLIBCXX_USE_NOEXCEPT {
    return _Tp();
  }

  /** The maximum rounding error measurement (see LIA-1).  */
  static _GLIBCXX_CONSTEXPR _Tp round_error() _GLIBCXX_USE_NOEXCEPT {
    return _Tp();
  }

  /** The representation of positive infinity, if @c has_infinity.  */
  static _GLIBCXX_CONSTEXPR _Tp infinity() _GLIBCXX_USE_NOEXCEPT {
    return _Tp();
  }

  /** The representation of a quiet Not a Number,
      if @c has_quiet_NaN. */
  static _GLIBCXX_CONSTEXPR _Tp quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return _Tp();
  }

  /** The representation of a signaling Not a Number, if
      @c has_signaling_NaN. */
  static _GLIBCXX_CONSTEXPR _Tp signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return _Tp();
  }

  /** The minimum positive denormalized value.  For types where
      @c has_denorm is false, this is the minimum positive normalized
      value.  */
  static _GLIBCXX_CONSTEXPR _Tp denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return _Tp();
  }
};

template <> struct numeric_limits<float> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR float min() _GLIBCXX_USE_NOEXCEPT {
    return __FLT_MIN__;
  }

  static _GLIBCXX_CONSTEXPR float max() _GLIBCXX_USE_NOEXCEPT {
    return __FLT_MAX__;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr float lowest() noexcept {
    return -__FLT_MAX__;
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __FLT_MANT_DIG__;
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __FLT_DIG__;
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = __glibcxx_max_digits10(__FLT_MANT_DIG__);
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = false;
  static _GLIBCXX_USE_CONSTEXPR int radix = __FLT_RADIX__;

  static _GLIBCXX_CONSTEXPR float epsilon() _GLIBCXX_USE_NOEXCEPT {
    return __FLT_EPSILON__;
  }

  static _GLIBCXX_CONSTEXPR float round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0.5F;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = __FLT_MIN_EXP__;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = __FLT_MIN_10_EXP__;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = __FLT_MAX_EXP__;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = __FLT_MAX_10_EXP__;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = true;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = true;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = has_quiet_NaN;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_present;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR float infinity() _GLIBCXX_USE_NOEXCEPT {
    return __builtin_huge_valf();
  }

  static _GLIBCXX_CONSTEXPR float quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return __builtin_nanf("");
  }

  static _GLIBCXX_CONSTEXPR float signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return __builtin_nansf("");
  }

  static _GLIBCXX_CONSTEXPR float denorm_min() _GLIBCXX_USE_NOEXCEPT {
    // TODO opencv breaks __FLT_DENORM_MIN__
    return __FLT_MIN__;
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 =
      has_infinity && has_quiet_NaN && has_denorm == denorm_present;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  static _GLIBCXX_USE_CONSTEXPR bool traps = false;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_to_nearest;
};

template <> struct numeric_limits<double> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR double min() _GLIBCXX_USE_NOEXCEPT {
    return __DBL_MIN__;
  }

  static _GLIBCXX_CONSTEXPR double max() _GLIBCXX_USE_NOEXCEPT {
    return __DBL_MAX__;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr double lowest() noexcept { return -__DBL_MAX__; }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __DBL_MANT_DIG__;
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __DBL_DIG__;
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = __glibcxx_max_digits10(__DBL_MANT_DIG__);
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = false;
  static _GLIBCXX_USE_CONSTEXPR int radix = __FLT_RADIX__;

  static _GLIBCXX_CONSTEXPR double epsilon() _GLIBCXX_USE_NOEXCEPT {
    return __DBL_EPSILON__;
  }

  static _GLIBCXX_CONSTEXPR double round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0.5;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = __DBL_MIN_EXP__;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = __DBL_MIN_10_EXP__;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = __DBL_MAX_EXP__;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = __DBL_MAX_10_EXP__;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = true;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = true;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = has_quiet_NaN;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_present;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR double infinity() _GLIBCXX_USE_NOEXCEPT {
    return __builtin_huge_val();
  }

  static _GLIBCXX_CONSTEXPR double quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return __builtin_nan("");
  }

  static _GLIBCXX_CONSTEXPR double signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return __builtin_nans("");
  }

  static _GLIBCXX_CONSTEXPR double denorm_min() _GLIBCXX_USE_NOEXCEPT {
    // TODO opencv breaks __DBL_DENORM_MIN__
    return __DBL_MIN__;
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 =
      has_infinity && has_quiet_NaN && has_denorm == denorm_present;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  static _GLIBCXX_USE_CONSTEXPR bool traps = false;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_to_nearest;
};

template <> struct numeric_limits<signed char> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR signed char min() _GLIBCXX_USE_NOEXCEPT {
    return -SCHAR_MAX - 1;
  }

  static _GLIBCXX_CONSTEXPR signed char max() _GLIBCXX_USE_NOEXCEPT {
    return SCHAR_MAX;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr signed char lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(signed char);
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __glibcxx_digits10(signed char);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR signed char epsilon() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR signed char round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR signed char infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<signed char>(0);
  }

  static _GLIBCXX_CONSTEXPR signed char quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<signed char>(0);
  }

  static _GLIBCXX_CONSTEXPR signed char signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<signed char>(0);
  }

  static _GLIBCXX_CONSTEXPR signed char denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<signed char>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

/// numeric_limits<unsigned char> specialization.
template <> struct numeric_limits<unsigned char> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR unsigned char min() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned char max() _GLIBCXX_USE_NOEXCEPT {
    return SCHAR_MAX * 2U + 1;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr unsigned char lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(unsigned char);
  static _GLIBCXX_USE_CONSTEXPR int digits10 =
      __glibcxx_digits10(unsigned char);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR unsigned char epsilon() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned char round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR unsigned char infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned char>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned char quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned char>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned char
  signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned char>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned char denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned char>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = true;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<short> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR short min() _GLIBCXX_USE_NOEXCEPT {
    return -SHRT_MAX - 1;
  }

  static _GLIBCXX_CONSTEXPR short max() _GLIBCXX_USE_NOEXCEPT {
    return SHRT_MAX;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr short lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(short);
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __glibcxx_digits10(short);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR short epsilon() _GLIBCXX_USE_NOEXCEPT { return 0; }

  static _GLIBCXX_CONSTEXPR short round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR short infinity() _GLIBCXX_USE_NOEXCEPT {
    return short();
  }

  static _GLIBCXX_CONSTEXPR short quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return short();
  }

  static _GLIBCXX_CONSTEXPR short signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return short();
  }

  static _GLIBCXX_CONSTEXPR short denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return short();
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<unsigned short> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR unsigned short min() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned short max() _GLIBCXX_USE_NOEXCEPT {
    return SHRT_MAX * 2U + 1;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr unsigned short lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(unsigned short);
  static _GLIBCXX_USE_CONSTEXPR int digits10 =
      __glibcxx_digits10(unsigned short);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR unsigned short epsilon() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned short round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR unsigned short infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned short>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned short quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned short>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned short
  signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned short>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned short denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned short>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = true;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<int> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR int min() _GLIBCXX_USE_NOEXCEPT {
    return -INT_MIN - 1;
  }

  static _GLIBCXX_CONSTEXPR int max() _GLIBCXX_USE_NOEXCEPT { return INT_MAX; }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr int lowest() noexcept { return min(); }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(int);
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __glibcxx_digits10(int);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR int epsilon() _GLIBCXX_USE_NOEXCEPT { return 0; }

  static _GLIBCXX_CONSTEXPR int round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR int infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<int>(0);
  }

  static _GLIBCXX_CONSTEXPR int quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<int>(0);
  }

  static _GLIBCXX_CONSTEXPR int signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<int>(0);
  }

  static _GLIBCXX_CONSTEXPR int denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<int>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<unsigned int> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR unsigned int min() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned int max() _GLIBCXX_USE_NOEXCEPT {
    return INT_MAX * 2U + 1;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr unsigned int lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(unsigned int);
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __glibcxx_digits10(unsigned int);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR unsigned int epsilon() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned int round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR unsigned int infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned int>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned int quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned int>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned int signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned int>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned int denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned int>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = true;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<long> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR long min() _GLIBCXX_USE_NOEXCEPT {
    return -LONG_MAX - 1;
  }

  static _GLIBCXX_CONSTEXPR long max() _GLIBCXX_USE_NOEXCEPT {
    return LONG_MAX;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr long lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(long);
  static _GLIBCXX_USE_CONSTEXPR int digits10 = __glibcxx_digits10(long);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR long epsilon() _GLIBCXX_USE_NOEXCEPT { return 0; }

  static _GLIBCXX_CONSTEXPR long round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR long infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<long>(0);
  }

  static _GLIBCXX_CONSTEXPR long quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<long>(0);
  }

  static _GLIBCXX_CONSTEXPR long signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<long>(0);
  }

  static _GLIBCXX_CONSTEXPR long denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<long>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<unsigned long> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR unsigned long min() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned long max() _GLIBCXX_USE_NOEXCEPT {
    return LONG_MAX * 2UL + 1;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr unsigned long lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits = __glibcxx_digits(unsigned long);
  static _GLIBCXX_USE_CONSTEXPR int digits10 =
      __glibcxx_digits10(unsigned long);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR unsigned long epsilon() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned long round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR unsigned long infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned long quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned long
  signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned long denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = true;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

template <> struct numeric_limits<unsigned long long> {
  static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

  static _GLIBCXX_CONSTEXPR unsigned long long min() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned long long max() _GLIBCXX_USE_NOEXCEPT {
    return LLONG_MAX * 2ULL + 1;
  }

#if __cplusplus >= 201103L
  TV_HOST_DEVICE_INLINE static constexpr unsigned long long lowest() noexcept {
    return min();
  }
#endif

  static _GLIBCXX_USE_CONSTEXPR int digits =
      __glibcxx_digits(unsigned long long);
  static _GLIBCXX_USE_CONSTEXPR int digits10 =
      __glibcxx_digits10(unsigned long long);
#if __cplusplus >= 201103L
  static constexpr int max_digits10 = 0;
#endif
  static _GLIBCXX_USE_CONSTEXPR bool is_signed = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_integer = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_exact = true;
  static _GLIBCXX_USE_CONSTEXPR int radix = 2;

  static _GLIBCXX_CONSTEXPR unsigned long long epsilon() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_CONSTEXPR unsigned long long
  round_error() _GLIBCXX_USE_NOEXCEPT {
    return 0;
  }

  static _GLIBCXX_USE_CONSTEXPR int min_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent = 0;
  static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = 0;

  static _GLIBCXX_USE_CONSTEXPR bool has_infinity = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = false;
  static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm = denorm_absent;
  static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss = false;

  static _GLIBCXX_CONSTEXPR unsigned long long
  infinity() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long long>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned long long
  quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long long>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned long long
  signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long long>(0);
  }

  static _GLIBCXX_CONSTEXPR unsigned long long
  denorm_min() _GLIBCXX_USE_NOEXCEPT {
    return static_cast<unsigned long long>(0);
  }

  static _GLIBCXX_USE_CONSTEXPR bool is_iec559 = false;
  static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
  static _GLIBCXX_USE_CONSTEXPR bool is_modulo = true;

  static _GLIBCXX_USE_CONSTEXPR bool traps = __glibcxx_integral_traps;
  static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false;
  static _GLIBCXX_USE_CONSTEXPR float_round_style round_style =
      round_toward_zero;
};

} // namespace std

#undef _GLIBCXX_USE_CONSTEXPR
#undef _GLIBCXX_USE_NOEXCEPT
#undef __glibcxx_signed_b
#undef __glibcxx_min_b
#undef __glibcxx_max_b
#undef __glibcxx_digits_b
#undef __glibcxx_digits10_b
#undef __glibcxx_signed
#undef __glibcxx_min
#undef __glibcxx_max
#undef __glibcxx_digits
#undef __glibcxx_digits10
#undef __glibcxx_max_digits10
#undef __glibcxx_integral_traps
#endif