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

#pragma once

#include <tensorview/core/all.h>
#include <tensorview/gemm/dtypes/all.h>
#if !defined(__CUDACC_RTC__)
#include <fenv.h>
#include <cfenv>
#else 
#endif 
namespace tv {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace UnaryTransform {
    struct Identity;    ///< None (i.e., identity)
    struct Conjugate;   ///< Complex conjugate
}

/// Floating-point rounding style similare to Standard Library's formats but supporting
/// additional rounding options.
enum class FloatRoundStyle {
  round_indeterminate,          ///< rounding mode unknown
  round_toward_zero,            ///< round toward zero
  round_to_nearest,             ///< round to nearest even
  round_toward_infinity,        ///< round toward infinity
  round_toward_neg_infinity,    ///< round toward negative infinity
  round_half_ulp_truncate,      ///< add 0.5ulp to integer representation then round toward zero
  round_half_ulp_trunc_dntz     ///< like round_half_ulp_truncate, except denorms are rounded *toward* zero
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
struct NumericConverter {

  using result_type = T;
  using source_type = S;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
    static result_type convert(source_type const & s) {

    return static_cast<result_type>(s);
  }

  TV_HOST_DEVICE_INLINE
    result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__)
template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    return __float2int_rn(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    return __float2int_rz(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#elif !defined(__CUDACC_RTC__)

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__)
template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    int32_t intermediate;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(intermediate) : "f"(s));

    return static_cast<result_type>(intermediate);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    int32_t intermediate;
    asm volatile("cvt.rzi.sat.s8.f32 %0, %1;" : "=r"(intermediate) : "f"(s));

    return static_cast<result_type>(intermediate);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, half_t, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = half_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {
    float ss = float(s);
    int32_t intermediate;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(intermediate) : "f"(ss));

    return static_cast<result_type>(intermediate);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, half_t, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = half_t;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {
    float ss = float(s);
    int32_t intermediate;
    asm volatile("cvt.rzi.sat.s8.f32 %0, %1;" : "=r"(intermediate) : "f"(ss));

    return static_cast<result_type>(intermediate);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#elif !defined(__CUDACC_RTC__)

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    int32_t intermediate = (int32_t)std::nearbyint(s);

    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());

    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());

    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    int32_t intermediate = (int32_t)std::nearbyint(s);

    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());

    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());

    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= half_t
template <typename T, FloatRoundStyle Round>
struct NumericConverter<T, T, Round> {

  using result_type = T;
  using source_type = T;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    return s;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= half_t
template <FloatRoundStyle Round>
struct NumericConverter<float, half_t, Round> {

  using result_type = float;
  using source_type = half_t;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<float>(s);

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<half_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<half_t>(s);

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<half_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & flt) {

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half_t(__float2half_rz(flt));
  #else
    // software implementation rounds toward nearest even
    unsigned const& s = reinterpret_cast<unsigned const &>(flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return half_t::bitcast(sign);
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      return half_t::bitcast(u);
    }

    if (exp >= -14) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(15));
      u = uint16_t(((exp & 0x1f) << 10));
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);
        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    u |= sign;

    return half_t::bitcast(u);

  #endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<float, bfloat16_t, Round> {

  using result_type = float;
  using source_type = bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<bfloat16_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {
    return static_cast<bfloat16_t>(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<bfloat16_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {
    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);

    #if defined(__CUDA_ARCH__)
    if (::isfinite(s)) {
      x32 += 0x8000;
    }
    #else
    if (std::isfinite(s)) {
      x32 += 0x8000;
    }
    #endif

    uint16_t x16 = uint16_t((x32 >> 16) & 0xffff);
    return bfloat16_t::bitcast(x16);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<bfloat16_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);
    uint16_t x16 = uint16_t(x32 >> 16);

    return bfloat16_t::bitcast(x16);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> tfloat32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= tfloat32_t
template <FloatRoundStyle Round>
struct NumericConverter<float, tfloat32_t, Round> {

  using result_type = float;
  using source_type = tfloat32_t;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    unsigned storage = reinterpret_cast<unsigned const &>(s);

    if ((storage & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((storage & (1 << 13)) != 0);
      bool round_bit = ((storage & (1 << 12)) != 0);
      bool sticky_bit = ((storage & ((1 << 12) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        storage += uint32_t(1 << 13);
      }

      // Note, the following is intentionally commented out. TF32
      // does not define the low order bits, so they may be left in
      // an undefined state. 
      //
      // By not truncating these bit explicitly, we avoid an extra logical
      // operation.
      //
      // TF32 may be implicitly converted to float by performing this
      // operation as needed.
      //
      // storage = (storage & ~0x1fff);
    }
    else if (storage & ~0xff800000) {
      storage = 0x7fffffff;
    }

    return tfloat32_t::bitcast(storage);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {
    return tfloat32_t::round_half_ulp_truncate(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// This rounding operation is similar to half_ulp_truncate except it rounds denorms toward zero.
/// It avoids predicated code, though it requires a temporary register.
template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_half_ulp_trunc_dntz> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_trunc_dntz;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    unsigned y = reinterpret_cast<unsigned const &>(s);
    y = y & 0xff800000;
    float d = reinterpret_cast<float const &>(y);
    float z = d / float(1 << 11) + s;

    return reinterpret_cast<result_type const &>(z);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);
    return tfloat32_t::bitcast(x & 0xffffe000);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for float to tfloat32_t big and small values
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  FloatRoundStyle RoundBig = FloatRoundStyle::round_toward_zero,
  FloatRoundStyle RoundSmall = FloatRoundStyle::round_half_ulp_truncate
>
struct NumericConverterFastF32 {

  // result_type holds big tfloat32_t at idx(0) and small tfloat32_t at idx(1)
  using result_type = tv::array<tfloat32_t, 2>; 

  // source data type
  using source_type = float;

  // rounding styles for big and small part
  static FloatRoundStyle const kRoundBig = RoundBig;
  static FloatRoundStyle const kRoundSmall = RoundSmall;

  TV_HOST_DEVICE_INLINE
    static result_type convert(source_type const & source) {

    result_type result;
    NumericConverter<tfloat32_t, float, kRoundBig> convert_big_;
    NumericConverter<tfloat32_t, float, kRoundSmall> convert_small_;

    // convert and fill tfloat32_t big at idx 0
    result[0] = convert_big_(source);

    // convert and fill tfloat32_t small at idx 1
    result[1] = convert_small_(source - static_cast<float>(result[0]));

    return result;
  }

  TV_HOST_DEVICE_INLINE
    result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion and Clamp operator for Integers
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S
>
struct NumericConverterClamp {

  using result_type = T;
  using source_type = S;

  TV_HOST_DEVICE_INLINE
    static result_type convert(source_type const & s) {
    NumericConverter<result_type, source_type> convert_op;
    result_type const kClamp_max = std::numeric_limits<result_type>::max();
    result_type const kClamp_min = std::numeric_limits<result_type>::lowest();
    if (s < (source_type)kClamp_min) 
      return kClamp_min;
    if (s > (source_type)kClamp_max)
      return kClamp_max;
    return convert_op(s);
  }

  TV_HOST_DEVICE_INLINE
    result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for tv::array
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion operator for tv::array
template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = UnaryTransform::Identity
>
struct NumericArrayConverter {

  using result_type = tv::array<T, N>;
  using source_type = tv::array<S, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(std::is_same<Transform, UnaryTransform::Identity>::value ||
                std::is_same<Transform, UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      if( std::is_same<Transform, UnaryTransform::Identity>::value )
      {
        result[i] = convert_(s[i]);
      } else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <
  typename T,
  int N,
  FloatRoundStyle Round,
  typename Transform
>
struct NumericArrayConverter<T, T, N, Round, Transform> {

  using result_type = tv::array<T, N>;
  using source_type = tv::array<T, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(std::is_same<Transform, UnaryTransform::Identity>::value ||
                std::is_same<Transform, UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
      if( std::is_same<Transform, UnaryTransform::Identity>::value )
      {
          return s;
      } else {
          result_type result;
          for (int i = 0; i < N; ++i) {
              result[i] = conj(s[i]);
          }
          return result;
      }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<half, 2> <= tv::array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<half_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = tv::array<half_t, 2>;
  using source_type = tv::array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    tv::array<half_t, 2> result;

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
      reinterpret_cast<__half2 &>(result) = __float22half2_rn(reinterpret_cast<float2 const &>(source));
    #else
      NumericConverter<half_t, float, round_style> convert_;
      result[0] = convert_(source[0]);
      result[1] = convert_(source[1]);
    #endif
    
    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float, 2> <= tv::array<half_t, 2>, round to nearest
template <FloatRoundStyle Round>
struct NumericArrayConverter<float, half_t, 2, Round> {

  using result_type = tv::array<float, 2>;
  using source_type = tv::array<half_t, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    tv::array<float, 2> result;

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
      reinterpret_cast<float2 &>(result) = __half22float2(reinterpret_cast<__half2 const &>(source));
    #else
      NumericConverter<float, half_t, round_style> convert_;
      result[0] = convert_(source[0]);
      result[1] = convert_(source[1]);
    #endif
    
    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<half> <= tv::array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<half_t, float, N, Round> {

  using result_type = tv::array<half_t, N>;
  using source_type = tv::array<float, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<half_t, float, 2, Round> convert_vector_;
    NumericConverter<half_t, float, Round> convert_element_;

    result_type result;

    tv::array<half_t, 2> *result_ptr = reinterpret_cast<tv::array<half_t, 2> *>(&result);
    tv::array<float, 2> const *source_ptr = reinterpret_cast<tv::array<float, 2> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};


/// Partial specialization for tv::array<half> <= tv::array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, half_t, N, Round> {

  using result_type = tv::array<float, N>;
  using source_type = tv::array<half_t, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float, half_t, 2, Round> convert_vector_;
    NumericConverter<float, half_t, Round> convert_element_;

    result_type result;

    tv::array<float, 2> *result_ptr = reinterpret_cast<tv::array<float, 2> *>(&result);
    tv::array<half_t, 2> const *source_ptr = reinterpret_cast<tv::array<half_t, 2> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<bfloat16_t, 2> <= tv::array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<bfloat16_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = tv::array<bfloat16_t, 2>;
  using source_type = tv::array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    unsigned d;

    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(source[1]), "f"(source[0]) );

    return reinterpret_cast<result_type const &>(d);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<bfloat16_t> <= tv::array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<bfloat16_t, float, N, Round> {

  using result_type = tv::array<bfloat16_t, N>;
  using source_type = tv::array<float, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<bfloat16_t, float, 2, Round> convert_vector_;
    NumericConverter<bfloat16_t, float, Round> convert_element_;

    result_type result;

    tv::array<bfloat16_t, 2> *result_ptr = reinterpret_cast<tv::array<bfloat16_t, 2> *>(&result);
    tv::array<float, 2> const *source_ptr = reinterpret_cast<tv::array<float, 2> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif // if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conditional guards to enable partial specialization for packed integers 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Partial specialization for tv::array<int8_t, 1> <= tv::array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 1, Round> {

  using result_type = tv::array<int8_t, 1>;
  using source_type = tv::array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {
    NumericConverter<int8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);
   
    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<int8_t, 2> <= tv::array<int, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 2, Round> {

  using result_type = tv::array<int8_t, 2>;
  using source_type = tv::array<int, 2>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    uint32_t tmp;

    asm volatile(
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, 0;\n"
      : "=r"(tmp) : "r"(source[0]), "r"(source[1]));

    uint16_t out = (tmp & 0xffff);
    return reinterpret_cast<result_type const &>(out);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<int8_t, 4> <= tv::array<int, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 4, Round> {

  using result_type = tv::array<int8_t, 4>;
  using source_type = tv::array<int, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.s8.s32.b32   r4, %4, %3, 0;"
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;"
      "}"
      : "=r"(out) : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<int8_t> <= tv::array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = tv::array<int8_t, N>;
  using source_type = tv::array<int, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<int8_t, int, 4, Round> convert_vector_;

    result_type result;

    tv::array<int8_t, 4> *result_ptr = reinterpret_cast<tv::array<int8_t, 4> *>(&result);
    tv::array<int, 4> const *source_ptr = reinterpret_cast<tv::array<int, 4> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<uint8_t, 1> <= tv::array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 1, Round> {

  using result_type = tv::array<uint8_t, 1>;
  using source_type = tv::array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {
    NumericConverter<uint8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);
   
    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<uint8_t, 2> <= tv::array<int, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 2, Round> {

  using result_type = tv::array<uint8_t, 2>;
  using source_type = tv::array<int, 2>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    uint32_t tmp;

    asm volatile(
      "cvt.pack.sat.u8.s32.b32   %0, %2, %1, 0;\n"
      : "=r"(tmp) : "r"(source[0]), "r"(source[1]));

    uint16_t out = (tmp & 0xffff);
    return reinterpret_cast<result_type const &>(out);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<uint8_t, 4> <= tv::array<int, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 4, Round> {

  using result_type = tv::array<uint8_t, 4>;
  using source_type = tv::array<int, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.u8.s32.b32   r4, %4, %3, 0;"
      "cvt.pack.sat.u8.s32.b32   %0, %2, %1, r4;"
      "}"
      : "=r"(out) : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<int8_t> <= tv::array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = tv::array<uint8_t, N>;
  using source_type = tv::array<int, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<uint8_t, int, 4, Round> convert_vector_;

    result_type result;

    tv::array<uint8_t, 4> *result_ptr = reinterpret_cast<tv::array<uint8_t, 4> *>(&result);
    tv::array<int, 4> const *source_ptr = reinterpret_cast<tv::array<int, 4> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for tv::array<float, N> <=> tv::array<float_e4m3_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<float, 4> <= tv::array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, float_e4m3_t, 4, Round> {
  using result_element = float;
  using source_element = float_e4m3_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e4m3_t, 4> <= tv::array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float, 4, Round> {
  using result_element = float_e4m3_t;
  using source_element = float;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float, 4> <= tv::array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, float_e5m2_t, 4, Round> {
  using result_element = float;
  using source_element = float_e5m2_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
        "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e5m2_t, 4> <= tv::array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, float, 4, Round> {
  using result_element = float_e5m2_t;
  using source_element = float;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for tv::array<half_t, 4> <=> tv::array<float_e4m3_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<half_t, 4> <= tv::array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<half_t, float_e4m3_t, 4, Round> {
  using result_element = half_t;
  using source_element = float_e4m3_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
        "}\n" : "=r"(out[0]), "=r"(out[1]) : "r"(src_packed));
    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e4m3_t, 4> <= tv::array<half_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, half_t, 4, Round> {
  using result_element = float_e4m3_t;
  using source_element = half_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;
    uint32_t const* src_packed = reinterpret_cast<uint32_t const*>(&source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n" \
        "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "r"(src_packed[0]), "r"(src_packed[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<half_t, 4> <= tv::array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<half_t, float_e5m2_t, 4, Round> {
  using result_element = half_t;
  using source_element = float_e5m2_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
        "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
        "}\n" : "=r"(out[0]), "=r"(out[1]) : "r"(src_packed));
    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e5m2_t, 4> <= tv::array<half_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, half_t, 4, Round> {
  using result_element = float_e5m2_t;
  using source_element = half_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;
    uint32_t const* src_packed = reinterpret_cast<uint32_t const*>(&source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n" \
        "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "r"(src_packed[0]), "r"(src_packed[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for tv::array<bfloat16_t, 4> <=> tv::array<float_e4m3_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<bfloat16_t, 4> <= tv::array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<bfloat16_t, float_e4m3_t, 4, Round> {
  using result_element = bfloat16_t;
  using source_element = float_e4m3_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert f8 to float
    NumericArrayConverter<float, source_element, 4, Round> src2float;
    tv::array<float, 4> tmp_floats = src2float(source);

    // Convert float to bf16
    result_type out;
    tv::array<float, 2>* packed_tmp = reinterpret_cast<tv::array<float, 2>*>(&tmp_floats);
    tv::array<result_element, 2>* packed_out = reinterpret_cast<tv::array<result_element, 2>*>(&out);
    NumericArrayConverter<result_element, float, 2, Round> float2result;
    packed_out[0] = float2result(packed_tmp[0]);
    packed_out[1] = float2result(packed_tmp[1]);

    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e4m3_t, 4> <= tv::array<bfloat16_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, bfloat16_t, 4, Round> {
  using result_element = float_e4m3_t;
  using source_element = bfloat16_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert bf16 to float
    tv::array<float, 4> tmp;
    tv::array<float, 2>* packed_tmp = reinterpret_cast<tv::array<float, 2>*>(&tmp);
    tv::array<source_element, 2> const* packed_source = reinterpret_cast<tv::array<source_element, 2> const*>(&source);
    NumericArrayConverter<float, source_element, 2, Round> src2float;
    packed_tmp[0] = src2float(packed_source[0]);
    packed_tmp[1] = src2float(packed_source[1]);

    // Convert float to f8
    NumericArrayConverter<result_element, float, 4, Round> float2result;
    return float2result(tmp);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<bfloat16_t, 4> <= tv::array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<bfloat16_t, float_e5m2_t, 4, Round> {
  using result_element = bfloat16_t;
  using source_element = float_e5m2_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert f8 to float
    NumericArrayConverter<float, source_element, 4, Round> src2float;
    tv::array<float, 4> tmp_floats = src2float(source);

    // Convert float to bf16
    result_type out;
    tv::array<float, 2>* packed_tmp = reinterpret_cast<tv::array<float, 2>*>(&tmp_floats);
    tv::array<result_element, 2>* packed_out = reinterpret_cast<tv::array<result_element, 2>*>(&out);
    NumericArrayConverter<result_element, float, 2, Round> float2result;
    packed_out[0] = float2result(packed_tmp[0]);
    packed_out[1] = float2result(packed_tmp[1]);

    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e5m2_t, 4> <= tv::array<bfloat16_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, bfloat16_t, 4, Round> {
  using result_element = float_e5m2_t;
  using source_element = bfloat16_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert bf16 to float
    tv::array<float, 4> tmp;
    tv::array<float, 2>* packed_tmp = reinterpret_cast<tv::array<float, 2>*>(&tmp);
    tv::array<source_element, 2> const* packed_source = reinterpret_cast<tv::array<source_element, 2> const*>(&source);
    NumericArrayConverter<float, source_element, 2, Round> src2float;
    packed_tmp[0] = src2float(packed_source[0]);
    packed_tmp[1] = src2float(packed_source[1]);

    // Convert float to f8
    NumericArrayConverter<result_element, float, 4, Round> float2result;
    return float2result(tmp);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for tv::array<float_e4m3_t, 4> <=> tv::array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<float_e4m3_t, 4> <= tv::array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float_e5m2_t, 4, Round> {
  using result_element = float_e4m3_t;
  using source_element = float_e5m2_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<float_e5m2_t, 4> <= tv::array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, float_e4m3_t, 4, Round> {
  using result_element = float_e5m2_t;
  using source_element = float_e4m3_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for:
//      tv::array<float_e4m3_t, 4> <=> tv::array<float_e4m3_t, 4>
//      tv::array<float_e5m2_t, 4> <=> tv::array<float_e5m2_t, 4>
//
// These are needed to avoid multiple-matching-template compilation errors (e.g., when
// compiling float_e4m3_t <=> float_e4m3_t, which among T <= float_e4m3_t and float_e4m3_t <= T
// should be used?)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<float_e4m3_t, 4> <= tv::array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float_e4m3_t, 4, Round> {
  using result_element = float_e4m3_t;
  using source_element = float_e4m3_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return s;
  }
};

/// Partial specialization for tv::array<float_e5m2_t, 4> <= tv::array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, float_e5m2_t, 4, Round> {
  using result_element = float_e5m2_t;
  using source_element = float_e5m2_t;

  using result_type = tv::array<result_element, 4>;
  using source_type = tv::array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specialziations for:
//       tv::array<T, N> <=> tv::array<float_e4m3_t, N>
//       tv::array<T, N> <=> tv::array<float_e5m2_t, N>
// using packed converter under the hood
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round
>
struct PackedNumericArrayConverter {
  using result_element = T;
  using source_element = S;

  using result_type = tv::array<result_element, N>;
  using source_type = tv::array<source_element, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using packed_result_type = tv::array<result_element, 4>;
  using packed_source_type = tv::array<source_element, 4>;

public:
  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {
    result_type result;
    packed_result_type* packed_result = reinterpret_cast<packed_result_type*>(&result);
    const packed_source_type* packed_source = reinterpret_cast<const packed_source_type*>(&source);

    NumericArrayConverter<result_element, source_element, 4, Round> packed_converter;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      packed_result[i] = packed_converter(packed_source[i]);
    }

    // Handle leftovers
    NumericConverter<result_element, source_element, Round> converter;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N % 4; ++i) {
      int idx = ((N / 4) * 4) + i;
      result[idx] = converter(source[idx]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<T, N> <= tv::array<float_e4m3_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<T, float_e4m3_t, N, Round> {};

/// Partial specialization for tv::array<T, N> <= tv::array<float_e5m2_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<T, float_e5m2_t, N, Round> {};

/// Partial specialization for tv::array<float_e4m3_t, N> <= tv::array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, S, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, S, N, Round> {};

/// Partial specialization for tv::array<float_e5m2_t, N> <= tv::array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, S, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, S, N, Round> {};

/// Partial specialization for tv::array<float_e4m3_t, N> <= tv::array<float_e5m2_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, float_e5m2_t, N, Round> {};

/// Partial specialization for tv::array<float_e5m2_t, N> <= tv::array<float_e4m3_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, float_e4m3_t, N, Round> {};

/// Partial specialization for tv::array<float_e4m3_t, N> <= tv::array<float_e4m3_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, float_e4m3_t, N, Round> {};

/// Partial specialization for tv::array<float_e5m2_t, N> <= tv::array<float_e5m2_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, float_e5m2_t, N, Round> {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for tv::array<int8_t> <= tv::array<float>
/// Conversion is performed with saturation regardless of setting of
/// the `Round` template parameter.
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, N, Round> {

  using result_type = tv::array<int8_t, N>;
  using source_type = tv::array<float, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {
    // Convert float to int
    tv::array<int32_t, N> temporary;

    NumericArrayConverter<int, float, N, Round> compute_converter;
    temporary = compute_converter(source);

    // Convert to int to int8_t
    NumericArrayConverter<int8_t, int32_t, N, Round> destination_converter;
    return destination_converter(temporary);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Partial specialization for tv::array<int4b_t, 8> <= tv::array<int, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, int, 8, Round> {

  using result_type = tv::array<int4b_t, 8>;
  using source_type = tv::array<int, 8>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.s4.s32.b32   r4, %8, %7, 0;"
        "cvt.pack.sat.s4.s32.b32   r4, %6, %5, r4;"
        "cvt.pack.sat.s4.s32.b32   r4, %4, %3, r4;"
        "cvt.pack.sat.s4.s32.b32   %0, %2, %1, r4;"
        "}"
        : "=r"(out)
        : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]),
          "r"(source[4]), "r"(source[5]), "r"(source[6]), "r"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<int4b_t> <= tv::array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, int, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = tv::array<int4b_t, N>;
  using source_type = tv::array<int, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<int4b_t, int, 8, Round> convert_vector_;

    result_type result;

    tv::array<int4b_t, 8> *result_ptr = reinterpret_cast<tv::array<int4b_t, 8> *>(&result);
    tv::array<int, 8> const *source_ptr = reinterpret_cast<tv::array<int, 8> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<uint4b_t, 8> <= tv::array<int, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, int, 8, Round> {

  using result_type = tv::array<uint4b_t, 8>;
  using source_type = tv::array<int, 8>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.u4.s32.b32   r4, %8, %7, 0;"
        "cvt.pack.sat.u4.s32.b32   r4, %6, %5, r4;"
        "cvt.pack.sat.u4.s32.b32   r4, %4, %3, r4;"
        "cvt.pack.sat.u4.s32.b32   %0, %2, %1, r4;"
        "}"
        : "=r"(out)
        : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]),
          "r"(source[4]), "r"(source[5]), "r"(source[6]), "r"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for tv::array<int4b_t> <= tv::array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, int, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = tv::array<uint4b_t, N>;
  using source_type = tv::array<int, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<uint4b_t, int, 8, Round> convert_vector_;

    result_type result;

    tv::array<uint4b_t, 8> *result_ptr = reinterpret_cast<tv::array<uint4b_t, 8> *>(&result);
    tv::array<int, 8> const *source_ptr = reinterpret_cast<tv::array<int, 8> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif  // Conditional guards to enable partial specialization for packed integers

/////////////////////////////////////////////////////////////////////////////////////////////////

/// FastNumericArrayConverter only works when the source is within center range.
/// Conversion operator for tv::array.  See the comments before
/// FastLinearCombinationClamp.
template <typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct FastNumericArrayConverter {
  using result_type = tv::array<T, N>;
  using source_type = tv::array<S, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const &s) {
    result_type result;
    NumericArrayConverter<T, S, N, Round> convert_;

    return convert_(s);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) { return convert(s); }
};

/// Partial specialization for tv::array<float> <= tv::array<int>
template <typename T, int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<float, T, N, Round> {
  using result_type = tv::array<float, N>;
  using source_type = tv::array<T, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const &source) {
    result_type result;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int tmp = source[i] + 1262485504 /*0x4B400000*/;
      result[i] = reinterpret_cast<float const &>(tmp) - 12582912.0f;
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) { return convert(s); }
};

/// Partial specialization for tv::array<int8_t, 4> <= tv::array<float, 4>
template <FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, 4, Round> {
  using result_type = tv::array<int8_t, 4>;
  using source_type = tv::array<float, 4>;
  static FloatRoundStyle const round_style = Round;

  TV_DEVICE_INLINE
  static result_type convert(source_type const &source) {
    tv::array<int32_t, 4> result;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      float tmp = source[i] + 12582912.0f;
      result[i] = reinterpret_cast<int32_t const &>(tmp);
    }

    result[0] = __byte_perm(result[0], result[1], 0x40);
    result[2] = __byte_perm(result[2], result[3], 0x40);
    result[0] = __byte_perm(result[0], result[2], 0x5410);

    return reinterpret_cast<result_type const &>(result[0]);
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) { return convert(s); }
};

/// Partial specialization for tv::array<int8_t> <= tv::array<float>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = tv::array<int8_t, N>;
  using source_type = tv::array<float, N>;
  static FloatRoundStyle const round_style = Round;

  TV_HOST_DEVICE_INLINE
  static result_type convert(source_type const &source) {
    FastNumericArrayConverter<int8_t, float, 4, Round> convert_vector_;

    result_type result;

    tv::array<int8_t, 4> *result_ptr =
        reinterpret_cast<tv::array<int8_t, 4> *>(&result);
    tv::array<float, 4> const *source_ptr =
        reinterpret_cast<tv::array<float, 4> const *>(&source);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  result_type operator()(source_type const &s) { return convert(s); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines preferred rounding mode for a pair of types
template <typename T, typename S>
struct PreferredRoundingMode {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_to_nearest;
};

/// Defines preferred rounding mode for a pair of types
template <>
struct PreferredRoundingMode<tfloat32_t, float> {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_half_ulp_truncate;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Packs predicates into an array.
template <int N>
struct PackPredicates {
  using result_type = tv::array<uint1b_t, N>;

  static_assert(!(N % 4), "Must pack predicates in a count that is a multiple of 4");

  TV_HOST_DEVICE_INLINE
  result_type operator()(bool const predicates[]) {

    result_type packed;
    packed.clear();

    int const kWordSize = 8;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(packed.data());

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      uint8_t mask = ((predicates[i] ? 1u : 0u) << bit_idx);
      bytes[word_idx] = (bytes[word_idx] | mask);
    }
    return packed;
  }
};

/// Packs predicates into an array
template <int N>
struct UnpackPredicates {
  using result_type = tv::array<uint1b_t, N>;

  static_assert(!(N % 4), "Must unpack predicates in a count that is a multiple of 4");

  TV_HOST_DEVICE_INLINE
  void operator()(bool predicates[], result_type const &packed) {

    int const kWordSize = 8;
    uint8_t const *bytes = reinterpret_cast<uint8_t const *>(packed.data());

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      predicates[i] = bool((bytes[word_idx] >> bit_idx) & 0x1);
    }

  }
};


} // namespace gemm
} // namespace tv