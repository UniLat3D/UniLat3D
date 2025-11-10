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
/*! \file
    \brief Define basic numeric operators with specializations for array<T, N>.
   SIMD-ize where possible.

    This is inspired by the Standard Library's <functional> header.
*/

#pragma once
#include <tensorview/core/all.h>
#include <tensorview/gemm/dtypes/all.h>
#include <tensorview/math/all.h>

namespace tv {
namespace math {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct plus {
  TV_HOST_DEVICE_INLINE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T> struct minus {
  TV_HOST_DEVICE_INLINE
  T operator()(T lhs, T const &rhs) const {
    lhs -= rhs;
    return lhs;
  }
};

template <typename T> struct multiplies {
  TV_HOST_DEVICE_INLINE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};
// template <class T> struct multiplies<array<T, 1>> {
//   static constexpr bool kWTF = true;
// };
template <typename T, size_t N> struct multiplies<array<T, N>> {
  static constexpr bool kWTF = true;
  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, array<T, N> const &rhs) const {

    array<T, N> result;
    multiplies<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, T const &scalar) const {

    array<T, N> result;
    multiplies<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(T const &scalar, array<T, N> const &rhs) const {

    array<T, N> result;
    multiplies<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

/// Squares with optional conversion
template <typename T, typename Output = T> struct square {
  TV_HOST_DEVICE_INLINE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Returns the magnitude squared of an element.
template <typename T, typename Output = T> struct magnitude_squared {
  TV_HOST_DEVICE_INLINE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Squares with optional conversion
template <typename T, typename Output>
struct magnitude_squared<complex<T>, Output> {
  TV_HOST_DEVICE_INLINE
  Output operator()(complex<T> lhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real());
    Output y_i = Output(lhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T> struct square_difference {
  TV_HOST_DEVICE_INLINE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T> struct magnitude_squared_difference {
  TV_HOST_DEVICE_INLINE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output>
struct magnitude_squared_difference<complex<T>, Output> {
  TV_HOST_DEVICE_INLINE
  Output operator()(complex<T> lhs, complex<T> rhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real()) - Output(rhs.real());
    Output y_i = Output(lhs.imag()) - Output(rhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

template <typename T> struct divides {
  TV_HOST_DEVICE_INLINE
  T operator()(T lhs, T const &rhs) const {
    lhs /= rhs;
    return lhs;
  }
};

template <typename T> struct negate {
  TV_HOST_DEVICE_INLINE
  T operator()(T lhs) const { return -lhs; }
};

/// Greater equal
template <typename T> struct greater_equal {
  TV_HOST_DEVICE_INLINE
  bool operator()(T const &lhs, T const &rhs) const { return (lhs >= rhs); }
};

/// Greater
template <typename T> struct greater {
  TV_HOST_DEVICE_INLINE
  bool operator()(T const &lhs, T const &rhs) const { return (lhs > rhs); }
};

/// Less equal
template <typename T> struct less_equal {
  TV_HOST_DEVICE_INLINE
  bool operator()(T const &lhs, T const &rhs) const { return (lhs <= rhs); }
};

/// Less
template <typename T> struct less {
  TV_HOST_DEVICE_INLINE
  bool operator()(T const &lhs, T const &rhs) const { return (lhs < rhs); }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A> struct multiply_add {
  TV_HOST_DEVICE_INLINE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

/// Fused multiply-add
template <typename T> struct and_add {
  TV_HOST_DEVICE_INLINE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a & b) + c);
  }
};

/// Fused multiply-add
template <typename T> struct xor_add {
  TV_HOST_DEVICE_INLINE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a ^ b) + c);
  }
};

template <typename T> struct conjugate {
  TV_HOST_DEVICE_INLINE
  T operator()(T const &a) const { return a; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct conjugate<complex<T>> {
  TV_HOST_DEVICE_INLINE
  complex<T> operator()(complex<T> const &a) const { return conj(a); }
};

template <typename T, size_t N> struct conjugate<array<T, N>> {
  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &a) const {

    conjugate<T> conj_op;

    array<T, N> ca;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      ca[i] = conj_op(a[i]);
    }
    return ca;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specialization for complex<T> to target four scalar fused
// multiply-adds.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Fused multiply-add
template <typename T> struct multiply_add<complex<T>, complex<T>, complex<T>> {
  TV_HOST_DEVICE_INLINE
  complex<T> operator()(complex<T> const &a, complex<T> const &b,
                        complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b.real();
    real += -a.imag() * b.imag();
    imag += a.real() * b.imag();
    imag += a.imag() * b.real();

    return complex<T>{real, imag};
  }
};

/// Fused multiply-add
template <typename T> struct multiply_add<complex<T>, T, complex<T>> {
  TV_HOST_DEVICE_INLINE
  complex<T> operator()(complex<T> const &a, T const &b,
                        complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b;
    imag += a.imag() * b;

    return complex<T>{real, imag};
  }
};

/// Fused multiply-add
template <typename T> struct multiply_add<T, complex<T>, complex<T>> {
  TV_HOST_DEVICE_INLINE
  complex<T> operator()(T const &a, complex<T> const &b,
                        complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a * b.real();
    imag += a * b.imag();

    return complex<T>{real, imag};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for array<T, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N> struct plus<array<T, N>> {
  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, array<T, N> const &rhs) const {

    array<T, N> result;
    plus<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, T const &scalar) const {

    array<T, N> result;
    plus<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(T const &scalar, array<T, N> const &rhs) const {

    array<T, N> result;
    plus<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T> struct maximum {

  TV_HOST_DEVICE_INLINE
  T operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs ? rhs : lhs);
  }
};

template <> struct maximum<float> {
  TV_HOST_DEVICE_INLINE
  float operator()(float const &lhs, float const &rhs) const {
    return fmaxf(lhs, rhs);
  }
};
// template must match array's template parameters, nvcc bug
template <typename T, size_t N> struct maximum<array<T, N>> {

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, array<T, N> const &rhs) const {

    array<T, N> result;
    maximum<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, T const &scalar) const {

    array<T, N> result;
    maximum<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(T const &scalar, array<T, N> const &rhs) const {

    array<T, N> result;
    maximum<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T> struct minimum {

  TV_HOST_DEVICE_INLINE
  constexpr T operator()(T const &lhs, T const &rhs) const {
    return (rhs < lhs ? rhs : lhs);
  }
};

template <> struct minimum<float> {
  TV_HOST_DEVICE_INLINE
  float operator()(float const &lhs, float const &rhs) const {
    return fminf(lhs, rhs);
  }
};

template <typename T, size_t N> struct minimum<array<T, N>> {

  TV_HOST_DEVICE_INLINE
  static constexpr T scalar_op(T const &lhs, T const &rhs) {
    return (rhs < lhs ? rhs : lhs);
  }

  TV_HOST_DEVICE_INLINE
  constexpr array<T, N> operator()(array<T, N> const &lhs,
                                   array<T, N> const &rhs) const {

    array<T, N> result;
    minimum<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  constexpr array<T, N> operator()(array<T, N> const &lhs,
                                   T const &scalar) const {

    array<T, N> result;
    minimum<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  constexpr array<T, N> operator()(T const &scalar,
                                   array<T, N> const &rhs) const {

    array<T, N> result;
    minimum<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, size_t N> struct minus<array<T, N>> {

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, array<T, N> const &rhs) const {

    array<T, N> result;
    minus<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, T const &scalar) const {

    array<T, N> result;
    minus<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(T const &scalar, array<T, N> const &rhs) const {

    array<T, N> result;
    minus<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, size_t N> struct divides<array<T, N>> {

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, array<T, N> const &rhs) const {

    array<T, N> result;
    divides<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs, T const &scalar) const {

    array<T, N> result;
    divides<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(T const &scalar, array<T, N> const &rhs) const {

    array<T, N> result;
    divides<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, size_t N> struct negate<array<T, N>> {

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &lhs) const {

    array<T, N> result;
    negate<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

/// Fused multiply-add
template <typename T, size_t N>
struct multiply_add<array<T, N>, array<T, N>, array<T, N>> {

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &a, array<T, N> const &b,
                         array<T, N> const &c) const {

    array<T, N> result;
    multiply_add<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(array<T, N> const &a, T const &scalar,
                         array<T, N> const &c) const {

    array<T, N> result;
    multiply_add<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<T, N> operator()(T const &scalar, array<T, N> const &b,
                         array<T, N> const &c) const {

    array<T, N> result;
    multiply_add<T> scalar_op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for array<half_t, N> targeting SIMD instructions in
// device code.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <size_t N> struct plus<array<half_t, N>> {
  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hadd(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] + rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(half_t const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual =
          __hadd(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs + rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              half_t const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half d_residual =
          __hadd(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] + rhs;
    }
#endif

    return result;
  }
};

template <size_t N> struct minus<array<half_t, N>> {
  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hsub(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] - rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(half_t const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual =
          __hsub(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs - rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              half_t const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half d_residual =
          __hsub(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] - rhs;
    }
#endif

    return result;
  }
};

template <size_t N> struct multiplies<array<half_t, N>> {
  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hmul(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] * rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(half_t const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual =
          __hmul(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs * rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              half_t const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual =
          __hmul(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] * rhs;
    }
#endif

    return result;
  }
};

template <size_t N> struct divides<array<half_t, N>> {
  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hdiv(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] / rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(half_t const &lhs,
                              array<half_t, N> const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual =
          __hdiv(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs / rhs[i];
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs,
                              half_t const &rhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual =
          __hdiv(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] / rhs;
    }
#endif

    return result;
  }
};

template <size_t N> struct negate<array<half_t, N>> {
  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &lhs) const {
    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *source_ptr = reinterpret_cast<__half2 const *>(&lhs);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hneg2(source_ptr[i]);
    }

    if (N % 2) {
      half_t x = lhs[N - 1];
      __half lhs_val = -reinterpret_cast<__half const &>(x);
      result[N - 1] = reinterpret_cast<half_t const &>(lhs_val);
    }

#else

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = -lhs[i];
    }
#endif

    return result;
  }
};

/// Fused multiply-add
template <size_t N>
struct multiply_add<array<half_t, N>, array<half_t, N>, array<half_t, N>> {

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &a,
                              array<half_t, N> const &b,
                              array<half_t, N> const &c) const {

    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma(a_residual_ptr[N - 1], b_residual_ptr[N - 1],
                                 c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(half_t const &a, array<half_t, N> const &b,
                              array<half_t, N> const &c) const {

    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 a_pair = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_pair, b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {

      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);
      __half d_residual = __hfma(reinterpret_cast<__half const &>(a),
                                 b_residual_ptr[N - 1], c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &a, half_t const &b,
                              array<half_t, N> const &c) const {

    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 b_pair = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_pair, c_ptr[i]);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual =
          __hfma(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(b),
                 c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<half_t, N> operator()(array<half_t, N> const &a,
                              array<half_t, N> const &b,
                              half_t const &c) const {

    array<half_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 c_pair = __half2half2(reinterpret_cast<__half const &>(c));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_ptr[i], c_pair);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);

      __half d_residual = __hfma(a_residual_ptr[N - 1], b_residual_ptr[N - 1],
                                 reinterpret_cast<__half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
#endif

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Fused multiply-add
template <size_t N>
struct multiply_add<array<bfloat16_t, N>, array<bfloat16_t, N>,
                    array<bfloat16_t, N>> {

  TV_HOST_DEVICE_INLINE
  array<bfloat16_t, N> operator()(array<bfloat16_t, N> const &a,
                                  array<bfloat16_t, N> const &b,
                                  array<bfloat16_t, N> const &c) const {

    array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);
    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
          : "=r"(result_ptr[i])
          : "r"(a_ptr[i]), "r"(b_ptr[i]), "r"(c_ptr[i]));
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm("fma.rn.bf16 %0, %1, %2, %3;\n"
          : "=h"(result_ptr[N - 1])
          : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[N - 1]),
            "h"(c_residual_ptr[N - 1]));
    }

#else

    multiply_add<bfloat16_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<bfloat16_t, N> operator()(bfloat16_t const &a,
                                  array<bfloat16_t, N> const &b,
                                  array<bfloat16_t, N> const &c) const {

    array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    unsigned a_packed = static_cast<unsigned>(a.raw());
    a_packed = (a_packed | (a_packed << 16));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
          : "=r"(result_ptr[i])
          : "r"(a_packed), "r"(b_ptr[i]), "r"(c_ptr[i]));
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm("fma.rn.bf16 %0, %1, %2, %3;\n"
          : "=h"(result_ptr[N - 1])
          : "h"(a_residual_ptr[0]), "h"(b_residual_ptr[N - 1]),
            "h"(c_residual_ptr[N - 1]));
    }

#else

    multiply_add<bfloat16_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<bfloat16_t, N> operator()(array<bfloat16_t, N> const &a,
                                  bfloat16_t const &b,
                                  array<bfloat16_t, N> const &c) const {

    array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    unsigned b_packed = static_cast<unsigned>(b.raw());
    b_packed = (b_packed | (b_packed << 16));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
          : "=r"(result_ptr[i])
          : "r"(a_ptr[i]), "r"(b_packed), "r"(c_ptr[i]));
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm("fma.rn.bf16 %0, %1, %2, %3;\n"
          : "=h"(result_ptr[N - 1])
          : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[0]),
            "h"(c_residual_ptr[N - 1]));
    }

#else

    multiply_add<bfloat16_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
#endif

    return result;
  }

  TV_HOST_DEVICE_INLINE
  array<bfloat16_t, N> operator()(array<bfloat16_t, N> const &a,
                                  array<bfloat16_t, N> const &b,
                                  bfloat16_t const &c) const {

    array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);

    unsigned c_packed = static_cast<unsigned>(c.raw());
    c_packed = (c_packed | (c_packed << 16));

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
          : "=r"(result_ptr[i])
          : "r"(a_ptr[i]), "r"(b_ptr[i]), "r"(c_packed));
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm("fma.rn.bf16 %0, %1, %2, %3;\n"
          : "=h"(result_ptr[N - 1])
          : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[N - 1]),
            "h"(c_residual_ptr[0]));
    }

#else

    multiply_add<bfloat16_t> op;

    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
#endif

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator+(array<T, N> const &lhs,
                                            array<T, N> const &rhs) {
  plus<array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator-(array<T, N> const &lhs,
                                            array<T, N> const &rhs) {
  minus<array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator-(array<T, N> const &lhs) {
  negate<array<T, N>> op;
  return op(lhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator*(array<T, N> const &lhs,
                                            array<T, N> const &rhs) {
  multiplies<array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator*(T lhs, array<T, N> const &rhs) {
  multiplies<array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator*(array<T, N> const &lhs, T rhs) {
  multiplies<array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> operator/(array<T, N> const &lhs,
                                            array<T, N> const &rhs) {
  divides<array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N>
fma(array<T, N> const &a, array<T, N> const &b, array<T, N> const &c) {
  multiply_add<array<T, N>> op;
  return op(a, b, c);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> fma(T a, array<T, N> const &b,
                                      array<T, N> const &c) {
  multiply_add<array<T, N>> op;
  return op(a, b, c);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> fma(array<T, N> const &a, T b,
                                      array<T, N> const &c) {
  multiply_add<array<T, N>> op;
  return op(a, b, c);
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE array<T, N> fma(array<T, N> const &a,
                                      array<T, N> const &b, T c) {
  multiply_add<array<T, N>> op;
  return op(a, b, c);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace math

} // namespace tv

/////////////////////////////////////////////////////////////////////////////////////////////////
