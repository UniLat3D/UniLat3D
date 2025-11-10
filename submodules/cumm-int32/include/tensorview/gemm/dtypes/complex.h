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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif
#include <tensorview/core/all.h>
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuComplex.h>
#endif
#include "bfloat16.h"
#include "half.h"
#include "tf32.h"

#if !defined(__CUDACC_RTC__)
#include <iosfwd>
#endif

namespace tv {

template <typename T> struct RealType {
  using Type = T;

  TV_HOST_DEVICE_INLINE
  static T from_real(double x) { return static_cast<T>(x); }
};

template <typename T> TV_HOST_DEVICE_INLINE static T from_real(double r) {
  return T(r);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeraed type describing a transformation on a complex value.
enum class ComplexTransform { kNone, kConjugate };

//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for CUDA complex types
//

/// Returns the real part of the complex number
TV_HOST_DEVICE_INLINE
float const &real(cuFloatComplex const &z) { return z.x; }

/// Returns the real part of the complex number
TV_HOST_DEVICE_INLINE
float &real(cuFloatComplex &z) { return z.x; }

/// Returns the real part of the complex number
TV_HOST_DEVICE_INLINE
double const &real(cuDoubleComplex const &z) { return z.x; }

/// Returns the real part of the complex number
TV_HOST_DEVICE_INLINE
double &real(cuDoubleComplex &z) { return z.x; }

/// Returns the imaginary part of the complex number
TV_HOST_DEVICE_INLINE
float const &imag(cuFloatComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
TV_HOST_DEVICE_INLINE
float &imag(cuFloatComplex &z) { return z.y; }

/// Returns the imaginary part of the complex number
TV_HOST_DEVICE_INLINE
double const &imag(cuDoubleComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
TV_HOST_DEVICE_INLINE
double &imag(cuDoubleComplex &z) { return z.y; }

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for representing and manipulating complex numbers with conversions
/// from built-in CUDA complex types.

template <typename T> class complex {
public:
  /// Type alias for scalar type

private:
  //
  // Data members
  //

  /// Real part
  T _real;

  /// Imaginary part
  T _imag;

public:
  //
  // Methods
  //

  /// Constructor
  TV_HOST_DEVICE_INLINE
  complex(T r = T(0)) : _real(r), _imag(T(0)) {}

  /// Constructor
  TV_HOST_DEVICE_INLINE
  complex(T r, T i) : _real(r), _imag(i) {}
  //
  /// Constructor
  template <typename A>
  TV_HOST_DEVICE_INLINE complex(complex<A> const &z)
      : _real(static_cast<T>(z.real())), _imag(static_cast<T>(z.imag())) {}

  /// Conversion from cuFloatComplex
  TV_HOST_DEVICE_INLINE
  complex(cuFloatComplex const &z)
      : _real(static_cast<T>(cuCrealf(z))), _imag(static_cast<T>(cuCimagf(z))) {
  }

  /// Conversion from cuDoubleComplex
  TV_HOST_DEVICE_INLINE
  complex(cuDoubleComplex const &z)
      : _real(static_cast<T>(cuCreal(z))), _imag(static_cast<T>(cuCimag(z))) {}

  /// Assignment
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> &operator=(complex<A> const &z) {
    _real = static_cast<T>(z.real());
    _imag = static_cast<T>(z.imag());
    return *this;
  }

  /// Equality operator
  TV_HOST_DEVICE_INLINE bool operator==(complex<T> const &rhs) const {
    return this->real() == rhs.real() && this->imag() == rhs.imag();
  }

  /// Inequality operator
  TV_HOST_DEVICE_INLINE bool operator!=(complex<T> const &rhs) const {
    return !(*this == rhs);
  }

  /// Addition
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> operator+(complex<A> const &rhs) const {
    return complex<T>(this->real() + rhs.real(), this->imag() + rhs.imag());
  }

  /// Subtraction
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> operator-(complex<A> const &rhs) const {
    return complex<T>(this->real() - rhs.real(), this->imag() - rhs.imag());
  }

  /// Multiplication
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> operator*(complex<A> const &rhs) const {
    return complex<T>(this->real() * rhs.real() - this->imag() * rhs.imag(),
                      this->real() * rhs.imag() + this->imag() * rhs.real());
  }

  /// Scalar Multiplication
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> operator*(A const &s) const {
    return complex<T>(this->real() * s, this->imag() * s);
  }

  /// Division
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> operator/(complex<A> const &rhs) const {
    T d = T(rhs.real() * rhs.real() + rhs.imag() * rhs.imag());

    return complex<T>((real() * rhs.real() + imag() * rhs.imag()) / d,
                      (imag() * rhs.real() - real() * rhs.imag()) / d);
  }

  /// Scalar Division
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> operator/(A const &s) const {
    return complex<T>(this->real() / s, this->imag() / s);
  }

  /// Addition
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> &operator+=(complex<A> const &rhs) {
    *this = *this + rhs;
    return *this;
  }

  /// Subtraction
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> &operator-=(complex<A> const &rhs) {
    *this = *this - rhs;
    return *this;
  }

  /// Multiplication
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> &operator*=(complex<A> const &rhs) {
    *this = *this * rhs;
    return *this;
  }

  /// Scalar multiplication
  template <typename A> TV_HOST_DEVICE_INLINE complex<T> &operator*=(A s) {
    *this = *this * s;
    return *this;
  }

  /// Division
  template <typename A>
  TV_HOST_DEVICE_INLINE complex<T> &operator/=(complex<A> const &rhs) {
    *this = *this / rhs;
    return *this;
  }

  /// Accesses the real part of the complex number
  TV_HOST_DEVICE_INLINE
  T const &real() const { return _real; }

  /// Accesses the real part of the complex number
  TV_HOST_DEVICE_INLINE
  T &real() { return _real; }

  /// Accesses the imaginary part of the complex number
  TV_HOST_DEVICE_INLINE
  T const &imag() const { return _imag; }

  /// Accesses the imaginary part of the complex number
  TV_HOST_DEVICE_INLINE
  T &imag() { return _imag; }

  /// Converts to cuFloatComplex
  TV_HOST_DEVICE_INLINE
  explicit operator cuFloatComplex() const {
    return make_cuFloatComplex(float(real()), float(imag()));
  }

  /// Converts to cuDoubleComplex
  TV_HOST_DEVICE_INLINE
  explicit operator cuDoubleComplex() const {
    return make_cuDoubleComplex(real(), imag());
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for complex template
//

/// Returns the real part of the complex number
template <typename T> TV_HOST_DEVICE_INLINE T const &real(complex<T> const &z) {
  return z.real();
}

/// Returns the real part of the complex number
template <typename T> TV_HOST_DEVICE_INLINE T &real(complex<T> &z) {
  return z.real();
}

/// Returns the imaginary part of the complex number
template <typename T> TV_HOST_DEVICE_INLINE T const &imag(complex<T> const &z) {
  return z.imag();
}

/// Returns the imaginary part of the complex number
template <typename T> TV_HOST_DEVICE_INLINE T &imag(complex<T> &z) {
  return z.imag();
}

//
// Output operators
//

#if !defined(__CUDACC_RTC__)
template <typename T>
std::ostream &operator<<(std::ostream &out, complex<T> const &z) {
  T _r = real(z);
  T _i = imag(z);

  if (bool(_i)) {
    return out << _r << "+i" << _i;
  }
  return out << _r;
}
#endif

//
// Non-member operators defined for complex types
//

//
// Non-member functions defined for complex numbers
//

/// Returns the magnitude of the complex number
template <typename T> TV_HOST_DEVICE_INLINE T abs(complex<T> const &z) {
  return sqrt(norm(z));
}

/// Returns the magnitude of the complex number
template <typename T> TV_HOST_DEVICE_INLINE T arg(complex<T> const &z) {
  return atan2(imag(z), real(z));
}

/// Returns the squared magnitude of a real number
template <typename T> TV_HOST_DEVICE_INLINE T norm(T const &z) { return z * z; }

/// Returns the squared magnitude of a real number
template <> TV_HOST_DEVICE_INLINE int8_t norm(int8_t const &z) {
  return static_cast<int8_t>(z * z);
}

/// Returns the squared magnitude of a complex number
template <typename T> TV_HOST_DEVICE_INLINE double norm(complex<T> const &z) {
  return real(z) * real(z) + imag(z) * imag(z);
}

/// Norm-accumulate calculation
template <typename T, typename R>
TV_HOST_DEVICE_INLINE R norm_accumulate(T const &x, R const &accumulator) {
  return accumulator + static_cast<R>(x) * static_cast<R>(x);
}

/// Norm accumulate specialized for complex types
template <typename T, typename R>
TV_HOST_DEVICE_INLINE R norm_accumulate(complex<T> const &z,
                                        R const &accumulator) {
  return accumulator + static_cast<R>(real(z)) * static_cast<R>(real(z)) +
         static_cast<R>(imag(z)) * static_cast<R>(imag(z));
}

/// Returns the complex conjugate
TV_HOST_DEVICE_INLINE float conj(float const &z) { return z; }

/// Returns the complex conjugate
TV_HOST_DEVICE_INLINE double conj(double const &z) { return z; }

/// Returns the complex conjugate
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> conj(complex<T> const &z) {
  return complex<T>(real(z), -imag(z));
}
/// Indentity transform for non-complex types
template <typename T> TV_HOST_DEVICE_INLINE T conj(T const &z) {
  static_assert(!std::is_same<T, cuComplex>::value &&
                    !std::is_same<T, cuDoubleComplex>::value &&
                    !std::is_same<T, complex<double>>::value &&
                    !std::is_same<T, complex<float>>::value,
                "May not be a complex data type");
  return z;
}

/// Projects the complex number z onto the Riemann sphere
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> proj(complex<T> const &z) {
  T d = real(z) * real(z) + imag(z) * imag(z) + T(1);
  return complex<T>((T(2) * real(z)) / d, (T(2) * imag(z)) / d);
}

/// Returns a complex number with magnitude r and phase theta
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> polar(T const &r, T const &theta = T()) {
  return complex<T>(r * cos(theta), r * sin(theta));
}

/// Computes the complex exponential of z.
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> exp(complex<T> const &z) {
  return complex<T>(real(z) * cos(imag(z)), real(z) * sin(imag(z)));
}

/// Computes the complex exponential of z.
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> log(complex<T> const &z) {
  return complex<T>(log(abs(z)), arg(z));
}

/// Computes the complex exponential of z.
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> log10(complex<T> const &z) {
  return log(z) / T(log(T(10)));
}

/// Computes the square root of complex number z
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> sqrt(complex<T> const &z) {
  return sqrt(T(2)) / T(2) *
         complex<T>(sqrt(sqrt(norm(z)) + real(z)),
                    (imag(z) < 0 ? T(-1) : T(1)) *
                        sqrt(sqrt(norm(z)) - real(z)));
}

/// Computes the cosine of complex z.
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> cos(complex<T> const &z) {
  return (exp(z) + exp(-z)) / T(2);
}

/// Computes the sin of complex z.
template <typename T>
TV_HOST_DEVICE_INLINE complex<T> sin(complex<T> const &z) {
  return (exp(-z) - exp(z)) * complex<T>(T(0), T(1) / T(2));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex-valued type.
template <typename T> struct RealType<complex<T>> {
  using Type = T;

  TV_HOST_DEVICE_INLINE
  static complex<T> from_real(double x) {
    return complex<T>(static_cast<T>(x));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
TV_HOST_DEVICE_INLINE tv::complex<half_t>
from_real<tv::complex<half_t>>(double r) {
  return tv::complex<half_t>(half_t(r));
}

template <>
TV_HOST_DEVICE_INLINE tv::complex<float>
from_real<tv::complex<float>>(double r) {
  return tv::complex<float>(float(r));
}

template <>
TV_HOST_DEVICE_INLINE tv::complex<double>
from_real<tv::complex<double>>(double r) {
  return tv::complex<double>(r);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct is_complex { static bool const value = false; };

template <typename T> struct is_complex<complex<T>> {
  static bool const value = true;
};

//////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tv

//////////////////////////////////////////////////////////////////////////////////////////////////
