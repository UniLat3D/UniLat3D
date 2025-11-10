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
#include <tensorview/core/defs.h>
#ifndef TV_PARALLEL_RTC
#include <cmath>
#endif
#ifdef __CUDACC__
#include <cuda_fp16.h>
#if (__CUDACC_VER_MAJOR__ >= 11)
#include <cuda_bf16.h>
#endif
#endif
#ifdef TV_APP_RTC
#include <metal_stdlib>
#endif

namespace tv {

namespace arrayops {

template <typename T> struct MathScalarOp {
  TV_HOST_DEVICE_INLINE static T square(T x) { return x * x; }

#ifndef __CUDACC__
  TV_HOST_DEVICE_INLINE static T copysign(T x, T y) { return std::copysign(x, y); }

  TV_HOST_DEVICE_INLINE static T atan2(T y, T x) { return std::atan2(y, x); }
#ifndef TV_METAL_RTC
  TV_HOST_DEVICE_INLINE static T scalbn(T x, int n) { return std::scalbn(x, n); }
#endif
  TV_HOST_DEVICE_INLINE static T pow(T x, T n) { return std::pow(x, n); }

  TV_HOST_DEVICE_INLINE static T fmod(T x, T n) { return std::fmod(x, n); }

  TV_HOST_DEVICE_INLINE static T neg(T x) { return -x; }

  TV_HOST_DEVICE_INLINE static T sqrt(T x) { return std::sqrt(x); }

  TV_HOST_DEVICE_INLINE static T ceil(T x) { return std::ceil(x); }

  TV_HOST_DEVICE_INLINE static T cos(T x) { return std::cos(x); }

  TV_HOST_DEVICE_INLINE static T exp(T x) { return std::exp(x); }

  TV_HOST_DEVICE_INLINE static T fast_exp(T x) { 
#ifdef TV_METAL_RTC
    return metal::fast::exp(x);
#else 
    return std::exp(x);
#endif
  }

  TV_HOST_DEVICE_INLINE static T fast_sqrt(T x) { 
#ifdef TV_METAL_RTC
    return metal::fast::sqrt(x);
#else 
    return std::sqrt(x);
#endif
  }

  TV_HOST_DEVICE_INLINE static T exp2(T x) { return std::exp2(x); }

  TV_HOST_DEVICE_INLINE static T floor(T x) { return std::floor(x); }

  TV_HOST_DEVICE_INLINE static T log(T x) { return std::log(x); }

  TV_HOST_DEVICE_INLINE static T log10(T x) { return std::log10(x); }

  TV_HOST_DEVICE_INLINE static T log2(T x) { return std::log2(x); }

  TV_HOST_DEVICE_INLINE static T rint(T x) { return std::rint(x); }

  TV_HOST_DEVICE_INLINE static T sin(T x) { return std::sin(x); }

  TV_HOST_DEVICE_INLINE static T trunc(T x) { return std::trunc(x); }

  TV_HOST_DEVICE_INLINE static T abs(T x) { return std::abs(x); }

  TV_HOST_DEVICE_INLINE static T fabs(T x) { return std::abs(x); }

  TV_HOST_DEVICE_INLINE static T tan(T x) { return std::tan(x); }

  TV_HOST_DEVICE_INLINE static T asin(T x) { return std::asin(x); }

  TV_HOST_DEVICE_INLINE static T acos(T x) { return std::acos(x); }

  TV_HOST_DEVICE_INLINE static T atan(T x) { return std::atan(x); }

  TV_HOST_DEVICE_INLINE static T round(T x) { return std::round(x); }

  TV_HOST_DEVICE_INLINE static T sinh(T x) { return std::sinh(x); }

  TV_HOST_DEVICE_INLINE static T cosh(T x) { return std::cosh(x); }

  TV_HOST_DEVICE_INLINE static T tanh(T x) { return std::tanh(x); }

  TV_HOST_DEVICE_INLINE static T asinh(T x) { return std::asinh(x); }

  TV_HOST_DEVICE_INLINE static T acosh(T x) { return std::acosh(x); }

  TV_HOST_DEVICE_INLINE static T atanh(T x) { return std::atanh(x); }

  TV_HOST_DEVICE_INLINE static T rsqrt(T x) { return T(1) / sqrt(x); }

  TV_HOST_DEVICE_INLINE static T max(T x, T y) { return std::max(x, y); }

  TV_HOST_DEVICE_INLINE static T min(T x, T y) { return std::min(x, y); }

  TV_HOST_DEVICE_INLINE static T fmax(T x, T y) { return std::fmax(x, y); }

  TV_HOST_DEVICE_INLINE static T fmin(T x, T y) { return std::fmin(x, y); }

  TV_HOST_DEVICE_INLINE static T clamp(T v, T lo, T hi) { return min(hi, max(lo, v)); }

  TV_HOST_DEVICE_INLINE static T fma(T x, T y, T z) { return std::fma(x, y, z); }

  TV_HOST_DEVICE_INLINE static T mix(T x, T y, T t) { 
#ifdef TV_METAL_RTC
    return metal::mix(x, y, t);
#else 
    return fma(t, y, fma(-t, x, x));
#endif
  }


  TV_HOST_DEVICE_INLINE static float expm1(T x) {
#ifdef TV_METAL_RTC
     return std::exp(x) - T(1);
#else
     return std::expm1(x);
#endif
   }

  TV_HOST_DEVICE_INLINE static float log1p(T x) {
#ifdef TV_METAL_RTC
     return std::log(x + T(1));
#else
     return std::log1p(x);
#endif
   }

  TV_HOST_DEVICE_INLINE static float cbrt(T x) {
#ifdef TV_METAL_RTC
     return std::pow(x, T(1) / T(3));
#else
     return std::cbrt(x);
#endif
   }

  TV_HOST_DEVICE_INLINE static float hypot(T x, T y) {
#ifdef TV_METAL_RTC
     return std::sqrt(x * x + y * y);
#else
     return std::hypot(x, y);
#endif
   }

#else

  TV_HOST_DEVICE_INLINE static T max(T x, T y) { return fmaxf(float(x), float(y)); }

  TV_HOST_DEVICE_INLINE static T min(T x, T y) { return fminf(float(x), float(y)); }

  TV_HOST_DEVICE_INLINE static T clamp(T v, T lo, T hi) { return min(hi, max(lo, v)); }

  TV_HOST_DEVICE_INLINE static T copysign(T x, T y) {
    return T(copysignf(float(x), float(y)));
  }

  TV_HOST_DEVICE_INLINE static T atan2(T y, T x) {
    return T(atan2f(float(y), float(x)));
  }

  TV_HOST_DEVICE_INLINE static T scalbn(T x, int n) {
    return T(scalbnf(float(x), n));
  }

  TV_HOST_DEVICE_INLINE static T pow(T x, T n) {
    return T(pow(float(x), float(n)));
  }

  TV_HOST_DEVICE_INLINE static T fmod(T x, T n) {
    return T(fmodf(float(x), float(n)));
  }

  TV_HOST_DEVICE_INLINE static T sqrt(T x) { return T(sqrtf(float(x))); }

  TV_HOST_DEVICE_INLINE static T ceil(T x) { return T(ceilf(float(x))); }

  TV_HOST_DEVICE_INLINE static T cos(T x) { return T(cosf(float(x))); }

  TV_HOST_DEVICE_INLINE static T exp(T x) { return T(expf(float(x))); }

#ifdef __CUDACC__
  TV_DEVICE_INLINE static T fast_exp(T x) { return __expf(float(x)); }

  TV_DEVICE_INLINE static T fast_sqrt(T x) { return __fsqrt_rn(float(x)); }
#endif
  TV_HOST_DEVICE_INLINE static T exp2(T x) { return T(exp2f(float(x))); }

  TV_HOST_DEVICE_INLINE static T floor(T x) { return T(floorf(float(x))); }

  TV_HOST_DEVICE_INLINE static T log(T x) { return T(logf(float(x))); }

  TV_HOST_DEVICE_INLINE static T log10(T x) { return T(log10f(float(x))); }

  TV_HOST_DEVICE_INLINE static T log2(T x) { return T(log2f(float(x))); }

  TV_HOST_DEVICE_INLINE static T rint(T x) { return T(rintf(float(x))); }

  TV_HOST_DEVICE_INLINE static T sin(T x) { return T(sinf(float(x))); }

  TV_HOST_DEVICE_INLINE static T trunc(T x) { return T(truncf(float(x))); }

  TV_HOST_DEVICE_INLINE static T abs(T x) { return T(fabsf(float(x))); }

  TV_HOST_DEVICE_INLINE static T tan(T x) { return T(tanf(float(x))); }

  TV_HOST_DEVICE_INLINE static T asin(T x) { return T(asinf(float(x))); }

  TV_HOST_DEVICE_INLINE static T acos(T x) { return T(acosf(float(x))); }

  TV_HOST_DEVICE_INLINE static T atan(T x) { return T(atanf(float(x))); }

  TV_HOST_DEVICE_INLINE static T round(T x) { return T(roundf(float(x))); }

  TV_HOST_DEVICE_INLINE static T sinh(T x) { return T(sinhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T cosh(T x) { return T(coshf(float(x))); }

  TV_HOST_DEVICE_INLINE static T tanh(T x) { return T(tanhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T asinh(T x) { return T(asinhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T acosh(T x) { return T(acoshf(float(x))); }

  TV_HOST_DEVICE_INLINE static T atanh(T x) { return T(atanhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T neg(T x) { return -x; }

  TV_HOST_DEVICE_INLINE static T rsqrt(T x) { return T(1) / sqrt(x); }

#endif
};

#ifdef __CUDACC__
template <> struct MathScalarOp<float> {

  TV_HOST_DEVICE_INLINE static float square(float x) { return x * x; }

  TV_HOST_DEVICE_INLINE static float copysign(float x, float y) {
    return copysignf(x, y);
  }

  TV_HOST_DEVICE_INLINE static float atan2(float y, float x) {
    return atan2f(y, x);
  }

  TV_HOST_DEVICE_INLINE static float scalbn(float x, int n) {
    return scalbnf(x, n);
  }

  TV_HOST_DEVICE_INLINE static float pow(float x, float n) {
    return powf(x, n);
  }

  TV_HOST_DEVICE_INLINE static float fmod(float x, float n) {
    return fmodf(x, n);
  }
  TV_HOST_DEVICE_INLINE static float neg(float x) { return -x; }

  TV_HOST_DEVICE_INLINE static float sqrt(float x) { return sqrtf(x); }

  TV_HOST_DEVICE_INLINE static float rsqrt(float x) { 
    return rsqrtf(x);
  }

  TV_HOST_DEVICE_INLINE static float ceil(float x) { return ceilf(x); }

  TV_HOST_DEVICE_INLINE static float cos(float x) { return cosf(x); }

  TV_HOST_DEVICE_INLINE static float exp(float x) { return expf(x); }

  TV_DEVICE_INLINE static float fast_exp(float x) { return __expf(x); }

  TV_DEVICE_INLINE static float fast_sqrt(float x) { return __fsqrt_rn(x); }

  TV_HOST_DEVICE_INLINE static float exp10(float x) { return exp10f(x); }

  TV_HOST_DEVICE_INLINE static float exp2(float x) { return exp2f(x); }

  TV_HOST_DEVICE_INLINE static float floor(float x) { return floorf(x); }

  TV_HOST_DEVICE_INLINE static float log(float x) { return logf(x); }

  TV_HOST_DEVICE_INLINE static float log10(float x) { return log10f(x); }

  TV_HOST_DEVICE_INLINE static float log2(float x) { return log2f(x); }

  TV_HOST_DEVICE_INLINE static float rint(float x) { return rintf(x); }

  TV_HOST_DEVICE_INLINE static float sin(float x) { return sinf(x); }

  TV_HOST_DEVICE_INLINE static float trunc(float x) { return truncf(x); }

  TV_HOST_DEVICE_INLINE static float abs(float x) { return fabsf(x); }

  TV_HOST_DEVICE_INLINE static float fabs(float x) { return fabsf(x); }

  TV_HOST_DEVICE_INLINE static float tan(float x) { return tanf(x); }

  TV_HOST_DEVICE_INLINE static float asin(float x) { return asinf(x); }

  TV_HOST_DEVICE_INLINE static float acos(float x) { return acosf(x); }

  TV_HOST_DEVICE_INLINE static float atan(float x) { return atanf(x); }

  TV_HOST_DEVICE_INLINE static float round(float x) { return roundf(x); }

  TV_HOST_DEVICE_INLINE static float sinh(float x) { return sinhf(x); }

  TV_HOST_DEVICE_INLINE static float cosh(float x) { return coshf(x); }

  TV_HOST_DEVICE_INLINE static float tanh(float x) { return tanhf(x); }

  TV_HOST_DEVICE_INLINE static float asinh(float x) { return asinhf(x); }

  TV_HOST_DEVICE_INLINE static float acosh(float x) { return acoshf(x); }

  TV_HOST_DEVICE_INLINE static float atanh(float x) { return atanhf(x); }

  TV_HOST_DEVICE_INLINE static float max(float x, float y) { return fmaxf(x, y); }

  TV_HOST_DEVICE_INLINE static float min(float x, float y) { return fminf(x, y); }
  
  TV_HOST_DEVICE_INLINE static float fmax(float x, float y) { return fmaxf(x, y); }

  TV_HOST_DEVICE_INLINE static float fmin(float x, float y) { return fminf(x, y); }


  TV_HOST_DEVICE_INLINE static float clamp(float v, float lo, float hi) { return min(hi, max(lo, v)); }

  TV_HOST_DEVICE_INLINE static float fma(float x, float y, float z) { return fmaf(x, y, z); }

  TV_HOST_DEVICE_INLINE static float expm1(float x) { return expm1f(x); }

  TV_HOST_DEVICE_INLINE static float log1p(float x) { return log1pf(x); }

  TV_HOST_DEVICE_INLINE static float cbrt(float x) { return cbrtf(x); }

  TV_HOST_DEVICE_INLINE static float hypot(float x, float y) { return hypotf(x, y); }

  TV_HOST_DEVICE_INLINE static float mix(float x, float y, float t) { return fma(t, y, fma(-t, x, x)); }

};

template <> struct MathScalarOp<double> {
  TV_HOST_DEVICE_INLINE static double square(double x) { return x * x; }

  TV_HOST_DEVICE_INLINE static double copysign(double x, double y) {
    return ::copysign(x, y);
  }

  TV_HOST_DEVICE_INLINE static double atan2(double y, double x) {
    return ::atan2(y, x);
  }

  TV_HOST_DEVICE_INLINE static double scalbn(double x, int n) {
    return ::scalbn(x, n);
  }

  TV_HOST_DEVICE_INLINE static double pow(double x, double n) {
    return ::pow(x, n);
  }

  TV_HOST_DEVICE_INLINE static double fmod(double x, double n) {
    return ::fmod(x, n);
  }
  TV_HOST_DEVICE_INLINE static double neg(double x) { return -x; }

  TV_HOST_DEVICE_INLINE static double sqrt(double x) { return ::sqrt(x); }

  TV_HOST_DEVICE_INLINE static double rsqrt(double x) { return ::rsqrt(x); }

  TV_HOST_DEVICE_INLINE static double ceil(double x) { return ::ceil(x); }

  TV_HOST_DEVICE_INLINE static double cos(double x) { return ::cos(x); }

  TV_HOST_DEVICE_INLINE static double exp(double x) { return ::exp(x); }

  TV_HOST_DEVICE_INLINE static double exp10(double x) { return ::exp10(x); }

  TV_HOST_DEVICE_INLINE static double exp2(double x) { return ::exp2(x); }

  TV_HOST_DEVICE_INLINE static double floor(double x) { return ::floor(x); }

  TV_HOST_DEVICE_INLINE static double log(double x) { return ::log(x); }

  TV_HOST_DEVICE_INLINE static double log10(double x) { return ::log10(x); }

  TV_HOST_DEVICE_INLINE static double log2(double x) { return ::log2(x); }

  TV_HOST_DEVICE_INLINE static double rint(double x) { return ::rint(x); }

  TV_HOST_DEVICE_INLINE static double sin(double x) { return ::sin(x); }

  TV_HOST_DEVICE_INLINE static double trunc(double x) { return ::trunc(x); }

  TV_HOST_DEVICE_INLINE static double abs(double x) { return ::fabs(x); }

  TV_HOST_DEVICE_INLINE static double fabs(double x) { return ::fabs(x); }

  TV_HOST_DEVICE_INLINE static double tan(double x) { return ::tan(x); }

  TV_HOST_DEVICE_INLINE static double asin(double x) { return ::asin(x); }

  TV_HOST_DEVICE_INLINE static double acos(double x) { return ::acos(x); }

  TV_HOST_DEVICE_INLINE static double atan(double x) { return ::atan(x); }

  TV_HOST_DEVICE_INLINE static double round(double x) { return ::round(x); }

  TV_HOST_DEVICE_INLINE static double sinh(double x) { return ::sinh(x); }

  TV_HOST_DEVICE_INLINE static double cosh(double x) { return ::cosh(x); }

  TV_HOST_DEVICE_INLINE static double tanh(double x) { return ::tanh(x); }

  TV_HOST_DEVICE_INLINE static double asinh(double x) { return ::asinh(x); }

  TV_HOST_DEVICE_INLINE static double acosh(double x) { return ::acosh(x); }

  TV_HOST_DEVICE_INLINE static double atanh(double x) { return ::atanh(x); }

  TV_HOST_DEVICE_INLINE static double max(double x, double y) { return ::max(x, y); }

  TV_HOST_DEVICE_INLINE static double min(double x, double y) { return ::min(x, y); }

  TV_HOST_DEVICE_INLINE static double fmax(double x, double y) { return ::max(x, y); }

  TV_HOST_DEVICE_INLINE static double fmin(double x, double y) { return ::min(x, y); }

  TV_HOST_DEVICE_INLINE static double clamp(double v, double lo, double hi) { return min(hi, max(lo, v)); }

  TV_HOST_DEVICE_INLINE static double fma(double x, double y, double z) { return ::fma(x, y, z); }

  TV_HOST_DEVICE_INLINE static double expm1(double x) { return ::expm1(x); }

  TV_HOST_DEVICE_INLINE static double log1p(double x) { return ::log1p(x); }

  TV_HOST_DEVICE_INLINE static double cbrt(double x) { return ::cbrt(x); }

  TV_HOST_DEVICE_INLINE static double hypot(double x, double y) { return ::hypot(x, y); }

  TV_HOST_DEVICE_INLINE static double mix(double x, double y, double t) { return fma(t, y, fma(-t, x, x)); }

};

template <> struct MathScalarOp<__half> {

  TV_DEVICE_INLINE static __half sqrt(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hsqrt(x);
#else
    return __half(sqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half rsqrt(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hrsqrt(x);
#else
    return __half(rsqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half ceil(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hceil(x);
#else
    return __half(ceilf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half cos(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hcos(x);
#else
    return __half(cosf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half exp(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hexp(x);
#else
    return __half(expf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half exp10(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hexp10(x);
#else
    return __half(exp10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half exp2(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hexp2(x);
#else
    return __half(exp2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half floor(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hfloor(x);
#else
    return __half(floorf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half log(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hlog(x);
#else
    return __half(logf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half log10(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hlog10(x);
#else
    return __half(log10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half log2(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hlog2(x);
#else
    return __half(log2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half rint(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hrint(x);
#else
    return __half(rintf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half sin(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hsin(x);
#else
    return __half(sinf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half trunc(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return htrunc(x);
#else
    return __half(truncf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half abs(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __habs(x);
#else
    return __half(fabsf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half2 v2sqrt(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2sqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sqrtf(x0);
    x1 = sqrtf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2rsqrt(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2rsqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rsqrtf(x0);
    x1 = rsqrtf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2ceil(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2ceil(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = ceilf(x0);
    x1 = ceilf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2cos(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2cos(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = cosf(x0);
    x1 = cosf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2exp(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2exp(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = expf(x0);
    x1 = expf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2exp10(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2exp10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp10f(x0);
    x1 = exp10f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2exp2(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2exp2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp2f(x0);
    x1 = exp2f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2floor(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2floor(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = floorf(x0);
    x1 = floorf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2log(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2log(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = logf(x0);
    x1 = logf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2log10(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2log10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log10f(x0);
    x1 = log10f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2log2(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2log2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log2f(x0);
    x1 = log2f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2rint(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2rint(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rintf(x0);
    x1 = rintf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2sin(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2sin(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sinf(x0);
    x1 = sinf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2trunc(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2trunc(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = truncf(x0);
    x1 = truncf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2fabs(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __habs2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = fabsf(x0);
    x1 = fabsf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_HOST_DEVICE_INLINE static __half neg(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hneg(x);
#else
    return __half(-(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half fma(__half x, __half y, __half z) { 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    return __hfma(x, y, z); 
#else 
    return __half(fmaf(float(x), float(y), float(z)));
#endif
  }

  TV_DEVICE_INLINE static __half max(__half x, __half y) { 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmax(x, y);
#else
    return __half(fmaxf(float(x), float(y)));
#endif
 }

  TV_DEVICE_INLINE static __half min(__half x, __half y) { 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmin(x, y); 
#else
    return __half(fminf(float(x), float(y)));
#endif
  }

  TV_DEVICE_INLINE static __half clamp(__half v, __half lo, __half hi) { return min(hi, max(lo, v)); }

  TV_DEVICE_INLINE static __half mix(__half x, __half y, __half t) { return fma(t, y, fma(neg(t), x, x)); }

};

#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11))
template <> struct MathScalarOp<__nv_bfloat16> {

  TV_DEVICE_INLINE static __nv_bfloat16 sqrt(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hsqrt(x);
#else
    return __nv_bfloat16(sqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 rsqrt(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hrsqrt(x);
#else
    return __nv_bfloat16(rsqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 ceil(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hceil(x);
#else
    return __nv_bfloat16(ceilf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 cos(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hcos(x);
#else
    return __nv_bfloat16(cosf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hexp(x);
#else
    return __nv_bfloat16(expf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp10(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hexp10(x);
#else
    return __nv_bfloat16(exp10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp2(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hexp2(x);
#else
    return __nv_bfloat16(exp2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 floor(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hfloor(x);
#else
    return __nv_bfloat16(floorf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hlog(x);
#else
    return __nv_bfloat16(logf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log10(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hlog10(x);
#else
    return __nv_bfloat16(log10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log2(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hlog2(x);
#else
    return __nv_bfloat16(log2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 rint(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hrint(x);
#else
    return __nv_bfloat16(rintf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 sin(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hsin(x);
#else
    return __nv_bfloat16(sinf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 trunc(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return htrunc(x);
#else
    return __nv_bfloat16(truncf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 abs(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __habs(x);
#else
    return __nv_bfloat16(fabsf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2sqrt(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2sqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sqrtf(x0);
    x1 = sqrtf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2rsqrt(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2rsqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rsqrtf(x0);
    x1 = rsqrtf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2ceil(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2ceil(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = ceilf(x0);
    x1 = ceilf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2cos(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2cos(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = cosf(x0);
    x1 = cosf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2exp(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = expf(x0);
    x1 = expf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp10(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2exp10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp10f(x0);
    x1 = exp10f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp2(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2exp2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp2f(x0);
    x1 = exp2f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2floor(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2floor(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = floorf(x0);
    x1 = floorf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2log(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = logf(x0);
    x1 = logf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log10(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2log10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log10f(x0);
    x1 = log10f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log2(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2log2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log2f(x0);
    x1 = log2f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2rint(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2rint(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rintf(x0);
    x1 = rintf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2sin(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2sin(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sinf(x0);
    x1 = sinf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2trunc(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2trunc(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = truncf(x0);
    x1 = truncf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2fabs(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __habs2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = fabsf(x0);
    x1 = fabsf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_HOST_DEVICE_INLINE static __nv_bfloat16 neg(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __hneg(x);
#else
    return __nv_bfloat16(-(float(x)));
#endif
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  TV_DEVICE_INLINE static __nv_bfloat16 max(__nv_bfloat16 x, __nv_bfloat16 y) { return __hmax(x, y); }

  TV_DEVICE_INLINE static __nv_bfloat16 min(__nv_bfloat16 x, __nv_bfloat16 y) { return __hmin(x, y); }

  TV_DEVICE_INLINE static __nv_bfloat16 clamp(__nv_bfloat16 v, __nv_bfloat16 lo, __nv_bfloat16 hi) { return min(hi, max(lo, v)); }

  TV_DEVICE_INLINE static __nv_bfloat16 fma(__nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 z) { return __hfma(x, y, z); }

  TV_DEVICE_INLINE static __nv_bfloat16 mix(__nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 t) { return fma(t, y, fma(neg(t), x, x)); }

#endif

};
#endif

#endif

} // namespace arrayops
} // namespace tv