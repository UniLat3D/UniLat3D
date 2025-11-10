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
#include "defs.h"
#include <sstream>
#ifdef TV_USE_STACKTRACE
#if defined(WIN32) || defined(_WIN32) ||                                       \
    defined(__WIN32) && !defined(__CYGWIN__)
#define BOOST_STACKTRACE_USE_WINDBG
#else
// require linking with -ldl and -lbacktrace in linux
#define BOOST_STACKTRACE_USE_BACKTRACE
#endif
#include <boost/stacktrace.hpp>
#endif
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuda.h>
#endif
// llvmlite currently don't support iostream
#ifndef TV_LLVM_JIT
#include <iostream>
#else 
#include "printf2.h"
#endif
#if defined(TV_USE_BOOST_TYPEOF) ||                                            \
    (!defined(__clang__) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000)
// a workaround when built with cuda 11
// #include <boost/typeof/typeof.hpp>
// #define TV_DECLTYPE(x) BOOST_TYPEOF(x)
// two options: use BOOST_TYPEOF or identity_t.
// this is a nvcc bug, msvc/gcc/clang don't have this problem.
namespace tv {
template <typename T> using identity_t = T;
}
#define TV_DECLTYPE(x) tv::identity_t<decltype(x)>
#else
#define TV_DECLTYPE(x) decltype(x)
#endif

#ifdef TV_USE_BACKWARD_TRACE
#include <backward.hpp>
#endif

namespace tv {

template <char Sep = ' ', class SStream, class T>
void sstream_print(SStream &ss, T val) {
  ss << val;
}

template <char Sep = ' ', class SStream, class T, class... TArgs>
void sstream_print(SStream &ss, T val, TArgs... args) {
  if TV_IF_CONSTEXPR (Sep == '\0') {
    ss << val;
  } else {
    ss << val << Sep;
  }
  sstream_print<Sep>(ss, args...);
}

template <char Sep = ' ', class... TArgs> void ssprint(TArgs... args) {
  std::stringstream ss;
  sstream_print<Sep>(ss, args...);
#ifndef TV_LLVM_JIT
  std::cout << ss.str() << std::endl;
#else 
  auto char_str = ss.str();
  tv::printf2(char_str.c_str());
#endif
}

#ifdef TV_USE_STACKTRACE
#define TV_BACKTRACE_PRINT(ss)                                                 \
  ss << std::endl << boost::stacktrace::stacktrace();
#elif defined(TV_USE_BACKWARD_TRACE)
#define TV_BACKTRACE_PRINT(ss)                                                 \
  {                                                                            \
    ss << std::endl;                                                           \
    backward::StackTrace __backtrace_st;                                       \
    __backtrace_st.load_here(32);                                              \
    backward::Printer __backtrace_printer;                                     \
    __backtrace_printer.object = true;                                         \
    __backtrace_printer.color_mode = backward::ColorMode::always;              \
    __backtrace_printer.address = true;                                        \
    __backtrace_printer.print(__backtrace_st, ss);                             \
  }
#else
#define TV_BACKTRACE_PRINT(ss)
#endif

#define TV_THROW_RT_ERR(...)                                                   \
  {                                                                            \
    std::stringstream __macro_s;                                               \
    __macro_s << __FILE__ << "(" << __LINE__ << ")\n";                         \
    TV_BACKTRACE_PRINT(__macro_s);                                             \
    tv::sstream_print(__macro_s, __VA_ARGS__);                                 \
    throw std::runtime_error(__macro_s.str());                                 \
  }

#define TV_THROW_INVALID_ARG(...)                                              \
  {                                                                            \
    std::stringstream __macro_s;                                               \
    __macro_s << __FILE__ << "(" << __LINE__ << ")\n";                         \
    TV_BACKTRACE_PRINT(__macro_s);                                             \
    tv::sstream_print(__macro_s, __VA_ARGS__);                                 \
    throw std::invalid_argument(__macro_s.str());                              \
  }

#define TV_ASSERT_RT_ERR(expr, ...)                                            \
  {                                                                            \
    if (!(expr)) {                                                             \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << "(" << __LINE__ << ")\n";                       \
      __macro_s << #expr << " assert faild. ";                                 \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  }

#define TV_ASSERT_INVALID_ARG(expr, ...)                                       \
  {                                                                            \
    if (!(expr)) {                                                             \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << "(" << __LINE__ << ")\n";                       \
      __macro_s << #expr << " assert faild. ";                                 \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      throw std::invalid_argument(__macro_s.str());                            \
    }                                                                          \
  }

#define TV_DEBUG_PRINT(...)                                                    \
  tv::ssprint(std::string(__FILE__) + "(" + std::to_string(__LINE__) + "):",   \
              __VA_ARGS__)

#define TV_TYPE_STRING(type) boost::core::demangle(typeid(type).name())

#define TV_REQUIRE(expr, ...)                                                  \
  {                                                                            \
    if (!(expr)) {                                                             \
      printf(__VA_ARGS__);                                                     \
      assert(expr);                                                            \
    }                                                                          \
  }

#define TV_CHECK_CUDA_ERR()                                                    \
  {                                                                            \
    auto __macro_err = cudaGetLastError();                                     \
    if (__macro_err != cudaSuccess) {                                          \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << ":" << __LINE__ << "\n";                        \
      __macro_s << "cuda execution failed with error " << __macro_err;         \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  }

#define TV_CHECK_CUDA_ERR_V2(...)                                              \
  {                                                                            \
    auto __macro_err = cudaGetLastError();                                     \
    if (__macro_err != cudaSuccess) {                                          \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << ":" << __LINE__ << "\n";                        \
      __macro_s << "cuda execution failed with error " << __macro_err;         \
      __macro_s << " " << cudaGetErrorString(__macro_err) << "\n";             \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  }

#define TV_CUDART_RESULT_CHECK(EXPR)                                           \
  do {                                                                         \
    cudaError_t __macro_err = EXPR;                                            \
    if (__macro_err != cudaSuccess) {                                          \
      auto error_unused = cudaGetLastError();                                  \
      std::stringstream __macro_s;                                             \
      __macro_s << __func__ << " " << __FILE__ << ":" << __LINE__ << "\n";     \
      __macro_s << "cuda failed with error " << __macro_err;                   \
      __macro_s << " " << cudaGetErrorString(__macro_err);                     \
      __macro_s << ". use CUDA_LAUNCH_BLOCKING=1 to get correct traceback.\n"; \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  } while (0)

#define TV_CUDADRV_RESULT_CHECK(EXPR)                                          \
  do {                                                                         \
    CUresult __macro_err = EXPR;                                               \
    if (__macro_err != CUresult::CUDA_SUCCESS) {                               \
      const char *errstr;                                                      \
      cuGetErrorString(__macro_err, &errstr);                                  \
      std::stringstream __macro_s;                                             \
      __macro_s << __func__ << " " << __FILE__ << ":" << __LINE__ << "\n";     \
      __macro_s << "cuda failed with error " << __macro_err;                   \
      __macro_s << " " << errstr << "\n";                                      \
      __macro_s << ". use CUDA_LAUNCH_BLOCKING=1 to get correct traceback.";   \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  } while (0)

#define TV_CUDA_RESULT_CHECK(EXPR)                                             \
  do {                                                                         \
    auto __macro_err = EXPR;                                                   \
    if (__macro_err) {                                                         \
      std::stringstream __macro_s;                                             \
      __macro_s << __func__ << " " << __FILE__ << ":" << __LINE__ << "\n";     \
      __macro_s << "cuda failed with error code "                               \
                << static_cast<int>(__macro_err);                              \
      __macro_s << ". use CUDA_LAUNCH_BLOCKING=1 to get correct traceback.\n"; \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  } while (0)

#define TV_CUDA_RESULT_CHECK_V2(EXPR, ...)                                     \
  do {                                                                         \
    auto __macro_err = EXPR;                                                   \
    if (__macro_err) {                                                         \
      std::stringstream __macro_s;                                             \
      __macro_s << __func__ << " " << __FILE__ << ":" << __LINE__ << "\n";     \
      __macro_s << "cuda failed with error code "                               \
                << static_cast<int>(__macro_err);                              \
      __macro_s << ". use CUDA_LAUNCH_BLOCKING=1 to get correct traceback.\n"; \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  } while (0)


} // namespace tv