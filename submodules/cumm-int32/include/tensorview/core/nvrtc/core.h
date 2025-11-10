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
#include <tensorview/core/defs.h>
#ifdef TV_PARALLEL_RTC
#ifndef __APPLE__
#include <cuda/std/cassert>
#endif
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
#ifndef __APPLE__
typedef signed long long int64_t;
#endif
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#ifndef __APPLE__
typedef unsigned long long uint64_t;
#endif
namespace std {
typedef unsigned long size_t;
typedef long ptrdiff_t;

template<class T> 
TV_HOST_DEVICE_INLINE constexpr const T max(const T a, const T b)
{
    return (a < b) ? b : a;
}

template<class T> 
TV_HOST_DEVICE_INLINE constexpr const T min(const T a, const T b)
{
    return (b < a) ? b : a;
}

template<class T> 
TV_HOST_DEVICE_INLINE constexpr T abs(const T a)
{
    return a >= T(0) ? a : -a;
}

template <typename T1, typename T2>
struct pair {
    T1 first;
    T2 second;
};

} // namespace std
#endif