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
#ifndef __APPLE__

#if ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6))
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/tuple>
#include <cuda/std/array>
namespace std {
    using namespace cuda::std;
}
#else 
#include "nvrtc/core.h"
#include "nvrtc/type_traits.h"
#include "nvrtc/limits.h"
#include "nvrtc/tuple.h"

#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif


#ifdef __APPLE__

#include "nvrtc/metal_std.h"
#endif