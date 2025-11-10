/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_runtime_api.h>
#include <tensorview/core/all.h>

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

namespace tv {
namespace gemm {

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2) ||               \
    (__CUDACC_VER_MAJOR__ >= 11)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
#define CUDA_LDMATRIX_ACTIVATED 1
#endif

#define CUDA_LDMATRIX_SUPPORTED 1
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
/*
#if ! defined(CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED) && (__CUDACC_VER_MAJOR__ >
10) #define CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED 1 #endif #if !
defined(CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED) #define
CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED ((__CUDACC_VER_MAJOR__ == 10) &&
(__CUDACC_VER_MINOR__ >= 1)) #endif

#if ! defined(CUDA_NVVM_GET_SMEM_POINTER_ENABLED)
  #define CUDA_NVVM_GET_SMEM_POINTER_ENABLED
CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED #endif
*/

#if (!defined(__clang__) && __CUDACC_VER_MAJOR__ == 10 &&                      \
     __CUDACC_VER_MINOR__ >= 2)
extern "C" {
//
// This NVVM intrinsic is subject to change in future versions of CUDA.
// Clients should not call it directly. Rather, they should use the
// cutlass::arch::ldsm<>() template.
//
__device__ uint32_t __nvvm_get_smem_pointer(void *);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned get_smem_pointer(void *ptr) {

// We prefer to use the new CVTA intrinsics if they are available, otherwise we
// will fall back to the previous internal intrinsics if they are available.
#if (!defined(__clang__) && defined(__CUDA_ARCH__) &&                          \
     __CUDACC_VER_MAJOR__ >= 11)
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
  // available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);

  /// CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));

#elif (!defined(__clang__) && defined(__CUDA_ARCH__) &&                        \
       __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)

  return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

  uint32_t smem_ptr;

  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_ptr)
      : "l"(ptr));

  return smem_ptr;

#else
  assert(0);
  return 0;
#endif
}

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned get_smem_pointer(void const *ptr) {
  return get_smem_pointer(const_cast<void *>(ptr));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace tv