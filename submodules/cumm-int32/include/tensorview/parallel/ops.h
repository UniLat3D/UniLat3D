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

/*
TODO add some ops (e.g. atomic) that can be used in both cuda and openmp.
TODO add CAS
*/
#pragma once
#include <tensorview/core/all.h>
#include <tensorview/core/arrayops/simple.h>
#if defined(TV_METAL_RTC)
#include <metal_stdlib>
#endif
#if defined(TV_HARDWARE_ACC_CUDA)
#if defined(__CUDACC_RTC__)
#include <tensorview/core/nvrtc_std.h>
#endif
#include <cooperative_groups.h>
#endif

namespace tv {
namespace parallel {
template <typename T1, typename T2> struct DataPair {
  T1 first;
  T2 second;
  static_assert(sizeof(T1) == sizeof(T2), "error");
  static_assert((sizeof(T1) == 4 || sizeof(T1) == 2 || sizeof(T1) == 1),
                "error");
};
namespace detail {

template <int Size> struct AtomicDataType;

template <> struct AtomicDataType<2> {
  using type = uint32_t;
};

template <> struct AtomicDataType<4> {
  using type = uint64_t;
};

template <typename T1, typename T2> union DataPairUnion {
  DataPair<T1, T2> data;
  typename AtomicDataType<sizeof(T1)>::type val;
};

} // namespace detail

#if defined(TV_METAL_RTC) && defined(__METAL_VERSION__) &&                     \
    __METAL_VERSION__ >= 310
namespace internal {
uint __apple_metal_warp_size [[threads_per_simdgroup]];
uint __apple_metal_warp_index [[simdgroup_index_in_threadgroup]];
uint __apple_metal_lane_index [[thread_index_in_simdgroup]];

uint3 __apple_metal_block_indexes [[threadgroup_position_in_grid]];
uint3 __apple_metal_block_dims [[threadgroups_per_grid]];
uint3 __apple_metal_grid_dims [[grid_size]];
uint3 __apple_metal_thread_indexes [[thread_position_in_threadgroup]];

}

using vote_t = metal::simd_vote::vote_t;

TV_DEVICE_INLINE uint warp_size() { return internal::__apple_metal_warp_size; }

TV_DEVICE_INLINE uint warp_index() { return internal::__apple_metal_warp_index; }

TV_DEVICE_INLINE uint lane_index() { return internal::__apple_metal_lane_index; }

TV_DEVICE_INLINE uint3 block_idx() { return internal::__apple_metal_block_indexes; }

TV_DEVICE_INLINE uint3 block_dim() { return internal::__apple_metal_block_dims; }

TV_DEVICE_INLINE uint3 grid_dim() { return internal::__apple_metal_grid_dims; }

TV_DEVICE_INLINE uint3 thread_idx() { return internal::__apple_metal_thread_indexes; }


TV_DEVICE_INLINE void block_sync() { metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup | metal::mem_flags::mem_device); }
TV_DEVICE_INLINE void block_sync_shared_io() { metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup); }

namespace detail {
TV_DEVICE_INLINE vote_t lanemask_lt() {
  return ~(metal::numeric_limits<vote_t>::max() << lane_index());
}
} // namespace detail

#endif

#if defined(TV_CUDA_CC)

using vote_t = uint32_t;
TV_DEVICE_INLINE constexpr uint32_t warp_size() { return 32; }

TV_DEVICE_INLINE uint32_t thread_index() { 
  return ((uint32_t)threadIdx.z * blockDim.y * blockDim.x) +
               ((uint32_t)threadIdx.y * blockDim.x) +
                (uint32_t)threadIdx.x; 
}

TV_DEVICE_INLINE uint32_t warp_index() { 
  return thread_index() / warp_size(); 
}

TV_DEVICE_INLINE uint32_t lane_index() { 
    unsigned int laneid;
    asm ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

TV_DEVICE_INLINE dim3 block_idx() { return blockIdx; }

TV_DEVICE_INLINE dim3 block_dim() { return blockDim; }

TV_DEVICE_INLINE dim3 grid_dim() { return gridDim; }

TV_DEVICE_INLINE dim3 thread_idx() { return threadIdx; }

TV_DEVICE_INLINE void block_sync() { __syncthreads(); }
TV_DEVICE_INLINE void block_sync_shared_io() { __syncthreads(); }

namespace detail {
TV_DEVICE_INLINE vote_t lanemask_lt() {
  vote_t lanemask32_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
  return (lanemask32_lt);
}

} // namespace detail

#endif
namespace detail {
template <typename T> TV_DEVICE_INLINE T atomicAggIncCuda(TV_METAL_DEVICE T *ctr) {
#ifdef TV_CUDA_CC
  namespace cg = cooperative_groups;
  auto g = cg::coalesced_threads();
  T warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, T(g.size()));
  return g.shfl(warp_res, 0) + g.thread_rank();

#endif
  return T(0);
}

}

#if defined(TV_CUDA_CC) || defined(TV_METAL_RTC)

template <typename T> TV_DEVICE_INLINE T atomicAggInc(TV_METAL_DEVICE T *ctr) {
#ifdef TV_METAL_RTC
  T warp_res = 0;
  auto mask = vote_t(metal::simd_active_threads_mask());
  auto thread_rank = metal::popcount(mask & detail::lanemask_lt());
  if (thread_rank == 0) {
    warp_res = metal::atomic_fetch_add_explicit(
        reinterpret_cast<device metal::atomic<T>*>(ctr), 
        T(metal::popcount(metal::simd_vote::vote_t(mask))), metal::memory_order_relaxed);
  }
  return metal::simd_shuffle(warp_res, metal::ctz(mask)) + thread_rank;
#else

  return detail::atomicAggIncCuda(ctr);

#endif
}

template <typename T> TV_DEVICE_INLINE T atomicAdd(TV_METAL_DEVICE T *ctr, T val) {
#ifdef TV_METAL_RTC
  return metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
#else
  return ::atomicAdd(ctr, val);
#endif
}

template <typename T, size_t N> TV_DEVICE_INLINE array<T, N> atomicAdd(TV_METAL_DEVICE array<T, N> *ctr, const TV_METAL_THREAD array<T, N>& val) {
  auto ptr_array = arrayops::create_ptr_arange<N>(reinterpret_cast<TV_METAL_DEVICE T*>(ctr));
  return arrayops::apply(atomicAdd<float>, reinterpret_cast<TV_METAL_DEVICE tv::array<TV_METAL_DEVICE T*, N>&>(ptr_array), val);
}

template <typename T> TV_DEVICE_INLINE T atomicMax(TV_METAL_DEVICE T *ctr, T val) {
#ifdef TV_METAL_RTC
  return metal::atomic_fetch_max_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
#else
  return ::atomicMax(ctr, val);
#endif
}

template <typename T> TV_DEVICE_INLINE T atomicMin(TV_METAL_DEVICE T *ctr, T val) {
#ifdef TV_METAL_RTC
  return metal::atomic_fetch_min_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
#else
  return ::atomicMin(ctr, val);
#endif
}

#ifdef TV_CUDA_RTC
TV_DEVICE_INLINE float atomicMax (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

TV_DEVICE_INLINE float atomicMin (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#endif

#ifdef TV_METAL_RTC

TV_DEVICE_INLINE float __int_as_float(int x){
  return *reinterpret_cast<thread float*>(&x);
}

TV_DEVICE_INLINE int __float_as_int(float x){
  return *reinterpret_cast<thread int*>(&x);
}

TV_DEVICE_INLINE float __uint_as_float(unsigned int x){
  return *reinterpret_cast<thread float*>(&x);
}

TV_DEVICE_INLINE unsigned int __float_as_uint(float x){
  return *reinterpret_cast<thread unsigned int*>(&x);
}

TV_DEVICE_INLINE float atomicMax (device float * addr, float value) {
    float old;
    // *reinterpret_cast<float*>( &val )
    old = (value >= 0) ? __int_as_float(atomicMax((device int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((device unsigned int *)addr, __float_as_uint(value)));

    return old;
}

TV_DEVICE_INLINE float atomicMin (device float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((device int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((device unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#endif


template <typename T, size_t N> TV_DEVICE_INLINE array<T, N> atomicMax(TV_METAL_DEVICE array<T, N> *ctr, const TV_METAL_THREAD array<T, N>& val) {
  auto ptr_array = arrayops::create_ptr_arange<N>(reinterpret_cast<TV_METAL_DEVICE T*>(ctr));
  return arrayops::apply(atomicMax<float>, reinterpret_cast<TV_METAL_DEVICE tv::array<TV_METAL_DEVICE T*, N>&>(ptr_array), val);
}

template <typename T, size_t N> TV_DEVICE_INLINE array<T, N> atomicMin(TV_METAL_DEVICE array<T, N> *ctr, const TV_METAL_THREAD array<T, N>& val) {
  auto ptr_array = arrayops::create_ptr_arange<N>(reinterpret_cast<TV_METAL_DEVICE T*>(ctr));
  return arrayops::apply(atomicMin<float>, reinterpret_cast<TV_METAL_DEVICE tv::array<TV_METAL_DEVICE T*, N>&>(ptr_array), val);
}

#ifdef TV_METAL_RTC

template <typename T> TV_DEVICE_INLINE T atomicAggInc(TV_METAL_THREADGROUP T *ctr) {
  T warp_res;
  auto mask = vote_t(metal::simd_active_threads_mask());
  auto thread_rank = metal::popcount(mask & detail::lanemask_lt());
  if (thread_rank == 0) {
    warp_res = metal::atomic_fetch_add_explicit(
        reinterpret_cast<device metal::atomic<T>*>(ctr),
        T(metal::popcount(metal::simd_vote::vote_t(mask))), metal::memory_order_relaxed);
  }
  return metal::simd_shuffle(warp_res, metal::ctz(mask)) + thread_rank;
}
template <typename T> TV_DEVICE_INLINE T atomicAdd(device metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_add_explicit(ctr, val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicAdd(threadgroup T *ctr, T val) {
  return metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicAdd(threadgroup metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_add_explicit(ctr, val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicMax(device metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_max_explicit(ctr, val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicMax(threadgroup T *ctr, T val) {
  return metal::atomic_fetch_max_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicMax(threadgroup metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_max_explicit(ctr, val, metal::memory_order_relaxed);
}

#endif

template <typename T1, typename T2>
TV_DEVICE_INLINE DataPair<T1, T2> atomicArgMax(TV_METAL_DEVICE DataPair<T1, T2> *addr, T1 first,
                                         T2 second) {
  // for apple, this function only support 32bit kv.
  using atomic_ptr_t = typename detail::AtomicDataType<sizeof(T1)>::type;
  detail::DataPairUnion<T1, T2> ret =
      *(reinterpret_cast<TV_METAL_DEVICE detail::DataPairUnion<T1, T2> *>(addr));
  detail::DataPairUnion<T1, T2> expected;
  expected.data.first = first;
  expected.data.second = second;
  while (first > ret.data.first) {
    atomic_ptr_t old = ret.val;
#if defined(TV_METAL_RTC)
    atomic_ptr_t old_for_apple = ret.val;
    atomic_ptr_t cur;
    bool success;
    do {
      success = metal::atomic_compare_exchange_weak_explicit(
          (device metal::atomic<atomic_ptr_t> *)addr, &old_for_apple, expected.val, metal::memory_order_relaxed,
          metal::memory_order_relaxed);
      cur = old_for_apple;
      old_for_apple = old;
    } while (!success && old_for_apple == cur);
    ret.val = cur;
#else
    ret.val = atomicCAS((atomic_ptr_t *)addr, old, expected.val);
#endif
    if (ret.val == old)
      // insert success
      break;
  }
  return ret.data;
}

template <typename T1, typename T2>
TV_DEVICE_INLINE DataPair<T1, T2> atomicArgMin(TV_METAL_DEVICE DataPair<T1, T2> *addr, T1 first,
                                         T2 second) {
  using atomic_ptr_t = typename detail::AtomicDataType<sizeof(T1)>::type;
  detail::DataPairUnion<T1, T2> ret =
      *(reinterpret_cast<TV_METAL_DEVICE detail::DataPairUnion<T1, T2> *>(addr));
  detail::DataPairUnion<T1, T2> expected;
  expected.data.first = first;
  expected.data.second = second;
  while (first < ret.data.first) {
    atomic_ptr_t old = ret.val;
#if defined(TV_METAL_RTC)
    atomic_ptr_t old_for_apple = ret.val;
    atomic_ptr_t cur;
    bool success;
    do {
      success = metal::atomic_compare_exchange_weak_explicit(
          (device metal::atomic<atomic_ptr_t> *)addr, &old_for_apple, expected.val, metal::memory_order_relaxed,
          metal::memory_order_relaxed);
      cur = old_for_apple;
      old_for_apple = old;
    } while (!success && old_for_apple == cur);
#else
    ret.val = atomicCAS((atomic_ptr_t *)addr, old, expected.val);
#endif
    if (ret.val == old)
      break;
  }
  return ret.data;
}

template <typename T>
TV_DEVICE_INLINE static uint32_t popcount(T x) { 
#ifdef TV_METAL_RTC
  return metal::popcount(x);
#else 
  return __popc(x);
#endif
}


TV_DEVICE_INLINE vote_t ballot_sync(bool expr){
#if defined(TV_CUDA_CC)
  return __ballot_sync(0xffffffff, expr);
#elif defined(TV_METAL_RTC)
  return vote_t(metal::simd_ballot(expr));
#else
  static_assert(false, "not implemented");
#endif
}

#if defined(TV_CUDA_CC)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
TV_DEVICE_INLINE vote_t match_all_sync(int compare, TV_METAL_THREAD int* pred){
  return __match_all_sync(0xffffffff, compare, pred);
}
#endif 
#elif defined(TV_METAL_RTC)
TV_DEVICE_INLINE vote_t match_all_sync(int compare, TV_METAL_THREAD int* pred){
  int compare_in_first = metal::simd_broadcast_first(compare);
  bool is_same = compare == compare_in_first;
  auto mask = vote_t(metal::simd_ballot(is_same));
  pred[0] = mask == 0xffffffff;
  return mask;
}
#endif

template <typename T> TV_DEVICE_INLINE T warp_broadcast(T val, int src_lane) {
#if defined(TV_CUDA_CC)
  return __shfl_sync(0xffffffff, val, src_lane);
#elif defined(TV_METAL_RTC)
  return metal::simd_broadcast(val, src_lane);
#else 
  static_assert(false, "not implemented");
#endif
}

template <typename T> TV_DEVICE_INLINE T shfl_down(T val, unsigned int offset) {
#if defined(TV_CUDA_CC)
  return __shfl_down_sync(0xffffffff, val, offset);
#elif defined(TV_METAL_RTC)
  return metal::simd_shuffle_down(val, offset);
#else 
  static_assert(false, "not implemented");
#endif
}

template <typename T> TV_DEVICE_INLINE T shfl_up(T val, unsigned int offset) {
#if defined(TV_CUDA_CC)
  return __shfl_up_sync(0xffffffff, val, offset); 
#elif defined(TV_METAL_RTC)
  return metal::simd_shuffle_up(val, offset);
#else
  static_assert(false, "not implemented");
#endif
}

template <typename T> TV_DEVICE_INLINE T shfl_xor(T val, unsigned int offset) {
#if defined(TV_CUDA_CC)
  return __shfl_xor_sync(0xffffffff, val, offset);
#elif defined(TV_METAL_RTC)
  return metal::simd_shuffle_xor(val, offset);
#else
  static_assert(false, "not implemented");
#endif
}

template <typename T> TV_DEVICE_INLINE T shfl(T val, unsigned int src_lane) {
#if defined(TV_CUDA_CC)
  return __shfl_sync(0xffffffff, val, src_lane);
#elif defined(TV_METAL_RTC)
  return metal::simd_shuffle(val, src_lane);
#else
  static_assert(false, "not implemented");
#endif
}


template <typename T> TV_DEVICE_INLINE T warp_sum(T val) {
#if defined(TV_CUDA_CC)
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#elif defined(TV_METAL_RTC)
  return metal::simd_sum(val);
#else 
  static_assert(false, "not implemented");
#endif
}

template <typename T> TV_DEVICE_INLINE T warp_sum_with_offset(T val, uint32_t init_offset) {
#if defined(TV_CUDA_CC)
  #pragma unroll
  for (int offset = init_offset; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#elif defined(TV_METAL_RTC)
  #pragma unroll
  for (int offset = init_offset; offset > 0; offset /= 2) {
    val += metal::simd_shuffle_down(val, offset);
  }
  return val;
#else 
  static_assert(false, "not implemented");
#endif
}

#ifdef TV_CUDA_CC
template <> TV_DEVICE_INLINE uint32_t warp_sum(uint32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_add_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#endif
}

template <> TV_DEVICE_INLINE int32_t warp_sum(int32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_add_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#endif
}

#endif 

template <typename T> TV_DEVICE_INLINE T  warp_max(T val) {
#if defined(TV_CUDA_CC)
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#elif defined(TV_METAL_RTC)
  return metal::simd_max(val);
#else
  static_assert(false, "not implemented");  
#endif
}

template <typename T> TV_DEVICE_INLINE T  warp_max_with_offset(T val, uint32_t init_offset) {
#if defined(TV_CUDA_CC)
  #pragma unroll
  for (int offset = init_offset; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#elif defined(TV_METAL_RTC)
  #pragma unroll
  for (int offset = init_offset; offset > 0; offset /= 2) {
    val = max(val, metal::simd_shuffle_down(val, offset));
  }
  return val;
#else
  static_assert(false, "not implemented");  
#endif
}


#ifdef TV_CUDA_CC
template <> TV_DEVICE_INLINE uint32_t warp_max(uint32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_max_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#endif
}
template <> TV_DEVICE_INLINE int32_t warp_max(int32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_max_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#endif
}

#endif 




template <typename T> TV_DEVICE_INLINE T warp_min(T val) {
#if defined(TV_CUDA_CC)
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#elif defined(TV_METAL_RTC)
  return metal::simd_min(val);
#else
  static_assert(false, "not implemented");
#endif
}

#ifdef TV_CUDA_CC
template <> TV_DEVICE_INLINE uint32_t warp_min(uint32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_min_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#endif
}
template <> TV_DEVICE_INLINE int32_t warp_min(int32_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_min_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
#endif
}

#endif 


TV_DEVICE_INLINE unsigned warp_and(unsigned val) {
#if defined(TV_CUDA_CC)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_and_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val &= __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#endif
#elif defined(TV_METAL_RTC)
  return metal::simd_and(val);
#else
  static_assert(false, "not implemented");
#endif
}

TV_DEVICE_INLINE unsigned warp_or(unsigned val) {
#if defined(TV_CUDA_CC)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_or_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val |= __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#endif
#elif defined(TV_METAL_RTC)
  return metal::simd_or(val);
#else
  static_assert(false, "not implemented");
#endif
}

TV_DEVICE_INLINE unsigned warp_xor(unsigned val) {
#if defined(TV_CUDA_CC)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_xor_sync(0xffffffff, val);
#else
  #pragma unroll
  for (int offset = warp_size() / 2; offset > 0; offset /= 2) {
    val ^= __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
#endif
#elif defined(TV_METAL_RTC)
  return metal::simd_xor(val);
#else
  static_assert(false, "not implemented");
#endif
}

TV_DEVICE_INLINE bool warp_all(bool val) {
#if defined(TV_CUDA_CC)
  return __all_sync(0xffffffff, val);
#elif defined(TV_METAL_RTC)
  return metal::simd_all(val);
#else
  static_assert(false, "not implemented");
#endif
}

TV_DEVICE_INLINE bool warp_any(bool val) {
#if defined(TV_CUDA_CC)
  return __any_sync(0xffffffff, val);
#elif defined(TV_METAL_RTC)
  return metal::simd_any(val);
#else
  static_assert(false, "not implemented");
#endif
}

TV_DEVICE_INLINE vote_t active_mask() {
#if defined(TV_CUDA_CC)
  return __activemask();
#elif defined(TV_METAL_RTC)
  return vote_t(metal::simd_active_threads_mask());
#else
  static_assert(false, "not implemented");
#endif
}

TV_DEVICE_INLINE vote_t warp_ballot(bool val) {
#if defined(TV_CUDA_CC)
  return __ballot_sync(0xffffffff, val);
#elif defined(TV_METAL_RTC)
  return vote_t(metal::simd_ballot(val));
#else
  static_assert(false, "not implemented");
#endif
}

template <size_t BlockSize, class T, size_t WarpSize = 32, class F>
TV_DEVICE_INLINE T block_reduce_full(TV_METAL_THREAD F&& f, T val, TV_METAL_THREADGROUP T* shared_mem_ptr){
  static_assert(BlockSize % WarpSize == 0 && BlockSize > WarpSize, "BlockSize must be multiple of WarpSize");
  static_assert(BlockSize / WarpSize <= WarpSize, "num warp must be less than warp size");
  auto warp_res = std::forward<F>(f)(val, WarpSize / 2);
  block_sync_shared_io();
  if (lane_index() == 0) {
    shared_mem_ptr[warp_index()] = warp_res;
  }
  block_sync_shared_io();
  auto lane_idx = lane_index();
  auto smem_data = lane_idx < (BlockSize / WarpSize) ? shared_mem_ptr[lane_idx] : T{};
  return warp_index() == 0 ? std::forward<F>(f)(smem_data, (BlockSize / WarpSize / 2)) : T{};
}

template <size_t BlockSize, class T, size_t N, size_t WarpSize = 32, class F>
TV_DEVICE_INLINE array<T, N> block_reduce_array_full(TV_METAL_THREAD F&& f, const TV_METAL_THREAD array<T, N>& val, TV_METAL_THREADGROUP T* shared_mem_ptr){
  static_assert(BlockSize % WarpSize == 0 && BlockSize > WarpSize, "BlockSize must be multiple of WarpSize");
  static_assert(BlockSize / WarpSize <= WarpSize, "num warp must be less than warp size");
  auto warp_res = apply(std::forward<F>(f), val, WarpSize / 2);
  block_sync_shared_io();
  if (lane_index() == 0) {
    reinterpret_cast<TV_METAL_THREADGROUP array<T, N>*>(shared_mem_ptr)[warp_index()] = warp_res;
  }
  block_sync_shared_io();
  auto lane_idx = lane_index();
  auto smem_data = lane_idx < (BlockSize / WarpSize) ? reinterpret_cast<TV_METAL_THREADGROUP array<T, N>*>(shared_mem_ptr)[lane_idx] : array<T, N>{};
  return warp_index() == 0 ? apply(std::forward<F>(f), smem_data, (BlockSize / WarpSize / 2)) : array<T, N>{};
}

template <size_t BlockSize, class T, size_t WarpSize = 32>
TV_DEVICE_INLINE T block_reduce_sum_full(T val, TV_METAL_THREADGROUP T* shared_mem_ptr){
  return block_reduce_full<BlockSize, T, WarpSize>(warp_sum_with_offset<T>, val, shared_mem_ptr);
}

template <size_t BlockSize, class T, size_t N, size_t WarpSize = 32>
TV_DEVICE_INLINE array<T, N> block_reduce_array_sum_full(const TV_METAL_THREAD array<T, N>& val, TV_METAL_THREADGROUP T* shared_mem_ptr){
  return block_reduce_array_full<BlockSize, T, WarpSize>(warp_sum_with_offset<T>, val, shared_mem_ptr);
}

template <size_t BlockSize, class T, size_t WarpSize = 32>
TV_DEVICE_INLINE T block_reduce_max_full(T val, TV_METAL_THREADGROUP T* shared_mem_ptr){
  return block_reduce_full<BlockSize, T, WarpSize>(warp_max_with_offset<T>, val, shared_mem_ptr);
}

template <size_t BlockSize, class T, size_t N, size_t WarpSize = 32>
TV_DEVICE_INLINE array<T, N> block_reduce_array_max_full(const TV_METAL_THREAD array<T, N>& val, TV_METAL_THREADGROUP T* shared_mem_ptr){
  return block_reduce_array_full<BlockSize, T, WarpSize>(warp_max_with_offset<T>, val, shared_mem_ptr);
}
#endif
} // namespace parallel

} // namespace tv