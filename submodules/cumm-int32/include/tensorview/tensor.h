// Copyright 2019-2021 Yan Yan
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
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "context.h"
#include "core/cc17.h"
#include "dtypes.h"
#include "mp_helper.h"
#include "tensorview.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <type_traits>
#if defined(TV_HARDWARE_ACC_CUDA)
#include "cuda/driverops.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#endif
#if (CUDA_VERSION >= 11000 && defined(TV_HARDWARE_ACC_CUDA))
#include <cuda_bf16.h>
#endif
#if (CUDA_VERSION >= 11080 && defined(TV_HARDWARE_ACC_CUDA))
#include <cuda_fp8.h>
#endif

#ifdef TV_HARDWARE_ACC_METAL
#include "Metal/Metal.hpp"
#endif

#include <random>

namespace tv {

enum DeviceType {
  kDeviceCPU = -1,
  kDeviceCUDA = 0,
};

class Tensor;

namespace detail {

using dtype_collection_t =
    mp_list_c<int, float32, int32, int16, int8, float64, bool_, uint8, float16,
              int64, uint16, uint32, uint64>;

#if defined(TV_HARDWARE_ACC_CUDA) && CUDA_VERSION < 11000
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool, __half>;
#elif defined(TV_HARDWARE_ACC_CUDA) && CUDA_VERSION >= 11000
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool, __half, __nv_bfloat16>;
#else
#ifdef TV_HARDWARE_ACC_METAL
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;
#else
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;

#endif
#endif

using all_tensor_types_print_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;

using all_int_tensor_types_t =
    std::tuple<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
               uint64_t>;

template <typename T, typename U>
struct is_convertible : std::is_convertible<T, U> {};

#if defined(TV_HARDWARE_ACC_CUDA)
template <typename T>
struct is_convertible<T, __half> : std::is_floating_point<T> {};
#endif

#if (CUDA_VERSION >= 11000 && defined(TV_HARDWARE_ACC_CUDA))
template <typename T>
struct is_convertible<T, __nv_bfloat16> : std::is_floating_point<T> {};
#endif

template <typename T> class TensorStorage {
  // friend Tensor;
public:
  TensorStorage(size_t size, int device = -1, bool managed = false,
                bool pinned = false)
      : size_(size), device_(device), managed_(managed), pinned_(pinned) {
    if (size == 0) {
      ptr_ = nullptr;
    } else {
      if (device == -1) {
        if (pinned || managed) {
          TV_ASSERT_INVALID_ARG(
              pinned ^ managed,
              "cpu tensor can't be managed (shared in apple silicon)");
        }
        if (pinned_) {
#if defined(TV_HARDWARE_ACC_CUDA)
          checkCudaErrors(cudaMallocHost(&ptr_, size * sizeof(T)));
#else
          TV_THROW_INVALID_ARG("you need to define TV_ENABLE_HARDWARE_ACC to "
                               "use pinned, only cuda support this.");
#endif
        } else if (managed) {
#if defined(TV_HARDWARE_ACC_CUDA)
          checkCudaErrors(cudaMallocManaged(&this->ptr_, size * sizeof(T)));
#elif defined(TV_HARDWARE_ACC_METAL)
          ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
          TV_ASSERT_RT_ERR(ptr_mtl_device_, "Metal device not found");
          ptr_mtl_ =
              ptr_mtl_device_->newBuffer(size, MTL::ResourceStorageModeShared);
          TV_ASSERT_RT_ERR(ptr_mtl_, "Metal buffer not created");
          ptr_ = reinterpret_cast<T *>(ptr_mtl_->contents());
#else
          TV_THROW_INVALID_ARG("don't compiled with cuda or metal");
#endif
        } else {
          ptr_ = new T[size];
        }
      } else {
        TV_ASSERT_INVALID_ARG(!managed && !pinned,
                              "only cpu tensor can be managed or pinned");
#if defined(TV_HARDWARE_ACC_CUDA)
        checkCudaErrors(cudaMalloc(&ptr_, size * sizeof(T)));
#elif defined(TV_HARDWARE_ACC_METAL)
        ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
        TV_ASSERT_RT_ERR(ptr_mtl_device_, "Metal device not found");
        ptr_mtl_ =
            ptr_mtl_device_->newBuffer(size, MTL::ResourceStorageModePrivate);
        TV_ASSERT_RT_ERR(ptr_mtl_, "Metal buffer not created");
        ptr_ = nullptr;
        TV_ASSERT_INVALID_ARG(is_private(), "should be private buffer");
#else
        TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
      }
    }
  }
  TensorStorage(T *ptr, size_t size, int device, int storage_offset = 0,
                bool managed = false)
      : size_(size), ptr_(ptr), from_blob_(true), device_(device),
        managed_(managed) {
    if (size == 0) {
      ptr_ = nullptr;
    }
#ifdef TV_HARDWARE_ACC_METAL
    if (device == 0) {
      if (managed) {
        // pytorch don't have managed, so we assume the ptr come from
        // tensorview, so it's data ptr, not buffer ptr.
        ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
        TV_ASSERT_RT_ERR(ptr_mtl_device_, "Metal device not found");

        int alignedLength = 0;
        void *host_dst = ptr;
        void *alignedPtr = detail::pageAlignedBlockPtr(
            host_dst, int(size * sizeof(T)), &alignedLength);
        int destOffset = (uintptr_t(host_dst) - uintptr_t(alignedPtr));
        // 4 bytes alignment required on macos for blits.
        TV_ASSERT_RT_ERR(destOffset % 4 == 0, "Unaligned blit request");
        TV_ASSERT_RT_ERR(destOffset % sizeof(T) == 0, "should not happen");
        ptr_ = reinterpret_cast<T *>(alignedPtr);
        ptr_mtl_ = ptr_mtl_device_->newBuffer(
            alignedPtr, alignedLength, MTL::ResourceStorageModeShared, nullptr);
        TV_ASSERT_RT_ERR(ptr_mtl_, "Metal buffer not created", PAGE_SIZE);
        offset_ = destOffset / sizeof(T);
      } else {
        // assume the ptr is come from torch mps
        // then the ptr is Buffer ptr, not data ptr
        // pytorch mps tensor use private buffer, so we can't
        // get the pointer, pytorch can't get the pointer too.

        // https://stackoverflow.com/questions/76989609/how-to-convert-a-idmtlcommandqueue-to-a-mtlcommandqueue-of-metal-cpp
        MTL::Buffer *buffer = reinterpret_cast<MTL::Buffer *>(ptr);
        ptr_mtl_ = buffer;
        offset_ = storage_offset;
        return;
      }
    }
#endif
  }

  virtual ~TensorStorage() {
    if (empty()) {
      return;
    }
    if (from_blob_) {
      return;
    }
    if (device_ == -1) {
      if (pinned_) {
#if defined(TV_HARDWARE_ACC_CUDA)
        cudaFreeHost(ptr_);
#endif
      } else if (managed_) {
#if defined(TV_HARDWARE_ACC_CUDA)
        cudaFree(ptr_);
#elif defined(TV_HARDWARE_ACC_METAL)
        if (ptr_mtl_ != nullptr) {
          ptr_mtl_->release();
          ptr_mtl_ = nullptr;
        }
        if (ptr_mtl_device_ != nullptr) {
          ptr_mtl_device_->release();
          ptr_mtl_device_ = nullptr;
        }
        ptr_ = nullptr;
#endif
      } else {
        delete[] ptr_;
      }
    } else {
#if defined(TV_HARDWARE_ACC_CUDA)
      cudaFree(ptr_);
#endif
#ifdef TV_HARDWARE_ACC_METAL
      if (ptr_mtl_ != nullptr) {
        ptr_mtl_->release();
        ptr_mtl_ = nullptr;
      }
      if (ptr_mtl_device_ != nullptr) {
        ptr_mtl_device_->release();
        ptr_mtl_device_ = nullptr;
      }
#endif
    }
  };

  inline size_t size() const { return size_; }
  inline size_t byte_size() const { return size_ * sizeof(T); }

#ifdef TV_HARDWARE_ACC_METAL
  const auto *apple_metal_buffer_ptr() const { return ptr_mtl_; }
  auto *apple_metal_buffer_ptr() { return ptr_mtl_; }
#endif
  T *data() {
#ifdef TV_HARDWARE_ACC_METAL
    TV_ASSERT_INVALID_ARG(
        !is_private(),
        "you can't access pointer of private buffer, currently only apple "
        "buffer can be private. only gpu kernels can access private data.");
#endif
    return ptr_ + offset_;
  }
  const T *data() const {
#ifdef TV_HARDWARE_ACC_METAL
    TV_ASSERT_INVALID_ARG(
        !is_private(),
        "you can't access pointer of private buffer, currently only apple "
        "buffer can be private. only gpu kernels can access private data.");
#endif
    return ptr_ + offset_;
  }
  bool is_cpu() const { return device_ == -1; }

  bool empty() const {
#ifdef TV_HARDWARE_ACC_METAL
    if (!is_cpu()) {
      return ptr_mtl_ == nullptr || size_ == 0;
    } else {
      if (managed_) {
        return ptr_mtl_ == nullptr || size_ == 0;
      } else {
        return ptr_ == nullptr || size_ == 0;
      }
    }
#else
    return ptr_ == nullptr || size_ == 0;
#endif
  }
  bool managed() const { return managed_; }
  bool pinned() const { return pinned_; }
  size_t offset() const { return offset_; }
  size_t byte_offset() const { return offset_ * sizeof(T); }
  bool is_private() const {
#ifdef TV_HARDWARE_ACC_METAL
    return ptr_ == nullptr && ptr_mtl_ != nullptr;
#else
    return false;
#endif
  }
  std::uintptr_t gpu_address() const {
    TV_ASSERT_INVALID_ARG(!is_cpu(), "only support gpu tensor");
#if defined(TV_HARDWARE_ACC_CUDA)
    std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr_);
    return address;
#elif defined(TV_HARDWARE_ACC_METAL)
    TV_ASSERT_INVALID_ARG(ptr_mtl_ != nullptr, "invalid metal buffer");
    return std::uintptr_t(ptr_mtl_->gpuAddress());
#else
    TV_THROW_RT_ERR("only support gpu tensor");
#endif
  }
  int device() const { return device_; }
  void zero_(int64_t offset, int64_t length, Context ctx = Context()) {
    if (size_ == 0) {
      TV_ASSERT_INVALID_ARG(
          offset == 0 && length == 0,
          "when you zero a empty tensor, offset and length should be 0");
      return;
    }
    TV_ASSERT_RT_ERR(length <= size_ - offset, "eror");
    int target_device = device_;
#ifdef TV_HARDWARE_ACC_METAL
    target_device = 0; // for apple m cpu, all buffer are shared.
#endif
    if (target_device == -1) {
      std::memset(data() + offset, 0, length);
      // std::fill(data(), data() + size_, 0);
    } else {
#if defined(TV_HARDWARE_ACC_CUDA)
      if (ctx.has_cuda_stream()) {
        checkCudaErrors(cudaMemsetAsync(data() + offset * sizeof(T), 0,
                                        length * sizeof(T), ctx.cuda_stream()));
      } else {
        checkCudaErrors(
            cudaMemset(data() + offset * sizeof(T), 0, length * sizeof(T)));
      }
#elif defined(TV_HARDWARE_ACC_METAL)
      AppleMetalContext::SyncType sync_type =
          AppleMetalContext::SyncType::COMMIT;
      if (!ctx.has_item(ContextType::kAppleMetal)) {
        ctx = Context().create_apple_metal_context();
        sync_type = AppleMetalContext::SyncType::COMMIT_AND_WAIT;
      }
      auto *metal_ctx = reinterpret_cast<AppleMetalContext *>(
          ctx.get_item(ContextType::kAppleMetal));
      if (ptr_mtl_ == nullptr) {
        TV_ASSERT_RT_ERR(ptr_ != nullptr, "invalid ptr, shouldn't happen");
        metal_ctx->fill_raw(ptr_, byte_offset() + offset * sizeof(T), length, 0,
                            sync_type);
      } else {
        metal_ctx->fill(ptr_mtl_, byte_offset() + offset * sizeof(T), length, 0,
                        sync_type);
      }
#else
      TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
    }
  }

  void zero_whole_(Context ctx = Context()) { return zero_(0, size_, ctx); }
  void copy_(const TensorStorage<T> &src, Context ctx = Context()) {
    TV_ASSERT_INVALID_ARG(size_ == src.size_, "size not match");
    if (size_ == 0) {
      return;
    }
    copy(src, 0, *this, 0, size_, ctx);
  }
  static void copy(const TensorStorage<T> &src, size_t src_offset,
                   TensorStorage<T> &dst, size_t dst_offset, size_t copy_size,
                   Context ctx = Context()) {
    if (copy_size == 0) {
      return;
    }
    TV_ASSERT_INVALID_ARG(src.size_ >= src_offset + copy_size,
                          "src size not match");
    TV_ASSERT_INVALID_ARG(dst.size_ >= dst_offset + copy_size,
                          "dst size not match");
#if defined(TV_HARDWARE_ACC_CUDA)
    if (src.device_ == -1 && dst.device_ == -1) {
      std::copy(src.data() + src_offset, src.data() + src_offset + copy_size,
                dst.data() + dst_offset);
    } else if (src.device_ == -1 && dst.device_ != -1) {
      if (ctx.has_cuda_stream()) {
        host2dev(dst.data() + dst_offset, src.data() + src_offset,
                 copy_size, ctx.cuda_stream());
      } else {
        host2dev(dst.data() + dst_offset, src.data() + src_offset,
                 copy_size);
      }
    } else if (src.device_ != -1 && dst.device_ == -1) {
      if (ctx.has_cuda_stream()) {
        dev2host(dst.data() + dst_offset, src.data() + src_offset,
                 copy_size, ctx.cuda_stream());
      } else {
        dev2host(dst.data() + dst_offset, src.data() + src_offset,
                 copy_size);
      }
    } else {
      if (ctx.has_cuda_stream()) {
        dev2dev(dst.data() + dst_offset, src.data() + src_offset,
                copy_size, ctx.cuda_stream());
      } else {
        dev2dev(dst.data() + dst_offset, src.data() + src_offset,
                copy_size);
      }
    }
#elif defined(TV_HARDWARE_ACC_METAL)
    size_t copy_size_in_bytes = copy_size * sizeof(T);
    AppleMetalContext::SyncType sync_type = AppleMetalContext::SyncType::COMMIT;
    if (!ctx.has_item(ContextType::kAppleMetal)) {
      ctx = Context().create_apple_metal_context();
      sync_type = AppleMetalContext::SyncType::COMMIT_AND_WAIT;
    }
    auto *metal_ctx = reinterpret_cast<AppleMetalContext *>(
        ctx.get_item(ContextType::kAppleMetal));
    if (src.ptr_mtl_ == nullptr && dst.ptr_mtl_ == nullptr) {
      // always cpu to cpu, just use std copy
      std::copy(src.data() + src_offset, src.data() + src_offset + copy_size,
                dst.data() + dst_offset);
    } else if (src.ptr_mtl_ == nullptr && dst.ptr_mtl_ != nullptr) {
      auto ptr = src.ptr_;
      TV_ASSERT_RT_ERR(ptr != nullptr, "invalid ptr, shouldn't happen");
      metal_ctx->copy_src_raw(ptr, src.byte_size(), dst.ptr_mtl_,
                              src.byte_offset() + src_offset * sizeof(T),
                              dst.byte_offset() + dst_offset * sizeof(T),
                              copy_size_in_bytes, sync_type);
    } else if (src.ptr_mtl_ != nullptr && dst.ptr_mtl_ == nullptr) {
      // when copy to a cpu tensor, we always do sync.
      auto ptr = dst.ptr_;
      TV_ASSERT_RT_ERR(ptr != nullptr, "invalid ptr, shouldn't happen");
      metal_ctx->copy_dst_raw(src.ptr_mtl_, ptr, dst.byte_size(),
                              src.byte_offset() + src_offset * sizeof(T),
                              dst.byte_offset() + dst_offset * sizeof(T),
                              copy_size_in_bytes,
                              AppleMetalContext::SyncType::COMMIT_AND_WAIT);
    } else {
      metal_ctx->copy(src.ptr_mtl_, dst.ptr_mtl_,
                      src.byte_offset() + src_offset * sizeof(T),
                      dst.byte_offset() + dst_offset * sizeof(T),
                      copy_size_in_bytes, sync_type);
    }
#else
    if (src.device_ == -1 && dst.device_ == -1) {
      std::copy(src.data() + src_offset, src.data() + src_offset + copy_size,
                dst.data() + dst_offset);
    } else {
      TV_THROW_RT_ERR("only support cpu tensor");
    }
#endif
  }
  std::shared_ptr<TensorStorage<T>> clone(Context ctx = Context()) {
    // clone whole storage
    auto new_storage_ptr =
        std::make_shared<TensorStorage<T>>(size_, device_, managed_, pinned_);
    if (size_ == 0) {
      return new_storage_ptr;
    }
    new_storage_ptr->copy_(*this, ctx);
    return new_storage_ptr;
  }

  std::shared_ptr<TensorStorage<T>> cpu(Context ctx = Context()) {
    // clone whole storage
    auto new_storage_ptr =
        std::make_shared<TensorStorage<T>>(size_, -1, managed_, pinned_);
    if (size_ == 0) {
      return new_storage_ptr;
    }
    new_storage_ptr->copy_(*this, ctx);
    return new_storage_ptr;
  }

private:
  size_t size_ = 0;
  T *ptr_ = nullptr;
  bool from_blob_ = false;
  int device_ = -1;
  bool managed_ = false;
  bool pinned_ = false;
  size_t offset_ = 0;

#ifdef TV_HARDWARE_ACC_METAL
  MTL::Buffer *ptr_mtl_ = nullptr;
  MTL::Device *ptr_mtl_device_ = nullptr;
#endif
};

template <typename T> size_t sizeof_dtype(T dtype) {
  switch (dtype) {
  case float32:
    return sizeof(float);
  case int8:
    return sizeof(int8_t);
  case int16:
    return sizeof(int16_t);
  case int32:
    return sizeof(int32_t);
  case float64:
    return sizeof(double);
  case int64:
    return sizeof(int64_t);
  case bool_:
    return sizeof(bool);
  case uint8:
    return sizeof(uint8_t);
  case uint16:
    return sizeof(uint16_t);
  case uint32:
    return sizeof(uint32_t);
  case uint64:
    return sizeof(uint64_t);
  case float16:
    return 2;
  case bfloat16:
    return 2;
  case tf32:
    return sizeof(float);
  case float_e4m3:
    return 1;
  case float_e5m2:
    return 1;
  case custom16:
    return 2;
  case custom32:
    return 4;
  case custom48:
    return 6;
  case custom64:
    return 8;
  case custom80:
    return 10;
  case custom96:
    return 12;
  case custom128:
    return 16;
  default:
    TV_THROW_RT_ERR("unsupported dtype", dtype);
  }
  return 0;
}

template <class Tsrc, class Tdst> struct ConvertTmpType {
  using type = Tdst;
  static constexpr bool kSpec = false;
};

#if (CUDA_VERSION >= 11000 && defined(TV_HARDWARE_ACC_CUDA))
template <class T> struct ConvertTmpType<T, __nv_bfloat16> {
  using type = float;
  static constexpr bool kSpec = true;
};
#endif
} // namespace detail
template <class... Ts, typename F> bool dispatch_noexcept(DType t, F &&f) {
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    using T = TV_DECLTYPE(I);
    auto type = detail::TypeToDtype<std::decay_t<T>>::dtype;
    if (type == t && notFound) {
      std::forward<F>(f)(TV_DECLTYPE(I)());
      notFound = false;
    }
  });
  return !notFound;
}

template <class... Ts, typename F> void dispatch(DType t, F &&f) {
  if (!dispatch_noexcept<Ts...>(t, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << type_s<std::decay_t<TV_DECLTYPE(I)>> << " ";
    });
    TV_THROW_RT_ERR("unknown type", dtype_str(t), ", available:", ss.str());
  }
}

template <typename T, T... Is, typename F> void dispatch_scalar(T idx, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<T, Is...>>([=, &notFound, &f](auto I) {
    if (T(I) == idx && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  if (notFound) {
    std::stringstream ss;
    mp_for_each<mp_list_c<T, Is...>>([=, &ss](auto I) { ss << T(I) << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

template <int... Is, typename F> bool dispatch_int_noexcept(int idx, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<int, Is...>>([=, &notFound, &f](auto I) {
    if (TV_DECLTYPE(I)::value == idx && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  return !notFound;
}

template <int... Is, typename F, class BinaryPredicate>
bool dispatch_int_noexcept(int idx, BinaryPredicate p, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<int, Is...>>([=, &notFound, &f](auto I) {
    if (p(idx, TV_DECLTYPE(I)::value) && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  return !notFound;
}

template <int... Is, typename F> void dispatch_int(int idx, F &&f) {
  if (!dispatch_int_noexcept<Is...>(idx, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list_c<int, Is...>>(
        [=, &ss](auto I) { ss << TV_DECLTYPE(I)::value << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

template <int... Is, typename F, class BinaryPredicate>
void dispatch_int(int idx, BinaryPredicate p, F &&f) {
  // BinaryPredicate: BinaryPredicate(idx, candidate)
  if (!dispatch_int_noexcept<Is...>(idx, p, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list_c<int, Is...>>(
        [=, &ss](auto I) { ss << TV_DECLTYPE(I)::value << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

// Ts is pack of mp_list_c
template <class... Ts, typename Iterator, typename F>
bool dispatch_container_noexcept(Iterator begin, Iterator end, F &&f) {
  static_assert(sizeof...(Ts) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    using val_lst_t = TV_DECLTYPE(I);
    auto val_lst_size = mp_size<val_lst_t>::value;
    bool equal = true;
    std::size_t count = 0;
    auto iter = begin;
    mp_for_each<val_lst_t>([&](auto E) {
      if (iter == end || !equal) {
        return;
      }
      if (count >= val_lst_size) {
        equal = false;
        return;
      }
      constexpr auto c = TV_DECLTYPE(E)::value;
      if (c != *iter) {
        equal = false;
      }
      ++count;
      std::advance(iter, 1);
    });
    if (count != val_lst_size || iter != end) {
      equal = false;
    }
    if (equal && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });

  return !notFound;
}

template <class... Ts, typename Iterator, typename F>
void dispatch_container(Iterator begin, Iterator end, F &&f) {
  if (!dispatch_container_noexcept<Ts...>(begin, end, std::forward<F>(f))) {
    std::stringstream ss;
    ss << "unknown value [";
    for (auto iter = begin; iter != end; std::advance(iter, 1)) {
      ss << *iter << ",";
    }
    ss << "], available: ";
    mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << "[";
      mp_for_each<TV_DECLTYPE(I)>(
          [=, &ss](auto E) { ss << TV_DECLTYPE(E)::value << ","; });
      ss << "]";
    });
    TV_THROW_RT_ERR(ss.str());
  }
}

/*
template <int... Is, typename F> void dispatch_int(int idx, F &&f) {
  return dispatch_scalar<int, Is...>(idx, f);
}
*/

template <class T> struct Dispatch;

template <template <class...> class T, class... Args>
struct Dispatch<T<Args...>> {
  template <typename F> inline void operator()(DType t, F &&f) {
    return dispatch<Args...>(t, std::forward<F>(f));
  }
};

template <class T> struct DispatchNoExcept;

template <template <class...> class T, class... Args>
struct DispatchNoExcept<T<Args...>> {
  template <typename F> inline bool operator()(DType t, F &&f) {
    return dispatch_noexcept<Args...>(t, std::forward<F>(f));
  }
};

template <class T> struct DispatchContainer;

template <template <class...> class T, class... Args>
struct DispatchContainer<T<Args...>> {
  template <typename Iterator, typename F>
  inline void operator()(Iterator begin, Iterator end, F &&f) {
    return dispatch_container<Args...>(begin, end, std::forward<F>(f));
  }
};

template <class T> struct DispatchContainerNoexcept;

template <template <class...> class T, class... Args>
struct DispatchContainerNoexcept<T<Args...>> {
  template <typename Iterator, typename F>
  inline bool operator()(Iterator begin, Iterator end, F &&f) {
    return dispatch_container_noexcept<Args...>(begin, end, std::forward<F>(f));
  }
};

template <class T> struct DispatchInt;

// Args should be std::integral_constant<int, value>
// you need to use type_container<std::integral_constant<int, value>...>
// as template parameter of DispatchInt.
// tv::mp_list_c is ok.
template <template <class...> class T, class... Args>
struct DispatchInt<T<Args...>> {
  template <typename F> inline void operator()(int t, F &&f) {
    return dispatch_int<Args::value...>(t, std::forward<F>(f));
  }
  template <typename F, typename BinaryPredicate>
  inline void operator()(int t, BinaryPredicate p, F &&f) {
    return dispatch_int<Args::value...>(t, p, std::forward<F>(f));
  }
};

template <class T> struct DispatchIntNoexcept;

template <template <class...> class T, class... Args>
struct DispatchIntNoexcept<T<Args...>> {
  template <typename F> inline bool operator()(int t, F &&f) {
    return dispatch_int_noexcept<Args::value...>(t, std::forward<F>(f));
  }
  template <typename F, typename BinaryPredicate>
  inline bool operator()(int t, BinaryPredicate p, F &&f) {
    return dispatch_int_noexcept<Args::value...>(t, p, std::forward<F>(f));
  }
};

constexpr size_t kTensorMaxDim = 10;
using TensorShape = ShapeBase<kTensorMaxDim, int64_t>;

struct Tensor {
  // empty tensor will have float32 dtype by default.
  Tensor() : dtype_(tv::float32) {}
  Tensor(TensorShape shape, TensorShape stride, DType dtype, int device = -1,
         bool pinned = false, bool managed = false)
      : dtype_(dtype) {

    // TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed, pinned);
    shape_ = shape;
    stride_ = stride;
    contiguous_ = compute_is_contiguous();
    TV_ASSERT_RT_ERR(
        contiguous_,
        "stride must be contiguous when you create tensor from shape");
  }

  Tensor(TensorShape shape, DType dtype, int device = -1, bool pinned = false,
         bool managed = false)
      : dtype_(dtype) {
    // TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed, pinned);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
    contiguous_ = compute_is_contiguous();
  }
  Tensor(void *ptr, TensorShape shape, TensorShape stride, DType dtype,
         int device = -1, int storage_offset = 0)
      : dtype_(dtype) {
    // TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    auto itemsize = detail::sizeof_dtype(dtype);
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr), shape.size() * itemsize, device,
        storage_offset * itemsize);
    shape_ = shape;
    stride_ = stride;
    contiguous_ = compute_is_contiguous();
  }
  Tensor(void *ptr, TensorShape shape, DType dtype, int device = -1,
         int storage_offset = 0)
      : dtype_(dtype) {
    // TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    auto itemsize = detail::sizeof_dtype(dtype);
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr), shape.size() * itemsize, device,
        storage_offset * itemsize);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
    contiguous_ = compute_is_contiguous();
  }

  Tensor(const void *ptr, TensorShape shape, TensorShape stride, DType dtype,
         int device = -1, int storage_offset = 0)
      : dtype_(dtype), writeable_(false) {
    // TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    auto itemsize = detail::sizeof_dtype(dtype);
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(ptr)),
        shape.size() * itemsize, device, storage_offset * itemsize);
    shape_ = shape;
    stride_ = stride;
    contiguous_ = compute_is_contiguous();
  }
  Tensor(const void *ptr, TensorShape shape, DType dtype, int device = -1,
         int storage_offset = 0)
      : dtype_(dtype), writeable_(false) {
    // TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    auto itemsize = detail::sizeof_dtype(dtype);
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(ptr)),
        shape.size() * itemsize, device, storage_offset * itemsize);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
    contiguous_ = compute_is_contiguous();
  }

  Tensor(std::initializer_list<int32_t> init)
      : Tensor({int(init.size())}, int32) {
    std::copy(init.begin(), init.end(), data<int32_t>());
  }
  Tensor(std::initializer_list<int64_t> init)
      : Tensor({int(init.size())}, int64) {
    std::copy(init.begin(), init.end(), data<int64_t>());
  }
  Tensor(std::initializer_list<float> init)
      : Tensor({int(init.size())}, float32) {
    std::copy(init.begin(), init.end(), data<float>());
  }
  Tensor(std::initializer_list<double> init)
      : Tensor({int(init.size())}, float64) {
    std::copy(init.begin(), init.end(), data<double>());
  }

  template <typename Iterator, typename = detail::_RequireInputIter<Iterator>>
  Tensor(Iterator first, Iterator second, bool pinned = false) {
    using T = typename std::iterator_traits<Iterator>::value_type;
    dtype_ = type_v<T>;
    shape_ = {second - first};
    stride_ = shape_.stride_rowmajor();
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape_.size() * detail::sizeof_dtype(dtype_), -1, false, pinned);
    writeable_ = true;
    std::copy(first, second, data_ptr<T>());
  }

  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = TV_GLOBAL_INDEX, bool DoTypeCheck = true>
  decltype(auto) tview() const {
    static_assert(Rank == -1 || Rank > 0, "error");
    if (DoTypeCheck) {
      template_dtype_check<T>();
    }
    return if_constexpr<(Rank > 0)>(
        [&](auto _) {
          // detail::_if_constexpr_workaround<Rank, (Rank > 0)> _val;
          // TV_ASSERT_RT_ERR(_(_val).value == ndim(), "error");
          ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex> shape(_(Rank)),
              stride(_(Rank));
          for (int i = 0; i < Rank; ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<const std::remove_const_t<T>, Rank, PtrTraits,
                            Tindex>(
              reinterpret_cast<const std::remove_const_t<T> *>(
                  this->data<T, DoTypeCheck>()),
              _(shape), _(stride));
        },
        [&](auto _) {
          ShapeBase<TV_MAX_DIM, Tindex> shape(this->ndim()),
              stride(this->ndim());
          for (int i = 0; i < int(this->ndim()); ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<const std::remove_const_t<T>, Rank, PtrTraits,
                            Tindex>(
              reinterpret_cast<const std::remove_const_t<T> *>(
                  this->data<T, DoTypeCheck>()),
              _(shape), _(stride));
        });
  }

  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = TV_GLOBAL_INDEX, bool DoTypeCheck = true>
  decltype(auto) tview() {
    static_assert(Rank == -1 || Rank > 0, "error");
    if (DoTypeCheck) {
      template_dtype_check<T>();
    }
    return if_constexpr<(Rank > 0)>(
        [&](auto _) {
          // detail::_if_constexpr_workaround<Rank, (Rank > 0)> _val;
          // TV_ASSERT_RT_ERR(_(_val).value == ndim(), "error");
          ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex> shape(_(Rank)),
              stride(_(Rank));
          for (int i = 0; i < Rank; ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<T, Rank, PtrTraits, Tindex>(
              reinterpret_cast<T *>(this->data<T, DoTypeCheck>()), _(shape),
              _(stride));
        },
        [&](auto _) {
          ShapeBase<TV_MAX_DIM, Tindex> shape(this->ndim()),
              stride(this->ndim());
          for (int i = 0; i < int(this->ndim()); ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<T, Rank, PtrTraits, Tindex>(
              reinterpret_cast<T *>(this->data<T, DoTypeCheck>()), _(shape),
              _(stride));
        });
  }

  Tensor view(TensorShape shape) const {
    TensorShape newshape = shape;
    bool found_minus_1 = false;
    for (size_t i = 0; i < newshape.ndim(); ++i) {
      if (!found_minus_1) {
        if (newshape[i] == -1) {
          newshape[i] = 1;
          newshape[i] = size() / newshape.size();
          found_minus_1 = true;
        } else {
          TV_ASSERT_INVALID_ARG(newshape[i] > 0,
                                "shape except -1 must larger than 0");
        }
      } else {
        TV_ASSERT_INVALID_ARG(newshape[i] > 0, "multiple -1 in your argument.");
      }
    }
    TV_ASSERT_RT_ERR(newshape.size() == size(), "error");
    bool stride_valid = true;
    auto res_stride = computeStride(shape_, stride_, newshape, stride_valid);
    TV_ASSERT_RT_ERR(stride_valid, "non-contiguous stride can't handled.");

    Tensor res(*this);
    res.shape_ = newshape;
    res.stride_ = res_stride;
    return res;
  }

  template <class... Inds> Tensor view(Inds... newShapes) const {
    static_assert(sizeof...(newShapes) > 0, "dont support empty for now");
    TensorShape shape{int(newShapes)...};
    return view(shape);
  }

  Tensor operator[](int64_t index) { return select(0, index); }

  Tensor squeeze() const { return view(shape_.squeeze()); }

  Tensor squeeze(int axis) const {
    if (axis < 0) {
      axis = ndim() + axis;
    }
    return view(shape_.squeeze(axis));
  }

  Tensor unsqueeze(int axis) const {
    if (axis < 0) {
      axis = ndim() + axis + 1;
    }
    TV_ASSERT_INVALID_ARG(axis >= 0 && axis < ndim() + 1, "error");
    return view(shape_.unsqueeze(axis));
  }

  bool pinned() const { return storage_ ? storage_->pinned() : false; }

  Tensor slice_first_axis(int64_t start, int64_t end) const {
    return slice(0, start, end, 1, false, false);
  }

private:
  bool compute_is_contiguous() const {
    bool is_contiguous = true;
    if (empty())
      return is_contiguous;
    int64_t z = 1;
    for (int64_t d = ndim() - 1; d >= 0; d--) {
      const auto size_d = dim(d);
      if (size_d != 1) {
        if (stride(d) == z) {
          z *= size_d;
        } else {
          is_contiguous = false;
          break;
        }
      }
    }
    return is_contiguous;
  }

  static size_t computeStorageNbytes(TensorShape sizes, TensorShape strides,
                                     size_t itemsize_bytes) {
    // size of the underlying storage is 1 bigger than the offset
    // of the last element according to stride
    size_t size = 1;
    for (int i = 0; i < sizes.ndim(); ++i) {
      if (sizes[i] == 0) {
        return 0;
      }
      size += strides[i] * (sizes[i] - 1);
    }
    return size * itemsize_bytes;
  }

  static void checkInBoundsForStorage(TensorShape size, TensorShape stride,
                                      int64_t storage_offset_bytes,
                                      const DType data_type,
                                      int64_t new_storage_size_bytes) {
    int64_t storage_size_bytes =
        computeStorageNbytes(size, stride, tv::detail::sizeof_dtype(data_type));
    if (storage_size_bytes == 0) {
      // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
      return;
    }
    TV_ASSERT_INVALID_ARG(
        storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
        "setStorage: sizes ", size, ", strides ", stride,
        ","
        " storage byte offset ",
        storage_offset_bytes, ", and itemsize ",
        tv::detail::sizeof_dtype(data_type), " requiring a storage size of ",
        storage_size_bytes + storage_offset_bytes,
        " are out of bounds for storage of size ", new_storage_size_bytes);
  }

  static inline TensorShape computeStride(TensorShape oldshape,
                                          TensorShape oldstride,
                                          const TensorShape &newshape,
                                          bool &success) {
    if (oldshape.empty()) {
      return TensorShape(newshape.ndim(), 1);
    }

    // NOTE: stride is arbitrary in the numel() == 0 case;
    // to match NumPy behavior we copy the strides if the size matches,
    // otherwise we use the stride as if it were computed via resize. This could
    // perhaps be combined with the below code, but the complexity didn't seem
    // worth it.
    const int64_t numel = oldshape.size();
    if (numel == 0 && oldshape == newshape) {
      return oldstride;
    }

    TensorShape newstride(newshape.ndim());
    if (numel == 0) {
      for (int64_t view_d = newshape.ndim() - 1; view_d >= 0; view_d--) {
        if (view_d == (int64_t)(newshape.ndim() - 1)) {
          newstride[view_d] = 1;
        } else {
          newstride[view_d] = std::max<int64_t>(newshape[view_d + 1], 1) *
                              newstride[view_d + 1];
        }
      }
      return newstride;
    }

    int64_t view_d = (int64_t)newshape.ndim() - 1;
    // stride for each subspace in the chunk
    int64_t chunk_base_stride = oldstride.back();
    // numel in current chunk
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;
    for (int64_t tensor_d = oldshape.ndim() - 1; tensor_d >= 0; tensor_d--) {
      tensor_numel *= oldshape[tensor_d];
      // if end of tensor size chunk, check view
      if ((tensor_d == 0) ||
          (oldshape[tensor_d - 1] != 1 &&
           oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
        while (view_d >= 0 &&
               (view_numel < tensor_numel || newshape[view_d] == 1)) {
          newstride[view_d] = view_numel * chunk_base_stride;
          view_numel *= newshape[view_d];
          view_d--;
        }
        if (view_numel != tensor_numel) {
          success = false;
          return TensorShape();
        }
        if (tensor_d > 0) {
          chunk_base_stride = oldstride[tensor_d - 1];
          tensor_numel = 1;
          view_numel = 1;
        }
      }
    }
    if (view_d != -1) {
      success = false;
      return TensorShape();
    }
    return newstride;
  }

public:
  Tensor as_strided(TensorShape shape, TensorShape stride,
                    int64_t storage_byte_offset) const {
    TV_ASSERT_INVALID_ARG(!empty(), "tensor must not empty");
    // pytorch/c10/core/TensorImpl.h
    // pytorch/aten/native/Resize.h
    // TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_INVALID_ARG(shape.ndim() == stride.ndim() &&
                              storage_byte_offset >= 0,
                          "rt error", shape, stride, storage_byte_offset);
    checkInBoundsForStorage(shape, stride, storage_byte_offset, dtype_,
                            storage_->size());

    auto new_shape = shape;
    auto new_stride = stride;
    int new_dim = new_shape.ndim();
    if (new_dim > 0) {
      for (size_t dim = new_dim - 1;; dim--) {
        if (stride[dim] >= 0) {
          new_stride[dim] = stride[dim];
        } else {
          // XXX: This behavior is surprising and may need to be removed to
          // support negative strides. Some pytorch functions rely on it:
          // for example, torch.cat (run TestTorch.test_cat_empty).
          if (dim == new_dim - 1) {
            new_stride[dim] = 1;
          } else {
            // Keep stride monotonically increasing to match NumPy.
            new_stride[dim] = std::max<TV_GLOBAL_INDEX>(new_shape[dim + 1], 1) *
                              new_stride[dim + 1];
          }
        }
        if (dim == 0)
          break;
      }
    }
    size_t new_offset = storage_byte_offset;
    Tensor res(*this);
    res.shape_ = new_shape;
    res.stride_ = new_stride;
    res.offset_ = new_offset;
    res.writeable_ = writeable_;
    res.contiguous_ = res.compute_is_contiguous();
    return res;
  }

  Tensor transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 == dim1) {
      return *this;
    }
    auto new_shape = this->shape();
    auto new_stride = this->stride();
    std::swap(new_stride[dim0], new_stride[dim1]);
    std::swap(new_shape[dim0], new_shape[dim1]);
    return as_strided(new_shape, new_stride, offset_);
  }

  auto storage() const { return storage_; }
  auto storage() { return storage_; }

  Tensor slice(int dim, TV_GLOBAL_INDEX start, TV_GLOBAL_INDEX end,
               TV_GLOBAL_INDEX step, bool start_is_none,
               bool end_is_none) const {
    TV_GLOBAL_INDEX start_val = start_is_none ? 0 : start;
    TV_GLOBAL_INDEX end_val =
        end_is_none ? std::numeric_limits<TV_GLOBAL_INDEX>::max() : end;

    // TODO: support negative strides
    auto new_shape = this->shape();
    auto new_stride = this->stride();
    TV_ASSERT_RT_ERR(step > 0 && dim < ndim(), "slice step must be positive");

    // INT64_MAX stands for default value.
    if (start_val == std::numeric_limits<TV_GLOBAL_INDEX>::max()) {
      start_val = 0;
    }
    if (start_val < 0) {
      start_val += this->dim(dim);
    }
    if (end_val < 0) {
      end_val += this->dim(dim);
    }
    if (start_val < 0) {
      start_val = 0;
    } else if (start_val >= this->dim(dim)) {
      start_val = this->dim(dim);
    }
    if (end_val < start_val) {
      end_val = start_val;
    } else if (end_val >= this->dim(dim)) {
      end_val = this->dim(dim);
    }
    auto storage_offset_bytes =
        offset_ + start_val * this->stride(dim) * itemsize();
    auto len = end_val - start_val;
    new_shape[dim] = (len + step - 1) / step; // round-up
    new_stride[dim] *= step;
    return as_strided(new_shape, new_stride, storage_offset_bytes);
  }

  Tensor select(int dim, TV_GLOBAL_INDEX index) const {
    TV_ASSERT_INVALID_ARG(ndim() > 1, "error");
    if (index < 0) {
      index += this->dim(dim);
    }
    TV_ASSERT_INVALID_ARG(index < this->dim(dim), "error");
    auto storage_offset = offset_ + index * stride(dim) * itemsize();
    TensorShape res_shape = shape();
    TensorShape res_stride = stride();
    res_shape.erase(res_shape.begin() + dim);
    res_stride.erase(res_stride.begin() + dim);
    return as_strided(res_shape, res_stride, storage_offset);
  }

  bool is_contiguous() const { return contiguous_; }

  bool empty() const {
    return !storage_ || storage_->empty() || shape().empty();
  }
  DType dtype() const { return dtype_; }
  int device() const { return storage_ ? storage_->device() : -1; }
  size_t ndim() const { return shape_.ndim(); }

  const TensorShape &shape() const { return shape_; }
  const std::vector<int64_t> shape_vector() const {
    return std::vector<int64_t>(shape_.begin(), shape_.end());
  }

  const TensorShape &strides() const { return stride_; }
  int64_t stride(int idx) const {
    if (idx < 0) {
      TV_ASSERT_RT_ERR(stride_.ndim() + idx < stride_.ndim(), idx, stride_);
      return stride_[stride_.ndim() + idx];
    } else {
      TV_ASSERT_RT_ERR(idx < int(stride_.ndim()), idx, stride_);
      return stride_[idx];
    }
  }
  const TensorShape &sizes() const { return shape_; }
  const TensorShape &stride() const { return stride_; }

  int64_t dim(int idx) const {
    if (idx < 0) {
      TV_ASSERT_RT_ERR(shape_.ndim() + idx < shape_.ndim(), idx, shape_);
      return shape_[shape_.ndim() + idx];
    } else {
      TV_ASSERT_RT_ERR(idx < int(shape_.ndim()), idx, shape_);
      return shape_[idx];
    }
  }
  const uint8_t *raw_data() const {
    if (empty()) {
      return nullptr;
    }
    return storage_->data() + byte_offset();
  }
  size_t raw_size() const { return size() * itemsize(); }
  size_t nbytes() const { return raw_size(); }
  size_t size() const { return shape_.size(); }
  size_t numel() const { return shape_.size(); }
  size_t size(int64_t idx) const { return dim(idx); }
  size_t storage_size() const { return storage_ ? storage_->size() : 0; }

  size_t itemsize() const { return detail::sizeof_dtype(dtype_); }
  size_t byte_offset() const { return offset_; }
  bool is_cpu() const { return storage_ ? storage_->is_cpu() : true; }
  bool is_readonly() const { return !writeable_; }
  std::uintptr_t gpu_address() const { return storage()->gpu_address(); }
  Tensor get_readonly() const {
    // used for cumm inliner, we can capture const tensor
    // as const ptr
    auto res = *this;
    res.writeable_ = false;
    return res;
  }
  Tensor &zero_(Context ctx = Context()) {
    if (empty()) {
      return *this;
    }
    writable_check();
    storage_->zero_(byte_offset(), raw_size(), ctx);
    return *this;
  }
  bool is_col_major_matrix() const {
    TV_ASSERT_INVALID_ARG(ndim() == 2, "tensor must be 2d");
    return stride(0) == 1 && stride(1) == dim(0);
  }
  uint8_t *raw_data(bool writeable_check = true) {
    if (empty()) {
      return nullptr;
    }
    if (writeable_check) {
      writable_check();
    }
    return storage_->data() + byte_offset();
  }
  const uint8_t *const_raw_data() const {
    if (empty()) {
      return nullptr;
    }
    return storage_->data() + byte_offset();
  }
  template <typename T> Tensor &fill_template_(T val, Context ctx) {
    writable_check();
    auto target_device = this->device();
    // #ifdef TV_HARDWARE_ACC_METAL
    //     target_device = -1;
    // #endif
    if (target_device == -1) {
      std::fill(this->data_ptr<T>(), this->data_ptr<T>() + this->size(), val);
    } else {
#if defined(TV_HARDWARE_ACC_CUDA) && (TV_USE_CUDA_DRIVER)
      auto tview = this->tview<T, -1, tv::DefaultPtrTraits, int64_t>();
      if (ctx.has_cuda_stream()) {
        tv::FillDev<T, -1, tv::DefaultPtrTraits, int64_t>::run_async(
            tview, val, ctx.cuda_stream());
      } else {
        tv::FillDev<T, -1, tv::DefaultPtrTraits, int64_t>::run(tview, val);
      }
#else
      TV_THROW_INVALID_ARG("don't compiled with cuda and cuda driver");
#endif
    }
    return *this;
  }
  Tensor to(int device, Context ctx = Context()) const {
    if (device == -1) {
      return cpu(ctx);
    } else {
#if defined(TV_ENABLE_HARDWARE_ACC)
      return cuda(ctx);
#else
      TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
    }
  }
  template <typename T> Tensor &fill_template_(T val) {
    return fill_template_<T>(val, Context());
  }

  Tensor &fill_(int64_t val, Context ctx = Context()) {
    auto target_device = this->device();
    // #ifdef TV_HARDWARE_ACC_METAL
    //     target_device = -1;
    // #endif
    using int_types_t =
        std::tuple<int32_t, int16_t, int8_t, uint32_t, uint16_t, uint8_t>;
    using int_types_cpu_t = std::tuple<uint64_t, int64_t, int32_t, int16_t,
                                       int8_t, uint32_t, uint16_t, uint8_t>;
    if (target_device == -1) {
      Dispatch<int_types_cpu_t>()(dtype_, [&](auto I) -> void {
        using T = TV_DECLTYPE(I);
        fill_template_<T>(val, ctx);
      });
    } else {
      Dispatch<int_types_t>()(dtype_, [&](auto I) -> void {
        using T = TV_DECLTYPE(I);
        fill_template_<T>(val, ctx);
      });
    }
    return *this;
  }

  Tensor &fill_(float val, Context ctx = Context()) {
    using float_types_t = std::tuple<float>;
    Dispatch<float_types_t>()(dtype_, [&](auto I) -> void {
      using T = TV_DECLTYPE(I);
      fill_template_<T>(val, ctx);
    });
    return *this;
  }

  template <typename T, bool DoTypeCheck = true> T *data() {
    if (empty()) {
      return nullptr;
    }
    if (DoTypeCheck) {
      template_dtype_check<T>();
    }
    if (!std::is_const<T>::value) {
      writable_check();
    }
    return reinterpret_cast<T *>(raw_data(false));
  }

  template <typename T, bool DoTypeCheck = true> const T *data() const {
    if (empty()) {
      return nullptr;
    }
    if (DoTypeCheck) {
      template_dtype_check<T>();
    }
    return reinterpret_cast<const T *>(raw_data());
  }

  template <typename T> T *data_ptr() { return data<T>(); }

  template <typename T> const T *data_ptr() const { return data<T>(); }

  void *data_ptr() { return reinterpret_cast<void *>(raw_data()); }

  const void *data_ptr() const {
    return reinterpret_cast<const void *>(raw_data());
  }
  bool managed() const { return storage_ ? storage_->managed() : false; }
  void copy_(const Tensor &tensor, Context ctx = Context()) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_RT_ERR(!this->empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(this->size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(this->dtype() == tensor.dtype(), "must have same dtype",
                     dtype_str(this->dtype()), dtype_str(tensor.dtype()));
    detail::TensorStorage<uint8_t>::copy(
        *(tensor.storage_), tensor.byte_offset(), *(this->storage_),
        this->byte_offset(), this->raw_size(), ctx);
    // #ifdef TV_HARDWARE_ACC_METAL
    //     // use apple metal api to copy
    //     bool sync_op = is_cpu(); // always sync if cpu to simulate cuda
    //     behavior if (!ctx.has_item(ContextType::kAppleMetal)) {
    //       ctx = Context().create_apple_metal_context();
    //       sync_op = true;
    //     }
    //     auto *metal_ctx = reinterpret_cast<AppleMetalContext *>(
    //                               ctx.get_item(ContextType::kAppleMetal));
    //     auto cb = metal_ctx->commandBuffer();
    //     TV_ASSERT_RT_ERR(cb, "error");
    //     auto bce = detail::make_apple_mtl_ptr(cb->blitCommandEncoder());
    //     TV_ASSERT_RT_ERR(bce, "error");

    //     auto src = tensor.storage_->apple_metal_buffer_ptr();
    //     auto dst = storage_->apple_metal_buffer_ptr();
    //     auto src_offset = tensor.byte_offset() +
    //     tensor.storage_->byte_offset(); auto dst_offset = byte_offset() +
    //     storage_->byte_offset(); auto size = raw_size();
    //     bce->copyFromBuffer(src, src_offset, dst, dst_offset, size);
    //     bce->endEncoding();

    //     if (sync_op) {
    //       metal_ctx->synchronize(AppleMetalContext::SyncType::COMMIT_AND_WAIT);
    //     }else{
    //       metal_ctx->synchronize(AppleMetalContext::SyncType::COMMIT);
    //     }
    //     return;
    // #else

    //     if (this->device() == -1 && tensor.device() == -1) {
    // #if defined(TV_HARDWARE_ACC_CUDA)
    //       // use memcpy instead to avoid cuda context init
    //       if (this->pinned()) {
    //         if (ctx.has_cuda_stream()) {
    //           host2host(this->raw_data(), tensor.raw_data(),
    //                     this->size() * detail::sizeof_dtype(dtype_),
    //                     ctx.cuda_stream());

    //         } else {
    //           host2host(this->raw_data(), tensor.raw_data(),
    //                     this->size() * detail::sizeof_dtype(dtype_));
    //         }
    //       } else {
    //         std::copy(tensor.raw_data(),
    //                   tensor.raw_data() + size() *
    //                   detail::sizeof_dtype(dtype_), raw_data());
    //       }
    // #else
    //       std::copy(tensor.raw_data(),
    //                 tensor.raw_data() + size() *
    //                 detail::sizeof_dtype(dtype_), raw_data());
    // #endif
    //     }
    // #if defined(TV_HARDWARE_ACC_CUDA)
    //     else if (device() >= 0 && tensor.device() == -1) {
    //       if (ctx.has_cuda_stream()) {
    //         host2dev(raw_data(), tensor.raw_data(),
    //                  size() * detail::sizeof_dtype(dtype_),
    //                  ctx.cuda_stream());

    //       } else {
    //         host2dev(raw_data(), tensor.raw_data(),
    //                  size() * detail::sizeof_dtype(dtype_));
    //       }

    //     } else if (device() == -1 && tensor.device() >= 0) {
    //       if (ctx.has_cuda_stream()) {
    //         dev2host(raw_data(), tensor.raw_data(),
    //                  size() * detail::sizeof_dtype(dtype_),
    //                  ctx.cuda_stream());

    //       } else {
    //         dev2host(raw_data(), tensor.raw_data(),
    //                  size() * detail::sizeof_dtype(dtype_));
    //       }
    //     } else if (device() >= 0 && tensor.device() >= 0) {
    //       if (ctx.has_cuda_stream()) {
    //         dev2dev(raw_data(), tensor.raw_data(),
    //                 size() * detail::sizeof_dtype(dtype_),
    //                 ctx.cuda_stream());
    //       } else {
    //         dev2dev(raw_data(), tensor.raw_data(),
    //                 size() * detail::sizeof_dtype(dtype_));
    //       }
    //     }
    // #endif
    //     else {
    //       TV_THROW_RT_ERR("only support cpu tensor");
    //     }
    // #endif
  }

  void copy_storage_(const Tensor &tensor, Context ctx = Context()) {
    writable_check();
    TV_ASSERT_RT_ERR(!this->empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(this->storage_->size() == tensor.storage_->size(),
                     "storage must have same size", this->shape(),
                     tensor.shape(), this->storage_->size(),
                     tensor.storage_->size());

    detail::TensorStorage<uint8_t>::copy(*(tensor.storage_), 0,
                                         *(this->storage_), 0,
                                         this->storage_->size(), ctx);
  }

  void copy_2d_pitched_(const Tensor &tensor, Context ctx = Context()) {
    writable_check();
    TV_ASSERT_RT_ERR(!this->empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(this->dtype() == tensor.dtype(), "must have same dtype",
                     dtype_str(this->dtype()), dtype_str(tensor.dtype()));
    TV_ASSERT_RT_ERR(this->ndim() == 2 && tensor.ndim() == 2,
                     "must be 2d tensor");
    TV_ASSERT_RT_ERR(this->stride(1) == 1 && tensor.stride(1) == 1,
                     "stride[1] must be 1");
    auto w = this->dim(1) * detail::sizeof_dtype(dtype_);
    auto h = this->dim(0);
    auto sw = tensor.dim(1) * detail::sizeof_dtype(dtype_);
    auto sh = tensor.dim(0);
    TV_ASSERT_RT_ERR(w == sw && h == sh, "shape must be same");
    if (this->contiguous_ && tensor.contiguous_) {
      return copy_(tensor, ctx);
    }
#ifdef TV_HARDWARE_ACC_METAL
    TV_THROW_INVALID_ARG(
        "copy 2d non-contiguous array not implemented for apple metal");
#endif

    auto dst = this->raw_data();
    auto src = tensor.const_raw_data();
    auto dst_pitch = this->stride(0) * detail::sizeof_dtype(dtype_);
    auto src_pitch = tensor.stride(0) * detail::sizeof_dtype(dtype_);
    if (this->device() == -1 && tensor.device() == -1) {
#if defined(TV_HARDWARE_ACC_CUDA)
      checkCudaErrors(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, w, h,
                                   cudaMemcpyHostToHost));
#else
      TV_THROW_INVALID_ARG("not implemented for cpu tensor")
#endif
    }
#if defined(TV_HARDWARE_ACC_CUDA)
    else if (device() >= 0 && tensor.device() == -1) {
      if (ctx.has_cuda_stream()) {
        checkCudaErrors(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, w, h,
                                          cudaMemcpyHostToDevice,
                                          ctx.cuda_stream()));
      } else {
        checkCudaErrors(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, w, h,
                                     cudaMemcpyHostToDevice));
      }
    } else if (device() == -1 && tensor.device() >= 0) {
      if (ctx.has_cuda_stream()) {
        checkCudaErrors(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, w, h,
                                          cudaMemcpyDeviceToHost,
                                          ctx.cuda_stream()));
      } else {
        checkCudaErrors(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, w, h,
                                     cudaMemcpyDeviceToHost));
      }
    } else if (device() >= 0 && tensor.device() >= 0) {
      if (ctx.has_cuda_stream()) {
        checkCudaErrors(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, w, h,
                                          cudaMemcpyDeviceToDevice,
                                          ctx.cuda_stream()));
      } else {
        checkCudaErrors(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, w, h,
                                     cudaMemcpyDeviceToDevice));
      }
    }
#endif
    else {
      TV_THROW_RT_ERR("only support cpu tensor");
    }
  }

  void copy_cpu_(const Tensor &tensor) {
    // this function exists for cloud environment that
    // use cheap non-gpu instance with library compiled
    // with cuda.
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_RT_ERR(!this->empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(this->size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(this->dtype() == tensor.dtype(), "must have same dtype",
                     dtype_str(this->dtype()), dtype_str(tensor.dtype()));
    TV_ASSERT_RT_ERR(this->device() == -1 && tensor.device() == -1,
                     "all tensors must be cpu");

    std::copy(tensor.raw_data(),
              tensor.raw_data() + size() * detail::sizeof_dtype(dtype_),
              raw_data());
  }

  Tensor cpu(Context ctx = Context()) const {
    if (empty()) {
      return Tensor();
    }
    if (storage_->device() == -1) {
      // cpu() should always copy tensor.
      return clone(false, false, ctx);
    }
    Tensor res;
    if (shape_.ndim() == 2 && this->stride(1) == 1) {
      res = Tensor(shape_, dtype_, -1, storage_->managed());
      res.copy_2d_pitched_(*this, ctx);
    } else {
      if (!contiguous_) {
        void *ptr = nullptr;
        res = Tensor(ptr, shape_, stride_, dtype_, -1);
        res.storage_ = storage_->cpu(ctx);
      } else {
        res = Tensor(shape_, stride_, dtype_, -1, storage_->managed());
        res.copy_(*this, ctx);
      }
    }
    return res;
  }

#if defined(TV_ENABLE_HARDWARE_ACC)
  Tensor cuda(Context ctx = Context()) const {
    if (empty()) {
      return Tensor();
    }
    if (storage_->device() >= 0) {
      // cuda() should always copy tensor.
      return clone();
    }
    Tensor res(shape_, stride_, dtype_, 0, storage_->managed());
    res.copy_(*this, ctx);
    return res;
  }
#endif

  template <typename T> T item() const {
    TV_ASSERT_RT_ERR(size() == 1, "size must be 1");
    auto tensor = *this;
    if (storage_->is_cpu()) {
      tensor = cpu();
    }
    return *(tensor.data_ptr<T>());
  }

  template <typename T> void copy_(const TensorView<T> &tensor, int device) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    Tensor src = from_blob(tensor, device);
    return copy_(src);
  }

  Tensor &operator=(const Tensor &tensor) {
    dtype_ = tensor.dtype_;
    storage_ = tensor.storage_;
    shape_ = tensor.shape_;
    writeable_ = tensor.writeable_;
    offset_ = tensor.offset_;
    stride_ = tensor.stride_;
    contiguous_ = tensor.contiguous_;
    return *this;
  }

  Tensor(const Tensor &tensor) {
    dtype_ = tensor.dtype_;
    storage_ = tensor.storage_;
    shape_ = tensor.shape_;
    writeable_ = tensor.writeable_;
    offset_ = tensor.offset_;
    stride_ = tensor.stride_;
    contiguous_ = tensor.contiguous_;
  }

  Tensor type_view(DType dtype, TensorShape newshape) const {
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_INVALID_ARG(detail::sizeof_dtype(dtype) * newshape.size() ==
                              itemsize() * this->size(),
                          "dtype itemsize multiple size must same");
    auto ten = *this;
    ten.dtype_ = dtype;
    ten.shape_ = newshape;
    ten.stride_ = newshape.stride_rowmajor();

    return ten;
  }

  Tensor type_view(DType dtype) const {
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    auto dtype_size = detail::sizeof_dtype(dtype);
    auto self_dtype_size = itemsize();
    auto new_shape = shape_;
    if (dtype_size >= self_dtype_size) {
      TV_ASSERT_INVALID_ARG(dtype_size % self_dtype_size == 0, "error",
                            dtype_size, self_dtype_size);
      int rate = dtype_size / self_dtype_size;
      TV_ASSERT_INVALID_ARG(this->dim(this->ndim() - 1) % rate == 0, "error",
                            this->dim(this->ndim() - 1), rate);
      auto new_shape = shape_;
      new_shape[this->ndim() - 1] = this->dim(this->ndim() - 1) / rate;
    } else {
      TV_ASSERT_INVALID_ARG(self_dtype_size % dtype_size == 0, "error",
                            dtype_size, self_dtype_size);
      int rate = self_dtype_size / dtype_size;
      new_shape[this->ndim() - 1] = this->dim(this->ndim() - 1) * rate;
    }
    auto ten = *this;
    ten.dtype_ = dtype;
    ten.shape_ = new_shape;
    ten.stride_ = new_shape.stride_rowmajor();
    return ten;
  }

  Tensor clone(bool pinned = false, bool use_cpu_copy = false,
               Context ctx = Context()) const {
    if (empty()) {
      return Tensor();
    }
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    Tensor newtensor(shape_, stride_, dtype_, device(), pinned,
                     storage_->managed());
    if (!empty()) {
      if (use_cpu_copy) {
        TV_ASSERT_INVALID_ARG(device() == -1, "tensor must be cpu");
        newtensor.copy_cpu_(*this);
      } else {
        newtensor.copy_(*this, ctx);
      }
    }
    return newtensor;
  }

  Tensor clone_whole_storage(Context ctx = Context()) const {
    // this method clone whole storage,
    // so it can be used with non-contiguous tensor.
    if (empty()) {
      return Tensor();
    }
    Tensor res = *this;
    res.storage_ = storage_->clone(ctx);
    return res;
  }

  void zero_whole_storage_(Context ctx = Context()) const {
    // this method zero whole storage,
    // so it can be used with non-contiguous tensor.
    if (empty()) {
      return;
    }
    storage_->zero_whole_(ctx);
  }

  Tensor rand_(int seed = -1) {
    std::random_device rd;
    if (seed == -1) {
      seed = rd();
    }
    dispatch<float, double>(dtype_, [&](auto I) {
      using T = TV_DECLTYPE(I);
      TensorView<T> tensor_tv = this->tview<T>();
      if (this->device() == -1) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<T> distr(0, 1);
        for (size_t i = 0; i < this->size(); ++i) {
          tensor_tv[i] = distr(generator);
        }
      } else {
#if defined(TV_HARDWARE_ACC_CUDA)
        curandGenerator_t gen;
        auto status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        if (status != CURAND_STATUS_SUCCESS) {
          TV_THROW_RT_ERR("curand failed", status);
        }
        status = curandSetPseudoRandomGeneratorSeed(gen, seed);
        if (status != CURAND_STATUS_SUCCESS) {
          TV_THROW_RT_ERR("curand failed", status);
        }
        if_constexpr<std::is_same<T, float>::value>(
            [&](auto _) {
              status =
                  curandGenerateUniform(gen, _(this->data_ptr<T>()), size());
              if (status != CURAND_STATUS_SUCCESS) {
                TV_THROW_RT_ERR("curand failed", status);
              }
            },
            [&](auto _) {
              status = curandGenerateUniformDouble(gen, _(this->data_ptr<T>()),
                                                   size());
              if (status != CURAND_STATUS_SUCCESS) {
                TV_THROW_RT_ERR("curand failed", status);
              }
            });
        TV_CUDA_RESULT_CHECK(curandDestroyGenerator(gen));
#else
        TV_THROW_RT_ERR("not implemented");
#endif
      }
    });
    return *this;
  }

  Tensor rand_int_(int begin, int end, int seed = -1) {
    std::random_device rd;
    if (seed == -1) {
      seed = rd();
    }
    if (this->device() == -1) {
      using all_int16_32_64_t =
          std::tuple<int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t>;
      using all_int8_t = std::tuple<int8_t, uint8_t>;
      bool found = DispatchNoExcept<all_int16_32_64_t>()(dtype_, [&](auto I) {
        using T = TV_DECLTYPE(I);
        TensorView<T> tensor_tv = this->tview<T>();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<T> distr(begin, end);
        for (size_t i = 0; i < this->size(); ++i) {
          tensor_tv[i] = distr(generator);
        }
      });
      if (!found) {
        Dispatch<all_int8_t>()(dtype_, [&](auto I) {
          found = true;
          using T = TV_DECLTYPE(I);
          TensorView<T> tensor_tv = this->tview<T>();
          std::mt19937 generator(seed);
          constexpr int kTRange = int32_t(std::numeric_limits<T>::max()) -
                                  int32_t(std::numeric_limits<T>::min());
          std::uniform_int_distribution<int32_t> distr(begin, end);
          for (size_t i = 0; i < this->size(); ++i) {
            tensor_tv[i] = T((distr(generator) % kTRange) -
                             int32_t(std::numeric_limits<T>::min()));
          }
        });
      }
    } else {
      TV_THROW_RT_ERR("not implemented");
    }
    return *this;
  }

  Tensor astype(DType dtype, bool use_cpu_copy = false) const {
    if (dtype == dtype_) {
      return this->clone(this->pinned(), use_cpu_copy);
    }
    TV_ASSERT_INVALID_ARG(this->device() == -1, "only support cpu tensor");
    TV_ASSERT_INVALID_ARG(!this->empty(), "can't be used in empty tensor");
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    auto tensor = Tensor();
    Dispatch<detail::all_tensor_types_t>()(dtype, [&](auto Idst) {
      using Tdst = TV_DECLTYPE(Idst);
      Dispatch<detail::all_tensor_types_t>()(this->dtype_, [&](auto Icur) {
        using Tcur = TV_DECLTYPE(Icur);
        // if constexpr (std::is_convertible<Tcur, Tdst>::value){
        //   auto ptr = this->data_ptr<Tcur>();
        //   tensor =
        //       Tensor(this->shape_, this->stride_, dtype, this->device(),
        //               this->pinned(), this->storage_->managed());
        //   if constexpr (detail::ConvertTmpType<Tcur, Tdst>::kSpec){
        //     Tdst* tensor_data = tensor.data_ptr<Tdst>();
        //     using TmpType = typename detail::ConvertTmpType<Tcur,
        //     Tdst>::type; for (int i = 0; i < this->size(); ++i){
        //       tensor_data[i] = TmpType(ptr[i]);
        //     }
        //   }else{
        //     std::copy(ptr, ptr + this->size(), tensor.data<Tdst>());
        //   }
        // }else{
        //   TV_THROW_INVALID_ARG("not convertable from",
        //                         type_s<std::decay_t<Tcur>>, "to",
        //                         type_s<std::decay_t<Tdst>>);
        // }
        if_constexpr<detail::is_convertible<Tcur, Tdst>::value>(
            [&](auto _) {
              auto ptr = this->data<Tcur>();
              tensor =
                  Tensor(this->shape_, this->stride_, dtype, this->device(),
                         this->pinned(), this->storage_->managed());
              std::copy(ptr, ptr + this->size(), tensor.data<Tdst>());
            },
            [&](auto _) {
              TV_THROW_INVALID_ARG("not convertable from",
                                   type_s<std::decay_t<Tcur>>, "to",
                                   type_s<std::decay_t<Tdst>>);
            });
      });
    });
    return tensor;
  }
  bool writeable() const { return writeable_; }

protected:
  inline void writable_check() {
    TV_ASSERT_RT_ERR(writeable_,
                     "you cant do non-const operation when not writable");
  }

  template <typename T> inline void template_dtype_check() const {
    if (dtype_ >= custom16 && dtype_ <= custom128) {
      auto dsize = detail::sizeof_dtype(dtype_);
      TV_ASSERT_RT_ERR(dsize == sizeof(T), "expect size", sizeof(T),
                       "but sizeof(dtype_) =", dsize);
    } else {
      TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "expect", type_s<T>,
                       "but dtype_ =", dtype_str(dtype_));
    }
  }

  DType dtype_;
  std::shared_ptr<detail::TensorStorage<uint8_t>> storage_;
  TensorShape shape_;
  size_t offset_ = 0;
  TensorShape stride_;

private:
  bool writeable_ = true;
  bool contiguous_ = true;
};

template <typename Os> Os &operator<<(Os &os, const Tensor &tensor) {
  TV_ASSERT_INVALID_ARG(tensor.device() == -1 ||
                            (tensor.device() == 0 && tensor.managed()),
                        "must be cpu tensor");
  Dispatch<detail::all_tensor_types_t>()(tensor.dtype(), [&](auto I) {
    using T = TV_DECLTYPE(I);
    std::stringstream ss;
    if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      ss << std::setprecision(4);
    }
#if defined(TV_HARDWARE_ACC_CUDA) && CUDA_VERSION < 11000
    if_constexpr<std::is_same<T, __half>::value>(
        [&](auto _) {
          auto tensorf = tensor.astype(float32);
          auto tview =
              tensorf.tview<const float, -1, DefaultPtrTraits, int64_t>();
          os << tview.repr(ss);
        },
        [&](auto _) {
          auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
          os << tview.repr(ss);
        });
#elif defined(TV_HARDWARE_ACC_CUDA) && CUDA_VERSION >= 11000
    if_constexpr<std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value>([&](auto _){
      auto tensorf = tensor.astype(float32);
      auto tview = tensorf.tview<const float, -1, DefaultPtrTraits, int64_t>();
      os << tview.repr(ss);
    }, [&](auto _){
      auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
      os << tview.repr(ss);
    });
#elif defined(TV_HARDWARE_ACC_CUDA) && CUDA_VERSION >= 11080
    if_constexpr<std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value || std::is_same<T, __nv_fp8_e5m2>::value || std::is_same<T, __nv_fp8_e4m3>::value>([&](auto _){
      auto tensorf = tensor.astype(float32);
      auto tview = tensorf.tview<const float, -1, DefaultPtrTraits, int64_t>();
      os << tview.repr(ss);
    }, [&](auto _){
      auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
      os << tview.repr(ss);
    });
#else 
    auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
    os << tview.repr(ss);
#endif
  });
  return os;
}

inline Tensor from_blob(void *ptr, TensorShape shape, DType dtype,
                        int device = -1, int storage_offset = 0) {
  return Tensor(ptr, shape, dtype, device, storage_offset);
}

inline Tensor from_blob(const void *ptr, TensorShape shape, DType dtype,
                        int device = -1, int storage_offset = 0) {
  return Tensor(ptr, shape, dtype, device, storage_offset);
}

inline Tensor from_blob(void *ptr, TensorShape shape, TensorShape stride,
                        DType dtype, int device = -1, int storage_offset = 0) {
  return Tensor(ptr, shape, stride, dtype, device, storage_offset);
}

inline Tensor from_blob(const void *ptr, TensorShape shape, TensorShape stride,
                        DType dtype, int device = -1, int storage_offset = 0) {
  return Tensor(ptr, shape, stride, dtype, device, storage_offset);
}

inline Tensor empty(TensorShape shape, DType dtype, int device = -1,
                    bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed);
}

inline Tensor zeros(TensorShape shape, DType dtype, int device = -1,
                    bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed).zero_();
}

template <typename T>
std::tuple<Tensor, T *> empty_with_ptr(TensorShape shape, int device = -1,
                                       bool pinned = false,
                                       bool managed = false) {
  auto res = Tensor(shape, type_v<T>, device, pinned, managed);
  auto res_ptr = res.data_ptr<T>();
  return std::make_tuple(res, res_ptr);
}

template <typename T>
std::tuple<Tensor, T *> zeros_with_ptr(TensorShape shape, int device = -1,
                                       bool pinned = false,
                                       bool managed = false) {
  auto res = Tensor(shape, type_v<T>, device, pinned, managed).zero_();
  auto res_ptr = res.data_ptr<T>();
  return std::make_tuple(res, res_ptr);
}

inline Tensor zeros_managed(TensorShape shape, DType dtype) {
  return Tensor(shape, dtype, 0, true, true).zero_();
}

inline Tensor full(TensorShape shape, int64_t val, DType dtype, int device = -1,
                   bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed).fill_(val);
}

inline Tensor full(TensorShape shape, float val, DType dtype, int device = -1,
                   bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed).fill_(val);
}

inline Tensor cat_first_axis(std::vector<Tensor> tensors) {
  TV_ASSERT_RT_ERR(tensors.size() > 0, "error");
  int first_shape = 0;
  TensorShape remain_shape = tensors[0].shape();
  auto dtype = tensors[0].dtype();
  auto ndim = tensors[0].ndim();
  for (auto &t : tensors) {
    first_shape += t.dim(0);
    TV_ASSERT_RT_ERR(t.dtype() == dtype, "error");
    TV_ASSERT_RT_ERR(t.ndim() == ndim, "error");
    for (int i = 1; i < ndim; ++i) {
      TV_ASSERT_RT_ERR(t.dim(i) == remain_shape[i], "error");
    }
  }
  remain_shape[0] = first_shape;
  Tensor res(remain_shape, tensors[0].dtype(), tensors[0].device());
  int count = 0;
  for (auto &t : tensors) {
    if (t.dim(0) == 0) {
      continue;
    }
    res.slice_first_axis(count, count + t.dim(0)).copy_(t);
    count += t.dim(0);
  }
  return res;
}

} // namespace tv