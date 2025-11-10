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
#include <array>
#include <memory>
#include <tensorview/core/common.h>
#include <tensorview/core/defs.h>
#include <tensorview/cuda/driverops.h>
#include <unordered_map>
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuda.h>
#endif
#ifdef TV_HARDWARE_ACC_METAL
#include "Metal/Metal.hpp"
#endif

namespace tv {

enum class ContextType : int { kCudaStream = 1, kCublasLt = 2, kCudnn = 3, kAppleMetal = 4 };

namespace detail {
#ifdef TV_HARDWARE_ACC_METAL
template <class T>
std::unique_ptr<T, void (*)(T *)> make_apple_mtl_ptr(T *ptr) {
  return std::unique_ptr<T, void (*)(T *)>(ptr, [](T *ptr) {
    if (ptr) {
      ptr->release();
    }
  });
}

inline const void *pageAlignedBlockPtr(const void *ptr, int size,
                                 int *alignedBlockSize) {
  uintptr_t address = (uintptr_t)ptr;
  uintptr_t alignedAddress = address & ~(PAGE_SIZE - 1);
  uintptr_t alignedEnd = ((address + size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  uint64_t alignedLength = alignedEnd - alignedAddress;

  TV_ASSERT_INVALID_ARG(address >= alignedAddress, "err");
  TV_ASSERT_INVALID_ARG(address + size <= alignedAddress + alignedLength,
                        "err");

  *alignedBlockSize = alignedLength;
  return (const void*)(alignedAddress);
}
inline void *pageAlignedBlockPtr(void *ptr, int size,
                                 int *alignedBlockSize) {
  uintptr_t address = (uintptr_t)ptr;
  uintptr_t alignedAddress = address & ~(PAGE_SIZE - 1);
  uintptr_t alignedEnd = ((address + size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  uint64_t alignedLength = alignedEnd - alignedAddress;

  TV_ASSERT_INVALID_ARG(address >= alignedAddress, "err");
  TV_ASSERT_INVALID_ARG(address + size <= alignedAddress + alignedLength,
                        "err");

  *alignedBlockSize = alignedLength;
  return (void*)(alignedAddress);
}

inline std::tuple<MTL::Buffer*, int, void*> create_apple_buffer_from_cpu_ptr(void* ptr, int byte_size, MTL::Device* dev_ptr){
  // for copy from/to cpu
  int alignedLength = 0;
  void* host_dst = ptr;
  void* alignedPtr = pageAlignedBlockPtr(host_dst, int(byte_size),
  &alignedLength); 
  int destOffset = (std::uintptr_t(host_dst) -
  std::uintptr_t(alignedPtr));
  // 4 bytes alignment required on macos for blits.
  TV_ASSERT_RT_ERR(destOffset % 4 == 0, "Unaligned blit request");
  auto ptr_ptl = dev_ptr->newBuffer(alignedPtr, alignedLength, MTL::ResourceStorageModeShared, nullptr);
  TV_ASSERT_RT_ERR(ptr_ptl, "Metal buffer not created", PAGE_SIZE);
  return std::make_tuple(ptr_ptl, destOffset, alignedPtr);
}
inline std::tuple<MTL::Buffer*, int, const void*> create_apple_buffer_from_cpu_ptr(const void* ptr, int byte_size, MTL::Device* dev_ptr){
  // for copy from/to cpu
  int alignedLength = 0;
  const void* host_dst = ptr;
  const void* alignedPtr = pageAlignedBlockPtr(host_dst, int(byte_size),
  &alignedLength); 
  int destOffset = (std::uintptr_t(host_dst) -
  std::uintptr_t(alignedPtr));
  // 4 bytes alignment required on macos for blits.
  TV_ASSERT_RT_ERR(destOffset % 4 == 0, "Unaligned blit request");
  auto ptr_ptl = dev_ptr->newBuffer((void*)(alignedPtr), alignedLength, MTL::ResourceStorageModeShared, nullptr);
  TV_ASSERT_RT_ERR(ptr_ptl, "Metal buffer not created", PAGE_SIZE);
  return std::make_tuple(ptr_ptl, destOffset, alignedPtr);
}

#endif 

}

#ifdef TV_HARDWARE_ACC_METAL

struct AppleMetalContext {
    enum class SyncType {
      NONE,               // no commit to command buffer
      COMMIT,             // commit and flush the command buffer
      COMMIT_AND_WAIT,    // flush and wait for command buffer execution to finish
      COMMIT_AND_CONTINUE,// commit and continue with a new underlying command buffer
    };
private:
    MTL::Device* device_ptr_ = nullptr;
    // The command queue used to pass commands to the device.
    MTL::CommandQueue* command_queue_ptr_ = nullptr;
    dispatch_queue_t dq_ptr_ = nullptr;
    MTL::CommandBuffer* command_buffer_ptr_external_ = nullptr;
    NS::SharedPtr<MTL::CommandBuffer> _commandBuffer, _prevCommandBuffer; 
    bool from_blob_ = false;
    bool _enableCommitAndContinue = true;
public:
    AppleMetalContext() {
        device_ptr_ = MTL::CreateSystemDefaultDevice();
        TV_ASSERT_RT_ERR(device_ptr_, "Metal device not found");
        command_queue_ptr_ = device_ptr_->newCommandQueue();
        TV_ASSERT_RT_ERR(command_queue_ptr_, "Metal command queue not found");
        dq_ptr_ = dispatch_queue_create("tv metal gpu stream", nullptr);
    }

    AppleMetalContext(MTL::CommandBuffer* cb, dispatch_queue_t dq): from_blob_(true) {
        device_ptr_ = MTL::CreateSystemDefaultDevice();
        TV_ASSERT_RT_ERR(device_ptr_, "Metal device not found");
        command_buffer_ptr_external_ = cb;
        TV_ASSERT_RT_ERR(command_buffer_ptr_external_, "Metal command queue not found");
        dq_ptr_ = dq;
    }

    void update_context_from_blob(MTL::CommandBuffer* cb, dispatch_queue_t dq) {
      TV_ASSERT_RT_ERR(from_blob_, "you can't update context from blob when context is not from blob");
      command_buffer_ptr_external_ = cb;
      dq_ptr_ = dq;
    }

    __attribute__((visibility("default"))) static std::shared_ptr<AppleMetalContext> getInstance()
#ifndef TV_STATIC_VARIABLE_IMPLEMENTATION
    {
      TV_THROW_RT_ERR("you must include this in a source file with TV_STATIC_VARIABLE_IMPLEMENTATION defined");
    }
#else 
    ;
#endif
    ~AppleMetalContext() {
      if (device_ptr_) {
          device_ptr_->release();
          device_ptr_ = nullptr;
      }
      if (command_queue_ptr_) {
          command_queue_ptr_->release();
          command_queue_ptr_ = nullptr;
      }
      if (!from_blob_){
        if (dq_ptr_){
          dispatch_release(dq_ptr_);
        }
      }
    }
    bool is_from_blob() const { return from_blob_; }
    MTL::Device* device() const {return device_ptr_;}
    dispatch_queue_t queue() const { return dq_ptr_; }
    MTL::CommandQueue* commandQueue() const { 
      TV_ASSERT_RT_ERR(command_queue_ptr_, "you can't get command queue when context is from blob");
      return command_queue_ptr_; 
    }

    MTL::CommandBuffer* commandBuffer() {
      if (from_blob_){
        return command_buffer_ptr_external_;
      }
      if (!_commandBuffer) {
        _commandBuffer = NS::RetainPtr(command_queue_ptr_->commandBuffer());
      }
      return _commandBuffer.get();
    }
    void synchronize(SyncType syncType) {
      TV_ASSERT_RT_ERR(!from_blob_, "you can't synchronize when context is from blob");
      // endKernelCoalescing();
      switch (syncType) {
        case SyncType::NONE:
          // typically in GPU to GPU copies we won't commit explicitly
          break;
        case SyncType::COMMIT:
          commit();
          break;
        case SyncType::COMMIT_AND_WAIT:
          commitAndWait();
          break;
        case SyncType::COMMIT_AND_CONTINUE:
          TV_ASSERT_RT_ERR(_enableCommitAndContinue,
                                          "CommitAndContinue is called but it is disabled globally!");
          commitAndContinue();
          break;
      }
    }   

    void commit() {
      TV_ASSERT_RT_ERR(!from_blob_, "you can't synchronize when context is from blob");
      if (_enableCommitAndContinue) {
        commitAndContinue();
      } else {
        flush();
      }
    }

    void flush() {
      TV_ASSERT_RT_ERR(!from_blob_, "you can't synchronize when context is from blob");
      if (_commandBuffer) {
        _commandBuffer->commit();
        // if commitAndContinue is disabled (e.g., for Profiler), we keep the command
        // buffer so we could wait on it later, if required.
        if (!_enableCommitAndContinue) {
          _prevCommandBuffer = _commandBuffer;
        } else {
          _commandBuffer.reset();
        }
        _commandBuffer = NS::SharedPtr<MTL::CommandBuffer>{};
      }
    }
    void commitAndContinue() {
      TV_ASSERT_RT_ERR(!from_blob_, "you can't synchronize when context is from blob");
      TV_ASSERT_RT_ERR(_commandBuffer, "error");
      _commandBuffer->commit();
      _commandBuffer = NS::RetainPtr(command_queue_ptr_->commandBuffer());
    }
    void commitAndWait() {
      TV_ASSERT_RT_ERR(!from_blob_, "you can't synchronize when context is from blob");
      if (_prevCommandBuffer) {
        // the previous command buffer (if exists) has already been committed,
        // so we just wait until it's completed and then dispose it.
        _prevCommandBuffer->waitUntilCompleted();
        _prevCommandBuffer = NS::SharedPtr<MTL::CommandBuffer>{};
      }

      if (_commandBuffer) {
        _commandBuffer->commit();
        _commandBuffer->waitUntilCompleted();
        _commandBuffer = NS::SharedPtr<MTL::CommandBuffer>{};
      }
    }
    void copy(MTL::Buffer* src, MTL::Buffer* dst, size_t src_offset, size_t dst_offset, size_t length, SyncType sync_type) {
      TV_ASSERT_INVALID_ARG(src && dst, "Metal buffer is null");
      auto blitEncoder = commandBuffer()->blitCommandEncoder();
      // For some reason copyFromBuffer for 4Gb fails without returning an error
      // See https://github.com/pytorch/pytorch/issues/124335
      // Workaround by batching copy commands into 2Gb chunks
      constexpr size_t max_copy_size = 0x80000000; // 2GB
      size_t bytes_copied = 0;
      size_t bytes_remains = length;
      while (bytes_remains > 0) {
        size_t bytes_to_copy = std::min(max_copy_size, bytes_remains);
        blitEncoder->copyFromBuffer(src, src_offset + bytes_copied, dst, dst_offset + bytes_copied, bytes_to_copy);
        bytes_copied += bytes_to_copy;
        bytes_remains -= bytes_to_copy;
      }
      blitEncoder->endEncoding();
      synchronize(sync_type);
    }
    void copy_src_raw(const void* src, size_t src_size, MTL::Buffer* dst, size_t src_offset, size_t dst_offset, size_t size, SyncType sync_type) {
      TV_ASSERT_INVALID_ARG(src && dst, "Metal buffer is null");

      auto res = detail::create_apple_buffer_from_cpu_ptr(src, src_size,
                                                  device_ptr_);
      auto ptr_mtl = detail::make_apple_mtl_ptr(std::get<0>(res));
      auto boffset = std::get<1>(res);
      return copy(ptr_mtl.get(), dst, boffset + src_offset, dst_offset, size, sync_type);
    }

    void copy_dst_raw(MTL::Buffer* src, void* dst, size_t dst_size, size_t src_offset, size_t dst_offset, size_t size, SyncType sync_type) {
      TV_ASSERT_INVALID_ARG(src && dst, "Metal buffer is null");

      auto res = detail::create_apple_buffer_from_cpu_ptr(dst, dst_size,
                                                  device_ptr_);
      auto ptr_mtl = detail::make_apple_mtl_ptr(std::get<0>(res));
      auto boffset = std::get<1>(res);
      return copy(src, ptr_mtl.get(), src_offset, boffset + dst_offset, size, sync_type);
    }

    void fill(MTL::Buffer* buffer, size_t offset, size_t size, uint8_t value, SyncType sync_type) {
      TV_ASSERT_INVALID_ARG(buffer, "Metal buffer is null");

      auto blitEncoder = commandBuffer()->blitCommandEncoder();
      blitEncoder->fillBuffer(buffer, NS::Range::Make(offset,
                                        size), value);
      blitEncoder->endEncoding();
      synchronize(sync_type);
    }

    void fill_raw(void* buffer, size_t offset, size_t size, uint8_t value, SyncType sync_type) {
      TV_ASSERT_INVALID_ARG(buffer, "Metal buffer is null");
      auto res = detail::create_apple_buffer_from_cpu_ptr(buffer, size,
                                                  device_ptr_);
      auto ptr_mtl = detail::make_apple_mtl_ptr(std::get<0>(res));
      auto boffset = std::get<1>(res);
      return fill(ptr_mtl.get(), boffset + offset, size, value, sync_type);
    }

};

#endif

namespace detail {

struct ContextValue {
  std::uintptr_t ptr_int;
  bool from_blob;
  void(*deleter)(std::uintptr_t) = nullptr;
};

struct ContextManager {
  // windows nvcc have bug when use std::function
  std::uintptr_t (*creater)();
  void (*deleter)(std::uintptr_t);
  // std::function<void(std::uintptr_t)> deleter;
};

struct ContextTypeHash {
  template <typename T> int operator()(T t) const {
    return static_cast<int>(t);
  }
};

struct ContextCore {
private:
  std::unordered_map<ContextType, ContextValue, ContextTypeHash> ctx_ptrs_;
  std::unordered_map<ContextType, ContextManager, ContextTypeHash> ctx_mgrs_;

public:
  ContextCore() {
#if defined(TV_HARDWARE_ACC_CUDA)
    ContextManager stream_mgr;
    stream_mgr.creater = []() -> std::uintptr_t {
      cudaStream_t stream_cuda;
      checkCudaErrors(cudaStreamCreate(&stream_cuda));
      return reinterpret_cast<std::uintptr_t>(stream_cuda);
    };
    stream_mgr.deleter = [](std::uintptr_t ptr_int) {
      cudaStreamDestroy(reinterpret_cast<cudaStream_t>(ptr_int));
    };
    ctx_mgrs_[ContextType::kCudaStream] = stream_mgr;
#endif
#ifdef TV_HARDWARE_ACC_METAL
    ContextManager mtl_compute_ctx_mgr;
    mtl_compute_ctx_mgr.creater = []() -> std::uintptr_t {
      auto* ptr = new AppleMetalContext();
      return reinterpret_cast<std::uintptr_t>(ptr);
    };
    mtl_compute_ctx_mgr.deleter = [](std::uintptr_t ptr_int) {
      auto* ptr = reinterpret_cast<AppleMetalContext*>(ptr_int);
      delete ptr;
    };
    ctx_mgrs_[ContextType::kAppleMetal] = mtl_compute_ctx_mgr;
#ifdef TV_STATIC_VARIABLE_IMPLEMENTATION
    create_raw_item(ContextType::kAppleMetal, reinterpret_cast<std::uintptr_t>(AppleMetalContext::getInstance().get()));
#endif
#endif

  }
  void register_manager(ContextType type, ContextManager mgr) {
    TV_ASSERT_INVALID_ARG(ctx_mgrs_.find(type) == ctx_mgrs_.end(),
                          "manager exists");
    ctx_mgrs_[type] = mgr;
  }

  void create_item(ContextType type) {
    TV_ASSERT_RT_ERR(ctx_ptrs_.find(type) == ctx_ptrs_.end(),
                     "context item exists");
    TV_ASSERT_RT_ERR(
        ctx_mgrs_.find(type) != ctx_mgrs_.end(),
        "can't find context manager. call register_manager before.");

    ctx_ptrs_[type] = ContextValue{ctx_mgrs_[type].creater(), false};
  }

  void create_raw_item(ContextType type, std::uintptr_t handle, void(*deleter)(std::uintptr_t) = nullptr) {
    TV_ASSERT_RT_ERR(ctx_ptrs_.find(type) == ctx_ptrs_.end(),
                     "context item exists");
    ctx_ptrs_[type] = ContextValue{handle, true, deleter};
  }

  bool has_item(ContextType type) {
    return ctx_ptrs_.find(type) != ctx_ptrs_.end();
  }

  void remove_item(ContextType type) {
    auto ptr = ctx_ptrs_.find(type);
    if (ptr != ctx_ptrs_.end()) {
      if (!ptr->second.from_blob) {
        ctx_mgrs_[type].deleter(ptr->second.ptr_int);
      }
      ctx_ptrs_.erase(ptr);
    }
  }

  std::uintptr_t get_item(ContextType type) {
    auto ptr = ctx_ptrs_.find(type);
    if (ptr != ctx_ptrs_.end()) {
      return ptr->second.ptr_int;
    }
    return 0;
  }

  ~ContextCore() {
    for (auto &pair : ctx_ptrs_) {
      if (!pair.second.from_blob) {
        ctx_mgrs_[pair.first].deleter(pair.second.ptr_int);
      }else{
        if (pair.second.deleter){
          pair.second.deleter(pair.second.ptr_int);
        }
      }
    }
  }
};

} // namespace detail

struct Context {
protected:
  std::shared_ptr<detail::ContextCore> context_ptr_;
  void check_ptr_valid() {
    TV_ASSERT_RT_ERR(bool(context_ptr_), "context ptr must not empty");
  }

public:
  Context() : context_ptr_(std::make_shared<detail::ContextCore>()) {}

  bool has_cuda_stream() {
    check_ptr_valid();
    return context_ptr_->has_item(ContextType::kCudaStream);
  }

  Context &create_cuda_stream() {
    check_ptr_valid();
    if (context_ptr_->has_item(ContextType::kCudaStream)){
      context_ptr_->remove_item(ContextType::kCudaStream);
    }
    context_ptr_->create_item(ContextType::kCudaStream);
    return *this;
  }

#if defined(TV_HARDWARE_ACC_CUDA)
  Context &set_cuda_stream(cudaStream_t stream) {
    check_ptr_valid();
    context_ptr_->create_raw_item(ContextType::kCudaStream,
                                  reinterpret_cast<std::uintptr_t>(stream));
    return *this;
  }
  cudaStream_t cuda_stream() {
    check_ptr_valid();
    return reinterpret_cast<cudaStream_t>(cuda_stream_int());
  }
#endif

  Context &set_cuda_stream_int(std::uintptr_t stream) {
#if defined(TV_HARDWARE_ACC_CUDA)
    check_ptr_valid();
    context_ptr_->create_raw_item(ContextType::kCudaStream,
                                  stream);
#endif
    return *this;
  }
  void synchronize_stream() {
#if defined(TV_HARDWARE_ACC_CUDA)
    check_ptr_valid();
    checkCudaErrors(cudaStreamSynchronize(cuda_stream()));
#endif
  }
  void synchronize() {
#if defined(TV_HARDWARE_ACC_CUDA)
    check_ptr_valid();
    checkCudaErrors(cudaStreamSynchronize(cuda_stream()));
#else 
#ifdef TV_HARDWARE_ACC_METAL
    check_ptr_valid();
    if (has_item(ContextType::kAppleMetal)) {
      auto* ptr = reinterpret_cast<AppleMetalContext*>(context_ptr_->get_item(ContextType::kAppleMetal));
      ptr->synchronize(AppleMetalContext::SyncType::COMMIT_AND_WAIT);
    }
#endif
#endif
  }

  std::uintptr_t cuda_stream_int() {
    check_ptr_valid();
#if defined(TV_HARDWARE_ACC_CUDA)
    return get_item(ContextType::kCudaStream);
#else 
    return 0;
#endif 
  }
  bool has_item(ContextType type) {
    check_ptr_valid();
    return context_ptr_->has_item(type);
  }
  std::uintptr_t get_item(ContextType type) {
    check_ptr_valid();
    return context_ptr_->get_item(type);
  }
  Context &create_apple_metal_context() {
#ifdef TV_HARDWARE_ACC_METAL

    check_ptr_valid();
    if (context_ptr_->has_item(ContextType::kAppleMetal)){
      context_ptr_->remove_item(ContextType::kAppleMetal);
    }
    context_ptr_->create_item(ContextType::kAppleMetal);
    return *this;
#else 
    TV_THROW_INVALID_ARG("Apple Metal is not supported in non-apple platform");
#endif
  }

  Context &create_or_update_metal_context_from_blob(std::uintptr_t cb, std::uintptr_t dq) {
#ifdef TV_HARDWARE_ACC_METAL
    auto deleter = [](std::uintptr_t ptr_int) {
      auto* ptr = reinterpret_cast<AppleMetalContext*>(ptr_int);
      delete ptr;
    };
    auto ctx = new AppleMetalContext(reinterpret_cast<MTL::CommandBuffer*>(cb), reinterpret_cast<dispatch_queue_t>(dq));
    check_ptr_valid();
    if (context_ptr_->has_item(ContextType::kAppleMetal)){
      AppleMetalContext* ptr = reinterpret_cast<AppleMetalContext*>(context_ptr_->get_item(ContextType::kAppleMetal));
      if (!ptr->is_from_blob()){
        context_ptr_->remove_item(ContextType::kAppleMetal);
      }else{
        ptr->update_context_from_blob(reinterpret_cast<MTL::CommandBuffer*>(cb), reinterpret_cast<dispatch_queue_t>(dq));
        return *this;
      }
    }
    context_ptr_->create_raw_item(ContextType::kAppleMetal, reinterpret_cast<std::uintptr_t>(ctx), deleter);
    return *this;
#else 
    TV_THROW_INVALID_ARG("Apple Metal is not supported in non-apple platform");
#endif
  }

};
} // namespace tv