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

#include <memory>
#include <string>
#include <tensorview/tensor.h>

#include <unordered_map>
#include <vector>
#ifdef TV_HARDWARE_ACC_METAL
#include "Metal/Metal.hpp"
#endif
#include <tensorview/cuda/nvrtc.h>

namespace tv {
namespace detail {
#ifdef TV_HARDWARE_ACC_METAL
inline MTL::DataType tv_dtype_to_mtl_data_type(tv::DType t, int cnt) {
  switch (cnt){
    case 1: {
      switch (t) {
      case DType::bool_:
        return MTL::DataType::DataTypeBool;
      case DType::float32:
        return MTL::DataType::DataTypeFloat;
      case DType::int8:
        return MTL::DataType::DataTypeChar;
      case DType::int16:
        return MTL::DataType::DataTypeShort;
      case DType::int32:
        return MTL::DataType::DataTypeInt;
      case DType::int64:
        return MTL::DataType::DataTypeLong;
      case DType::uint8:
        return MTL::DataType::DataTypeUChar;
      case DType::uint16:
        return MTL::DataType::DataTypeUShort;
      case DType::uint32:
        return MTL::DataType::DataTypeUInt;
      case DType::uint64:
        return MTL::DataType::DataTypeULong;
      case DType::float16:
        return MTL::DataType::DataTypeHalf;
      case DType::bfloat16:
        return MTL::DataType::DataTypeBFloat;

      default:
        TV_THROW_INVALID_ARG("not implemented dtype: ", dtype_str(t));
      }

      break;
    }
    case 2: {
      switch (t) {
      case DType::float32:
        return MTL::DataType::DataTypeFloat2;
      case DType::int8:
        return MTL::DataType::DataTypeChar2;
      case DType::int16:
        return MTL::DataType::DataTypeShort2;
      case DType::int32:
        return MTL::DataType::DataTypeInt2;
      case DType::int64:
        return MTL::DataType::DataTypeLong2;
      case DType::uint8:
        return MTL::DataType::DataTypeUChar2;
      case DType::uint16:
        return MTL::DataType::DataTypeUShort2;
      case DType::uint32:
        return MTL::DataType::DataTypeUInt2;
      case DType::uint64:
        return MTL::DataType::DataTypeULong2;
      case DType::float16:
        return MTL::DataType::DataTypeHalf2;
      case DType::bfloat16:
        return MTL::DataType::DataTypeBFloat2;

      default:
        TV_THROW_INVALID_ARG("not implemented dtype: ", dtype_str(t));
      }
      break;
    }
    case 3: {
      switch (t) {
      case DType::float32:
        return MTL::DataType::DataTypeFloat3;
      case DType::int8:
        return MTL::DataType::DataTypeChar3;
      case DType::int16:
        return MTL::DataType::DataTypeShort3;
      case DType::int32:
        return MTL::DataType::DataTypeInt3;
      case DType::int64:
        return MTL::DataType::DataTypeLong3;
      case DType::uint8:
        return MTL::DataType::DataTypeUChar3;
      case DType::uint16:
        return MTL::DataType::DataTypeUShort3;
      case DType::uint32:
        return MTL::DataType::DataTypeUInt3;
      case DType::uint64:
        return MTL::DataType::DataTypeULong3;
      case DType::float16:
        return MTL::DataType::DataTypeHalf3;
      case DType::bfloat16:
        return MTL::DataType::DataTypeBFloat3;

      default:
        TV_THROW_INVALID_ARG("not implemented dtype: ", dtype_str(t));
      }
      break;
    }
    case 4: {
      switch (t) {
      case DType::float32:
        return MTL::DataType::DataTypeFloat4;
      case DType::int8:
        return MTL::DataType::DataTypeChar4;
      case DType::int16:
        return MTL::DataType::DataTypeShort4;
      case DType::int32:
        return MTL::DataType::DataTypeInt4;
      case DType::int64:
        return MTL::DataType::DataTypeLong4;
      case DType::uint8:
        return MTL::DataType::DataTypeUChar4;
      case DType::uint16:
        return MTL::DataType::DataTypeUShort4;
      case DType::uint32:
        return MTL::DataType::DataTypeUInt4;
      case DType::uint64:
        return MTL::DataType::DataTypeULong4;
      case DType::float16:
        return MTL::DataType::DataTypeHalf4;
      case DType::bfloat16:
        return MTL::DataType::DataTypeBFloat4;
      default:
        TV_THROW_INVALID_ARG("not implemented dtype: ", dtype_str(t));
      }
      break;
    }
    default:
      TV_THROW_INVALID_ARG("not implemented cnt: ", cnt);
  }
}

#endif
} // namespace detail
class MetalModule {
public:
  using ArgType = NVRTCModule::ArgType;
  MetalModule(tv::Tensor binary) {
#ifdef TV_HARDWARE_ACC_METAL
    ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
    auto data = dispatch_data_create(binary.const_raw_data(), binary.raw_size(),
                                     dispatch_get_main_queue(),
                                     DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *error = nullptr;
    mtl_lib_ = ptr_mtl_device_->newLibrary(data, &error);
#else
    TV_THROW_RT_ERR("you must compile with apple first to use metal program");
#endif
  }
  MetalModule(std::string code, std::unordered_map<std::string, std::string> preprocessorMacros, bool fastMathEnabled = true) {
#ifdef TV_HARDWARE_ACC_METAL
    ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
    auto options =
        detail::make_apple_mtl_ptr(MTL::CompileOptions::alloc()->init());
    NS::SharedPtr< NS::AutoreleasePool > pPool = NS::TransferPtr( NS::AutoreleasePool::alloc()->init() );
    auto codeNS = NS::String::string(code.c_str(), NS::ASCIIStringEncoding);
    std::vector<NS::Object* > macroKeys;
    std::vector<NS::Object* > macroValues;
    for (auto &pair : preprocessorMacros) {
      macroKeys.push_back(NS::String::string(pair.first.c_str(), NS::ASCIIStringEncoding));
      macroValues.push_back(NS::String::string(pair.second.c_str(), NS::ASCIIStringEncoding));
    }
    auto nd = NS::Dictionary::dictionary(macroKeys.data(), macroValues.data(), macroKeys.size());
    options->setPreprocessorMacros(nd);
    if (!fastMathEnabled){
      // default is yes, so only set if false
      options->setFastMathEnabled(fastMathEnabled);
    }
    NS::Error *error = nullptr;
    mtl_lib_ = ptr_mtl_device_->newLibrary(codeNS, options.get(), &error);
    TV_ASSERT_INVALID_ARG(mtl_lib_ && error == nullptr,
                          "new library failed.");

#else
    TV_THROW_RT_ERR("you must compile with apple first to use metal program");
#endif
  }
  void run_kernel(std::string name, std::array<int, 3> blocks,
                  std::array<int, 3> threads, int smem_size, Context ctx,
                  std::vector<std::tuple<tv::Tensor, int, std::uintptr_t, size_t>> args,
                  bool use_nonuniform_threadgroup = true) {
#ifdef TV_HARDWARE_ACC_METAL
    bool sync_op = false;
    if (!ctx.has_item(ContextType::kAppleMetal)) {
      ctx = Context().create_apple_metal_context();
      sync_op = true;
    }
    TV_ASSERT_INVALID_ARG(
        ctx.has_item(ContextType::kAppleMetal),
        "you must use a context with metal created explicitly");
    auto ctx_ptr = reinterpret_cast<AppleMetalContext *>(
        ctx.get_item(ContextType::kAppleMetal));
    auto cb = ctx_ptr->commandBuffer();
    TV_ASSERT_RT_ERR(cb, "command buffer is null");
    int cnt = 0;
    std::string cache_ley = name;
    for (auto &arg : args) {
      auto &ten = std::get<0>(arg);
      auto arg_type = std::get<1>(arg);
      switch (arg_type) {
      case ArgType::kConstant: {
        TV_ASSERT_INVALID_ARG(ten.device() == -1 && ten.size() == 1 && ten.dtype() == tv::uint8,
                              "array tensor must be CPU and scalar and uint8 (bool)");
        // auto mtl_dtype = detail::tv_dtype_to_mtl_data_type(ten.dtype(), ten.dim(0));
        auto mtl_dtype = MTL::DataType::DataTypeBool;
        bool val = ten.item<uint8_t>() > 0;
        cache_ley += std::string("_") + std::to_string(ten.item<uint8_t>());
        cnt += 1;
        break;
      }
      default:;
      }
    }
    if (func_pso_map_.find(cache_ley) == func_pso_map_.end()) {
      auto constants = detail::make_apple_mtl_ptr(
          MTL::FunctionConstantValues::alloc()->init());
      TV_ASSERT_RT_ERR(constants, "command buffer is null");
      for (auto &arg : args) {
        auto &ten = std::get<0>(arg);
        auto arg_type = std::get<1>(arg);
        switch (arg_type) {
        case ArgType::kConstant: {
          TV_ASSERT_INVALID_ARG(ten.device() == -1 && ten.size() == 1 && ten.dtype() == tv::uint8,
                                "array tensor must be CPU and scalar and uint8 (bool)");
          // auto mtl_dtype = detail::tv_dtype_to_mtl_data_type(ten.dtype(), ten.dim(0));
          auto mtl_dtype = MTL::DataType::DataTypeBool;
          bool val = ten.item<uint8_t>() > 0;
          constants->setConstantValue(&val, mtl_dtype, cnt);
          cache_ley += std::string("_") + std::to_string(ten.item<uint8_t>());
          cnt += 1;
          break;
        }
        default:;
        }
      }
      auto nameNS = detail::make_apple_mtl_ptr(
          NS::String::string(name.c_str(), NS::ASCIIStringEncoding)->retain());
      NS::Error *error = nullptr;
      auto computeFunction = NS::TransferPtr(
          mtl_lib_->newFunction(nameNS.get(), constants.get(), &error));
      TV_ASSERT_INVALID_ARG(computeFunction && error == nullptr,
                            "can't find function", name);
      error = nullptr;
      auto func_pso =
          NS::TransferPtr(ptr_mtl_device_->newComputePipelineState(
              computeFunction.get(), &error));
      TV_ASSERT_INVALID_ARG(func_pso && error == nullptr, "can't create pso",
                            name);
      func_pso_map_[cache_ley] = std::make_tuple(computeFunction, func_pso);
    }
    auto pso_ptr = std::get<1>(func_pso_map_[cache_ley]).get();

    dispatch_sync(ctx_ptr->queue(), ^() {
      bool ce_is_end_encoding = false;
      auto deleter = [&ce_is_end_encoding](MTL::ComputeCommandEncoder *p) {
                if (!ce_is_end_encoding)
                  p->endEncoding();
                p->release();
              };
      auto computeEncoder =
          std::unique_ptr<MTL::ComputeCommandEncoder, decltype(deleter)>(
              cb->computeCommandEncoder()->retain(), deleter);
      TV_ASSERT_RT_ERR(computeEncoder, "compute encoder is null");
      int buffer_cnt = 0;

      for (auto &arg : args) {
        auto &ten = std::get<0>(arg);
        auto arg_type = std::get<1>(arg);
        // tv::ssprint(ten.shape(), ten.dtype(), arg_type);
        switch (arg_type) {
        case ArgType::kDevicePointer: {
          auto ptr = reinterpret_cast<MTL::Buffer *>(std::get<2>(arg));
          auto offset = std::get<3>(arg);
            computeEncoder->setBuffer(
                reinterpret_cast<MTL::Buffer *>(
                    ptr),
                offset, buffer_cnt);
          buffer_cnt += 1;
          break;
        }

        case ArgType::kTensor: {
          if (ten.empty()) {
          } else {
            TV_ASSERT_INVALID_ARG(ten.device() == 0, "tensor must be GPU");
            computeEncoder->setBuffer(
                reinterpret_cast<MTL::Buffer *>(
                    ten.storage()->apple_metal_buffer_ptr()),
                ten.byte_offset() + ten.storage()->byte_offset(), buffer_cnt);
          }
          buffer_cnt += 1;
          break;
        }
        case ArgType::kScalar:
        case ArgType::kArray: {
          TV_ASSERT_INVALID_ARG(ten.device() == -1, "array tensor must be CPU");
          TV_ASSERT_INVALID_ARG(ten.storage()->apple_metal_buffer_ptr() == nullptr,
                                "array metal buffer must be empty");
          computeEncoder->setBytes(ten.raw_data(), ten.raw_size(), buffer_cnt);
          buffer_cnt += 1;
          break;
        }
        case ArgType::kConstant: {
          break;
        }
        case ArgType::kTensorView: {
          TV_THROW_INVALID_ARG("not implemented");
          break;
        }
        default:
          TV_THROW_RT_ERR("not implemented");
        }
      }
      computeEncoder->setComputePipelineState(pso_ptr);

      MTL::Size threadgroupSize = MTL::Size(threads[0], threads[1], threads[2]);
      MTL::Size gridSize = MTL::Size(blocks[0], blocks[1], blocks[2]);
      // Encode the compute command.
      if (use_nonuniform_threadgroup){
        computeEncoder->dispatchThreads(gridSize, threadgroupSize);
      }else{
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
      }
      computeEncoder->endEncoding();
      ce_is_end_encoding = true;
      // if from blob (use external context), do sync with external library by
      // user.
      if (!ctx_ptr->is_from_blob()) {
        // Execute the command.
        if (sync_op) {
          ctx_ptr->synchronize(AppleMetalContext::SyncType::COMMIT_AND_WAIT);
        } else {
          ctx_ptr->synchronize(AppleMetalContext::SyncType::COMMIT);
        }
      }
    });
#endif
  }

  ~MetalModule() {
#ifdef TV_HARDWARE_ACC_METAL
    if (mtl_lib_) {
      mtl_lib_->release();
      mtl_lib_ = nullptr;
    }
    if (ptr_mtl_device_) {
      ptr_mtl_device_->release();
      ptr_mtl_device_ = nullptr;
    }
#endif
  }

private:
#ifdef TV_HARDWARE_ACC_METAL
  MTL::Device *ptr_mtl_device_ = nullptr;
  MTL::Library *mtl_lib_ = nullptr;
  std::unordered_map<std::string, std::tuple<NS::SharedPtr<MTL::Function>, NS::SharedPtr<MTL::ComputePipelineState>>> func_pso_map_;

#endif
};

} // namespace tv
