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
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuda.h>
#include <nvrtc.h>
#include <tensorview/cuda/driver.h>
#endif

#if !(defined(__CUDA__) || defined(__NVCC__))
#include <tensorview/io/jsonarray.h>
#include <tensorview/thirdparty/nlohmann/json.hpp>
#endif

#define TV_NVRTC_SAFE_CALL(x)                                                  \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      TV_THROW_RT_ERR("error: " #x " failed with error ",                      \
                      nvrtcGetErrorString(result));                            \
    }                                                                          \
  } while (0);

namespace tv {

class NVRTCProgram {
public:
  enum SerializationType { kSource = 0, kPTX = 1, kCuBin = 2 };
  NVRTCProgram(std::string code,
               std::unordered_map<std::string, std::string> headers = {},
               std::vector<std::string> opts = {},
               std::string program_name = "kernel",
               std::vector<std::string> name_exprs = {})
      : code_(code), headers_(headers), program_name_(program_name + ".cu"),
        name_exprs_(name_exprs), opts_(opts) {
    std::vector<const char *> header_buffers;
    std::vector<const char *> header_names;
    std::vector<const char *> opts_ptrs;
    for (auto &opt : opts) {
      opts_ptrs.push_back(opt.c_str());
    }
    for (auto &pair : headers_) {
      header_names.push_back(pair.first.c_str());
      header_buffers.push_back(pair.second.c_str());
    }
    const char *const *header_ptr = nullptr;
    const char *const *header_name_ptr = nullptr;
    if (headers_.size() > 0) {
      header_ptr = header_buffers.data();
      header_name_ptr = header_names.data();
    }
#if defined(TV_HARDWARE_ACC_CUDA)
    TV_NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog_,                // prog
                                          code_.c_str(),         // buffer
                                          program_name_.c_str(), // name
                                          headers_.size(),       // numHeaders
                                          header_ptr,            // headers
                                          header_name_ptr));     // includeNames
    for (size_t i = 0; i < name_exprs_.size(); ++i) {
      TV_NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog_, name_exprs_[i].c_str()));
    }

    nvrtcResult compileResult =
        nvrtcCompileProgram(prog_,             // prog
                            opts.size(),       // numOptions
                            opts_ptrs.data()); // options

    size_t logSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog_, &logSize));
    std::string log(logSize, '0');
    auto nvrtc_compile_res = nvrtcGetProgramLog(prog_, &log[0]);
    // if (compileResult != NVRTC_SUCCESS) {
    //   tv::ssprint(log);
    // }
    TV_NVRTC_SAFE_CALL(nvrtc_compile_res);
    compile_log_ = log;
    TV_ASSERT_RT_ERR(compileResult == NVRTC_SUCCESS, "nvrtc compile failed. log: \n", log);
    // post check
    predefined_name_expr_map_.clear();
    for (size_t i = 0; i < name_exprs_.size(); ++i) {
      predefined_name_expr_map_[name_exprs_[i]] =
          get_lowered_name(name_exprs_[i]);
    }

#else
    TV_THROW_RT_ERR("you must compile with CUDA first to use nvrtc program");
#endif
  }

  NVRTCProgram(std::unordered_map<std::string, std::string> predefined_name_map,
               std::string ptx)
      : ptx_(ptx), predefined_name_expr_map_(predefined_name_map),
        serial_type_(kPTX) {}

  NVRTCProgram(std::unordered_map<std::string, std::string> predefined_name_map,
               tv::Tensor cubin)
      : cubin_(cubin), predefined_name_expr_map_(predefined_name_map),
        serial_type_(kCuBin) {}

  static std::shared_ptr<NVRTCProgram>
  create(std::string code,
         std::unordered_map<std::string, std::string> headers = {},
         std::vector<std::string> opts = {},
         std::string program_name = "kernel",
         std::vector<std::string> name_exprs = {}) {
    return std::make_shared<NVRTCProgram>(code, headers, opts, program_name,
                                          name_exprs);
  }
#if !(defined(__CUDA__) || defined(__NVCC__))
  static std::shared_ptr<NVRTCProgram> from_string(std::string json_string) {
    nlohmann::json j = nlohmann::json::parse(json_string);
    return std::make_shared<NVRTCProgram>(
        j["code"].get<std::string>(),
        j["headers"].get<std::unordered_map<std::string, std::string>>(),
        j["opts"].get<std::vector<std::string>>(),
        j["program_name"].get<std::string>(),
        j["name_exprs"].get<std::vector<std::string>>());
  }

  std::string to_string() const {
    TV_ASSERT_RT_ERR(serial_type_ == kSource,
                     "only kSource program can be converted to string")
    nlohmann::json j;
    j["type"] = kSource;
    j["code"] = code_;
    j["headers"] = headers_;
    j["opts"] = opts_;
    j["program_name"] = program_name_;
    j["name_exprs"] = name_exprs_;
    return j.dump();
  }

  std::vector<uint8_t> to_binary(SerializationType type) const {
    tv::io::JsonArray jarr;
    if (type == kSource) {
      jarr.data["type"] = static_cast<int>(kSource);
      jarr.data["code"] = code_;
      jarr.data["headers"] = headers_;
      jarr.data["opts"] = opts_;
      jarr.data["program_name"] = program_name_;
      jarr.data["name_exprs"] = name_exprs_;
    } else if (type == kPTX) {
      jarr.data["type"] = static_cast<int>(kPTX);
      jarr.data["ptx"] = ptx();
      jarr.data["name_map"] = get_predefined_lowered_name_map();
    } else {
      jarr.data["type"] = static_cast<int>(kCuBin);
      jarr.assign("cubin", cubin());
      jarr.data["name_map"] = get_predefined_lowered_name_map();
    }
    return tv::io::encode(jarr.tensors, jarr.data);
  }

  static std::shared_ptr<NVRTCProgram> from_binary(const uint8_t *buffer,
                                                   size_t size) {
    auto jarr = tv::io::decode(buffer, size);
    SerializationType type =
        static_cast<SerializationType>(jarr.data["type"].get<int>());
    if (type == kSource) {
      return std::make_shared<NVRTCProgram>(
          jarr.data["code"].get<std::string>(),
          jarr.data["headers"]
              .get<std::unordered_map<std::string, std::string>>(),
          jarr.data["opts"].get<std::vector<std::string>>(),
          jarr.data["program_name"].get<std::string>(),
          jarr.data["name_exprs"].get<std::vector<std::string>>());
    } else if (type == kPTX) {
      return std::make_shared<NVRTCProgram>(
          jarr.data["name_map"]
              .get<std::unordered_map<std::string, std::string>>(),
          jarr.data["ptx"].get<std::string>());
    } else {
      return std::make_shared<NVRTCProgram>(
          jarr.data["name_map"]
              .get<std::unordered_map<std::string, std::string>>(),
          jarr.tensors[0]);
    }
  }
#endif
  std::unordered_map<std::string, std::string>
  get_predefined_lowered_name_map() const {
    std::unordered_map<std::string, std::string> res;
    for (size_t i = 0; i < name_exprs_.size(); ++i) {
      res[name_exprs_[i]] = get_lowered_name(name_exprs_[i]);
    }
    return res;
  }

  ~NVRTCProgram() {
#if defined(TV_HARDWARE_ACC_CUDA)
    if (prog_) {
      nvrtcDestroyProgram(&prog_);
    }
#endif
  }

  std::string ptx() const {
    if (!ptx_.empty()){
      return ptx_;
    }
#if defined(TV_HARDWARE_ACC_CUDA)
    if (prog_ == nullptr) {
      TV_ASSERT_RT_ERR(!ptx_.empty(), "PTX is empty!!!");
      return ptx_;
    }
    size_t ptxSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog_, &ptxSize));
    std::string ptx(ptxSize, '0');
    TV_NVRTC_SAFE_CALL(nvrtcGetPTX(prog_, &ptx[0]));

    return ptx;
#else
    return "";
#endif
  }

  tv::Tensor cubin() const {
    if (!cubin_.empty()){
      return cubin_;
    }
#if defined(TV_HARDWARE_ACC_CUDA)
#if (CUDA_VERSION < 11000)
    TV_THROW_RT_ERR("cubin not implemented for CUDA < 11");
#else 
    if (prog_ == nullptr) {
      TV_ASSERT_RT_ERR(!cubin_.empty(), "Cubin is empty!!!");
      return cubin_;
    }
    size_t cubinSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog_, &cubinSize));
    tv::Tensor bin({int64_t(cubinSize)}, tv::uint8, -1);
    TV_NVRTC_SAFE_CALL(nvrtcGetCUBIN(
        prog_, reinterpret_cast<char *>(bin.data_ptr<uint8_t>())));
    return bin;
#endif
#else
    return tv::Tensor();
#endif
  }

  std::string compile_log() const { return compile_log_; }

  std::string program_name() const { return program_name_; }

  std::vector<std::string> name_exprs() const { return name_exprs_; }

  std::string get_lowered_name(std::string name) const {
#if defined(TV_HARDWARE_ACC_CUDA)
    if (prog_ == nullptr) {
      TV_ASSERT_RT_ERR(predefined_name_expr_map_.find(name) !=
                           predefined_name_expr_map_.end(),
                       "can't find your name");
      return predefined_name_expr_map_.at(name);
    }
    const char *lowered_name;
    TV_NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog_,
                                           name.c_str(), // name expression
                                           &lowered_name // lowered name
                                           ));
    return std::string(lowered_name);
#else
    return "";
#endif
  }

private:
#if defined(TV_HARDWARE_ACC_CUDA)
  nvrtcProgram prog_ = nullptr;
#endif
  std::string code_;
  std::string compile_log_;
  // used if program is loaded from binary data or ptx.
  std::string ptx_;
  tv::Tensor cubin_;
  SerializationType serial_type_ = kSource;
  std::unordered_map<std::string, std::string> headers_;
  std::string program_name_;
  std::vector<std::string> name_exprs_;
  std::unordered_map<std::string, std::string> predefined_name_expr_map_;
  std::vector<std::string> opts_;
};

class NVRTCModule {
public:
  enum ArgType { kTensor = 0, kArray = 1, kTensorView = 2, kScalar = 3, kConstant = 4, kDevicePointer = 5 };

  NVRTCModule(std::shared_ptr<NVRTCProgram> program,
              std::string cudadevrt_path = "")
      : program_(program), module_(nullptr), cudadevrt_path_(cudadevrt_path) {
    TV_ASSERT_RT_ERR(program, "program ptr must not empty");
#ifndef TV_HARDWARE_ACC_CUDA
    TV_THROW_RT_ERR("you must compile with CUDA first to use NVRTCModule");
#endif
    ptx_name_ = program->program_name() + ".ptx";
  }
  static std::shared_ptr<NVRTCModule>
  create(std::string code,
         std::unordered_map<std::string, std::string> headers = {},
         std::vector<std::string> opts = {},
         std::string program_name = "kernel",
         std::vector<std::string> name_exprs = {},
         std::string cudadevrt_path = "") {
    return std::make_shared<NVRTCModule>(
        NVRTCProgram::create(code, headers, opts, program_name, name_exprs),
        cudadevrt_path);
  }
  static std::shared_ptr<NVRTCModule>
  from_program(std::shared_ptr<NVRTCProgram> prog,
               std::string cudadevrt_path = "") {
    return std::make_shared<NVRTCModule>(prog, cudadevrt_path);
  }

  NVRTCModule &load() {
#if defined(TV_HARDWARE_ACC_CUDA)
    if (module_ != nullptr) {
      TV_THROW_RT_ERR("this module is already compiled");
    }
    auto cubin_ten = program_->cubin();
    if (!cubin_ten.empty()){
      TV_CUDA_RESULT_CHECK(
          wrapper_.cuDrvModuleLoadDataEx(&module_, cubin_ten.const_raw_data(), 0, 0, 0));
      return *this;
    }
    auto ptx = program_->ptx();
    if (!cudadevrt_path_.empty()) {
      size_t cubinSize;
      void *cubin;
      TV_CUDA_RESULT_CHECK(wrapper_.cuDrvLinkCreate(0, 0, 0, &linkState_));
      TV_CUDA_RESULT_CHECK(wrapper_.cuDrvLinkAddFile(
          linkState_, CU_JIT_INPUT_LIBRARY, cudadevrt_path_.c_str(), 0, 0, 0));
      TV_CUDA_RESULT_CHECK(
          wrapper_.cuDrvLinkAddData(linkState_, CU_JIT_INPUT_PTX, &ptx[0],
                                    ptx.size(), ptx_name_.c_str(), 0, 0, 0));
      TV_CUDA_RESULT_CHECK(
          wrapper_.cuDrvLinkComplete(linkState_, &cubin, &cubinSize));
      TV_CUDA_RESULT_CHECK(
          wrapper_.cuDrvModuleLoadDataEx(&module_, cubin, 0, 0, 0));

    } else {
      TV_CUDA_RESULT_CHECK(
          wrapper_.cuDrvModuleLoadDataEx(&module_, ptx.data(), 0, 0, 0));
    }
#endif
    return *this;
  }
#if defined(TV_HARDWARE_ACC_CUDA)

  CUfunction kernel(std::string name) {

    TV_ASSERT_RT_ERR(module_ != nullptr, "moculde must be loaded");
    CUfunction k = nullptr;
    TV_CUDA_RESULT_CHECK(
        wrapper_.cuDrvModuleGetFunction(&k, module_, name.c_str()));
    return k;
  }
#endif
  std::unordered_map<std::string, int> get_kernel_attributes(std::string name) {
    std::unordered_map<std::string, int> res;
#if defined(TV_HARDWARE_ACC_CUDA)
    auto k = kernel(name);
    int pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, k));
    res["max_threads_per_block"] = pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k));
    res["shared_size_bytes"] = pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, k));
    res["const_size_bytes"] = pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, k));
    res["local_size_bytes"] = pi;
    TV_CUDA_RESULT_CHECK(
        wrapper_.cuDrvFuncGetAttribute(&pi, CU_FUNC_ATTRIBUTE_NUM_REGS, k));
    res["num_regs"] = pi;
    TV_CUDA_RESULT_CHECK(
        wrapper_.cuDrvFuncGetAttribute(&pi, CU_FUNC_ATTRIBUTE_PTX_VERSION, k));
    res["ptx_version"] = pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_BINARY_VERSION, k));
    res["binary_version"] = pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k));
    res["max_dynamic_shared_size_bytes"] = pi;
    TV_CUDA_RESULT_CHECK(wrapper_.cuDrvFuncGetAttribute(
        &pi, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, k));
    res["preferred_shared_memory_carveout"] = pi;
#endif
    return res;
  }

  void set_max_dynamic_shared_size_bytes(std::string name, int size) {
#if defined(TV_HARDWARE_ACC_CUDA)
    auto k = kernel(name);
    TV_CUDA_RESULT_CHECK_V2(wrapper_.cuDrvFuncSetAttribute(
        k, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size), 
        "set_max_dynamic_shared_size_bytes error");
#endif
  }

  void set_preferred_smem_carveout(std::string name, int carveout) {
#if defined(TV_HARDWARE_ACC_CUDA)
    TV_ASSERT_INVALID_ARG(carveout > 0 && carveout <= 100, "carveout must in (0, 100]")
    auto k = kernel(name);
    TV_CUDA_RESULT_CHECK_V2(wrapper_.cuDrvFuncSetAttribute(
        k, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout), 
        "set_preferred_smem_carveout error");
#endif
  }

  std::shared_ptr<NVRTCProgram> get_program() { return program_; }

  std::string get_lowered_name(std::string name) const {
    TV_ASSERT_RT_ERR(program_ != nullptr, "program_ must not empty");
    return program_->get_lowered_name(name);
  }

  void run_kernel(std::string name, std::array<int, 3> blocks,
                  std::array<int, 3> threads, int smem_size,
                  std::uintptr_t stream_int,
                  std::vector<std::tuple<tv::Tensor, int, std::uintptr_t, size_t>> args) {
#if defined(TV_HARDWARE_ACC_CUDA)
    if (module_ == nullptr) {
      load();
    }
    CUstream stream = reinterpret_cast<CUstream>(stream_int);
    std::vector<void *> params;
    std::vector<const void *> tensor_ptrs(args.size());
    std::vector<tv::Tensor> tensor_view_datas;

    int cnt = 0;
    for (auto &arg : args) {
      auto &ten = std::get<0>(arg);
      auto arg_type = std::get<1>(arg);
      switch (arg_type) {
      case ArgType::kDevicePointer: {
        if (std::get<2>(arg) != 0){
          tensor_ptrs[cnt] = reinterpret_cast<const char *>(std::get<2>(arg)) + std::get<3>(arg);
        }else{
          tensor_ptrs[cnt] = nullptr;
        }
        params.push_back(&tensor_ptrs[cnt]);
        cnt += 1;
        break;
      }
      case ArgType::kTensor: {
        if (ten.empty()) {
          tensor_ptrs[cnt] = nullptr;
        } else {
          TV_ASSERT_INVALID_ARG(ten.device() == 0, "tensor must be GPU");
          tensor_ptrs[cnt] = ten.const_raw_data();
        }
        params.push_back(&tensor_ptrs[cnt]);
        cnt += 1;
        break;
      }
      case ArgType::kScalar:
      case ArgType::kConstant:
      case ArgType::kArray: {
        TV_ASSERT_INVALID_ARG(ten.device() == -1, "array tensor must be CPU");
        // const check is performed in python
        params.push_back(const_cast<void *>(
            reinterpret_cast<const void *>(ten.const_raw_data())));
        break;
      }
      case ArgType::kTensorView: {
        TV_ASSERT_INVALID_ARG(ten.device() == 0 && ten.ndim() <= 10, "array tensor must be GPU and <= 10 dim");
        tv::DispatchInt<tv::mp_list_int_range<1, 11>>()(ten.ndim(), [&](auto I){
          constexpr auto V = int(I);
          auto tview = ten.tview<const float, V, tv::DefaultPtrTraits, TV_GLOBAL_INDEX, false>();
          using tview_t = std::decay_t<decltype(tview)>;
          tv::Tensor storage = tv::empty({sizeof(tview_t)}, tv::uint8, tv::kDeviceCPU);
          auto tview_data_ptr = reinterpret_cast<tview_t*>(storage.raw_data());
          tview_data_ptr[0] = tview;
          tensor_view_datas.push_back(storage);
          params.push_back(storage.raw_data());
        });
        break;
      }
      default:
        TV_THROW_RT_ERR("not implemented");
      }
    }
    auto k = kernel(name);
    if (smem_size >= (48 << 10)) {
      TV_CUDA_RESULT_CHECK_V2(wrapper_.cuDrvFuncSetAttribute(
          k, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size), smem_size, get_kernel_attributes(name));
      TV_CUDA_RESULT_CHECK_V2(wrapper_.cuDrvFuncSetAttribute(
          k, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100), "cudaFuncAttributePreferredSharedMemoryCarveout");
    }
    TV_CUDA_RESULT_CHECK_V2(wrapper_.cuDrvLaunchKernel(
        k, blocks[0], blocks[1], blocks[2], threads[0], threads[1],
        threads[2], smem_size, stream, params.data(), 0),
        "Params:", blocks[0], blocks[1], blocks[2], threads[0], threads[1],
        threads[2], "Smem:", smem_size);
#endif
  }
#if defined(TV_HARDWARE_ACC_CUDA)
  const CUDADriverWrapper &get_driver_wrapper() { return wrapper_; }

  CUresult cuDrvLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                             uint32_t gridDimZ, uint32_t blockDimX,
                             uint32_t blockDimY, uint32_t blockDimZ,
                             uint32_t sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra) const {
    return wrapper_.cuDrvLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);
  }

  void *get_global_ptr(std::string name) const {
    size_t bytes;
    CUdeviceptr ptr;
    TV_CUDA_RESULT_CHECK(
        wrapper_.cuDrvModuleGetGlobal(&ptr, &bytes, module_, name.c_str()));
    return reinterpret_cast<void *>(ptr);
  }

#endif

  ~NVRTCModule() {
#if defined(TV_HARDWARE_ACC_CUDA)
    if (module_ != nullptr) {
      wrapper_.cuDrvModuleUnload(module_);
    }
    if (linkState_ != nullptr) {
      wrapper_.cuDrvLinkDestroy(linkState_);
    }

#endif
  }

private:
  std::shared_ptr<NVRTCProgram> program_;
  std::string cudadevrt_path_;
  std::string ptx_name_;

#if defined(TV_HARDWARE_ACC_CUDA)
  CUmodule module_ = nullptr;
  CUDADriverWrapper wrapper_;
  CUlinkState linkState_ = nullptr;
#else
  void *module_ = nullptr;
#endif
};

} // namespace tv
