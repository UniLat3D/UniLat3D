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
#include "core.h"
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cublasLt.h>
#endif

namespace tv {

struct BlasContext : Context {
  BlasContext() : Context() {
    ContextManager cublaslt_mgr;
    cublaslt_mgr.creater = []() -> std::uintptr_t {
      cublasStatus_t stat;
      cublasLtHandle_t blas_handle;
      stat = cublasLtCreate(&blas_handle);
      TV_ASSERT_RT_ERR(CUBLAS_STATUS_SUCCESS == stat, "error create cublaslt");
      return reinterpret_cast<std::uintptr_t>(blas_handle);
    };
    cublaslt_mgr.deleter = [](std::uintptr_t ptr_int) {
      cublasLtDestroy(reinterpret_cast<cublasLtHandle_t>(ptr_int));
    };
    context_ptr_->register_manager(ContextType::kCublasLt, cublaslt_mgr);
  }

#if defined(TV_HARDWARE_ACC_CUDA)
  bool has_cublaslt_handle() {
    check_ptr_valid();
    return context_ptr_->has_item(ContextType::kCublasLt);
  }

  Context &create_cublaslt() {
    check_ptr_valid();
    context_ptr_->create_item(ContextType::kCublasLt);
    return *this;
  }
  Context &set_cublaslt(cublasLtHandle_t handle) {
    check_ptr_valid();
    context_ptr_->create_raw_item(ContextType::kCublasLt,
                                  reinterpret_cast<std::uintptr_t>(handle));
    return *this;
  }
  cublasLtHandle_t cublaslt_handle() {
    check_ptr_valid();
    return reinterpret_cast<cublasLtHandle_t>(
        context_ptr_->get_item(ContextType::kCublasLt));
  }
#endif
};

} // namespace tv
