/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/plugin/common/cudaDriverWrapper.cpp
// https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/plugin/common/cudaDriverWrapper.h
#pragma once

#define CUDA_LIB_NAME "cuda"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void *)LoadLibraryA("nv" name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name)                                                \
  GetProcAddress(static_cast<HMODULE>(handle), name)
#else
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so.1", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif

#include <cuda.h>
#include <tensorview/core/common.h>

namespace tv {

class CUDADriverWrapper {
public:
  CUDADriverWrapper() {
    handle = dllOpen(CUDA_LIB_NAME);
    TV_ASSERT_RT_ERR(handle != nullptr, "load CUDA Driver library failed!!! "
                                        "You must install cuda driver first.");

    auto load_sym = [](void *handle, const char *name) {
      void *ret = dllGetSym(handle, name);
      TV_ASSERT_RT_ERR(ret != nullptr, "load", name,
                       "from CUDA Driver library failed!!!");
      return ret;
    };

    *(void **)(&_cuModuleUnload) = load_sym(handle, "cuModuleUnload");
    *(void **)(&_cuModuleLoadDataEx) = load_sym(handle, "cuModuleLoadDataEx");
    *(void **)(&_cuModuleGetFunction) = load_sym(handle, "cuModuleGetFunction");
    *(void **)(&_cuLaunchKernel) = load_sym(handle, "cuLaunchKernel");
    *(void **)(&_cuLinkCreate) = load_sym(handle, "cuLinkCreate_v2");
    *(void **)(&_cuLinkAddFile) = load_sym(handle, "cuLinkAddFile_v2");
    *(void **)(&_cuLinkAddData) = load_sym(handle, "cuLinkAddData_v2");
    *(void **)(&_cuLinkComplete) = load_sym(handle, "cuLinkComplete");
    *(void **)(&_cuLinkDestroy) = load_sym(handle, "cuLinkDestroy");
    *(void **)(&_cuLaunchCooperativeKernel) =
        load_sym(handle, "cuLaunchCooperativeKernel");
    *(void **)(&_cuFuncSetAttribute) = load_sym(handle, "cuFuncSetAttribute");
    *(void **)(&_cuGetErrorName) = load_sym(handle, "cuGetErrorName");
    *(void **)(&_cuFuncGetAttribute) = load_sym(handle, "cuFuncGetAttribute");
    *(void **)(&_cuModuleGetGlobal) = load_sym(handle, "cuModuleGetGlobal_v2");
    // if cuda change cuLinkAddFile_v2 again (e.g. cuLinkAddFile_v3), we can
    // get compile error here.
    static_assert(&cuModuleGetGlobal_v2 == &cuModuleGetGlobal, "error");
    static_assert(&cuLinkAddFile_v2 == &cuLinkAddFile, "error");
    static_assert(&cuLinkCreate_v2 == &cuLinkCreate, "error");
    static_assert(&cuLinkAddData_v2 == &cuLinkAddData, "error");
  }

  ~CUDADriverWrapper() {
    if (handle) {
      dllClose(handle);
    }
  }

  // Delete default copy constructor and copy assignment constructor
  CUDADriverWrapper(const CUDADriverWrapper &) = delete;
  CUDADriverWrapper &operator=(const CUDADriverWrapper &) = delete;

  CUresult cuDrvModuleUnload(CUmodule hmod) const {
    return (*_cuModuleUnload)(hmod);
  }
  CUresult cuDrvModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                const char *name) const {
    return (*_cuModuleGetGlobal)(dptr, bytes, hmod, name);
  }

  CUresult cuDrvModuleLoadDataEx(CUmodule *module, const void *image,
                                 unsigned int numOptions, CUjit_option *options,
                                 void **optionValues) const {
    return (*_cuModuleLoadDataEx)(module, image, numOptions, options,
                                  optionValues);
  }
  CUresult cuDrvModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                  const char *name) const {
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
  }
  CUresult cuDrvLinkAddFile(CUlinkState state, CUjitInputType type,
                            const char *path, uint32_t numOptions,
                            CUjit_option *options, void **optionValues) const {
    return (*_cuLinkAddFile)(state, type, path, numOptions, options,
                             optionValues);
  }
  CUresult cuDrvLinkAddData(CUlinkState state, CUjitInputType type, void *data,
                            size_t size, const char *name, uint32_t numOptions,
                            CUjit_option *options, void **optionValues) const {
    return (*_cuLinkAddData)(state, type, data, size, name, numOptions, options,
                             optionValues);
  }
  CUresult cuDrvLinkCreate(uint32_t numOptions, CUjit_option *options,
                           void **optionValues, CUlinkState *stateOut) const {
    return (*_cuLinkCreate)(numOptions, options, optionValues, stateOut);
  }

  CUresult cuDrvLinkDestroy(CUlinkState state) const {
    return (*_cuLinkDestroy)(state);
  }
  CUresult cuDrvLinkComplete(CUlinkState state, void **cubinOut,
                             size_t *sizeOut) const {
    return (*_cuLinkComplete)(state, cubinOut, sizeOut);
  }

  CUresult cuDrvLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                             uint32_t gridDimZ, uint32_t blockDimX,
                             uint32_t blockDimY, uint32_t blockDimZ,
                             uint32_t sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra) const {
    return (*_cuLaunchKernel)(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                              blockDimY, blockDimZ, sharedMemBytes, hStream,
                              kernelParams, extra);
  }
  CUresult cuDrvLaunchCooperativeKernel(
      CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
      uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
      uint32_t sharedMemBytes, CUstream hStream, void **kernelParams) const {
    return (*_cuLaunchCooperativeKernel)(f, gridDimX, gridDimY, gridDimZ,
                                         blockDimX, blockDimY, blockDimZ,
                                         sharedMemBytes, hStream, kernelParams);
  }
  CUresult cuDrvFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                                 int value) const {
    return (*_cuFuncSetAttribute)(hfunc, attrib, value);
  }
  CUresult cuDrvGetErrorName(CUresult error, const char **pStr) const {
    return (*_cuGetErrorName)(error, pStr);
  }
  CUresult cuDrvFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                 CUfunction hfunc) const {
    return (*_cuFuncGetAttribute)(pi, attrib, hfunc);
  }

private:
  void *handle;
  CUresult (*_cuGetErrorName)(CUresult, const char **);
  CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
  CUresult (*_cuFuncGetAttribute)(int *, CUfunction_attribute, CUfunction);
  CUresult (*_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int,
                                         unsigned int, unsigned int,
                                         unsigned int, unsigned int,
                                         unsigned int, CUstream, void **);

  CUresult (*_cuLinkComplete)(CUlinkState, void **, size_t *);
  CUresult (*_cuLinkDestroy)(CUlinkState);
  CUresult (*_cuLinkCreate)(unsigned int, CUjit_option *, void **,
                            CUlinkState *);
  CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, const char *,
                             unsigned int, CUjit_option *, void **);
  CUresult (*_cuLinkAddData)(CUlinkState, CUjitInputType, void *, size_t,
                             const char *, unsigned int, CUjit_option *,
                             void **);
  CUresult (*_cuModuleGetGlobal)(CUdeviceptr *, size_t *, CUmodule,
                                 const char *);
  CUresult (*_cuModuleUnload)(CUmodule);
  CUresult (*_cuModuleLoadDataEx)(CUmodule *module, const void *image,
                                  unsigned int numOptions,
                                  CUjit_option *options, void **optionValues);
  CUresult (*_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
  CUresult (*_cuLaunchKernel)(CUfunction f, uint32_t gridDimX,
                              uint32_t gridDimY, uint32_t gridDimZ,
                              uint32_t blockDimX, uint32_t blockDimY,
                              uint32_t blockDimZ, uint32_t sharedMemBytes,
                              CUstream hStream, void **kernelParams,
                              void **extra);
};

} // namespace tv
