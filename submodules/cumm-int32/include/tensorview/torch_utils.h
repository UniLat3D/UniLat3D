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

#pragma once
#include "mp_helper.h"
#include <ATen/ATen.h>
#include <tensorview/tensor.h>
#include <tensorview/tensorview.h>
#include <torch/script.h>
#if defined(TV_HARDWARE_ACC_CUDA)
#include <ATen/cuda/CUDAContext.h>
#endif
namespace tv {

#if defined(TV_HARDWARE_ACC_CUDA)
struct TorchGPU : public tv::GPU {
  virtual cudaStream_t getStream() const override {
    return at::cuda::getCurrentCUDAStream();
  }
};
#endif

namespace detail {
template <> struct TypeToString<at::Half> {
  static constexpr const char *value = "half";
};


template <typename T> struct TypeToTorchDtype;

template <> struct TypeToTorchDtype<int32_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt32;
};
template <> struct TypeToTorchDtype<int16_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt16;
};
template <> struct TypeToTorchDtype<int8_t> {
  static constexpr decltype(torch::kInt8) value = torch::kInt8;
};
template <> struct TypeToTorchDtype<int64_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt64;
};
template <> struct TypeToTorchDtype<uint8_t> {
  static constexpr decltype(torch::kInt32) value = torch::kUInt8;
};
template <> struct TypeToTorchDtype<bool> {
  static constexpr decltype(torch::kInt32) value = torch::kBool;
};
template <> struct TypeToTorchDtype<float> {
  static constexpr decltype(torch::kInt32) value = torch::kFloat32;
};
template <> struct TypeToTorchDtype<double> {
  static constexpr decltype(torch::kInt32) value = torch::kFloat64;
};
template <> struct TypeToTorchDtype<at::Half> {
  static constexpr decltype(torch::kInt32) value = torch::kHalf;
};

using all_torch_types_t = std::tuple<float, double, int8_t, int16_t, int32_t,
                                     int64_t, uint8_t, bool, at::Half>;

} // namespace detail

template <typename T>
constexpr decltype(torch::kInt32) torch_type_v =
    detail::TypeToTorchDtype<std::decay_t<T>>::value;


inline tv::DType constexpr torch_type_to_tv(decltype(torch::kInt32) t){
  switch (t) {
  case torch::kInt32:
    return tv::int32;
  case torch::kInt16:
    return tv::int16;
  case torch::kInt8:
    return tv::int8;
  case torch::kInt64:
    return tv::int64;
  case torch::kUInt8:
    return tv::uint8;
  case torch::kBool:
    return tv::bool_;
  case torch::kFloat32:
    return tv::float32;
  case torch::kFloat64:
    return tv::float64;
  case torch::kHalf:
    return tv::float16;
  default:
    return tv::unknown;
  }
}

inline decltype(torch::kFloat32) tv_type_to_torch(tv::DType t){
  switch (t) {
  case tv::int32:
    return torch::kInt32;
  case tv::int16:
    return torch::kInt16;
  case tv::int8:
    return torch::kInt8;
  case tv::int64:
    return torch::kInt64;
  case tv::uint8:
    return torch::kUInt8;
  case tv::bool_:
    return torch::kBool;
  case tv::float32:
    return torch::kFloat32;
  case tv::float64:
    return torch::kFloat64;
  case tv::float16:
    return torch::kHalf;
  default:
    TV_THROW_INVALID_ARG("unknown dtype", t);
  }
}

inline decltype(torch::kFloat32) tv_type_to_torch_uint_to_int(tv::DType t){
  switch (t) {
  case tv::int32:
    return torch::kInt32;
  case tv::int16:
    return torch::kInt16;
  case tv::int8:
    return torch::kInt8;
  case tv::int64:
    return torch::kInt64;
  case tv::uint32:
    return torch::kInt32;
  case tv::uint16:
    return torch::kInt16;
  case tv::uint64:
    return torch::kInt64;
  case tv::uint8:
    return torch::kUInt8;
  case tv::bool_:
    return torch::kBool;
  case tv::float32:
    return torch::kFloat32;
  case tv::float64:
    return torch::kFloat64;
  case tv::float16:
    return torch::kHalf;
  default:
    TV_THROW_INVALID_ARG("unknown dtype", t);
  }
}

template <class... Ts, typename F>
void dispatch_torch(at::ScalarType t, F &&f) {
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  tv::mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    if (detail::TypeToTorchDtype<std::decay_t<TV_DECLTYPE(I)>>::value == t) {
      std::forward<F>(f)(TV_DECLTYPE(I)());
      notFound = false;
    }
  });
  if (notFound) {
    std::stringstream ss;
    tv::mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << tv::type_s<TV_DECLTYPE(I)> << " ";
    });
    TV_THROW_RT_ERR("unknown type", t, ", available:", ss.str());
  }
}

template <class T> struct DispatchTorch;

template <template <class...> class T, class... Args>
struct DispatchTorch<T<Args...>> {
  template <typename F> inline void operator()(at::ScalarType t, F &&f) {
    return dispatch_torch<Args...>(t, std::forward<F>(f));
  }
};

template <typename T> void check_torch_dtype(const torch::Tensor &tensor) {
  DispatchTorch<detail::all_torch_types_t>()(tensor.scalar_type(), [&](auto I) {
    using Ttensor = TV_DECLTYPE(I);
    constexpr bool val = std::is_same<std::decay_t<T>, Ttensor>::value;
    TV_ASSERT_RT_ERR(val, "your torch tensor has dtype",
                     tv::type_s<Ttensor>,
                     "but expect",
                     tv::type_s<T>);
  });
}

template <typename T, int Rank = -1,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = TV_GLOBAL_INDEX>
TensorView<T, Rank, PtrTraits, Tindex> torch2tv(const torch::Tensor &tensor) {
  using tv_shape_t =
      typename TensorView<T, Rank, PtrTraits, Tindex>::tv_shape_t;
  check_torch_dtype<T>(tensor);
  // TODO stride
  if (Rank > 0) {
    TV_ASSERT_INVALID_ARG(tensor.dim() == Rank, "error");
  }
  tv_shape_t shape;
  for (auto i : tensor.sizes()) {
    shape.push_back(i);
  }
  return tv::TensorView<T, Rank, PtrTraits, Tindex>(
      tensor.data_ptr<std::remove_const_t<T>>(), shape);
}

inline tv::Tensor torch2tensor(const torch::Tensor &tensor) {
  tv::TensorShape shape;
  tv::TensorShape stride;

  for (auto i : tensor.sizes()) {
    shape.push_back(i);
  }
  for (auto i : tensor.strides()) {
    stride.push_back(i);
  }
  int device = -1;
  if (tensor.device().type() == torch::kCUDA){
    device = 0;
  }
  if (tensor.device().type() == torch::kMPS){
    device = 0;
  }
  if (tensor.device().type() == torch::kMPS){
    auto storage_ptr = tensor.storage().mutable_data();
    auto storage_offset = tensor.storage_offset();
    return tv::from_blob(storage_ptr, shape, stride, torch_type_to_tv(tensor.scalar_type()), device, storage_offset);

  } else{
    return tv::from_blob(tensor.data_ptr(), shape, stride, torch_type_to_tv(tensor.scalar_type()), device);

  }
}

inline torch::Tensor tensor2torch(tv::Tensor tensor, bool cast_uint_to_int = false) {

  tv::TensorShape shape_tv = tensor.shape();
  tv::TensorShape stride_tv = tensor.stride();

  std::vector<int64_t> shape(shape_tv.begin(), shape_tv.end());
  std::vector<int64_t> stride(stride_tv.begin(), stride_tv.end());

  auto torch_dtype = tv_type_to_torch_uint_to_int(tensor.dtype());
  if (!cast_uint_to_int){
    torch_dtype = tv_type_to_torch(tensor.dtype());
  }
  torch::TensorOptions opt = torch::TensorOptions().dtype(torch_dtype);
  if (tensor.device() == -1){
    opt = opt.device(torch::kCPU);
  }else{
#if defined(TV_HARDWARE_ACC_CUDA)
    opt = opt.device(torch::kCUDA);
#elif defined(TV_HARDWARE_ACC_METAL)
    opt = opt.device(torch::kMPS);
#else 
    TV_THROW_RT_ERR("unknown device, cumm only support mps and cuda.");
#endif
  }
  if (tensor.device() == -1){
    return torch::from_blob(tensor.raw_data(), torch::IntArrayRef(shape), torch::IntArrayRef(stride), opt);
  }else{
#ifdef TV_HARDWARE_ACC_METAL
    // currently pytorch don't support mps from_blob.
    TV_THROW_INVALID_ARG("currently pytorch don't support mps from_blob");
    // return at::from_blob(tensor.storage()->apple_metal_buffer_ptr(), 
    //   torch::IntArrayRef(shape), torch::IntArrayRef(stride), tensor.storage()->offset(), 
    //   [](auto* ptr){}, opt);
#else
    return torch::from_blob(tensor.raw_data(), torch::IntArrayRef(shape), torch::IntArrayRef(stride), opt);
#endif
  }

}


template <typename T>
torch::Tensor torch_slice_first_axis(torch::Tensor tensor, T start, T end) {
  // only torch >= 1.5 have tensor slice.
  torch::Tensor res;
  auto tensor_shape = tensor.sizes();
  std::vector<int64_t> shape(tensor_shape.begin(), tensor_shape.end());
  shape[0] = end - start;
  uint8_t *ptr = reinterpret_cast<uint8_t *>(tensor.data_ptr());
  res = torch::from_blob(ptr + start * tensor.stride(0) * tensor.itemsize(),
                         torch::IntArrayRef(shape), tensor.options());
  return res;
}
} // namespace tv