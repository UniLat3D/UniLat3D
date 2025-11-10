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
#include "tensor.h"
#include "tensorview.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace tv {

template <typename Tarr> bool is_c_style(const Tarr &arr) {
  return bool(arr.flags() & py::array::c_style);
}


template <typename T, int Rank = -1>
TensorView<T, Rank> arrayt2tv(py::array_t<T> arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  if (Rank >= 0) {
    TV_ASSERT_INVALID_ARG(shape.ndim() == Rank, "error");
  }
  return TensorView<T, Rank>(arr.mutable_data(), shape);
}

template <typename T, int Rank = -1>
TensorView<const T> carrayt2tv(py::array_t<T> arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  if (Rank >= 0) {
    TV_ASSERT_INVALID_ARG(shape.ndim() == Rank, "error");
  }
  return TensorView<const T, Rank>(arr.data(), shape);
}

template <typename Tarr> tv::DType get_array_tv_dtype(const Tarr &arr) {
  switch (arr.dtype().kind()) {
  case 'b':
    return tv::bool_;
  case 'i': {
    switch (arr.itemsize()) {
    case 1:
      return tv::int8;
    case 2:
      return tv::int16;
    case 4:
      return tv::int32;
    case 8:
      return tv::int64;
    default:
      break;
    }
  }
  case 'u': {
    switch (arr.itemsize()) {
    case 1:
      return tv::uint8;
    case 2:
      return tv::uint16;
    case 4:
      return tv::uint32;
    case 8:
      return tv::uint64;
    default:
      break;
    }
  }
  case 'f': {
    switch (arr.itemsize()) {
    case 2:
      return tv::float16;
    case 4:
      return tv::float32;
    case 8:
      return tv::float64;
    default:
      break;
    }
  }
  }
  TV_THROW_RT_ERR("unknown dtype", arr.dtype().kind(), arr.itemsize());
}

template <typename Tarr> Tensor array2tensor(Tarr &arr) {
  // TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  TensorShape shape, stride;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
    stride.push_back(arr.strides(i) / arr.itemsize());

  }
  if (arr.writeable()){
    return tv::from_blob(arr.mutable_data(), shape, stride, get_array_tv_dtype(arr), -1);
  }else{
    return tv::from_blob(arr.data(), shape, stride, get_array_tv_dtype(arr), -1);
  }
}

template <typename T> Tensor arrayt2tensor(py::array_t<T> &arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  TensorShape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return from_blob(arr.mutable_data(), shape, type_v<T>, -1);
}

template <typename T, typename Tarr> TensorView<T> array2tview(Tarr &arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  auto type = type_v<T>;
  auto np_type = get_array_tv_dtype(arr);
  TV_ASSERT_INVALID_ARG(np_type == type,
                        "array type mismatch, expected:", dtype_str(type),
                        "get:", dtype_str(np_type));
  return TensorView<T>(reinterpret_cast<T *>(arr.mutable_data()), shape);
}

template <typename TDType> py::dtype tv_dtype_to_py(TDType d) {
  switch (d) {
  case float32:
    return py::dtype("float32");
  case float64:
    return py::dtype("float64");
  case float16:
    return py::dtype("float16");
  case int32:
    return py::dtype("int32");
  case int16:
    return py::dtype("int16");
  case int8:
    return py::dtype("int8");
  case int64:
    return py::dtype("int64");
  case uint32:
    return py::dtype("uint32");
  case uint16:
    return py::dtype("uint16");
  case uint8:
    return py::dtype("uint8");
  case uint64:
    return py::dtype("uint64");
  case bool_:
    return py::dtype("bool_");
  case tf32:
    return py::dtype("float32");
  default:;
  }
  TV_THROW_INVALID_ARG("unknown dtype", d);
}

// add template to define function in header
template <typename Ttensor> py::array tensor2array(const Ttensor &tensor) {
  // you cant call this function during GIL released.
  TV_ASSERT_INVALID_ARG(tensor.device() == -1, "must be cpu tensor");
  auto shape = tensor.shape();
  std::vector<int> shape_vec(shape.begin(), shape.end());
  auto stride = tensor.stride();
  std::vector<int> stride_vec(stride.begin(), stride.end());
  int itemsize = tensor.itemsize();
  // py::array stride is BYTES!!!
  for (auto &s : stride_vec) {
    s *= itemsize;
  }
  auto dtype = tv_dtype_to_py(tensor.dtype());
  // construct py::array will copy content from ptr.
  // its expected because we can't transfer ownership from
  // c++ tv::Tensor to numpy array when c++ object is deleted.
  return py::array(dtype, shape_vec, stride_vec, tensor.raw_data());
}

template <typename T> py::array tview2array(TensorView<T> tview) {
  // you cant call this function during GIL released.
  auto shape = tview.shape();
  auto shape_vec = std::vector<int64_t>(shape.begin(), shape.end());
  auto stride = tview.stride();
  auto stride_vec = std::vector<int64_t>(stride.begin(), stride.end());
  auto tensor = Tensor(tview.data(), TensorShape(shape_vec),
                       TensorShape(stride_vec), tv::type_v<T>, -1);
  return tensor2array(tensor);
}

template <typename T> decltype(auto) to_tensor(const T &x) {
  return if_constexpr<std::is_same<T, py::array>>(
      [&](auto _) { return array2tensor(_(x)); }, [&]() { return x; });
}

template <typename F, typename... Args>
constexpr decltype(auto) pyarray_invoke(F &&f, Args &&...args) {
  return std::forward<F>(f)(to_tensor(std::forward<Args>(args))...);
}

} // namespace tv

namespace pybind11 {
namespace detail {

template <typename Type, size_t Size>
struct type_caster<tv::array<Type, Size>>
    : array_caster<tv::array<Type, Size>, Type, false, Size> {};

/*
template <> struct type_caster<tv::Tensor> {
public:
  PYBIND11_TYPE_CASTER(tv::Tensor, _("tv::Tensor"));

  bool load(handle src, bool convert) {
    // if (base::load(src, convert)) {
    //   return true;
    // }
    if (py::isinstance<py::array>(src)) {
      auto arr = py::cast<py::array>(src);
      if (!tv::is_c_style(arr)) {
        tv::ssprint("[ERROR] only support c style contiguous array");
        return false;
      }
      value = tv::array2tensor(arr);
      return true;
    }
    return false;
  }

  static handle cast(tv::Tensor src, return_value_policy,
                     handle h ) {
    //
https://stackoverflow.com/questions/42645228/cast-numpy-array-to-from-custom-c-matrix-class-using-pybind11
    return tv::tensor2array(src).release();
  }
};
*/

#define DECLARE_TENSORVIEW_BIND(T)                                             \
  template <> struct type_caster<tv::TensorView<T>> {                          \
  public:                                                                      \
    PYBIND11_TYPE_CASTER(tv::TensorView<T>, _("tv::TensorView<#T>"));          \
    bool load(handle src, bool convert) {                                      \
      if (py::isinstance<py::array>(src)) {                                    \
        auto arr = py::cast<py::array>(src);                                   \
        if (!tv::is_c_style(arr)) {                                            \
          tv::ssprint("[ERROR] only support c style contiguous array");        \
          return false;                                                        \
        }                                                                      \
        value = tv::array2tview<T>(arr);                                       \
        return true;                                                           \
      }                                                                        \
      return false;                                                            \
    }                                                                          \
    static handle cast(tv::TensorView<T> src,                                  \
                       return_value_policy /* policy */,                       \
                       handle h /* parent */) {                                \
      return tv::tview2array<T>(src).release();                                \
    }                                                                          \
  };

DECLARE_TENSORVIEW_BIND(float);
DECLARE_TENSORVIEW_BIND(double);
DECLARE_TENSORVIEW_BIND(int8_t);
DECLARE_TENSORVIEW_BIND(int16_t);
DECLARE_TENSORVIEW_BIND(int32_t);
DECLARE_TENSORVIEW_BIND(int64_t);
DECLARE_TENSORVIEW_BIND(uint8_t);
DECLARE_TENSORVIEW_BIND(uint16_t);
DECLARE_TENSORVIEW_BIND(uint32_t);
DECLARE_TENSORVIEW_BIND(uint64_t);
DECLARE_TENSORVIEW_BIND(bool);
#undef DECLARE_TENSORVIEW_BIND

} // namespace detail
} // namespace pybind11