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
#include "core/const_ops.h"
#include "mp_helper.h"

#include "core/vecarray.h"

#include "core/defs.h"

#ifdef TV_PARALLEL_RTC
#include <tensorview/core/nvrtc_std.h>
#else

#include "common.h"
#include "dtypes.h"
#include "prettyprint.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>
#endif

#if defined(TV_HARDWARE_ACC_CUDA)

#include <cuda_fp16.h>
#ifndef __CUDACC_RTC__

#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
#endif
#if (CUDA_VERSION >= 11000 && defined(TV_HARDWARE_ACC_CUDA))
#include <cuda_bf16.h>
#endif

namespace tv {

#if defined(TV_HARDWARE_ACC_CUDA)
#ifndef __CUDACC_RTC__

struct GPU {
  GPU(cudaStream_t s = 0) : mStream(s) {}
  virtual cudaStream_t getStream() const { return mStream; }
  cudaStream_t mStream = 0;
};
#endif
#endif
struct CPU {};

template <typename T> struct DefaultPtrTraits { typedef T *type; };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T> struct RestrictPtrTraits {
  typedef T *__restrict__ type;
};
#endif

/*
template <typename T>
constexpr size_t calc_align(size_t ndim)
{
  if (ndim * sizeof(T) == 1)
    return 1;
  else if (ndim * sizeof(T) == 2)
    return 2;
  else if (ndim * sizeof(T) <= 4 && ndim * sizeof(T) > 2)
    return 4;
  else if (ndim * sizeof(T) <= 8 && ndim * sizeof(T) > 4)
    return 8;
  else if (ndim * sizeof(T) <= 16 && ndim * sizeof(T) > 8)
    return 16;
  else if (ndim * sizeof(T) <= 32 && ndim * sizeof(T) > 16)
    return 32;
  else
    return 64;
}
*/

struct Slice {
  template <class... Integers> TV_HOST_DEVICE_INLINE Slice(Integers... ints) {
    static_assert(sizeof...(ints) <= 3, "slice init must smaller than 3");
    vecarray<int, 3> slices{int(ints)...};
    slices_[0] = -1;
    slices_[1] = -1;
    slices_[2] = -1;
    for (size_t i = 0; i < slices.size(); ++i) {
      slices_[i] = slices[i];
    }
  }

  TV_HOST_DEVICE_INLINE Slice() {
    slices_[0] = -1;
    slices_[1] = -1;
    slices_[2] = -1;
  }
  template <typename T>
  TV_HOST_DEVICE_INLINE Slice(std::initializer_list<T> slice) {
    slices_[0] = -1;
    slices_[1] = -1;
    slices_[2] = -1;
    TV_ASSERT(slice.size() <= 3);
    int idx = 0;
    for (T s : slice) {
      slices_[idx] = int(s);
      ++idx;
    }
  }
  TV_HOST_DEVICE_INLINE int &operator[](int idx) {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < 3);
#endif
    return slices_[idx];
  }
  TV_HOST_DEVICE_INLINE const int &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < 3);
#endif
    return slices_[idx];
  }

protected:
  int slices_[3];
};

template <size_t MaxDim = TV_MAX_DIM, typename Tindex = TV_GLOBAL_INDEX>
struct ShapeBase : public vecarray<Tindex, MaxDim> {
  static constexpr int kMaxDim = MaxDim;
  TV_HOST_DEVICE_INLINE ShapeBase() : vecarray<Tindex, MaxDim>(){};
  TV_HOST_DEVICE_INLINE ShapeBase(std::initializer_list<Tindex> shape)
      : vecarray<Tindex, MaxDim>(shape) {}
  TV_HOST_DEVICE_INLINE ShapeBase(vecarray<Tindex, MaxDim> vec)
      : vecarray<Tindex, MaxDim>(vec) {}
  TV_HOST_DEVICE_INLINE ShapeBase(size_t size, Tindex init = Tindex())
      : vecarray<Tindex, MaxDim>(size, init) {}

  TV_HOST_DEVICE_INLINE ShapeBase(const ShapeBase<MaxDim> &shape) {
    TV_ASSERT(shape.ndim() <= MaxDim);
    for (size_t i = 0; i < shape.ndim(); ++i) {
      this->array_[i] = shape[i];
    }
    this->size_ = shape.ndim();
  }
#ifndef __CUDACC_RTC__
  template <typename Iterator, typename = detail::_RequireInputIter<Iterator>>
  ShapeBase(Iterator first, Iterator last)
      : vecarray<Tindex, MaxDim>(first, last) {}
  ShapeBase(const std::vector<int32_t> &arr)
      : vecarray<Tindex, MaxDim>(arr.begin(), arr.end()) {}
  ShapeBase(const std::vector<int64_t> &arr)
      : vecarray<Tindex, MaxDim>(arr.begin(), arr.end()) {}
#endif
  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> &
  operator=(const ShapeBase<MaxDim, Tindex> &shape) {
    TV_ASSERT(shape.ndim() <= MaxDim);
    for (size_t i = 0; i < shape.ndim(); ++i) {
      this->array_[i] = shape[i];
    }
    this->size_ = shape.ndim();
    return *this;
  };
  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> subshape(Tindex start,
                                                    Tindex end) const {
#ifdef TV_DEBUG
    TV_ASSERT(start >= 0 && end <= this->size_ && end > start);
#endif
    ShapeBase<MaxDim, Tindex> shape;
    for (Tindex i = start; i < end; ++i) {
      shape.push_back(this->array_[i]);
    }
    return shape;
  }
  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> subshape(Tindex start) const {
#ifdef TV_DEBUG
    TV_ASSERT(start >= 0 && start <= this->size_);
#endif
    ShapeBase<MaxDim, Tindex> shape;
    for (size_t i = start; i < this->size_; ++i) {
      shape.push_back(this->array_[i]);
    }
    return shape;
  }

  TV_HOST_DEVICE size_t size() const {
    if (this->size_ == 0)
      return 0;
    size_t s = 1;
    for (int i = 0; i < int(this->size_); ++i) {
      s *= this->array_[i];
    }
    return s;
  }
  TV_HOST_DEVICE_INLINE size_t ndim() const { return this->size_; }
  TV_HOST_DEVICE_INLINE bool empty() const { 
    if (this->size_ == 0){
      return true;
    }
    for (size_t i = 0; i < this->size_; ++i) {
      if (this->array_[i] == 0)
        return true;
    }
    return false;
  }

  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> squeeze() const {
    ShapeBase<MaxDim, Tindex> shape;
    for (size_t i = 0; i < this->size_; ++i) {
      if (this->array_[i] != 1)
        shape.push_back(this->array_[i]);
    }
    if (shape.empty()) {
      // dont support empty shape for now
      shape.push_back(1);
    }
    return shape;
  }
  template <size_t MaxDim2 = MaxDim>
  TV_HOST_DEVICE ShapeBase<MaxDim2, Tindex> squeeze(int dim) const {
    static_assert(MaxDim2 >= MaxDim - 1, "error");

    ShapeBase<MaxDim2, Tindex> shape;
    for (size_t i = 0; i < this->size_; ++i) {
      if (i != size_t(dim) || this->array_[i] != 1)
        shape.push_back(this->array_[i]);
    }
    return shape;
  }
  template <size_t MaxDim2 = MaxDim>
  TV_HOST_DEVICE ShapeBase<MaxDim2, Tindex> unsqueeze(int dim) const {
    static_assert(MaxDim2 >= MaxDim - 1, "error");
    ShapeBase<MaxDim2, Tindex> shape;
    for (size_t i = 0; i < this->size_ + 1; ++i) {
      if (i == size_t(dim))
        shape.push_back(1);
      if (i < this->size_)
        shape.push_back(this->array_[i]);
    }
    return shape;
  }

  TV_HOST_DEVICE size_t prod(Tindex start = 0) const {
    size_t res = 1;
    for (size_t i = start; i < this->size_; ++i) {
      res *= this->array_[i];
    }
    return res;
  }
  template <size_t MaxDim2 = MaxDim>
  TV_HOST_DEVICE ShapeBase<MaxDim2, Tindex> stride_rowmajor() {
    static_assert(MaxDim2 >= MaxDim, "error");
    Tindex p = Tindex(1);
    ShapeBase<MaxDim2, Tindex> res(this->size_);
    for (Tindex i = this->size_ - 1; i >= 0; --i) {
      res[i] = p;
      p *= this->array_[i];
    }
    return res;
  }
};

using Shape = ShapeBase<TV_MAX_DIM, TV_GLOBAL_INDEX>;
#ifndef __CUDACC_RTC__

template <class... Inds>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(std::vector<TV_GLOBAL_INDEX> &shape,
                                           Inds... indexes) {
  unsigned offset = 0;
  unsigned m = 1;
  TV_GLOBAL_INDEX indexes_vec[sizeof...(indexes)] = {indexes...};
#ifdef TV_DEBUG
  TV_ASSERT(sizeof...(indexes) == shape.size());
#endif
  TV_PRAGMA_UNROLL
  for (int i = sizeof...(indexes) - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

inline TV_GLOBAL_INDEX rowArrayIdx(std::vector<TV_GLOBAL_INDEX> &shape,
                                   std::vector<TV_GLOBAL_INDEX> &indexes_vec) {
  TV_GLOBAL_INDEX offset = 0;
  TV_GLOBAL_INDEX m = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}
#endif
template <class... Inds>
TV_HOST_DEVICE_INLINE TV_GLOBAL_INDEX rowArrayIdx(const Shape &shape,
                                                  Inds... indexes) {
  TV_GLOBAL_INDEX offset = 0;
  TV_GLOBAL_INDEX m = 1;
  TV_GLOBAL_INDEX indexes_vec[sizeof...(indexes)] = {indexes...};
  TV_PRAGMA_UNROLL
  for (int i = sizeof...(indexes) - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

TV_HOST_DEVICE_INLINE TV_GLOBAL_INDEX rowArrayIdx(const Shape &shape,
                                                  const Shape &indexes_vec) {
  TV_GLOBAL_INDEX offset = 0;
  TV_GLOBAL_INDEX m = 1;
  for (int i = indexes_vec.ndim() - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Index *indexes,
                                           const Index *shape) {
  TV_GLOBAL_INDEX offset = 0;
  TV_GLOBAL_INDEX m = 1;
  TV_PRAGMA_UNROLL
  for (int i = NDim - 1; i >= 0; --i) {
    offset += m * indexes[i];
    m *= shape[i];
  }
  return offset;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE Index rowArrayIdxInv(Index index, Index *output,
                                           const Index *shape) {
  TV_PRAGMA_UNROLL
  for (int i = NDim - 1; i >= 0; --i) {
    output[i] = index % shape[i];
    index -= output[i];
    index /= shape[i];
  }
  return index;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE Index rowArrayIdxInvStride(Index index, Index *output,
                                                 const Index *stride) {
  TV_PRAGMA_UNROLL
  for (int i = 0; i < NDim; ++i) {
    output[i] = index / stride[i];
    index -= output[i] * stride[i];
  }
  return index;
}

template <typename Index>
TV_HOST_DEVICE Index rowArrayIdxInv(Index index, Index *output,
                                    const Index *shape, int ndim) {
  for (int i = ndim - 1; i >= 0; --i) {
    output[i] = index % shape[i];
    index -= output[i];
    index /= shape[i];
  }
  return index;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE constexpr Index
rowArrayIdxStride(Index index, const array<Index, NDim> &stride) {
  array<Index, NDim> res{};
  TV_PRAGMA_UNROLL
  for (int i = NDim - 1; i >= 0; --i) {
    res[i] = index / stride[i];
    if (i > 0) {
      index -= res[i] * stride[i];
    }
  }
  return res;
}

template <typename T, unsigned... Shapes> struct FixedArray {

  template <std::size_t I> TV_HOST_DEVICE_INLINE constexpr unsigned dim() {
    return mp_nth_c<I, unsigned, Shapes...>;
  }
  TV_HOST_DEVICE_INLINE T &operator()(unsigned i1) {
    static_assert(sizeof...(Shapes) == 1, "error");
    return arr_[i1];
  }
  TV_HOST_DEVICE_INLINE T &operator[](unsigned i1) { return arr_[i1]; }

  TV_HOST_DEVICE_INLINE T &operator()(unsigned i1, unsigned i2) {
    static_assert(sizeof...(Shapes) == 2, "error");
    return arr_[i1 * dim<1>() + i2];
  }
  TV_HOST_DEVICE_INLINE T &operator()(unsigned i1, unsigned i2, unsigned i3) {
    static_assert(sizeof...(Shapes) == 3, "error");
    return arr_[i1 * dim<1>() * dim<2>() + i2 * dim<2>() + i3];
  }

private:
  T arr_[mp_prod_int<Shapes...>::value];
};

template <int N> struct ArrayIndexRowMajorReverse {
  template <typename TShape, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, T index,
                                            Ts... inds) {
    return index +
           shape[N - 1] * ArrayIndexRowMajorReverse<N - 1>::run(shape, inds...);
  }
  template <typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape, T index,
                                                 Ts... inds) {
    return index +
           shape[N - 1] * ArrayIndexRowMajorReverse<N - 1>::run(shape, inds...);
  }
};

template <> struct ArrayIndexRowMajorReverse<1> {
  template <typename TShape, typename T>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, T idx) {
    return idx;
  }
  template <typename T>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape, T idx) {
    return idx;
  }
};
template <int N, int Ndim> struct ArrayIndexRowMajor {
  // this array index provide almost same compiled code. compile it in
  // https://godbolt.org/ for more details.
  template <typename TShape, typename Tinit, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, Tinit start,
                                            T index, Ts... inds) {
    return ArrayIndexRowMajor<N - 1, Ndim>::run(
        shape, (index + start) * shape[Ndim - N + 1], inds...);
  }
  template <typename Tinit, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned
  runShape(const Shape &shape, Tinit start, T index, Ts... inds) {
    return ArrayIndexRowMajor<N - 1, Ndim>::runShape(
        shape, (index + start) * shape[Ndim - N + 1], inds...);
  }
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned
  runPtrs(const TShape *indexes, const TShape *shape, Tinit start) {
    return ArrayIndexRowMajor<N - 1, Ndim>::runPtrs(
        indexes, shape, (indexes[Ndim - N] + start) * shape[Ndim - N + 1]);
  }
};

template <int Ndim> struct ArrayIndexRowMajor<1, Ndim> {
  template <typename TShape, typename Tinit, typename T>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, Tinit start,
                                            T idx) {
    return start + idx;
  }
  template <typename Tinit, typename T>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape,
                                                 Tinit start, T idx) {
    return start + idx;
  }
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned
  runPtrs(const TShape *indexes, const TShape *shape, Tinit start) {
    return start + indexes[Ndim - 1];
  }
};

template <> struct ArrayIndexRowMajor<0, 0> {
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, Tinit start) {
    return 0;
  }
  template <typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape,
                                                 Tinit start) {
    return 0;
  }
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned
  runPtrs(const TShape *indexes, const TShape *shape, Tinit start) {
    return 0;
  }
};

template <int N, int Ndim> struct ArrayIndexStride {
  // this array index provide almost same compiled code. compile it in
  // https://godbolt.org/ for more details.
  template <typename TShape, typename Tinit, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static TV_GLOBAL_INDEX
  run(const TShape *stride, Tinit start, T index, Ts... inds) {
    return ArrayIndexStride<N - 1, Ndim>::run(
        stride, start + index * stride[Ndim - N], inds...);
  }
};

template <int Ndim> struct ArrayIndexStride<1, Ndim> {
  template <typename TShape, typename Tinit, typename T>
  TV_HOST_DEVICE_INLINE static TV_GLOBAL_INDEX run(const TShape *stride,
                                                   Tinit start, T idx) {
    return start + idx * stride[Ndim - 1];
  }
};

#if __cplusplus >= 201703L
template <size_t... N, class T, class... Ts>
TV_HOST_DEVICE_INLINE T array_index_stride(const T *stride, Ts... ids) {
  return ((stride[N] * std::get<N>(std::forward_as_tuple(ids...))) + ...);
}
#endif

namespace detail {

template <typename T, int Rank,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = TV_GLOBAL_INDEX>
struct TensorAccesserBase {
  static constexpr int rank_value = Rank;
  using ptr_t = typename PtrTraits<T>::type;

  static_assert(Rank > 0, "error");

  explicit TV_HOST_DEVICE_INLINE TensorAccesserBase(ptr_t ptr,
                                                    const Tindex *stride_ptr)
      : ptr_(ptr), stride_ptr_(stride_ptr) {}

  TV_HOST_DEVICE_INLINE ptr_t data() { return ptr_; }
  TV_HOST_DEVICE_INLINE const ptr_t data() const { return ptr_; }

  template <class... Inds> TV_HOST_DEVICE_INLINE T &operator()(Inds... inds) {
    static_assert(sizeof...(inds) == Rank, "error");
    return ptr_[ArrayIndexStride<Rank, Rank>::run(stride_ptr_, 0, inds...)];
  }

  template <class... Inds>
  TV_HOST_DEVICE_INLINE const T &operator()(Inds... inds) const {
    static_assert(sizeof...(inds) == Rank, "error");
    return ptr_[ArrayIndexStride<Rank, Rank>::run(stride_ptr_, 0, inds...)];
  }

protected:
  ptr_t ptr_;
  const Tindex *stride_ptr_;
};

template <typename T, int Rank,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = TV_GLOBAL_INDEX>
struct ReadonlyTensorAccesserBase {
  static constexpr int rank_value = Rank;
  using ptr_t = typename PtrTraits<std::add_const_t<T>>::type;

  static_assert(Rank > 0, "error");

  explicit TV_HOST_DEVICE_INLINE
  ReadonlyTensorAccesserBase(ptr_t ptr, const Tindex *stride_ptr)
      : ptr_(ptr), stride_ptr_(stride_ptr) {}

  TV_HOST_DEVICE_INLINE ptr_t data() const { return ptr_; }

  template <class... Inds>
  TV_HOST_DEVICE_INLINE T operator()(Inds... inds) const {
    static_assert(sizeof...(inds) == Rank, "error");
#ifdef __CUDACC__
    return __ldg(
        &ptr_[ArrayIndexStride<Rank, Rank>::run(stride_ptr_, 0, inds...)]);
#else
    return ptr_[ArrayIndexStride<Rank, Rank>::run(stride_ptr_, 0, inds...)];
#endif
  }

protected:
  ptr_t ptr_;
  const Tindex *stride_ptr_;
};

} // namespace detail

template <typename T, int Rank,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = TV_GLOBAL_INDEX>
struct TensorAccesser
    : public detail::TensorAccesserBase<T, Rank, PtrTraits, Tindex> {
  using ptr_t = typename PtrTraits<T>::type;
  static_assert(Rank > 0, "error");
  explicit TV_HOST_DEVICE_INLINE TensorAccesser(ptr_t ptr,
                                                const Tindex *stride_ptr)
      : detail::TensorAccesserBase<T, Rank, PtrTraits, Tindex>(ptr,
                                                               stride_ptr) {}

  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  operator[](int i) {
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        this->ptr_ + this->stride_ptr_[0] * i, this->stride_ptr_ + 1);
  }
  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  operator[](int i) const {
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        this->ptr_ + this->stride_ptr_[0] * i, this->stride_ptr_ + 1);
  }
};

template <typename T, template <class> class PtrTraits, typename Tindex>
struct TensorAccesser<T, 1, PtrTraits, Tindex>
    : public detail::TensorAccesserBase<T, 1, PtrTraits, Tindex> {
  using ptr_t = typename PtrTraits<T>::type;

  explicit TV_HOST_DEVICE_INLINE TensorAccesser(ptr_t ptr,
                                                const Tindex *stride_ptr)
      : detail::TensorAccesserBase<T, 1, PtrTraits, Tindex>(ptr, stride_ptr) {}

  TV_HOST_DEVICE_INLINE T &operator[](int i) {
    return this->ptr_[this->stride_ptr_[0] * i];
  }
  TV_HOST_DEVICE_INLINE const T &operator[](int i) const {
    return this->ptr_[this->stride_ptr_[0] * i];
  }
};

template <typename T, int Rank,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = TV_GLOBAL_INDEX>
struct ReadonlyTensorAccesser
    : public detail::ReadonlyTensorAccesserBase<T, Rank, PtrTraits, Tindex> {
  using ptr_t = typename PtrTraits<std::add_const_t<T>>::type;
  static_assert(Rank > 0, "error");
  explicit TV_HOST_DEVICE_INLINE
  ReadonlyTensorAccesser(ptr_t ptr, const Tindex *stride_ptr)
      : detail::TensorAccesserBase<T, Rank, PtrTraits, Tindex>(ptr,
                                                               stride_ptr) {}

  TV_HOST_DEVICE_INLINE ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  operator[](int i) {
    return ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        this->ptr_ + this->stride_ptr_[0] * i, this->stride_ptr_ + 1);
  }
  TV_HOST_DEVICE_INLINE ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  operator[](int i) const {
    return ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        this->ptr_ + this->stride_ptr_[0] * i, this->stride_ptr_ + 1);
  }
};

template <typename T, template <class> class PtrTraits, typename Tindex>
struct ReadonlyTensorAccesser<T, 1, PtrTraits, Tindex>
    : public detail::ReadonlyTensorAccesserBase<T, 1, PtrTraits, Tindex> {
  using ptr_t = typename PtrTraits<std::add_const_t<T>>::type;

  explicit TV_HOST_DEVICE_INLINE
  ReadonlyTensorAccesser(ptr_t ptr, const Tindex *stride_ptr)
      : detail::ReadonlyTensorAccesserBase<T, 1, PtrTraits, Tindex>(
            ptr, stride_ptr) {}

  TV_HOST_DEVICE_INLINE T operator[](int i) const {
#ifdef __CUDACC__
    return __ldg(&this->ptr_[this->stride_ptr_[0] * i]);
#else
    return this->ptr_[this->stride_ptr_[0] * i];
#endif
  }
};

template <typename T, int Rank = -1,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = TV_GLOBAL_INDEX>
struct TensorView {
  static constexpr int rank_value = Rank;
  using ptr_t = typename PtrTraits<T>::type;
  using tv_shape_t = ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex>;
  using no_cv_type = typename std::remove_cv<T>::type;
  using const_type = TensorView<const no_cv_type, Rank, PtrTraits, Tindex>;
  static_assert(Rank == -1 || Rank > 0, "error");

  TV_HOST_DEVICE_INLINE TensorView() {}
  explicit TV_HOST_DEVICE_INLINE TensorView(ptr_t ptr, tv_shape_t shape)
      : ptr_(ptr), shape_(shape), stride_(shape.stride_rowmajor()) {}
  explicit TV_HOST_DEVICE_INLINE TensorView(ptr_t ptr, tv_shape_t shape,
                                            tv_shape_t stride)
      : ptr_(ptr), shape_(shape), stride_(stride) {}

  template <typename T2 = T,
            typename = typename std::enable_if_t<!std::is_const<T2>::value>>
  TV_HOST_DEVICE_INLINE
  operator std::enable_if_t<!std::is_const<T2>::value, const_type>() {
    return const_type(ptr_, shape_);
  } // conversion function

  template <class... Inds>
  TV_HOST_DEVICE_INLINE T &operator()(Inds... inds) const {
    static_assert(Rank == -1 || sizeof...(inds) == Rank, "error");
#if defined TV_DEBUG
    int idxes[sizeof...(Inds)]{int(inds)...};
    TV_REQUIRE(sizeof...(inds) == shape_.ndim(),
               "you provide %d indexes, but dim is %d\n", sizeof...(inds),
               shape_.ndim());
    for (int i = 0; i < sizeof...(inds); ++i) {
      TV_REQUIRE(idxes[i] >= 0 && idxes[i] < shape_[i],
                 "index-%d(%d) out-of-range: [0, %d)\n", i, idxes[i],
                 shape_[i]);
    }
#endif
    constexpr int Ndim = sizeof...(Inds);
    return ptr_[ArrayIndexStride<Ndim, Ndim>::run(stride_.data(), 0, inds...)];
  }

  TV_HOST_DEVICE_INLINE T &operator()() const {
    static_assert(Rank == -1 || 0 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(ptr_ != nullptr, "you want get value but the view is empty.%s",
               "\n");
    TV_REQUIRE(shape_.ndim() == 0, "you provide 0 indexes, but dim is %ld\n",
               shape_.ndim());
#endif
    return ptr_[0];
  }

  template <class T1> TV_HOST_DEVICE_INLINE T &operator()(T1 i1) const {
    static_assert(Rank == -1 || 1 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 1, "you provide 1 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
#endif
    return ptr_[i1 * stride_[0]];
  }
  template <class T1, class T2>
  TV_HOST_DEVICE_INLINE T &operator()(T1 i1, T2 i2) const {
    static_assert(Rank == -1 || 2 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 2, "you provide 2 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
#endif
    return ptr_[i1 * stride_[0] + i2 * stride_[1]];
  }
  template <class T1, class T2, class T3>
  TV_HOST_DEVICE_INLINE T &operator()(T1 i1, T2 i2, T3 i3) const {
    static_assert(Rank == -1 || 3 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 3, "you provide 3 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
    TV_REQUIRE(i3 >= 0 && i3 < shape_[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), shape_[2]);
#endif
    return ptr_[i1 * stride_[0] + i2 * stride_[1] + i3 * stride_[2]];
  }
  template <class T1, class T2, class T3, class T4>
  TV_HOST_DEVICE_INLINE T &operator()(T1 i1, T2 i2, T3 i3, T4 i4) const {
    static_assert(Rank == -1 || 4 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 4, "you provide 4 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
    TV_REQUIRE(i3 >= 0 && i3 < shape_[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), shape_[2]);
    TV_REQUIRE(i4 >= 0 && i4 < shape_[3],
               "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4), shape_[3]);
#endif
    return ptr_[i1 * stride_[0] + i2 * stride_[1] + i3 * stride_[2] +
                i4 * stride_[3]];
  }

  TV_HOST_DEVICE_INLINE T &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_REQUIRE(idx >= 0 && idx < size(), "index(%d) out-of-range: [0, %ld)\n",
               int(idx), size());
#endif
    return ptr_[idx];
  }

  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  accessor(Tindex idx) {
    static_assert(Rank > 1, "for Rank == 1, use accessor() or just use []");
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        ptr_ + stride_[0] * idx, stride_.data() + 1);
  }
  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank, PtrTraits, Tindex> accessor() {
    static_assert(Rank > 0, "rank must higher than zero");
    return TensorAccesser<T, Rank, PtrTraits, Tindex>(ptr_, stride_.data());
  }
  TV_HOST_DEVICE_INLINE
  TensorAccesser<T, Rank - 1, PtrTraits, Tindex> accessor(Tindex idx) const {
    static_assert(Rank > 1, "for Rank == 1, use accessor() or just use []");
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        ptr_ + stride_[0] * idx, stride_.data() + 1);
  }
  TV_HOST_DEVICE_INLINE
  TensorAccesser<T, Rank, PtrTraits, Tindex> accessor() const {
    static_assert(Rank > 0, "error");
    return TensorAccesser<T, Rank, PtrTraits, Tindex>(
        ptr_, stride_.data(), "rank must higher than zero");
  }

  TV_HOST_DEVICE_INLINE ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  accessor_readonly(Tindex idx) {
    static_assert(Rank > 1, "for Rank == 1, use accessor() or just use []");
    return ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        ptr_ + stride_[0] * idx, stride_.data() + 1);
  }
  TV_HOST_DEVICE_INLINE ReadonlyTensorAccesser<T, Rank, PtrTraits, Tindex>
  accessor_readonly() {
    static_assert(Rank > 0, "rank must higher than zero");
    return ReadonlyTensorAccesser<T, Rank, PtrTraits, Tindex>(ptr_,
                                                              stride_.data());
  }
  TV_HOST_DEVICE_INLINE
  ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  accessor_readonly(Tindex idx) const {
    static_assert(Rank > 1, "for Rank == 1, use accessor() or just use []");
    return ReadonlyTensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        ptr_ + stride_[0] * idx, stride_.data() + 1);
  }
  TV_HOST_DEVICE_INLINE
  ReadonlyTensorAccesser<T, Rank, PtrTraits, Tindex> accessor_readonly() const {
    static_assert(Rank > 0, "error");
    return ReadonlyTensorAccesser<T, Rank, PtrTraits, Tindex>(
        ptr_, stride_.data(), "rank must higher than zero");
  }

  TV_HOST_DEVICE_INLINE const tv_shape_t &strides() const { return stride_; }
  TV_HOST_DEVICE_INLINE auto stride(int idx) const { return stride_[idx]; }

  TV_HOST_DEVICE_INLINE bool empty() const { return ptr_ == nullptr; }
  TV_HOST_DEVICE_INLINE ptr_t data() const { return ptr_; }
  TV_HOST_DEVICE_INLINE const tv_shape_t &shape() const { return shape_; }
  TV_HOST_DEVICE_INLINE const tv_shape_t &stride() const { return stride_; }

  TV_HOST_DEVICE_INLINE auto dim(int idx) const { return shape_[idx]; }
  TV_HOST_DEVICE_INLINE int ndim() const { return shape_.ndim(); }
  template <class... Inds>
  TV_HOST_DEVICE_INLINE
      TensorView<T, Rank == -1 ? -1 : int(sizeof...(Inds)), PtrTraits, Tindex>
      view(Inds... newShapes) const {
    ShapeBase<Rank == -1 ? TV_MAX_DIM : int(sizeof...(Inds)), Tindex> shapes{
        int(newShapes)...};
    for (size_t i = 0; i < sizeof...(newShapes); ++i) {
      if (shapes[i] == -1) {
        shapes[i] = 1;
        shapes[i] = size() / shapes.size();
        break;
      }
    }
    TV_ASSERT(shapes.size() == size());
    return TensorView<T, (Rank == -1 ? -1 : int(sizeof...(Inds))), PtrTraits,
                      Tindex>(ptr_, shapes);
  }
  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex>
  view(Shape shapes) const {
    TV_ASSERT(shapes.size() == size());
    return TensorView<T, -1, PtrTraits, Tindex>(ptr_, shapes);
  }
  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex> squeeze() const {
    return TensorView<T, -1, PtrTraits, Tindex>(ptr_, shape_.squeeze());
  }
  TV_HOST_DEVICE_INLINE
  TensorView<T, (Rank == -1 ? -1 : Rank - 1), PtrTraits, Tindex>
  squeeze(int dim) const {
    return TensorView<T, (Rank == -1 ? -1 : Rank - 1), PtrTraits, Tindex>(
        ptr_, shape_.squeeze() < Rank == -1 ? TV_MAX_DIM : Rank - 1 > (dim));
  }
  TV_HOST_DEVICE_INLINE size_t size() const { return shape_.size(); }

#ifndef __CUDACC_RTC__
  template <typename Os>
  std::string repr(Os &ss, int limit = 1000, int limit_axis = 6) const {
    if (empty())
      return "";
    if (shape_.ndim() == 0) {
      ss << "Tensor[" << type_s<T> << "]" << std::endl;
      ss << *ptr_;
      return ss.str();
    }
    bool enable_limit = size() > limit;

    vecarray<int64_t, TV_MAX_DIM> prev(ndim(), -1);
    vecarray<int64_t, TV_MAX_DIM> nd_index(ndim());
    vecarray<int64_t, TV_MAX_DIM> _shape;
    for (auto s : shape()) {
      _shape.push_back(s);
    }
    ss << "Tensor[" << type_s<T> << "]: shape=" << shape()
       << ", stride=" << strides() << std::endl;
    auto ndimValue = ndim();
    for (int64_t i = 0; i < int64_t(size()); ++i) {
      rowArrayIdxInv(i, nd_index.data(), _shape.data(), ndimValue);
      bool newline = false;
      int end_count = 0;
      for (int j = 0; j < ndimValue; ++j) {
        if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0 &&
            prev[j] != -1) {
          ss << "]";
          ++end_count;
          newline = true;
        }
      }
      if (prev[0] == -1) {
        end_count = ndimValue;
      }
      if (newline) {
        ss << "\n";
      }
      int starts_count = 0;
      for (int j = 0; j < ndimValue; ++j) {
        if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0) {
          ++starts_count;
        }
      }
      if (starts_count > 0) {
        for (int j = 0; j < ndimValue - end_count; ++j) {
          ss << " ";
        }
        for (int j = 0; j < starts_count; ++j) {
          ss << "[";
        }
      }
      if (std::is_same<std::decay_t<T>, uint8_t>::value) {
        ss << unsigned((*this)[i]);
      } else {
        ss << (*this)[i];
      }
      if (nd_index[ndimValue - 1] != _shape[ndimValue - 1] - 1) {
        ss << ",";
      }
      for (int j = 0; j < ndimValue; ++j) {
        prev[j] = nd_index[j];
      }
    }
    for (int j = 0; j < ndimValue; ++j) {
      ss << "]";
    }
    return ss.str();
  }
  std::string repr() const {
    std::ostringstream ss;
    return repr(ss);
  }
#endif
protected:
  template <typename T1> TV_HOST_DEVICE_INLINE Slice to_slice(T1 s) const {
    return Slice{int(s), -1, -1};
  }

  TV_HOST_DEVICE_INLINE Slice to_slice(Slice s) const { return Slice(s); }

  ptr_t ptr_ = nullptr;
  tv_shape_t shape_;
  tv_shape_t stride_;
};
#ifndef __CUDACC_RTC__

template <typename T> TensorView<T> vector2tv(std::vector<T> &arr) {
  return TensorView<T>(arr.data(), {arr.size()});
}

template <typename T>
TensorView<T> vector2tv(std::vector<T> &arr, Shape shape) {
  TV_ASSERT_INVALID_ARG(shape.prod() == arr.size(), "error");
  return TensorView<T>(arr.data(), shape);
}

template <typename T> TensorView<const T> vector2tv(const std::vector<T> &arr) {
  return TensorView<const T>(arr.data(), {arr.size()});
}

template <typename Os, typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
Os &operator<<(Os &os, const TensorView<T, Rank, PtrTraits, Tindex> &dt) {
  os << dt.repr();
  return os;
}

template <typename Os, typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
Os &operator<<(Os &os, const TensorView<const T, Rank, PtrTraits, Tindex> &dt) {
  os << dt.repr();
  return os;
}

namespace detail {
template <typename T> struct TypePrintfFormat;
template <> struct TypePrintfFormat<float> {
  static constexpr const char *value = "%.2f";
};
template <> struct TypePrintfFormat<double> {
  static constexpr const char *value = "%.2f";
};
template <> struct TypePrintfFormat<int8_t> {
  static constexpr const char *value = "%d";
};
template <> struct TypePrintfFormat<int16_t> {
  static constexpr const char *value = "%d";
};
template <> struct TypePrintfFormat<int32_t> {
  static constexpr const char *value = "%d";
};
template <> struct TypePrintfFormat<uint8_t> {
  static constexpr const char *value = "%u";
};
template <> struct TypePrintfFormat<uint16_t> {
  static constexpr const char *value = "%u";
};
template <> struct TypePrintfFormat<uint32_t> {
  static constexpr const char *value = "%u";
};
template <> struct TypePrintfFormat<int64_t> {
  static constexpr const char *value = "%ld";
};
template <> struct TypePrintfFormat<uint64_t> {
  static constexpr const char *value = "%lu";
};
template <> struct TypePrintfFormat<bool> {
  static constexpr const char *value = "%d";
};

template <typename T>
constexpr const char *type_printf_format_v = TypePrintfFormat<T>::value;

}; // namespace detail

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
TV_HOST_DEVICE void
printTensorView(const TensorView<T, Rank, PtrTraits, Tindex> &tensor,
                const char *format) {
  // used to print tensor in cuda kernel.
  if (tensor.empty())
    return;
  if (tensor.ndim() == 0) {
    printf(format, tensor());
    printf("\n");
    return;
  }
  vecarray<int64_t, TV_MAX_DIM> prev(tensor.ndim(), -1);
  vecarray<int64_t, TV_MAX_DIM> nd_index(tensor.ndim());
  vecarray<int64_t, TV_MAX_DIM> shape(tensor.shape());

  auto ndim = tensor.ndim();
  for (int64_t i = 0; i < tensor.size(); ++i) {
    rowArrayIdxInv(i, nd_index.data(), shape.data(), ndim);
    bool newline = false;
    int end_count = 0;
    for (int j = 0; j < ndim; ++j) {
      if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0 &&
          prev[j] != -1) {
        printf("]");
        ++end_count;
        newline = true;
      }
    }
    if (prev[0] == -1) {
      end_count = ndim;
    }
    if (newline) {
      printf("\n");
    }
    int starts_count = 0;
    for (int j = 0; j < ndim; ++j) {
      if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0) {
        ++starts_count;
      }
    }
    if (starts_count > 0) {
      for (int j = 0; j < ndim - end_count; ++j) {
        printf(" ");
      }
      for (int j = 0; j < starts_count; ++j) {
        printf("]");
      }
    }
    printf(format, tensor[i]);
    if (nd_index[ndim - 1] != shape[ndim - 1] - 1) {
      printf(",");
    }
    for (int j = 0; j < ndim; ++j) {
      prev[j] = nd_index[j];
    }
  }
  for (int j = 0; j < ndim; ++j) {
    printf("]");
  }
  printf("\n");
}

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
TV_HOST_DEVICE void
printTensorView(TensorView<T, Rank, PtrTraits, Tindex> tensor) {
  using Traw = typename std::remove_const<T>::type;
  return printTensorView(tensor, detail::type_printf_format_v<Traw>);
}
template <typename T>
TV_HOST_DEVICE void printTensorView(const T *ptr, Shape shape) {
  using Traw = typename std::remove_const<T>::type;
  return printTensorView(TensorView<const T>(ptr, shape),
                         detail::type_printf_format_v<Traw>);
}
template <typename T>
TV_HOST_DEVICE void printTensorView(const T *ptr, Shape shape,
                                    const char *format) {
  return printTensorView(TensorView<const T>(ptr, shape), format);
}
#endif
} // namespace tv