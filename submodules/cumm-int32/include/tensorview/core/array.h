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
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#include <tensorview/core/defs.h>
#include <tensorview/core/mp_helper.h>
#include "const_ops.h"

#ifdef TV_PARALLEL_RTC
#include "nvrtc_std.h"
#else
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>
#endif

#ifdef __METAL_VERSION__ 
#pragma METAL internals : enable

#endif

namespace tv {

template <class T> struct sizeof_bits {
  static constexpr auto TV_METAL_CONSTANT value = sizeof(T) * 8;
};
template <> struct sizeof_bits<bool> { static int const TV_METAL_CONSTANT value = 1; };

template <class T>
constexpr size_t TV_METAL_CONSTANT sizeof_bits_v = sizeof_bits<std::decay_t<T>>::value;




namespace detail {

template <class T>
struct equivalent_data_type {
  using type = T;
};

template <typename T> struct sizeof_subbyte_impl {
  static_assert(sizeof_bits_v<T> % 8 == 0, "error");
  static constexpr int TV_METAL_CONSTANT value = sizeof_bits_v<T> / 8;
};

} // namespace detail

template <class T>
using equivalent_data_type_t = typename detail::equivalent_data_type<T>::type;

template <typename T>
constexpr int TV_METAL_CONSTANT sizeof_v = detail::sizeof_subbyte_impl<T>::value;
template <typename T, size_t N, size_t Align> struct array;
namespace detail {

template <typename T> struct get_array_value_type_or_scalar_impl {
  using type = std::decay_t<T>;
};

template <typename T, size_t N, size_t Align>
struct get_array_value_type_or_scalar_impl<array<T, N, Align>> {
  using type = T;
};

template <typename T> struct get_nested_element_type_impl {
  using type = std::decay_t<T>;
};

template <typename T, size_t N, size_t Align>
struct get_nested_element_type_impl<array<T, N, Align>> {
  using type = typename get_nested_element_type_impl<std::decay_t<T>>::type;
};

template <typename T>
using get_nested_element_t = typename get_nested_element_type_impl<T>::type;

template <typename T>
using get_array_value_type_or_scalar_t = typename get_array_value_type_or_scalar_impl<T>::type;


template <typename K, typename V> struct InitPair {
  K first;
  V second;
};

template <typename K, typename V>
constexpr TV_HOST_DEVICE_INLINE InitPair<K, V> make_init_pair(K k, V v) {
  return InitPair<K, V>{k, v};
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_sum(T l, T r) {
  return l + r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_sub(T l, T r) {
  return l - r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_mul(T l, T r) {
  return l * r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_div(T l, T r) {
  return l / r;
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_minus(T x) {
  return -x;
}

template <typename T, typename TCast>
TV_HOST_DEVICE_INLINE constexpr TCast array_cast(T x) {
  return static_cast<TCast>(x);
}

} // namespace detail

namespace arrayops {
template <class F, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto apply(TV_METAL_THREAD F &&f, TV_METAL_THREAD Args &&...args);
} // namespace arrayops

template <typename T, size_t N, size_t Align = 0> struct array {
  // TODO constexpr slice
  typedef T value_type;
  typedef value_type TV_METAL_CONSTANT *pointer;
  typedef const value_type TV_METAL_CONSTANT *const_pointer;
  typedef value_type TV_METAL_CONSTANT &reference;
  typedef const value_type TV_METAL_CONSTANT &const_reference;
  typedef value_type TV_METAL_CONSTANT *iterator;
  typedef const value_type TV_METAL_CONSTANT *const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
#ifndef TV_PARALLEL_RTC
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#endif
public:
#ifdef TV_DEBUG
  TV_HOST_DEVICE_INLINE T &operator[](int idx) {
    TV_ASSERT(idx >= 0 && idx < N);
    return array_[idx];
  }
  TV_HOST_DEVICE_INLINE const T &operator[](int idx) const {
    TV_ASSERT(idx >= 0 && idx < N);
    return array_[idx];
  }
#else
  TV_HOST_DEVICE_INLINE constexpr reference operator[](int idx) TV_METAL_CONSTANT {
    return array_[idx];
  }

#ifndef TV_METAL_RTC
  TV_HOST_DEVICE_INLINE constexpr const_reference operator[](int idx) const TV_METAL_CONSTANT {
    return array_[idx];
  }
#endif
#ifdef TV_METAL_RTC

  TV_HOST_DEVICE_INLINE constexpr thread T& operator[](int idx) thread {
    return array_[idx];
  }

  TV_HOST_DEVICE_INLINE constexpr threadgroup T& operator[](int idx) threadgroup {
    return array_[idx];
  }

  TV_HOST_DEVICE_INLINE constexpr device T& operator[](int idx) device {
    return array_[idx];
  }

  TV_HOST_DEVICE_INLINE constexpr thread const T& operator[](int idx) const thread {
    return array_[idx];
  }

  TV_HOST_DEVICE_INLINE constexpr threadgroup const T& operator[](int idx) const threadgroup {
    return array_[idx];
  }

  TV_HOST_DEVICE_INLINE constexpr device const T& operator[](int idx) const device {
    return array_[idx];
  }
#endif
#endif
  // constexpr array(T(&a)[N]) : array_(a) {}
  TV_HOST_DEVICE_INLINE constexpr size_t size() const { return N; }

  TV_HOST_DEVICE_INLINE constexpr const TV_METAL_CONSTANT T *data() const TV_METAL_CONSTANT { return array_; }
#ifndef TV_METAL_RTC
  TV_HOST_DEVICE_INLINE constexpr T *data() { return array_; }
#endif
#ifdef TV_METAL_RTC
  TV_HOST_DEVICE_INLINE constexpr thread const T *data() const thread { return array_; }
  TV_HOST_DEVICE_INLINE constexpr thread T *data() thread { return array_; }
  TV_HOST_DEVICE_INLINE constexpr device const T *data() const threadgroup { return array_; }
  TV_HOST_DEVICE_INLINE constexpr device T *data() threadgroup { return array_; }
  TV_HOST_DEVICE_INLINE constexpr threadgroup const T *data() const device { return array_; }
  TV_HOST_DEVICE_INLINE constexpr threadgroup T *data() device { return array_; }

#endif
  // TV_HOST_DEVICE_INLINE constexpr const T[N] data_array() const { return
  // array_; } TV_HOST_DEVICE_INLINE constexpr T[N] data_array() { return
  // array_; }

  TV_HOST_DEVICE_INLINE constexpr size_t empty() const { return N == 0; }

  TV_HOST_DEVICE_INLINE constexpr iterator begin() { return iterator(array_); }
  TV_HOST_DEVICE_INLINE constexpr iterator end() {
    return iterator(array_ + N);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator begin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator end() const {
    return const_iterator(array_ + N);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator cbegin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator cend() const {
    return const_iterator(array_ + N);
  }
#ifndef TV_PARALLEL_RTC
  constexpr const_reverse_iterator crbegin() const TV_NOEXCEPT_EXCEPT_METAL {
    return const_reverse_iterator(end());
  }

  constexpr const_reverse_iterator crend() const TV_NOEXCEPT_EXCEPT_METAL {
    return const_reverse_iterator(begin());
  }

  constexpr reverse_iterator rbegin() TV_NOEXCEPT_EXCEPT_METAL {
    return reverse_iterator(end());
  }

  constexpr const_reverse_iterator rbegin() const TV_NOEXCEPT_EXCEPT_METAL {
    return const_reverse_iterator(end());
  }

  constexpr reverse_iterator rend() TV_NOEXCEPT_EXCEPT_METAL {
    return reverse_iterator(begin());
  }

  constexpr const_reverse_iterator rend() const TV_NOEXCEPT_EXCEPT_METAL {
    return const_reverse_iterator(begin());
  }
#endif
  TV_HOST_DEVICE_INLINE constexpr reference front() TV_NOEXCEPT_EXCEPT_METAL {
    return *begin();
  }

  TV_HOST_DEVICE_INLINE constexpr const_reference front() const TV_NOEXCEPT_EXCEPT_METAL {
    return array_[0];
  }

  TV_HOST_DEVICE_INLINE constexpr reference back() TV_NOEXCEPT_EXCEPT_METAL {
    return N ? *(end() - 1) : *end();
  }

  TV_HOST_DEVICE_INLINE constexpr const_reference back() const TV_NOEXCEPT_EXCEPT_METAL {
    return N ? array_[N - 1] : array_[0];
  }

  TV_HOST_DEVICE_INLINE constexpr void clear() TV_NOEXCEPT_EXCEPT_METAL {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] = T{};
    }
  }

  TV_HOST_DEVICE_INLINE constexpr void fill(const TV_METAL_CONSTANT value_type &__u) {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] = __u;
    }
  }

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT array<T, N, Align> &
  operator+=(TV_METAL_CONSTANT TOther const &other) TV_METAL_CONSTANT {
    return *this = *this + other;
  }

#ifdef TV_METAL_RTC
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr thread array<T, N, Align> &
  operator+=(thread TOther const &other) thread {
    return *this = *this + other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr threadgroup array<T, N, Align> &
  operator+=(threadgroup TOther const &other) threadgroup {
    return *this = *this + other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator+=(device TOther const &other) device {
    return *this = *this + other;
  }

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator+=(thread TOther const &other) device {
    auto lfs = *this;
    return *this = lfs + other;
  }

#endif

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT array<T, N, Align> &
  operator-=(TV_METAL_CONSTANT TOther const &other) TV_METAL_CONSTANT {
    return *this = *this - other;
  }

#ifdef TV_METAL_RTC
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr thread array<T, N, Align> &
  operator-=(thread TOther const &other) thread {
    return *this = *this - other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr threadgroup array<T, N, Align> &
  operator-=(threadgroup TOther const &other) threadgroup {
    return *this = *this - other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator-=(device TOther const &other) device {
    return *this = *this - other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator-=(thread TOther const &other) device {
    auto lfs = *this;
    return *this = lfs - other;
  }
#endif

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT array<T, N, Align> &
  operator*=(TV_METAL_CONSTANT TOther const &other) TV_METAL_CONSTANT {
    return *this = *this * other;
  }

#ifdef TV_METAL_RTC
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr thread array<T, N, Align> &
  operator*=(thread TOther const &other) thread {
    return *this = *this * other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr threadgroup array<T, N, Align> &
  operator*=(threadgroup TOther const &other) threadgroup {
    return *this = *this * other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator*=(device TOther const &other) device {
    return *this = *this * other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator*=(thread TOther const &other) device {
    auto lfs = *this;
    return *this = lfs * other;
  }
#endif

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT array<T, N, Align> &
  operator/=(TV_METAL_CONSTANT TOther const &other) TV_METAL_CONSTANT {
    return *this = *this / other;
  }

#ifdef TV_METAL_RTC
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr thread array<T, N, Align> &
  operator/=(thread TOther const &other) thread {
    return *this = *this / other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr threadgroup array<T, N, Align> &
  operator/=(threadgroup TOther const &other) threadgroup {
    return *this = *this / other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator/=(device TOther const &other) device {
    return *this = *this / other;
  }
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr device array<T, N, Align> &
  operator/=(thread TOther const &other) device {
    auto lfs = *this;
    return *this = lfs / other;
  }
#endif

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> operator-() const {
    return arrayops::apply(detail::array_minus<detail::get_nested_element_t<T>>, *this);
  }

  // TODO add limit to TCast, it must not be array type
  template <typename TCast>
  TV_HOST_DEVICE_INLINE constexpr auto cast() const {
    return arrayops::apply(detail::array_cast<detail::get_nested_element_t<T>, TCast>, *this);
  }


#ifdef TV_METAL_RTC
  template <template <class, size_t, size_t> class Op, class... Args>
  TV_HOST_DEVICE_INLINE constexpr auto op(thread Args &&...args) {
    return Op<T, N, Align>()(*this, args...);
  }
  template <template <class, size_t, size_t> class Op, class... Args>
  TV_HOST_DEVICE_INLINE constexpr auto op(thread Args &&...args) const {
    return Op<T, N, Align>()(*this, args...);
  }
  // TODO how to fix this in metal?
  // template <template <class, size_t, size_t> class Op>
  // TV_HOST_DEVICE_INLINE constexpr auto op() {
  //   return Op<T, N, Align>()(*this);
  // }

  // template <template <class, size_t, size_t> class Op>
  // TV_HOST_DEVICE_INLINE constexpr auto op() const {
  //   return Op<T, N, Align>()(*this);
  // }

  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(TV_METAL_CONSTANT Args &&...args, thread typename std::enable_if<sizeof...(Args) != 0>::type * = 0) {
  //   return Op<T, N, Align>()(*this, args...);
  // }

  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(thread Args &&...args, thread typename std::enable_if<sizeof...(Args) != 0>::type * = 0) {
  //   return Op<T, N, Align>()(*this, args...);
  // }

  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(threadgroup Args &&...args, thread typename std::enable_if<sizeof...(Args) != 0>::type * = 0) {
  //   return Op<T, N, Align>()(*this, args...);
  // }

  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(device Args &&...args, thread typename std::enable_if<sizeof...(Args) != 0>::type * = 0) {
  //   return Op<T, N, Align>()(*this, args...);
  // }
  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(thread Args &&...args, thread typename std::enable_if<sizeof...(Args) != 0>::type * = 0) const {
  //   return Op<T, N, Align>()(*this, args...);
  // }

  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(threadgroup Args &&...args, thread typename std::enable_if<sizeof...(Args) != 0>::type * = 0) const {
  //   return Op<T, N, Align>()(*this, args...);
  // }

  // template <template <class, size_t, size_t> class Op, class... Args>
  // TV_HOST_DEVICE_INLINE constexpr auto op(device Args &&...args, thread  typename std::enable_if<sizeof...(Args) != 0>::type * = 0) const {
  //   return Op<T, N, Align>()(*this, args...);
  // }

#else 
  template <template <class, size_t, size_t> class Op, class... Args>
  TV_HOST_DEVICE_INLINE constexpr auto op(Args &&...args) {
    return Op<T, N, Align>()(*this, std::forward<Args>(args)...);
  }
  template <template <class, size_t, size_t> class Op, class... Args>
  TV_HOST_DEVICE_INLINE constexpr auto op(Args &&...args) const {
    return Op<T, N, Align>()(*this, std::forward<Args>(args)...);
  }

#endif


  T array_[N];
};
#ifndef TV_METAL_RTC
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr bool operator==(const array<T, N, Align> &lfs,
                                                const array<T, N, Align> &rfs) {

  for (size_t i = 0; i < N; ++i) {
    if (lfs[i] != rfs[i])
      return false;
  }
  return true;
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr bool operator!=(const array<T, N, Align> &lfs,
                                                const array<T, N, Align> &rfs) {

  return !(lfs == rfs);
}
#endif
namespace detail {
// mp_list_c to array
template <class... Is> struct mp_list_c_to_array_impl {
  static constexpr TV_METAL_CONSTANT array<typename mp_nth_t<0, Is...>::value_type, sizeof...(Is)>
      value{{Is::value...}};
};

// https://stackoverflow.com/questions/19936841/initialize-a-constexpr-array-as-sum-of-other-two-constexpr-arrays
template <int... Is> struct seq {};
template <int I, int... Is> struct gen_seq : gen_seq<I - 1, I - 1, Is...> {};
template <int... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT const T &
array_or_scalar(TV_METAL_CONSTANT const array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}

#ifdef TV_METAL_RTC
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr thread const T &
array_or_scalar(thread const array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr device const T &
array_or_scalar(device const array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr threadgroup const T &
array_or_scalar(threadgroup const array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}

#endif

#ifndef __CUDACC_RTC__
template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr const TV_METAL_THREAD T &
array_or_scalar(const TV_METAL_THREAD std::array<T, N> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}
#endif

template <typename T>
TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT const T &array_or_scalar(TV_METAL_CONSTANT const T &arr, int i) {
  return arr;
}
#ifdef TV_METAL_RTC
template <typename T>
TV_HOST_DEVICE_INLINE constexpr thread const T &array_or_scalar(thread const T &arr, int i) {
  return arr;
}

template <typename T>
TV_HOST_DEVICE_INLINE constexpr device const T &array_or_scalar(device const T &arr, int i) {
  return arr;
}

template <typename T>
TV_HOST_DEVICE_INLINE constexpr threadgroup const T &array_or_scalar(threadgroup const T &arr, int i) {
  return arr;
}

#endif
#ifndef TV_METAL_RTC
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT T &array_or_scalar(TV_METAL_CONSTANT array<T, N, Align> &arr,
                                                   int i) {
  return arr[N == 1 ? 0 : i];
}
#endif
#ifdef TV_METAL_RTC
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr thread T &array_or_scalar(thread array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr device T &array_or_scalar(device array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}
template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr threadgroup T &array_or_scalar(threadgroup array<T, N, Align> &arr, int i) {
  return arr[N == 1 ? 0 : i];
}

#endif
#ifndef TV_METAL_RTC
template <typename T>
TV_HOST_DEVICE_INLINE constexpr TV_METAL_CONSTANT T &array_or_scalar(TV_METAL_CONSTANT T &arr, int i) {
  return arr;
}
#endif
#ifdef TV_METAL_RTC
template <typename T>
TV_HOST_DEVICE_INLINE constexpr thread T &array_or_scalar(thread T &arr, int i) {
  return arr;
}

template <typename T>
TV_HOST_DEVICE_INLINE constexpr device T &array_or_scalar(device T &arr, int i) {
  return arr;
}

template <typename T>
TV_HOST_DEVICE_INLINE constexpr threadgroup T &array_or_scalar(threadgroup T &arr, int i) {
  return arr;
}

#endif
template <typename T> struct is_tv_array {
  static constexpr TV_METAL_CONSTANT bool value = false;
};

template <typename T, size_t N, size_t Align>
struct is_tv_array<tv::array<T, N, Align>> {
  static constexpr TV_METAL_CONSTANT bool value = true;
};


template <typename TCast, typename T> struct cast_nested_element_type_impl {
  using type = TCast;
};

template <typename TCast, typename T, size_t N, size_t Align>
struct cast_nested_element_type_impl<TCast, array<T, N, Align>> {
  using type = array<
      typename cast_nested_element_type_impl<TCast, std::decay_t<T>>::type, N>;
};

template <typename T, typename TCast>
using cast_nested_element_t =
    typename cast_nested_element_type_impl<TCast, T>::type;

template <class T>
struct get_tv_array_rank : public std::integral_constant<std::size_t, 0> {};

template <class T, size_t N, size_t Align>
struct get_tv_array_rank<array<T, N, Align>>
    : public std::integral_constant<std::size_t,
                                    get_tv_array_rank<T>::value + 1> {};

template <typename T> struct get_tv_array_shape { using type = mp_list<>; };

template <typename T, size_t N, size_t Align>
struct get_tv_array_shape<tv::array<T, N, Align>> {
  using type = mp_insert_front<typename get_tv_array_shape<T>::type,
                               std::integral_constant<std::size_t, N>>;
};

template <typename T> struct _get_tv_value_type { using type = void; };

template <typename T, size_t N, size_t Align>
struct _get_tv_value_type<tv::array<T, N, Align>> {
  using type = T;
};

template <typename TA, typename TB> struct determine_array_type {
  using __ta = typename std::decay<TA>::type;
  using __tb = typename std::decay<TB>::type;

  using type = typename std::conditional<
      is_tv_array<__ta>::value, __ta,
      typename std::conditional<is_tv_array<__tb>::value, __tb,
                                __tb>::type>::type;
};

template <typename T> struct get_array_extent {
  static constexpr TV_METAL_CONSTANT int value = -1;
};

#ifndef __CUDACC_RTC__
// std::array don't support nvrtc.
template <typename T, size_t N> struct get_array_extent<std::array<T, N>> {
  static constexpr TV_METAL_CONSTANT int value = N;
};
#endif

template <typename T, size_t N, size_t Align>
struct get_array_extent<array<T, N, Align>> {
  static constexpr TV_METAL_CONSTANT int value = N;
};

#ifdef TV_METAL_RTC
template <typename T, size_t N, size_t Align>
struct get_array_extent<TV_METAL_CONSTANT array<T, N, Align>> {
  static constexpr TV_METAL_CONSTANT int value = N;
};

#endif

template <class T> struct get_extent_helper_impl {
  static constexpr auto TV_METAL_CONSTANT __value1 = get_array_extent<const std::remove_cv_t<std::decay_t<T>>>::value;
  static constexpr auto TV_METAL_CONSTANT __value2 = get_array_extent<std::remove_cv_t<std::decay_t<T>>>::value;

    using type =
      std::integral_constant<int, __value1 == -1 ? __value2 : __value1>;
};
template <class T>
using get_extent_helper = typename get_extent_helper_impl<T>::type;

template <class... Ts>
constexpr TV_METAL_CONSTANT int get_max_extent_v =
    mp_reduce_max<mp_transform<get_extent_helper, mp_list<Ts...>>,
                  std::integral_constant<int, -1>>::value;


template <int N, class TElement, class... Args>
struct determine_broadcast_array_type_impl {
  static constexpr TV_METAL_CONSTANT int NShape = get_max_extent_v<Args...>;
  static_assert(NShape > 0, "error");
  using extent_is_valid_t = mp_list_c<bool, (get_extent_helper<Args>::type::value == -1 || detail::get_extent_helper<Args>::type::value == 1 || detail::get_extent_helper<Args>::type::value == NShape)...>;
  static_assert(true == mp_reduce_and<extent_is_valid_t, std::integral_constant<bool, true>>::value, "array shape must valid");
  using type = array<typename determine_broadcast_array_type_impl<N - 1, TElement, 
    typename std::conditional<is_tv_array<Args>::value, typename detail::get_array_value_type_or_scalar_t<Args>,
                                  Args>::type...>::type, NShape>;
};

template <class TElement, class... Args>
struct determine_broadcast_array_type_impl<0, TElement, Args...>{
  static constexpr TV_METAL_CONSTANT int N = get_max_extent_v<Args...>;
  using type = typename std::conditional<(N <= 0), TElement, array<TElement, size_t(N)>>::type;
};


template <class TElement, class... Args>
struct determine_broadcast_array_type {
  static constexpr TV_METAL_CONSTANT int Rank = mp_reduce_max<mp_transform<get_tv_array_rank, mp_list<std::decay_t<Args>...>>,
                                    std::integral_constant<size_t, 0>>::value;
  static_assert(Rank >= 0, "error");
  using type = typename determine_broadcast_array_type_impl<Rank, TElement, std::decay_t<Args>...>::type;
};


template <bool Enable> struct invoke_or_recursive;

template <> struct invoke_or_recursive<true> {
  template <class F, class... Args>
  TV_HOST_DEVICE_INLINE static constexpr auto run(TV_METAL_THREAD F &&f, int i,
                                                  TV_METAL_THREAD Args &&...args) {
    return arrayops::apply(TV_FORWARD_EXCEPT_METAL(F, f),
                           array_or_scalar(std::forward<Args>(args), i)...);
  }
};

template <> struct invoke_or_recursive<false> {
  template <class F, class... Args>
  TV_HOST_DEVICE_INLINE static constexpr auto run(TV_METAL_THREAD F &&f, int i,
                                                  TV_METAL_THREAD Args &&...args) {
    return TV_FORWARD_EXCEPT_METAL(F, f)(array_or_scalar(std::forward<Args>(args), i)...);
  }
};

template <class F, int... Is, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto index_transform_impl(TV_METAL_THREAD F &&f, seq<Is...>,
                                                          TV_METAL_THREAD Args &&...args)
    -> typename determine_broadcast_array_type<std::decay_t<return_type_t<F>>, Args...>::type {
  using arr_type_t = typename determine_broadcast_array_type<std::decay_t<return_type_t<F>>, Args...>::type;
  // arr_type_t::asdwasfasf;
  // using X = mp_list<Args...>;
  // X::asdasfasfa;
  // arr_type_t::asdwasfasf;
  static_assert(!std::is_same<arr_type_t, void>::value, "wtf");
  return {{invoke_or_recursive<(get_tv_array_rank<arr_type_t>::value > 1)>::run(
      TV_FORWARD_EXCEPT_METAL(F, f), Is, std::forward<Args>(args)...)...}};
}
// template <class F, int... Is, class... Args>
// TV_HOST_DEVICE_INLINE constexpr auto index_transform_impl(TV_METAL_THREAD F &&f, seq<Is...>,
//                                                           TV_METAL_CONSTANT Args &&...args)
//     -> cast_nested_element_t<
//         array<typename mp_reduce<determine_array_type, mp_list<Args...>,
//                                  void>::value_type,
//               sizeof...(Is)>,
//         std::decay_t<return_type_t<F>>> {
//   using arr_type_t = cast_nested_element_t<
//       mp_reduce<determine_array_type, mp_list<Args...>, void>,
//       std::decay_t<return_type_t<F>>>;
//   static_assert(!std::is_same<arr_type_t, void>::value, "wtf");
//   return {{invoke_or_recursive<(get_tv_array_rank<arr_type_t>::value > 1)>::run(
//       TV_FORWARD_EXCEPT_METAL(F, f), Is, std::forward<Args>(args)...)...}};
// }

// we can't use std::extent here because the default value of extent is ZERO...

template <size_t N> struct array_reduce_impl {
  static_assert(N != 0, "N can't equal to zero");
  template <typename F, typename T, size_t N1, size_t Align>
  TV_HOST_DEVICE_INLINE static constexpr T run(TV_METAL_THREAD F &&f,
                                               TV_METAL_THREAD const array<T, N1, Align> &arr) {
    return TV_FORWARD_EXCEPT_METAL(F, f)(
        array_reduce_impl<N - 1>::run(TV_FORWARD_EXCEPT_METAL(F, f), arr), arr[N - 1]);
  }
};

template <> struct array_reduce_impl<1> {
  template <typename F, typename T, size_t N1, size_t Align>
  TV_HOST_DEVICE_INLINE static constexpr T run(TV_METAL_THREAD F &&f,
                                               TV_METAL_THREAD const array<T, N1, Align> &arr) {
    return arr[0];
  }
};

} // namespace detail

template <typename T>
constexpr TV_METAL_CONSTANT bool is_tv_array_v = detail::is_tv_array<std::decay_t<T>>::value;

namespace arrayops {
template <class F, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto apply(TV_METAL_THREAD F &&f, TV_METAL_THREAD Args &&...args) {
  constexpr int N = tv::detail::get_max_extent_v<Args...>;
  static_assert(N > 0, "error");
  using extent_is_valid_t = mp_list_c<bool, (tv::detail::get_extent_helper<Args>::type::value == -1 || tv::detail::get_extent_helper<Args>::type::value == 1 || tv::detail::get_extent_helper<Args>::type::value == N)...>;
  static_assert(true == mp_reduce_and<extent_is_valid_t, std::integral_constant<bool, true>>::value, "array shape must valid");
  return tv::detail::index_transform_impl(TV_FORWARD_EXCEPT_METAL(F, f), tv::detail::gen_seq<N>{},
                                      std::forward<Args>(args)...);
}
// template <class F, class... Args>
// TV_HOST_DEVICE_INLINE constexpr auto apply(TV_METAL_THREAD F &&f, TV_METAL_CONSTANT Args &&...args) {
//   constexpr int N = detail::get_max_extent_v<Args...>;
//   static_assert(N > 0, "error");
//   return detail::index_transform_impl(TV_FORWARD_EXCEPT_METAL(F, f), detail::gen_seq<N>{},
//                                       std::forward<Args>(args)...);
// }

} // namespace arrayops
namespace detail {

template <typename T, size_t N, size_t... Ns> struct nested_array_type {
  using type = array<typename nested_array_type<T, Ns...>::type, N>;
};

template <typename T, size_t N> struct nested_array_type<T, N> {
  using type = array<T, N>;
};
} // namespace detail

template <typename T, size_t... Ns>
using array_nd = typename detail::nested_array_type<T, Ns...>::type;

#ifndef __CUDACC_RTC__
// std::array don't support nvrtc.
namespace detail {

template <typename T> struct nested_conversion {
  using type = std::decay_t<T>;
};

template <typename T, size_t N> struct nested_conversion<std::array<T, N>> {
  using type = tv::array<typename nested_conversion<std::decay_t<T>>::type, N>;
};

template <typename T, size_t N> struct nested_conversion<tv::array<T, N>> {
  using type = std::array<typename nested_conversion<std::decay_t<T>>::type, N>;
};

template <typename T>
using nested_conversion_t = typename nested_conversion<std::decay_t<T>>::type;

template <typename T, size_t N>
constexpr nested_conversion_t<std::array<T, N>>
from_std_array_impl_expand(const TV_METAL_THREAD std::array<T, N> &arr);
template <typename T> constexpr T from_std_array_impl_expand(const TV_METAL_THREAD T &arr) {
  return arr;
}

template <typename T, size_t N>
constexpr nested_conversion_t<tv::array<T, N>>
to_std_array_impl_expand(const TV_METAL_THREAD tv::array<T, N> &arr);
template <typename T> constexpr T to_std_array_impl_expand(const TV_METAL_THREAD T &arr) {
  return arr;
}

template <typename T, size_t N, int... Inds>
constexpr auto from_std_array_impl(const TV_METAL_THREAD std::array<T, N> &arr,
                                   mp_list_int<Inds...>) {
  return nested_conversion_t<std::array<T, N>>{
      from_std_array_impl_expand(arr[Inds])...};
}

template <typename T, size_t N>
constexpr nested_conversion_t<std::array<T, N>>
from_std_array_impl_expand(const TV_METAL_THREAD std::array<T, N> &arr) {
  return from_std_array_impl(arr, mp_make_list_c_sequence<int, N>{});
}

template <typename T, size_t N, int... Inds>
constexpr auto to_std_array_impl(const TV_METAL_THREAD tv::array<T, N> &arr,
                                 mp_list_int<Inds...>) {
  return nested_conversion_t<tv::array<T, N>>{
      to_std_array_impl_expand(arr[Inds])...};
}

template <typename T, size_t N>
constexpr nested_conversion_t<tv::array<T, N>>
to_std_array_impl_expand(const TV_METAL_THREAD tv::array<T, N> &arr) {
  return to_std_array_impl(arr, mp_make_list_c_sequence<int, N>{});
}

} // namespace detail

namespace arrayops {
template <typename T, size_t N>
constexpr detail::nested_conversion_t<std::array<T, N>>
from_std_array(const TV_METAL_THREAD std::array<T, N> &arr) {
  return detail::from_std_array_impl_expand(arr);
}
template <typename T, size_t N>
constexpr detail::nested_conversion_t<tv::array<T, N>>
to_std_array(const TV_METAL_THREAD tv::array<T, N> &arr) {
  return detail::to_std_array_impl_expand(arr);
}
} // namespace arrayops
#endif

template <class L>
constexpr TV_METAL_CONSTANT auto mp_list_c_to_array =
    mp_rename_v<L, detail::mp_list_c_to_array_impl>;
template <class Array>
constexpr TV_METAL_CONSTANT auto array_size_v = detail::get_array_extent<Array>::value;

template <int... Is>
constexpr TV_METAL_CONSTANT auto mp_array_int_v = tv::mp_list_c_to_array<tv::mp_list_int<Is...>>;

template <typename T, typename T2, size_t N1, size_t N2, size_t Align>
TV_HOST_DEVICE_INLINE constexpr auto
operator+(TV_METAL_THREAD const array<T, N1, Align> &lfs, TV_METAL_THREAD const array<T2, N2, Align> &rfs) {
  return arrayops::apply(detail::array_sum<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator+(TV_METAL_THREAD const array<T, N, Align> &lfs, TV_METAL_THREAD const T2 &rfs) {
  return arrayops::apply(detail::array_sum<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator+(TV_METAL_THREAD const T2 &lfs, TV_METAL_THREAD const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_sum<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, typename T2, size_t N1, size_t N2, size_t Align>
TV_HOST_DEVICE_INLINE constexpr auto
operator-(TV_METAL_THREAD const array<T, N1, Align> &lfs, TV_METAL_THREAD const array<T2, N2, Align> &rfs) {
  return arrayops::apply(detail::array_sub<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator-(TV_METAL_THREAD const array<T, N, Align> &lfs, TV_METAL_THREAD const T2 &rfs) {
  return arrayops::apply(detail::array_sub<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator-(TV_METAL_THREAD const T2 &lfs, TV_METAL_THREAD const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_sub<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, typename T2, size_t N1, size_t N2, size_t Align>
TV_HOST_DEVICE_INLINE constexpr auto
operator*(TV_METAL_THREAD const array<T, N1, Align> &lfs, TV_METAL_THREAD const array<T2, N2, Align> &rfs) {
  return arrayops::apply(detail::array_mul<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator*(TV_METAL_THREAD const array<T, N, Align> &lfs, TV_METAL_THREAD const T2 &rfs) {
  return arrayops::apply(detail::array_mul<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator*(TV_METAL_THREAD const T2 &lfs, TV_METAL_THREAD const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_mul<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, typename T2, size_t N1, size_t N2, size_t Align>
TV_HOST_DEVICE_INLINE constexpr auto
operator/(TV_METAL_THREAD const array<T, N1, Align> &lfs, TV_METAL_THREAD const array<T2, N2, Align> &rfs) {
  return arrayops::apply(detail::array_div<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator/(TV_METAL_THREAD const array<T, N, Align> &lfs, TV_METAL_THREAD const T2 &rfs) {
  return arrayops::apply(detail::array_div<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator/(TV_METAL_THREAD const T2 &lfs, TV_METAL_THREAD const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_div<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

// TODO sub-byte type
template <typename T, size_t N, size_t Align = sizeof_v<T> *N>
struct alignas(Align) alignedarray : public array<T, N> {};

namespace detail {
template <typename T, size_t N1, size_t N2, int... IndsL, int... IndsR>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
concat_impl(TV_METAL_THREAD const array<T, N1> &a, TV_METAL_THREAD const array<T, N2> &b, mp_list_int<IndsL...>,
            mp_list_int<IndsR...>) TV_NOEXCEPT_EXCEPT_METAL {
  return array<T, N1 + N2>{a[IndsL]..., b[IndsR]...};
}

template <typename T, size_t N, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, N>
reverse_impl(TV_METAL_THREAD const array<T, N> &a, mp_list_int<Inds...>) TV_NOEXCEPT_EXCEPT_METAL {
  return array<T, N>{a[Inds]...};
}

template <typename T, size_t N, size_t Align, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
constant_impl(T val, mp_list_int<Inds...>) TV_NOEXCEPT_EXCEPT_METAL {
  return array<T, N, Align>{(Inds, val)...};
}

template <typename T, size_t N, size_t Align, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
arange_impl(mp_list_int<Inds...>) TV_NOEXCEPT_EXCEPT_METAL {
  return array<T, N, Align>{T(Inds)...};
}


template <typename T, size_t N, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, sizeof...(Inds)>
slice_impl(TV_METAL_THREAD const array<T, N> &arr, mp_list_int<Inds...>) TV_NOEXCEPT_EXCEPT_METAL {
  return array<T, sizeof...(Inds)>{arr[Inds]...};
}

template <int Start, int End, typename T, size_t N1, size_t N2, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<array<T, End - Start>, sizeof...(Inds)>
slice_2d_impl(TV_METAL_THREAD const array<array<T, N2>, N1> &arr, mp_list_int<Inds...>) TV_NOEXCEPT_EXCEPT_METAL {
  return array<array<T, End - Start>, sizeof...(Inds)>{slice_impl(arr[Inds], mp_list_int_range<Start, End>{})...};
}


template <typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
concat2_base_impl(TV_METAL_THREAD const array<T, N1> &a, TV_METAL_THREAD const array<T, N2> &b) TV_NOEXCEPT_EXCEPT_METAL {
  return concat_impl(a, b, mp_make_list_c_sequence<int, N1>{},
                     mp_make_list_c_sequence<int, N2>{});
}

} // namespace detail

namespace arrayops {

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr array<T, N>
concat(TV_METAL_THREAD const array<T, N> &arr) TV_NOEXCEPT_EXCEPT_METAL {
  return arr;
}

template <typename T, size_t N, size_t... Ns>
TV_HOST_DEVICE_INLINE constexpr array<T, mp_reduce_sum_v<size_t, N, Ns...>>
concat(TV_METAL_THREAD const array<T, N> &arr, TV_METAL_THREAD const array<T, Ns> &...arrs) TV_NOEXCEPT_EXCEPT_METAL {
  return detail::concat2_base_impl(arr, concat(arrs...));
}

template <typename F, typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr T reduce(TV_METAL_THREAD F &&f,
                                         TV_METAL_THREAD const array<T, N, Align> &a) TV_NOEXCEPT_EXCEPT_METAL {
  return tv::detail::array_reduce_impl<N>::run(TV_FORWARD_EXCEPT_METAL(F, f), a);
}

template <typename T, size_t N, size_t Align = 0>
TV_HOST_DEVICE_INLINE constexpr auto constant_array(T val) TV_NOEXCEPT_EXCEPT_METAL {
  return tv::detail::constant_impl<T, N, Align>(
      val, mp_make_list_c_sequence_reverse<int, N>{});
}

template <typename T, size_t N, size_t Align = 0>
TV_HOST_DEVICE_INLINE constexpr auto arange_array() TV_NOEXCEPT_EXCEPT_METAL {
  return tv::detail::arange_impl<T, N, Align>(
      mp_make_list_c_sequence<int, N>{});
}


template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr array<T, N>
reverse(TV_METAL_THREAD const array<T, N> &a) TV_NOEXCEPT_EXCEPT_METAL {
  return tv::detail::reverse_impl(a, mp_make_list_c_sequence_reverse<int, N>{});
}
template <int Start, int End, typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr array<
    T, (End > 0 ? End : End + int64_t(N)) - Start>
slice(TV_METAL_THREAD const array<T, N> &arr) {
  // TODO this function have problem in intellisense engine
  return tv::detail::slice_impl(
      arr, mp_list_int_range<Start, (End > 0 ? End : End + int64_t(N))>{});
}

template <int Start1, int Start2, int End1, int End2, typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<
    array<T, (End2 > 0 ? End2 : End2 + int64_t(N2)) - Start2>, (End1 > 0 ? End1 : End1 + int64_t(N1)) - Start1>
slice_2d(TV_METAL_THREAD const array<array<T, N2>, N1> &arr) {
  return tv::detail::slice_2d_impl<Start2, (End2 > 0 ? End2 : End2 + int64_t(N2))>(
      arr, mp_list_int_range<Start1, (End1 > 0 ? End1 : End1 + int64_t(N1))>{});
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr bool contains(T val,
                                              TV_METAL_THREAD const array<T, N> &a) TV_NOEXCEPT_EXCEPT_METAL {
  bool res = false;
  TV_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    if (a[i] == val) {
      res = true;
    }
  }
  return res;
}

} // namespace arrayops

template <typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
operator|(TV_METAL_THREAD const array<T, N1> &lfs, TV_METAL_THREAD const array<T, N2> &rfs) {
  return arrayops::concat(lfs, rfs);
}

} // namespace tv

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef __METAL_VERSION__ 
#pragma METAL internals : disable

#endif
