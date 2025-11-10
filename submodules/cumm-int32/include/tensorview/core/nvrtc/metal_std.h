
#pragma once
#pragma METAL internals : enable

#include <metal_stdlib>
namespace std = metal;
namespace metal {
template <typename> struct is_array : false_type {};

template <typename _Tp, std::size_t _Size>
struct is_array<_Tp[_Size]> : true_type {};

template <typename _Tp> struct is_array<_Tp[]> : true_type {};

/// remove_extent
template <typename _Tp> struct remove_extent {
  typedef _Tp type;
};

template <typename _Tp, std::size_t _Size> struct remove_extent<_Tp[_Size]> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_extent<_Tp[]> {
  typedef _Tp type;
};

template <typename _Up, bool _IsArray = is_array<_Up>::value,
          bool _IsFunction = is_function<_Up>::value>
struct __decay_selector;

template <class T> struct is_metal_constant : false_type {};
template <class T> struct is_metal_constant<constant T> : true_type {};

// NB: DR 705.
template <typename _Up> struct __decay_selector<_Up, false, false> {
  typedef conditional_t<is_metal_constant<_Up>::value,
                        const typename remove_cv<_Up>::type,
                        typename remove_cv<_Up>::type>
      __type;
};

template <typename _Up> struct __decay_selector<_Up, true, false> {
  typedef typename remove_extent<_Up>::type device *__type;
};

// add support for add_point need lots of code.
// template<typename _Up>
//   struct __decay_selector<_Up, false, true>
//   { typedef typename add_pointer<_Up>::type __type; };

/// decay
template <typename _Tp> class decay {
  typedef typename remove_reference<_Tp>::type __remove_type;

public:
  typedef typename __decay_selector<__remove_type>::__type type;
};
template <typename _Tp> using decay_t = typename decay<_Tp>::type;
namespace detail {
template <class T> struct remove_metal_address_space_helper;

template <class T> struct remove_metal_address_space_helper<constant T &> {
  typedef T type;
};
template <class T> struct remove_metal_address_space_helper<thread T &> {
  typedef T type;
};

template <class T> struct remove_metal_address_space_helper<threadgroup T &> {
  typedef T type;
};

template <class T> struct remove_metal_address_space_helper<device T &> {
  typedef T type;
};

} // namespace detail
template <class T> struct remove_metal_address_space {
  typedef conditional_t<
      is_function_v<T>, T,
      typename detail::remove_metal_address_space_helper<
          constant std::conditional_t<is_metal_constant<decay_t<T>>::value,
                                      const remove_reference_t<T>,
                                      remove_reference_t<T>> &>::type>
      type;
};
template <class T>
using remove_metal_address_space_t =
    typename remove_metal_address_space<T>::type;

// template <class T>
// inline constant T&& forward(constant typename remove_reference<T>::type& t)
// TV_NOEXCEPT_EXCEPT_METAL
// {
//     return static_cast<constant T&&>(t);
// }
template <class T>
inline thread T &&
forward(thread typename remove_reference<T>::type &t) TV_NOEXCEPT_EXCEPT_METAL {
  return static_cast<thread T &&>(t);
}
template <class T>
inline threadgroup T &&
forward(threadgroup
        typename remove_reference<T>::type &t) TV_NOEXCEPT_EXCEPT_METAL {
  return static_cast<threadgroup T &&>(t);
}
template <class T>
inline device T &&
forward(device typename remove_reference<T>::type &t) TV_NOEXCEPT_EXCEPT_METAL {
  return static_cast<device T &&>(t);
}

// template <class T>
// inline constant T&& forward(constant typename remove_reference<T>::type&& t)
// TV_NOEXCEPT_EXCEPT_METAL
// {
//     static_assert(!is_lvalue_reference<T>::value,
//                   "Can not forward an rvalue as an lvalue.");
//     return static_cast<constant T&&>(t);
// }

template <class T>
inline thread T &&forward(thread typename remove_reference<T>::type &&t)
    TV_NOEXCEPT_EXCEPT_METAL {
  static_assert(!is_lvalue_reference<T>::value,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<thread T &&>(t);
}
template <class T>
inline threadgroup T &&
forward(threadgroup
        typename remove_reference<T>::type &&t) TV_NOEXCEPT_EXCEPT_METAL {
  static_assert(!is_lvalue_reference<T>::value,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<threadgroup T &&>(t);
}
template <class T>
inline device T &&forward(device typename remove_reference<T>::type &&t)
    TV_NOEXCEPT_EXCEPT_METAL {
  static_assert(!is_lvalue_reference<T>::value,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<device T &&>(t);
}

template <typename _Tp> class reference_wrapper;

// Helper which adds a reference to a type when given a reference_wrapper
template <typename _Tp> struct __strip_reference_wrapper {
  typedef _Tp __type;
};

template <typename _Tp>
struct __strip_reference_wrapper<reference_wrapper<_Tp>> {
  typedef thread _Tp &__type;
};

template <typename _Tp> struct __decay_and_strip {
  typedef typename __strip_reference_wrapper<typename decay<_Tp>::type>::__type
      __type;
};

#ifndef M_PI
#define M_PI M_PI_F // metal don't support double for now
#endif
/*
for non-array, non-function, non-point type T,
std::decay_t<T> is:
  const constant T& -> constant T
  constant T& -> constant T
  thread T& -> thread T
  ...

when we use regular template partial specification, the
`constant T` is actually `const constant T`, which isn't equal to `constant T`

 */
} // namespace metal

namespace metal {

// metal don't have tuple due to lack support of inheritance, so we need to
// implement it by a old way.

template <typename T, typename... Ts> struct tuple {
  TV_HOST_DEVICE_INLINE tuple(thread const T &t, thread const Ts &...ts)
      : value(t), rest(ts...) {}

  TV_HOST_DEVICE_INLINE constexpr int size() const { return 1 + rest.size(); }

  T value;
  tuple<Ts...> rest;
};
template <typename T> struct tuple<T> {
  TV_HOST_DEVICE_INLINE tuple(thread const T &t) : value(t) {}

  TV_HOST_DEVICE_INLINE constexpr int size() const { return 1; }

  T value;
};

template <size_t i, typename T, typename... Ts>
struct nthType : nthType<i - 1, Ts...> {
  static_assert(i < sizeof...(Ts) + 1, "index out of bounds");
};

template <typename T, typename... Ts> struct nthType<0, T, Ts...> {
  T value;
};

template <size_t i> struct getter {
  template <typename... Ts>
  TV_HOST_DEVICE_INLINE static constexpr thread decltype(nthType<
                                                         i, Ts...>::value) &
  get(thread tuple<Ts...> &t) {
    return getter<i - 1>::get(t.rest);
  }
};
template <> struct getter<0> {
  template <typename T, typename... Ts>
  TV_HOST_DEVICE_INLINE static constexpr thread T &
  get(thread tuple<T, Ts...> &t) {
    return t.value;
  }
  template <typename T, typename... Ts>
  TV_HOST_DEVICE_INLINE static constexpr const thread T &
  get(const thread tuple<T, Ts...> &t) {
    return t.value;
  }
};

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr thread decltype(nthType<N, Ts...>::value) &
get(thread tuple<Ts...> &t) {
  return getter<N>::get(t);
}
template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr thread const decltype(nthType<N,
                                                              Ts...>::value) &
get(thread const tuple<Ts...> &t) {
  return getter<N>::get(t);
}

template <typename... _Elements>
TV_HOST_DEVICE_INLINE constexpr tuple<
    typename __decay_and_strip<_Elements>::__type...>
make_tuple(thread _Elements &&...__args) {
  typedef tuple<typename __decay_and_strip<_Elements>::__type...> __result_type;
  return __result_type(std::forward<_Elements>(__args)...);
}
} // namespace metal

#pragma METAL internals : disable
