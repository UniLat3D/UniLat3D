// Copyright (C) 2008-2021 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

#pragma once

#include "core.h"
#include <tensorview/core/defs.h>

#ifdef __CUDACC_RTC__

namespace std {

template <typename _Tp, _Tp __v> struct integral_constant {
  static constexpr _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant<_Tp, __v> type;
  TV_HOST_DEVICE_INLINE constexpr operator value_type() const noexcept {
    return value;
  }
  TV_HOST_DEVICE_INLINE constexpr value_type operator()() const noexcept {
    return value;
  }
};
/// The type used as a compile-time boolean with true value.
typedef integral_constant<bool, true> true_type;

/// The type used as a compile-time boolean with false value.
typedef integral_constant<bool, false> false_type;

template <bool __v> using __bool_constant = integral_constant<bool, __v>;

template <bool _Cond, typename _Iftrue, typename _Iffalse> struct conditional {
  typedef _Iftrue type;
};

// Partial specialization for false.
template <typename _Iftrue, typename _Iffalse>
struct conditional<false, _Iftrue, _Iffalse> {
  typedef _Iffalse type;
};

template <bool, typename, typename> struct conditional;

template <typename...> struct __or_;

template <> struct __or_<> : public false_type {};

template <typename _B1> struct __or_<_B1> : public _B1 {};

template <typename _B1, typename _B2>
struct __or_<_B1, _B2> : public conditional<_B1::value, _B1, _B2>::type {};

template <typename _B1, typename _B2, typename _B3, typename... _Bn>
struct __or_<_B1, _B2, _B3, _Bn...>
    : public conditional<_B1::value, _B1, __or_<_B2, _B3, _Bn...>>::type {};

template <typename...> struct __and_;

template <> struct __and_<> : public true_type {};

template <typename _B1> struct __and_<_B1> : public _B1 {};

template <typename _B1, typename _B2>
struct __and_<_B1, _B2> : public conditional<_B1::value, _B2, _B1>::type {};

template <typename _B1, typename _B2, typename _B3, typename... _Bn>
struct __and_<_B1, _B2, _B3, _Bn...>
    : public conditional<_B1::value, __and_<_B2, _B3, _Bn...>, _B1>::type {};
template <typename _Pp>
struct __not_ : public __bool_constant<!bool(_Pp::value)> {};

template <typename _Tp> struct remove_const {
  typedef _Tp type;
};

template <typename _Tp> struct remove_const<_Tp const> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_volatile {
  typedef _Tp type;
};

template <typename _Tp> struct remove_volatile<_Tp volatile> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_cv {
  typedef typename remove_const<typename remove_volatile<_Tp>::type>::type type;
};

template <typename _Tp> struct add_const {
  typedef _Tp const type;
};

/// add_volatile
template <typename _Tp> struct add_volatile {
  typedef _Tp volatile type;
};

/// add_cv
template <typename _Tp> struct add_cv {
  typedef typename add_const<typename add_volatile<_Tp>::type>::type type;
};

template <typename _Tp> using remove_const_t = typename remove_const<_Tp>::type;

/// Alias template for remove_volatile
template <typename _Tp>
using remove_volatile_t = typename remove_volatile<_Tp>::type;

/// Alias template for remove_cv
template <typename _Tp> using remove_cv_t = typename remove_cv<_Tp>::type;

/// Alias template for add_const
template <typename _Tp> using add_const_t = typename add_const<_Tp>::type;

/// Alias template for add_volatile
template <typename _Tp> using add_volatile_t = typename add_volatile<_Tp>::type;

/// Alias template for add_cv
template <typename _Tp> using add_cv_t = typename add_cv<_Tp>::type;
/// is_const
template <typename> struct is_const : public false_type {};

template <typename _Tp> struct is_const<_Tp const> : public true_type {};

/// is_volatile
template <typename> struct is_volatile : public false_type {};

template <typename _Tp> struct is_volatile<_Tp volatile> : public true_type {};

template <typename _Tp, typename> struct __remove_pointer_helper {
  typedef _Tp type;
};

template <typename _Tp, typename _Up>
struct __remove_pointer_helper<_Tp, _Up *> {
  typedef _Up type;
};

template <typename> struct __is_void_helper : public false_type {};

template <> struct __is_void_helper<void> : public true_type {};

/// is_void
template <typename _Tp>
struct is_void : public __is_void_helper<typename remove_cv<_Tp>::type>::type {
};

template <typename> struct is_lvalue_reference : public false_type {};

template <typename _Tp> struct is_lvalue_reference<_Tp &> : public true_type {};

/// is_rvalue_reference
template <typename> struct is_rvalue_reference : public false_type {};

template <typename _Tp>
struct is_rvalue_reference<_Tp &&> : public true_type {};

/// is_reference
template <typename _Tp>
struct is_reference
    : public __or_<is_lvalue_reference<_Tp>, is_rvalue_reference<_Tp>>::type {};

/// remove_pointer
template <typename _Tp>
struct remove_pointer
    : public __remove_pointer_helper<_Tp, typename remove_cv<_Tp>::type> {};

template <typename _Tp>
using remove_pointer_t = typename remove_pointer<_Tp>::type;

template <typename _Tp> struct remove_reference {
  typedef _Tp type;
};

template <typename _Tp> struct remove_reference<_Tp &> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_reference<_Tp &&> {
  typedef _Tp type;
};

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

/// is_array
template <typename> struct is_array : public false_type {};

template <typename _Tp, std::size_t _Size>
struct is_array<_Tp[_Size]> : public true_type {};

template <typename _Tp> struct is_array<_Tp[]> : public true_type {};

template <typename> struct __is_pointer_helper : public false_type {};

template <typename _Tp> struct __is_pointer_helper<_Tp *> : public true_type {};

/// is_pointer
template <typename _Tp>
struct is_pointer
    : public __is_pointer_helper<typename remove_cv<_Tp>::type>::type {};
template <typename> struct is_function;

template <typename> struct is_function : public false_type {};

#define _GLIBCXX_NOEXCEPT_PARM
#define _GLIBCXX_NOEXCEPT_QUAL

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile &
                   _GLIBCXX_NOEXCEPT_QUAL> : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile &&
                   _GLIBCXX_NOEXCEPT_QUAL> : public true_type {};

template <typename _Up, bool _IsArray = is_array<_Up>::value,
          bool _IsFunction = is_function<_Up>::value>
struct __decay_selector;

template <typename _Tp>
struct is_object
    : public __not_<
          __or_<is_function<_Tp>, is_reference<_Tp>, is_void<_Tp>>>::type {};

template <typename _Tp>
struct __is_referenceable
    : public __or_<is_object<_Tp>, is_reference<_Tp>>::type{};

template <typename _Res, typename... _Args>
struct __is_referenceable<_Res(_Args...)> : public true_type{};

template <typename _Res, typename... _Args>
struct __is_referenceable<_Res(_Args......)> : public true_type{};

// NB: DR 705.
template <typename _Up> struct __decay_selector<_Up, false, false> {
  typedef typename remove_cv<_Up>::type __type;
};

template <typename _Up> struct __decay_selector<_Up, true, false> {
  typedef typename remove_extent<_Up>::type *__type;
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

// nvrtc seems contain a built-in std::forward...
// template <typename _Tp>
// constexpr _Tp &&
// forward(typename std::remove_reference<_Tp>::type &__t) noexcept {
//   return static_cast<_Tp &&>(__t);
// }

// /**
//  *  @brief  Forward an rvalue.
//  *  @return The parameter cast to the specified type.
//  *
//  *  This function is used to implement "perfect forwarding".
//  */
// template <typename _Tp>
// constexpr _Tp &&
// forward(typename std::remove_reference<_Tp>::type &&__t) noexcept {
//   static_assert(!std::is_lvalue_reference<_Tp>::value,
//                 "template argument"
//                 " substituting _Tp is an lvalue reference type");
//   return static_cast<_Tp &&>(__t);
// }

template <typename, typename> struct is_same : public false_type {};

template <typename _Tp> struct is_same<_Tp, _Tp> : public true_type {};
template <typename _Tp, typename _Up>
inline constexpr bool is_same_v = is_same<_Tp, _Up>::value;

// Primary template.
/// Define a member typedef @c type only if a boolean constant is true.
template <bool, typename _Tp = void> struct enable_if {};

// Partial specialization for true.
template <typename _Tp> struct enable_if<true, _Tp> {
  typedef _Tp type;
};

template <typename... _Cond>
using _Require = typename enable_if<__and_<_Cond...>::value>::type;

template <bool _Cond, typename _Tp = void>
using enable_if_t = typename enable_if<_Cond, _Tp>::type;

template <bool _Cond, typename _Iftrue, typename _Iffalse>
using conditional_t = typename conditional<_Cond, _Iftrue, _Iffalse>::type;

template <typename _Tp, typename _Up = _Tp &&>
_Up TV_HOST_DEVICE_INLINE __declval(int);

template <typename _Tp> _Tp TV_HOST_DEVICE_INLINE __declval(long);

template <typename _Tp>
TV_HOST_DEVICE_INLINE auto declval() noexcept -> decltype(__declval<_Tp>(0));

template <typename _Tp> class reference_wrapper;

// Helper which adds a reference to a type when given a reference_wrapper
template <typename _Tp> struct __strip_reference_wrapper {
  typedef _Tp __type;
};

template <typename _Tp>
struct __strip_reference_wrapper<reference_wrapper<_Tp>> {
  typedef _Tp &__type;
};

template <typename _Tp> struct __decay_and_strip {
  typedef typename __strip_reference_wrapper<typename decay<_Tp>::type>::__type
      __type;
};

template <typename _Tp> struct tuple_size;

template <typename _Tp, typename _Up = typename remove_cv<_Tp>::type,
          typename = typename enable_if<is_same<_Tp, _Up>::value>::type,
          size_t = tuple_size<_Tp>::value>
using __enable_if_has_tuple_size = _Tp;

template <typename _Tp>
struct tuple_size<const __enable_if_has_tuple_size<_Tp>>
    : public tuple_size<_Tp> {};

template <typename _Tp>
struct tuple_size<volatile __enable_if_has_tuple_size<_Tp>>
    : public tuple_size<_Tp> {};

template <typename _Tp>
struct tuple_size<const volatile __enable_if_has_tuple_size<_Tp>>
    : public tuple_size<_Tp> {};

template <std::size_t __i, typename _Tp> struct tuple_element;

// Duplicate of C++14's tuple_element_t for internal use in C++11 mode
template <std::size_t __i, typename _Tp>
using __tuple_element_t = typename tuple_element<__i, _Tp>::type;

template <std::size_t __i, typename _Tp> struct tuple_element<__i, const _Tp> {
  typedef typename add_const<__tuple_element_t<__i, _Tp>>::type type;
};

template <std::size_t __i, typename _Tp>
struct tuple_element<__i, volatile _Tp> {
  typedef typename add_volatile<__tuple_element_t<__i, _Tp>>::type type;
};

template <std::size_t __i, typename _Tp>
struct tuple_element<__i, const volatile _Tp> {
  typedef typename add_cv<__tuple_element_t<__i, _Tp>>::type type;
};

template <std::size_t __i, typename _Tp>
using tuple_element_t = typename tuple_element<__i, _Tp>::type;

// https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/detail/libcxx/include/__type_traits/is_assignable.h
template <typename, typename _Tp> struct __select_2nd {
  typedef _Tp type;
};

template <class _Tp, class _Arg>
typename __select_2nd<decltype((declval<_Tp>() = declval<_Arg>())),
                      true_type>::type TV_HOST_DEVICE_INLINE
__is_assignable_test(int);

template <class, class>
TV_HOST_DEVICE_INLINE false_type __is_assignable_test(...);

template <class _Tp, class _Arg,
          bool = is_void<_Tp>::value || is_void<_Arg>::value>
struct __is_assignable_imp
    : public decltype((__is_assignable_test<_Tp, _Arg>(0))) {};

template <class _Tp, class _Arg>
struct __is_assignable_imp<_Tp, _Arg, true> : public false_type {};

template <class _Tp, class _Arg>
struct is_assignable : public __is_assignable_imp<_Tp, _Arg> {};

template <class _Tp, class _Arg>
constexpr bool is_assignable_v = is_assignable<_Tp, _Arg>::value;

template <typename... _Elements> class tuple;

template <typename> struct __is_tuple_like_impl : false_type {};

template <typename... _Tps>
struct __is_tuple_like_impl<tuple<_Tps...>> : true_type {};
template <typename _Tp>
using __remove_cvref_t =
    typename remove_cv<typename remove_reference<_Tp>::type>::type;

template <typename _Tp, bool = __is_referenceable<_Tp>::value>
struct __is_move_assignable_impl;

template <typename _Tp>
struct __is_move_assignable_impl<_Tp, false> : public false_type {};

template <typename _Tp>
struct __is_move_assignable_impl<_Tp, true>
    : public is_assignable<_Tp &, _Tp &&> {};

/// is_move_assignable
template <typename _Tp>
struct is_move_assignable : public __is_move_assignable_impl<_Tp> {};

/// is_constructible
template <typename _Tp, typename... _Args>
struct is_constructible
    : public __bool_constant<__is_constructible(_Tp, _Args...)> {};

template <typename _Tp, bool = __is_referenceable<_Tp>::value>
struct __is_move_constructible_impl;

template <typename _Tp>
struct __is_move_constructible_impl<_Tp, false> : public false_type {};

template <typename _Tp>
struct __is_move_constructible_impl<_Tp, true>
    : public is_constructible<_Tp, _Tp &&> {};

/// is_move_constructible
template <typename _Tp>
struct is_move_constructible : public __is_move_constructible_impl<_Tp> {};

template <typename _Tp, typename _Up>
struct __is_nt_assignable_impl
    : public integral_constant<bool,
                               noexcept(declval<_Tp>() = declval<_Up>())> {};

/// is_nothrow_assignable
template <typename _Tp, typename _Up>
struct is_nothrow_assignable
    : public __and_<is_assignable<_Tp, _Up>,
                    __is_nt_assignable_impl<_Tp, _Up>> {};

template <bool, bool, class _Tp, class... _Args>
struct __libcpp_is_nothrow_constructible;

template <class _Tp, class... _Args>
struct __libcpp_is_nothrow_constructible</*is constructible*/ true,
                                         /*is reference*/ false, _Tp, _Args...>
    : public integral_constant<bool, noexcept(_Tp(declval<_Args>()...))> {};

template <class _Tp>
TV_HOST_DEVICE_INLINE void __implicit_conversion_to(_Tp) noexcept {}

template <class _Tp, class _Arg>
struct __libcpp_is_nothrow_constructible</*is constructible*/ true,
                                         /*is reference*/ true, _Tp, _Arg>
    : public integral_constant<bool, noexcept(__implicit_conversion_to<_Tp>(
                                         declval<_Arg>()))> {};

template <class _Tp, bool _IsReference, class... _Args>
struct __libcpp_is_nothrow_constructible</*is constructible*/ false,
                                         _IsReference, _Tp, _Args...>
    : public false_type {};

template <class _Tp, class... _Args>
struct is_nothrow_constructible
    : __libcpp_is_nothrow_constructible<is_constructible<_Tp, _Args...>::value,
                                        is_reference<_Tp>::value, _Tp,
                                        _Args...> {};

template <class _Tp, size_t _Ns>
struct is_nothrow_constructible<_Tp[_Ns]>
    : __libcpp_is_nothrow_constructible<is_constructible<_Tp>::value,
                                        is_reference<_Tp>::value, _Tp> {};

template <typename _Tp, bool = __is_referenceable<_Tp>::value>
struct __is_nt_move_assignable_impl;

template <typename _Tp>
struct __is_nt_move_assignable_impl<_Tp, false> : public false_type {};

template <typename _Tp>
struct __is_nt_move_assignable_impl<_Tp, true>
    : public is_nothrow_assignable<_Tp &, _Tp &&> {};

/// is_nothrow_move_assignable
template <typename _Tp>
struct is_nothrow_move_assignable : public __is_nt_move_assignable_impl<_Tp> {};

template <typename _Tp, bool = __is_referenceable<_Tp>::value>
struct __is_nothrow_move_constructible_impl;

template <typename _Tp>
struct __is_nothrow_move_constructible_impl<_Tp, false> : public false_type {};

template <typename _Tp>
struct __is_nothrow_move_constructible_impl<_Tp, true>
    : public is_nothrow_constructible<_Tp, _Tp &&> {};

/// is_nothrow_move_constructible
template <typename _Tp>
struct is_nothrow_move_constructible
    : public __is_nothrow_move_constructible_impl<_Tp> {};

// Internal type trait that allows us to sfinae-protect tuple_cat.
template <typename _Tp>
struct __is_tuple_like
    : public __is_tuple_like_impl<__remove_cvref_t<_Tp>>::type {};

template <typename _Tp>
TV_HOST_DEVICE_INLINE typename enable_if<
    __and_<__not_<__is_tuple_like<_Tp>>, is_move_constructible<_Tp>,
           is_move_assignable<_Tp>>::value>::type
swap(_Tp &__a,
     _Tp &__b) noexcept(__and_<is_nothrow_move_constructible<_Tp>,
                               is_nothrow_move_assignable<_Tp>>::value) {

  _Tp __tmp = std::move(__a);
  __a = std::move(__b);
  __b = std::move(__tmp);
}

} // namespace std
#endif