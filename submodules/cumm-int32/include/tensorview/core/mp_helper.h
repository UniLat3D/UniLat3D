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

//  Copyright 2015-2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

// disable weird warning when use alignas(0)

#ifndef MP_HELPER_H_
#define MP_HELPER_H_
#include "defs.h"

#ifdef TV_PARALLEL_RTC
#ifdef __APPLE__
#include "nvrtc_std.h"
#else 
#include "nvrtc_std.h"
#endif
#else
#include "cc17.h"
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#endif

namespace tv {

namespace detail {

template <std::size_t I, typename T, T N, T... Ns> struct mp_nth_c_impl {
  constexpr static T TV_METAL_CONSTANT value = mp_nth_c_impl<I - 1, T, Ns...>::value;
};

template <typename T, T N, T... Ns> struct mp_nth_c_impl<0, T, N, Ns...> {
  constexpr static T TV_METAL_CONSTANT value = N;
};

template <std::size_t I, typename T, class... Ts> struct mp_nth_impl {
  using type = typename mp_nth_impl<I - 1, Ts...>::type;
};

template <typename T, class... Ts> struct mp_nth_impl<0, T, Ts...> {
  using type = T;
};

template <std::size_t I, typename T> struct mp_at_impl;

template <std::size_t I, template <class...> class T, class... Args>
struct mp_at_impl<I, T<Args...>> {
  using type = typename mp_nth_impl<I, Args...>::type;
};

} // namespace detail

template <std::size_t I, typename T>
using mp_at = typename detail::mp_at_impl<I, T>::type;

template <std::size_t I, typename T>
constexpr auto TV_METAL_CONSTANT mp_at_c = mp_at<I, T>::value;

template <std::size_t I, typename T, T... Ns>
constexpr T TV_METAL_CONSTANT mp_nth_c = detail::mp_nth_c_impl<I, T, Ns...>::value;

template <std::size_t I, class... Ts>
using mp_nth_t = typename detail::mp_nth_impl<I, Ts...>::type;

template <class... T> struct mp_list {
  template <std::size_t I> using at = mp_nth_t<I, T...>;

  template <std::size_t I>
  static constexpr typename mp_nth_t<I, T...>::value_type TV_METAL_CONSTANT at_c =
      mp_nth_t<I, T...>::value;

  static TV_HOST_DEVICE_INLINE constexpr std::size_t size() {
    return sizeof...(T);
  }
};

template <> struct mp_list<> {
  static TV_HOST_DEVICE_INLINE constexpr std::size_t size() { return 0; }
};

template <class T, T... I>
using mp_list_c = mp_list<std::integral_constant<T, I>...>;

template <int... I>
using mp_list_int = mp_list<std::integral_constant<int, I>...>;

namespace detail {
#ifndef TV_PARALLEL_RTC
template <class... Ts, class F>
constexpr F mp_for_each_impl(mp_list<Ts...>, F &&f) {
  using A = int[sizeof...(Ts)];
  return (void)A{((void)f(Ts()), 0)...}, std::forward<F>(f);

  // return (void)(std::initializer_list<int>{(f(Ts()), 0)...}),
  //        std::forward<F>(f);
}

template <class F> constexpr F mp_for_each_impl(mp_list<>, F &&f) {
  return std::forward<F>(f);
}
#endif

#ifndef TV_METAL_RTC
template <class... Ts, class F>
TV_HOST_DEVICE_INLINE constexpr F mp_for_each_impl_cuda(mp_list<Ts...>, F &&f) {
  using A = int[sizeof...(Ts)];
  return (void)A{((void)std::forward<F>(f)(Ts()), 0)...}, std::forward<F>(f);

  // return (void)(std::initializer_list<int>{(f(Ts()), 0)...}),
  //        std::forward<F>(f);
}

template <class F> TV_HOST_DEVICE_INLINE constexpr F mp_for_each_impl_cuda(mp_list<>, F &&f) {
  return std::forward<F>(f);
}
#endif
} // namespace detail

template <class... T>
using mp_length = std::integral_constant<std::size_t, sizeof...(T)>;

namespace detail {

template <class A, template <class...> class B> struct mp_rename_impl {
  // An error "no type named 'type'" here means that the first argument to
  // mp_rename is not a list
};

template <template <class...> class A, class... T, template <class...> class B>
struct mp_rename_impl<A<T...>, B> {
  using type = B<T...>;
};

template <class A, template <class...> class B> struct mp_rename_v_impl {
  // An error "no type named 'type'" here means that the first argument to
  // mp_rename is not a list
};

template <template <class...> class A, class... T, template <class...> class B>
struct mp_rename_v_impl<A<T...>, B> {
  // unlike mp_rename_t, B is a templated type. we must use xx::value
  // because B can't be a templated non-type template parameter.
  static constexpr auto TV_METAL_CONSTANT value = B<T...>::value;
};

template <class L, class T> struct mp_append_impl;

template <class... Ts, class T, template <class...> class L>
struct mp_append_impl<L<Ts...>, T> {
  using type = mp_list<Ts..., T>;
};

template <class L, class T> struct mp_insert_front_impl;

template <class... Ts, class T, template <class...> class L>
struct mp_insert_front_impl<L<Ts...>, T> {
  using type = mp_list<T, Ts...>;
};

template <template <class> class F, class L> struct mp_transform_impl;

template <template <class> class F, template <class...> class L, class... T>
struct mp_transform_impl<F, L<T...>> {
  using type = L<F<T>...>;
};

template <template <class, class> class Op, class Init, class T, class... Ts>
struct mp_reduce_impl1 {
  using type =
      typename Op<T, typename mp_reduce_impl1<Op, Init, Ts...>::type>::type;
};

template <template <class, class> class Op, class Init, class T>
struct mp_reduce_impl1<Op, Init, T> {
  using type = typename Op<T, Init>::type;
};

template <template <class, class> class Op, class L, class Init>
struct mp_reduce_impl;

template <template <class, class> class Op, class Init,
          template <class...> class L, class... T>
struct mp_reduce_impl<Op, L<T...>, Init> {
  using type = typename mp_reduce_impl1<Op, Init, T...>::type;
};

} // namespace detail

template <template <class, class> class Op, class L, class Init>
using mp_reduce = typename detail::mp_reduce_impl<Op, L, Init>::type;

template <template <class> class F, class L>
using mp_transform = typename detail::mp_transform_impl<F, L>::type;

template <class L, class T>
using mp_append = typename detail::mp_append_impl<L, T>::type;

// mp_bool
template <bool B> using mp_bool = std::integral_constant<bool, B>;

using mp_true = mp_bool<true>;
using mp_false = mp_bool<false>;

// mp_if in boost/mp11

namespace detail {

template <bool C, class T, class... E> struct mp_if_c_impl {};

template <class T, class... E> struct mp_if_c_impl<true, T, E...> {
  using type = T;
};

template <class T, class E> struct mp_if_c_impl<false, T, E> {
  using type = E;
};

} // namespace detail

template <bool C, class T, class... E>
using mp_if_c = typename detail::mp_if_c_impl<C, T, E...>::type;
template <class C, class T, class... E>
using mp_if =
    typename detail::mp_if_c_impl<static_cast<bool>(C::value), T, E...>::type;

// mp_valid in boost/mp11

namespace detail {
#ifndef TV_METAL_RTC
template <template <class...> class F, class... T> struct mp_valid_impl {
  template <template <class...> class G, class = G<T...>>
  TV_HOST_DEVICE_INLINE static mp_true check(int);
  template <template <class...> class> TV_HOST_DEVICE_INLINE static mp_false check(...);

  using type = decltype(check<F>(0));
};
#endif
} // namespace detail
#ifndef TV_METAL_RTC

template <template <class...> class F, class... T>
using mp_valid = typename detail::mp_valid_impl<F, T...>::type;
#endif
// mp_defer in boost/mp11
#ifndef TV_METAL_RTC

// mp_defer
namespace detail {

template <template <class...> class F, class... T> struct mp_defer_impl {
  using type = F<T...>;
};

struct mp_no_type {};

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 9000 && CUDA_VERSION < 10000)

template <template <class...> class F, class... T>
struct mp_defer_cuda_workaround {
  using type = mp_if<mp_valid<F, T...>, detail::mp_defer_impl<F, T...>,
                     detail::mp_no_type>;
};

#endif

} // namespace detail

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 9000 && CUDA_VERSION < 10000)

template <template <class...> class F, class... T>
using mp_defer = typename detail::mp_defer_cuda_workaround<F, T...>::type;

#else

template <template <class...> class F, class... T>
using mp_defer = mp_if<mp_valid<F, T...>, detail::mp_defer_impl<F, T...>,
                       detail::mp_no_type>;

#endif

template <template <class...> class F> struct mp_quote {
  // the indirection through mp_defer works around the language inability
  // to expand T... into a fixed parameter list of an alias template

  template <class... T> using fn = typename mp_defer<F, T...>::type;
};

namespace detail {
// from boost/mp11
template <class... L> struct mp_concat_impl;
template <class L1 = mp_list<>, class L2 = mp_list<>, class L3 = mp_list<>,
          class L4 = mp_list<>, class L5 = mp_list<>, class L6 = mp_list<>,
          class L7 = mp_list<>, class L8 = mp_list<>, class L9 = mp_list<>,
          class L10 = mp_list<>, class L11 = mp_list<>>
struct append_11_impl {};

template <
    template <class...> class L1, class... T1, template <class...> class L2,
    class... T2, template <class...> class L3, class... T3,
    template <class...> class L4, class... T4, template <class...> class L5,
    class... T5, template <class...> class L6, class... T6,
    template <class...> class L7, class... T7, template <class...> class L8,
    class... T8, template <class...> class L9, class... T9,
    template <class...> class L10, class... T10, template <class...> class L11,
    class... T11>

struct append_11_impl<L1<T1...>, L2<T2...>, L3<T3...>, L4<T4...>, L5<T5...>,
                      L6<T6...>, L7<T7...>, L8<T8...>, L9<T9...>, L10<T10...>,
                      L11<T11...>> {
  using type = L1<T1..., T2..., T3..., T4..., T5..., T6..., T7..., T8..., T9...,
                  T10..., T11...>;
};

template <

    class L00 = mp_list<>, class L01 = mp_list<>, class L02 = mp_list<>,
    class L03 = mp_list<>, class L04 = mp_list<>, class L05 = mp_list<>,
    class L06 = mp_list<>, class L07 = mp_list<>, class L08 = mp_list<>,
    class L09 = mp_list<>, class L0A = mp_list<>, class L10 = mp_list<>,
    class L11 = mp_list<>, class L12 = mp_list<>, class L13 = mp_list<>,
    class L14 = mp_list<>, class L15 = mp_list<>, class L16 = mp_list<>,
    class L17 = mp_list<>, class L18 = mp_list<>, class L19 = mp_list<>,
    class L20 = mp_list<>, class L21 = mp_list<>, class L22 = mp_list<>,
    class L23 = mp_list<>, class L24 = mp_list<>, class L25 = mp_list<>,
    class L26 = mp_list<>, class L27 = mp_list<>, class L28 = mp_list<>,
    class L29 = mp_list<>, class L30 = mp_list<>, class L31 = mp_list<>,
    class L32 = mp_list<>, class L33 = mp_list<>, class L34 = mp_list<>,
    class L35 = mp_list<>, class L36 = mp_list<>, class L37 = mp_list<>,
    class L38 = mp_list<>, class L39 = mp_list<>, class L40 = mp_list<>,
    class L41 = mp_list<>, class L42 = mp_list<>, class L43 = mp_list<>,
    class L44 = mp_list<>, class L45 = mp_list<>, class L46 = mp_list<>,
    class L47 = mp_list<>, class L48 = mp_list<>, class L49 = mp_list<>,
    class L50 = mp_list<>, class L51 = mp_list<>, class L52 = mp_list<>,
    class L53 = mp_list<>, class L54 = mp_list<>, class L55 = mp_list<>,
    class L56 = mp_list<>, class L57 = mp_list<>, class L58 = mp_list<>,
    class L59 = mp_list<>, class L60 = mp_list<>, class L61 = mp_list<>,
    class L62 = mp_list<>, class L63 = mp_list<>, class L64 = mp_list<>,
    class L65 = mp_list<>, class L66 = mp_list<>, class L67 = mp_list<>,
    class L68 = mp_list<>, class L69 = mp_list<>, class L70 = mp_list<>,
    class L71 = mp_list<>, class L72 = mp_list<>, class L73 = mp_list<>,
    class L74 = mp_list<>, class L75 = mp_list<>, class L76 = mp_list<>,
    class L77 = mp_list<>, class L78 = mp_list<>, class L79 = mp_list<>,
    class L80 = mp_list<>, class L81 = mp_list<>, class L82 = mp_list<>,
    class L83 = mp_list<>, class L84 = mp_list<>, class L85 = mp_list<>,
    class L86 = mp_list<>, class L87 = mp_list<>, class L88 = mp_list<>,
    class L89 = mp_list<>, class L90 = mp_list<>, class L91 = mp_list<>,
    class L92 = mp_list<>, class L93 = mp_list<>, class L94 = mp_list<>,
    class L95 = mp_list<>, class L96 = mp_list<>, class L97 = mp_list<>,
    class L98 = mp_list<>, class L99 = mp_list<>, class LA0 = mp_list<>,
    class LA1 = mp_list<>, class LA2 = mp_list<>, class LA3 = mp_list<>,
    class LA4 = mp_list<>, class LA5 = mp_list<>, class LA6 = mp_list<>,
    class LA7 = mp_list<>, class LA8 = mp_list<>, class LA9 = mp_list<>

    >
struct append_111_impl {
  using type = typename append_11_impl<

      typename append_11_impl<L00, L01, L02, L03, L04, L05, L06, L07, L08, L09,
                              L0A>::type,
      typename append_11_impl<mp_list<>, L10, L11, L12, L13, L14, L15, L16, L17,
                              L18, L19>::type,
      typename append_11_impl<mp_list<>, L20, L21, L22, L23, L24, L25, L26, L27,
                              L28, L29>::type,
      typename append_11_impl<mp_list<>, L30, L31, L32, L33, L34, L35, L36, L37,
                              L38, L39>::type,
      typename append_11_impl<mp_list<>, L40, L41, L42, L43, L44, L45, L46, L47,
                              L48, L49>::type,
      typename append_11_impl<mp_list<>, L50, L51, L52, L53, L54, L55, L56, L57,
                              L58, L59>::type,
      typename append_11_impl<mp_list<>, L60, L61, L62, L63, L64, L65, L66, L67,
                              L68, L69>::type,
      typename append_11_impl<mp_list<>, L70, L71, L72, L73, L74, L75, L76, L77,
                              L78, L79>::type,
      typename append_11_impl<mp_list<>, L80, L81, L82, L83, L84, L85, L86, L87,
                              L88, L89>::type,
      typename append_11_impl<mp_list<>, L90, L91, L92, L93, L94, L95, L96, L97,
                              L98, L99>::type,
      typename append_11_impl<mp_list<>, LA0, LA1, LA2, LA3, LA4, LA5, LA6, LA7,
                              LA8, LA9>::type

      >::type;
};

template <

    class L00, class L01, class L02, class L03, class L04, class L05, class L06,
    class L07, class L08, class L09, class L0A, class L10, class L11, class L12,
    class L13, class L14, class L15, class L16, class L17, class L18, class L19,
    class L20, class L21, class L22, class L23, class L24, class L25, class L26,
    class L27, class L28, class L29, class L30, class L31, class L32, class L33,
    class L34, class L35, class L36, class L37, class L38, class L39, class L40,
    class L41, class L42, class L43, class L44, class L45, class L46, class L47,
    class L48, class L49, class L50, class L51, class L52, class L53, class L54,
    class L55, class L56, class L57, class L58, class L59, class L60, class L61,
    class L62, class L63, class L64, class L65, class L66, class L67, class L68,
    class L69, class L70, class L71, class L72, class L73, class L74, class L75,
    class L76, class L77, class L78, class L79, class L80, class L81, class L82,
    class L83, class L84, class L85, class L86, class L87, class L88, class L89,
    class L90, class L91, class L92, class L93, class L94, class L95, class L96,
    class L97, class L98, class L99, class LA0, class LA1, class LA2, class LA3,
    class LA4, class LA5, class LA6, class LA7, class LA8, class LA9,
    class... Lr

    >
struct append_inf_impl {
  using prefix = typename append_111_impl<

      L00, L01, L02, L03, L04, L05, L06, L07, L08, L09, L0A, L10, L11, L12, L13,
      L14, L15, L16, L17, L18, L19, L20, L21, L22, L23, L24, L25, L26, L27, L28,
      L29, L30, L31, L32, L33, L34, L35, L36, L37, L38, L39, L40, L41, L42, L43,
      L44, L45, L46, L47, L48, L49, L50, L51, L52, L53, L54, L55, L56, L57, L58,
      L59, L60, L61, L62, L63, L64, L65, L66, L67, L68, L69, L70, L71, L72, L73,
      L74, L75, L76, L77, L78, L79, L80, L81, L82, L83, L84, L85, L86, L87, L88,
      L89, L90, L91, L92, L93, L94, L95, L96, L97, L98, L99, LA0, LA1, LA2, LA3,
      LA4, LA5, LA6, LA7, LA8, LA9

      >::type;

  using type = typename mp_concat_impl<prefix, Lr...>::type;
};
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 9000 && CUDA_VERSION < 10000)

template <class... L> struct mp_concat_impl_cuda_workaround {
  using type = mp_if_c<(sizeof...(L) > 111), mp_quote<append_inf_impl>,
                       mp_if_c<(sizeof...(L) > 11), mp_quote<append_111_impl>,
                               mp_quote<append_11_impl>>>;
};

template <class... L>
struct mp_concat_impl
    : mp_concat_impl_cuda_workaround<L...>::type::template fn<L...> {};

#else

template <class... L>
struct mp_concat_impl
    : mp_if_c<(sizeof...(L) > 111), mp_quote<append_inf_impl>,
              mp_if_c<(sizeof...(L) > 11), mp_quote<append_111_impl>,
                      mp_quote<append_11_impl>>>::template fn<L...> {};

#endif

} // namespace detail

template <class... L>
using mp_concat = typename detail::mp_concat_impl<L...>::type;
#endif

namespace detail {

template <class L1, class L2> struct mp_assign_impl;

template <template <class...> class L1, class... T,
          template <class...> class L2, class... U>
struct mp_assign_impl<L1<T...>, L2<U...>> {
  using type = L1<U...>;
};

} // namespace detail

template <class L1, class L2>
using mp_assign = typename detail::mp_assign_impl<L1, L2>::type;

// mp_clear<L>
template <class L> using mp_clear = mp_assign<L, mp_list<>>;
#ifndef TV_METAL_RTC

namespace detail {

template <class L, std::size_t N> struct mp_repeat_c_impl {
  using _l1 = typename mp_repeat_c_impl<L, N / 2>::type;
  using _l2 = typename mp_repeat_c_impl<L, N % 2>::type;

  using type = mp_concat<_l1, _l1, _l2>;
};

template <class L> struct mp_repeat_c_impl<L, 0> { using type = mp_clear<L>; };

template <class L> struct mp_repeat_c_impl<L, 1> { using type = L; };

} // namespace detail

template <class L, std::size_t N>
using mp_repeat_c = typename detail::mp_repeat_c_impl<L, N>::type;
template <class L, class N>
using mp_repeat =
    typename detail::mp_repeat_c_impl<L, std::size_t{N::value}>::type;
#endif
template <class L, class T>
using mp_insert_front = typename detail::mp_insert_front_impl<L, T>::type;

template <class A, template <class...> class B>
using mp_rename = typename detail::mp_rename_impl<A, B>::type;

template <class A, template <class...> class B>
constexpr auto TV_METAL_CONSTANT mp_rename_v = detail::mp_rename_v_impl<A, B>::value;

template <class L> using mp_size = mp_rename<L, mp_length>;

#ifndef TV_PARALLEL_RTC
template <class L, class F> constexpr F mp_for_each(F &&f) {
  return detail::mp_for_each_impl(mp_rename<L, mp_list>(), std::forward<F>(f));
}
#endif
#ifndef TV_METAL_RTC

template <class L, class F> TV_HOST_DEVICE_INLINE constexpr F mp_for_each_cuda(F &&f) {
  return detail::mp_for_each_impl_cuda(mp_rename<L, mp_list>(), std::forward<F>(f));
}
#endif

template <unsigned N, unsigned... Ns> struct mp_prod_uint {
  static constexpr unsigned TV_METAL_CONSTANT value = N * mp_prod_uint<Ns...>::value;
};

template <unsigned N> struct mp_prod_uint<N> {
  static constexpr unsigned TV_METAL_CONSTANT value = N;
};
template <int N, int... Ns> struct mp_prod_int {
  static constexpr int TV_METAL_CONSTANT value = N * mp_prod_int<Ns...>::value;
};

template <int N> struct mp_prod_int<N> {
  static constexpr int TV_METAL_CONSTANT value = N;
};

namespace detail {
template <typename TA, typename TB> struct mp_max_op_impl {
  using type =
      std::integral_constant<typename TA::value_type,
                             (TA::value > TB::value ? TA::value : TB::value)>;
};
template <typename TA, typename TB> struct mp_min_op_impl {
  using type =
      std::integral_constant<typename TA::value_type,
                             (TA::value < TB::value ? TA::value : TB::value)>;
};
template <typename TA, typename TB> struct mp_sum_op_impl {
  using type =
      std::integral_constant<typename TA::value_type, TA::value + TB::value>;
};
template <typename TA, typename TB> struct mp_prod_op_impl {
  using type =
      std::integral_constant<typename TA::value_type, TA::value * TB::value>;
};
template <typename TA, typename TB> struct mp_or_op_impl {
  using type =
      std::integral_constant<typename TA::value_type, TA::value || TB::value>;
};

template <typename TA, typename TB> struct mp_and_op_impl {
  using type =
      std::integral_constant<typename TA::value_type, TA::value && TB::value>;
};

} // namespace detail

template <class L, class Init>
using mp_reduce_max = mp_reduce<detail::mp_max_op_impl, L, Init>;

template <class L, class Init>
using mp_reduce_min = mp_reduce<detail::mp_min_op_impl, L, Init>;

template <class L, class Init>
using mp_reduce_sum = mp_reduce<detail::mp_sum_op_impl, L, Init>;

template <class L, class Init>
using mp_reduce_prod = mp_reduce<detail::mp_prod_op_impl, L, Init>;

template <class L, class Init>
using mp_reduce_or = mp_reduce<detail::mp_or_op_impl, L, Init>;

template <class L, class Init>
using mp_reduce_and = mp_reduce<detail::mp_and_op_impl, L, Init>;

template <class T, T... Ns>
constexpr T TV_METAL_CONSTANT mp_reduce_sum_v =
    mp_reduce_sum<mp_list_c<T, Ns...>, std::integral_constant<T, T(0)>>::value;

namespace detail {

// template <typename T, typename L> struct mp_append_impl;
// template <typename T, class... Ts> struct mp_append_impl<T, mp_list<Ts...>> {
//   using type = mp_list<Ts..., T>;
// };

template <typename T, int N> struct mp_make_integer_sequence_impl {
  using type = typename mp_append_impl<
      typename mp_make_integer_sequence_impl<T, N - 1>::type,
      std::integral_constant<T, N - 1>>::type;
};

template <typename T> struct mp_make_integer_sequence_impl<T, 1> {
  using type = mp_list_c<T, 0>;
};

template <typename T> struct mp_make_integer_sequence_impl<T, 0> {
  using type = mp_list_c<T>;
};

template <typename T> struct mp_integer_sequence_impl;

template <class T, T... Ints>
struct mp_integer_sequence_impl<mp_list_c<T, Ints...>> {
  using type = mp_list<std::integral_constant<T, Ints>...>;
};
} // namespace detail

template <typename T, int N>
using mp_make_integer_sequence =
    typename detail::mp_make_integer_sequence_impl<T, N>::type;

template <class T, int N>
using mp_make_list_c_sequence = typename detail::mp_integer_sequence_impl<
    mp_make_integer_sequence<T, N>>::type;

namespace detail {}

namespace detail {

template <class T, T Start, T... Is>
constexpr auto TV_HOST_DEVICE_INLINE
mp_make_list_c_range_impl(mp_list_c<T, Is...> const )
    -> decltype(mp_list_c<T, (Is + Start)...>{});

template <class T, T... Is>
constexpr auto TV_HOST_DEVICE_INLINE
mp_make_list_c_sequence_reverse_impl(mp_list_c<T, Is...> const )
    -> decltype(mp_list_c<T, sizeof...(Is) - 1U - Is...>{});
} // namespace detail

template <int Start, int End>
using mp_list_int_range =
    decltype(detail::mp_make_list_c_range_impl<int, Start>(
        mp_make_list_c_sequence<int, (End - Start > 0 ? End - Start : 0)>{}));

template <class T, std::size_t N>
using mp_make_list_c_sequence_reverse =
    decltype(detail::mp_make_list_c_sequence_reverse_impl(
        mp_make_list_c_sequence<int, N>{}));

// #ifndef __CUDACC_RTC__

namespace detail {

template <typename Ret, typename... Args>
TV_HOST_DEVICE_INLINE std::integral_constant<std::size_t, sizeof...(Args)>
    func_argument_size_helper(Ret (*)(Args...));

template <typename Ret, typename F, typename... Args>
TV_HOST_DEVICE_INLINE std::integral_constant<std::size_t, sizeof...(Args)>
    func_argument_size_helper(Ret (F::*)(Args...));

template <typename Ret, typename F, typename... Args>
TV_HOST_DEVICE_INLINE std::integral_constant<std::size_t, sizeof...(Args)>
func_argument_size_helper(Ret (F::*)(Args...) const);

template <typename F>
TV_HOST_DEVICE_INLINE decltype(func_argument_size_helper(&F::operator()))
    func_argument_size_helper(F);

template <typename Ret, typename Arg, typename... Rest>
TV_HOST_DEVICE_INLINE Arg first_argument_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
TV_HOST_DEVICE_INLINE Arg first_argument_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
TV_HOST_DEVICE_INLINE Arg first_argument_helper(Ret (F::*)(Arg, Rest...) const);

template <typename Ret, typename Arg, typename... Rest>
TV_HOST_DEVICE_INLINE Ret result_type_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
TV_HOST_DEVICE_INLINE Ret result_type_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
TV_HOST_DEVICE_INLINE Ret result_type_helper(Ret (F::*)(Arg, Rest...) const);

template <typename F>
TV_HOST_DEVICE_INLINE decltype(first_argument_helper(&F::operator()))
    first_argument_helper(F);

template <typename F>
TV_HOST_DEVICE_INLINE decltype(result_type_helper(&F::operator()))
    result_type_helper(F);

} // namespace detail

template <typename T>
using first_argument_t =
    decltype(detail::first_argument_helper(std::declval<T>()));
template <typename T>
using return_type_t = decltype(detail::result_type_helper(std::declval<T>()));

template <typename T>
constexpr TV_METAL_CONSTANT std::size_t argument_size_v =
    decltype(detail::func_argument_size_helper(std::declval<T>()))::value;
// #endif

} // namespace tv
// using TP1 = tv::mp_concat<tv::mp_list<int, float>, tv::mp_list<double>>;
// using TP2 = tv::mp_repeat_c<tv::mp_list<int>, 5>;
// using TP = tv::mp_rename<std::tuple, tv::mp_repeat_c<int, 5>>;
// using TP3 = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<int>, 5>, thrust::tuple>;
#endif