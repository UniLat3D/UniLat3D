#pragma once
#ifdef __CUDACC_RTC__
#include "type_traits.h"
#include <tensorview/core/defs.h>
#include <tensorview/core/mp_helper.h>

namespace std {
namespace detail {

template <size_t N, typename T> class head {
public:
  template <typename Targ>
  TV_HOST_DEVICE_INLINE constexpr explicit head(Targ &&num)
      : _value{std::forward<Targ>(num)} {}
  TV_HOST_DEVICE_INLINE constexpr explicit head() : _value() {}

  T _value;
};

struct __nonesuch {
  __nonesuch() = delete;
  ~__nonesuch() = delete;
  __nonesuch(__nonesuch const &) = delete;
  void operator=(__nonesuch const &) = delete;
};

template <typename Seq, typename... Ts> class tuple_impl {};
struct __nonesuch_no_braces : __nonesuch {
  explicit __nonesuch_no_braces(const __nonesuch &) = delete;
};

template <size_t... Ns, typename... Ts>
class tuple_impl<tv::mp_list_c<size_t, Ns...>, Ts...> : public head<Ns, Ts>... {
public:
  template <typename... Targs,
            typename = std::enable_if_t<sizeof...(Targs) == sizeof...(Ns)>>
  TV_HOST_DEVICE_INLINE constexpr tuple_impl(Targs &&...nums)
      : head<Ns, Ts>{std::forward<Targs>(nums)}... {}
  constexpr tuple_impl(const tuple_impl &) = default;

  constexpr tuple_impl(tuple_impl &&) = default;

  TV_HOST_DEVICE_INLINE constexpr tuple_impl() : head<Ns, Ts>{}... {}

  template <typename... _UElements> struct _M_assigner {
    const tuple<tv::mp_list_c<size_t, Ns...>, _UElements...> &__in;
    tuple_impl &__out;
    template <class I> TV_HOST_DEVICE_INLINE void operator()(I i) {
      using base = head<I::value, typename tv::mp_nth_t<I::value, Ts...>>;
      using base2 =
          head<I::value, typename tv::mp_nth_t<I::value, _UElements...>>;
      static_cast<base &>(__out)._value =
          static_cast<const base2 &>(__in)._value;
    }
  };
  template <typename... _UElements> struct _M_assigner_r {
    const tuple_impl<tv::mp_list_c<size_t, Ns...>, _UElements...> &&__in;
    tuple_impl &__out;
    template <class I> TV_HOST_DEVICE_INLINE void operator()(I i) {
      using base = head<I::value, typename tv::mp_nth_t<I::value, Ts...>>;
      using base2 =
          head<I::value, typename tv::mp_nth_t<I::value, _UElements...>>;
      static_cast<base &>(__out)._value =
          std::move(static_cast<const base2 &&>(__in)._value);
    }
  };

  template <typename... _UElements>
  TV_HOST_DEVICE_INLINE void _M_assign(
      const tuple_impl<tv::mp_list_c<size_t, Ns...>, _UElements...> &__in) {
    tv::mp_for_each_cuda<tv::mp_list_c<size_t, Ns...>>(__assigner);
  }

  template <typename... _UElements>
  void TV_HOST_DEVICE_INLINE _M_assign(
      const tuple_impl<tv::mp_list_c<size_t, Ns...>, _UElements...> &&__in) {
    _M_assigner_r<_UElements...> __assigner{std::move(__in), *this};
    tv::mp_for_each_cuda<tv::mp_list_c<size_t, Ns...>>(__assigner);
  }
};

} // namespace detail

template <typename... Ts>
class tuple : public detail::tuple_impl<
                  tv::mp_make_integer_sequence<size_t, sizeof...(Ts)>, Ts...> {
  using __Base =
      detail::tuple_impl<tv::mp_make_integer_sequence<size_t, sizeof...(Ts)>,
                         Ts...>;
  template <class... _UElements>
  TV_HOST_DEVICE_INLINE static constexpr enable_if_t<
      sizeof...(_UElements) == sizeof...(Ts), bool>
  __assignable() {
    return __and_<is_assignable<Ts &, _UElements>...>::value;
  }

  template <class... _UElements>
  TV_HOST_DEVICE_INLINE static constexpr bool __nothrow_assignable() {
    return __and_<is_nothrow_assignable<Ts &, _UElements>...>::value;
  }

public:
  // FIXME if only one element in tuple, structure binding will raise error
  template <typename... Targs,
            typename = std::enable_if_t<sizeof...(Targs) == sizeof...(Ts)>>
  TV_HOST_DEVICE_INLINE constexpr tuple(Targs &&...nums)
      : __Base{std::forward<Targs>(nums)...} {}
  constexpr tuple(const tuple &) = default;

  constexpr tuple(tuple &&) = default;

  TV_HOST_DEVICE_INLINE constexpr tuple() : __Base{} {}

  TV_HOST_DEVICE_INLINE tuple &
  operator=(typename conditional<__assignable<const Ts &...>(), const tuple &,
                                 const detail::__nonesuch_no_braces &>::type
                __in) noexcept(__nothrow_assignable<const Ts &...>()) {
    this->_M_assign(__in);
    return *this;
  }

  TV_HOST_DEVICE_INLINE tuple &
  operator=(tuple &&__in) noexcept(__nothrow_assignable<Ts...>()) {
    this->_M_assign(std::move(__in));
    return *this;
  }

  template <typename... _UElements>
  enable_if_t<__assignable<const _UElements &...>(), tuple &>
      TV_HOST_DEVICE_INLINE
      operator=(const tuple<_UElements...> &__in) noexcept(
          __nothrow_assignable<const _UElements &...>()) {
    this->_M_assign(__in);
    return *this;
  }

  template <typename... _UElements>
  enable_if_t<__assignable<_UElements...>(), tuple &>
      TV_HOST_DEVICE_INLINE operator=(tuple<_UElements...> &&__in) noexcept(
          __nothrow_assignable<_UElements...>()) {
    this->_M_assign(std::move(__in));
    return *this;
  }
};

// template <typename... Ts>
// using tuple =
//     detail::tuple_impl<tv::mp_make_integer_sequence<size_t, sizeof...(Ts)>,
//                        Ts...>;

template <size_t __i, class... Ts> struct tuple_element<__i, tuple<Ts...>> {
  using type = typename tv::mp_nth_t<__i, Ts...>;
};

template <typename... _Elements>
struct tuple_size<tuple<_Elements...>>
    : public integral_constant<size_t, sizeof...(_Elements)> {};

template <typename _Tp> constexpr size_t tuple_size_v = tuple_size<_Tp>::value;

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr tuple_element_t<N, tuple<Ts...>> &
get(tuple<Ts...> &t) {
  using base = detail::head<N, tuple_element_t<N, tuple<Ts...>>>;
  return static_cast<base &>(t)._value;
}
template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr const tuple_element_t<N, tuple<Ts...>> &
get(const tuple<Ts...> &t) {
  using base = detail::head<N, tuple_element_t<N, tuple<Ts...>>>;
  return static_cast<const base &>(t)._value;
}

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr tuple_element_t<N, tuple<Ts...>> &&
get(tuple<Ts...> &&t) {
  typedef tuple_element_t<N, tuple<Ts...>> __element_type;
  return std::forward<__element_type &&>(get<N>(t));
}

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr const tuple_element_t<N, tuple<Ts...>> &&
get(const tuple<Ts...> &&t) {
  typedef tuple_element_t<N, tuple<Ts...>> __element_type;
  return std::forward<const __element_type &&>(get<N>(t));
}

template <typename... _Elements>
TV_HOST_DEVICE_INLINE constexpr tuple<
    typename __decay_and_strip<_Elements>::__type...>
make_tuple(_Elements &&...__args) {
  typedef tuple<typename __decay_and_strip<_Elements>::__type...> __result_type;
  return __result_type(std::forward<_Elements>(__args)...);
}

template <typename... _Elements>
TV_HOST_DEVICE_INLINE constexpr tuple<_Elements &...>
tie(_Elements &...__args) noexcept {
  return tuple<_Elements &...>(__args...);
}

} // namespace std
#endif
