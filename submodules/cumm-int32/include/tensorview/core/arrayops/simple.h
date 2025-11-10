#pragma once

#include "mathbase.h"
#include <tensorview/core/array.h>

namespace tv {
namespace arrayops {


namespace arrayops_detail {
template <class T, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<const TV_METAL_DEVICE T*, sizeof...(Inds)>
create_ptr_arange_impl(const TV_METAL_DEVICE T* ptr,
          mp_list_int<Inds...>) {
  return array<TV_METAL_DEVICE T*, sizeof...(Inds)>{(ptr + Inds)...};
}
template <class T, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<TV_METAL_DEVICE T*, sizeof...(Inds)>
create_ptr_arange_impl(TV_METAL_DEVICE T* ptr,
          mp_list_int<Inds...>) {
  return array<TV_METAL_DEVICE T*, sizeof...(Inds)>{(ptr + Inds)...};
}

}

template <int N, class T>
TV_HOST_DEVICE_INLINE constexpr array<const TV_METAL_DEVICE T*, N>
create_ptr_arange(const TV_METAL_DEVICE T* ptr) {
  return arrayops_detail::create_ptr_arange_impl(ptr, mp_make_list_c_sequence<int, N>{});
}

template <int N, class T>
TV_HOST_DEVICE_INLINE constexpr array<TV_METAL_DEVICE T*, N>
create_ptr_arange(TV_METAL_DEVICE T* ptr) {
  return arrayops_detail::create_ptr_arange_impl(ptr, mp_make_list_c_sequence<int, N>{});
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(TV_METAL_CONSTANT const T *ptr) {
  return reinterpret_cast<const TV_METAL_CONSTANT tv::array<T, N> *>(ptr)[0];
}
template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE TV_METAL_CONSTANT tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(TV_METAL_CONSTANT T *ptr) {
  return reinterpret_cast<TV_METAL_CONSTANT tv::array_nd<T, Ns...> *>(ptr);
}
template <size_t N, typename T>
TV_HOST_DEVICE_INLINE TV_METAL_CONSTANT tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(TV_METAL_CONSTANT T *ptr) {
  return reinterpret_cast<TV_METAL_CONSTANT tv::alignedarray<T, N> *>(ptr);
}

#ifndef TV_METAL_RTC
template <typename T>
class PointerValueReader {
public:
  TV_HOST_DEVICE_INLINE constexpr PointerValueReader(T *ptr) : ptr_(ptr) {}
  TV_HOST_DEVICE_INLINE constexpr T operator[](size_t i) const {
    T res = ptr_[i];
    return res;
  }
private:
  T* ptr_;
};
#else 
template <typename T>
class PointerValueReader {
public:
  TV_HOST_DEVICE_INLINE constexpr PointerValueReader(thread T *ptr) : ptr_(ptr) {}
  TV_HOST_DEVICE_INLINE constexpr T operator[](size_t i) const {
    T res = ptr_[i];
    return res;
  }
private:
  thread T* ptr_;
};
#endif

// template <class T>
// TV_HOST_DEVICE_INLINE PointerValueReader<T> create_pointer_value_reader(TV_METAL_CONSTANT T *ptr) {
//   return PointerValueReader<T>(ptr);
// }

#ifndef TV_METAL_RTC
template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE const tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(const T *ptr) {
  return reinterpret_cast<const tv::array_nd<T, Ns...> *>(ptr);
}
template <size_t N, typename T>
TV_HOST_DEVICE_INLINE const tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(const T *ptr) {
  return reinterpret_cast<const tv::alignedarray<T, N> *>(ptr);
}


#endif
#define OP_RESHAPE_DECLARE_FOR_METAL(metal_type, ref_type)                     \
  template <int... Ns, typename T, size_t N, size_t Align>                     \
  TV_HOST_DEVICE_INLINE constexpr metal_type auto ref_type reshape(                      \
      metal_type array<T, N, Align> ref_type arr) {                            \
    using shape =                                                              \
        typename detail::get_tv_array_shape<array<T, N, Align>>::type;         \
    constexpr size_t shapeProd =                                               \
        mp_reduce_prod<shape,                                                  \
                       std::integral_constant<size_t, size_t(1)>>::value;      \
    constexpr int reshapeProd =                                                \
        mp_reduce_prod<mp_list_c<int, int(Ns == -1 ? 1 : Ns)...>,              \
                       std::integral_constant<size_t, size_t(1)>>::value;      \
    static_assert(                                                             \
        shapeProd ==                                                           \
            mp_reduce_prod<                                                    \
                mp_list_c<int, int(Ns == -1 ? int(shapeProd) / reshapeProd     \
                                            : Ns)...>,                         \
                std::integral_constant<int, int(1)>>::value,                   \
        "reshape size mismatch");                                              \
    static_assert(                                                             \
        1 >= mp_reduce_sum<mp_list_c<int, int(Ns == -1)...>,                   \
                           std::integral_constant<int, int(0)>>::value,        \
        "shape can only have one -1");                                         \
    static_assert(                                                             \
        0 == mp_reduce_sum<mp_list_c<int, int(Ns < -1)...>,                    \
                           std::integral_constant<int, int(0)>>::value,        \
        "shape can't have negative number other than -1");                     \
    return reinterpret_cast<metal_type tv::array_nd<                           \
        tv::detail::get_nested_element_t<T>,                                       \
        (Ns == -1 ? int(shapeProd) / reshapeProd : Ns)...>                     \
                                ref_type>(arr);                                \
  }

OP_RESHAPE_DECLARE_FOR_METAL(TV_METAL_CONSTANT, &);
OP_RESHAPE_DECLARE_FOR_METAL(TV_METAL_CONSTANT, &&);
#ifndef TV_METAL_RTC
OP_RESHAPE_DECLARE_FOR_METAL(const, &);
OP_RESHAPE_DECLARE_FOR_METAL(const, &&);
#endif

#ifdef TV_METAL_RTC
template <typename T>
class PointerValueReader<device T> {
public:
  TV_HOST_DEVICE_INLINE constexpr PointerValueReader(device T *ptr) : ptr_(ptr) {}
  TV_HOST_DEVICE_INLINE T constexpr operator[](size_t i) const {
    T res = ptr_[i];
    return res;
  }
private:
  device T* ptr_;
};


template <typename T>
class PointerValueReader<threadgroup T> {
public:
  TV_HOST_DEVICE_INLINE constexpr PointerValueReader(threadgroup T *ptr) : ptr_(ptr) {}
  TV_HOST_DEVICE_INLINE constexpr T operator[](size_t i) const {
    T res = ptr_[i];
    return res;
  }
private:
  threadgroup T* ptr_;
};

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(const thread T *ptr) {
  return reinterpret_cast<const thread tv::array<T, N> *>(ptr)[0];
}
template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(const device T *ptr) {
  return reinterpret_cast<const device tv::array<T, N> *>(ptr)[0];
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(const threadgroup T *ptr) {
  return reinterpret_cast<const threadgroup tv::array<T, N> *>(ptr)[0];
}

template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE thread tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(thread T *ptr) {
  return reinterpret_cast<thread tv::array_nd<T, Ns...> *>(ptr);
}

template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE device tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(device T *ptr) {
  return reinterpret_cast<device tv::array_nd<T, Ns...> *>(ptr);
}

template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE threadgroup tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(threadgroup T *ptr) {
  return reinterpret_cast<threadgroup tv::array_nd<T, Ns...> *>(ptr);
}

template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE thread const tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(thread const T *ptr) {
  return reinterpret_cast<thread const tv::array_nd<T, Ns...> *>(ptr);
}

template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE device const tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(device const T *ptr) {
  return reinterpret_cast<device const tv::array_nd<T, Ns...> *>(ptr);
}

template <size_t... Ns, typename T>
TV_HOST_DEVICE_INLINE threadgroup const tv::array_nd<T, Ns...> *
reinterpret_cast_array_nd(threadgroup const T *ptr) {
  return reinterpret_cast<threadgroup const tv::array_nd<T, Ns...> *>(ptr);
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE thread tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(thread T *ptr) {
  return reinterpret_cast<thread tv::alignedarray<T, N> *>(ptr);
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE threadgroup tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(threadgroup T *ptr) {
  return reinterpret_cast<threadgroup tv::alignedarray<T, N> *>(ptr);
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE device tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(device T *ptr) {
  return reinterpret_cast<device tv::alignedarray<T, N> *>(ptr);
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE thread const tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(thread const T *ptr) {
  return reinterpret_cast<thread const tv::alignedarray<T, N> *>(ptr);
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE threadgroup const tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(threadgroup const T *ptr) {
  return reinterpret_cast<threadgroup const tv::alignedarray<T, N> *>(ptr);
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE device const tv::alignedarray<T, N> *
reinterpret_cast_alignedarray(device const T *ptr) {
  return reinterpret_cast<device const tv::alignedarray<T, N> *>(ptr);
}

OP_RESHAPE_DECLARE_FOR_METAL(thread, &);
OP_RESHAPE_DECLARE_FOR_METAL(thread, &&);
OP_RESHAPE_DECLARE_FOR_METAL(threadgroup, &);
OP_RESHAPE_DECLARE_FOR_METAL(threadgroup, &&);

OP_RESHAPE_DECLARE_FOR_METAL(const thread, &);
OP_RESHAPE_DECLARE_FOR_METAL(const thread, &&);
OP_RESHAPE_DECLARE_FOR_METAL(const threadgroup, &);
OP_RESHAPE_DECLARE_FOR_METAL(const threadgroup, &&);

#endif

template <class... Ts>
TV_HOST_DEVICE_INLINE constexpr auto create_array(Ts... vals) {
  return array<mp_nth_t<0, Ts...>, sizeof...(Ts)>{vals...};
}


template <typename T, size_t N, size_t Align> struct identity {
  template <class... Args>
  TV_HOST_DEVICE_INLINE constexpr decltype(auto)
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TV_METAL_THREAD Args&& ...) {
    return self;
  }
  template <class... Args>
  TV_HOST_DEVICE_INLINE constexpr decltype(auto)
  operator()(const TV_METAL_THREAD array<T, N, Align> &&self, TV_METAL_THREAD Args&& ...) {
    return self;
  }
};

template <typename T, size_t N, size_t Align> struct reduce_max {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(max_impl, self);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const TV_METAL_THREAD Tnested &
  max_impl(const TV_METAL_THREAD Tnested &a, const TV_METAL_THREAD Tnested &b) {
    return (a < b) ? b : a;
  }
};

template <typename T, size_t N, size_t Align> struct reduce_min {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(min_impl, self);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const TV_METAL_THREAD Tnested &
  min_impl(const TV_METAL_THREAD Tnested &a, const TV_METAL_THREAD Tnested &b) {
    return (b < a) ? b : a;
  }
};
template <typename T, size_t N, size_t Align> struct maximum {
  using Tnested = tv::detail::get_nested_element_t<T>;
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TOther other) {
    return apply(max_impl, self, other);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const TV_METAL_THREAD Tnested &
  max_impl(const TV_METAL_THREAD Tnested &a, const TV_METAL_THREAD Tnested &b) {
    return (a < b) ? b : a;
  }
};

template <typename T, size_t N, size_t Align> struct minimum {
  using Tnested = tv::detail::get_nested_element_t<T>;

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TOther other) {
    return apply(min_impl, self, other);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const TV_METAL_THREAD Tnested &
  min_impl(const TV_METAL_THREAD Tnested &a, const TV_METAL_THREAD Tnested &b) {
    return (b < a) ? b : a;
  }
};


template <typename T, size_t N, size_t Align> struct clamp {
  using Tnested = tv::detail::get_nested_element_t<T>;

  template <typename T1, typename T2>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, T1 min_v, T2 max_v) {
    return apply(clamp_impl, self, min_v, max_v);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  clamp_impl(const TV_METAL_THREAD Tnested& val, const TV_METAL_THREAD Tnested &min_v, const TV_METAL_THREAD Tnested &max_v) {
    return MathScalarOp<Tnested>::clamp(val, min_v, max_v);
  }
};

template <typename T, size_t N, size_t Align> struct abs {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(abs_impl, self);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr Tnested
  abs_impl(const TV_METAL_THREAD Tnested &a) {
    return a >= Tnested(0) ? a : -a;
  }
};

template <typename T, size_t N, size_t Align> struct floor {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(MathScalarOp<Tnested>::floor, self);
  }
};

template <typename T, size_t N, size_t Align> struct ceil {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(MathScalarOp<Tnested>::ceil, self);
  }
};

template <typename T, size_t N, size_t Align> struct sum {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(tv::detail::array_sum<T>, self);
  }
};

template <typename T, size_t N, size_t Align> struct mean {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(tv::detail::array_sum<T>, self) / T(N);
  }
};

template <typename T, size_t N, size_t Align> struct prod {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(tv::detail::array_mul<T>, self);
  }
};

template <typename T, size_t N, size_t Align> struct dot {
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TOther other) {
    return (self * other).template op<sum>();
  }
};

template <typename T, size_t N, size_t Align> struct l2norm {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return MathScalarOp<T>::sqrt(self.template op<dot>(self));
  }
};

template <typename T, size_t N, size_t Align> struct rl2norm {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return MathScalarOp<T>::rsqrt(self.template op<dot>(self));
  }
};

template <typename T, size_t N, size_t Align> struct length {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return MathScalarOp<T>::sqrt(self.template op<dot>(self));
  }
};

template <typename T, size_t N, size_t Align> struct length2 {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return self.template op<dot>(self);
  }
};

template <typename T, size_t N, size_t Align> struct normalize {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return self * self.template op<rl2norm>();
  }
};

template <typename T, size_t N, size_t Align> struct normalize_grad {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &inp) {
    auto length_squ = inp.template op<length2>();
    T length2_minus_3_div_2 = MathScalarOp<T>::rsqrt(length_squ * length_squ * length_squ);
    auto grad_mul_inp = grad * inp;
    auto grad_mul_inp_sum = grad_mul_inp.template op<sum>();
    return ((length_squ - inp * inp) * grad - inp * (grad_mul_inp_sum - grad_mul_inp)) * length2_minus_3_div_2;
  }
};

template <typename T, size_t N, size_t Align> struct normalize_grad_out {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &out, const TV_METAL_THREAD array<T, N, Align> &inp) {
    auto length_squ = inp.template op<length2>();
    T length2_minus_3_div_2 = MathScalarOp<T>::rsqrt(length_squ * length_squ * length_squ);
    auto grad_mul_inp = grad * inp;
    auto grad_mul_inp_sum = grad_mul_inp.template op<sum>();
    return ((length_squ - inp * inp) * grad - inp * (grad_mul_inp_sum - grad_mul_inp)) * length2_minus_3_div_2;
  }
};
template <typename T, size_t N, size_t Align> struct exponential;

template <typename T, size_t N, size_t Align> struct log {
  using Tnested = tv::detail::get_nested_element_t<T>;
  using inverse_t = exponential<T, N, Align>;
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(impl, self);
  }
private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& self) {
    return MathScalarOp<Tnested>::log(self);
  }
};

template <typename T, size_t N, size_t Align> struct log_grad{
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(impl, grad, self);
  }
private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& grad, const TV_METAL_THREAD Tnested& self) {
    return grad / self;
  }
};

template <typename T, size_t N, size_t Align> struct logit;
template <typename T, size_t N, size_t Align> struct sigmoid {
  using Tnested = tv::detail::get_nested_element_t<T>;
  using inverse_t = logit<T, N, Align>;
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(impl, self);
  }
private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& self) {
    return Tnested(1) / (Tnested(1) + MathScalarOp<Tnested>::exp(-self));
  }
};

template <typename T, size_t N, size_t Align> struct sigmoid_grad_out {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &out) {
    return apply(impl, grad, out);
  }
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &out, const TV_METAL_THREAD array<T, N, Align> &inp) {
    return apply(impl, grad, out);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& grad, const TV_METAL_THREAD Tnested& out) {
    return grad * out * (1 - out);
  }
};

template <typename T, size_t N, size_t Align> struct logit {
  using Tnested = tv::detail::get_nested_element_t<T>;
  using inverse_t = sigmoid<T, N, Align>;
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(impl, self);
  }
private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& self) {
    return MathScalarOp<Tnested>::log(self / (Tnested(1) - self));
  }
};

template <typename T, size_t N, size_t Align> struct exponential {
  using Tnested = tv::detail::get_nested_element_t<T>;
  using inverse_t = log<T, N, Align>;
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(impl, self);
  }
private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& val) {
    return MathScalarOp<Tnested>::exp(val);
  }
};

template <typename T, size_t N, size_t Align> struct fast_exponential {
  using Tnested = tv::detail::get_nested_element_t<T>;
  using inverse_t = log<T, N, Align>;
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(impl, self);
  }
private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& self) {
    return MathScalarOp<Tnested>::fast_exp(self);
  }
};

template <typename T, size_t N, size_t Align> struct exponential_grad_out {
  using Tnested = tv::detail::get_nested_element_t<T>;

  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &out) {
    return apply(impl, grad, out);
  }
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &grad, const TV_METAL_THREAD array<T, N, Align> &out, const TV_METAL_THREAD array<T, N, Align> &inp) {
    return apply(impl, grad, out);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const Tnested
  impl(const TV_METAL_THREAD Tnested& grad, const TV_METAL_THREAD Tnested& out) {
    return grad * out;
  }
};

template <typename T, size_t N, size_t Align> struct cross {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, 2, Align> &a,
             const TV_METAL_THREAD array<T, 2, Align> &b) {
    return a[0] * b[1] - a[1] * b[0];
  }
  TV_HOST_DEVICE_INLINE constexpr array<T, 2, Align>
  operator()(const TV_METAL_THREAD array<T, 2, Align> &a,
             const TV_METAL_THREAD T &b) {
    return {a[1] * b, -a[0] * b};
  }
  TV_HOST_DEVICE_INLINE constexpr array<T, 2, Align>
  operator()(const TV_METAL_THREAD T &a,
             const TV_METAL_THREAD array<T, 2, Align> &b) {
    return {-a * b[1], a * b[0]};
  }
  TV_HOST_DEVICE_INLINE constexpr array<T, 3, Align>
  operator()(const TV_METAL_THREAD array<T, 3, Align> &a,
             const TV_METAL_THREAD array<T, 3, Align> &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  }
};

#ifndef TV_METAL_RTC
template <typename T, size_t N, size_t Align> struct zeros {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return array<T, N, Align>{};
  }
};

template <typename T, size_t N, size_t Align> struct ones {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return constant_array<T, N, Align>(T(1));
  }
};

template <typename T, size_t N, size_t Align> struct full {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, T val) {
    return constant_array<T, N, Align>(val);
  }
};

#endif

// constexpr tv::array<float, 3> a{1, 2, 3};
// constexpr tv::array<float, 3> aa{4, 5, 6};
// template<class T>
// TV_HOST_DEVICE_INLINE constexpr const T& maxX(const T& a, const T& b)
// {
//     return (a < b) ? b : a;
// }

// constexpr auto b = (a * aa).op<arrayops::min>(2);
// constexpr auto c = (a * aa).cast<int>().op<max>(0).op<min>(1).cast<float>() +
// 0.5f; constexpr auto d = c / aa; using WTF =
// std::decay_t<return_type_t<decltype(maxX<float>)>>;
} // namespace arrayops
} // namespace tv

#undef OP_RESHAPE_DECLARE_FOR_METAL