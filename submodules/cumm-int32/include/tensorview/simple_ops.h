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
#include <tensorview/tensor.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace tv {

namespace ops {

namespace ftor {
template <typename T> struct Norm {
  TV_HOST_DEVICE_INLINE T operator()(const T &x, const T &y) const {
    return (x - y) * (x - y);
  }
};

}

template <template <class> class OpClass, typename T = void>
void transform(cudaStream_t stream, tv::Tensor out, tv::Tensor a,
               tv::Tensor b) {
  TV_ASSERT_INVALID_ARG(a.shape() == b.shape() && a.shape() == out.shape(),
                        "erorr");
  using all_t = std::conditional_t<std::is_same<T, void>::value,
                                   tv::detail::all_tensor_types_t, mp_list<T>>;
  tv::Dispatch<all_t>()(a.dtype(), [&](auto TAValue) {
    using TA = decltype(TAValue);
    OpClass<TA> op;
    thrust::device_ptr<TA> ptr_a(a.data_ptr<TA>());
    thrust::device_ptr<TA> ptr_b(b.data_ptr<TA>());
    thrust::device_ptr<TA> ptr_o(out.data_ptr<TA>());
    auto ctx = thrust::cuda::par.on(stream);
    thrust::transform(ctx, ptr_a, ptr_a + a.size(), ptr_b, ptr_o, op);
  });
}

template <template <class> class OpClass, typename T = void>
tv::Tensor transform(cudaStream_t stream, tv::Tensor a, tv::Tensor b) {
  tv::Tensor out(a.shape(), a.dtype(), a.device());
  transform<OpClass, T>(stream, out, a, b);
  return out;
}

template <template <class> class OpClass, typename T = void>
void unary(cudaStream_t stream, tv::Tensor out, tv::Tensor a) {
  TV_ASSERT_INVALID_ARG(a.size() == out.size(), "erorr");
  using all_t = std::conditional_t<std::is_same<T, void>::value,
                                   tv::detail::all_tensor_types_t, mp_list<T>>;
  tv::Dispatch<all_t>()(a.dtype(), [&](auto TAValue) {
    using TA = decltype(TAValue);
    OpClass<TA> op;
    thrust::device_ptr<TA> ptr_a(a.data_ptr<TA>());
    thrust::device_ptr<TA> ptr_o(out.data_ptr<TA>());
    auto ctx = thrust::cuda::par.on(stream);
    thrust::transform(ctx, ptr_a, ptr_a + a.size(), ptr_o, op);
  });
}

template <template <class> class OpClass, typename T = void>
tv::Tensor reduce(cudaStream_t stream, tv::Tensor a, tv::Tensor init) {
  using all_t = std::conditional_t<std::is_same<T, void>::value,
                                   tv::detail::all_tensor_types_t, mp_list<T>>;
  tv::Tensor res({1}, a.dtype());
  tv::Dispatch<all_t>()(a.dtype(), [&](auto TAValue) {
    using TA = decltype(TAValue);
    OpClass<TA> op;
    thrust::device_ptr<TA> ptr_a(a.data_ptr<TA>());
    auto ctx = thrust::cuda::par.on(stream);
    TA sum = thrust::reduce(ctx, ptr_a, ptr_a + a.size(), init.item<TA>(), op);
    *(res.data_ptr<TA>()) = sum;
  });
  return res;
}

template <template <class> class OpClass, typename T = void>
tv::Tensor reduce(cudaStream_t stream, tv::Tensor a) {
  tv::Tensor init({1}, a.dtype());
  init.zero_();
  return reduce<OpClass, T>(stream, a, init);
}

} // namespace ops
} // namespace tv