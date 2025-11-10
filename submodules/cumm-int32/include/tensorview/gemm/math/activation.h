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

#include "functional.h"
namespace tv {

namespace math {

template <typename T, typename Tout, size_t N> struct Clamp {
  using argument_t = tv::array<T, N>;
  using result_t = tv::array<T, N>;
  //   static constexpr T kClamp = T(std::numeric_limits<Tout>::max());
  TV_HOST_DEVICE_INLINE constexpr result_t operator()(const argument_t &src, T alpha = 1, T beta = 0) {
    constexpr T kClamp = T((1U << (sizeof(Tout) * 8 - 1)) - 1);
    minimum<argument_t> min_op;
    maximum<argument_t> max_op;
    argument_t intermediate = max_op(src, -kClamp - T(1));
    intermediate = min_op(intermediate, kClamp);
    return intermediate;
  }
};

template <typename T, typename Tout, size_t N> struct ReLU {
  using argument_t = tv::array<T, N>;
  using result_t = tv::array<T, N>;
  //   static constexpr T kClamp = T(std::numeric_limits<Tout>::max());
  TV_HOST_DEVICE_INLINE constexpr result_t operator()(const argument_t &src, T alpha = 1, T beta = 0) {
    maximum<argument_t> max_op;
    argument_t intermediate = max_op(src, T(0));
    return intermediate;
  }
};


} // namespace math
} // namespace tv