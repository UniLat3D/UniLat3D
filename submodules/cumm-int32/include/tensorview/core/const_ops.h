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
#include "defs.h"
namespace tv {
template <typename T> TV_HOST_DEVICE_INLINE constexpr T div_up(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T align_up(T a, T b) {
  return div_up(a, b) * b;
}

} // namespace tv