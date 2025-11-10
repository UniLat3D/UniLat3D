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
#include <tensorview/core/all.h>

namespace tv {

template <typename T, typename Fragment>
tv::array<T, 3> TV_DEVICE_INLINE get_fragment_meta(Fragment const &frag) {
  tv::array<T, 3> res{T(0), T(frag[0]), T(frag[0])};
  for (int i = 0; i < frag.size(); ++i) {
    res[0] += T(frag[i]);
    res[1] = T(frag[i]) > res[1] ? T(frag[i]) : res[1];
    res[2] = T(frag[i]) < res[2] ? T(frag[i]) : res[2];
  }
  res[0] /= T(frag.size());
  return res;
}

template <typename T, typename TPtr, int Size>
tv::array<T, 3> TV_DEVICE_INLINE get_ptr_meta(const TPtr *ptr) {
  tv::array<T, 3> res{T(0), T(ptr[0]), T(ptr[0])};
  for (int i = 0; i < Size; ++i) {
    res[0] += T(ptr[i]);
    res[1] = T(ptr[i]) > res[1] ? T(ptr[i]) : res[1];
    res[2] = T(ptr[i]) < res[2] ? T(ptr[i]) : res[2];
  }
  res[0] /= T(Size);
  return res;
}

template <typename T, typename Fragment, class... Ts>
void TV_DEVICE_INLINE print_fragment_meta(Fragment const &frag, Ts... args) {
  auto res = get_fragment_meta<T>(frag);
  tv::printf2("mean:", res[0], ", max:", res[1], ", min:", res[2]);
}

template <typename T, unsigned Tx = 0, typename Fragment, class... Ts>
void TV_DEVICE_INLINE print_fragment_meta_once(Fragment const &frag,
                                               Ts... args) {
  auto res = get_fragment_meta<T>(frag);
  tv::printf2_once<' ', Tx>(args..., "mean:", res[0], ", max:", res[1],
                            ", min:", res[2]);
}

template <typename T, typename Fragment, class... Ts>
void TV_DEVICE_INLINE print_fragment_meta_block_once(Fragment const &frag,
                                                     Ts... args) {
  auto res = get_fragment_meta<T>(frag);
  tv::printf2_block_once(threadIdx.x, args..., "mean:", res[0],
                         ", max:", res[1], ", min:", res[2]);
}

template <typename T, int Size, unsigned Tx = 0, typename TPtr, class... Ts>
void TV_DEVICE_INLINE print_ptr_meta_once(const TPtr *ptr, Ts... args) {
  auto res = get_ptr_meta<T, TPtr, Size>(ptr);
  tv::printf2_once<' ', Tx>(args..., "ptr mean:", res[0], ", max:", res[1],
                            ", min:", res[2]);
}

template <typename T, int Size, typename TPtr, class... Ts>
void TV_DEVICE_INLINE print_ptr_meta_block_once(const TPtr *ptr, Ts... args) {
  auto res = get_ptr_meta<T, TPtr, Size>(ptr);
  tv::printf2_block_once(threadIdx.x, args..., "ptr mean:", res[0],
                         ", max:", res[1], ", min:", res[2]);
}

// template <typename T, typename Fragment>
// void TV_DEVICE_INLINE print_fragment_meta_once(Fragment const& frag, const
// char* msg){
//     auto res = get_fragment_meta<T>(frag);
//     tv::printf2_once(msg, "mean:", res[0], ", max:", res[1], ", min:",
//     res[2]);
// }

// template <typename T, typename Fragment>
// void TV_DEVICE_INLINE print_fragment_meta_block_once(Fragment const& frag,
// const char* msg){
//     auto res = get_fragment_meta<T>(frag);
//     tv::printf2_block_once(msg, threadIdx.x, "mean:", res[0], ", max:",
//     res[1], ", min:", res[2]);
// }

namespace detail {

template <typename T, unsigned Tx, typename Fragment, int... Inds>
void TV_DEVICE_INLINE print_fragment_once_impl(Fragment const &frag,
                                               mp_list_int<Inds...>) {
  tv::printf2_once<' ', Tx>(T(frag[Inds])...);
}

template <typename T, typename Fragment, int... Inds>
void TV_DEVICE_INLINE print_fragment_block_once_impl(Fragment const &frag,
                                                     mp_list_int<Inds...>) {
  tv::printf2_block_once(threadIdx.x, T(frag[Inds])...);
}

template <typename T, unsigned Tx, typename TPtr, int... Inds>
void TV_DEVICE_INLINE print_ptr_once_impl(const TPtr *ptr,
                                          mp_list_int<Inds...>) {
  tv::printf2_once<' ', Tx>(T(ptr[Inds])...);
}

template <typename T, typename TPtr, int... Inds>
void TV_DEVICE_INLINE print_ptr_block_once_impl(const TPtr *ptr,
                                                mp_list_int<Inds...>) {
  tv::printf2_block_once(threadIdx.x, T(ptr[Inds])...);
}

} // namespace detail

template <typename T, int Start, int End, unsigned Tx = 0, typename Fragment>
void TV_DEVICE_INLINE print_fragment_once(Fragment const &frag) {
  return detail::print_fragment_once_impl<T, Tx>(
      frag, mp_list_int_range<Start, End>{});
}

template <typename T, int Start, int End, typename Fragment>
void TV_DEVICE_INLINE print_fragment_block_once(Fragment const &frag) {
  return detail::print_fragment_block_once_impl<T>(
      frag, mp_list_int_range<Start, End>{});
}

template <typename T, int Start, int End, unsigned Tx = 0, typename TPtr>
void TV_DEVICE_INLINE print_ptr_once(const TPtr *ptr) {
  return detail::print_ptr_once_impl<T, Tx>(ptr,
                                            mp_list_int_range<Start, End>{});
}

template <typename T, int Start, int End, typename TPtr>
void TV_DEVICE_INLINE print_ptr_block_once(const TPtr *ptr) {
  return detail::print_ptr_block_once_impl<T>(ptr,
                                              mp_list_int_range<Start, End>{});
}

} // namespace tv
