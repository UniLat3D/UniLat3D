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

#include "linear.cu.h"
#include <tensorview/tensor.h>
#include <tensorview/cuda/launch.h>

namespace tv {

namespace hash {

template <typename THashTable>
void clear_set(THashTable table, cudaStream_t stream = 0) {
  auto launcher = tv::cuda::Launch(table.size(), stream);
  launcher(clear_set_kernel<THashTable>, table);
}

template <typename THashTable>
void clear_map(THashTable table, cudaStream_t stream = 0) {
  auto launcher = tv::cuda::Launch(table.size(), stream);
  launcher(clear_map_kernel<THashTable>, table);
}

template <typename THashTable>
void clear_map_split(THashTable table, cudaStream_t stream = 0) {
  auto launcher = tv::cuda::Launch(table.size(), stream);
  launcher(clear_map_kernel_split<THashTable>, table);
}

template <typename THashTable>
tv::Tensor
items_map(THashTable table, typename THashTable::mapped_type empty_value,
          tv::Tensor out = tv::Tensor(), cudaStream_t stream = nullptr) {
  auto count = tv::zeros({1}, tv::int32, 0);
  if (out.empty()) {
    out = tv::Tensor({table.size()},
                     tv::type_v<typename THashTable::value_type>, 0);
  } else {
    TV_ASSERT_INVALID_ARG(out.device() == 0 && out.ndim() == 1 &&
                              out.dtype() ==
                                  tv::type_v<typename THashTable::value_type>,
                          "error");
  }
  auto launcher = tv::cuda::Launch(table.size(), stream);
  auto out_tv = out.data_ptr<typename THashTable::value_type>();
  launcher(iterate_table<THashTable>, table, out_tv, count.data_ptr<int32_t>(),
           empty_value, out.dim(0));
  auto count_cpu = count.cpu();
  auto count_val = count_cpu.item<int32_t>();
  return out.slice_first_axis(0, count_val);
}

template <typename THashTable>
tv::Tensor items_map(THashTable table, tv::Tensor out = tv::Tensor(),
                     cudaStream_t stream = nullptr) {
  auto count = tv::zeros({1}, tv::int32, 0);
  if (out.empty()) {
    out = tv::Tensor({table.size()},
                     tv::type_v<typename THashTable::value_type>, 0);
  } else {
    TV_ASSERT_INVALID_ARG(out.device() == 0 && out.ndim() == 1 &&
                              out.dtype() ==
                                  tv::type_v<typename THashTable::value_type>,
                          "error");
  }
  auto launcher = tv::cuda::Launch(table.size(), stream);
  auto out_tv = out.data_ptr<typename THashTable::value_type>();
  launcher(iterate_table_oneshot<THashTable>, table, out_tv,
           count.data_ptr<int32_t>(), out.dim(0));
  auto count_cpu = count.cpu();
  auto count_val = count_cpu.item<int32_t>();
  return out.slice_first_axis(0, count_val);
}

template <typename THashTable>
std::tuple<tv::Tensor, tv::Tensor>
items_map_split(THashTable table, tv::Tensor out_k = tv::Tensor(),
                tv::Tensor out_v = tv::Tensor(),
                tv::Tensor count = tv::Tensor(),
                cudaStream_t stream = nullptr) {
  if (count.empty()) {
    count = tv::empty({1}, tv::type_v<typename THashTable::size_type>, 0);
  }
  auto ctx = tv::Context();
  ctx.set_cuda_stream(stream);
  count.zero_(ctx);
  if (out_k.empty()) {
    out_k = tv::Tensor({table.size()},
                       tv::type_v<typename THashTable::key_type>, 0);
  } else {
    TV_ASSERT_INVALID_ARG(
        out_k.device() == 0 && out_k.ndim() == 1 &&
            out_k.itemsize() ==
                sizeof(tv::type_v<typename THashTable::key_type>),
        "error");
  }
  if (out_v.empty()) {
    out_v = tv::Tensor({table.size()},
                       tv::type_v<typename THashTable::mapped_type>, 0);
  } else {
    TV_ASSERT_INVALID_ARG(
        out_v.device() == 0 && out_v.ndim() == 1 &&
            out_v.itemsize() ==
                sizeof(tv::type_v<typename THashTable::mapped_type>),
        "error");
  }
  TV_ASSERT_INVALID_ARG(out_k.dim(0) == out_v.dim(0), "error");

  auto launcher = tv::cuda::Launch(table.size(), stream);
  launcher(
      iterate_table_split<THashTable, typename THashTable::size_type>, table,
      reinterpret_cast<typename THashTable::key_type *>(out_k.raw_data()),
      reinterpret_cast<typename THashTable::mapped_type *>(out_v.raw_data()),
      out_k.dim(0), count.data_ptr<typename THashTable::size_type>());
  auto count_cpu = count.cpu(ctx);
  auto count_val = count_cpu.item<typename THashTable::size_type>();
  return std::make_tuple(out_k.slice_first_axis(0, count_val),
                         out_v.slice_first_axis(0, count_val));
}

template <typename THashTable>
tv::Tensor get_map_probe_length(THashTable table, tv::Tensor out = tv::Tensor(),
                                cudaStream_t stream = nullptr) {
  auto count = tv::zeros({1}, tv::int32, 0);
  if (out.empty()) {
    out = tv::Tensor({table.size()}, tv::int32, 0);
  } else {
    TV_ASSERT_INVALID_ARG(out.device() == 0 && out.ndim() == 1 &&
                              out.dtype() == tv::int32,
                          "error");
  }
  auto launcher = tv::cuda::Launch(table.size(), stream);
  auto out_tv = out.data_ptr<int32_t>();
  launcher(table_probe_length<THashTable>, table, out_tv,
           count.data_ptr<int32_t>(), out.dim(0));
  auto count_cpu = count.cpu();
  auto count_val = count_cpu.item<int32_t>();
  return out.slice_first_axis(0, count_val);
}

} // namespace hash

} // namespace tv