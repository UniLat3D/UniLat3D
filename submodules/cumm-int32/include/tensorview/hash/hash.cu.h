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

/*
GPUHashTable concepts

class HashTable {
  __device__ insert(K k, V v);
  __device__ lookup(K k);
  __host__ clear();
  __host__ __device__ data();
  __host__ __device__ size();
}

Problems of other hashing
RobinHood: Exch based, need to save a atomic value. so we can't use robinhood in
cuda.
Cuckoo: if cycle detected during insert, we must rehash whole table.
  linear probing don't need rehash if insert once.

*/

#pragma once
#include "hash_core.h"
#include <tensorview/device_ops.h>
#include <tensorview/kernel_utils.h>

namespace tv {
namespace hash {

template <typename THashTable>
__global__ void insert(THashTable table,
                       const typename THashTable::value_type *items,
                       size_t size) {
  for (auto i : tv::KernelLoopX<int>(size)) {
    auto item = items[i];
    table.insert(item.first, item.second);
  }
}

template <typename THashTableSplit>
__global__ void insert_split(
    THashTableSplit table,
    const typename THashTableSplit::key_type *__restrict__ key_ptr,
    const typename THashTableSplit::mapped_type *__restrict__ value_ptr,
    size_t size) {
  if (value_ptr == nullptr) {
    for (auto i : tv::KernelLoopX<int>(size)) {
      table.insert_key_only(key_ptr[i]);
    }
  } else {
    for (auto i : tv::KernelLoopX<int>(size)) {
      table.insert(key_ptr[i], value_ptr[i]);
    }
  }
}

template <typename THashSet>
__global__ void insert_set(THashSet set,
                           const typename THashSet::value_type *items,
                           size_t size) {
  for (auto i : tv::KernelLoopX<int>(size)) {
    set.insert(items[i]);
  }
}

template <typename THashTable>
__global__ void lookup(THashTable table,
                       typename THashTable::value_type *item_to_query,
                       size_t size) {
  for (auto i : tv::KernelLoopX<int>(size)) {
    auto &item = item_to_query[i];
    auto query = table.lookup(item.first);
    if (!query.empty()) {
      item.second = query.second;
    }
  }
}

template <typename THashTableSplit>
__global__ void
query_split(THashTableSplit table,
            typename THashTableSplit::key_type *__restrict__ key_ptr,
            typename THashTableSplit::mapped_type *__restrict__ value_ptr,
            uint8_t *is_empty, size_t size) {
  auto value_data = table.value_ptr();

  for (auto i : tv::KernelLoopX<int>(size)) {
    auto offset = table.lookup_offset(key_ptr[i]);
    is_empty[i] = offset == -1;
    if (offset != -1) {
      value_ptr[i] = value_data[offset];
    }
  }
}

template <typename THashTableSplit>
__global__ void query_split_benchmark(
    THashTableSplit table,
    typename THashTableSplit::key_type *__restrict__ key_ptr,
    typename THashTableSplit::mapped_type *__restrict__ value_ptr,
    size_t size) {
  auto value_data = table.value_ptr();

  for (auto i : tv::KernelLoopX<int>(size)) {
    auto offset = table.lookup_offset(key_ptr[i]);
    if (offset != -1) {
      value_ptr[i] = value_data[offset];
    }
  }
}

template <typename THashTable>
__global__ void
lookup_default(THashTable table,
               const typename THashTable::value_type *item_to_query,
               typename THashTable::mapped_type empty_value, size_t size) {
  for (auto i : tv::KernelLoopX<int>(size)) {
    auto &item = item_to_query[i];
    auto query = table.lookup(item.first);
    if (!query.empty()) {
      item.second = query.second;
    } else {
      item.second = empty_value;
    }
  }
}

template <typename THashTable> __global__ void clear_map_kernel(THashTable table) {
  auto data = table.data();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
    data[i].first = THashTable::empty_key;
  }
}

template <typename THashTable>
__global__ void clear_map_kernel_split(THashTable table) {
  auto data = table.key_ptr();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
    data[i] = THashTable::empty_key;
  }
}

template <typename THashSet> __global__ void clear_set_kernel(THashSet set) {
  auto data = set.data();
  for (auto i : tv::KernelLoopX<int>(set.size())) {
    data[i] = THashSet::empty_key;
  }
}

template <typename THashTable>
__global__ void
iterate_table(THashTable table, typename THashTable::value_type *out,
              int32_t *count, typename THashTable::mapped_type empty_value,
              int size_limit) {
  auto data = table.data();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
    auto &item = data[i];
    if (!item.empty() && item.second != empty_value) {
      int32_t old = tv::cuda::atomicAggInc(count);
      if (old < size_limit) {
        out[old] = item;
      }
    }
  }
}

template <typename THashTableSplit, typename TSize>
__global__ void
iterate_table_split(THashTableSplit table,
                    typename THashTableSplit::key_type *__restrict__ out_k,
                    typename THashTableSplit::mapped_type *__restrict__ out_v,
                    TSize size_limit, TSize *count) {
  auto key_ptr = table.key_ptr();
  auto v_ptr = table.value_ptr();

  for (auto i : tv::KernelLoopX<int>(table.size())) {
    auto key = key_ptr[i];
    if (key != THashTableSplit::empty_key) {
      TSize old = tv::cuda::atomicAggInc(count);
      if (old < size_limit) {
        out_k[old] = key;
        out_v[old] = v_ptr[i];
      }
    }
  }
}

template <typename THashTableSplit, typename TSize>
__global__ void assign_arange_split(THashTableSplit table, TSize *count) {
  auto key_ptr = table.key_ptr();
  auto v_ptr = table.value_ptr();

  for (auto i : tv::KernelLoopX<int>(table.size())) {
    auto key = key_ptr[i];
    if (key != THashTableSplit::empty_key) {
      TSize old = tv::cuda::atomicAggInc(count);
      v_ptr[i] = old;
    }
  }
}

template <typename THashTable>
__global__ void iterate_table_oneshot(THashTable table,
                                      typename THashTable::value_type *out,
                                      int32_t *count, int size_limit) {
  auto data = table.data();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
    auto &item = data[i];
    if (!item.empty()) {
      int32_t old = tv::cuda::atomicAggInc(count);
      if (old < size_limit) {
        out[old] = item;
      }
    }
  }
}

template <typename THashSet>
__global__ void iterate_set(THashSet set, typename THashSet::value_type *out,
                            int32_t *count, int size_limit) {
  auto data = set.data();
  for (auto i : tv::KernelLoopX<int>(set.size())) {
    auto &item = data[i];
    if (item != THashSet::empty_key) {
      int32_t old = tv::cuda::atomicAggInc(count);
      if (old < size_limit) {
        out[old] = item;
      }
    }
  }
}

template <typename THashTable>
__global__ void table_probe_length(THashTable table, int32_t *out,
                                   int32_t *count, int size_limit) {
  auto data = table.data();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
    auto &item = data[i];
    if (!item.empty()) {
      int32_t old = tv::cuda::atomicAggInc(count);
      if (old < size_limit) {
        auto hash_val = THashTable::hash_type::hash_scalar(
            reinterpret_cast<typename THashTable::key_type_uint&>(item.first));
        out[old] = (i - hash_val % table.size());
      }
    }
  }
}

template <typename THashTable>
__global__ void table_split_probe_length(THashTable table, int32_t *out,
                                         int32_t *count, int size_limit) {
  auto data = table.key_ptr();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
    auto key = data[i];
    if (key != THashTable::empty_key) {
      int32_t old = tv::cuda::atomicAggInc(count);
      if (old < size_limit) {
        auto hash_val = THashTable::hash_type::hash_scalar(
            reinterpret_cast<typename THashTable::key_type_uint&>(key));
        out[old] = (i - hash_val % table.size());
      }
    }
  }
}

} // namespace hash
} // namespace tv
