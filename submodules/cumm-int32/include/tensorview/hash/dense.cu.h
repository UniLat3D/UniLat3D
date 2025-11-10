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
#include "hash.cu.h"

namespace tv {

namespace hash {

template <typename K, typename V, typename Hash,
          typename detail::MapTypeToUnsignedInt<K>::type EmptyKey =
              default_empty_key_v<K>>
struct DenseTable {
public:
  using key_type_uint =
      typename tv::hash::detail::MapTypeToUnsignedInt<K>::type;
  using key_type = K;
  using value_type = pair<K, V>;
  using mapped_type = V;
  using size_type = int32_t;
  static constexpr auto empty_key = EmptyKey;

private:
  mapped_type *table_ = nullptr;
  size_type hash_size_;
  Hash hash_ftor_ = Hash();

public:
  explicit TV_HOST_DEVICE DenseTable(mapped_type *table, size_type hash_size)
      : table_(table), hash_size_(hash_size) {}

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

  TV_HOST_DEVICE_INLINE mapped_type *data() { return table_; }

  TV_HOST_DEVICE_INLINE mapped_type *data() const { return table_; }

  void TV_DEVICE_INLINE insert(const K &key, const V &value) {
    table_[key] = value;
  }

  value_type TV_DEVICE_INLINE lookup(K key) {
    auto item = table_[key];
    if (item == empty_key) {
      return {empty_key, V()};
    } else {
      return {key, item};
    }
  }

  void TV_DEVICE delete_item(K key, V empty_value) {
    table_[key] = empty_value;
    return;
  }

  void clear(cudaStream_t stream = 0) {
    tv::FillDev<mapped_type, 1, tv::DefaultPtrTraits, TV_GLOBAL_INDEX> filler;

    if (stream) {
      filler.run_async(
          tv::TensorView<mapped_type, 1, tv::DefaultPtrTraits, TV_GLOBAL_INDEX>(
              table_, {hash_size_}),
          empty_key, stream);
    } else {
      filler.run(
          tv::TensorView<mapped_type, 1, tv::DefaultPtrTraits, TV_GLOBAL_INDEX>(
              table_, {hash_size_}),
          empty_key);
    }
  }
};

} // namespace hash
} // namespace tv