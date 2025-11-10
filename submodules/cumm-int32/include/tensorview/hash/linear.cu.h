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
#include <tensorview/core/defs.h>

#ifndef TV_PARALLEL_RTC
#include "hash.cu.h"
#endif
#include "hash_functions.h"
#ifdef TV_METAL_CC
#include <metal_stdlib>
// #include "metal_atomic_cas.h"
#endif
namespace tv {

namespace hash {

namespace detail {
#ifdef TV_METAL_CC
template <typename T> class AtomicCASDispatch {
public:
  static TV_DEVICE_INLINE bool
  atomic_cas_weak_for_hash(device T *object, thread T empty_key, T desired) {
    // auto cur = metal::atomic_load_explicit(
    //     reinterpret_cast<device metal::atomic<T> *>(object),
    //     metal::memory_order_relaxed);

    if (desired == empty_key) {
      return true;
    }
    T expected = empty_key;
    // }
    bool success;
    T cur;
    do {
      success = metal::atomic_compare_exchange_weak_explicit(
          reinterpret_cast<device metal::atomic<T> *>(object), &expected,
          desired, metal::memory_order_relaxed, metal::memory_order_relaxed);
      cur = expected;
      expected = empty_key;
    } while (!success && empty_key == cur);
    return success || cur == desired;
  }
};

template <> class AtomicCASDispatch<ulong> {
  /*
  In Apple Metal, native atomicCAS 64bit isn't supported, so we use another way
  to implement a fake 64bit (actually 62bit) lock-free hash table
   */
public:
  static TV_DEVICE_INLINE bool atomic_cas_weak_for_hash(device ulong *object,
                                                        thread ulong empty_key,
                                                        ulong desired) {
    uint empty_first_part = reinterpret_cast<thread uint *>(&empty_key)[0];
    uint empty_second_part = reinterpret_cast<thread uint *>(&empty_key)[1];

    device uint *object_uint_ptr = reinterpret_cast<device uint *>(object);
    auto desired_first_part = reinterpret_cast<thread uint *>(&desired)[0];
    auto desired_second_part = reinterpret_cast<thread uint *>(&desired)[1];
    auto expected_first_part = empty_first_part;

    bool success;

    uint cur_first_part;
    do {
      success = metal::atomic_compare_exchange_weak_explicit(
          reinterpret_cast<device metal::atomic<uint> *>(object_uint_ptr),
          &expected_first_part, desired_first_part, metal::memory_order_relaxed,
          metal::memory_order_relaxed);
      cur_first_part = expected_first_part;
      expected_first_part = empty_first_part;
      // user key are mapped to ensure first and second part always not equal to
      // expected.
    } while (!success && (cur_first_part == empty_first_part));
    if (success) {
      metal::atomic_store_explicit(
          reinterpret_cast<device metal::atomic<uint> *>(object_uint_ptr) + 1,
          desired_second_part, metal::memory_order_relaxed);
      return true;
    }
    // insert failed.
    uint cur_second_part;
    do {
      cur_second_part = metal::atomic_load_explicit(
          reinterpret_cast<device metal::atomic<uint> *>(object_uint_ptr) + 1,
          metal::memory_order_relaxed);
    } while (cur_second_part == empty_second_part);
    return cur_first_part == desired_first_part &&
           cur_second_part == desired_second_part;
  }
};

#endif
} // namespace detail

template <typename K, typename V, typename Hash = tv::hash::Murmur3Hash<K>,
          K EmptyKey = default_empty_key_v<K>, bool Power2 = false>
struct LinearHashTable {
public:
  using key_type_uint = to_unsigned_t<K>;
  using key_type = K;
  using value_type = pair<K, V, EmptyKey>;
  using mapped_type = V;
  using size_type = int32_t;
  using hash_type = Hash;
  // TODO no way to get constexpr uint repr of key for now.
  // static constexpr auto empty_key_uint = value_type::empty_key_uint;
  static constexpr TV_METAL_CONSTANT auto empty_key = EmptyKey;

private:
  TV_METAL_DEVICE value_type *table_ = nullptr;
  size_type hash_size_;
  key_type_uint empty_key_uint;
  static constexpr TV_METAL_CONSTANT auto kNumHashArgs =
      argument_size_v<decltype(hash_type::hash)>;

public:
  TV_HOST_DEVICE_INLINE explicit LinearHashTable(
      TV_METAL_DEVICE value_type *table, size_type hash_size)
      : table_(table), hash_size_(hash_size) {
    auto eky = empty_key;
    empty_key_uint =
        *(reinterpret_cast<const TV_METAL_THREAD key_type_uint *>(&eky));
  }

  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(TV_METAL_CONSTANT Args &&...keys) const {
    return hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
  }
#ifdef TV_METAL_CC
  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(TV_METAL_DEVICE Args &&...keys) const {
    return hash_type::hash(
        reinterpret_cast<TV_METAL_DEVICE const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
  }
  template <class... Args>
  TV_DEVICE_INLINE key_type_uint
  hash(TV_METAL_THREADGROUP Args &&...keys) const {
    return hash_type::hash(
        reinterpret_cast<TV_METAL_THREADGROUP const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
  }
  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(TV_METAL_THREAD Args &&...keys) const {
    return hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
  }

#endif
  TV_DEVICE_INLINE int probe_length(int index, key_type key) {
    return (index -
            hash_type::hash_scalar(
                reinterpret_cast<TV_METAL_THREAD key_type_uint &>(key)) %
                hash_size_);
  }

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

#ifndef TV_METAL_RTC
  TV_HOST_DEVICE_INLINE TV_METAL_CONSTANT value_type *data() TV_METAL_CONSTANT {
    return table_;
  }
#endif
  TV_HOST_DEVICE_INLINE TV_METAL_CONSTANT const value_type *
  data() TV_METAL_CONSTANT const {
    return table_;
  }
#ifdef TV_METAL_CC
  TV_HOST_DEVICE_INLINE constexpr thread const value_type *data() const thread {
    return table_;
  }
  TV_HOST_DEVICE_INLINE constexpr thread value_type *data() thread {
    return table_;
  }
  TV_HOST_DEVICE_INLINE constexpr device const value_type *
  data() const threadgroup {
    return table_;
  }
  TV_HOST_DEVICE_INLINE constexpr device value_type *data() threadgroup {
    return table_;
  }
  TV_HOST_DEVICE_INLINE constexpr threadgroup const value_type *
  data() const device {
    return table_;
  }
  TV_HOST_DEVICE_INLINE constexpr threadgroup value_type *data() device {
    return table_;
  }
#endif

private:
  template <class... Args>
  TV_DEVICE_INLINE void insert_raw(TV_METAL_THREAD const V &value,
                                   TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
#ifdef TV_METAL_CC
    // wrong usage (kernel run infinite loop) may cause apple machine
    // hang/restart, so we add a limit here.
    for (int i = 0; i < hash_size_; ++i)
#else
    while (true)
#endif
    {
#ifdef TV_METAL_CC
      bool success =
          detail::AtomicCASDispatch<key_type_uint>::atomic_cas_weak_for_hash(
              reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(
                  &table_[slot].first),
              empty_key_uint, key_u);

      // bool success = atomic_compare_exchange_weak_explicit(
      //     reinterpret_cast<TV_METAL_DEVICE metal::atomic<key_type_uint> *>(
      //         &table_[slot].first),
      //     &empty_key_uint, key_u, metal::memory_order_relaxed,
      //     metal::memory_order_relaxed);
      // auto cur = reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(
      //     &table_[slot].first)[0];
      // success = success || cur == key_u;
#else
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&table_[slot].first),
                    empty_key_uint, key_u);
      bool success = prev == empty_key_uint || prev == key_u;
#endif
      if (success) {
        table_[slot].second = value;
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class F, class KT, int... Inds, class... FArgs>
  TV_DEVICE_INLINE void insert_raw_custom_value_base(
      TV_METAL_THREAD const array<KT, kNumHashArgs> &key_arr,
      mp_list_int<Inds...>, TV_METAL_THREAD F &&f,
      TV_METAL_THREAD FArgs &&...fargs) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<KT> &>(
            key_arr[Inds])...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<KT> &>(
            key_arr[Inds])...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
#ifdef TV_METAL_CC
    // wrong usage (kernel run infinite loop) may cause apple machine
    // hang/restart, so we add a limit here.
    for (int i = 0; i < hash_size_; ++i)
#else
    while (true)
#endif
    {
#ifdef TV_METAL_CC
      bool success =
          detail::AtomicCASDispatch<key_type_uint>::atomic_cas_weak_for_hash(
              reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(
                  &table_[slot].first),
              empty_key_uint, key_u);
#else
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&table_[slot].first),
                    empty_key_uint, key_u);
      bool success = prev == empty_key_uint || prev == key_u;
#endif
      if (success) {
        TV_FORWARD_EXCEPT_METAL(F, f)
        (&table_[slot].second, std::forward<FArgs>(fargs)...);
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <int... Inds>
  TV_DEVICE_INLINE void
  insert_raw(TV_METAL_THREAD const array<K, kNumHashArgs> &key_arr,
             TV_METAL_THREAD const V &value, mp_list_int<Inds...>) {
    return insert_raw(value, key_arr[Inds]...);
  }

  // template <class F, int... Inds>
  // TV_DEVICE_INLINE void
  // insert_raw_custom_value(TV_METAL_THREAD const array<K, kNumHashArgs>
  // &key_arr,
  //                         TV_METAL_THREAD F &&f, mp_list_int<Inds...>) {
  //   return insert_raw_custom_value(TV_FORWARD_EXCEPT_METAL(F, f),
  //                                  key_arr[Inds]...);
  // }

public:
  TV_DEVICE_INLINE void insert(TV_METAL_THREAD const K &key,
                               TV_METAL_THREAD const V &value) {
    static_assert(kNumHashArgs == 1,
                  "you must use tv::array if hash multiple values.");
    return insert_raw(value, key);
  }

  TV_DEVICE_INLINE void
  insert(TV_METAL_THREAD const array<K, kNumHashArgs> &key_arr,
         TV_METAL_THREAD const V &value) {
    return insert_raw(key_arr, value,
                      mp_make_list_c_sequence<int, kNumHashArgs>{});
  }

  template <class F, class... FArgs>
  TV_DEVICE_INLINE void insert_custom_value(TV_METAL_THREAD const K &key,
                                            TV_METAL_THREAD F &&f,
                                            TV_METAL_THREAD FArgs &&...fargs) {
    static_assert(kNumHashArgs == 1,
                  "you must use tv::array if hash multiple values.");
    return insert_raw_custom_value_base(
        tv::array<K, 1>{key}, mp_make_list_c_sequence<int, 1>{},
        TV_FORWARD_EXCEPT_METAL(F, f), std::forward<FArgs>(fargs)...);
  }

  template <class F, class KT, class... FArgs>
  TV_DEVICE_INLINE void
  insert_custom_value(TV_METAL_THREAD const array<KT, kNumHashArgs> &key_arr,
                      TV_METAL_THREAD F &&f, TV_METAL_THREAD FArgs &&...fargs) {
    return insert_raw_custom_value_base(
        key_arr, mp_make_list_c_sequence<int, kNumHashArgs>{},
        TV_FORWARD_EXCEPT_METAL(F, f), std::forward<FArgs>(fargs)...);
  }

  template <class... Args>
  TV_DEVICE_INLINE value_type lookup(TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    const TV_METAL_THREAD value_type *table_ptr = table_;
    while (true) {
      auto val = table_ptr[slot];
      if (reinterpret_cast<TV_METAL_THREAD key_type_uint &>(val.first) ==
          key_u) {
        return val;
      }
      if (val.first == empty_key) {
        return {empty_key, V()};
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return {empty_key, V()};
  }

  template <class... Args>
  TV_DEVICE_INLINE size_type lookup_offset(TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    value_type val;
    while (true) {
      val = table_[slot];
      if (reinterpret_cast<TV_METAL_THREAD key_type_uint &>(val.first) ==
          key_u) {
        return slot;
      }
      if (val.first == empty_key) {
        return -1;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return -1;
  }
};

template <typename K, typename Hash = tv::hash::Murmur3Hash<K>,
          K EmptyKey = default_empty_key_v<K>, bool Power2 = false>
struct LinearHashSet {
public:
  using key_type_uint = to_unsigned_t<K>;
  using value_type = K;
  using size_type = int32_t;
  static constexpr auto TV_METAL_CONSTANT empty_key = EmptyKey;
  using hash_type = Hash;

private:
  TV_METAL_DEVICE value_type *table_ = nullptr;
  size_type hash_size_;
  key_type_uint empty_key_uint;
  static constexpr TV_METAL_CONSTANT auto kNumHashArgs =
      argument_size_v<decltype(hash_type::hash)>;

public:
  TV_HOST_DEVICE_INLINE explicit LinearHashSet(
      TV_METAL_DEVICE value_type *table, size_type hash_size)
      : table_(table), hash_size_(hash_size) {
    auto eky = empty_key;
    empty_key_uint =
        *(reinterpret_cast<TV_METAL_THREAD const key_type_uint *>(&eky));
  }

  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(TV_METAL_THREAD Args &&...keys) const {
    return hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
  }
  TV_DEVICE_INLINE int probe_length(int index, value_type key) {
    return (index -
            hash_type::hash_scalar(
                reinterpret_cast<TV_METAL_THREAD key_type_uint &>(key)) %
                hash_size_);
  }

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

  TV_HOST_DEVICE_INLINE TV_METAL_DEVICE value_type *data() { return table_; }

  TV_HOST_DEVICE_INLINE TV_METAL_DEVICE const value_type *data() const {
    return table_;
  }

  template <class... Args>
  TV_DEVICE_INLINE void insert(TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
#ifdef TV_METAL_CC
    // wrong usage (kernel run infinite loop) may cause apple machine
    // hang/restart, so we add a limit here.
    for (int i = 0; i < hash_size_; ++i)
#else
    while (true)
#endif
    {
#ifdef TV_METAL_CC
      bool success =
          detail::AtomicCASDispatch<key_type_uint>::atomic_cas_weak_for_hash(
              reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(table_ + slot),
              empty_key_uint, key_u);
#else
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(table_ + slot),
                    empty_key_uint, key_u);
      bool success = prev == empty_key_uint || prev == key_u;
#endif
      if (success) {
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class... Args>
  TV_DEVICE_INLINE size_type lookup_offset(TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    value_type val;
    while (true) {
      val = table_[slot];
      if (reinterpret_cast<TV_METAL_THREAD key_type_uint &>(val) == key_u) {
        return slot;
      }
      if (val == empty_key) {
        return -1;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return -1;
  }
};

template <typename K, typename V, typename Hash = tv::hash::Murmur3Hash<K>,
          K EmptyKey = default_empty_key_v<K>, bool Power2 = false>
struct LinearHashTableSplit {
public:
  using key_type_uint = to_unsigned_t<K>;
  using key_type = K;
  // using value_type = pair<K, V, EmptyKey>;
  using mapped_type = V;
  using size_type = int32_t;
  // TODO no way to get constexpr uint repr of key for now.
  // static constexpr auto empty_key_uint = value_type::empty_key_uint;
  static constexpr TV_METAL_CONSTANT auto empty_key = EmptyKey;
  using hash_type = Hash;

private:
  TV_METAL_DEVICE key_type *key_ptr_ = nullptr;
  TV_METAL_DEVICE mapped_type *value_ptr_ = nullptr;

  size_type hash_size_;
  key_type_uint empty_key_uint;
  static constexpr TV_METAL_CONSTANT auto kNumHashArgs =
      argument_size_v<decltype(hash_type::hash)>;

public:
  TV_HOST_DEVICE explicit LinearHashTableSplit(
      TV_METAL_DEVICE key_type *key, TV_METAL_DEVICE mapped_type *value,
      size_type hash_size)
      : key_ptr_(key), value_ptr_(value), hash_size_(hash_size) {
    auto eky = empty_key;
    empty_key_uint =
        *(reinterpret_cast<TV_METAL_THREAD const key_type_uint *>(&eky));
  }

  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(TV_METAL_THREAD Args &&...keys) const {
    return hash_type::hash(keys...);
  }

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

  TV_HOST_DEVICE_INLINE TV_METAL_DEVICE key_type *key_ptr() { return key_ptr_; }
  TV_HOST_DEVICE_INLINE TV_METAL_DEVICE const key_type *key_ptr() const {
    return key_ptr_;
  }
  TV_HOST_DEVICE_INLINE TV_METAL_DEVICE mapped_type *value_ptr() {
    return value_ptr_;
  }
  TV_HOST_DEVICE_INLINE TV_METAL_DEVICE const mapped_type *value_ptr() const {
    return value_ptr_;
  }

private:
  template <class... Args>
  TV_DEVICE_INLINE void insert_raw(TV_METAL_THREAD const V &value,
                                   TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
#ifdef TV_METAL_CC
    // wrong usage (kernel run infinite loop) may cause apple machine
    // hang/restart, so we add a limit here.
    for (int i = 0; i < hash_size_; ++i)
#else
    while (true)
#endif
    {
#ifdef TV_METAL_CC
      // bool success =
      //     atomic_compare_exchange_weak_explicit(reinterpret_cast<TV_METAL_DEVICE
      //     metal::atomic<key_type_uint> *>(key_ptr_ + slot),
      //               &empty_key_uint, key_u, metal::memory_order_relaxed,
      //               metal::memory_order_relaxed);
      // auto cur = reinterpret_cast<TV_METAL_DEVICE
      // key_type_uint*>(key_ptr_)[slot]; success = success || cur == key_u;
      bool success =
          detail::AtomicCASDispatch<key_type_uint>::atomic_cas_weak_for_hash(
              reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(key_ptr_ +
                                                                slot),
              empty_key_uint, key_u);
#else
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&key_ptr_[slot]),
                    empty_key_uint, key_u);
      bool success = prev == empty_key_uint || prev == key_u;
#endif
      if (success) {
        value_ptr_[slot] = value;
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class F, class KT, int... Inds, class... FArgs>
  TV_DEVICE_INLINE void insert_raw_custom_value_base(
      TV_METAL_THREAD const array<KT, kNumHashArgs> &key_arr,
      mp_list_int<Inds...>, TV_METAL_THREAD F &&f,
      TV_METAL_THREAD FArgs &&...fargs) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<KT> &>(
            key_arr[Inds])...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<KT> &>(
            key_arr[Inds])...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
#ifdef TV_METAL_CC
    // wrong usage (kernel run infinite loop) may cause apple machine
    // hang/restart, so we add a limit here.
    for (int i = 0; i < hash_size_; ++i)
#else
    while (true)
#endif
    {
#ifdef TV_METAL_CC
      bool success =
          detail::AtomicCASDispatch<key_type_uint>::atomic_cas_weak_for_hash(
              reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(key_ptr_ +
                                                                slot),
              empty_key_uint, key_u);
#else
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&key_ptr_[slot]),
                    empty_key_uint, key_u);
      bool success = prev == empty_key_uint || prev == key_u;
#endif
      if (success) {
        TV_FORWARD_EXCEPT_METAL(F, f)
        (value_ptr_ + slot, std::forward<FArgs>(fargs)...);
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <int... Inds>
  TV_DEVICE_INLINE void
  insert_raw(TV_METAL_THREAD const array<K, kNumHashArgs> &key_arr,
             TV_METAL_THREAD const V &value, mp_list_int<Inds...>) {
    insert_raw(value, key_arr[Inds]...);
  }

public:
  TV_DEVICE_INLINE void insert(TV_METAL_THREAD const K &key,
                               TV_METAL_THREAD const V &value) {
    static_assert(kNumHashArgs == 1,
                  "you must use tv::array if hash multiple values.");
    return insert_raw(value, key);
  }

  TV_DEVICE_INLINE void
  insert(TV_METAL_THREAD const array<K, kNumHashArgs> &key_arr,
         TV_METAL_THREAD const V &value) {
    return insert_raw(key_arr, value,
                      mp_make_list_c_sequence<int, kNumHashArgs>{});
  }

  template <class F, class... FArgs>
  TV_DEVICE_INLINE void insert_custom_value(TV_METAL_THREAD const K &key,
                                            TV_METAL_THREAD F &&f,
                                            TV_METAL_THREAD FArgs &&...fargs) {
    static_assert(kNumHashArgs == 1,
                  "you must use tv::array if hash multiple values.");
    return insert_raw_custom_value_base(
        tv::array<K, 1>{key}, mp_make_list_c_sequence<int, 1>{},
        TV_FORWARD_EXCEPT_METAL(F, f), std::forward<FArgs>(fargs)...);
  }

  template <class F, class KT, class... FArgs>
  TV_DEVICE_INLINE void
  insert_custom_value(TV_METAL_THREAD const array<KT, kNumHashArgs> &key_arr,
                      TV_METAL_THREAD F &&f, TV_METAL_THREAD FArgs &&...fargs) {
    return insert_raw_custom_value_base(
        key_arr, mp_make_list_c_sequence<int, kNumHashArgs>{},
        TV_FORWARD_EXCEPT_METAL(F, f), std::forward<FArgs>(fargs)...);
  }

  TV_DEVICE_INLINE int probe_length(int index, key_type key) {
    return (index -
            hash_type::hash_scalar(
                reinterpret_cast<TV_METAL_THREAD key_type_uint &>(key)) %
                hash_size_);
  }

  template <class... Args>
  TV_DEVICE_INLINE void insert_key_only(TV_METAL_THREAD Args &&...keys) {
    const key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
#ifdef TV_METAL_CC
    // wrong usage (kernel run infinite loop) may cause apple machine
    // hang/restart, so we add a limit here.
    for (int i = 0; i < hash_size_; ++i)
#else
    while (true)
#endif
    {
#ifdef TV_METAL_CC
      bool success =
          detail::AtomicCASDispatch<key_type_uint>::atomic_cas_weak_for_hash(
              reinterpret_cast<TV_METAL_DEVICE key_type_uint *>(key_ptr_ +
                                                                slot),
              empty_key_uint, key_u);
#else
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&key_ptr_[slot]),
                    empty_key_uint, key_u);
      bool success = prev == empty_key_uint || prev == key_u;
#endif
      if (success) {
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class... Args>
  TV_DEVICE_INLINE size_type lookup_offset(TV_METAL_THREAD Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(
        reinterpret_cast<TV_METAL_THREAD const to_unsigned_t<Args> &>(
            std::forward<Args>(keys))...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      auto key_target = key_ptr_[slot];
      if (reinterpret_cast<TV_METAL_THREAD key_type_uint &>(key_target) ==
          key_u) {
        return slot;
      }
      if (key_target == empty_key) {
        return -1;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return -1;
  }
};

} // namespace hash
} // namespace tv