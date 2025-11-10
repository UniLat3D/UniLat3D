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

#include "array.h"

namespace tv {
namespace detail {
#ifndef TV_PARALLEL_RTC
template <typename _InIter>
using _RequireInputIter = typename std::enable_if<std::is_convertible<
    typename std::iterator_traits<_InIter>::iterator_category,
    std::input_iterator_tag>::value>::type;
#endif
} // namespace detail
template <typename T, size_t N> struct vecarray : public array<T, N> {
  typedef T value_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *iterator;
  typedef const value_type *const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
#ifndef TV_PARALLEL_RTC
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#endif
public:
  TV_HOST_DEVICE_INLINE vecarray(){};
  TV_HOST_DEVICE_INLINE constexpr vecarray(size_t count, T init = T())
      : array<T, N>(), size_(count) {
    for (size_t i = 0; i < count; ++i) {
      this->array_[i] = init;
    }
  };
  constexpr TV_HOST_DEVICE vecarray(std::initializer_list<T> arr)
      : array<T, N>(), size_(std::min(N, arr.size())) {
    for (auto p = detail::make_init_pair(0, arr.begin());
         p.first < N && p.second != arr.end(); ++p.first, ++p.second) {
      this->array_[p.first] = *(p.second);
    }
  }
#ifndef TV_PARALLEL_RTC
  template <typename Iterator, typename = detail::_RequireInputIter<Iterator>>
  vecarray(Iterator first, Iterator last) {
    size_ = 0;
    for (; first != last; ++first) {
      if (size_ >= N) {
        continue;
      }
      this->array_[size_++] = *first;
    }
  };

  vecarray(const std::vector<T> &arr) {
    TV_ASSERT(arr.size() <= N);
    for (size_t i = 0; i < arr.size(); ++i) {
      this->array_[i] = arr[i];
    }
    size_ = arr.size();
  }
#endif
#ifdef TV_DEBUG
  TV_HOST_DEVICE_INLINE T &operator[](int idx) {
    TV_ASSERT(idx >= 0 && idx < size_);
    return this->array_[idx];
  }
  TV_HOST_DEVICE_INLINE const T &operator[](int idx) const {
    TV_ASSERT(idx >= 0 && idx < size_);
    return this->array_[idx];
  }
#else
  TV_HOST_DEVICE_INLINE constexpr T &operator[](int idx) {
    return this->array_[idx];
  }
  TV_HOST_DEVICE_INLINE constexpr const T &operator[](int idx) const {
    return this->array_[idx];
  }
#endif

  TV_HOST_DEVICE_INLINE void push_back(T s) {
#ifdef TV_DEBUG
    TV_ASSERT(size_ < N);
#endif
    this->array_[size_++] = s;
  }
  TV_HOST_DEVICE_INLINE void pop_back() {
#ifdef TV_DEBUG
    TV_ASSERT(size_ > 0);
#endif
    size_--;
  }

  TV_HOST_DEVICE_INLINE size_t size() const { return size_; }
  TV_HOST_DEVICE_INLINE constexpr size_t max_size() const { return N; }

  TV_HOST_DEVICE_INLINE const T *data() const { return this->array_; }
  TV_HOST_DEVICE_INLINE T *data() { return this->array_; }
  TV_HOST_DEVICE_INLINE size_t empty() const { return size_ == 0; }

  TV_HOST_DEVICE_INLINE iterator begin() { return iterator(this->array_); }
  TV_HOST_DEVICE_INLINE iterator end() {
    return iterator(this->array_ + size_);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator begin() const {
    return const_iterator(this->array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator end() const {
    return const_iterator(this->array_ + size_);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator cbegin() const {
    return const_iterator(this->array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator cend() const {
    return const_iterator(this->array_ + size_);
  }
#ifndef TV_PARALLEL_RTC
  constexpr const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  constexpr reverse_iterator rbegin() noexcept {
    return reverse_iterator(end());
  }

  constexpr const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr reverse_iterator rend() noexcept {
    return reverse_iterator(begin());
  }

  constexpr const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }

  iterator erase(const_iterator CIit) {
    // Just cast away constness because this is a non-const member function.
    iterator Iit = const_cast<iterator>(CIit);

    assert(Iit >= this->begin() && "Iterator to erase is out of bounds.");
    assert(Iit < this->end() && "Erasing at past-the-end iterator.");

    iterator Nit = Iit;
    // Shift all elts down one.
    std::move(Iit + 1, this->end(), Iit);
    // Drop the last elt.
    this->pop_back();
    return (Nit);
  }

#endif
  TV_HOST_DEVICE_INLINE constexpr reference front() noexcept {
    return *begin();
  }

  TV_HOST_DEVICE_INLINE constexpr const_reference front() const noexcept {
    return this->array_[0];
  }

  TV_HOST_DEVICE_INLINE reference back() noexcept {
    return size_ ? *(end() - 1) : *end();
  }

  TV_HOST_DEVICE_INLINE const_reference back() const noexcept {
    return size_ ? this->array_[size_ - 1] : this->array_[0];
  }

protected:
  size_t size_ = 0;
};

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE bool operator==(const vecarray<T, N> &lfs,
                                      const vecarray<T, N> &rfs) {
  if (lfs.size() != rfs.size())
    return false;
  for (size_t i = 0; i < lfs.size(); ++i) {
    if (lfs[i] != rfs[i])
      return false;
  }
  return true;
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE bool operator!=(const vecarray<T, N> &lfs,
                                      const vecarray<T, N> &rfs) {

  return !(lfs == rfs);
}

} // namespace tv