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

#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuda_runtime_api.h>
#endif
#include <chrono>
#include <memory>
#include <string>
#include <tensorview/tensor.h>
#include <unordered_map>
#ifdef TV_USE_LIBRT
#include <time.h>
#endif

namespace tv {


#ifdef TV_USE_LIBRT
#define SEC_TO_MS 1000

/* To microseconds (10^-6) */
#define MS_TO_US 1000
#define SEC_TO_US (SEC_TO_MS * MS_TO_US)

/* To nanoseconds (10^-9) */
#define US_TO_NS 1000
#define MS_TO_NS (MS_TO_US * US_TO_NS)
#define SEC_TO_NS (SEC_TO_MS * MS_TO_NS)

/* Conversion from nanoseconds */
#define NS_TO_MS (1000 * 1000)
#define NS_TO_US (1000)

#define _PyTime_MIN INT64_MIN
#define _PyTime_MAX INT64_MAX

#define CODEAI_EXPORT
#define CODEAI_INIT_EXPORT

using _PyTime_t = int64_t;
#define _PyTime_check_mul_overflow(a, b)                                       \
  (assert(b > 0), (_PyTime_t)(a) < _PyTime_MIN / (_PyTime_t)(b) ||             \
                      _PyTime_MAX / (_PyTime_t)(b) < (_PyTime_t)(a))

inline int pytime_fromtimespec(_PyTime_t &tp, const timespec &ts) {
  int res = 0;
  _PyTime_t t = ts.tv_sec;
  _PyTime_t nsec;
  if (_PyTime_check_mul_overflow(t, SEC_TO_NS)) {
    res = -1;
    t = (t > 0) ? _PyTime_MAX : _PyTime_MIN;
  } else {
    t = t * SEC_TO_NS;
  }

  nsec = ts.tv_nsec;
  /* The following test is written for positive only nsec */
  assert(nsec >= 0);
  if (t > _PyTime_MAX - nsec) {
    res = -1;
    t = _PyTime_MAX;
  } else {
    t += nsec;
  }
  tp = t;
  return res;
}

inline int pygettimeofday(_PyTime_t &tp) {
  // adapted from cpython pytime.c
  struct timespec ts;
  auto err = clock_gettime(CLOCK_REALTIME, &ts);
  if (err) {
    return -1;
  }
  return pytime_fromtimespec(tp, ts);
}


#endif

class CPUEvent {
private:
#ifdef TV_USE_LIBRT
  _PyTime_t cur_time_;
#else 
  std::chrono::time_point<std::chrono::steady_clock> cur_time_;
#endif

public:
  std::string name;

  CPUEvent(std::string name = "") : name(name) {}

  CPUEvent& record(std::uintptr_t stream = 0) {
#ifdef TV_USE_LIBRT
    pygettimeofday(cur_time_);
#else 
    cur_time_ = std::chrono::steady_clock::now();
#endif
    return *this;
  }

  CPUEvent& stream_wait_me(std::uintptr_t stream, int flag = 0) {
    return *this;
  }

  void sync() {}

  static float duration(CPUEvent start, CPUEvent stop) {
#ifdef TV_USE_LIBRT
    float ms = (stop.cur_time_ - start.cur_time_) /
               1000.0f;

#else 
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(
                   stop.cur_time_ - start.cur_time_)
                   .count() /
               1000.0f;
#endif
    return ms;
  }

  static float sync_and_duration(CPUEvent start, CPUEvent stop) {
    return duration(start, stop);
  }

};

#if defined(TV_HARDWARE_ACC_CUDA)

class CUDAEventCore {
private:
  cudaEvent_t event_ = 0;

public:
  CUDAEventCore() { checkCudaErrors(cudaEventCreate(&event_)); }

  ~CUDAEventCore() {
    if (event_) {
      { cudaEventDestroy(event_); }
    }
  }

  void record(cudaStream_t stream = nullptr) {
    checkCudaErrors(cudaEventRecord(event_, stream));
  }

  void stream_wait_me(cudaStream_t stream, int flag = 0) {
    checkCudaErrors(cudaStreamWaitEvent(stream, event_, flag));
  }

  void sync() {
    checkCudaErrors(cudaEventSynchronize(event_));
  }

  cudaEvent_t get_event() { return event_; }
};

class CUDAEvent {
private:
  std::shared_ptr<CUDAEventCore> event_ = 0;

public:
  std::string name;

  CUDAEvent(std::string name = "")
      : event_(std::make_shared<CUDAEventCore>()), name(name) {}

  CUDAEvent& record(std::uintptr_t stream = 0) {
    TV_ASSERT_RT_ERR(event_, "event is empty");
    event_->record(reinterpret_cast<cudaStream_t>(stream));
    return *this;
  }

  CUDAEvent& stream_wait_me(std::uintptr_t stream, int flag = 0) {
    TV_ASSERT_RT_ERR(event_, "event is empty");
    event_->stream_wait_me(reinterpret_cast<cudaStream_t>(stream), flag);
    return *this;
  }

  void sync() {
    TV_ASSERT_RT_ERR(event_, "event is empty");
    event_->sync();
  }

  static float duration(CUDAEvent start, CUDAEvent stop) {
    float ms;
    TV_ASSERT_RT_ERR(start.event_, "event is empty");
    TV_ASSERT_RT_ERR(stop.event_, "event is empty");
    checkCudaErrors(cudaEventElapsedTime(&ms, start.event_->get_event(),
                                         stop.event_->get_event()));
    return ms;
  }

  static float sync_and_duration(CUDAEvent start, CUDAEvent stop) {
    start.sync();
    stop.sync();
    return duration(start, stop);
  }
};
#else

class CUDAEvent: public CPUEvent {
public:
  CUDAEvent(std::string name = "") : CPUEvent(name) {}
};
#endif

class CUDAKernelTimerCore {
private:
  std::vector<std::string> namespaces_;
  std::unordered_map<std::string, CUDAEvent> name_to_event_;
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      name_to_pair_;
  std::vector<std::string> names_;

public:
  void push(std::string name) noexcept { namespaces_.push_back(name); }
  void pop() {
    TV_ASSERT_RT_ERR(!namespaces_.empty(),
                     "you pop value from empty namespaces");
    namespaces_.pop_back();
  }

  std::string get_current_namespace() {
    if (namespaces_.empty()) {
      { return ""; }
    }
    std::string res = namespaces_[0];
    for (int i = 1; i < namespaces_.size(); ++i) {
      { res += "." + namespaces_[i]; }
    }
    return res;
  }

  std::string add_namespace_to_name(std::string name) {
    auto unique_name = name;
    auto cur_ns = get_current_namespace();
    if (name.empty()) {
      { return cur_ns; }
    }
    if (cur_ns.size() > 0) {
      { unique_name = cur_ns + "." + name; }
    }
    return unique_name;
  }

  void record(std::string name, std::uintptr_t stream = 0) {
    auto unique_name = add_namespace_to_name(name);
    TV_ASSERT_RT_ERR(name_to_event_.find(unique_name) == name_to_event_.end(),
                     "your name", unique_name, "already exists");
    CUDAEvent newev(unique_name);
    newev.record(stream);
    name_to_event_.insert({{unique_name, newev}});
    names_.push_back(unique_name);
  }

  bool has_pair(std::string name) {
    name = add_namespace_to_name(name);
    return name_to_pair_.find(name) != name_to_pair_.end();
  }

  void sync_all_event() {
    for (auto &p : name_to_event_) {
      { p.second.sync(); }
    }
  }

  std::string insert_pair(std::string name, std::string start, std::string stop) {
    name = add_namespace_to_name(name);
    start = add_namespace_to_name(start);
    stop = add_namespace_to_name(stop);
    TV_ASSERT_RT_ERR(name_to_pair_.find(name) == name_to_pair_.end(),
                     "your name", name, "already exists");
    name_to_pair_[name] = {start, stop};
    return name;
  }

  std::unordered_map<std::string, float> get_all_pair_duration() {
    std::unordered_map<std::string, float> res;
    sync_all_event();
    for (auto &p : name_to_pair_) {
      {
        auto &ev_start = name_to_event_.at(p.second.first);
        auto &ev_stop = name_to_event_.at(p.second.second);
        res[p.first] = CUDAEvent::duration(ev_start, ev_stop);
      }
    }
    return res;
  }

  float get_pair_duration(std::string name) {
    TV_ASSERT_RT_ERR(name_to_pair_.find(name) != name_to_pair_.end(), "can't find your pair", name);
    auto& p = name_to_pair_.at(name);
    auto &ev_start = name_to_event_.at(p.first);
    auto &ev_stop = name_to_event_.at(p.second);
    ev_start.sync();
    ev_stop.sync();
    return CUDAEvent::duration(ev_start, ev_stop);
  }

};

class CUDAKernelTimer {
private:
  std::shared_ptr<CUDAKernelTimerCore> timer_ptr_;
  bool enable_;

public:
  CUDAKernelTimer(bool enable)
      : timer_ptr_(std::make_shared<CUDAKernelTimerCore>()), enable_(enable) {}

  void push(std::string name) {
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->push(name);
      }
    }
  }
  bool enable() { return enable_; }

  void pop() {
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->pop();
      }
    }
  }

  void record(std::string name, std::uintptr_t stream = 0) {
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->record(name, stream);
      }
    }
  }
  std::string insert_pair(std::string name, std::string start, std::string stop) {
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        return timer_ptr_->insert_pair(name, start, stop);
      }
    }else{
      return "";
    }
  }

  bool has_pair(std::string name) {
    if (!enable_) {
      return false;
    }
    TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
    return timer_ptr_->has_pair(name);
  }

  void sync_all_event() {
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->sync_all_event();
      }
    }
  }

  std::unordered_map<std::string, float> get_all_pair_duration() {
    std::unordered_map<std::string, float> res;
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        return timer_ptr_->get_all_pair_duration();
      }
    }
    return res;
  }

  float get_pair_duration(std::string name) {
    if (enable_) {
      {
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        return timer_ptr_->get_pair_duration(name);
      }
    }else{
      return -1;
    }
  }

};

class CUDAKernelTimerGuard {
private:
  std::string name_;
  CUDAKernelTimer timer_;
  std::uintptr_t stream_ = 0;
  bool print_exit_;

  std::string pair_name_;

public:
  CUDAKernelTimerGuard(std::string name, CUDAKernelTimer timer,
                       std::uintptr_t stream = 0, bool print_exit = false)
      : name_(name), timer_(timer), stream_(stream), print_exit_(print_exit) {
    if (timer_.enable()) {
      {
        if (!name_.empty()) {
          timer_.push(name_);
        }
        pair_name_ = timer_.insert_pair("", "start", "stop");
        timer_.record("start", stream_);
      }
    }
  }

#if defined(TV_HARDWARE_ACC_CUDA)
  CUDAKernelTimerGuard(std::string name, CUDAKernelTimer timer,
                       cudaStream_t stream = nullptr, bool print_exit = false): CUDAKernelTimerGuard(name, timer, reinterpret_cast<std::uintptr_t>(stream), print_exit){}
#endif

  ~CUDAKernelTimerGuard() {
    if (timer_.enable()) {
      {
        timer_.record("stop", stream_);
        if (!name_.empty()) {
          timer_.pop();
        }
        if (print_exit_){
          tv::ssprint(pair_name_, ":", timer_.get_pair_duration(pair_name_));
        }
      }
    }
  }
};

#if defined(TV_HARDWARE_ACC_CUDA)
inline auto measure_and_print_guard(std::string name, cudaStream_t stream = nullptr){
  return std::make_shared<CUDAKernelTimerGuard>(name, CUDAKernelTimer(true), stream, true);
};
#endif
inline auto measure_and_print_guard(std::string name, std::uintptr_t stream = 0){
  return std::make_shared<CUDAKernelTimerGuard>(name, CUDAKernelTimer(true), stream, true);
};

inline auto measure_guard(std::string name, std::uintptr_t stream = 0){
  return std::make_shared<CUDAKernelTimerGuard>(name, CUDAKernelTimer(true), stream, true);
};


} // namespace tv
