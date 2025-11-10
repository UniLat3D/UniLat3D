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
JsonArray Design

meta_start_offset-meta_end_offset-tensors-meta(json)

meta layout:

{
  user_meta: ...
  tensor_offsets: ...
}

we can access single tensor without read all data of a json array.



*/

#pragma once
#include <tuple>
#include <vector>

#include <string>
#include <tensorview/tensor.h>
#include <tensorview/thirdparty/nlohmann/json.hpp>
#include <unordered_map>
#include <fstream>

namespace tv {
namespace io {
using json = nlohmann::json;
constexpr const char *kJsonArrayKey = "__cumm_io_json_index";

inline json access_idx(json j) {
  return j[kJsonArrayKey].template get<int64_t>();
}

inline json json_idx(size_t i) {
  json res;
  res[kJsonArrayKey] = i;
  return res;
}

struct JsonArrayProxy {
  json& data;
  std::vector<tv::Tensor>& tensors;
  void assign(std::string name, tv::Tensor ten){
    data[name] = json_idx(tensors.size());
    tensors.push_back(ten);
  }

  void assign(std::string name, std::vector<tv::Tensor> tens){
    std::vector<json> is_idxes;
    for (auto& t : tens){
      is_idxes.push_back(json_idx(tensors.size()));
      tensors.push_back(t);
    }
    data[name] = is_idxes;

  }

  void assign(std::string name, std::unordered_map<std::string, tv::Tensor> ten_map){
    std::unordered_map<std::string, json> is_idxes;
    for (auto& p : ten_map){
      is_idxes[p.first] = json_idx(tensors.size());
      tensors.push_back(p.second);
    }
    data[name] = is_idxes;
  }

  JsonArrayProxy operator[](std::string key){
    return JsonArrayProxy{data[key], tensors};
  }

  JsonArrayProxy operator[](int64_t key){
    return JsonArrayProxy{data[key], tensors};
  }

  std::vector<tv::Tensor> get_vector_of_tensor(){
    std::vector<tv::Tensor> vec;
    for (auto it = data.begin(); it != data.end(); ++it){
      vec.push_back(tensors[access_idx(*it)]);
    }
    return vec;
  }
  tv::Tensor get_tensor(){
    return tensors[access_idx(data)];
  }
  std::unordered_map<std::string, tv::Tensor> get_map_of_tensor(){
    std::unordered_map<std::string, tv::Tensor> res;
    for (auto it = data.begin(); it != data.end(); ++it){
      res.insert({it.key(), tensors[access_idx(it.value())]});
    }
    return res;
  }

};

struct JsonArray {
  json data;
  std::vector<tv::Tensor> tensors;

  void assign(std::string name, tv::Tensor ten){
    data[name] = json_idx(tensors.size());
    tensors.push_back(ten);
  }

  void assign(std::string name, std::vector<tv::Tensor> tens){
    std::vector<json> is_idxes;
    for (auto& t : tens){
      is_idxes.push_back(json_idx(tensors.size()));
      tensors.push_back(t);
    }
    data[name] = is_idxes;
  }

  void assign(std::string name, std::unordered_map<std::string, tv::Tensor> ten_map){
    std::unordered_map<std::string, json> is_idxes;
    for (auto& p : ten_map){
      is_idxes[p.first] = json_idx(tensors.size());
      tensors.push_back(p.second);
    }
    data[name] = is_idxes;
  }

  JsonArrayProxy operator[](std::string key){
    return JsonArrayProxy{data[key], tensors};
  }

  JsonArrayProxy operator[](int64_t key){
    return JsonArrayProxy{data[key], tensors};
  }
  
  std::vector<tv::Tensor> get_vector_of_tensor(){
    std::vector<tv::Tensor> vec;
    for (auto it = data.begin(); it != data.end(); ++it){
      vec.push_back(tensors[access_idx(*it)]);
    }
    return vec;
  }
  tv::Tensor get_tensor(){
    return tensors[access_idx(data)];
  }
  std::unordered_map<std::string, tv::Tensor> get_map_of_tensor(){
    std::unordered_map<std::string, tv::Tensor> res;
    for (auto it = data.begin(); it != data.end(); ++it){
      res.insert({it.key(), tensors[access_idx(it.value())]});
    }
    return res;
  }


};


namespace detail {
template <typename K, typename V, typename Hash>
std::unordered_map<V, K, Hash>
inverse_map(const std::unordered_map<K, V, Hash> &dict) {
  std::unordered_map<V, K, Hash> res;
  for (auto &p : dict) {
    res[p.second] = p.first;
  }
  return res;
}

inline int64_t align_offset(int64_t offset, int64_t n) {
  if (n <= 0) {
    return offset;
  }
  return n * ((offset + n - 1) / n);
}

} // namespace detail

inline JsonArray decode(const uint8_t* buffer_ptr, size_t size) {
  JsonArray res;
  if (size < 16) {
    TV_THROW_RT_ERR("buffer length invalid");
  }
  int64_t content_end = *(reinterpret_cast<const int64_t *>(buffer_ptr));
  int64_t meta_end = *(reinterpret_cast<const int64_t *>(buffer_ptr + 8));
  TV_ASSERT_RT_ERR(content_end < (1 << 30) && meta_end < (1 << 30),
                   "must small than 1GB");
  TV_ASSERT_RT_ERR(content_end < meta_end, "error");

  auto meta =
      json::parse(buffer_ptr + content_end, buffer_ptr + meta_end);
  auto array_metas = meta["array"];
  auto data_skeleton = meta["data"];
  std::vector<tv::Tensor> res_tensors;
  for (auto it = array_metas.begin(); it < array_metas.end(); ++it) {
    auto &array_meta = (*it);
    auto shape = array_meta["shape"].template get<std::vector<int64_t>>();
    auto dtype = array_meta["dtype"].template get<int64_t>();
    auto offset = array_meta["offset"].template get<int64_t>();
    TV_ASSERT_RT_ERR(shape.size() <= tv::kTensorMaxDim, "error");
    auto tv_dtype = tv::DType(dtype);
    tv::TensorShape shape_tensor;
    for (auto &c : shape) {
      shape_tensor.push_back(c);
    }
    tv::Tensor tensor;
    if (shape_tensor.size() > 0){
      tensor =
          tv::from_blob(buffer_ptr + offset, shape_tensor, tv_dtype, -1);
    }
    res_tensors.push_back(tensor.clone());
  }
  return {data_skeleton, res_tensors};
}


inline JsonArray decode(const std::vector<uint8_t> &buffer) {
  return decode(buffer.data(), buffer.size());
}


inline std::vector<uint8_t> encode(const std::vector<tv::Tensor> tensors,
                          const json &data_json) {
  json res_json;
  int64_t start = 16;
  constexpr int64_t align_size = 128;
  std::vector<int64_t> offsets;
  res_json["array"] = json::array();
  for (size_t i = 0; i < tensors.size(); ++i) {
    json array_meta;
    auto &tensor = tensors[i];
    auto &shape = tensor.shape();

    int64_t start_aligned = detail::align_offset(start, align_size);
    array_meta["shape"] = std::vector<int64_t>(shape.begin(), shape.end());
    array_meta["dtype"] = int(tensor.dtype());
    array_meta["offset"] = start_aligned;
    array_meta["is_np"] = false;
    start = start_aligned + tensor.nbytes();
    res_json["array"].push_back(array_meta);
    offsets.push_back(start_aligned);
  }
  res_json["data"] = data_json;
  std::string meta_json = res_json.dump();
  size_t total_length = start + meta_json.size();
  std::vector<uint8_t> res;
  res.resize(total_length);

  auto res_data = &res[0];
  *(reinterpret_cast<int64_t *>(res_data)) = start;
  *(reinterpret_cast<int64_t *>(res_data + 8)) = total_length;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &tensor = tensors[i];
    auto offset = offsets[i];
    if (!tensor.is_cpu()){
      auto ten_cpu = tensor.cpu();
      std::copy(ten_cpu.raw_data(), ten_cpu.raw_data() + ten_cpu.nbytes(),
                reinterpret_cast<uint8_t *>(res_data + offset));
    }else{
      std::copy(tensor.raw_data(), tensor.raw_data() + tensor.nbytes(),
                reinterpret_cast<uint8_t *>(res_data + offset));
    }
  }
  std::copy(meta_json.begin(), meta_json.end(), res_data + start);
  return res;
}

inline std::vector<uint8_t> encode(const JsonArray& jarr) {
  return encode(jarr.tensors, jarr.data);
}

inline void dump_to_file(std::string path, const JsonArray& jarr){
  std::ofstream file;
  file.open(path, std::ios::out | std::ios::binary);
  auto buffer = encode(jarr);
  file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
}

inline JsonArray load_from_file(std::string path){
  std::ifstream input( path, std::ios::binary );
  std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(input), {});
  return decode(buffer);
}



} // namespace io
} // namespace tv