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
#include "tensor.h"
/*
check_eq_dtype(a, b, c)
check_eq_shape(a, b, c)
check_eq_device(a, b, c)

check_shape(a, {-1, 2})
*/

namespace tv {


inline void check_shape(tv::Tensor ten, tv::TensorShape shape){
    TV_ASSERT_INVALID_ARG(ten.ndim() == shape.ndim(), "error ndim", ten.ndim(), "expect", shape.ndim());
    auto& shape_ten = ten.shape();
    for (int i = 0; i < shape.ndim(); ++i){
        if (shape[i] != -1){
            TV_ASSERT_INVALID_ARG(shape_ten[i] == shape[i], "error shape", shape_ten, "expect", shape);
        }
    }
}

template <class ...Ts>
void check_eq_dtype(Ts... tens){
    std::vector<mp_nth_t<0, Ts...>> ten_vecs{tens...};
    for (int i = 0; i < sizeof...(tens); ++i){
        TV_ASSERT_INVALID_ARG(ten_vecs[i].dtype() == ten_vecs[0].dtype(), "dtype mismatch", 
            tv::dtype_str(ten_vecs[i].dtype()), tv::dtype_str(ten_vecs[0].dtype()));
    }
}

template <class ...Ts>
void check_eq_shape(Ts... tens){
    std::vector<mp_nth_t<0, Ts...>> ten_vecs{tens...};
    for (int i = 0; i < sizeof...(tens); ++i){
        TV_ASSERT_INVALID_ARG(ten_vecs[i].shape() == ten_vecs[0].shape(), "shape mismatch", 
            ten_vecs[i].shape(), ten_vecs[0].shape());
    }
}

template <class ...Ts>
void check_eq_device(Ts... tens){
    std::vector<mp_nth_t<0, Ts...>> ten_vecs{tens...};
    for (int i = 0; i < sizeof...(tens); ++i){
        TV_ASSERT_INVALID_ARG(ten_vecs[i].device() == ten_vecs[0].device(), "device mismatch", 
            ten_vecs[i].device(), ten_vecs[0].device());
    }
}

}