# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import read
from typing import List, Optional

import pccm

from cumm.common import TensorView, TensorViewCPU
from cumm.tensorview_bind import TensorViewBind


class ArrayPtr(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        # here we add TensorViewBind as dep to ensure tv::Tensor is binded before arrayptr.

        self.add_dependency(TensorViewCPU, TensorViewBind)
        self.add_include("unordered_map")
        self.add_pybind_member("dtype_", "tv::DType", readwrite=False)
        self.add_pybind_member("length_", "int64_t", readwrite=False)
        self.add_pybind_member("access_byte_size_", "int", readwrite=False)
        self.add_pybind_member("byte_offset_", "int64_t", readwrite=False)
        self.add_pybind_member("align_", "int", readwrite=False)
        self.add_pybind_member("itemsize_", "int", readwrite=False)
        self.add_pybind_member("data_", "tv::Tensor", readwrite=False)
        # the meta data will be copied during getitem/setitem.
        # suitable for low precision dtypes.
        self.add_pybind_member("meta_data_",
                               "tv::Tensor",
                               pyanno="cumm.tensorview.Tensor",
                               readwrite=False)

        # self.add_pybind_member("num_smem_bank_", "int", readwrite=False)
        # self.add_pybind_member("bank_byte_size_", "int", readwrite=False)

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("dtype", "int")
        code.arg("length", "int64_t")
        code.arg("access_byte_size", "int", "-1")
        code.arg("byte_offset", "int64_t", "0")
        code.arg("align", "int", "-1")
        code.arg("external_data",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("meta_data",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")

        # code.arg("num_smem_bank", "int", "32")
        # code.arg("bank_byte_size", "int", "4")

        code.ctor_init("length_", "length")
        code.ctor_init("dtype_", "tv::DType(dtype)")
        code.ctor_init("itemsize_",
                       "tv::detail::sizeof_dtype(tv::DType(dtype))")
        code.ctor_init("byte_offset_", "byte_offset")
        # code.ctor_init("num_smem_bank_", "num_smem_bank")
        # code.ctor_init("bank_byte_size_", "bank_byte_size")

        code.raw(f"""
        TV_ASSERT_INVALID_ARG(byte_offset >= 0 && byte_offset < length * itemsize_, "invalid byte_offset", byte_offset);
        if (external_data.empty()){{
            data_ = tv::empty({{length}}, dtype_, -1);
        }}else{{
            TV_ASSERT_INVALID_ARG(
                external_data.itemsize() == itemsize_, "error");
            data_ = external_data.view(-1); 
        }}
        if (!meta_data.empty()){{
            TV_ASSERT_INVALID_ARG(
                meta_data.size() == data_.size(), "error");
            TV_ASSERT_INVALID_ARG(
                meta_data.dtype() == tv::int64, "error");
            meta_data_ = meta_data.view(-1); 
        }}else{{
            meta_data_ = tv::zeros({{length}}, tv::int64, -1);
        }}
        if (access_byte_size == -1){{
            access_byte_size = itemsize_;
        }}
        access_byte_size_ = access_byte_size;
        TV_ASSERT_INVALID_ARG(length * itemsize_ % access_byte_size == 0, "error");
        if (align == -1){{
            align = itemsize_;
        }}
        align_ = align;
        TV_ASSERT_INVALID_ARG(byte_offset % align == 0, "misaligned", byte_offset, align);
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function(name="__repr__")
    def repr(self):
        code = pccm.FunctionCode("""
        std::stringstream ss;
        tv::sstream_print<'\\0'>(ss, "Ptr[", get_length(), "|", get_offset(), 
            "|", tv::dtype_str(dtype_), "|", access_byte_size_, "|", byte_offset_, "]");
        return ss.str();
        """)
        return code.ret("std::string")

    @pccm.pybind.mark
    @pccm.member_function
    def clear(self):
        code = pccm.FunctionCode("""
        auto offset = get_offset();
        data_.slice_first_axis(offset, offset + get_length()).zero_();
        if (!meta_data_.empty()){
            meta_data_.slice_first_axis(offset, offset + get_length()).zero_();
        }
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def shadow_copy(self):
        code = pccm.FunctionCode("""
        return *this;
        """)
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function
    def copy(self):
        code = pccm.FunctionCode("""
        auto new_tensor = *this;
        auto new_data = tv::empty(data_.shape(), data_.dtype(), data_.device());
        new_tensor.data_ = new_data;
        std::memcpy(new_data.raw_data(), data_.raw_data(), new_data.raw_size());
        if (!meta_data_.empty()){
            auto new_meta_data = tv::empty(meta_data_.shape(), meta_data_.dtype(), meta_data_.device());
            new_tensor.meta_data_ = new_meta_data;
            std::memcpy(new_meta_data.raw_data(), meta_data_.raw_data(), new_meta_data.raw_size());
        }
        return new_tensor;
        """)
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark_prop_getter(prop_name="length")
    @pccm.member_function
    def get_length(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(byte_offset_ % itemsize_ == 0, "misaligned", byte_offset_, itemsize_);
        return length_ - byte_offset_ / itemsize_;
        """)
        return code.ret("int64_t")

    @pccm.pybind.mark_prop_getter(prop_name="byte_length")
    @pccm.member_function
    def get_byte_length(self):
        code = pccm.FunctionCode("""
        return get_length() * itemsize_;
        """)
        return code.ret("int64_t")

    @pccm.pybind.mark_prop_getter(prop_name="offset")
    @pccm.member_function
    def get_offset(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(byte_offset_ % align_ == 0, "misaligned", byte_offset_, align_);
        return byte_offset_ / itemsize_;
        """)
        return code.ret("int64_t")

    @pccm.pybind.mark_prop_getter(prop_name="access_offset")
    @pccm.member_function
    def get_access_offset(self):
        code = pccm.FunctionCode("""
        auto res = get_offset();
        TV_ASSERT_INVALID_ARG(res % get_access_size() == 0, "misaligned");
        return res / get_access_size();
        """)
        return code.ret("int64_t")

    @pccm.pybind.mark_prop_getter(prop_name="access_size")
    @pccm.member_function
    def get_access_size(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(access_byte_size_ % align_ == 0, "misaligned", access_byte_size_, align_);
        return access_byte_size_ / itemsize_;
        """)
        return code.ret("int")

    @pccm.member_function(inline=True)
    def get_access_byte_size(self):
        code = pccm.FunctionCode("""
        return access_byte_size_;
        """)
        return code.ret("int")

    @pccm.pybind.mark_prop_getter(prop_name="num_access")
    @pccm.member_function
    def get_num_access(self):
        code = pccm.FunctionCode("""
        auto acs = get_access_size();
        TV_ASSERT_RT_ERR(acs > 0, "access size zero.", acs);
        return get_length() / acs;
        """)
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.member_function(name="__add__")
    def add(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG((index * access_byte_size_) % align_ == 0, 
            "misaligned", (index * access_byte_size_), align_);
        auto new_ten = *this;
        new_ten.byte_offset_ += index * access_byte_size_;
        TV_ASSERT_INVALID_ARG(new_ten.byte_offset_ >= 0 , 
            "invalid byte_offset", new_ten.byte_offset_);

        return new_ten;
        """)
        code.arg("index", "int64_t")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function(name="__iadd__")
    def iadd(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG((index * access_byte_size_) % align_ == 0, 
            "misaligned", (index * access_byte_size_), align_);
        byte_offset_ += index * access_byte_size_;
        TV_ASSERT_INVALID_ARG(byte_offset_ >= 0, 
            "invalid byte_offset", byte_offset_);

        return *this;
        """)
        code.arg("index", "int64_t")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function(name="__sub__")
    def sub(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG((index * access_byte_size_) % align_ == 0, 
            "misaligned", (index * access_byte_size_), align_);
        auto new_ten = *this;
        new_ten.byte_offset_ -= index * access_byte_size_;
        TV_ASSERT_INVALID_ARG(new_ten.byte_offset_ >= 0, 
            "invalid byte_offset", new_ten.byte_offset_);

        return new_ten;
        """)
        code.arg("index", "int64_t")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function(name="__isub__")
    def isub(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG((index * access_byte_size_) % align_ == 0, 
            "misaligned", (index * access_byte_size_), align_);
        byte_offset_ -= index * access_byte_size_;
        TV_ASSERT_INVALID_ARG(byte_offset_ >= 0, 
            "invalid byte_offset", byte_offset_);
        return *this;
        """)
        code.arg("index", "int64_t")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function
    def change_access_size(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(length_ % new_acc_size == 0, "misaligned");
        auto new_ten = *this;
        new_ten.access_byte_size_ = new_acc_size * itemsize_;
        if (align == -1){
            align = itemsize_;
        }
        new_ten.align_ = align;
        return new_ten;
        """)
        code.arg("new_acc_size", "int")
        code.arg("align", "int", "-1")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function
    def change_access_byte_size(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(get_byte_length() % new_acc_byte_size == 0, "misaligned");
        // TV_ASSERT_INVALID_ARG(new_acc_byte_size >= itemsize_, "too small, cast dtype instead.", 
        //     new_acc_byte_size, itemsize_);

        auto new_ten = *this;
        new_ten.access_byte_size_ = new_acc_byte_size;
        if (align == -1){
            align = itemsize_;
        }
        new_ten.align_ = align;
        return new_ten;
        """)
        code.arg("new_acc_byte_size", "int")
        code.arg("align", "int", "-1")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function(name="__getitem__")
    def getitem(self):
        code = pccm.FunctionCode(f"""
        TV_ASSERT_INVALID_ARG(get_length() > 0, "error");
        auto new_byte_offset = byte_offset_ + idx_access * access_byte_size_;
        return {self.class_name}(dtype_, get_offset() + (idx_access + 1) * get_access_size(),
            itemsize_, new_byte_offset, itemsize_, data_, meta_data_);
        """)
        code.arg("idx_access", "int")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.member_function(name="__setitem__")
    def setitem(self):
        code = pccm.FunctionCode(f"""
        TV_ASSERT_INVALID_ARG(get_length() > 0, "error");
        TV_ASSERT_INVALID_ARG(idx_access < get_num_access(), "ptr", idx_access, "exceed", get_num_access());
        TV_ASSERT_INVALID_ARG(val.get_byte_length() == access_byte_size_, "nbytes mismatch", val.__repr__(), this->__repr__());
        auto val_ptr = val.data_.raw_data() + val.byte_offset_;
        auto ptr = data_.raw_data() + byte_offset_ + access_byte_size_ * idx_access;
        std::memcpy(ptr, val_ptr, access_byte_size_);
        TV_ASSERT_INVALID_ARG(!meta_data_.empty() && !val.meta_data_.empty(), "error");

        if (!meta_data_.empty() && !val.meta_data_.empty()){{
            TV_ASSERT_INVALID_ARG(meta_data_.dtype() == val.meta_data_.dtype(), "error");
            int dtype_rate = meta_data_.itemsize() / tv::detail::sizeof_dtype(dtype_);
            auto val_meta_ptr = val.meta_data_.raw_data() + val.byte_offset_ * dtype_rate;
            auto meta_ptr = meta_data_.raw_data() + byte_offset_ * dtype_rate + access_byte_size_ * idx_access * dtype_rate;
            std::memcpy(meta_ptr, val_meta_ptr, access_byte_size_ * dtype_rate);
        }}
        """)
        code.arg("idx_access", "int")
        code.arg("val", self.class_name)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="data")
    @pccm.member_function
    def get_data(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(get_length() > 0, "error");
        TV_ASSERT_INVALID_ARG(byte_offset_ % itemsize_ == 0, "misaligned", byte_offset_, itemsize_);
        auto start = byte_offset_ / itemsize_;
        return data_.slice_first_axis(start, start + get_length());
        """)
        return code.ret("tv::Tensor", "cumm.tensorview.Tensor")

    @pccm.pybind.mark_prop_getter(prop_name="meta_data")
    @pccm.member_function
    def get_meta_data(self):
        code = pccm.FunctionCode("""
        TV_ASSERT_INVALID_ARG(get_length() > 0, "error");
        TV_ASSERT_INVALID_ARG(byte_offset_ % itemsize_ == 0, "misaligned", byte_offset_, itemsize_);
        auto start = byte_offset_ / itemsize_;
        return meta_data_.slice_first_axis(start, start + get_length());
        """)
        return code.ret("tv::Tensor", "cumm.tensorview.Tensor")

    @pccm.pybind.mark
    @pccm.member_function
    def astype(self):
        code = pccm.FunctionCode(f"""
        if (dtype == dtype_){{
            return copy();
        }}
        TV_ASSERT_INVALID_ARG(get_length() > 0, "error");
        TV_ASSERT_INVALID_ARG(byte_offset_ % itemsize_ == 0, "misaligned", byte_offset_, itemsize_);
        auto start = byte_offset_ / itemsize_;
        auto new_data = data_.slice_first_axis(start, start + get_length()).astype(tv::DType(dtype), true);
        auto new_ref = {self.class_name}(tv::DType(dtype), get_length(), -1, 0, -1, new_data);
        return new_ref;
        """)
        code.arg("dtype", "int")
        return code.ret(self.class_name, self.class_name)

    @pccm.pybind.mark
    @pccm.static_function
    def check_smem_bank_conflicit(self):
        code = pccm.FunctionCode(f"""
        auto& first_pair = idx_ptrs[0];
        auto& first_ptr = std::get<1>(first_pair);
        auto ref_access_byte_size = first_ptr.get_access_byte_size(); 
        constexpr int kNumSmemBank = 32;
        constexpr int kBankByteSize = 4;
        constexpr int kBankSize = kNumSmemBank * kBankByteSize;
        constexpr int kWarpSize = 32;
        for (int i = 0; i < idx_ptrs.size(); ++i){{
            auto& pair = idx_ptrs[i];
            auto& array_ptr = std::get<1>(pair);
            TV_ASSERT_RT_ERR(ref_access_byte_size == array_ptr.get_access_byte_size(), "all smem access size must be same");
            TV_ASSERT_RT_ERR(array_ptr.get_access_byte_size() <= 16, "only support <= 128bit access for now");

        }}
        int num_phase = std::max(ref_access_byte_size / kBankByteSize, 1); 
        TV_ASSERT_RT_ERR(num_phase > 0, num_phase, ref_access_byte_size);
        int num_thread_per_phase = kWarpSize / num_phase;
        for (int phase = 0; phase < num_phase; ++phase){{
            std::vector<std::unordered_map<int, std::vector<int>>> bank_groups(kNumSmemBank);
            for (int i = phase * num_thread_per_phase; i < (phase + 1) * num_thread_per_phase; ++i){{
                auto& pair = idx_ptrs[i];
                auto idx_access = std::get<0>(pair);
                auto& array_ptr = std::get<1>(pair);
                int b4_acc_idx = (array_ptr.byte_offset_ + array_ptr.access_byte_size_ * idx_access) / kBankByteSize;
                int bank_idx = b4_acc_idx % kNumSmemBank;
                bank_groups[bank_idx][b4_acc_idx].push_back(i);
            }}
            for (int i = 0; i < bank_groups.size(); ++i){{
                if (bank_groups[i].size() > 1){{
                    return bank_groups[i];
                }}
            }}
        }}
        return {{}};
        """)
        code.arg("idx_ptrs",
                 f"std::vector<std::tuple<int, {self.class_name}>>")
        return code.ret(f"std::unordered_map<int, std::vector<int>>",
                        f"Dict[int, List[int]]")
