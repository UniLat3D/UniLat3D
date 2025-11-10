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

class ConvAlgoDesp(GemmAlgoDesp):
    def __init__(self):
        super().__init__()
        self.add_dependency(kernel.ConvUtils)
        self.add_pybind_member("ndim", "int")
        self.add_pybind_member("op_type", "int")
        self.add_pybind_member("iter_algo", "int")
        self.add_pybind_member("layout_i, layout_w, layout_o", "int")
        self.add_pybind_member("interleave_i, interleave_w, interleave_o",
                               "int")
        self.add_pybind_member("mask_sparse", "bool", "false")
        self.add_pybind_member("increment_k_first", "bool", "false")

        self.add_member("conv2gemm_inds", "std::array<int, 3>")
        self.add_member("gemm2conv_inds", "std::array<int, 3>")

    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.code()
        code.arg("ndim", "int")
        code.arg("op_type", "int")
        code.ctor_init("GemmAlgoDesp", "")
        code.ctor_init("ndim", "ndim")
        code.ctor_init("op_type", f"op_type")
        code.ctor_init("iter_algo", f"{ConvIterAlgo.Optimized.value}")
        code.ctor_init("layout_i", f"{ConvLayoutType.ChannelLast.value}")
        code.ctor_init("layout_w", f"{ConvLayoutType.ChannelLast.value}")
        code.ctor_init("layout_o", f"{ConvLayoutType.ChannelLast.value}")
        code.ctor_init("interleave_i", f"1")
        code.ctor_init("interleave_w", f"1")
        code.ctor_init("interleave_o", f"1")
        code.ctor_init("conv2gemm_inds", f"conv_iwo_012_to_abc(op_type)")
        code.ctor_init("gemm2conv_inds", f"gemm_abc_012_to_iwo(op_type)")
        return code

    @pccm.pybind.mark
    @pccm.member_function(name="__repr__")
    def repr(self):
        code = pccm.code()
        code.raw(f"""
        check_valid();
        std::stringstream ss;
        ss << GemmAlgoDesp::__repr__();
        ss << "_C" << ndim << "_" << op_type << iter_algo;
        std::string layout_i_str = layout_i == {ConvLayoutType.ChannelFirst.value} ? "F" : "L";
        std::string layout_w_str = layout_w == {ConvLayoutType.ChannelFirst.value} ? "F" : "L";
        std::string layout_o_str = layout_o == {ConvLayoutType.ChannelFirst.value} ? "F" : "L";
        if (interleave_i > 1){{
            layout_i_str += std::to_string(interleave_i);
        }}
        if (interleave_w > 1){{
            layout_w_str += std::to_string(interleave_w);
        }}
        if (interleave_o > 1){{
            layout_o_str += std::to_string(interleave_o);
        }}

        ss << layout_i_str << layout_w_str << layout_o_str;
        if (mask_sparse){{
            ss << "_" << increment_k_first ? "SF" : "SK";
        }}
        return ss.str();
        """)
        return code.ret("std::string")

    @pccm.pybind.mark
    @pccm.static_function
    def conv_iwo_012_to_abc(self):
        code = pccm.code()
        code.arg("op_type", "int")
        code.raw(f"""
        if (op_type == {ConvOpType.kForward.value}){{
            return {{0, 1, 2}};
        }}
        if (op_type == {ConvOpType.kBackwardInput.value}){{
            return {{2, 1, 0}};
        }}
        if (op_type == {ConvOpType.kBackwardWeight.value}){{
            return {{1, 2, 0}};
        }}
        TV_THROW_RT_ERR("unknown op type",op_type);
        """)
        return code.ret("std::array<int, 3>")

    @pccm.pybind.mark
    @pccm.static_function
    def gemm_abc_012_to_iwo(self):
        code = pccm.code()
        code.arg("op_type", "int")
        code.raw(f"""
        if (op_type == {ConvOpType.kForward.value}){{
            return {{0, 1, 2}};
        }}
        if (op_type == {ConvOpType.kBackwardInput.value}){{
            return {{2, 1, 0}};
        }}
        if (op_type == {ConvOpType.kBackwardWeight.value}){{
            return {{2, 0, 1}};
        }}
        TV_THROW_RT_ERR("unknown op type",op_type);
        """)
        return code.ret("std::array<int, 3>")

    @pccm.pybind.mark_prop_getter(prop_name="dtype_input")
    @pccm.member_function
    def dtype_input(self):
        code = pccm.code()
        code.raw(f"""
        std::array<int, 3> dtypes{{dtype_a, dtype_b, dtype_c}};
        return dtypes[conv2gemm_inds[0]];
        """)
        return code.ret("int")

    @pccm.pybind.mark_prop_getter(prop_name="dtype_weight")
    @pccm.member_function
    def dtype_weight(self):
        code = pccm.code()
        code.raw(f"""
        std::array<int, 3> dtypes{{dtype_a, dtype_b, dtype_c}};
        return dtypes[conv2gemm_inds[1]];
        """)
        return code.ret("int")

    @pccm.pybind.mark_prop_getter(prop_name="dtype_output")
    @pccm.member_function
    def dtype_output(self):
        code = pccm.code()
        code.raw(f"""
        std::array<int, 3> dtypes{{dtype_a, dtype_b, dtype_c}};
        return dtypes[conv2gemm_inds[2]];
        """)
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.member_function
    def supported(self):
        code = pccm.code()
        code.arg("m,n,k, C, K, mask_width", "int")
        code.raw(f"""
        bool res = GemmAlgoDesp::supported(m, n, k);
        if (mask_sparse){{
            if (op_type == {ConvOpType.kForward.value}){{
                // NC -> NRSC @ KRSC
                res &= C % element_per_access_a == 0;
            }}else if (op_type == {ConvOpType.kBackwardInput.value}){{
                // NK -> NRSK @ KRSC -> RSKC
                res &= K % element_per_access_a == 0;
            }}else{{
                // NK @ NC -> NRSC
                // we must ensure every k iteration only have one mask (from forward),
                res &= mask_width % tile_shape[2] == 0;
                res &= K % element_per_access_a == 0;
                res &= C % element_per_access_b == 0;
            }}
        }}
        return res;
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.member_function
    def query_conv_workspace_size(self):
        code = pccm.code()
        code.arg("m,n,k,split_k_slices,kv", "int")
        code.raw(f"""
        if (!mask_sparse){{
            return query_workspace_size(m, n, k, split_k_slices);
        }}
        auto logical_tile_count = tv::gemm::get_spconv_logical_tile_count(m, n, k, 
            tile_shape[0], tile_shape[1], split_k_slices, kv, op_type);
        int workspace_size = 0;
        if (split_k_slices > 1){{
            if (split_k_serial()){{
                workspace_size = sizeof(int) * logical_tile_count[0] * logical_tile_count[1];
            }} else if (split_k_parallel()){{
                workspace_size = tv::detail::sizeof_dtype(tv::DType(dacc)) * m * n * logical_tile_count[2];
            }} else{{
                TV_THROW_INVALID_ARG("not impemented");
            }}
        }}
        return workspace_size;
        """)
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.member_function
    def supported_ldx_conv(self):
        code = pccm.code()
        code.arg("ldi, ldw, ldo", "int")
        code.raw(f"""
        bool res = true;
        std::array<int, 3> epas{{element_per_access_a, element_per_access_b, element_per_access_c}};
        int epa_i = epas[conv2gemm_inds[0]];
        int epa_w = epas[conv2gemm_inds[1]];
        int epa_o = epas[conv2gemm_inds[2]];

        if (epa_i > 0){{
            res &= ldi % epa_i == 0;
        }}
        if (epa_w > 0){{
            res &= ldw % epa_w == 0;
        }}
        if (epa_o > 0){{
            res &= ldo % epa_o == 0;
        }}
        return res;
        """)
        return code.ret("bool")


class ConvParams(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(ConvAlgoDesp, CummNVRTCLib)
        self.add_pybind_member("conv_algo_desp", "ConvAlgoDesp")
        self.add_pybind_member("input,weight,output",
                               "tv::Tensor",
                               pyanno="cumm.tensorview.Tensor")
        self.add_pybind_member("split_k_slices", "int", "1")
        self.add_pybind_member("padding,stride,dilation", "std::vector<int>")
        self.add_pybind_member("alpha,beta", "float")
        self.add_pybind_member("mask_width", "int")
        self.add_pybind_member("mask_filter", "uint32_t")
        self.add_pybind_member("reverse_mask", "bool")
        self.add_pybind_member("verbose", "bool")
        self.add_pybind_member("timer",
                               "tv::CUDAKernelTimer",
                               "tv::CUDAKernelTimer(false)",
                               pyanno="cumm.tensorview.CUDAKernelTimer")

        self.add_pybind_member("workspace",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("mask",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("mask_argsort",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("indices",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("mask_output",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("stream", "std::uintptr_t", "0", pyanno="int")
        self.add_pybind_member("nvrtc_params", "tv::gemm::NVRTCParams", "tv::gemm::NVRTCParams()")


    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.code()
        code.arg("ndim", "int")
        code.arg("op_type", "int")
        code.arg(
            "timer",
            "tv::CUDAKernelTimer",
            "tv::CUDAKernelTimer(false)",
            pyanno="cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")

        code.ctor_init("conv_algo_desp", "ndim, op_type")
        code.ctor_init("input", "tv::Tensor()")
        code.ctor_init("weight", "tv::Tensor()")
        code.ctor_init("output", "tv::Tensor()")
        code.ctor_init("padding", "std::vector<int>()")
        code.ctor_init("stride", "std::vector<int>()")
        code.ctor_init("dilation", "std::vector<int>()")
        code.ctor_init("alpha", "1.0")
        code.ctor_init("beta", "0.0")
        code.ctor_init("mask_width", "-1")
        code.ctor_init("mask_filter", "0xffffffff")
        code.ctor_init("reverse_mask", "false")
        code.ctor_init("verbose", "false")
        code.ctor_init("timer", "timer")

        return code
@pccm.pybind.bind_class_module_local
class GemmAlgoDesp(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmUtilsCPU)
        # why not std::optional?
        # c++17 requires cuda11. to support cuda 10.2, we can't use c++17 for now.
        self.add_pybind_member("dtype_a,dtype_b,dtype_c",
                               "int")  # -1 means unset
        self.add_member("trans_a_,trans_b_,trans_c_", "int")  # -1 means unset
        self.add_pybind_member("tile_shape,warp_tile_shape",
                               "std::array<int, 3>",
                               pyanno="Tuple[int, int, int]")
        self.add_pybind_member("num_stage", "int")
        self.add_pybind_member("dacc,dcomp", "int")
        self.add_pybind_member("algo", "std::string")
        self.add_pybind_member("tensorop", "std::array<int, 3>",
                               "std::array<int, 3>{}")
        self.add_pybind_member("split_k_serial_", "int", "0")  # -1 means unset
        self.add_pybind_member("split_k_parallel_", "int",
                               "0")  # -1 means unset
        self.add_pybind_member("shuffle_type", "std::string",
                               f"\"{ShuffleStrideType.NoShuffle.value}\"")
        self.add_pybind_member("element_per_access_a", "int", "-1")
        self.add_pybind_member("element_per_access_b", "int", "-1")
        self.add_pybind_member("element_per_access_c", "int", "-1")
        self.add_pybind_member("access_per_vector", "int", "1")

    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.code()
        code.ctor_init("dtype_a", "int(tv::unknown)")
        code.ctor_init("dtype_b", "int(tv::unknown)")
        code.ctor_init("dtype_c", "int(tv::unknown)")

        code.ctor_init("trans_a_", "-1")
        code.ctor_init("trans_b_", "-1")
        code.ctor_init("trans_c_", "-1")
        code.ctor_init("tile_shape", "{-1, -1, -1}")
        code.ctor_init("warp_tile_shape", "{-1, -1, -1}")
        code.ctor_init("num_stage", "-1")
        code.ctor_init("dacc", "int(tv::unknown)")
        code.ctor_init("dcomp", "int(tv::unknown)")
        code.ctor_init("algo", "\"\"")
        code.ctor_init("tensorop", "{-1, -1, -1}")
        code.ctor_init("shuffle_type",
                       f"\"{ShuffleStrideType.NoShuffle.value}\"")
        code.ctor_init("split_k_serial_", "0")
        code.ctor_init("split_k_parallel_", "0")
        code.ctor_init("element_per_access_a", "-1")
        code.ctor_init("element_per_access_b", "-1")
        code.ctor_init("element_per_access_c", "-1")
        code.ctor_init("access_per_vector", "1")

        return code

    @pccm.pybind.mark
    @pccm.member_function(name="__repr__")
    def repr(self):
        code = pccm.code()
        code.raw(f"""
        check_valid();
        std::stringstream ss;
        ss << algo << "_" << tv::dtype_short_str(dtype_a) << tv::dtype_short_str(dtype_b)
            << tv::dtype_short_str(dtype_c) << tv::dtype_short_str(dacc) << tv::dtype_short_str(dcomp);
        ss << (trans_a() ? "n" : "t") << (trans_b() ? "n" : "t") << (trans_c() ? "n" : "t");
        ss << "_m" << tile_shape[0] << "n" << tile_shape[1] << "k" << tile_shape[2];
        ss << "m" << warp_tile_shape[0] << "n" << warp_tile_shape[1] << "k" << warp_tile_shape[2];
        ss << "A" << access_per_vector;
        if (tensorop[0] != -1){{
            ss << "T" << tensorop[0] << tensorop[1] << tensorop[2];
        }}
        if (shuffle_type != "{ShuffleStrideType.NoShuffle.value}"){{
            ss << "_" << shuffle_type;
        }}
        ss << (split_k_serial() ? 1 : 0) << (split_k_parallel() ? 1 : 0);
        return ss.str();
        """)
        return code.ret("std::string")

    @pccm.pybind.mark_prop_getter(prop_name="split_k_serial")
    @pccm.member_function
    def split_k_serial(self):
        code = pccm.code()
        code.raw(f"""
        return split_k_serial_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="split_k_serial")
    @pccm.member_function
    def split_k_serial_set(self):
        code = pccm.code()
        code.arg("val", "bool")
        code.raw(f"""
        split_k_serial_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="split_k_parallel")
    @pccm.member_function
    def split_k_parallel(self):
        code = pccm.code()
        code.raw(f"""
        return split_k_parallel_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="split_k_parallel")
    @pccm.member_function
    def split_k_parallel_set(self):
        code = pccm.code()
        code.arg("val", "bool")
        code.raw(f"""
        split_k_parallel_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def check_valid(self):
        code = pccm.code()
        code.raw(f"""
        TV_ASSERT_RT_ERR(trans_a_ != -1 && !trans_b_ != -1 && trans_c_ != -1 && !algo.empty(), 
            "trans_a, trans_b, trans_c and algo must be set");
        for (int i = 0; i < 3; ++i){{
            TV_ASSERT_RT_ERR(tile_shape[i] > 0 && warp_tile_shape[i] > 0, 
                "tile_shape and warp_tile_shape must be set, but they are", tile_shape, warp_tile_shape);
        }}
        if (algo != "{GemmAlgo.Simt.value}" && algo != "{GemmAlgo.SimtDP4A.value}" && algo != "{GemmAlgo.SimtDP2A.value}"){{
            // tensor op must not empty
            for (int i = 0; i < 3; ++i){{
                TV_ASSERT_RT_ERR(tensorop[i] > 0, 
                    "tensorop must be set, but they are", tensorop);
            }}
        }}
        TV_ASSERT_RT_ERR(dtype_a != int(tv::unknown) && dtype_b != int(tv::unknown) && dtype_c != int(tv::unknown), 
            "dacc and dcomp must be set to valid value");

        TV_ASSERT_RT_ERR(dacc != int(tv::unknown) && dcomp != int(tv::unknown), "dacc and dcomp must be set to valid value");
        TV_ASSERT_RT_ERR(num_stage > 0, "num_stage must larger than zero");
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="trans_a")
    @pccm.member_function
    def trans_a(self):
        code = pccm.code()
        code.raw(f"""
        return trans_a_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="trans_a")
    @pccm.member_function
    def trans_a_set(self):
        code = pccm.code()
        code.arg("val", "bool")
        code.raw(f"""
        trans_a_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="trans_b")
    @pccm.member_function
    def trans_b(self):
        code = pccm.code()
        code.raw(f"""
        return trans_b_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="trans_b")
    @pccm.member_function
    def trans_b_set(self):
        code = pccm.code()
        code.arg("val", "bool")
        code.raw(f"""
        trans_b_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="trans_c")
    @pccm.member_function
    def trans_c(self):
        code = pccm.code()
        code.raw(f"""
        return trans_c_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="trans_c")
    @pccm.member_function
    def trans_c_set(self):
        code = pccm.code()
        code.arg("val", "bool")
        code.raw(f"""
        trans_c_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def query_workspace_size(self):
        code = pccm.code()
        code.arg("m,n,k,split_k_slices", "int")
        code.raw(f"""
        auto logical_tile_count =  GemmUtilsCPU::get_logical_tile_count(m, n, k, tile_shape[0], tile_shape[1], split_k_slices);
        int workspace_size = 0;
        if (split_k_slices > 1){{
            if (split_k_serial()){{
                workspace_size = sizeof(int) * logical_tile_count[0] * logical_tile_count[1];
            }} else if (split_k_parallel()){{
                workspace_size = tv::detail::sizeof_dtype(tv::DType(dacc)) * m * n * logical_tile_count[2];
            }} else{{
                TV_THROW_INVALID_ARG("not impemented");
            }}
        }}
        return workspace_size;
        """)
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.member_function
    def supported(self):
        code = pccm.code()
        code.arg("m,n,k", "int")
        code.raw(f"""
        bool res = true;
        auto lda = trans_a() ? m : k;
        auto ldb = trans_b() ? k : n;
        auto ldc = trans_c() ? m : n;
        if (element_per_access_a > 0){{
            res &= lda % element_per_access_a == 0;
        }}
        if (element_per_access_b > 0){{
            res &= ldb % element_per_access_b == 0;
        }}
        if (element_per_access_c > 0){{
            res &= ldc % element_per_access_c == 0;
        }}
        return res;
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.member_function
    def supported_ldx(self):
        code = pccm.code()
        code.arg("lda, ldb, ldc", "int")
        code.raw(f"""
        bool res = true;
        if (element_per_access_a > 0){{
            res &= lda % element_per_access_a == 0;
        }}
        if (element_per_access_b > 0){{
            res &= ldb % element_per_access_b == 0;
        }}
        if (element_per_access_c > 0){{
            res &= ldc % element_per_access_c == 0;
        }}
        return res;
        """)
        return code.ret("bool")

class GemmParams(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmAlgoDesp, CummNVRTCLib)
        # why not std::optional?
        # c++17 requires cuda11. to support cuda 10.2, we can't use c++17 for now.
        self.add_pybind_member("algo_desp",
                               "GemmAlgoDesp",
                               pyanno="GemmAlgoDesp")
        self.add_member("a,b,c", "tv::Tensor", pyanno="cumm.tensorview.Tensor")
        self.add_pybind_member("split_k_slices", "int", "1")
        self.add_pybind_member("workspace",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        # for spatial sparse convolution (split kernel algorithm)
        self.add_pybind_member("a_inds",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("b_inds",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("c_inds",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("alpha,beta", "float")
        self.add_pybind_member("stream", "std::uintptr_t", pyanno="int")
        self.add_pybind_member("timer",
                               "tv::CUDAKernelTimer",
                               "tv::CUDAKernelTimer(false)",
                               pyanno="cumm.tensorview.CUDAKernelTimer")
        self.add_pybind_member("nvrtc_params", "tv::gemm::NVRTCParams", "tv::gemm::NVRTCParams()")

    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.code()
        code.arg(
            "timer",
            "tv::CUDAKernelTimer",
            "tv::CUDAKernelTimer(false)",
            pyanno="cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")
        code.ctor_init("a", "tv::Tensor()")
        code.ctor_init("b", "tv::Tensor()")
        code.ctor_init("c", "tv::Tensor()")
        code.ctor_init("split_k_slices", "1")
        code.ctor_init("workspace", "tv::Tensor()")
        code.ctor_init("a_inds", "tv::Tensor()")
        code.ctor_init("b_inds", "tv::Tensor()")
        code.ctor_init("c_inds", "tv::Tensor()")
        code.ctor_init("alpha", "1.0")
        code.ctor_init("beta", "0.0")
        code.ctor_init("stream", "0")
        code.ctor_init("timer", "timer")

        return code

    @pccm.pybind.mark
    @pccm.member_function
    def check_valid(self):
        code = pccm.code()
        code.raw(f"""
        algo_desp.check_valid();
        TV_ASSERT_RT_ERR(!a.empty() && !b.empty() && !c.empty(), 
            "a,b,c must not empty");
        if (algo_desp.shuffle_type == "{ShuffleStrideType.ShuffleAC.value}"){{
            TV_ASSERT_RT_ERR(!c_inds.empty(), "a_inds,c_inds tensor must not empty");
        }}else if (algo_desp.shuffle_type == "{ShuffleStrideType.ShuffleAB.value}"){{
            TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(), "a_inds,b_inds tensor must not empty");
        }}
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="a")
    @pccm.member_function
    def a_get(self):
        code = pccm.code()
        code.raw(f"""
        return a;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark_prop_setter(prop_name="a")
    @pccm.member_function
    def a_set(self):
        code = pccm.code()
        code.arg("val", "tv::Tensor")
        code.raw(f"""
        a = val;
        algo_desp.dtype_a = int(a.dtype());
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="b")
    @pccm.member_function
    def b_get(self):
        code = pccm.code()
        code.raw(f"""
        return b;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark_prop_setter(prop_name="b")
    @pccm.member_function
    def b_set(self):
        code = pccm.code()
        code.arg("val", "tv::Tensor")
        code.raw(f"""
        b = val;
        algo_desp.dtype_b = int(b.dtype());
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="c")
    @pccm.member_function
    def c_get(self):
        code = pccm.code()
        code.raw(f"""
        return c;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark_prop_setter(prop_name="c")
    @pccm.member_function
    def c_set(self):
        code = pccm.code()
        code.arg("val", "tv::Tensor")
        code.raw(f"""
        c = val;
        algo_desp.dtype_c = int(c.dtype());
        """)
        return code


