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

import os
from pathlib import Path
from typing import List, Optional

import pccm
from ccimport import compat
from pccm.utils import project_is_editable, project_is_installed

from cumm.common import CompileInfo, PyBind11, TensorView, TensorViewCPU, TensorViewCompileLinkFlags, TensorViewHeader, TensorViewImplFlags, get_cuda_version_by_nvcc, TensorViewArrayLinalg
from cumm.constants import CUMM_CPU_ONLY_BUILD, PACKAGE_ROOT
from .constants import CUMM_APPLE_METAL_CPP_ROOT, CUMM_CUDA_VERSION, PACKAGE_NAME
from cumm.conv.nvrtc_code import nvrtc_conv_template
from cumm.gemm.nvrtc_code import nvrtc_gemm_template

_TENSORVIEW_BIND_CODE_ANNO_PATH = PACKAGE_ROOT / "tensorview_bind_anno.pyi"
with _TENSORVIEW_BIND_CODE_ANNO_PATH.open("r") as f:
    _TENSORVIEW_BIND_CODE_ANNO = f.read()

class AppleMetalImpl(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewHeader, TensorViewCompileLinkFlags)
        if compat.InMacOS:
            path = Path.home() / "metal-cpp"
            if CUMM_APPLE_METAL_CPP_ROOT is not None:
                path = Path(CUMM_APPLE_METAL_CPP_ROOT)
            assert path.exists(), ("if you use mac os, you must download metal-cpp and save"
                f" it to home folder or use CUMM_APPLE_METAL_CPP_ROOT. {CUMM_APPLE_METAL_CPP_ROOT}")
            self.build_meta.add_includes(str(path))
            self.build_meta.add_ldflags("clang++", "-framework Metal", "-framework CoreGraphics")
    
    @pccm.pybind.mark
    @pccm.static_function
    def hello(self):
        code = pccm.code()
        if compat.InMacOS:
            code.code_after_include = """
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <tensorview/contexts/core.h>

namespace tv {
std::shared_ptr<AppleMetalContext> AppleMetalContext::getInstance()
{
  static std::shared_ptr<AppleMetalContext> ctx = std::make_shared<AppleMetalContext>();
  return ctx;
}
}
            """
        return code

class TensorViewBind(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, TensorViewArrayLinalg, PyBind11, TensorViewImplFlags, AppleMetalImpl, CompileInfo)
        self.add_include("tensorview/pybind_utils.h")
        self.add_include("tensorview/profile/all.h")
        self.add_include("limits")
        self.add_include("tensorview/cuda/nvrtc.h")
        self.add_include("tensorview/metal/metallib.h")

        self.add_include("tensorview/gemm/core/nvrtc_bases.h")
        self.add_include("tensorview/gemm/core/params.h")
        self.add_include("tensorview/gemm/core/nvrtc_params.h")
        if not compat.InWindows:
            self.add_include("cxxabi.h")
        

        if not CUMM_CPU_ONLY_BUILD and not compat.InMacOS:
            # cufilt (nv_decode.h) is used to demangle
            # c++ names in ptx.

            cuda_ver = get_cuda_version_by_nvcc().split(".")
            cuda_ver_ints = list(map(int, cuda_ver))
            if cuda_ver_ints >= [11, 4]:
                # cuflit has problem in arm platforms and windows, so we remove it.
                # self.add_include("nv_decode.h")
                # self.build_meta.add_libraries("nvrtc", "cufilt")
                self.build_meta.add_libraries("nvrtc")
            else:
                self.build_meta.add_libraries("nvrtc")
            if compat.InLinux:
                self.build_meta.add_ldflags("g++", "-Wl,--no-as-needed", "-lnvrtc-builtins")
                self.build_meta.add_ldflags("clang++", "-Wl,--no-as-needed", "-lnvrtc-builtins")
                self.build_meta.add_ldflags("nvcc", "-Wl,--no-as-needed", "-lnvrtc-builtins")
            if not compat.InWindows:
                self.build_meta.add_libraries("dl")
            else:
                # disable min/max macro if include Windows.h
                self.build_meta.add_cflags("cl", "/DNOMINMAX")

    @pccm.pybind.mark
    @pccm.static_function
    def hello(self):
        code = pccm.code()
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def _compile_test(self):
        # compile some code to check msvc problem during cumm build
        # msvc is the best compiler in the world.
        code = pccm.code()
        code.raw(f"""
        namespace op = tv::arrayops;
        auto a = tv::array_nd<float, 1, 3>{{}};
        auto b = tv::array_nd<float, 3, 1>{{}};
        
        tv::ssprint(a + b, a - b);
        auto p = tv::array_nd<float, 3>{{}};
        auto m = tv::array_nd<float, 4, 4>{{}};
        tv::ssprint(p.op<op::maximum>(p), p.op<op::transform_3d>(m));
        """)
        return code

    @pccm.static_function
    def run_nvrtc_conv_kernel(self):
        code = pccm.code()
        code.arg("params",
                 "tv::gemm::ConvParams",
                 pyanno="cumm.tensorview.gemm.ConvParams")
        if CUMM_CPU_ONLY_BUILD or compat.IsAppleSiliconMacOs:
            code.raw(f"""
            TV_THROW_RT_ERR("cpu-only or apple build don't support this");
            """)
            return code
        nvrtc_conv_template(code)

        code.raw(f"""
        TV_THROW_RT_ERR("you must use nvrtc kernel to run conv by this function");
        """)
        return code

    @pccm.static_function
    def run_nvrtc_gemm_kernel(self):
        code = pccm.code()
        code.arg("params",
                 "tv::gemm::GemmParams",
                 pyanno="cumm.tensorview.gemm.GemmParams")
        if CUMM_CPU_ONLY_BUILD or compat.IsAppleSiliconMacOs:
            code.raw(f"""
            TV_THROW_RT_ERR("cpu-only or apple build don't support this");
            """)
            return code
        nvrtc_gemm_template(code)
        code.raw(f"""
        TV_THROW_RT_ERR("you must use nvrtc kernel to run gemm by this function");
        """)
        return code

    @pccm.static_function
    def bind_enums(self):
        code = pccm.code()
        code.arg("module_", "pybind11::module_")
        code.raw("""
        py::enum_<tv::gemm::ConvOpType>(module_, "ConvOpType")
            .value("Forward", tv::gemm::ConvOpType::kForward)
            .value("BackwardInput", tv::gemm::ConvOpType::kBackwardInput)
            .value("BackwardWeight", tv::gemm::ConvOpType::kBackwardWeight);
        py::enum_<tv::gemm::ConvMode>(module_, "ConvMode")
            .value("Convolution", tv::gemm::ConvMode::kConvolution)
            .value("CrossCorrelation", tv::gemm::ConvMode::kCrossCorrelation);
        py::enum_<tv::gemm::ConvIterAlgo>(module_, "ConvIterAlgo")
            .value("Analytic", tv::gemm::ConvIterAlgo::kAnalytic)
            .value("Optimized", tv::gemm::ConvIterAlgo::kOptimized);

        py::enum_<tv::gemm::ConvLayoutType>(module_, "ConvLayoutType")
            .value("ChannelFirst", tv::gemm::ConvLayoutType::kChannelFirst)
            .value("ChannelLast", tv::gemm::ConvLayoutType::kChannelLast)
            .value("SpatialFirst", tv::gemm::ConvLayoutType::kSpatialFirst);
        py::enum_<tv::gemm::ShuffleStrideType>(module_, "ShuffleStrideType")
            .value("NoShuffle", tv::gemm::ShuffleStrideType::kNoShuffle)
            .value("ShuffleAC", tv::gemm::ShuffleStrideType::kShuffleAC)
            .value("ShuffleAB", tv::gemm::ShuffleStrideType::kShuffleAB);
        py::enum_<tv::gemm::Activation>(module_, "Activation")
            .value("None_", tv::gemm::Activation::kNone)
            .value("ReLU", tv::gemm::Activation::kReLU)
            .value("Sigmoid", tv::gemm::Activation::kSigmoid)
            .value("LeakyReLU", tv::gemm::Activation::kLeakyReLU);
        """)
        """
            .value("Tanh", tv::gemm::Activation::kTanh)
            .value("ELU", tv::gemm::Activation::kELU)
            .value("SeLU", tv::gemm::Activation::kSeLU)
            .value("Softsign", tv::gemm::Activation::kSoftsign)
            .value("Softplus", tv::gemm::Activation::kSoftplus)
            .value("Clip", tv::gemm::Activation::kClip)
            .value("HardSigmoid", tv::gemm::Activation::kHardSigmoid)
            .value("ScaledTanh", tv::gemm::Activation::kScaledTanh)
            .value("ThresholdedReLU", tv::gemm::Activation::kThresholdedReLU)
        """
        return code

    @pccm.static_function
    def bind_nvrtc_params(self):
        code = pccm.code()
        code.arg("module_", "pybind11::module_")
        code.raw("""
        py::class_<tv::gemm::NVRTCParams, std::shared_ptr<tv::gemm::NVRTCParams>>(module_, "NVRTCParams")
          .def(py::init<>())
          .def_readwrite("cumodule", &tv::gemm::NVRTCParams::cumodule)
          .def_readwrite("kernel_name", &tv::gemm::NVRTCParams::kernel_name)
          .def_readwrite("init_kernel_name", &tv::gemm::NVRTCParams::init_kernel_name)
          .def_readwrite("constant_name", &tv::gemm::NVRTCParams::constant_name)
          .def_readwrite("param_size", &tv::gemm::NVRTCParams::param_size)
          .def_readwrite("param_storage", &tv::gemm::NVRTCParams::param_storage)
          .def_readwrite("param_storage_cpu", &tv::gemm::NVRTCParams::param_storage_cpu)
          .def_readwrite("num_threads", &tv::gemm::NVRTCParams::num_threads)
          .def_readwrite("smem_size", &tv::gemm::NVRTCParams::smem_size)
          .def_readwrite("mode", &tv::gemm::NVRTCParams::mode);
        """)
        return code

    @pccm.static_function
    def bind_gemm_algo_desp(self):
        code = pccm.code()
        code.arg("module_", "pybind11::module_")
        code.raw("""
        pybind11::class_<tv::gemm::GemmAlgoDesp> m_cls(
            module_, "GemmAlgoDesp");
        m_cls.def(pybind11::init<>());
        m_cls.def("__repr__", &tv::gemm::GemmAlgoDesp::__repr__,
                  pybind11::return_value_policy::automatic);
        m_cls.def_property("split_k_serial",
                          &tv::gemm::GemmAlgoDesp::split_k_serial,
                          &tv::gemm::GemmAlgoDesp::split_k_serial_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def_property("split_k_parallel",
                          &tv::gemm::GemmAlgoDesp::split_k_parallel,
                          &tv::gemm::GemmAlgoDesp::split_k_parallel_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def("check_valid", &tv::gemm::GemmAlgoDesp::check_valid,
                  pybind11::return_value_policy::automatic);
        m_cls.def("copy", [](const tv::gemm::GemmAlgoDesp& self) -> tv::gemm::GemmAlgoDesp{
          return self;
        }, pybind11::return_value_policy::automatic);
        m_cls.def_property("trans_a", &tv::gemm::GemmAlgoDesp::trans_a,
                          &tv::gemm::GemmAlgoDesp::trans_a_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def_property("trans_b", &tv::gemm::GemmAlgoDesp::trans_b,
                          &tv::gemm::GemmAlgoDesp::trans_b_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def_property("trans_c", &tv::gemm::GemmAlgoDesp::trans_c,
                          &tv::gemm::GemmAlgoDesp::trans_c_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def("query_workspace_size",
                  &tv::gemm::GemmAlgoDesp::query_workspace_size,
                  pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
                  pybind11::arg("split_k_slices"),
                  pybind11::return_value_policy::automatic);
        m_cls.def("supported", &tv::gemm::GemmAlgoDesp::supported,
                  pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
                  pybind11::return_value_policy::automatic);
        m_cls.def("supported_ldx", &tv::gemm::GemmAlgoDesp::supported_ldx,
                  pybind11::arg("lda"), pybind11::arg("ldb"), pybind11::arg("ldc"),
                  pybind11::return_value_policy::automatic);
        m_cls.def_readwrite("dtype_a", &tv::gemm::GemmAlgoDesp::dtype_a);
        m_cls.def_readwrite("dtype_b", &tv::gemm::GemmAlgoDesp::dtype_b);
        m_cls.def_readwrite("dtype_c", &tv::gemm::GemmAlgoDesp::dtype_c);
        m_cls.def_readwrite("tile_shape",
                            &tv::gemm::GemmAlgoDesp::tile_shape);
        m_cls.def_readwrite("warp_tile_shape",
                            &tv::gemm::GemmAlgoDesp::warp_tile_shape);
        m_cls.def_readwrite("num_stage",
                            &tv::gemm::GemmAlgoDesp::num_stage);
        m_cls.def_readwrite("dacc", &tv::gemm::GemmAlgoDesp::dacc);
        m_cls.def_readwrite("dcomp", &tv::gemm::GemmAlgoDesp::dcomp);
        m_cls.def_readwrite("algo", &tv::gemm::GemmAlgoDesp::algo);
        m_cls.def_readwrite("tensorop", &tv::gemm::GemmAlgoDesp::tensorop);
        m_cls.def_readwrite("split_k_serial_",
                            &tv::gemm::GemmAlgoDesp::split_k_serial_);
        m_cls.def_readwrite("split_k_parallel_",
                            &tv::gemm::GemmAlgoDesp::split_k_parallel_);
        m_cls.def_readwrite("shuffle_type",
                            &tv::gemm::GemmAlgoDesp::shuffle_type);
        m_cls.def_readwrite("element_per_access_a",
                            &tv::gemm::GemmAlgoDesp::element_per_access_a);
        m_cls.def_readwrite("element_per_access_b",
                            &tv::gemm::GemmAlgoDesp::element_per_access_b);
        m_cls.def_readwrite("element_per_access_c",
                            &tv::gemm::GemmAlgoDesp::element_per_access_c);
        m_cls.def_readwrite("access_per_vector",
                            &tv::gemm::GemmAlgoDesp::access_per_vector);
        m_cls.def_readwrite("is_nvrtc",
                            &tv::gemm::GemmAlgoDesp::is_nvrtc);
        m_cls.def_readwrite("min_arch",
                            &tv::gemm::GemmAlgoDesp::min_arch);

        """)
        return code

    @pccm.static_function
    def bind_conv_algo_desp(self):
        code = pccm.code()
        code.arg("module_", "pybind11::module_")
        code.raw("""
        pybind11::class_<tv::gemm::ConvAlgoDesp,
                        tv::gemm::GemmAlgoDesp>
            m_cls(module_, "ConvAlgoDesp");
        m_cls.def(pybind11::init<int, tv::gemm::ConvOpType>(), pybind11::arg("ndim"),
                  pybind11::arg("op_type"));
        m_cls.def("__repr__", &tv::gemm::ConvAlgoDesp::__repr__,
                  pybind11::return_value_policy::automatic);
        m_cls.def("copy", [](const tv::gemm::ConvAlgoDesp& self) -> tv::gemm::ConvAlgoDesp{
          return self;
        }, pybind11::return_value_policy::automatic);
        m_cls.def_static("conv_iwo_012_to_abc",
                        &tv::gemm::ConvAlgoDesp::conv_iwo_012_to_abc,
                        pybind11::arg("op_type"),
                        pybind11::return_value_policy::automatic);
        m_cls.def_static("gemm_abc_012_to_iwo",
                        &tv::gemm::ConvAlgoDesp::gemm_abc_012_to_iwo,
                        pybind11::arg("op_type"),
                        pybind11::return_value_policy::automatic);
        m_cls.def_property_readonly("dtype_input",
                                    &tv::gemm::ConvAlgoDesp::dtype_input,
                                    pybind11::return_value_policy::automatic);
        m_cls.def_property_readonly("dtype_weight",
                                    &tv::gemm::ConvAlgoDesp::dtype_weight,
                                    pybind11::return_value_policy::automatic);
        m_cls.def_property_readonly("dtype_output",
                                    &tv::gemm::ConvAlgoDesp::dtype_output,
                                    pybind11::return_value_policy::automatic);
        m_cls.def("supported", &tv::gemm::ConvAlgoDesp::supported,
                  pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
                  pybind11::arg("C"), pybind11::arg("K"),
                  pybind11::arg("mask_width"),
                  pybind11::return_value_policy::automatic);
        m_cls.def("query_conv_workspace_size",
                  &tv::gemm::ConvAlgoDesp::query_conv_workspace_size,
                  pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
                  pybind11::arg("split_k_slices"), pybind11::arg("kv"),
                  pybind11::return_value_policy::automatic);
        m_cls.def("supported_ldx_conv",
                  &tv::gemm::ConvAlgoDesp::supported_ldx_conv,
                  pybind11::arg("ldi"), pybind11::arg("ldw"), pybind11::arg("ldo"),
                  pybind11::return_value_policy::automatic);
        m_cls.def_readwrite("ndim", &tv::gemm::ConvAlgoDesp::ndim);
        m_cls.def_readwrite("op_type", &tv::gemm::ConvAlgoDesp::op_type);
        m_cls.def_readwrite("iter_algo",
                            &tv::gemm::ConvAlgoDesp::iter_algo);
        m_cls.def_readwrite("layout_i", &tv::gemm::ConvAlgoDesp::layout_i);
        m_cls.def_readwrite("layout_w", &tv::gemm::ConvAlgoDesp::layout_w);
        m_cls.def_readwrite("layout_o", &tv::gemm::ConvAlgoDesp::layout_o);
        m_cls.def_readwrite("interleave_i",
                            &tv::gemm::ConvAlgoDesp::interleave_i);
        m_cls.def_readwrite("interleave_w",
                            &tv::gemm::ConvAlgoDesp::interleave_w);
        m_cls.def_readwrite("interleave_o",
                            &tv::gemm::ConvAlgoDesp::interleave_o);
        m_cls.def_readwrite("mask_sparse",
                            &tv::gemm::ConvAlgoDesp::mask_sparse);
        m_cls.def_readwrite("increment_k_first",
                            &tv::gemm::ConvAlgoDesp::increment_k_first);
        m_cls.def_readwrite("increment_k_first",
                            &tv::gemm::ConvAlgoDesp::increment_k_first);
        m_cls.def_readwrite("is_int8_inference",
                            &tv::gemm::ConvAlgoDesp::is_int8_inference);
        m_cls.def_readwrite("dynamic_mask",
                            &tv::gemm::ConvAlgoDesp::dynamic_mask);
        """)
        return code

    @pccm.static_function
    def bind_conv_params(self):
        code = pccm.code()
        code.arg("module_", "pybind11::module_")
        code.raw("""
        pybind11::class_<tv::gemm::ConvParams> m_cls(module_, "ConvParams");
        m_cls.def(pybind11::init<int, tv::gemm::ConvOpType, tv::CUDAKernelTimer>(),
                  pybind11::arg("ndim"), pybind11::arg("op_type"),
                  pybind11::arg("timer") = tv::CUDAKernelTimer(false));
        m_cls.def_readwrite("conv_algo_desp",
                            &tv::gemm::ConvParams::conv_algo_desp);
        m_cls.def_readwrite("input", &tv::gemm::ConvParams::input);
        m_cls.def_readwrite("weight", &tv::gemm::ConvParams::weight);
        m_cls.def_readwrite("output", &tv::gemm::ConvParams::output);
        m_cls.def_readwrite("output_add", &tv::gemm::ConvParams::output_add);
        m_cls.def_readwrite("bias", &tv::gemm::ConvParams::bias);
        m_cls.def_readwrite("scale", &tv::gemm::ConvParams::scale);

        m_cls.def_readwrite("split_k_slices",
                            &tv::gemm::ConvParams::split_k_slices);
        m_cls.def_readwrite("padding", &tv::gemm::ConvParams::padding);
        m_cls.def_readwrite("stride", &tv::gemm::ConvParams::stride);
        m_cls.def_readwrite("dilation", &tv::gemm::ConvParams::dilation);
        m_cls.def_readwrite("alpha", &tv::gemm::ConvParams::alpha);
        m_cls.def_readwrite("beta", &tv::gemm::ConvParams::beta);
        m_cls.def_readwrite("act_alpha", &tv::gemm::ConvParams::act_alpha);
        m_cls.def_readwrite("act_beta", &tv::gemm::ConvParams::act_beta);
        m_cls.def_readwrite("act_type", &tv::gemm::ConvParams::act_type);

        m_cls.def_readwrite("mask_width", &tv::gemm::ConvParams::mask_width);
        m_cls.def_readwrite("mask_filter", &tv::gemm::ConvParams::mask_filter);
        m_cls.def_readwrite("reverse_mask", &tv::gemm::ConvParams::reverse_mask);
        m_cls.def_readwrite("verbose", &tv::gemm::ConvParams::verbose);
        m_cls.def_readwrite("timer", &tv::gemm::ConvParams::timer);
        m_cls.def_readwrite("workspace", &tv::gemm::ConvParams::workspace);
        m_cls.def_readwrite("mask", &tv::gemm::ConvParams::mask);
        m_cls.def_readwrite("mask_argsort", &tv::gemm::ConvParams::mask_argsort);
        m_cls.def_readwrite("indices", &tv::gemm::ConvParams::indices);
        m_cls.def_readwrite("mask_output", &tv::gemm::ConvParams::mask_output);
        m_cls.def_readwrite("stream", &tv::gemm::ConvParams::stream);
        m_cls.def_readwrite("nvrtc_params", &tv::gemm::ConvParams::nvrtc_params);
        m_cls.def_readwrite("mask_int_count", &tv::gemm::ConvParams::mask_int_count);

        """)
        return code

    @pccm.static_function
    def bind_gemm_params(self):
        code = pccm.code()
        code.arg("module_", "pybind11::module_")
        code.raw("""
        pybind11::class_<tv::gemm::GemmParams> m_cls(module_, "GemmParams");
        m_cls.def(pybind11::init<tv::CUDAKernelTimer>(),
                  pybind11::arg("timer") = tv::CUDAKernelTimer(false));
        m_cls.def("check_valid", &tv::gemm::GemmParams::check_valid,
                  pybind11::return_value_policy::automatic);
        m_cls.def_property("a", &tv::gemm::GemmParams::a_get,
                          &tv::gemm::GemmParams::a_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def_property("b", &tv::gemm::GemmParams::b_get,
                          &tv::gemm::GemmParams::b_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def_property("c", &tv::gemm::GemmParams::c_get,
                          &tv::gemm::GemmParams::c_set,
                          pybind11::return_value_policy::automatic);
        m_cls.def_property("d", &tv::gemm::GemmParams::d_get,
                          &tv::gemm::GemmParams::d_set,
                          pybind11::return_value_policy::automatic);

        m_cls.def_readwrite("algo_desp", &tv::gemm::GemmParams::algo_desp);
        m_cls.def_readwrite("split_k_slices",
                            &tv::gemm::GemmParams::split_k_slices);
        m_cls.def_readwrite("workspace", &tv::gemm::GemmParams::workspace);
        m_cls.def_readwrite("a_inds", &tv::gemm::GemmParams::a_inds);
        m_cls.def_readwrite("b_inds", &tv::gemm::GemmParams::b_inds);
        m_cls.def_readwrite("c_inds", &tv::gemm::GemmParams::c_inds);
        m_cls.def_readwrite("alpha", &tv::gemm::GemmParams::alpha);
        m_cls.def_readwrite("beta", &tv::gemm::GemmParams::beta);
        m_cls.def_readwrite("act_alpha", &tv::gemm::GemmParams::act_alpha);
        m_cls.def_readwrite("act_beta", &tv::gemm::GemmParams::act_beta);
        m_cls.def_readwrite("act_type", &tv::gemm::GemmParams::act_type);
        m_cls.def_readwrite("stream", &tv::gemm::GemmParams::stream);
        m_cls.def_readwrite("timer", &tv::gemm::GemmParams::timer);
        m_cls.def_readwrite("nvrtc_params", &tv::gemm::GemmParams::nvrtc_params);
        """)
        return code

    @pccm.pybind.mark_bind_raw(raw_bind_anno=_TENSORVIEW_BIND_CODE_ANNO)
    @pccm.static_function
    def bind_tensorview(self):
        code = pccm.code()
        code.arg("m", "pybind11::module_")
        if not compat.InWindows:
            code.code_after_include = f"""
#if !defined(TV_HARDWARE_ACC_CUDA) || (CUDA_VERSION < 11040)
#include <cxxabi.h>
#endif
            """
        # we remove fill in below code because it depends on libcuda.so.
        code.raw("""
  py::class_<tv::Context, std::shared_ptr<tv::Context>>(m, "Context")
    .def(py::init<>())
    .def("create_cuda_stream", &tv::Context::create_cuda_stream)
    .def("has_cuda_stream", &tv::Context::has_cuda_stream)
    .def("set_cuda_stream", &tv::Context::set_cuda_stream_int)
    .def("synchronize_stream", &tv::Context::synchronize_stream)
    .def("create_apple_metal_context", &tv::Context::create_apple_metal_context)
    .def("create_or_update_metal_context_from_blob", &tv::Context::create_or_update_metal_context_from_blob)

    .def("synchronize", &tv::Context::synchronize)
    .def("has_apple_metal_context", [](tv::Context& ctx){
        return ctx.has_item(tv::ContextType::kAppleMetal);
    })
    .def("cuda_stream_int", &tv::Context::cuda_stream_int);
  
  py::class_<tv::CUDAEvent, std::shared_ptr<tv::CUDAEvent>>(m, "CUDAEvent")
    .def(py::init<std::string>(), py::arg("name") = std::string())
    .def("record", &tv::CUDAEvent::record, py::arg("stream"))
    .def("stream_wait_me", &tv::CUDAEvent::stream_wait_me, py::arg("stream"), py::arg("flag") = 0)
    .def("sync", &tv::CUDAEvent::sync)
    .def_static("duration", &tv::CUDAEvent::duration, py::arg("start"), py::arg("stop"))
    .def_static("sync_and_duration", &tv::CUDAEvent::sync_and_duration, py::arg("start"), py::arg("stop"));

  py::class_<tv::CPUEvent, std::shared_ptr<tv::CPUEvent>>(m, "CPUEvent")
    .def(py::init<std::string>(), py::arg("name") = std::string())
    .def("record", &tv::CPUEvent::record, py::arg("stream"))
    .def("stream_wait_me", &tv::CPUEvent::stream_wait_me, py::arg("stream"), py::arg("flag") = 0)
    .def("sync", &tv::CPUEvent::sync)
    .def_static("duration", &tv::CPUEvent::duration, py::arg("start"), py::arg("stop"))
    .def_static("sync_and_duration", &tv::CPUEvent::sync_and_duration, py::arg("start"), py::arg("stop"));

  py::class_<tv::CUDAKernelTimer, std::shared_ptr<tv::CUDAKernelTimer>>(m, "CUDAKernelTimer")
    .def(py::init<bool>(), py::arg("enable"))
    .def("push", &tv::CUDAKernelTimer::push, py::arg("name"))
    .def("pop", &tv::CUDAKernelTimer::pop)
    .def("record", &tv::CUDAKernelTimer::record, py::arg("name"), py::arg("stream") = 0)
    .def("insert_pair", &tv::CUDAKernelTimer::insert_pair, py::arg("name"), py::arg("start"), py::arg("stop"))
    .def("has_pair", &tv::CUDAKernelTimer::has_pair, py::arg("name"))
    .def("sync_all_event", &tv::CUDAKernelTimer::sync_all_event)
    .def_property_readonly("enable", &tv::CUDAKernelTimer::enable)
    .def("get_pair_duration", &tv::CUDAKernelTimer::get_pair_duration, py::arg("name"))
    .def("get_all_pair_duration", &tv::CUDAKernelTimer::get_all_pair_duration);

  py::class_<tv::NVRTCProgram, std::shared_ptr<tv::NVRTCProgram>> nvrtc_prog_m(m, "NVRTCProgram");

  nvrtc_prog_m.def(py::init(&tv::NVRTCProgram::create), py::arg("code"), py::arg("headers") = std::unordered_map<std::string, std::string>{}, 
         py::arg("opts") = std::vector<std::string>{}, py::arg("program_name") = std::string("kernel.cu"),
         py::arg("name_exprs") = std::vector<std::string>{})
    .def("ptx", &tv::NVRTCProgram::ptx)
    .def("cubin", &tv::NVRTCProgram::cubin)
    .def("get_predefined_lowered_name_map", &tv::NVRTCProgram::get_predefined_lowered_name_map)
    .def("to_string", &tv::NVRTCProgram::to_string)
    .def_static("from_string", &tv::NVRTCProgram::from_string, py::arg("json_string"))
    .def("to_binary", [](const tv::NVRTCProgram& prog, int serial_type){
      auto buffer = prog.to_binary(static_cast<tv::NVRTCProgram::SerializationType>(serial_type));
      return py::bytes(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    }, py::arg("serial_type"))
    .def_static("from_binary", [](pybind11::bytes buffer){
        py::buffer_info info(py::buffer(buffer).request());
        const uint8_t *data = reinterpret_cast<const uint8_t *>(info.ptr);
        size_t length = static_cast<size_t>(info.size);
        return tv::NVRTCProgram::from_binary(data, length);
    }, py::arg("buffer"))

    .def("compile_log", &tv::NVRTCProgram::compile_log)
    .def("get_lowered_name", &tv::NVRTCProgram::get_lowered_name);
  py::enum_<tv::NVRTCProgram::SerializationType>(nvrtc_prog_m, "SerializationType")
      .value("kSource", tv::NVRTCProgram::SerializationType::kSource)
      .value("kPTX", tv::NVRTCProgram::SerializationType::kPTX)
      .value("kCuBin", tv::NVRTCProgram::SerializationType::kCuBin)
      .export_values();


  py::class_<tv::NVRTCModule, std::shared_ptr<tv::NVRTCModule>> nvrtc_m(m, "NVRTCModule");
  nvrtc_m.def(py::init(&tv::NVRTCModule::create), py::arg("code"), py::arg("headers") = std::unordered_map<std::string, std::string>{}, 
         py::arg("opts") = std::vector<std::string>{}, py::arg("program_name") = std::string("kernel.cu"),
         py::arg("name_exprs") = std::vector<std::string>{},
         py::arg("cudadevrt_path") = std::string(""))
    .def(py::init(&tv::NVRTCModule::from_program), py::arg("prog"), py::arg("cudadevrt_path") = std::string())
    .def("load", &tv::NVRTCModule::load)
    .def_property_readonly("program", &tv::NVRTCModule::get_program)
    .def("get_lowered_name", &tv::NVRTCModule::get_lowered_name)
    .def("get_kernel_attributes", &tv::NVRTCModule::get_kernel_attributes, py::arg("name"))
    .def("set_max_dynamic_shared_size_bytes", &tv::NVRTCModule::set_max_dynamic_shared_size_bytes, py::arg("name"), py::arg("size"))
    .def("set_preferred_smem_carveout", &tv::NVRTCModule::set_preferred_smem_carveout, py::arg("name"), py::arg("carveout"))
    .def("run_kernel", &tv::NVRTCModule::run_kernel);

  py::class_<tv::MetalModule, std::shared_ptr<tv::MetalModule>> metal_rtc_m(m, "MetalModule");
  metal_rtc_m.def(py::init<tv::Tensor>(), py::arg("binary"))
    .def(py::init<std::string, std::unordered_map<std::string, std::string>, bool>(), py::arg("code"), py::arg("preprocessorMacros"), py::arg("fastMathEnabled") = true)
    .def("run_kernel", &tv::MetalModule::run_kernel, py::arg("name"), py::arg("blocks"), 
      py::arg("threads"), py::arg("smem_size"), py::arg("ctx"),
      py::arg("args"), py::arg("use_nonuniform_threadgroup") = true);

  py::enum_<tv::NVRTCModule::ArgType>(nvrtc_m, "ArgType")
      .value("kTensor", tv::NVRTCModule::ArgType::kTensor)
      .value("kArray", tv::NVRTCModule::ArgType::kArray)
      .value("kTensorView", tv::NVRTCModule::ArgType::kTensorView)
      .value("kScalar", tv::NVRTCModule::ArgType::kScalar)
      .value("kConstant", tv::NVRTCModule::ArgType::kConstant)
      .value("kDevicePointer", tv::NVRTCModule::ArgType::kDevicePointer)
      .export_values();

  py::class_<tv::Tensor, std::shared_ptr<tv::Tensor>>(m, "Tensor")
    .def(py::init([](std::vector<int64_t> shape, int dtype, int device, bool pinned, bool managed) {
        return tv::Tensor(tv::TensorShape(shape), tv::DType(dtype), device, pinned, managed);
    }), py::arg("shape"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false)
    .def(py::init([]() {
        return tv::Tensor();
    }))
    .def("clone", [](const tv::Tensor& ten, bool pinned, bool use_cpu_copy){
      return ten.clone(pinned, use_cpu_copy);
    }, py::arg("pinned") = false, py::arg("use_cpu_copy") = false)
    .def("clone_whole_storage", &tv::Tensor::clone_whole_storage, py::arg("ctx") = tv::Context())
    .def("zero_whole_storage_", &tv::Tensor::zero_whole_storage_)
    .def("view", [](const tv::Tensor& ten, std::vector<int64_t> shape){
      return ten.view(tv::TensorShape(shape));
    })
    .def("__getitem__", &tv::Tensor::operator[])
    .def("__getitem__", [](const tv::Tensor& src, const pybind11::slice& key){
      namespace py = pybind11;
      Py_ssize_t start;
      Py_ssize_t stop;
      Py_ssize_t step;
      PySlice_Unpack(key.ptr(), &start, &stop, &step);
      PySliceObject* slice_key = reinterpret_cast<PySliceObject*>(key.ptr());
      bool start_is_none = py::detail::PyNone_Check(slice_key->start);
      bool end_is_none = py::detail::PyNone_Check(slice_key->stop);
      bool step_is_none = py::detail::PyNone_Check(slice_key->step);
      if (step_is_none){
        step = 1;
      }
      return src.slice(0, start, stop, step, start_is_none, end_is_none);
    })
    .def("__getitem__", [](const tv::Tensor& src, const pybind11::tuple& key_list){

      namespace py = pybind11;
      int64_t dim = 0;
      // tv::Tensor prev_dim_result;
      int64_t specified_dims = 0;

      for (auto& key : key_list){
        if (py::detail::PyEllipsis_Check(key.ptr()) || key.is_none()){
          ++specified_dims;
        }
      }
      tv::Tensor result = src;
      for (auto& key : key_list){
        if (py::detail::PyEllipsis_Check(key.ptr())){
          dim += src.ndim() - specified_dims;
        }else if (py::isinstance<py::slice>(key)){
          Py_ssize_t start;
          Py_ssize_t stop;
          Py_ssize_t step;
          PySlice_Unpack(key.ptr(), &start, &stop, &step);
          PySliceObject* slice_key = reinterpret_cast<PySliceObject*>(key.ptr());
          bool start_is_none = py::detail::PyNone_Check(slice_key->start);
          bool end_is_none = py::detail::PyNone_Check(slice_key->stop);
          bool step_is_none = py::detail::PyNone_Check(slice_key->step);
          if (step_is_none){
            step = 1;
          }
          result = result.slice(dim, start, stop, step, start_is_none, end_is_none);
          ++dim;
        }else if (py::isinstance<py::int_>(key)){
          int select_idx = py::cast<int64_t>(key);
          result = result.select(dim, select_idx);
          // result = result.slice(dim, select_idx, select_idx + 1, 1, false, false).squeeze(dim);
        }else if (key.is_none()){
          result = result.unsqueeze(dim);
          dim++;
        }else{
          TV_THROW_INVALID_ARG("tv::Tensor only support .../None/int/slice slicing");
        }
      }
      return result;
    })

    .def("as_strided", [](const tv::Tensor& ten, std::vector<int64_t> shape, std::vector<int64_t> stride, int64_t storage_byte_offset){
      return ten.as_strided(shape, stride, storage_byte_offset);
    }, py::arg("shape"), py::arg("stride"), py::arg("storage_byte_offset") = 0)
    .def("slice_first_axis", &tv::Tensor::slice_first_axis)
    .def("transpose", &tv::Tensor::transpose)
    .def_property_readonly("T", [](const tv::Tensor& ten){
      TV_ASSERT_INVALID_ARG(ten.ndim() == 2, "you can only use .T with 2d tensor.");
      return ten.transpose(0, 1);
    })
    .def("slice_axis", [](const tv::Tensor& ten, int dim, py::object start, py::object stop, py::object step){
      bool start_is_none = start.is_none();
      bool stop_is_none = stop.is_none();
      bool step_is_none = step.is_none();
      int64_t start_val = start_is_none ? 0 : py::cast<int64_t>(start);
      int64_t stop_val = stop_is_none ? 0 : py::cast<int64_t>(stop);
      int64_t step_val = step_is_none ? 1 : py::cast<int64_t>(step); 
      return ten.slice(dim, start_val, stop_val, step_val, start_is_none, stop_is_none);
    }, py::arg("dim"), py::arg("start"), py::arg("stop"), py::arg("step") = py::none())
    .def("select", &tv::Tensor::select)

    // .def("slice_axis", &tv::Tensor::slice)
    .def("dim", &tv::Tensor::dim)
    .def("squeeze", py::overload_cast<>(&tv::Tensor::squeeze, py::const_))
    .def("squeeze", py::overload_cast<int>(&tv::Tensor::squeeze, py::const_))
    .def("zero_", &tv::Tensor::zero_, py::arg("ctx") = tv::Context())
    .def("fill_int_", py::overload_cast<int64_t, tv::Context>(&tv::Tensor::fill_), py::arg("val"), py::arg("ctx") = tv::Context())
    .def("fill_float_", py::overload_cast<float, tv::Context>(&tv::Tensor::fill_), py::arg("val"), py::arg("ctx") = tv::Context())
    .def("copy_", [](tv::Tensor& t, const tv::Tensor& other, tv::Context ctx) -> void{
      t.copy_(other, ctx);
    }, py::arg("other"), py::arg("ctx") = tv::Context())
    .def("copy_storage_", [](tv::Tensor& t, const tv::Tensor& other, tv::Context ctx) -> void{
      t.copy_storage_(other, ctx);
    }, py::arg("other"), py::arg("ctx") = tv::Context())
    .def("copy_2d_pitched_", [](tv::Tensor& t, const tv::Tensor& other, tv::Context ctx) -> void{
      t.copy_2d_pitched_(other, ctx);
    }, py::arg("other"), py::arg("ctx") = tv::Context())

    .def("cpu", py::overload_cast<tv::Context>(&tv::Tensor::cpu, py::const_), py::arg("ctx") = tv::Context())
    .def("numpy", [](const tv::Tensor& ten){
      TV_ASSERT_RT_ERR(ten.device() == -1 || (ten.device() == 0 && ten.managed()), "you need to call .cpu() before convert cuda tensor to numpy");
      return tv::tensor2array(ten);
    })
    .def("type_view", [](const tv::Tensor& ten, int dtype){
      return ten.type_view(tv::DType(dtype));
    })
    .def("type_view", [](const tv::Tensor& ten, int dtype, std::vector<int64_t> shape){
      return ten.type_view(tv::DType(dtype), tv::TensorShape(shape));
    })
    .def_property_readonly("size", py::overload_cast<>(&tv::Tensor::size, py::const_))
    .def_property_readonly("itemsize", &tv::Tensor::itemsize)
    .def_property_readonly("ndim", &tv::Tensor::ndim)
    .def_property_readonly("device", &tv::Tensor::device)

    .def("pinned", &tv::Tensor::pinned)
    .def("is_contiguous", &tv::Tensor::is_contiguous)
    .def("is_col_major_matrix", &tv::Tensor::is_col_major_matrix)
    .def("is_readonly", &tv::Tensor::is_readonly)
    .def("get_readonly", &tv::Tensor::get_readonly)
    .def("byte_offset", &tv::Tensor::byte_offset)
    .def("storage_bytesize", &tv::Tensor::storage_size)
    .def("bytesize", &tv::Tensor::raw_size)
    .def("unsqueeze", &tv::Tensor::unsqueeze)
    .def("empty", &tv::Tensor::empty)
    .def("byte_pointer", [](const tv::Tensor& ten){
      return reinterpret_cast<std::uintptr_t>(ten.raw_data());
    })
    .def("gpu_address", &tv::Tensor::gpu_address)

#if defined(TV_ENABLE_HARDWARE_ACC)
    .def("cuda", py::overload_cast<tv::Context>(&tv::Tensor::cuda, py::const_), py::arg("ctx") = tv::Context())
#endif
#if (PYBIND11_VERSION_MAJOR > 2 || (PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 6))
    .def("get_memoryview", [](tv::Tensor& ten) {
        TV_ASSERT_RT_ERR(ten.device() == -1 || (ten.device() == 0 && ten.managed()), "you need to call .cpu() before convert cuda tensor to numpy");
        return py::memoryview::from_memory(
            ten.raw_data(),
            ten.raw_size()
        );
    })
#endif
    .def_property_readonly("dtype", [](const tv::Tensor& t){
      return int(t.dtype());
    })
    .def_property_readonly("shape", [](const tv::Tensor& t){
      auto& shape = t.shape();
      return std::vector<int64_t>(shape.begin(), shape.end());
    })
    .def_property_readonly("stride", [](const tv::Tensor& t){
      auto& shape = t.strides();
      return std::vector<int64_t>(shape.begin(), shape.end());
    });
  // from_blob is used for pytorch.
  m.def("from_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, std::vector<int64_t> stride, int dtype, int device, int storage_offset){
      return tv::from_blob(reinterpret_cast<void*>(ptr_uint), shape, stride, tv::DType(dtype), device, storage_offset);
  }, py::arg("ptr"), py::arg("shape"), py::arg("stride"), py::arg("dtype"), py::arg("device"), py::arg("storage_offset") = 0); 
  m.def("from_const_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, std::vector<int64_t> stride, int dtype, int device, int storage_offset){
      return tv::from_blob(reinterpret_cast<const void*>(ptr_uint), shape, stride, tv::DType(dtype), device, storage_offset);
  }, py::arg("ptr"), py::arg("shape"), py::arg("stride"), py::arg("dtype"), py::arg("device"), py::arg("storage_offset") = 0); 
  m.def("from_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, int dtype, int device, int storage_offset){
      return tv::from_blob(reinterpret_cast<void*>(ptr_uint), shape, tv::DType(dtype), device, storage_offset);
  }, py::arg("ptr"), py::arg("shape"), py::arg("dtype"), py::arg("device"), py::arg("storage_offset") = 0); 
  m.def("from_const_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, int dtype, int device, int storage_offset){
      return tv::from_blob(reinterpret_cast<const void*>(ptr_uint), shape, tv::DType(dtype), device, storage_offset);
  }, py::arg("ptr"), py::arg("shape"), py::arg("dtype"), py::arg("device"), py::arg("storage_offset") = 0); 
  m.def("zeros", [](std::vector<int64_t> shape, int dtype, int device, bool pinned, bool managed){
    return tv::zeros(shape, tv::DType(dtype), device, pinned, managed);
  }, py::arg("shape"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
  m.def("empty", [](std::vector<int64_t> shape, int dtype, int device, bool pinned, bool managed){
    return tv::empty(shape, tv::DType(dtype), device, pinned, managed);
  }, py::arg("shape"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
  m.def("full_int", [](std::vector<int64_t> shape, int64_t val, int dtype, int device, bool pinned, bool managed){
    return tv::full(shape, val, tv::DType(dtype), device, pinned, managed);
  }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
  m.def("full_float", [](std::vector<int64_t> shape, float val, int dtype, int device, bool pinned, bool managed){
    return tv::full(shape, val, tv::DType(dtype), device, pinned, managed);
  }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
#if defined(TV_HARDWARE_ACC_CUDA)
  m.def("zeros_managed", [](std::vector<int64_t> shape, int dtype){
    return tv::zeros(shape, tv::DType(dtype), 0, false, true);
  }, py::arg("shape"), py::arg("dtype") = 0); 
  // m.def("full_int_managed", [](std::vector<int64_t> shape, int64_t val, int dtype){
  //   return tv::full(shape, val, tv::DType(dtype), 0, false, true);
  // }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0); 
  // m.def("full_float_managed", [](std::vector<int64_t> shape, float val, int dtype){
  //   return tv::full(shape, val, tv::DType(dtype), 0, false, true);
  // }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0); 
#endif
  m.def("get_compute_capability", [](int index){
    if (index == -1){
      checkCudaErrors(cudaGetDevice(&index));
    }
#if defined(TV_HARDWARE_ACC_CUDA)
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, index));
    return std::make_tuple(prop.major, prop.minor);
#else 
    return std::make_tuple(-1, -1);
#endif
  }, py::arg("index") = -1); 

  m.def("from_numpy", [](py::array arr){
    return tv::array2tensor(arr);
  }); 
  m.def("tvdtype_bitsize", [](int dtype){
    return tv::bit_size(tv::DType(dtype));
  }); 
  m.def("tvdtype_itemsize", [](int dtype){
    return tv::bit_size(tv::DType(dtype)) / 8;
  }); 
  m.def("check_cuda_error", [](){
#if defined(TV_HARDWARE_ACC_CUDA)
    TV_CHECK_CUDA_ERR_V2("error");
#endif
  }); 

  m.def("cat_first_axis", &tv::cat_first_axis);
  m.def("is_cpu_only", [](){
#if defined(TV_ENABLE_HARDWARE_ACC)
    return false;
#else
    return true;
#endif
  }); 
        """)
        if compat.InLinux or compat.InMacOS:
            code.raw("""
            m.def("cufilt", [](std::string name){
              int status;
              std::shared_ptr<char> realname = std::shared_ptr<char>(abi::__cxa_demangle(name.c_str(), 0, 0, &status), std::free);
              if (status != 0){
                return std::string();
              }
              return std::string(realname.get());
            }, py::arg("name")); 
            """)
        elif compat.InWindows:
            code.raw("""
            m.def("cufilt", [](std::string name){
              return std::string();
            }, py::arg("name")); 
            """)
        else:
            code.raw("""
            m.def("cufilt", [](std::string name){
              return std::string();
            }, py::arg("name")); 
            """)
        code.raw("""

  bind_gemm_algo_desp(m);
  bind_conv_algo_desp(m);
  bind_conv_params(m);
  bind_nvrtc_params(m);
  bind_enums(m);
  bind_gemm_params(m);
  m.def("run_nvrtc_conv_kernel", &run_nvrtc_conv_kernel, py::arg("params"));
  m.def("run_nvrtc_gemm_kernel", &run_nvrtc_gemm_kernel, py::arg("params"));
        """)
        return code
