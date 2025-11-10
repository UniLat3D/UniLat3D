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

import pccm 
from cumm.constants import CUMM_MAXIMUM_NVRTC_CONV_NDIM
from cumm.gemm.constants import NVRTCMode


def nvrtc_conv_template(code: pccm.FunctionCode):
    code.raw(f"""
    // auto rtxtimer = tv::CPUTimer<>();
    // auto ev1 = tv::CUDAEvent("wtf").record();
    static_assert({CUMM_MAXIMUM_NVRTC_CONV_NDIM} == CUMM_MAXIMUM_NVRTC_CONV_NDIM, "error");
    int groups = 1;
    bool found = false;
    auto& algo_desp = params.conv_algo_desp;
    auto dacc = tv::DType(algo_desp.dacc);
    auto dcomp = tv::DType(algo_desp.dcomp);
    tv::gemm::ConvOpType op_type = static_cast<tv::gemm::ConvOpType>(algo_desp.op_type);
    int split_k_slices = params.split_k_slices;
    auto& workspace = params.workspace;
    auto& input = params.input;
    auto& weight = params.weight;
    auto& output = params.output;
    auto& output_add = params.output_add;

    auto& bias = params.bias;
    auto& scale = params.scale;
    auto& indices = params.indices;
    auto& mask = params.mask;
    auto& mask_argsort = params.mask_argsort;
    auto& mask_output = params.mask_output;
    auto& padding = params.padding;
    auto& stride = params.stride;
    auto& dilation = params.dilation;
    auto& mask_width = params.mask_width;
    auto& evtimer = params.timer;
    int io_dim = algo_desp.mask_sparse ? 2 : algo_desp.ndim + 2;
    int weight_ndim = algo_desp.mask_sparse ? 3 : algo_desp.ndim + 2;
    int dim_start =  algo_desp.layout_w == tv::gemm::ConvLayoutType::kChannelFirst ? 2 : 1;
    int ndim = algo_desp.ndim;
    TV_ASSERT_RT_ERR(input.ndim() == io_dim, "error");
    TV_ASSERT_RT_ERR(weight.ndim() == weight_ndim, "error");
    TV_ASSERT_RT_ERR(output.ndim() == io_dim, "error");
    if (!(algo_desp.split_k_serial() || algo_desp.split_k_parallel()) && split_k_slices > 1){{
        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
    }}
    int kernel_volume = 1;
    int N = input.dim(0);
    int K = weight.dim(0);
    int C = algo_desp.layout_i == tv::gemm::ConvLayoutType::kChannelFirst ? input.dim(1) : input.dim(io_dim - 1);
    int K2 = algo_desp.layout_o == tv::gemm::ConvLayoutType::kChannelFirst ? output.dim(1) : output.dim(io_dim - 1);
    TV_ASSERT_RT_ERR(K2 == K, "error");
    tv::array<int, 3> mnk;
    auto inv_indices = tv::gemm::gemm_abc_012_to_iwo(tv::gemm::ConvOpType(algo_desp.op_type));
    std::array<tv::Tensor, 3> conv_inputs{{input, weight, output}};
    auto& a_ten = conv_inputs[inv_indices[0]];
    auto& b_ten = conv_inputs[inv_indices[1]];
    auto& c_ten = conv_inputs[inv_indices[2]];
    auto& nvrtc_params = params.nvrtc_params;
    tv::gemm::ConvNVRTCParams kernel_params;
    tv::gemm::SparseConvNVRTCParams sp_kernel_params;

    kernel_params.ptr_A = a_ten.const_raw_data();
    kernel_params.ptr_B = b_ten.const_raw_data();
    kernel_params.ptr_C = c_ten.raw_data();
    if (!algo_desp.is_int8_inference){{
        TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
    }}else{{
        TV_ASSERT_RT_ERR(!bias.empty() && !scale.empty(), "int8 inference must have both scale and bias");
    }}
    if (output_add.empty()){{
        kernel_params.ptr_D = algo_desp.is_int8_inference ? c_ten.const_raw_data() : (bias.empty() ? c_ten.const_raw_data() : bias.const_raw_data());
    }}else{{
        TV_ASSERT_RT_ERR(output_add.dtype() == output.dtype() && output_add.shape() == output.shape(),
            "output and output_add must have same dtype and shape", output_add.dtype(), output.dtype(),
            output_add.shape(), output.shape());
        kernel_params.ptr_D = output_add.const_raw_data();
    }}
    kernel_params.bias_pointer = bias.empty() ? nullptr : bias.const_raw_data();
    kernel_params.scale_pointer = scale.empty() ? nullptr : scale.const_raw_data();
    kernel_params.alpha = params.alpha;
    kernel_params.beta = params.beta;
    kernel_params.ndim = ndim;
    kernel_params.d_is_bias = !bias.empty();
    kernel_params.act_alpha = params.act_alpha;
    kernel_params.act_beta = params.act_beta;
    kernel_params.act_type = static_cast<int>(params.act_type);


    sp_kernel_params.ptr_A = kernel_params.ptr_A;
    sp_kernel_params.ptr_B = kernel_params.ptr_B;
    sp_kernel_params.ptr_C = kernel_params.ptr_C;
    sp_kernel_params.ptr_D = kernel_params.ptr_D;
    sp_kernel_params.scale_pointer = kernel_params.scale_pointer;
    sp_kernel_params.bias_pointer = kernel_params.bias_pointer;

    sp_kernel_params.alpha = kernel_params.alpha;
    sp_kernel_params.beta = kernel_params.beta;
    sp_kernel_params.ndim = kernel_params.ndim;
    sp_kernel_params.d_is_bias = !bias.empty();
    sp_kernel_params.act_alpha = kernel_params.act_alpha;
    sp_kernel_params.act_beta = kernel_params.act_beta;
    sp_kernel_params.act_type = kernel_params.act_type;

    constexpr int int_max = std::numeric_limits<int32_t>::max();

    if (algo_desp.mask_sparse){{
        if (algo_desp.op_type == tv::gemm::ConvOpType::kBackwardWeight){{
            TV_ASSERT_RT_ERR(mask_width > 0 && mask_width % algo_desp.tile_shape[2] == 0, "error");
        }}
        TV_ASSERT_RT_ERR(!indices.empty(), "error");
        TV_ASSERT_RT_ERR(!mask.empty(), "error");
        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
        kernel_volume = weight.dim(dim_start);
        tv::check_shape(indices, {{kernel_volume, -1}});
        N = indices.dim(1);
        if (algo_desp.op_type == tv::gemm::ConvOpType::kBackwardWeight){{
            TV_ASSERT_RT_ERR(N == output.dim(0), "error");
            # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_b) / 8 < int_max, 
            #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
            # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
            #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");

        }}else if (algo_desp.op_type == tv::gemm::ConvOpType::kForward){{
            TV_ASSERT_RT_ERR(N == output.dim(0), "error");
            # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
            #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
        }}else{{
                # TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
                #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                TV_ASSERT_RT_ERR(N == input.dim(0), "error");

        }}
        mnk = tv::gemm::implicit_gemm_mnk(tv::gemm::ConvOpType(algo_desp.op_type), N, C, K, kernel_volume, -1, -1, true);
    }}else{{
        TV_ASSERT_RT_ERR(algo_desp.ndim <= {CUMM_MAXIMUM_NVRTC_CONV_NDIM}, "ndim too large for nvrtc");
        tv::array<int, {CUMM_MAXIMUM_NVRTC_CONV_NDIM}> ksize, padding_arr, stride_arr, dilation_arr, input_dims, output_dims;
        TV_ASSERT_RT_ERR(ndim == padding.size() && ndim == stride.size() && ndim == dilation.size(), "error");
        for (int i = dim_start; i < dim_start + ndim; ++i){{
            ksize[i - dim_start] = weight.dim(i);
            input_dims[i - dim_start] = input.dim(i);
            output_dims[i - dim_start] = output.dim(i);
        }}
        for (int i = 0; i < ndim; ++i){{
            padding_arr[i] = padding[i];
            stride_arr[i] = stride[i];
            dilation_arr[i] = dilation[i];
        }}
        kernel_volume = 1;
        int in_prod = 1;
        int out_prod = 1;
        for (int i = 0; i < ndim; ++i){{
            kernel_volume *= ksize[i];
            in_prod *= input_dims[i];
            out_prod *= output_dims[i];
        }}
        mnk = tv::gemm::implicit_gemm_mnk(tv::gemm::ConvOpType(algo_desp.op_type), N, C, K, kernel_volume, in_prod, out_prod, false);
        kernel_params.input_dims = input_dims;
        kernel_params.output_dims = output_dims;
        kernel_params.ksize = ksize;
        kernel_params.padding = padding_arr;
        kernel_params.stride = stride_arr;
        kernel_params.dilation = dilation_arr;
    }}
    TV_ASSERT_RT_ERR(algo_desp.supported(mnk[0], mnk[1], mnk[2], C, K, mask_width), "error");

    int workspace_size = algo_desp.query_conv_workspace_size(mnk[0], mnk[1], mnk[2], split_k_slices, kernel_volume);

    auto ctx = tv::Context();
    ctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(params.stream));
    if (workspace_size > 0){{
        if (!workspace.empty()){{
            workspace.zero_(ctx);
            TV_ASSERT_RT_ERR(workspace.nbytes() >= workspace_size, 
                "workspace at least", workspace_size, "bytes.");
        }}else{{
            workspace = tv::empty({{workspace_size}}, tv::uint8, 0);
            workspace.zero_(ctx);
        }}
    }}
    void* workspace_ptr = nullptr;
    if (!workspace.empty()){{
        workspace_ptr = workspace.raw_data();
    }}

    if (nvrtc_params.cumodule){{
        TV_ASSERT_RT_ERR(!nvrtc_params.kernel_name.empty(), "you must provide name of your kernel");
        kernel_params.N = N;
        kernel_params.C = C;
        kernel_params.K = K;
        kernel_params.kernel_volume = kernel_volume;
        kernel_params.mode = static_cast<int>(tv::gemm::ConvMode::kCrossCorrelation);
        kernel_params.split_k_slices = split_k_slices;
        kernel_params.groups = groups;
        kernel_params.workspace = workspace_ptr;

        sp_kernel_params.N = kernel_params.N;
        sp_kernel_params.C = kernel_params.C;
        sp_kernel_params.K = kernel_params.K;
        sp_kernel_params.kernel_volume = kernel_params.kernel_volume;
        sp_kernel_params.mode = kernel_params.mode;
        sp_kernel_params.split_k_slices = kernel_params.split_k_slices;
        sp_kernel_params.groups = kernel_params.groups;
        sp_kernel_params.workspace = kernel_params.workspace;

        tv::array<int, 3> grid_dims_arr;
        if (algo_desp.mask_sparse){{
            sp_kernel_params.mask_out_ptr = mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>();
            sp_kernel_params.mask_width = mask_width;
            sp_kernel_params.mask_ptr = mask.data_ptr<const uint32_t>();
            sp_kernel_params.reverse_mask = params.reverse_mask;
            sp_kernel_params.mask_filter = params.mask_filter;
            sp_kernel_params.indice_ptr = indices.data_ptr<const int32_t>();
            sp_kernel_params.mask_argsort_ptr = mask_argsort.data_ptr<const int32_t>();

            grid_dims_arr = tv::gemm::get_spconv_logical_tile_count(mnk[0], mnk[1], mnk[2], 
                            algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices, kernel_volume, algo_desp.op_type);
        }}else{{
            grid_dims_arr = tv::gemm::get_logical_tile_count(mnk[0], mnk[1], mnk[2], 
                            algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices);

        }}
        dim3 grid_dims;
        grid_dims.x = grid_dims_arr[0];
        grid_dims.y = grid_dims_arr[1];
        grid_dims.z = grid_dims_arr[2];
        if (algo_desp.op_type == tv::gemm::ConvOpType::kBackwardWeight && algo_desp.mask_sparse){{
            int num_reduced_mask = tv::div_up(sp_kernel_params.N, sp_kernel_params.mask_width);
            TV_ASSERT_RT_ERR(mask.dim(0) >= num_reduced_mask, "error");
        }}
        std::string algo_name;
        if (evtimer.enable()){{
            algo_name = algo_desp.__repr__();
        }}
        auto kernel = nvrtc_params.cumodule->kernel(nvrtc_params.kernel_name);
        auto& driver = nvrtc_params.cumodule->get_driver_wrapper();
        cudaError_t result;
        if (nvrtc_params.smem_size > 0){{
            if (nvrtc_params.smem_size >= (48 << 10)) {{
                TV_CUDA_RESULT_CHECK(driver.cuDrvFuncSetAttribute(kernel,
                                                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                nvrtc_params.smem_size));
                TV_CUDA_RESULT_CHECK(driver.cuDrvFuncSetAttribute(
                    kernel,
                    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
            }}
        }}
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(params.stream);
        void* kernel_params_ptr;
        if (algo_desp.mask_sparse){{
            kernel_params_ptr = &sp_kernel_params;
        }}else{{
            kernel_params_ptr = &kernel_params;
        }}

        // auto ev2 = tv::CUDAEvent("wtf").record();
        // ev1.sync();
        // ev2.sync();
        // tv::ssprint("prep time", tv::CUDAEvent::duration(ev1, ev2));

        if (nvrtc_params.mode == {NVRTCMode.DynamicParallism.value}){{
            tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
                std::vector<void*> args{{kernel_params_ptr, &grid_dims, &params.stream}};
            TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, 1, 1, 1, 
                1, 1, 1, 0, stream, args.data(), 0));
        }}else if (nvrtc_params.mode == {NVRTCMode.KernelAndCPU.value}){{
            // use kernel-cpu-kernel
            auto init_kernel = nvrtc_params.cumodule->kernel(nvrtc_params.init_kernel_name);
            tv::Tensor temp_data = nvrtc_params.param_storage;
            if (nvrtc_params.param_storage.empty()){{
                temp_data = tv::empty({{nvrtc_params.param_size}}, tv::uint8, 0);
            }}else{{
                TV_ASSERT_RT_ERR(temp_data.nbytes() >= nvrtc_params.param_size, "your params storage too small");
            }}
            void* raw_data_ptr;
            void* temp_data_ptr = temp_data.raw_data();
            tv::Tensor temp_data_cpu = nvrtc_params.param_storage_cpu;

            {{
                tv::CUDAKernelTimerGuard timerguard(algo_name + "/init", evtimer, stream);
                std::vector<void*> args{{kernel_params_ptr, &temp_data_ptr}};
                TV_CUDA_RESULT_CHECK(driver.cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
                if (nvrtc_params.param_storage_cpu.empty()){{
                    temp_data_cpu = temp_data.cpu(ctx);
                }}else{{
                    temp_data_cpu.copy_(temp_data, ctx);
                }}
                // we must sync here because following kernel launch requires cpu data.
                checkCudaErrors(cudaStreamSynchronize(stream));
                raw_data_ptr = temp_data_cpu.raw_data();
            }}
            {{
                tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
                std::vector<void*> args{{raw_data_ptr}};
                TV_CUDA_RESULT_CHECK(driver.cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                    nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
            }}
        }}else if (nvrtc_params.mode == {NVRTCMode.Direct.value}){{
            tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
            std::vector<void*> args{{kernel_params_ptr}};
            TV_CUDA_RESULT_CHECK(driver.cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));

        }}else if (nvrtc_params.mode == {NVRTCMode.ConstantMemory.value}){{
            auto init_kernel = nvrtc_params.cumodule->kernel(nvrtc_params.init_kernel_name);
            tv::Tensor temp_data = nvrtc_params.param_storage;
            if (nvrtc_params.param_storage.empty()){{
                temp_data = tv::empty({{nvrtc_params.param_size}}, tv::uint8, 0);
            }}else{{
                TV_ASSERT_RT_ERR(temp_data.nbytes() >= nvrtc_params.param_size, "your params storage too small");
            }}
            void* temp_data_ptr = temp_data.raw_data();
            {{
                tv::CUDAKernelTimerGuard timerguard(algo_name + "/init", evtimer, stream);
                std::vector<void*> args{{kernel_params_ptr, &temp_data_ptr}};
                TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
            }}
            {{
                tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
                auto ptr = nvrtc_params.cumodule->get_global_ptr(nvrtc_params.constant_name);
                auto constant_ten = tv::from_blob(ptr, {{nvrtc_params.param_size}}, tv::uint8, 0);
                constant_ten.copy_(temp_data, ctx);
                std::vector<void*> args{{}};
                TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                    nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, nullptr, 0));
            }}
        }}else{{
            TV_THROW_RT_ERR("not implemented");
        }}
        TV_CHECK_CUDA_ERR_V2(algo_desp.__repr__(), "error with params", input.shape(), output.shape(), weight.shape());
        return;
    }}

    """)
