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


def nvrtc_gemm_template(code: pccm.FunctionCode):
    code.raw(f"""
    params.check_valid();
    auto& algo_desp = params.algo_desp;
    bool found = false;
    auto dacc = tv::DType(algo_desp.dacc);
    auto dcomp = tv::DType(algo_desp.dcomp);
    auto a = params.a;
    auto b = params.b;
    auto c = params.c;
    auto d = params.d;
    if (d.empty()){{
        d = c; // TODO fix this
    }}
    auto ta = algo_desp.trans_a();
    auto tb = algo_desp.trans_b();
    auto tc = algo_desp.trans_c();
    tv::check_shape(a, {{-1, -1}});
    tv::check_shape(b, {{-1, -1}});
    tv::check_shape(c, {{-1, -1}});
    tv::check_eq_device(a, b, c);
    tv::Tensor a_ten = a;
    tv::Tensor b_ten = b;
    tv::Tensor c_ten = c;
    tv::Tensor d_ten = d;
    auto trans_a = ta;
    auto trans_b = tb;
    auto trans_c = tc;
    if (tc) {{
        trans_a = !trans_a;
        trans_b = !trans_b;
        std::swap(trans_a, trans_b);
        std::swap(a_ten, b_ten);
    }}
    int split_k_slices = params.split_k_slices;
    auto workspace = params.workspace;
    auto a_inds = params.a_inds;
    auto c_inds = params.c_inds;
    auto b_inds = params.b_inds;
    auto& evtimer = params.timer;
    if (!(algo_desp.split_k_serial() || algo_desp.split_k_parallel()) && split_k_slices > 1){{
        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);

    }}
    int m, n, k, k2;
    constexpr int int_max = std::numeric_limits<int32_t>::max();
    if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAC){{
        TV_ASSERT_RT_ERR(!trans_a, "a of shuffle AB must be row major");
        if (!a_inds.empty()){{
            m = a_inds.dim(0);
        }}else{{
            m = a.dim(0);
        }}
        # TV_ASSERT_RT_ERR(int64_t(a.dim(0)) * int64_t(a.dim(1)) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");

        k = a_ten.dim(int(!trans_a));
        k2 = b_ten.dim(int(trans_b));
        n = b_ten.dim(int(!trans_b) );
        if (trans_c){{
            tv::check_shape(c_ten, {{-1, m}});
        }}else{{
            tv::check_shape(c_ten, {{-1, n}});
        }}
    }}else if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAB){{
        TV_ASSERT_RT_ERR(trans_a && !trans_b, "shuffle AB must be nt, i.e. backward weight");
        m = a_ten.dim(int(trans_a));
        k = a_inds.dim(0);
        k2 = b_inds.dim(0);
        n = b_ten.dim(int(!trans_b) );
        # TV_ASSERT_RT_ERR(int64_t(a.dim(0)) * int64_t(a.dim(1)) * tv::bit_size(algo_desp.dtype_a)/ 8 < int_max, 
        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
        # TV_ASSERT_RT_ERR(int64_t(b.dim(0)) * int64_t(b.dim(1)) * tv::bit_size(algo_desp.dtype_b) / 8 < int_max, 
        #     "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
        if (trans_c){{
            tv::check_shape(c_ten, {{n, m}});
        }}else{{
            tv::check_shape(c_ten, {{m, n}});
        }}
    }}else{{
        m = a_ten.dim(int(trans_a));
        k = a_ten.dim(int(!trans_a));
        k2 = b_ten.dim(int(trans_b));
        n = b_ten.dim(int(!trans_b) );
        if (trans_c){{
            tv::check_shape(c_ten, {{n, m}});
        }}else{{
            tv::check_shape(c_ten, {{m, n}});
        }}
    }}
    TV_ASSERT_INVALID_ARG(algo_desp.supported(m, n, k), "this m, n, k isn't supported due to misaligned contiguous dim.")
    TV_ASSERT_INVALID_ARG(k == k2, "error");
    if (d.ndim() == 1){{
        TV_ASSERT_RT_ERR(d.dim(0) == n, "d must be a valid bias");
    }}
    int workspace_size = algo_desp.query_workspace_size(m, n, k, split_k_slices);
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
    auto& nvrtc_params = params.nvrtc_params;

    if (nvrtc_params.cumodule){{
        TV_ASSERT_RT_ERR(nvrtc_params.kernel_name != "", "you must provide name of your kernel");
        tv::gemm::GemmNVRTCParams kernel_params;
        if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAC){{
            const int* a_ptr = nullptr;
            if (!a_inds.empty()){{
                a_ptr = a_inds.data_ptr<const int>();
            }}
            TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
            auto indice_ptr = c_inds.data_ptr<const int>();
            kernel_params = tv::gemm::GemmNVRTCParams{{m, n, k, a_ten.const_raw_data(),  b_ten.const_raw_data(),  
                c_ten.raw_data(), d_ten.raw_data(), 
                a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0),
                a_ptr, indice_ptr, d.ndim() == 1 ? nullptr : indice_ptr, 
                float(params.alpha), float(params.beta), 
                float(params.act_alpha), float(params.act_beta), 
                static_cast<int>(params.act_type),
                split_k_slices, workspace_ptr}};

        }}else if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAB){{
            TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(), "error");
            kernel_params = tv::gemm::GemmNVRTCParams{{m, n, k, a_ten.const_raw_data(),  b_ten.const_raw_data(),  
                c_ten.raw_data(), d_ten.raw_data(), 
                a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0),
                a_inds.data_ptr<const int>(), b_inds.data_ptr<const int>(), nullptr,
                float(params.alpha), float(params.beta), 
                float(params.act_alpha), float(params.act_beta), 
                static_cast<int>(params.act_type),
                split_k_slices, workspace_ptr}};

        }}else{{
            kernel_params = tv::gemm::GemmNVRTCParams{{m, n, k, a_ten.const_raw_data(),  b_ten.const_raw_data(),  
                c_ten.raw_data(), c_ten.raw_data(), 
                a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0),
                nullptr, nullptr, nullptr,
                float(params.alpha), float(params.beta), 
                float(params.act_alpha), float(params.act_beta), 
                static_cast<int>(params.act_type),
                split_k_slices, workspace_ptr}};
        }}
        std::string algo_name;
        if (evtimer.enable()){{
            algo_name = algo_desp.__repr__();
        }}
        auto grid_dims_arr = tv::gemm::get_logical_tile_count(m, n, k, algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices);
        TV_ASSERT_RT_ERR(grid_dims_arr[0] != 0 && grid_dims_arr[1] != 0 && grid_dims_arr[2] != 0, "unexpected error",
            m, n, k, algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices);
        dim3 grid_dims;
        grid_dims.x = grid_dims_arr[0];
        grid_dims.y = grid_dims_arr[1];
        grid_dims.z = grid_dims_arr[2];
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(params.stream);
        auto kernel = nvrtc_params.cumodule->kernel(nvrtc_params.kernel_name);
        if (nvrtc_params.mode == {NVRTCMode.DynamicParallism.value}){{
            tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
            std::vector<void*> args{{&kernel_params, &grid_dims, &params.stream}};
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
                std::vector<void*> args{{&kernel_params, &temp_data_ptr}};
                TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
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
                // tv::ssprint(reinterpret_cast<tv::array<int, 4>*>(raw_data_ptr)[0]);
                // tv::ssprint(grid_dims.x, grid_dims.y, grid_dims.z, temp_data.size(), temp_data_cpu.size());
                TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                    nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
            }}
        }}else if (nvrtc_params.mode == {NVRTCMode.Direct.value}){{
            tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
            std::vector<void*> args{{&kernel_params}};
            TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
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
                std::vector<void*> args{{&kernel_params, &temp_data_ptr}};
                TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
            }}
            {{
                tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
                auto ptr = nvrtc_params.cumodule->get_global_ptr(nvrtc_params.constant_name);
                auto constant_ten = tv::from_blob(ptr, {{nvrtc_params.param_size}}, tv::uint8, 0);
                constant_ten.copy_(temp_data, ctx);
                std::vector<void*> args{{}};
                TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                    nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
            }}
        }}else{{
            TV_THROW_RT_ERR("not implemented");
        }}
        TV_CHECK_CUDA_ERR_V2(algo_desp.__repr__(), "error with params", a.shape(), b.shape(), c.shape());
        return;
    }}
    """)
