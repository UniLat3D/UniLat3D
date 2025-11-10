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

import math
from typing import List

import pccm
from pccm.core import FunctionCode

from cumm import dtypes
from cumm.common import GemmBasic, TensorView, TensorViewKernel
from cumm.gemm import thread_map
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.frozen import mask_iters


class GatherKernel(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 epa: int, num_threads: int):
        super().__init__()
        self.add_dependency(TensorView, TensorViewKernel)
        self.dtype = dtype
        self.tile_shape = tile_shape
        self.epa = epa
        sub_tile_shape = seq(1, epa)
        self.tmap = thread_map.PitchLinear(tile_shape, sub_tile_shape,
                                           num_threads)
        self.num_threads = num_threads
        self.inp_iter_in_param = mask_iters.MaskTileIteratorParams(
            dtype, tile_shape, sub_tile_shape, self.tmap, 1, True)

        self.inp_iter_out_param = mask_iters.MaskTileIteratorParams(
            dtype, tile_shape, sub_tile_shape, self.tmap, 1, False)

        self.inp_iter_in = mask_iters.MaskTileIteratorGather(
            dtype, tile_shape, sub_tile_shape, self.tmap,
            self.inp_iter_in_param, 1, epa, False, False, True)

        self.inp_iter_out = mask_iters.MaskTileIteratorGather(
            dtype,
            tile_shape,
            sub_tile_shape,
            self.tmap,
            self.inp_iter_out_param,
            1,
            epa,
            False,
            False,
            False,
            read_only=False)

        self.add_param_class("inpp1", self.inp_iter_in_param, "InputParams")
        self.add_param_class("outp1", self.inp_iter_out_param, "OutputParams")
        self.add_param_class("inpiter1", self.inp_iter_in, "InputIter")
        self.add_param_class("outiter1", self.inp_iter_out, "OutputIter")

    @pccm.cuda.cuda_global_function(header_only=True)
    def kernel(self):
        code = pccm.code()
        code.arg("input_ptr", f"const {self.dtype}*")
        code.arg("output_ptr", f"{self.dtype}*")

        code.arg("m, k, k_iterations", f"int")

        code.arg("inp_params", "InputParams")
        code.arg("out_params", "OutputParams")
        code.raw(f"""
        int tile_offset_m = blockIdx.x;
        int block_offset_m = tile_offset_m * {self.tile_shape[0]};
        InputIter input_iter(
            inp_params, input_ptr,
            {{m, k}},
            threadIdx.x,
            {{block_offset_m, 0}});
        OutputIter output_iter(
            out_params, output_ptr,
            {{m, k}},
            threadIdx.x,
            {{block_offset_m, 0}});
        {self.inp_iter_in.fragment_t} input_frag;
        // input_frag.clear();
        for (; k_iterations > 0; --k_iterations){{
            input_iter.load(input_frag);
            input_iter.store(input_frag);
            ++input_iter;
            ++output_iter;
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def gather(self):
        code = pccm.code()
        code.arg("output", "tv::Tensor")
        code.arg("input", "tv::Tensor")
        code.arg("indices", "tv::Tensor")

        code.raw(f"""
        int m = input.dim(0);
        auto timer = tv::CudaContextTimer<>();
        InputParams parmas_inp(input.dim(1), indices.data_ptr<const int>());
        OutputParams parmas_out(input.dim(1));
        int k_iterations = tv::div_up(input.dim(1), {self.tile_shape[1]});
        dim3 grid(tv::div_up(m, {self.tile_shape[0]}));
        tv::cuda::Launch launcher(grid, dim3({self.num_threads}, 1, 1));
        tv::ssprint(grid.x, grid.y, grid.z, k_iterations, m);
        launcher(kernel, input.data_ptr<const {self.dtype}>(), output.data_ptr<{self.dtype}>(), m, 
            input.dim(1), k_iterations, parmas_inp, parmas_out);
        tv::ssprint("gather time", timer.report() / 1000.0);
        """)
        return code


class GatherKernelV2(pccm.ParameterizedClass):
    def __init__(self,
                 tile_shape_bytes: MetaArray[int],
                 bytes_per_access: int,
                 num_threads: int,
                 dtype: dtypes.DType = dtypes.int8):
        super().__init__()
        assert bytes_per_access <= 16
        tile_size = tile_shape_bytes.prod()
        assert tile_size % bytes_per_access == 0
        num_access = tile_size // bytes_per_access
        assert num_access % num_threads == 0
        num_access_per_thread = num_access // num_threads
        assert bytes_per_access % dtype.itemsize() == 0
        epa = bytes_per_access // dtype.itemsize()
        self.bytes_per_access = bytes_per_access
        self.add_dependency(TensorView, TensorViewKernel)
        self.dtype = dtype
        self.tile_shape_bytes = tile_shape_bytes
        self.epa = epa
        sub_tile_shape = seq(1, epa)
        self.tmap = thread_map.PitchLinear(tile_shape_bytes, sub_tile_shape,
                                           num_threads)
        self.num_threads = num_threads
        self.iter_param = mask_iters.MaskTileIteratorParams(
            dtype, tile_shape_bytes, sub_tile_shape, self.tmap, 1, True)

        self.tile_iter = mask_iters.MaskTileIteratorGather(
            dtype,
            tile_shape_bytes,
            sub_tile_shape,
            self.tmap,
            self.iter_param,
            1,
            epa,
            False,
            False,
            True,
            have_output_ptr=True)
        self.add_param_class("iterpns", self.iter_param, "IterParams")
        self.add_param_class("iterns", self.tile_iter, "TileIterator")

    @pccm.cuda.cuda_global_function(header_only=True)
    def kernel(self):
        code = pccm.code()
        code.arg("input_ptr", f"const {self.dtype}*")
        code.arg("output_ptr", f"{self.dtype}*")

        code.arg("m, k_bytes, k_iterations", f"int")

        code.arg("params", "IterParams")
        code.raw(f"""
        int tile_offset_m = blockIdx.x;
        int block_offset_m = tile_offset_m * {self.tile_shape_bytes[0]};
        TileIterator iter(
            params, input_ptr,
            output_ptr,
            {{m, k_bytes}},
            threadIdx.x,
            {{block_offset_m, 0}});
        {self.tile_iter.fragment_t} frag;
        for (; k_iterations > 0; --k_iterations){{
            iter.load(frag);
            iter.store(frag);
            ++iter;
        }}
        """)
        return code


class Gather(pccm.ParameterizedClass):
    def __init__(self,
                 tile_shape_bytes: MetaArray[int],
                 bytes_per_access: int,
                 num_threads: int,
                 dtype: dtypes.DType = dtypes.int8):
        super().__init__()
        self.tile_shape_bytes = tile_shape_bytes
        self.bytes_per_access = bytes_per_access
        self.num_threads = num_threads
        self.dtype = dtype
        self.add_dependency(TensorView)

    @pccm.cuda.static_function(inline=True)
    def supported(self):
        code = pccm.code()
        code.arg("channel_size", "int")
        code.arg("dtype", "int")
        code.raw(f"""
        auto tv_dtype_size = tv::detail::sizeof_dtype(tv::DType(dtype));
        return (channel_size * tv_dtype_size) % {self.bytes_per_access} == 0;
        """)
        return code.ret("bool")

    # @pccm.pybind.mark

    @pccm.cuda.static_function
    def gather(self):
        code = pccm.code()
        kernel = GatherKernelV2(self.tile_shape_bytes, self.bytes_per_access,
                                self.num_threads, self.dtype)
        code.add_param_class("kernel", kernel, "GatherKernel")
        code.add_param_class("kernel", kernel.iter_param, "IterParams")

        code.arg("output", "tv::Tensor")
        code.arg("input", "tv::Tensor")
        code.arg("indices", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")

        code.raw(f"""
        TV_ASSERT_RT_ERR(input.dim(1) == output.dim(1) && output.ndim() == 2 && input.ndim() == 2, "error");
        TV_ASSERT_RT_ERR(input.dtype() == output.dtype(), "error");

        int m = indices.dim(0);
        int itemsize = input.itemsize();
        int channel_size_bytes = input.dim(1) * itemsize;
        TV_ASSERT_RT_ERR(supported(input.dim(1), input.dtype()), "not supported. use supported() to check first");
        // auto timer = tv::CudaContextTimer<>();

        IterParams parmas_iter(channel_size_bytes, indices.data_ptr<const int>());
        int k_iterations = tv::div_up(channel_size_bytes, {self.tile_shape_bytes[1]});
        dim3 grid(tv::div_up(m, {self.tile_shape_bytes[0]}));
        tv::cuda::Launch launcher(grid, dim3({self.num_threads}, 1, 1), 0, reinterpret_cast<cudaStream_t>(stream));
        auto inp_ptr = reinterpret_cast<const {self.dtype}*>(input.raw_data());
        auto out_ptr = reinterpret_cast<{self.dtype}*>(output.raw_data());
        launcher(kernel::kernel, inp_ptr, out_ptr, m, 
            channel_size_bytes, k_iterations, parmas_iter);
        // tv::ssprint("gather time", timer.report() / 1000.0);
        """)
        return code


class ScatterKernel(pccm.ParameterizedClass):
    def __init__(self, tile_shape_bytes: MetaArray[int], bytes_per_access: int,
                 num_threads: int, dtype: dtypes.DType):
        super().__init__()
        self.add_include("tensorview/gemm/math/all.h")
        assert bytes_per_access <= 16
        tile_size = tile_shape_bytes.prod()
        assert tile_size % bytes_per_access == 0
        num_access = tile_size // bytes_per_access
        assert num_access % num_threads == 0
        num_access_per_thread = num_access // num_threads
        assert bytes_per_access % dtype.itemsize() == 0
        epa = bytes_per_access // dtype.itemsize()
        self.bytes_per_access = bytes_per_access
        self.add_dependency(TensorView, TensorViewKernel)
        self.dtype = dtype
        self.tile_shape = seq(tile_shape_bytes[0],
                              tile_shape_bytes[1] // dtype.itemsize())
        self.epa = epa
        sub_tile_shape = seq(1, epa)
        self.tmap = thread_map.PitchLinear(self.tile_shape, sub_tile_shape,
                                           num_threads)
        self.num_threads = num_threads
        self.iter_param = mask_iters.MaskTileIteratorParams(
            dtype, self.tile_shape, sub_tile_shape, self.tmap, 1, True)

        self.tile_iter = mask_iters.MaskTileIteratorGather(
            dtype,
            self.tile_shape,
            sub_tile_shape,
            self.tmap,
            self.iter_param,
            1,
            epa,
            False,
            False,
            True,
            read_only=False,
            have_output_ptr=True,
            is_scatter=True)
        self.add_param_class("iterpns", self.iter_param, "IterParams")
        self.add_param_class("iterns", self.tile_iter, "TileIterator")

    @pccm.cuda.cuda_global_function(header_only=True)
    def kernel(self):
        code = pccm.code()
        code.arg("input_ptr", f"const {self.dtype}*")
        code.arg("output_ptr", f"{self.dtype}*")

        code.arg("m, k, k_iterations", f"int")

        code.arg("params", "IterParams")
        code.raw(f"""
        int tile_offset_m = blockIdx.x;
        int block_offset_m = tile_offset_m * {self.tile_shape[0]};
        TileIterator iter(
            params, output_ptr,
            input_ptr,
            {{m, k}},
            threadIdx.x,
            {{block_offset_m, 0}});
        {self.tile_iter.fragment_t} frag, out_frag;
        // out_frag.clear();
        tv::math::plus<{self.tile_iter.fragment_t}> accer;
        for (; k_iterations > 0; --k_iterations){{
            iter.load(frag);
            iter.load_output_with_byte_offset(out_frag, 0);
            out_frag = accer(frag, out_frag);
            iter.store(out_frag);
            ++iter;
        }}
        """)
        return code


class Scatter(pccm.ParameterizedClass):
    def __init__(self, tile_shape_bytes: MetaArray[int], bytes_per_access: int,
                 num_threads: int, dtype: dtypes.DType):
        super().__init__()
        self.tile_shape_bytes = tile_shape_bytes
        self.bytes_per_access = bytes_per_access
        self.num_threads = num_threads
        self.dtype = dtype
        self.add_dependency(TensorView)

    @pccm.cuda.static_function(inline=True)
    def supported(self):
        code = pccm.code()
        code.arg("channel_size", "int")
        code.arg("dtype", "int")
        code.raw(f"""
        auto tv_dtype_size = tv::detail::sizeof_dtype(tv::DType(dtype));
        return (channel_size * tv_dtype_size) % {self.bytes_per_access} == 0;
        """)
        return code.ret("bool")

    @pccm.cuda.static_function
    def scatter(self):
        code = pccm.code()
        kernel = ScatterKernel(self.tile_shape_bytes, self.bytes_per_access,
                               self.num_threads, self.dtype)
        code.add_param_class("kernel", kernel, "ScatterKernel")
        code.add_param_class("kernel", kernel.iter_param, "IterParams")

        code.arg("output", "tv::Tensor")
        code.arg("input", "tv::Tensor")
        code.arg("indices", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")

        code.raw(f"""
        TV_ASSERT_RT_ERR(input.dim(1) == output.dim(1) && output.ndim() == 2 && input.ndim() == 2, "error");
        TV_ASSERT_RT_ERR(input.dtype() == output.dtype(), "error");

        int m = indices.dim(0);
        int itemsize = input.itemsize();
        int channel_size_bytes = input.dim(1) * itemsize;
        TV_ASSERT_RT_ERR(supported(input.dim(1), input.dtype()), "not supported. use supported() to check first");
        // auto timer = tv::CudaContextTimer<>();

        IterParams parmas_iter(input.dim(1), indices.data_ptr<const int>());
        int k_iterations = tv::div_up(channel_size_bytes, {self.tile_shape_bytes[1]});
        dim3 grid(tv::div_up(m, {self.tile_shape_bytes[0]}));
        tv::cuda::Launch launcher(grid, dim3({self.num_threads}, 1, 1), 0, reinterpret_cast<cudaStream_t>(stream));
        auto inp_ptr = reinterpret_cast<const {self.dtype}*>(input.raw_data());
        auto out_ptr = reinterpret_cast<{self.dtype}*>(output.raw_data());
        launcher(kernel::kernel, inp_ptr, out_ptr, m, 
            input.dim(1), k_iterations, parmas_iter);
        // tv::ssprint("gather time", timer.report() / 1000.0);
        """)
        return code


class GatherAll(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("vector", "tuple", "unordered_map", "functional")
        self.add_dependency(TensorView)
        self.kernels: List[Gather] = []
        uniques = set()
        for tile_size_k_bytes in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            for tile_m in [1, 2, 4, 8]:
                for byte_per_access in [4, 8, 16]:
                    for num_threads in [64, 128, 256]:
                        # for tile_size_k_bytes in [512]:
                        #     for tile_m in [8]:
                        #         for byte_per_access in [4]:
                        #             for num_threads in [128]:

                        if num_threads * byte_per_access > tile_size_k_bytes * tile_m:
                            continue
                        key = (tile_m, tile_size_k_bytes, byte_per_access,
                               num_threads)
                        if key not in uniques:
                            self.kernels.append(
                                Gather(seq(tile_m, tile_size_k_bytes),
                                       byte_per_access, num_threads))
                        uniques.add(key)
        self.add_member(
            "param_to_function_",
            "std::unordered_map<int64_t, std::function<void(tv::Tensor, tv::Tensor, tv::Tensor, std::uintptr_t)>>"
        )

    def _get_alias_of_ker(self, ker: Gather):
        return f"Gather{ker.tile_shape_bytes[0]}_{ker.tile_shape_bytes[1]}_{ker.bytes_per_access}_{ker.num_threads}"

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = FunctionCode()
        for ker in self.kernels:
            ns = f"g{ker.tile_shape_bytes[0]}_{ker.tile_shape_bytes[1]}_{ker.bytes_per_access}_{ker.num_threads}"
            code.add_param_class(ns, ker, self._get_alias_of_ker(ker))
        code.raw(f"""
        int64_t key;
        """)
        for k in self.kernels:
            key = (k.tile_shape_bytes[0] << 0) | (
                k.tile_shape_bytes[1] << 16) | (k.bytes_per_access << 32) | (
                    k.num_threads << 48)
            code.raw(f"""
            param_to_function_[{key}] = {self._get_alias_of_ker(k)}::gather;
            """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def get_all_gather_params(self):
        code = FunctionCode()
        code.raw(f"""
        std::vector<std::tuple<int, int, int, int>> res;
        """)
        for k in self.kernels:
            code.raw(f"""
            res.push_back({{{k.tile_shape_bytes[0]}, {k.tile_shape_bytes[1]}, {k.bytes_per_access}, {k.num_threads}}});
            """)
        code.raw(f"return res;")
        return code.ret("std::vector<std::tuple<int, int, int, int>>")

    @pccm.pybind.mark
    @pccm.static_function
    def supported(self):
        code = pccm.code()
        code.arg("bytes_per_access", "int")
        code.arg("channel_size", "int")
        code.arg("dtype", "int")
        code.raw(f"""
        auto tv_dtype_size = tv::detail::sizeof_dtype(tv::DType(dtype));
        return (channel_size * tv_dtype_size) % bytes_per_access == 0;
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def stream_synchronize(self):
        code = pccm.code()
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        checkCudaErrors(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def gather(self):
        code = pccm.code()
        code.arg("output", "tv::Tensor")
        code.arg("input", "tv::Tensor")
        code.arg("indices", "tv::Tensor")
        code.arg("tile_m, tile_k_bytes, bytes_per_access, num_threads",
                 "int64_t")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.raw(f"""
        int64_t key = (tile_m << 0) | (tile_k_bytes << 16) | (bytes_per_access << 32) | (num_threads << 48);
        auto method_iter = param_to_function_.find(key);
        TV_ASSERT_RT_ERR(method_iter != param_to_function_.end(), "can't find a sutable gather");
        return method_iter->second(output, input, indices, stream);
        """)
        return code


class ScatterAll(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("vector", "tuple", "unordered_map", "functional")
        self.add_dependency(TensorView)
        self.kernels: List[Scatter] = []
        uniques = set()
        for tile_size_k_bytes in [64, 128, 256, 512, 1024, 2048]:
            for tile_m in [1, 2, 4, 8]:
                for byte_per_access in [4, 8, 16]:
                    for num_threads in [128, 256]:
                        # for tile_size_k_bytes in [256]:
                        #     for tile_m in [8]:
                        #         for byte_per_access in [16]:
                        #             for num_threads in [128]:

                        for dtype in [dtypes.float32]:
                            if num_threads * byte_per_access > tile_size_k_bytes * tile_m:
                                continue
                            key = (tile_m, tile_size_k_bytes, byte_per_access,
                                   num_threads, dtype)
                            if key not in uniques:
                                self.kernels.append(
                                    Scatter(seq(tile_m, tile_size_k_bytes),
                                            byte_per_access, num_threads,
                                            dtype))
                            uniques.add(key)
        self.add_member(
            "param_to_function_",
            "std::unordered_map<int64_t, std::function<void(tv::Tensor, tv::Tensor, tv::Tensor, std::uintptr_t)>>"
        )

    def _get_alias_of_ker(self, ker: Scatter):
        return f"Scatter{ker.dtype.shortcut()}{ker.tile_shape_bytes[0]}_{ker.tile_shape_bytes[1]}_{ker.bytes_per_access}_{ker.num_threads}"

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = FunctionCode()
        for ker in self.kernels:
            ns = f"s{ker.dtype.shortcut()}{ker.tile_shape_bytes[0]}_{ker.tile_shape_bytes[1]}_{ker.bytes_per_access}_{ker.num_threads}"
            code.add_param_class(ns, ker, self._get_alias_of_ker(ker))
        for k in self.kernels:
            key = (k.tile_shape_bytes[0] << 0) | (
                k.tile_shape_bytes[1] << 8) | (k.bytes_per_access << 24) | (
                    k.num_threads << 32) | (k.dtype.tv_dtype << 48)
            code.raw(f"""
            param_to_function_[{key}] = {self._get_alias_of_ker(k)}::scatter;
            """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def get_all_scatter_params(self):
        code = FunctionCode()
        code.raw(f"""
        std::vector<std::tuple<int, int, int, int>> res;
        """)
        uniques = set()

        for k in self.kernels:
            key = (k.tile_shape_bytes[0], k.tile_shape_bytes[1],
                   k.bytes_per_access, k.num_threads)
            if key in uniques:
                continue
            uniques.add(key)
            code.raw(f"""
            res.push_back({{{k.tile_shape_bytes[0]}, {k.tile_shape_bytes[1]}, {k.bytes_per_access}, {k.num_threads}}});
            """)

        code.raw(f"return res;")
        return code.ret("std::vector<std::tuple<int, int, int, int>>")

    @pccm.pybind.mark
    @pccm.member_function
    def supported_scatter(self):
        code = pccm.code()
        code.arg("tile_m, tile_k_bytes, bytes_per_access, num_threads",
                 "int64_t")
        code.arg("channel_size", "int")
        code.arg("dtype", "int")
        code.raw(f"""
        int64_t dtype_64 = int64_t(dtype);
        int64_t key = (tile_m << 0) | (tile_k_bytes << 8) | (bytes_per_access << 24) | (num_threads << 32) | (dtype_64 << 48);
        if (param_to_function_.find(key) == param_to_function_.end()){{
            return false;
        }}
        auto tv_dtype_size = tv::detail::sizeof_dtype(tv::DType(dtype));
        return (channel_size * tv_dtype_size) % bytes_per_access == 0;
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def stream_synchronize(self):
        code = pccm.code()
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        checkCudaErrors(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def scatter(self):
        code = pccm.code()
        code.arg("output", "tv::Tensor")
        code.arg("input", "tv::Tensor")
        code.arg("indices", "tv::Tensor")
        code.arg("tile_m, tile_k_bytes, bytes_per_access, num_threads",
                 "int64_t")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.raw(f"""
        int64_t dtype = int64_t( input.dtype());
        int64_t key = (tile_m << 0) | (tile_k_bytes << 8) | (bytes_per_access << 24) | (num_threads << 32) | (dtype << 48);
        auto method_iter = param_to_function_.find(key);
        TV_ASSERT_RT_ERR(method_iter != param_to_function_.end(), "can't find a sutable scatter");
        return method_iter->second(output, input, indices, stream);
        """)
        return code
