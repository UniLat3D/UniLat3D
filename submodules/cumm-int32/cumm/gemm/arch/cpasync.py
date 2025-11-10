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

from typing import List, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import constants, core
from enum import Enum
from cumm.gemm.thread_map import PitchLinearWarpRaked
import asyncio


class CacheOperation(Enum):
    Always = 0
    Global = 1
    
    # we don't use below
    Streaming = 2
    LastUse = 3
    Volatile = 4
    WriteBack = 5
    WriteThrough = 6

class AsyncCopyConfig(Enum):
    NFill = 0           # if!valid   do nothing
    ZFill = 1           # if!valid:  fill 0
    NanFill = 2         # if!valid:  fill nan   TODO: not impl,   may never used.

class CpAsyncGroup(pccm.ParameterizedClass):
    def __init__(self, stages=3):
        super().__init__()
        self.stages = stages
        self.add_global_code("""
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                #define CUDA_CP_ASYNC_ACTIVATED 1
            #else
                #define CUDA_CP_ASYNC_ACTIVATED 0
            #endif""")
    
    @staticmethod
    def wait_group(N: int):
        if N >= 0:
            code = pccm.FunctionCode()
            with code.macro_if_("CUDA_CP_ASYNC_ACTIVATED"):
                code.raw(rf"""
                    asm volatile("cp.async.wait_group %0;\n" ::"n"({N}));
                """)
            with code.macro_else_():
                code.raw(rf"""
                    assert(0);
                """)
            code.macro_endif_()
            return code
        else:
            raise ValueError
    
    @pccm.cuda.static_function(device=True, forceinline=True)
    def make_fence(self):
        code = pccm.FunctionCode()
        with code.macro_if_("CUDA_CP_ASYNC_ACTIVATED"):
            code.raw(r"""
                asm volatile("cp.async.commit_group;\n" ::);
            """)
        with code.macro_else_():
            code.raw(r"""
                assert(0);
            """)
        code.macro_endif_()
        return code
    
    @pccm.cuda.static_function(device=True, forceinline=True)
    def wait_final_group(self):
        return CpAsyncGroup.wait_group(self.stages - 2)
    
    @pccm.cuda.static_function(device=True, forceinline=True)
    def wait_all(self):
        code = pccm.FunctionCode()
        with code.macro_if_("CUDA_CP_ASYNC_ACTIVATED"):
            code.raw(r"""
                asm volatile("cp.async.wait_all;\n" ::);
            """)
        with code.macro_else_():
            code.raw(r"""
                assert(0);
            """)
        code.macro_endif_()
        return code
    
    ################################# PYTHON implements ##################################
    def python_ctor(self):
        new_obj = CpAsyncGroup(self.stages)
        new_obj.fenced_groups = []
        new_obj.unbind_opr = []
        return new_obj
    
    def _make_async_copy_python(self, dest_arr, source_arr, valid: bool, Type=AsyncCopyConfig.NFill):
        self.unbind_opr.append((dest_arr, source_arr, valid, Type))
    
    def make_fence_python(self):
        self.fenced_groups.append(self.unbind_opr)
        self.unbind_opr = []
    
    async def _wait_group(self, N):
        assert N >= 0
        async def global_load(dest: ArrayPtr, source: ArrayPtr, valid, Type=AsyncCopyConfig.NFill):
            if valid:
                dest[0] = source[0]
            else:
                if Type == AsyncCopyConfig.ZFill:
                    dest[0].clear()
                elif Type == AsyncCopyConfig.NanFill:
                    raise NotImplementedError
                elif Type == AsyncCopyConfig.NFill:
                    return
                else:
                    raise NotImplementedError
        
        while len(self.fenced_groups) > N:
            tasks = [global_load(*args) for args in self.fenced_groups[0]]
            await asyncio.gather(*tasks)
            self.fenced_groups = self.fenced_groups[1:]
    
    async def wait_final_group_python(self):
        await self._wait_group(self.stages - 2)

    async def wait_all_python(self):
        self.make_fence_python()
        await self._wait_group(0)


class CpAsyncCopy(pccm.ParameterizedClass):
    def __init__(self, warp_raked_tmap: PitchLinearWarpRaked, dtype: dtypes.DType):
        super().__init__()
        access_size_in_bytes = warp_raked_tmap.element_per_acc * dtype.itemsize()
        cache_opr = CacheOperation.Always
        if access_size_in_bytes == 16:
            cache_opr = CacheOperation.Global
        self.access_size_in_byte = access_size_in_bytes
        self.cache_opr = cache_opr
        if self.cache_opr == CacheOperation.Always:
            assert self.access_size_in_byte in [4, 8, 16], "only support 4, 8, 16 size"
        elif self.cache_opr == CacheOperation.Global:
            assert self.access_size_in_byte == 16, "it only support 16 B"
        else:
            raise NotImplementedError
        self.warp_raked_tmap = warp_raked_tmap
        self.dtype = dtype
        self.add_global_code
        self.add_global_code("""
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                #define CUDA_CP_ASYNC_ACTIVATED 1
            #else
                #define CUDA_CP_ASYNC_ACTIVATED 0
            #endif
            #if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \\
                (__CUDACC_VER_MAJOR__ > 11)) &&                                  \\
                defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && \\
                ! (defined(__clang__) && defined(__CUDA__))
                #define CUMM_ENABLE_L2_PREFETCH 1
            #else
                #define CUMM_ENABLE_L2_PREFETCH 0
            #endif
        """)

    def python_ctor(self):
        return CpAsyncCopy(self.warp_raked_tmap, self.dtype)
    
    @pccm.cuda.static_function(device=True, forceinline=True)
    def copy(self):
        code = pccm.FunctionCode()
        code.arg("dest_smem", "void*")
        code.arg("src_global", "const void*")
        code.arg("pred_guard", "bool", "true")
        code.raw("""
            unsigned smem_addr = tv::gemm::get_smem_pointer(dest_smem);
        """)
        if self.cache_opr == CacheOperation.Always:
            with code.macro_if_("CUDA_CP_ASYNC_ACTIVATED"):
                code.raw(f"""
                    asm volatile(
                        "{{\\n"
                        "  .reg .pred p;\\n"
                        "  setp.ne.b32 p, %0, 0;\\n"
#if CUMM_ENABLE_L2_PREFETCH
                        "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\\n"
#else
                        "  @p cp.async.ca.shared.global [%1], [%2], %3;\\n"
#endif
                        "}}\\n" ::"r"((int)pred_guard), "r"(smem_addr), "l"(src_global), "n"({self.access_size_in_byte}));
                """)
            with code.macro_else_():
                code.raw("assert(0);")
            code.macro_endif_()
        elif self.cache_opr == CacheOperation.Global:
            with code.macro_if_("CUDA_CP_ASYNC_ACTIVATED"):
                code.raw(f"""
                    asm volatile(
                        "{{\\n"
                        "  .reg .pred p;\\n"
                        "  setp.ne.b32 p, %0, 0;\\n"
#if CUMM_ENABLE_L2_PREFETCH
                        "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\\n"
#else
                        "  @p cp.async.cg.shared.global [%1], [%2], %3;\\n"
#endif
                        "}}\\n" ::"r"((int)pred_guard), "r"(smem_addr), "l"(src_global), "n"({self.access_size_in_byte}));
                """)
            with code.macro_else_():
                code.raw("assert(0);")
            code.macro_endif_()
        else:
            raise NotImplementedError
        return code
    
    def copy_python(self, dest_ptr, src_ptr, valid, GroupArrange: CpAsyncGroup):
        # we use CpAsyncGroup to implement async copy
        # print(cudasim.get_thread_id(), 'copy', f"{src_ptr.offset}-{src_ptr.length}", " to ", f"{dest_ptr.offset} {dest_ptr.length}", valid)
        if valid:
            assert(src_ptr.length > 0)
        GroupArrange._make_async_copy_python(dest_ptr, src_ptr, valid, AsyncCopyConfig.NFill)

    @pccm.cuda.static_function(device=True, forceinline=True)
    def copy_zfill(self):
        code = pccm.FunctionCode()
        code.arg("dest_smem", "void*")
        code.arg("src_global", "const void*")
        code.arg("pred_guard", "bool", "true")
        code.raw(f"""
        """)
        with code.macro_if_("CUDA_CP_ASYNC_ACTIVATED"):
            code.raw(f"""
            unsigned smem_addr = tv::gemm::get_smem_pointer(dest_smem);
            unsigned real_size = (pred_guard ? {self.access_size_in_byte} : 0);
            """)
            if self.cache_opr == CacheOperation.Always:
                code.raw(f"""
                    asm volatile(
#if CUMM_ENABLE_L2_PREFETCH
                        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\\n"
#else
                        "cp.async.ca.shared.global [%0], [%1], %2, %3;\\n"
#endif
                         ::"r"(smem_addr), "l"(src_global), "n"({self.access_size_in_byte}), "r"(real_size));
                """)
            elif self.cache_opr == CacheOperation.Global:
                code.raw(f"""
                    asm volatile(
#if CUMM_ENABLE_L2_PREFETCH
                        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\\n"
#else
                        "cp.async.cg.shared.global [%0], [%1], %2, %3;\\n"
#endif
                         ::"r"(smem_addr), "l"(src_global), "n"({self.access_size_in_byte}), "r"(real_size));
                """)
            else:
                raise NotImplementedError
        with code.macro_else_():
            code.raw("assert(0);")
        code.macro_endif_()
        return code

    def copy_zfill_python(self, dest_ptr, src_ptr, valid, GroupArrange: CpAsyncGroup):
        if valid:
            assert(src_ptr.length > 0)
        GroupArrange._make_async_copy_python(dest_ptr, src_ptr, valid, AsyncCopyConfig.ZFill)
