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

import asyncio
import contextlib
import contextvars
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from ccimport import compat

from .debug import debug_print, debug_tx, enable_debug, enter_debug_context

CUDA_SIMULATION_VARS: Optional[
    contextvars.ContextVar["CudaExecutionContext"]] = None


def _lazy_init_contextvars():
    global CUDA_SIMULATION_VARS
    if CUDA_SIMULATION_VARS is None:
        CUDA_SIMULATION_VARS = contextvars.ContextVar("CUDA")


def _div_up(x, y):
    return (x + y - 1) // y


@dataclass
class Dim3:
    x: int
    y: int
    z: int

    def count(self):
        return self.x * self.y * self.z

    def calc_offset(self, x: int, y: int, z: int):
        return x + y * self.x + z * self.x * self.y


class KernelLaunchParams:
    def __init__(self, blockDim: Dim3, gridDim: Dim3, smem_size: int = 0):
        self.blockDim = blockDim
        self.gridDim = gridDim
        self.smem_size = smem_size
        self.num_threads = self.blockDim.count()
        self.warp_size = 32
        self.num_warps = _div_up(self.num_threads, self.warp_size)


class SyncLock:
    def __init__(self, thread_count: int):
        self.thread_count = thread_count
        self.ev = asyncio.Event()
        self.lock = asyncio.Lock()
        self.count = thread_count

    async def wait(self):
        async with self.lock:
            self.count -= 1
            if self.count == 0:
                self.ev.set()
        await self.ev.wait()
        async with self.lock:
            if self.count == 0:
                self.ev.clear()
                self.count = self.thread_count


class SyncResource(SyncLock):
    def __init__(self, thread_count: int):
        super().__init__(thread_count)
        self.resource = [None for _ in range(thread_count)]

    async def gather(self, tid: int, obj: Any, root_id: int = 0):
        self.resource[tid] = obj
        self.count -= 1
        if self.count == 0:
            self.ev.set()

        await self.ev.wait()
        if self.count == 0:
            self.ev.clear()
            self.count = self.thread_count
        if tid == root_id:
            res = self.resource.copy()
            await self.wait()
            return res
        await self.wait()
        return None

    async def broadcast(self, tid: int, obj, root_id: int = 0):
        if root_id == tid:
            self.resource[0] = obj
        self.count -= 1
        if self.count == 0:
            self.ev.set()
        await self.ev.wait()
        if self.count == 0:
            self.ev.clear()
            self.count = self.thread_count
        res = self.resource[0]
        await self.wait()
        return res


class CudaExecutionContext:
    def __init__(self, block_sync: SyncResource, block_idx: Dim3,
                 thread_idx: Dim3, params: KernelLaunchParams,
                 smem: np.ndarray, warp_locks: List[SyncResource]):
        if compat.Python3_6AndLater and not compat.Python3_7AndLater:
            raise NotImplementedError("python 3.6 don't support cudasim.")
        self.blockIdx = block_idx
        self.threadIdx = thread_idx
        self.blockDim = params.blockDim
        self.gridDim = params.gridDim
        self.smem = smem
        self.block_sync_lock = block_sync
        self.warp_locks = warp_locks

    def get_thread_id(self):
        return self.blockDim.calc_offset(self.threadIdx.x, self.threadIdx.y,
                                         self.threadIdx.z)


@contextlib.contextmanager
def enter_cuda_context(ctx: CudaExecutionContext):
    """dont need async manager here because ContextVar support asyncio
    natively.
    """
    _lazy_init_contextvars()
    token = CUDA_SIMULATION_VARS.set(ctx)
    yield ctx
    CUDA_SIMULATION_VARS.reset(token)


def get_cuda_context() -> Optional[CudaExecutionContext]:
    _lazy_init_contextvars()
    return CUDA_SIMULATION_VARS.get(None)


def inside_cuda() -> bool:
    return get_cuda_context() is not None


def threadIdx():
    ctx = get_cuda_context()
    if ctx is None:
        return Dim3(-1, -1, -1)
    return ctx.threadIdx


def blockDim():
    ctx = get_cuda_context()
    if ctx is None:
        return Dim3(-1, -1, -1)
    return ctx.blockDim


def gridDim():
    ctx = get_cuda_context()
    if ctx is None:
        return Dim3(-1, -1, -1)
    return ctx.gridDim


def get_smem() -> np.ndarray:
    return get_cuda_context().smem


def blockIdx():
    ctx = get_cuda_context()
    if ctx is None:
        return Dim3(-1, -1, -1)
    return ctx.blockIdx


def debug_once(tx: int = -1):
    tid = threadIdx()
    bidx = blockIdx()
    exp_tx = tx
    if exp_tx == -1:
        exp_tx = debug_tx()
    return tid.x == exp_tx and tid.y == 0 and tid.z == 0 and bidx.x == 0 and bidx.y == 0


def get_thread_id():
    return get_cuda_context().get_thread_id()


def get_warp_id():
    return get_thread_id() // 32


def get_lane_id():
    return get_thread_id() % 32


def get_warp_resource():
    return get_cuda_context().warp_locks[get_warp_id()]


async def block_broadcast(obj, root_id: int = 0):
    ctx = get_cuda_context()
    block_lock = ctx.block_sync_lock
    return await block_lock.broadcast(ctx.get_thread_id(), obj, root_id)


async def warp_broadcast(obj, root_id: int = 0):
    resource = get_warp_resource()
    return await resource.broadcast(get_lane_id(), obj, root_id)


async def warp_gather(obj, root_id: int = 0):
    resource = get_warp_resource()
    return await resource.gather(get_lane_id(), obj, root_id)


async def warp_wait():
    resource = get_warp_resource()
    return await resource.wait()


async def thread_launch(func, ctx: CudaExecutionContext):
    with enter_cuda_context(ctx):
        res = await func()
    return res


async def block_launch(func, block_idx: Dim3, params: KernelLaunchParams):
    smem = np.zeros((params.smem_size, ), dtype=np.int8)
    block_sync = SyncResource(params.blockDim.count())
    warp_locks = [
        SyncResource(params.warp_size) for _ in range(params.num_warps)
    ]
    block_tasks = []
    threads = params.blockDim
    for tz in range(threads.z):
        for ty in range(threads.y):
            for tx in range(threads.x):
                ctx = CudaExecutionContext(block_sync, block_idx,
                                           Dim3(tx, ty, tz), params, smem,
                                           warp_locks)
                block_tasks.append(thread_launch(func, ctx))
    res = await asyncio.gather(*block_tasks)
    block_res = {}
    i = 0
    for tz in range(threads.z):
        for ty in range(threads.y):
            for tx in range(threads.x):
                block_res[(block_idx.x, block_idx.y, block_idx.z, tx, ty,
                           tz)] = res[i]
                i += 1
    return block_res


async def kernel_launch(func, blocks: Dim3, threads: Dim3, smem_size: int = 0):
    params = KernelLaunchParams(threads, blocks, smem_size)
    kernel_res = {}
    for bz in range(blocks.z):
        for by in range(blocks.y):
            for bx in range(blocks.x):
                kernel_res.update(await block_launch(func, Dim3(bx, by, bz),
                                                     params))

    return kernel_res


async def syncthreads():
    ctx = get_cuda_context()
    await ctx.block_sync_lock.wait()
