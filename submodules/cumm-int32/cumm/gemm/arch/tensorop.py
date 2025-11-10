# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, TensorViewNVRTCKernel
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import constants
from cumm.gemm.algospec.core import TensorOp
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.bases import GemmComponentBase

class MmaLayoutDespBase:
    def __init__(self, shape: MetaArray[int], dtype_ab: dtypes.DType, dtype_c: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 min_sm: Tuple[int, int],
                 num_threads: int = 32,
                 satfinite: bool = False) -> None:
        self.shape = shape 

        self.dtype_ab = dtype_ab
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.min_sm = min_sm
        self.num_threads = num_threads
        self.satfinite = satfinite
        self.lane_ids = np.arange(num_threads, dtype=np.int32)

        self.mn = shape[0] * shape[1]
        self.km = shape[2] * shape[0]
        self.kn = shape[2] * shape[1]
        self.shape_a = [shape[0], shape[2]]
        # if trans_a:
        #     self.shape_a = self.shape_a[::-1]
        self.shape_b = [shape[2], shape[1]]
        # if trans_b:
        #     self.shape_b = self.shape_b[::-1]
        self.shape_c = [shape[0], shape[1]]
        # if trans_c:
        #     self.shape_c = self.shape_c[::-1]

        self.fragment_c_count = self.mn // self.num_threads
        self.fragment_a_count = self.km // self.num_threads
        self.fragment_b_count = self.kn // self.num_threads

        self._map_a = np.zeros((self.fragment_a_count, self.num_threads, 2), dtype=np.int32)
        self._map_b = np.zeros((self.fragment_b_count, self.num_threads, 2), dtype=np.int32)
        self._map_c = np.zeros((self.fragment_c_count, self.num_threads, 2), dtype=np.int32)
        self.gid = self.lane_ids >> 2
        self.tid_in_group = self.lane_ids % 4

    def __repr__(self) -> str:
        mma_stmt = f"mma.sync.aligned.m{self.shape[0]}n{self.shape[1]}k{self.shape[2]}"
        layout_a = "col" if self.trans_a else "row"
        layout_b = "col" if self.trans_b else "row"
        mma_stmt += f".{layout_a}.{layout_b}"
        if self.satfinite:
            mma_stmt += ".satfinite"
        mma_stmt += f".{self.dtype_c.shortcut()}.{self.dtype_ab.shortcut()}.{self.dtype_ab.shortcut()}.{self.dtype_c.shortcut()}"

        return mma_stmt

    def short_repr(self) -> str:
        mma_stmt = f"m{self.shape[0]}n{self.shape[1]}k{self.shape[2]}"
        layout_a = "col" if self.trans_a else "row"
        layout_b = "col" if self.trans_b else "row"
        mma_stmt += f".{layout_a}.{layout_b}"
        if self.satfinite:
            mma_stmt += ".satfinite"
        mma_stmt += f".{self.dtype_c.shortcut()}.{self.dtype_ab.shortcut()}.{self.dtype_ab.shortcut()}.{self.dtype_c.shortcut()}"

        return mma_stmt

    def get_tensor_op(self):
        dtype_str = f"{self.dtype_ab.shortcut()},{self.dtype_ab.shortcut()},{self.dtype_c.shortcut()}"
        return TensorOp((self.shape[0], self.shape[1], self.shape[2]), dtype_str)

    def a_map(self):
        return self._map_a

    def b_map(self):
        return self._map_b

    def c_map(self):
        return self._map_c

    def a_sim_dtype(self):
        return self.dtype_ab.npdtype()

    def b_sim_dtype(self):
        return self.dtype_ab.npdtype()

    def c_sim_dtype(self):
        return self.dtype_c.npdtype()

    def get_empty_a(self):
        return np.zeros((self.shape[0], self.shape[2]), dtype=self.a_sim_dtype())

    def get_empty_b(self):
        return np.zeros((self.shape[2], self.shape[1]), dtype=self.b_sim_dtype())
    
    def get_empty_c(self):
        return np.zeros((self.shape[0], self.shape[1]), dtype=self.c_sim_dtype())


class MmaM8N8KX(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f64
    def __init__(self, k: int, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(8, 8, k), dtype_ab, dtype_c, False, True, False, min_sm)
        self._map_a[:, :, 0] = self.gid
        self._map_b[:, :, 1] = self.gid
        for i in range(k // 4):
            self._map_a[i, :, 1] = self.tid_in_group * (k // 4) + i
            self._map_b[i, :, 0] = self.tid_in_group * (k // 4) + i
        self._map_c[:, :, 0] = self.gid 
        for i in range(self.fragment_c_count):
            self._map_c[i, :, 1] = self.tid_in_group * 2 + i


class MmaM16N8K4(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1684
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType) -> None:
        super().__init__(seq(16, 8, 4), dtype_ab, dtype_c, False, True, False, (8, 0))
        
        self._map_a[0, :, 0] = self.gid
        self._map_a[1, :, 0] = self.gid + 8
        self._map_a[:, :, 1] = self.tid_in_group

        self._map_b[:, :, 0] = self.tid_in_group
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)


    def a_sim_dtype(self):
        return np.float32

    def b_sim_dtype(self):
        return np.float32

class MmaM16N8K8F16(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 8), dtype_ab, dtype_c, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_a[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)

        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group * 2 + i
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)

    def a_sim_dtype(self):
        return np.float16

    def b_sim_dtype(self):
        return np.float16

class MmaM16N8K8F32(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
    def __init__(self, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 8), dtypes.tf32, dtypes.float32, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * (i & 0x1)
            self._map_a[i, :, 1] = self.tid_in_group + 4 * (i >> 1)
        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group + 4 * i
        self._map_b[:, :, 1] = self.gid
        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)
            
    def a_sim_dtype(self):
        return np.float32

    def b_sim_dtype(self):
        return np.float32

class MmaM16N8K16F16(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 16), dtype_ab, dtype_c, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * ((i >> 1) & 0x1)
            self._map_a[i, :, 1] = self.tid_in_group * 2 + (i & 0x1) + 8 * (i >> 2)
        
        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group * 2 + (i & 0x1) + 8 * (i >> 1)
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)

    def a_sim_dtype(self):
        return np.float16

    def b_sim_dtype(self):
        return np.float16

class MmaM16N8K16I8(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-i8
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 16), dtype_ab, dtype_c, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * (i >> 2)
            self._map_a[i, :, 1] = self.tid_in_group * 4 + (i & 0x3)
        
        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group * 4 + i
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)


class MmaM16N8K32I4(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-i8
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 32), dtype_ab, dtype_c, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * (i >> 3)
            self._map_a[i, :, 1] = self.tid_in_group * 8 + (i & 0x7)
        
        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group * 8 + (i & 0x7)
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)


class MmaM16N8K32I8(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-i8
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 32), dtype_ab, dtype_c, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * ((i >> 2) & 0x1)
            self._map_a[i, :, 1] = self.tid_in_group * 4 + (i & 0x3) + 16 * (i >> 3)
        
        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group * 4 + (i & 0x3) + 16 * (i >> 2)
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)


class MmaM16N8K64I4(MmaLayoutDespBase):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16864
    def __init__(self, dtype_ab: dtypes.DType, dtype_c: dtypes.DType, min_sm: Tuple[int, int]) -> None:
        super().__init__(seq(16, 8, 64), dtype_ab, dtype_c, False, True, False, min_sm)
        for i in range(self.fragment_a_count):
            self._map_a[i, :, 0] = self.gid + 8 * ((i >> 3) & 0x1)
            self._map_a[i, :, 1] = self.tid_in_group * 8 + (i & 0x7) + 32 * (i >> 4)
        
        for i in range(self.fragment_b_count):
            self._map_b[i, :, 0] = self.tid_in_group * 8 + (i & 0x7) + 32 * (i >> 3)
        self._map_b[:, :, 1] = self.gid

        for i in range(self.fragment_c_count):
            self._map_c[i, :, 0] = self.gid + 8 * (i >> 1)
            self._map_c[i, :, 1] = self.tid_in_group * 2 + (i & 0x1)

ALL_TENSOR_OP_MAP: Dict[Tuple[Tuple[int, int, int], dtypes.DType, dtypes.DType], MmaLayoutDespBase] = {
    ((8, 8, 4), dtypes.float64, dtypes.float64): MmaM8N8KX(4, dtypes.float64, dtypes.float64, (8, 0)),
    ((8, 8, 16), dtypes.int8, dtypes.int32): MmaM8N8KX(16, dtypes.int8, dtypes.int32, (7, 5)),
    ((8, 8, 16), dtypes.uint8, dtypes.int32): MmaM8N8KX(16, dtypes.uint8, dtypes.int32, (7, 5)),
    ((8, 8, 32), dtypes.int8, dtypes.int32): MmaM8N8KX(32, dtypes.int8, dtypes.int32, (7, 5)),
    ((8, 8, 32), dtypes.uint8, dtypes.int32): MmaM8N8KX(32, dtypes.uint8, dtypes.int32, (7, 5)),

    # ((16, 8, 4), dtypes.float32, dtypes.float32): MmaM16N8K4(dtypes.float32, dtypes.float32),
    ((16, 8, 4), dtypes.tf32, dtypes.float32): MmaM16N8K4(dtypes.tf32, dtypes.float32),

    ((16, 8, 8), dtypes.float16, dtypes.float16): MmaM16N8K8F16(dtypes.float16, dtypes.float16, (7, 5)),
    ((16, 8, 8), dtypes.float16, dtypes.float32): MmaM16N8K8F16(dtypes.float16, dtypes.float32, (7, 5)),

    ((16, 8, 8), dtypes.bfloat16, dtypes.float16): MmaM16N8K8F16(dtypes.bfloat16, dtypes.float16, (8, 0)),
    ((16, 8, 8), dtypes.bfloat16, dtypes.float32): MmaM16N8K8F16(dtypes.bfloat16, dtypes.float32, (8, 0)),

    # ((16, 8, 8), dtypes.float32, dtypes.float32): MmaM16N8K8F32((8, 0)),
    ((16, 8, 8), dtypes.tf32, dtypes.float32): MmaM16N8K8F32((8, 0)),

    ((16, 8, 16), dtypes.float16, dtypes.float16): MmaM16N8K16F16(dtypes.float16, dtypes.float16, (7, 5)),
    ((16, 8, 16), dtypes.float16, dtypes.float32): MmaM16N8K16F16(dtypes.float16, dtypes.float32, (7, 5)),

    ((16, 8, 16), dtypes.bfloat16, dtypes.float16): MmaM16N8K16F16(dtypes.bfloat16, dtypes.float16, (8, 0)),
    ((16, 8, 16), dtypes.bfloat16, dtypes.float32): MmaM16N8K16F16(dtypes.bfloat16, dtypes.float32, (8, 0)),
    
    ((16, 8, 16), dtypes.int8, dtypes.int32): MmaM16N8K16I8(dtypes.int8, dtypes.int32, (8, 0)),
    ((16, 8, 16), dtypes.uint8, dtypes.int32): MmaM16N8K16I8(dtypes.uint8, dtypes.int32, (8, 0)),

    ((16, 8, 32), dtypes.int4, dtypes.int32): MmaM16N8K32I4(dtypes.int4, dtypes.int32, (8, 0)),
    ((16, 8, 32), dtypes.uint4, dtypes.int32): MmaM16N8K32I4(dtypes.uint4, dtypes.int32, (8, 0)),

    ((16, 8, 32), dtypes.int8, dtypes.int32): MmaM16N8K32I8(dtypes.int8, dtypes.int32, (8, 0)),
    ((16, 8, 32), dtypes.uint8, dtypes.int32): MmaM16N8K32I8(dtypes.uint8, dtypes.int32, (8, 0)),

    ((16, 8, 64), dtypes.int4, dtypes.int32): MmaM16N8K64I4(dtypes.int4, dtypes.int32, (8, 0)),
    ((16, 8, 64), dtypes.uint4, dtypes.int32): MmaM16N8K64I4(dtypes.uint4, dtypes.int32, (8, 0)),
}

def get_tensorop_desp(shape: MetaArray[int], dtype_ab: dtypes.DType, dtype_c: dtypes.DType):
    shape_tuple = (shape[0], shape[1], shape[2])
    return ALL_TENSOR_OP_MAP[(shape_tuple, dtype_ab, dtype_c)]

class MmaSync(GemmComponentBase):
    """
    Volta: [8, 8, 4]
    Turing: tnt, [16, 8, 8] f16f16[f16|f32]
    [8, 8, 16] [u8|s8][u8|s8]s32
    [8, 8, 32] [u4|s4][u4|s4]s32

    Requires sm_70 or higher.

    Note:mma.sync.m8n8k4 is optimized for target architecture sm_70 and may have substantially reduced performance on other target architectures.
    Shapes .m16n8k8, .m16n8k16, .m8n8k128, .m8n8k16 and .m8n8k32 require sm_75 or higher.

    Alternate floating point types .bf16 and .tf32 on shape .m16n8k8 require sm_80 or higher.

    Shapes .m16n8k4, .m16n8k32, .m16n8k64, .m16n8k128 and .m16n8k256 require sm_80 or higher.

    Shapes .m8n8k4 with .f64 floating point type require sm_80 or higher.

    .and operation in single-bit mma requires sm_80 or higher.

    m16n8k16 tf32 only available in sparse kernel
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
    """
    def __init__(self,
                 shape: MetaArray[int],
                 num_threads: int,
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_c: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 satfinite: bool = False):
        super().__init__()
        self.add_dependency(TensorViewNVRTCKernel, GemmBasic)
        self.shape = shape
        self.num_threads = num_threads
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.satfinite = satfinite
        self.mn = shape[0] * shape[1]
        self.km = shape[2] * shape[0]
        self.kn = shape[2] * shape[1]
        self.is_volta = shape[0] == 8 and shape[1] == 8 and shape[2] == 4 and num_threads == 8
        if num_threads == 8:
            assert shape[0] == 8 and shape[1] == 8 and shape[2] == 4
            assert self.trans_c is False

        self.fragment_c_count = self.mn // self.num_threads
        self.fragment_a_count = self.km // self.num_threads
        self.fragment_b_count = self.kn // self.num_threads
        self.fragment_a_t = array_type(str(dtype_a), self.fragment_a_count)
        self.fragment_b_t = array_type(str(dtype_b), self.fragment_b_count)
        self.fragment_c_t = array_type(str(dtype_c), self.fragment_c_count)
        if num_threads == 8:
            self.tensorop_desp = None # no volta support
        else:
            self.tensorop_desp = get_tensorop_desp(shape, dtype_a, dtype_c)

    def min_arch(self) -> Optional[Tuple[int, int]]:
        if self.is_volta:
            return (7, 0)
        else:
            assert self.tensorop_desp is not None
            return self.tensorop_desp.min_sm

    def python_ctor(self):
        return self

    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator(self):
        mma_stmt = f"mma.sync.aligned.m{self.shape[0]}n{self.shape[1]}k{self.shape[2]}"
        layout_a = "col" if self.trans_a else "row"
        layout_b = "col" if self.trans_b else "row"

        mma_stmt += f".{layout_a}.{layout_b}"
        if self.satfinite:
            mma_stmt += ".satfinite"
        mma_stmt += f".{self.dtype_c.shortcut()}.{self.dtype_a.shortcut()}.{self.dtype_b.shortcut()}.{self.dtype_c.shortcut()}"

        # compute number of inputs
        num_32_registers_out = self.dtype_c.bitsize(
        ) * self.mn // self.num_threads // 32
        num_32_registers_a = self.dtype_a.bitsize(
        ) * self.km // self.num_threads // 32
        num_32_registers_b = self.dtype_b.bitsize(
        ) * self.kn // self.num_threads // 32
        out_label = "f" if self.dtype_c == dtypes.float32 else "r"
        cnt = 0
        # register names
        out_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_out)])
        cnt += num_32_registers_out
        a_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_a)])
        cnt += num_32_registers_a
        b_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_b)])
        cnt += num_32_registers_b
        c_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_out)])
        cnt += num_32_registers_out
        code = pccm.code()

        # output operands
        out_operands = ", ".join(f"\"={out_label}\"(D[{i}])"
                                 for i in range(num_32_registers_out))
        # input operands
        if self.shape[0] == 8 and self.shape[1] == 8 and self.shape[2] == 4:
            code.raw("""
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
            """)
        else:
            code.raw("""
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
            """)
        if num_32_registers_a == 1:
            a_operands = "\"r\"(A)"
            code.raw(
                "unsigned const & A = reinterpret_cast<unsigned const &>(a);")
        else:
            a_operands = ", ".join(f"\"r\"(A[{i}])"
                                   for i in range(num_32_registers_a))
            code.raw(
                "unsigned const *A = reinterpret_cast<unsigned const *>(&a);")
        if num_32_registers_b == 1:
            b_operands = "\"r\"(B)"
            code.raw(
                "unsigned const & B = reinterpret_cast<unsigned const &>(b);")
        else:
            b_operands = ", ".join(f"\"r\"(B[{i}])"
                                   for i in range(num_32_registers_b))
            code.raw(
                "unsigned const *B = reinterpret_cast<unsigned const *>(&b);")

        if self.dtype_c == dtypes.float32:
            code.raw("float const *C = reinterpret_cast<float const *>(&c);")
            code.raw("float *D = reinterpret_cast<float *>(&d);")
        else:
            code.raw(
                "unsigned const *C = reinterpret_cast<unsigned const *>(&c);")
            code.raw("unsigned *D = reinterpret_cast<unsigned *>(&d);")
        c_operands = ", ".join(f"\"{out_label}\"(C[{i}])"
                               for i in range(num_32_registers_out))
        code.raw(f"""
        asm volatile("{mma_stmt} {{{out_register_names}}}, {{{a_register_names}}}, {{{b_register_names}}}, {{{c_register_names}}};\\n"
            : {out_operands}
            : {a_operands}, {b_operands}, {c_operands});
        """)
        code.raw("#endif")
        code.arg("d", f"{self.fragment_c_t}&")
        code.arg("a", f"{self.fragment_a_t} const &")
        code.arg("b", f"{self.fragment_b_t} const &")
        code.arg("c", f"{self.fragment_c_t} const &")
        return code

    async def __call__(self, d: ArrayPtr, a: ArrayPtr, b: ArrayPtr,
                       c: ArrayPtr):

        lane_id = cudasim.get_lane_id()
        warp_id = cudasim.get_warp_id()
        resource = cudasim.get_warp_resource()
        warp_data = await resource.gather(
            lane_id, (d, a, b, c),
            0)  # type: List[Tuple[ArrayPtr, ArrayPtr, ArrayPtr,ArrayPtr]]
        dabc = (self.dtype_a, self.dtype_b, self.dtype_c)
        dab = (self.dtype_a, self.dtype_b)

        if lane_id == 0:
            Ds = [x[0].data.numpy_view() for x in warp_data]
            As = [x[1].data.numpy_view() for x in warp_data]
            Bs = [x[2].data.numpy_view() for x in warp_data]
            Cs = [x[3].data.numpy_view() for x in warp_data]
            if self.shape == (8, 8, 4):
                w_off = [0, 4, 8, 12]
                # Ds_ten = np.stack(Ds)
                # Ds_ten[:] = 0
                # As_ten = np.stack(As)
                # Bs_ten = np.stack(Bs)
                # Cs_ten = np.stack(Cs)
                # MmaSyncTester.mma_test(tv.from_numpy(As_ten), tv.from_numpy(Bs_ten),
                #     tv.from_numpy(Cs_ten), tv.from_numpy(Ds_ten))
                # for i in range(len(Ds)):
                #     Ds[i][:] = Ds_ten[i]
                # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
                if not self.trans_a:
                    A_qp0 = np.stack(As[w_off[0]:w_off[0] + 4] +
                                     As[w_off[0] + 16:w_off[0] + 16 + 4])
                    A_qp1 = np.stack(As[w_off[1]:w_off[1] + 4] +
                                     As[w_off[1] + 16:w_off[1] + 16 + 4])
                    A_qp2 = np.stack(As[w_off[2]:w_off[2] + 4] +
                                     As[w_off[2] + 16:w_off[2] + 16 + 4])
                    A_qp3 = np.stack(As[w_off[3]:w_off[3] + 4] +
                                     As[w_off[3] + 16:w_off[3] + 16 + 4])
                else:
                    A_qp0 = np.concatenate([
                        np.stack(As[w_off[0]:w_off[0] + 4]).T,
                        np.stack(As[w_off[0] + 16:w_off[0] + 16 + 4]).T
                    ])
                    A_qp1 = np.concatenate([
                        np.stack(As[w_off[1]:w_off[1] + 4]).T,
                        np.stack(As[w_off[1] + 16:w_off[1] + 16 + 4]).T
                    ])
                    A_qp2 = np.concatenate([
                        np.stack(As[w_off[2]:w_off[2] + 4]).T,
                        np.stack(As[w_off[2] + 16:w_off[2] + 16 + 4]).T
                    ])
                    A_qp3 = np.concatenate([
                        np.stack(As[w_off[3]:w_off[3] + 4]).T,
                        np.stack(As[w_off[3] + 16:w_off[3] + 16 + 4]).T
                    ])
                if not self.trans_b:
                    B_qp0 = np.concatenate([
                        np.stack(Bs[w_off[0]:w_off[0] + 4]),
                        np.stack(Bs[w_off[0] + 16:w_off[0] + 16 + 4])
                    ],
                                           axis=1)
                    B_qp1 = np.concatenate([
                        np.stack(Bs[w_off[1]:w_off[1] + 4]),
                        np.stack(Bs[w_off[1] + 16:w_off[1] + 16 + 4])
                    ],
                                           axis=1)
                    B_qp2 = np.concatenate([
                        np.stack(Bs[w_off[2]:w_off[2] + 4]),
                        np.stack(Bs[w_off[2] + 16:w_off[2] + 16 + 4])
                    ],
                                           axis=1)
                    B_qp3 = np.concatenate([
                        np.stack(Bs[w_off[3]:w_off[3] + 4]),
                        np.stack(Bs[w_off[3] + 16:w_off[3] + 16 + 4])
                    ],
                                           axis=1)
                else:
                    B_qp0 = np.stack(Bs[w_off[0]:w_off[0] + 4] +
                                     Bs[w_off[0] + 16:w_off[0] + 16 + 4]).T
                    B_qp1 = np.stack(Bs[w_off[1]:w_off[1] + 4] +
                                     Bs[w_off[1] + 16:w_off[1] + 16 + 4]).T
                    B_qp2 = np.stack(Bs[w_off[2]:w_off[2] + 4] +
                                     Bs[w_off[2] + 16:w_off[2] + 16 + 4]).T
                    B_qp3 = np.stack(Bs[w_off[3]:w_off[3] + 4] +
                                     Bs[w_off[3] + 16:w_off[3] + 16 + 4]).T
                # print(A_qp0.shape, B_qp0.shape)

                # assume A always 8x4, B always 4x8
                D_qp0_res = (A_qp0 @ B_qp0)
                D_qp1_res = (A_qp1 @ B_qp1)
                D_qp2_res = (A_qp2 @ B_qp2)
                D_qp3_res = (A_qp3 @ B_qp3)
                C_res = [np.zeros_like(D_qp0_res) for _ in range(4)]
                if dabc == (dtypes.float16, dtypes.float16, dtypes.float32):
                    for qp_idx in range(4):
                        C_qp_res = C_res[qp_idx]
                        for lane_id_mma in range(w_off[qp_idx],
                                                 w_off[qp_idx] + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10)
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                C_qp_res[row, col] = Cs[lane_id_mma][i]
                        for lane_id_mma in range(w_off[qp_idx] + 16,
                                                 w_off[qp_idx] + 16 + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10) + 4
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                C_qp_res[row, col] = Cs[lane_id_mma][i]
                    C_qp0, C_qp1, C_qp2, C_qp3 = C_res
                else:
                    C_qp0 = np.stack(Cs[w_off[0]:w_off[0] + 4] +
                                     Cs[w_off[0] + 16:w_off[0] + 16 + 4])
                    C_qp1 = np.stack(Cs[w_off[1]:w_off[1] + 4] +
                                     Cs[w_off[1] + 16:w_off[1] + 16 + 4])
                    C_qp2 = np.stack(Cs[w_off[2]:w_off[2] + 4] +
                                     Cs[w_off[2] + 16:w_off[2] + 16 + 4])
                    C_qp3 = np.stack(Cs[w_off[3]:w_off[3] + 4] +
                                     Cs[w_off[3] + 16:w_off[3] + 16 + 4])
                D_qp0_res += C_qp0
                D_qp1_res += C_qp1
                D_qp2_res += C_qp2
                D_qp3_res += C_qp3

                D_res = [D_qp0_res, D_qp1_res, D_qp2_res, D_qp3_res]
                D_res = [ddd.astype(self.dtype_c.npdtype()) for ddd in D_res]
                if dabc == (dtypes.float16, dtypes.float16, dtypes.float32):
                    for qp_idx in range(4):
                        D_qp_res = D_res[qp_idx]
                        for lane_id_mma in range(w_off[qp_idx],
                                                 w_off[qp_idx] + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10)
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                Ds[lane_id_mma][i] = D_qp_res[row, col]
                        for lane_id_mma in range(w_off[qp_idx] + 16,
                                                 w_off[qp_idx] + 16 + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10) + 4
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                Ds[lane_id_mma][i] = D_qp_res[row, col]

                else:
                    for qp_idx in range(4):
                        D_qp_res = D_res[qp_idx]
                        for lane_id_mma in range(w_off[qp_idx],
                                                 w_off[qp_idx] + 4):
                            for i in range(8):
                                row = lane_id_mma % 4
                                col = i
                                Ds[lane_id_mma][i] = D_qp_res[row, col]
                        for lane_id_mma in range(w_off[qp_idx] + 16,
                                                 w_off[qp_idx] + 16 + 4):
                            for i in range(8):
                                row = lane_id_mma % 4 + 4
                                col = i
                                Ds[lane_id_mma][i] = D_qp_res[row, col]
            else:
                assert self.tensorop_desp is not None 
                a_map = self.tensorop_desp.a_map()
                b_map = self.tensorop_desp.b_map()
                c_map = self.tensorop_desp.c_map()

                mma_A = self.tensorop_desp.get_empty_a()
                mma_B = self.tensorop_desp.get_empty_b()
                mma_C = self.tensorop_desp.get_empty_c()
                As_mat = np.stack(As)
                Bs_mat = np.stack(Bs)
                Cs_mat = np.stack(Cs)

                for i in range(self.tensorop_desp.fragment_a_count):
                    mma_A[a_map[i, :, 0], a_map[i, :, 1]] = As_mat[:, i]
                for i in range(self.tensorop_desp.fragment_b_count):
                    mma_B[b_map[i, :, 0], b_map[i, :, 1]] = Bs_mat[:, i]
                for i in range(self.tensorop_desp.fragment_c_count):
                    mma_C[c_map[i, :, 0], c_map[i, :, 1]] = Cs_mat[:, i]
                mma_D = mma_A @ mma_B + mma_C
                for i in range(self.tensorop_desp.fragment_c_count):
                    for j in range(self.tensorop_desp.num_threads):
                        Ds[j][i] = mma_D[c_map[i, j, 0], c_map[i, j, 1]]
        # wait for compute finish
        await resource.wait()
        return


