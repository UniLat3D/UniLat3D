#!/home/yy/library/anaconda3/bin/python
import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pccm

from cumm import dtypes
from cumm import tensorview as tv
from cumm.common import CUDALibs, GemmBasic, TensorView
from cumm.gemm.algospec.core import GemmAlgo, TensorOp
from cumm.gemm.main import GemmAlgoParams, GemmMainUnitTest
from cumm import cudasim 

CUTLASS_ROOT = Path("/home/yy/Projects/cutlass")
import numpy as np


class CutlassLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(CUDALibs)
        self.add_include("cutlass/cutlass.h")
        self.add_include("cutlass/gemm/device/gemm.h")
        self.add_include("random")
        self.build_meta.add_public_includes(CUTLASS_ROOT / "include")
        self.build_meta.add_public_includes(CUTLASS_ROOT / "tools/util/include")
        if cudasim.enable_debug():
            self.build_meta.add_cflags("nvcc", f"-DCUMM_DEBUG_TX={cudasim.debug_tx()}")


class CutlassGemm(pccm.ParameterizedClass):
    def __init__(self, params: GemmAlgoParams, num_out_width: int, arch: str):
        super().__init__()
        self.params = params
        self.num_out_width = num_out_width
        self.arch = arch
        self.add_dependency(TensorView, GemmBasic)
        self.add_impl_only_dependency(self.matmul, CutlassLib)
        # self.add_include("tensorview/simple_ops.h")

    def cutlass_layout(self, trans: bool):
        return "cutlass::layout::ColumnMajor" if trans else "cutlass::layout::RowMajor"

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def matmul(self):
        code = pccm.code()
        code.arg("a,b", "tv::Tensor", pyanno="cumm.tensorview.Tensor")
        code.arg("split_k_slices", "int", default="1")

        op = "cutlass::arch::OpClassTensorOp"
        if self.params.tensorop is None or self.params.tensorop.shape[0] == 1:
            op = "cutlass::arch::OpClassSimt"
        ts = self.params.ts
        wts = self.params.wts
        top_shape = [1, 1, 1]
        if self.params.tensorop is not None:
            top_shape = self.params.tensorop.shape
        code.raw(f"""
        using ElementAccumulator = {self.params.dtype_acc.cutlass};        
        using ElementComputeEpilogue = {self.params.dtype_comp.cutlass}; 
        using ElementInputA = {self.params.dtype_a.cutlass};
        using ElementInputB = {self.params.dtype_b.cutlass};
        using ElementOutput = {self.params.dtype_c.cutlass};

        using LayoutInputA = {self.cutlass_layout(self.params.trans_a)};
        using LayoutInputB = {self.cutlass_layout(self.params.trans_b)};
        using LayoutOutput = {self.cutlass_layout(self.params.trans_c)};

        using MMAOp = {op};

        // This code section describes CUDA SM architecture number
        using SmArch = cutlass::arch::{self.arch};

        // This code section describes the tile size a thread block will compute
        using ShapeMMAThreadBlock =
            cutlass::gemm::GemmShape<{ts[0]}, {ts[1]}, {ts[2]}>;
        // This code section describes tile size a warp will compute
        using ShapeMMAWarp = cutlass::gemm::GemmShape<{wts[0]}, {wts[1]}, {wts[2]}>; 
        // This code section describes the size of MMA op
        using ShapeMMAOp = cutlass::gemm::GemmShape<{top_shape[0]}, {top_shape[1]}, {top_shape[2]}>;

        // This code section describes how threadblocks are scheduled on GPU
        using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

        // This code section describes the epilogue part of the kernel
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput,                                     // <- data type of output matrix
            {self.num_out_width} / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                            // memory access. For a byte, it's 16
                                                            // elements. This becomes the vector width of
                                                            // math instructions in the epilogue too
            ElementAccumulator,                                // <- data type of accumulator
            ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function
        constexpr int NumStages = {self.params.num_stage};

        using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                LayoutInputA,
                                                ElementInputB,
                                                LayoutInputB,
                                                ElementOutput,
                                                LayoutOutput,
                                                ElementAccumulator,
                                                MMAOp,
                                                SmArch,
                                                ShapeMMAThreadBlock,
                                                ShapeMMAWarp,
                                                ShapeMMAOp,
                                                EpilogueOp,
                                                SwizzleThreadBlock,
                                                NumStages>;
        
        ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
        ElementComputeEpilogue beta = ElementComputeEpilogue(0);
        bool ta = {pccm.boolean(self.params.trans_a)};
        bool tb = {pccm.boolean(self.params.trans_b)};
        bool tc = {pccm.boolean(self.params.trans_c)};

        auto m = a.dim(int(ta));
        auto k = a.dim(int(!ta));
        auto k2 = b.dim(int(tb));
        TV_ASSERT_INVALID_ARG(k == k2, "error");
        auto n = b.dim(int(!tb));
        tv::Tensor c({{tc ? n : m, tc ? m : n}}, tv::type_v<{self.params.dtype_c}>, a.device());
        auto lda = a.size(1);
        auto ldb = b.size(1);
        auto ldc = c.size(1);

        ElementInputA *a_ptr = reinterpret_cast<ElementInputA *>(a.raw_data());
        ElementInputB *b_ptr = reinterpret_cast<ElementInputB *>(b.raw_data());
        ElementOutput *c_ptr = reinterpret_cast<ElementOutput *>(c.raw_data());

        typename Gemm::Arguments args(
            {{m, n, k}},         // Gemm Problem dimensions
            {{a_ptr, lda}},      // Tensor-ref for source matrix A
            {{b_ptr, ldb}},      // Tensor-ref for source matrix B
            {{c_ptr, ldc}},      // Tensor-ref for source matrix C
            {{c_ptr, ldc}},      // Tensor-ref for destination matrix D (may be
                                // different memory than source C matrix)
            {{alpha, beta}}, split_k_slices); // Scalars used in the Epilogue
        size_t workspace_size = Gemm::get_workspace_size(args);
        tv::Tensor workspace;
        if (workspace_size > 0){{
            workspace = tv::Tensor({{int(workspace_size)}}, tv::uint8, 0);
        }}
        Gemm gemm_op ;
        // Check the problem size is supported or  not 
        cutlass::Status status = gemm_op.can_implement(args);
        TV_ASSERT_INVALID_ARG(status == cutlass::Status::kSuccess, "error");
        if (workspace_size > 0){{
            status = gemm_op.initialize(args, workspace.raw_data());
        }}else{{
            status = gemm_op.initialize(args, nullptr);

        }}
        TV_ASSERT_INVALID_ARG(status == cutlass::Status::kSuccess, "error");
        // Launch initialized CUTLASS kernel
        auto timer = tv::CudaContextTimer<>();

        status = gemm_op();
        tv::ssprint(workspace_size, "cut time", timer.report() / 1000.0);

        TV_ASSERT_INVALID_ARG(status == cutlass::Status::kSuccess, "error");
        // checkCudaErrors(cudaDeviceSynchronize());
        return c;
        """)
        return code.ret("tv::Tensor")


def cutlass_build_exec(cu: CutlassGemm):
    params = cu.params
    cu.namespace = "CuTlassTestExec"
    output = Path(__file__).parent / "cutlass_test_exec"
    lib = pccm.builder.build_pybind([cu],
                                    Path(__file__).parent / "cutlassgemm_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_cutlass",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True,
                                    std="c++17")

    return output


def cutlass_profile_win(cu: CutlassGemm):
    out_file_name = cutlass_build_exec(cu)
    ncu_sections = [
        "ComputeWorkloadAnalysis", "InstructionStats", "LaunchStats",
        "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Chart",
        "MemoryWorkloadAnalysis_Tables", "Occupancy", "SchedulerStats",
        "SourceCounters", "SpeedOfLight", "WarpStateStats"
    ]
    section_flags = sum([["--section", s] for s in ncu_sections], [])
    cmds = [
        "ncu", "-o",
        str((out_file_name).parent / "profile"), *section_flags, "-f",
        str(out_file_name) + ".exe"
    ]
    print(" ".join(cmds))
    subprocess.check_call(cmds, env=os.environ, shell=True)
    target = "/home/yy"
    upload_file("127.0.0.1:50073",
                Path(__file__).parent / "profile.ncu-rep",
                target + "/profile.ncu-rep",
                exist_ok=True)
    return


def cutlass_test_gemm(cu: CutlassGemm, spk: int = 1):
    params = cu.params
    cu.namespace = "CuTlassTest"
    with tv.measure_and_print():
        lib = pccm.builder.build_pybind([cu],
                                        Path(__file__).parent / "cutlassgemm_test",
                                        build_dir=Path(__file__).parent / "build" /
                                        "build_cutlass",
                                        pybind_file_suffix=".cc",
                                        verbose=False,
                                        disable_anno=True,
                                        std="c++17")
    np.random.seed(12315)
    m = 256 + 32
    n = 256 + 40
    k = 136
    m *= 2
    n *= 2
    k *= 2
    m = 128
    n = 128
    k = 8
    m = max(params.ts[0], m)
    n = max(params.ts[1], n)
    k = max(params.ts[2], k)

    if params.dtype_a == dtypes.int8:
        a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
        b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
        dtype_c = params.dtype_c.npdtype()
        c = (a.astype(dtype_c) @ b.astype(dtype_c)).astype(
            dtypes.get_npdtype(params.dtype_c))

    else:
        a = np.random.uniform(-1, 1, size=[m, k]).astype(
            dtypes.get_npdtype(params.dtype_a))
        b = np.random.uniform(-1, 1, size=[k, n]).astype(
            dtypes.get_npdtype(params.dtype_b))
        c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
    if params.trans_a:
        a = np.ascontiguousarray(a.transpose(1, 0))
    if params.trans_b:
        b = np.ascontiguousarray(b.transpose(1, 0))
    if params.trans_c:
        c = np.ascontiguousarray(c.transpose(1, 0))
    # print("WTF PREPARED")

    a_tv = tv.from_numpy(a).cuda()
    b_tv = tv.from_numpy(b).cuda()
    for i in range(1):
        c_tv = lib.CuTlassTest.CutlassGemm.matmul(a_tv, b_tv, 1)
        c_cpu = c_tv.cpu().numpy()
        print(m, n, k, a.mean(), b.mean(), c.mean(),
              np.linalg.norm(c_cpu - c))


def cutlass_test_simt():
    params = GemmAlgoParams((32, 512, 8), (32, 64, 8), 2,
                            "f32,f32,f32,f32,f32", False, False, False,
                            GemmAlgo.Simt, TensorOp((1, 1, 1)))
    main_cu = CutlassGemm(params, 32, "Sm61")
    cutlass_test_gemm(main_cu)


def cutlass_test_simt_dp4a():
    params = GemmAlgoParams((128, 64, 32), (64, 32, 32), 2,
                            "s8,s8,s32,s32,s32", False, True, False,
                            GemmAlgo.SimtDP4A, TensorOp((1, 1, 4)))
    main_cu = CutlassGemm(params, 32, "Sm61")
    cutlass_test_gemm(main_cu)


def cutlass_test_turing():
    params = GemmAlgoParams((64, 128, 64), (32, 64, 32), 2,
                            "f16,f16,f16,f32,f32", False, True, False,
                            GemmAlgo.Turing, TensorOp((16, 8, 8)))
    main_cu = CutlassGemm(params, 128, "Sm75")
    cutlass_test_gemm(main_cu)

def cutlass_test_tf32():
    params = GemmAlgoParams((128, 128, 16), (64, 64, 16), 2,
                            "f32,f32,f32,f32,f32", False, True, False,
                            GemmAlgo.Ampere, TensorOp((16, 8, 8)))
    main_cu = CutlassGemm(params, 128, "Sm80")
    cutlass_test_gemm(main_cu)

if __name__ == "__main__":
    cutlass_test_tf32()
    # cutlass_profile_win_simt()

    # cutlass_profile_win_simt_f32()
    # cutlass_profile_win_turing()
