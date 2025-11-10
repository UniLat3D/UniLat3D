
import importlib.util 
from pathlib import Path
from typing import List, Optional 

from ccimport import compat
import pccm 

from .common import CUDALibs, TensorView

class PyTorchLibNoPybind(pccm.Class):
    def __init__(self):
        super().__init__()
        spec = importlib.util.find_spec("torch")
        if spec is None:
            raise ValueError(
                "you need to install torch python")
        origin = Path(spec.origin)
        libtorch = origin.parent
        self.add_dependency(CUDALibs, TensorView)

        self.build_meta.add_public_includes(str(libtorch / "include"))
        self.build_meta.add_public_includes(str(libtorch / "include/torch/csrc/api/include"))
        torch_lib_paths = [str(libtorch / "lib")]
        torch_libs = ["c10", "torch", 'torch_cpu', 'torch_python']
        if not compat.InMacOS:
            torch_cuda_libs = ["c10_cuda", "torch_cuda"]
        else:
            torch_cuda_libs = []
        self.build_meta.libraries.extend(torch_libs + torch_cuda_libs)
        self.build_meta.libpaths.extend(torch_lib_paths)
        self.build_meta.add_public_cflags("nvcc,clang++,g++", "-D_GLIBCXX_USE_CXX11_ABI=0")
        self.add_include("torch/script.h")
        if not compat.InMacOS:
            self.add_include("ATen/cuda/CUDAContext.h")
        self.add_include("ATen/ATen.h")
        if compat.InMacOS:
            self.build_meta.add_ldflags("clang++", "-Wl,-undefined,dynamic_lookup")
            self.build_meta.add_ldflags("clang++", "-framework Metal", "-framework CoreGraphics", "-framework Foundation")

class PyTorchLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(PyTorchLibNoPybind)
        self.add_include("torch/extension.h") # include this to add pybind for torch.Tensor
        self.add_include("tensorview/torch_utils.h")

class PyTorchMPSInclude(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("ATen/native/mps/OperationUtils.h")

class PyTorchTools(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(PyTorchLib)

    @pccm.pybind.mark
    @pccm.static_function
    def torch2tensor(self):
        code = pccm.FunctionCode()
        code.arg("ten", "torch::Tensor", pyanno="~torch.Tensor")
        code.raw(f"""
        return tv::torch2tensor(ten);
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark
    @pccm.static_function
    def tensor2torch(self):
        code = pccm.FunctionCode()
        code.arg("ten", "tv::Tensor")
        code.arg("clone", "bool", "true")
        code.arg("cast_uint_to_int", "bool", "false")

        code.raw(f"""
        auto res = tv::tensor2torch(ten, cast_uint_to_int);
        if (clone){{
            res = res.clone();
        }}
        return res;
        """)
        return code.ret("torch::Tensor", "~torch.Tensor")

    @pccm.pybind.mark
    @pccm.static_function
    def mps_get_default_command_buffer(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        #ifdef __APPLE__
            return reinterpret_cast<std::uintptr_t>(torch::mps::get_command_buffer());
        #else   
            return 0;
        #endif
        """)
        return code.ret("std::uintptr_t")

    @pccm.pybind.mark
    @pccm.static_function
    def mps_get_default_dispatch_queue(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        #ifdef __APPLE__
            return reinterpret_cast<std::uintptr_t>(torch::mps::get_dispatch_queue());
        #else   
            return 0;
        #endif
        """)
        return code.ret("std::uintptr_t")

    @pccm.pybind.mark
    @pccm.static_function
    def mps_commit(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        #ifdef __APPLE__
            return torch::mps::commit();
        #endif
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function(impl_file_suffix=".mm" if compat.InMacOS else ".cc")
    def mps_flush_command_encoder(self):
        # pytorch mps kernels won't clear command encoding, so we need to flush it.
        code = pccm.FunctionCode()
        if compat.InMacOS:
            code.add_dependency(PyTorchMPSInclude)
            code.raw(f"""
            getCurrentMPSStream()->endKernelCoalescing();
            """)
        return code


def build_pytorch_bindings(name: str, lib_store_root: Path, custom_cus: Optional[List[pccm.Class]] = None):
    import torch 
    torch_version = torch.__version__.split("+")[0]
    torch_version = torch_version.replace(".", "_")
    cu = PyTorchTools()
    cu.namespace = "pytorch_tools"
    if custom_cus is None:
        custom_cus = []
    pccm.builder.build_pybind([cu] + custom_cus,
                            lib_store_root / name,
                            build_dir=lib_store_root / "build" / f"{name}_{torch_version}",
                            # namespace_root=PACKAGE_ROOT / "csrc",
                            verbose=True,
                            load_library=False,
                            disable_pch=True, # object-c don't support pch
                            std="c++17")
