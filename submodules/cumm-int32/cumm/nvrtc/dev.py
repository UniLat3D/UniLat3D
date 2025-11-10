import pccm

from cumm import tensorview as tv
from cumm.common import TensorViewNVRTCKernel
from cumm.nvrtc import CummNVRTCModule


class SimpleKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTCKernel)
        self.add_include("cuda/std/cassert")

    @pccm.cuda.cuda_global_function
    def add(self):
        code = pccm.code()
        code.arg("x", "float*")
        code.arg("n", "int")
        code.arg("arr", "tv::array<tv::array<float, 3>, 4>")

        code.raw(f"""
        tv::array<float, 3> a, b;
        tv::array<float, 2> c;
        b = b + a;
        c = tv::arrayops::slice<0, 2>(a);
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;  
        if (tid < n) {{
            x[tid] += 1;
        }}
        """)
        return code


if __name__ == "__main__":
    import numpy as np
    m = CummNVRTCModule([SimpleKernel()])
    # print(m.program.ptx())
    print(m.name_to_meta)
    a = tv.zeros([2], tv.float32, 0)
    arr = np.zeros((4, 3), dtype=np.float32)
    launch = m.get_launch_param([1, 1, 1], [128, 1, 1])
    m.run_kernel("cumm::nvrtc::dev::add", launch, a, 2, arr)
    print(a.cpu().numpy())
