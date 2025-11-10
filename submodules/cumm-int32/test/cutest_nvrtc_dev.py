import pickle

from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT, TENSORVIEW_INCLUDE_PATH
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
import numpy as np

from cumm.nvrtc import CummNVRTCModule 

def test_nvrtc_std():
    # init cuda
    a = tv.zeros([3], tv.float32, 0)
    inliner = NVRTCInlineBuilder([TensorViewNVRTCHashKernel, TensorViewArrayLinalg], root=PACKAGE_ROOT.parent, std="c++17")
    
    inliner.kernel_1d("nvrtc_std", 1, 0, 
                      f"""
    float x = 0.356632;
    float y = 0.346854;
    float z = 0.998650;
    float scale = 15.0f;
    float val = fmaf(scale, z, 0.0f);
    tv::printf2(val, val*val*(3.0f - 2.0f * val));

    
    """)
    # print(inliner.get_nvrtc_kernel_attrs("nvrtc_std"))

    # mod = CummNVRTCModule([TensorViewNVRTCDev()], verbose=True)


if __name__ == "__main__":
    test_nvrtc_std()
