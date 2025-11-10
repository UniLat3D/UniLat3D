import pickle
import traceback

from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT, TENSORVIEW_INCLUDE_PATH
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, EigenLib
import numpy as np 
from cumm.perftools import perf_context

def test_nvrtc_process():
    inliner = NVRTCInlineBuilder([TensorViewNVRTCHashKernel, TensorViewArrayLinalg], root=PACKAGE_ROOT.parent, std="c++17")
    # init driver
    keys = tv.zeros([50], tv.int32, 0)
    try:
        inliner.kernel_1d("run_in_child_process_to_debug_unrecoverable_cuda_errors", keys.shape[0] + 5000000, 0, f"""
        namespace op = tv::arrayops;
        $keys[i] = 1;
        """, run_in_process=True)
    except:
        traceback.print_exc()
    inliner.kernel_1d("run_in_main_process", keys.shape[0], 0, f"""
    namespace op = tv::arrayops;
    $keys[i] = 2;
    """, run_in_process=False)


    print(keys.cpu().numpy())

if __name__ == "__main__":
    test_nvrtc_process()