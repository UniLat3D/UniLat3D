import pickle

from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT, TENSORVIEW_INCLUDE_PATH
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, EigenLib
import numpy as np 

def test_nvrtc_std():
    # init cuda
    a = tv.zeros([3], tv.float32, 0)
    inliner = NVRTCInlineBuilder([TensorViewNVRTCHashKernel, TensorViewArrayLinalg], root=PACKAGE_ROOT.parent, std="c++17")
    
    inliner.kernel_1d("nvrtc_std", 1, 0, f"""

    using wtftype = tv::detail::determine_broadcast_array_type<float, tv::array_nd<float, 1, 4>, float>::type;
    using wtftype2 = tv::detail::determine_broadcast_array_type<float, tv::array_nd<float, 1, 4>, tv::array_nd<float, 3, 1>>::type;

    wtftype arr_test1 = tv::array_nd<float, 1, 4>{{}};
    wtftype2 arr_test2 = tv::array_nd<float, 3, 4>{{}};
    auto arr_test3 = tv::array_nd<float, 4>{{}};
    auto arr_test4 = tv::array_nd<float, 1>{{}};

    tv::printf2(tv::detail::get_tv_array_rank<wtftype2>::value);
    auto wtf3 = arr_test1 + arr_test2;
    auto tup = std::make_tuple(1.0f, false);
    auto tup2 = tup;
    using debug_tuple_t = std::tuple<float>;      
    debug_tuple_t tup3 = std::make_tuple(1.0f);
    auto ctor = std::tuple<tv::array<float, 4>, uint32_t>{{}};
    float val0 = 5.99;
    float val01 = 4.99;

    bool val1 = false;
    auto [tup0, tup1] = tup;

    auto& [tup4] = tup3;
    // auto [tup5] = tup3;

    std::swap(val0, val01);
    std::tuple_element<0, debug_tuple_t>::type wtf = std::get<0>(tup3);
    tv::printf2("tuple_size", std::tuple_size<debug_tuple_t>::value, wtf, std::is_same_v<std::tuple_element<0, debug_tuple_t>::type, float>);
    std::tie(val0) = std::move(tup3);
    tv::printf2(val0, val1, std::get<0>(tup));
    // std::is_copy_assignable<float&, decltype(tup0)>::value;
    """)
    print(inliner.get_nvrtc_kernel_attrs("nvrtc_std"))


if __name__ == "__main__":
    test_nvrtc_std()
