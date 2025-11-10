import os
import pickle

from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT, TENSORVIEW_INCLUDE_PATH
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, EigenLib
import numpy as np 
from cumm.perftools import perf_context
# @lineprof.lineprof_wrapper_cpp
def test_nvrtc():

    # init driver
    a = tv.zeros([1], tv.float32, 0)
    prog = tv.NVRTCProgram(
        """
    #include <tensorview/core/all.h>
    #include <tensorview/gemm/debug.h>
    #include <tensorview/core/arrayops/all.h>
    #include <cuda/std/cfloat>
    #include <tensorview/hash/linear.cu.h>

    extern \"C\" __global__
    void add(float *x, int64_t n)
    {
        namespace op = tv::arrayops;
        tv::array<int, 2> lfs{};
        tv::array<int, 4> rfs{1, 1, 2, 3};
        // constexpr tv::array<tv::array<float, 3>, 3> a{tv::array<float, 3>{1, 2, 3}, tv::array<float, 3>{4, 5, 6}, tv::array<float, 3>{7, 8, 9}};
        // constexpr auto inv_a = a.op<row>(0);
constexpr tv::array<tv::array<float, 3>, 3> a{tv::array<float, 3>{1, 2, 3}, tv::array<float, 3>{4, 5, 6}, tv::array<float, 3>{7, 8, 3}};
 constexpr tv::array<tv::array<float, 3>, 3> b{tv::array<float, 3>{9, 7, 8}, tv::array<float, 3>{6, 5, 4}, tv::array<float, 3>{3, 2, 1}};
constexpr auto c = a.op<op::mv_colmajor>(b[0]);
constexpr auto c2 = a.op<op::inverse>();

        tv::printf2_once(sizeof(long));
        tv::printf2_once(sizeof(int));
        tv::printf2_once(sizeof(long long));
        auto v3 = tv::arrayops::apply(tv::detail::array_sum<int>, lfs, 2);
        tv::printf2_once(v3[0]);
        auto aa = a.op<op::col>(0);
        tv::printf2_once(aa[0], aa[1], aa[2]);
        aa -= 3;
        auto aa2 = -aa;
        tv::printf2_once(aa2[0], aa2[1], aa2[2]);
        constexpr tv::array<float, 3> a1{1, 2, 3};
        constexpr auto b1 = a1.op<tv::arrayops::max>(2);

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            x[tid] += 1;
        }
        float fval = -4.6;
        tv::printf2_once(int(fval), static_cast<int>(fval), floorf(fval));
        int xx = 5;
        uint32_t yy = reinterpret_cast<uint32_t&>(xx);
    }
    """,
        opts=["--std=c++17", "-I",
              str(TENSORVIEW_INCLUDE_PATH), 
              "-I", os.path.join(os.getenv("CUDA_HOME", "/usr/local/cuda"), "include")])
    print(prog.ptx())

    mod = tv.NVRTCModule(prog)
    mod.load()
    launch = mod.get_launch_param([1, 1, 1], [128, 1, 1], 0, 0)
    mod.run_kernel("add", launch, a, a.dim(0))

    print(a.cpu().numpy())

def test_nvrtc2():
    inliner = NVRTCInlineBuilder([TensorViewNVRTCHashKernel, TensorViewArrayLinalg], root=PACKAGE_ROOT.parent, std="c++17")
    # init driver
    a = tv.zeros([1], tv.float32, 0)
    keys = tv.zeros([5000], tv.int32, 0)
    values = tv.zeros([5000], tv.int32, 0)
    kv = tv.zeros([5000, 2], tv.int32, 0)

    inliner.kernel_1d_capture_tview("wtf", keys.dim(0), 0, f"""
    tv::printf2_once(tv::hash::LinearHashTableSplit<int, int>::empty_key);
    $keys(i) = tv::hash::LinearHashTableSplit<int, int>::empty_key;
    $kv(i, 0) = tv::hash::LinearHashTable<int, int>::empty_key;
    """)
    inliner.kernel_1d_capture_tview("wtf2", 1, 0, f"""
    tv::hash::LinearHashTableSplit<int, int> table($keys.data(), $values.data(), 2500);
    tv::hash::LinearHashTable<int, int> table2(
        reinterpret_cast<tv::hash::LinearHashTable<int, int>::value_type*>($kv.data()), 2500);

    table.insert(5, 1);
    table2.insert(5, 1);
    """)
    
    inliner.kernel_1d("wtf3", 1, 0, f"""
    namespace op = tv::arrayops;
    tv::array<float, 3> a{{2.010012, 0.530250, 0.630409}};
    auto a_inv = 1.0f / a;
    tv::printf2_once("DEBUG", a.op<op::min>());
    int c = 1;
    tv::printf2_once("DEBUG", c);
    auto d = op::concat(a, a);
    tv::printf2_array_once(d);
    std::tuple<float, int> ctx{{1.0f, 2}};
    // std::tuple<float> ctx2{{1.0f}};
    // const auto [ctx2_0] = ctx2;
    // int ctx[2] = {{1, 2}};
    // const auto [ctx_0, ctx_1] = ctx;
    tv::printf2_once(std::get<0>(ctx), std::get<1>(ctx));
    tv::printf2_once(__ffs(1u << 0));

    tv::array_nd<float, 8, 3> corners{{}};
    tv::array_nd<float, 4, 4> imu2enu{{}};
    auto corners2 = corners.op<op::transform_3d>(imu2enu);
                      
    auto tup = std::make_tuple(1.0f, false);
    auto tup2 = tup;
    using debug_tuple_t = std::tuple<float>;      
    debug_tuple_t tup3 = std::make_tuple(1.0f);

    float val0 = 5.99;
    float val01 = 4.99;

    bool val1 = false;
    auto [tup0, tup1] = tup;
    auto& [tup4] = tup3;
    std::swap(val0, val01);
    std::tuple_element<0, debug_tuple_t>::type wtf = std::get<0>(tup3);
    tv::printf2("tuple_size", std::tuple_size<debug_tuple_t>::value, wtf, std::is_same_v<std::tuple_element<0, debug_tuple_t>::type, float>);
    std::tie(val0) = std::move(tup3);
    tv::printf2(val0, val1, std::get<0>(tup));
    // std::is_copy_assignable<float&, decltype(tup0)>::value;
    """, perf_context=perf_context("wtf3"))
    print(inliner.get_nvrtc_kernel_attrs("wtf3"))
    print(a.cpu().numpy())

def test_nvrtc3():
    import numpy as np 
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], root=PACKAGE_ROOT.parent)
    # init driver
    a = tv.zeros([3], tv.float32, 0)
    t = np.zeros((4, 4), np.float32)
    b_out = tv.zeros([3], tv.float32, 0)
    cc = tv.zeros([3], tv.float32, 0)
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), np.float32)
    origin = np.array([0, 1, 0], np.float32)
    normdir = np.array([1, 0, 0], np.float32)
    pp = np.array([1, 1, 1], np.float32)
    inliner.kernel_1d("wtf0", 1, 0, f"""
    $a[0] = 1;
    """)
    inliner.kernel_raw("wtf", inliner.get_1d_param(1), f"""
    namespace op = tv::arrayops;

    auto tr = $t;
    auto tr2 = $R;
    auto q4 = tr2.op<op::rotation_quat>();
    auto p = reinterpret_cast<tv::array<float, 3>*>($a)[0];
    auto length = p.op<op::l2norm>();

    auto b = p.op<op::transform_3d>(tr);
    reinterpret_cast<tv::array<float, 3>*>($b_out)[0] = b;
    tv::printf2_once(q4);
    """)
    # print(inliner.get_nvrtc_module("wtf").get_ptx())

    inliner.kernel_raw("wtf2", inliner.get_1d_param(1), f"""
    namespace op = tv::arrayops;
    tv::array_nd<float, 4, 4> wf;
    auto tr = $t;
    auto p = reinterpret_cast<tv::array<float, 3>*>($a)[0];
    tv::array<float, 3> b;
    b[0] = p[0] * tr[0][0] + p[1] * tr[0][1] + p[2] * tr[0][2] + tr[0][3];
    b[1] = p[0] * tr[1][0] + p[1] * tr[1][1] + p[2] * tr[1][2] + tr[1][3];
    b[2] = p[0] * tr[2][0] + p[1] * tr[2][1] + p[2] * tr[2][2] + tr[2][3];
    reinterpret_cast<tv::array<float, 3>*>($b_out)[0] = b;

    auto lineprojp = $pp.op<op::lineproj>($origin, $normdir);
    """)
    # print(inliner.get_nvrtc_module("wtf2").get_ptx())
    # binary = pickle.dumps(inliner.get_nvrtc_module("wtf"))
    # model2 = pickle.loads(binary)
    # print(len(binary))

def test_nvrtc_problem():

    # init driver
    a = tv.zeros([1], tv.float32, 0)
    prog = tv.NVRTCProgram(
        """
    #include <tensorview/core/all.h>
    #include <tensorview/gemm/debug.h>
    #include <tensorview/core/arrayops/all.h>
    #include <cuda/std/cfloat>
    #include <tensorview/hash/linear.cu.h>
    #include <cuda_fp16.h>
    

    extern \"C\" __global__
    void add(float *x, int64_t n)
    {
        using T = __half;
        using MathOp = tv::arrayops::MathScalarOp<__half>;
        tv::array<__half, 8> hs;
        hs[0] = __half(x[0]);
        __half a = MathOp::exp(hs[0]);
        __half s = __half(-float(hs[0]));
        __half e = MathOp::exp(s);
        // auto o = T(1) / (T(1) + *reinterpret_cast<T*>( &e ));


    }
    """,
        opts=["--std=c++17", "-I",
              str(TENSORVIEW_INCLUDE_PATH), 
              "-I", os.path.join(os.getenv("CUDA_HOME", "/usr/local/cuda"), "include"), "--gpu-architecture=sm_52"])
    print(prog.ptx())

def test_cpu_v0():
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], root=PACKAGE_ROOT.parent)
    # init driver
    a = tv.zeros([1], tv.float32, -1)
    keys = tv.zeros([5000], tv.int32, -1)
    values = tv.zeros([5000], tv.int32, -1)
    kv = tv.zeros([5000, 2], tv.int32, -1)

    inliner.cpu_kernel_1d("wtf3", 1, f"""
    namespace op = tv::arrayops;
    tv::array<float, 3> a{{2.010012, 0.530250, 0.630409}};
    tv::printf2("DEBUG", a.op<op::min>());
    """)

def test_cpu_v1():
    from pccm.builder.inliner import InlineBuilder
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], root=PACKAGE_ROOT.parent)
    # init driver
    a = tv.zeros([1], tv.float32, -1)
    keys = tv.zeros([10], tv.float32, -1)
    values = tv.zeros([5000], tv.int32, -1)
    kv = tv.zeros([5000, 2], tv.int32, -1)
    # inliner = InlineBuilder([TensorViewArrayLinalg], root=PACKAGE_ROOT.parent)
    inliner.cpu_kernel_raw("wtf3", f"""
    namespace op = tv::arrayops;
    for (int i = 0; i < 10; ++i){{
        tv::array<float, 3> a{{2.010012, 0.530250, 0.630409}};
        tv::printf2(std::sin(a[0]) + a[1], op::length<float, 3, 0>()(a));
        // $keys[i] = a[0] + a[1] + a[2] + op::MathScalarOp<float>::sin(a.op<op::min>());
    }}
    """)
    print(keys.numpy())
    breakpoint()
    print(1)

def test_cpu_v2():
    from pccm.builder.inliner import InlineBuilder
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorView], root=PACKAGE_ROOT.parent)
    # init driver
    a = tv.zeros([3], tv.float32, -1)
    t = np.zeros((4, 4), np.float32)
    b_out = tv.zeros([3], tv.float32, -1)
    cc = tv.zeros([3], tv.float32, -1)
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), np.float32)# .reshape(-1)
    R2 = np.random.uniform(-1, 1, size=[2, 3])
    # inliner = InlineBuilder([TensorViewArrayLinalg], root=PACKAGE_ROOT.parent)
    # cpu_kernel_raw
    inliner.cpu_kernel_raw("wtf3", f"""
    tv::ssprint($R, sizeof(decltype($R)), $R2);
    """)
    breakpoint()
    print(1)

if __name__ == "__main__":
    test_nvrtc3()
