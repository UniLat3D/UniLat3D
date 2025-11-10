import pickle
import time

import torch

from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT, TENSORVIEW_INCLUDE_PATH
from cumm.inliner import NVRTCInlineBuilder, torch_tensor_to_tv
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
import numpy as np
from cumm import dtypes
import pccm 
import numba 

def test_metal_capture():
    """inliner for metal support small 1d/2d array capture (must smaller than 4x4)
    small capture is passed by func constants to get best performance.
    unlike nvrtc, only 4x4 or smaller matrix are allowed.
    """
    # init cuda
    data = torch.rand([1000, 3], dtype=torch.float32, device="mps")
    datares = torch.empty_like(data)
    arr4x4 = np.random.uniform(-1, 1, size=[4, 4]).astype(np.float32)
    arr4x4_th = torch.from_numpy(arr4x4).to("mps")
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], root=PACKAGE_ROOT.parent, std="c++17")
    wtf: bool = False
    inliner.kernel_1d("kernel_capture", data.shape[0], 0, 
                    f"""
    namespace op = tv::arrayops;
    auto p = op::reinterpret_cast_array_nd<3>($data)[i];
    auto tr = $arr4x4;
    $wtf;
    op::reinterpret_cast_array_nd<3>($datares)[i] = p.op<op::transform_3d>(tr);
    """)
    data_res_ref = data @ (arr4x4_th[:3, :3].T) + arr4x4_th[:3, 3]
    erro = torch.linalg.norm(data_res_ref - datares)
    assert torch.allclose(data_res_ref, datares, atol=1e-5)

def test_metal_basic():
    # init cuda
    a = tv.zeros([100000, 3], tv.float32, 0)
    cnt = tv.zeros([1], tv.int32, 0)
    data = tv.zeros([32], tv.uint64, 0)
    aa = np.eye(4).astype(np.float32)
    inliner = NVRTCInlineBuilder([], root=PACKAGE_ROOT.parent, std="c++17")
    for i in range(5):
        t = time.time()
        inliner.kernel_1d("nvrtc_std", a.shape[0], 0, 
                        f"""
        namespace op = tv::arrayops;
        auto a_ptr = op::reinterpret_cast_array_nd<3>($a);
        auto a_val = a_ptr[i];
        a_ptr[i] = a_val.op<op::transform_3d>($aa);

        auto c_atomic_ptr = reinterpret_cast<device std::atomic_int*>($cnt);
        int old = tv::parallel::atomicAggInc($cnt);
        if (i <= 32){{
            auto wtf_max = metal::numeric_limits<tv::parallel::vote_t>::max();
            $data[i] = tv::parallel::detail::lanemask_lt();
            // $data[i] = op::MathScalarOp<float>::atan2(1.0f, 1.0f);
            // $data[i] = tv::parallel::warp_size() - tv::parallel::lane_index();
        }}
        
        """)
        print(time.time() - t)
    print(a.cpu().numpy())
    print(cnt.cpu().numpy())
    print(data.cpu().numpy())

    # print(inliner.get_nvrtc_kernel_attrs("nvrtc_std"))

    # mod = CummNVRTCModule([TensorViewNVRTCDev()], verbose=True)

def test_metal_torch_cumm():
    th_ten = torch.rand(64, 4321).float().to("mps")
    th_ten2 = th_ten[1:]
    ref = th_ten2.cpu().numpy()
    tv_ten = torch_tensor_to_tv(th_ten2)
    my = tv_ten.cpu().numpy()

    print(np.linalg.norm(ref - my))


def test_metal_hash():
    keys_np = np.random.randint(0, 10, size=(2500)).astype(np.int32)
    # we still need to use .cuda to make code compatabile with cuda
    keys = tv.from_numpy(keys_np).cuda()
    wtf = tv.zeros([100], tv.float16, 0)
    keys_debug = tv.zeros([keys.shape[0]], tv.uint32, 0)
    hash_length = int(keys.shape[0] * 2)
    hashkeys = tv.empty([hash_length], tv.int32, 0)
    hashvalues = tv.empty([hash_length], tv.int32, 0)
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], root=PACKAGE_ROOT.parent, std="c++17")
    cnt = tv.zeros([1], tv.int32, 0)
    inliner.kernel_1d("clear_hash", hashkeys.dim(0), 0, f"""
    using table_t = tv::hash::LinearHashTableSplit<int32_t, int>;
    $hashkeys[i] = table_t::empty_key;
    """)
    # print(keys.cpu().numpy())
    inliner.kernel_1d("insert_table", keys.dim(0), 0, f"""
    using table_t = tv::hash::LinearHashTableSplit<int, int>;
    table_t table($hashkeys, $hashvalues, $hash_length);
    auto key = $keys[i];
    table.insert(key, 1);
    // $keys_debug[i] = $hash_length;
    """)

    inliner.kernel_1d("collect_unique_res", hashkeys.dim(0), 0, f"""
    using table_t = tv::hash::LinearHashTableSplit<int, int>;
    table_t table($hashkeys, $hashvalues, $hash_length);
    if ($hashkeys[i] != table_t::empty_key){{
        tv::parallel::atomicAdd($cnt, 1);
    }}
    """)

    print(cnt.cpu().numpy())

class _TestMetalHash(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)

    @pccm.static_function(attrs=["TV_DEVICE_INLINE"], header_only=True)
    def custom_value_insert(self):
        code = pccm.code()
        code.arg("ptr", "TV_METAL_DEVICE float*")
        code.arg("value", "float")
        code.raw(f"""
        tv::parallel::atomicAdd(ptr, value);
        """) 
        return code 

@numba.njit 
def _test_hash_table_insert(keys: np.ndarray, values: np.ndarray, acc_values: np.ndarray):
    res = np.ones_like(values)
    hash_dict = {}
    for i in range(keys.shape[0]):
        k = keys[i]
        if k not in hash_dict:
            hash_dict[k] = values[i]
        else:
            hash_dict[k] += values[i]
    for i in range(keys.shape[0]):
        k = keys[i]
        res[i] = hash_dict[k]

    return res
    

def _template_test_metal_hash(is_u64: bool, num: int = 25000):
    """parallel code do same thing in first for loop of `_test_hash_table_insert`
    """
    if is_u64:
        dtype = tv.uint64 
        cpp_dtype = "uint64_t"
    else:
        dtype = tv.uint32 
        cpp_dtype = "uint32_t"
    np_dtype = dtypes.get_npdtype_from_tvdtype(dtype)
    keys_np = np.random.randint(0, 2000, size=(num)).astype(np_dtype)
    value_fp = np.random.uniform(-1, 1, size=keys_np.size).astype(np.float32)
    # we still need to use .cuda to make code compatabile with cuda
    keys_dev = tv.zeros([keys_np.size], dtype, -1, managed=True)
    keys_dev.numpy_view()[:] = keys_np
    keys = tv.from_numpy(keys_np).cuda()
    # print(keys.shape)
    # breakpoint()
    values = tv.from_numpy(value_fp).cuda()
    acc_values = tv.from_numpy(value_fp).cuda()

    hash_length = int(keys.shape[0] * 1.3)
    hashkeys = tv.empty([hash_length], dtype, 0)
    hashvalues = tv.zeros([hash_length], tv.float32, 0)
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], root=PACKAGE_ROOT.parent, std="c++17")
    cnt = tv.zeros([1], tv.int32, 0)
    with inliner.enter_inliner_scope():
        inliner.kernel_1d("clear_hash", hashkeys.dim(0), 0, f"""
        using table_t = tv::hash::LinearHashTableSplit<{cpp_dtype}, float>;
        $hashkeys[i] = table_t::empty_key;
        """)
        code = pccm.code(f"""
        using table_t = tv::hash::LinearHashTableSplit<{cpp_dtype}, float>;
        table_t table($hashkeys, $hashvalues, $hash_length);
        auto key = $keys[i];
        auto value = $values[i];
        table.insert_custom_value(key, _TestMetalHash::custom_value_insert, value);
        """)
        code.add_dependency(_TestMetalHash)
        inliner.kernel_1d("insert_table", keys.dim(0), 0, code)
        inliner.kernel_1d("query_key", keys.dim(0), 0, f"""
        using table_t = tv::hash::LinearHashTableSplit<{cpp_dtype}, float>;
        table_t table($hashkeys, $hashvalues, $hash_length);
        auto key = $keys[i];
        int offset = table.lookup_offset(key);
        if (offset != -1){{
            $acc_values[i] = table.value_ptr()[offset];
        }}
        """)
    res = _test_hash_table_insert(keys_np, value_fp, acc_values.cpu().numpy())
    print(np.linalg.norm(res - acc_values.cpu().numpy()))
    assert np.allclose(res, acc_values.cpu().numpy(), atol=1e-5)

def test_metal_hash():
    _template_test_metal_hash(False)

    _template_test_metal_hash(True)

if __name__ == "__main__":
    torch.manual_seed(42)
    test_metal_capture()
    # test_metal_torch_cumm()
    test_metal_hash()