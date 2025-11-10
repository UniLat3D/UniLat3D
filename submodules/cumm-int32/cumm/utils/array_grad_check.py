import numpy as np 
from cumm.gemm.layout_tensorop import rowmajor_inverse
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
from cumm import tensorview as tv
import pccm 
from ccimport import compat 
def check_array_op_grad(inp: np.ndarray, out_shape: list[int], op: str, grad_op: str, delta: float = 1e-4):
    np.random.seed(50051)

    inp_shape = inp.shape[1:]
    num_element = np.prod(inp_shape)
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], std="c++17")
    dtype = np.float64
    tv_dtype = tv.float64
    dtype_str = "double"
    if compat.IsAppleSiliconMacOs:
        dtype = np.float32 
        tv_dtype = tv.float32 
        dtype_str = "float"

    inp = inp.astype(dtype)
    inp_tv = tv.from_numpy(inp).cuda()

    grad_scalar = np.random.uniform(0.5, 1.5, size=out_shape).astype(dtype)
    num_check = inp.shape[0]
    inp_shape_str = ", ".join(map(str, inp_shape))
    out_shape_str = ", ".join(map(str, out_shape))
    my_val_tv = tv.zeros([num_check], tv_dtype, 0)
    ref_val_tv = tv.zeros([num_check], tv_dtype, 0)

    assert list(inp_shape) == list(inp.shape[1:]), f"{inp_shape}, {inp.shape[1:]}"
    inp_shape_np = np.array(inp_shape, np.int32)
    for i in range(num_element):
        inp_delta = np.zeros(inp_shape, dtype)
        index = i
        inp_delta.reshape(-1)[i] = delta 
        slice_indexes = rowmajor_inverse(i, inp_shape_np)
        slice_indexes_str = "".join(map(lambda x: f"[{x}]", slice_indexes))
        inliner.kernel_1d(f"check_grad_op_{op}_{grad_op}_{inp_shape}_{dtype_str}", num_check, 0, f"""
        namespace op = tv::arrayops;
        auto inp_ptr = op::reinterpret_cast_array_nd<{inp_shape_str}>($inp_tv);
        auto inp_arr = inp_ptr[i];
        auto grad_scale = $grad_scalar;
        auto inp_delta_val = $inp_delta;

        tv::array_nd<float, {out_shape_str}> out_arr = inp_arr.op<op::{op}>() * grad_scale;
        auto out_arr_with_delta = (inp_arr + inp_delta_val).op<op::{op}>() * grad_scale;
        auto out_arr_with_delta_sum = op::reshape<-1>(out_arr_with_delta - out_arr).op<op::sum>(); 
        $my_val_tv[i] = op::reshape<-1>(grad_scale.op<op::{grad_op}>(inp_arr))[$index];
        $ref_val_tv[i] = out_arr_with_delta_sum / op::reshape<-1>(inp_delta_val)[$index];
        """)

        my_val = my_val_tv.cpu().numpy()
        ref_val = ref_val_tv.cpu().numpy()
        # currently only double can get high precision result,
        # apple silicon don't support double, so we just print here for now.
        print(my_val)
        print(ref_val)

def check_array_binary_op_grad(inp_list: list[np.ndarray], out_shape: list[int], op: str, grad_ops: list[str], delta: float = 1e-4):
    np.random.seed(50051)
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], std="c++17")
    dtype = np.float64
    tv_dtype = tv.float64
    dtype_str = "double"
    if compat.IsAppleSiliconMacOs:
        dtype = np.float32 
        tv_dtype = tv.float32 
        dtype_str = "float"
    inp_list = [inp.astype(dtype) for inp in inp_list]
    inp_shapes = [inp.shape[1:] for inp in inp_list]
    inp_shape_strs = [", ".join(map(str, inp_shape)) for inp_shape in inp_shapes]
    inp_tvs = [tv.from_numpy(inp).cuda() for inp in inp_list]

    grad_scale_np = np.random.uniform(0.5, 1.5, size=out_shape).astype(dtype)
    num_check = inp_list[0].shape[0]
    # grad_scale_np = np.eye(4, dtype=dtype)
    # grad_scale_np[:] = 1
    out_shape_str = ", ".join(map(str, out_shape))
    my_val_tv = tv.zeros([len(inp_list), num_check], tv_dtype, 0)
    ref_val_tv = tv.zeros([len(inp_list), num_check], tv_dtype, 0)
    code = pccm.code()
    code.raw(f"""
    namespace op = tv::arrayops;

    auto grad_scale = $grad_scale_np;

    """)
    inp_deltas = [np.zeros(inp_shape, dtype) for inp_shape in inp_shapes]
    for cur_inp_idx in range(len(inp_list)):
        code.raw(f"""
        auto inp_ptr_{cur_inp_idx} = op::reinterpret_cast_array_nd<{inp_shape_strs[cur_inp_idx]}>($(inp_tvs[{cur_inp_idx}]));
        auto inp_arr_{cur_inp_idx} = inp_ptr_{cur_inp_idx}[i];
        auto inp_delta_{cur_inp_idx} = $(inp_deltas[{cur_inp_idx}]);

        auto inp_with_delta_{cur_inp_idx} = inp_arr_{cur_inp_idx} + inp_delta_{cur_inp_idx};
        """)
    # op format: inp1.op<op>(inp2, inp3, ...)
    inp_arr_str = ", ".join([f"inp_arr_{i}" for i in range(len(inp_list))])
    inp_arr_with_delta_str = ", ".join([f"inp_with_delta_{i}" for i in range(len(inp_list))])


    inp_arr_start1_str = ", ".join([f"inp_arr_{i}" for i in range(1, len(inp_list))])
    inp_arr_with_delta_start1_str = ", ".join([f"inp_with_delta_{i}" for i in range(1, len(inp_list))])

    code.raw(f"""
    tv::array_nd<{dtype_str}, {out_shape_str}> out_arr = inp_arr_0.op<op::{op}>({inp_arr_start1_str}) * grad_scale;
    tv::array_nd<{dtype_str}, {out_shape_str}> out_arr_with_delta = inp_with_delta_0.op<op::{op}>({inp_arr_with_delta_start1_str}) * grad_scale;
    auto out_arr_with_delta_sum = op::reshape<-1>(out_arr_with_delta - out_arr).op<op::sum>(); 

    """)
    for cur_inp_idx in range(len(inp_list)):
        grad_op = grad_ops[cur_inp_idx]
        code.raw(f"""
        $my_val_tv[{cur_inp_idx} * {num_check} + i] = op::reshape<-1>(grad_scale.op<op::{grad_op}>({inp_arr_str}))[$index];
        $ref_val_tv[{cur_inp_idx} * {num_check} + i] = out_arr_with_delta_sum / op::reshape<-1>(inp_delta_{cur_inp_idx})[$index];
        """)
    for cur_inp_idx in range(len(inp_list)):
        num_element = np.prod(inp_shapes[cur_inp_idx])
        for j in range(num_element):
            inp_deltas[cur_inp_idx].reshape(-1)[j] = delta 
            index = j
            inliner.kernel_1d(f"check_grad_op_{dtype_str}", num_check, 0, code)
            inp_deltas[cur_inp_idx].reshape(-1)[j] = 0 
            my_val = my_val_tv[cur_inp_idx].cpu().numpy()
            ref_val = ref_val_tv[cur_inp_idx].cpu().numpy()
            # currently only double can get high precision result,
            # apple silicon don't support double, so we just print here for now.
            print(f"------ {op}-{grad_ops[cur_inp_idx]}-{j} ------")
            print(my_val)
            print(ref_val)


