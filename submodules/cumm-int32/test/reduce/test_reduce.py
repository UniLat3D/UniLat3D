from cumm.inliner import NVRTCInlineBuilder, torch_tensor_to_tv
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
import numpy as np
from cumm import dtypes
import pccm 
from cumm import tensorview as tv

def test_block_reduce_sum():
    N = 50000
    block_size = 256
    data = np.random.uniform(-1, 1, size=[N]).astype(np.float32)
    # we still need to use .cuda to make code compatabile with cuda
    data_tv = tv.from_numpy(data).cuda()
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], std="c++17")
    launch_param = tv.LaunchParam((tv.div_up(data.shape[0], block_size), 1, 1), (block_size, 1, 1))
    reduced_val = tv.zeros([1], tv.float32, 0)
    reduced_val_vec = tv.zeros([3], tv.float32, 0)

    inliner.kernel_raw("simple_block_reduce", launch_param, f"""
    namespace op = tv::arrayops;
    int index = tv::parallel::block_idx().x * {block_size} + tv::parallel::thread_idx().x;
    bool valid = index < $N;

    TV_SHARED_MEMORY float smem[{block_size} / 32];
    float val = valid ? $data_tv[index] : 0.0f;
    float val_reduced = tv::parallel::block_reduce_sum_full<{block_size}>(val, smem);
    tv::array<float, 3> val_reduced_vec3{{
        val_reduced, val_reduced, val_reduced,
    }};
    auto ptr_array = op::create_ptr_arange<3>($reduced_val_vec);
    if (tv::parallel::thread_idx().x == 0){{
        tv::parallel::atomicAdd($reduced_val, val_reduced);
        tv::parallel::atomicAdd(op::reinterpret_cast_array_nd<3>($reduced_val_vec), val_reduced_vec3);
        // op::apply(tv::parallel::atomicAdd<float>, reinterpret_cast<tv::array<float*, 3>&>(ptr_array), val_reduced_vec3);
    }}
    """)
    reduced_val_cpu = reduced_val.cpu().numpy()[0]
    print(reduced_val_cpu, data.sum(), reduced_val_vec.cpu().numpy())

if __name__ == "__main__":
    test_block_reduce_sum()