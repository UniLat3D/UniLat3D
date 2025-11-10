# Copyright 2024 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pccm.builder.inliner import InlineBuilder
from cumm.common import EigenLib, TensorView, TensorViewArrayLinalg, TensorViewParallel
import numpy as np 
import pccm 
def rot_to_quat(axis, angle):
    res = np.zeros((4,), np.float32)
    res[:3] = axis*np.sin(angle/2)
    res[3] = np.cos(angle/2)
    return res


def main():
    vec = np.array([0, 0, 1], np.float32)
    vec2 = np.random.uniform(-1, 1, size=[3]).astype(np.float32)
    vec2 = vec2 / np.linalg.norm(vec2)
    print(rot_to_quat(vec, 0.5))
    inliner = InlineBuilder([EigenLib, TensorView, TensorViewArrayLinalg])
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), np.float32)

    inliner.inline("wtf", f"""
    namespace op = tv::arrayops;
    const Eigen::Vector3f Z = {{{vec2[0]}, {vec2[1]}, {vec2[2]}}};
    auto l = Eigen::AngleAxisf(0.5, Z).toRotationMatrix();
    tv::ssprint(l);

    auto Zv = op::create_array({vec2[0]}f, {vec2[1]}f, {vec2[2]}f);
    auto r4 = Zv.op<op::rotation_quat>(0.5);

    auto my = r4.op<op::rotation_matrix_3x3>();
    tv::ssprint(my);
    tv::ssprint("------");
    Eigen::AngleAxisf result;
    result.fromRotationMatrix(l);
    tv::ssprint(result.angle(), "\\n", result.axis());

    auto q4 = my.op<op::rotation_quat_matrix>();
    auto axis4 = q4.op<op::qaxis>();
    tv::ssprint(axis4, q4.op<op::qangle>());

    """)

class CustomHeader(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/debug.h")

def main2():
    inliner = InlineBuilder([EigenLib, TensorView, TensorViewArrayLinalg, TensorViewParallel])
    
    inliner.inline("wtf", f"""
    namespace op = tv::arrayops;
    Eigen::Matrix4f a, b;
    for (int j = 0; j < 16; ++j){{
        a(j / 4, j % 4) = j + 1;
        b(j / 4, j % 4) = j + 1;
    }}
    Eigen::Matrix4f c = a * b;

    std::cout << c << std::endl;
    tv::Tensor rtx = tv::zeros({{1}}, tv::float32, 0);
    auto rtx_ptr = rtx.data_ptr<float>();
    tv::kernel_1d_map(0, 1, [=]TV_GPU_LAMBDA(size_t i){{
        Eigen::Matrix4f a, b;
        for (int j = 0; j < 16; ++j){{
            a(j / 4, j % 4) = j + 1;
            b(j / 4, j % 4) = j + 1;
        }}

        Eigen::Matrix4f c = a * b;
        for (int j = 0; j < 16; ++j){{
            tv::printf2(c(j / 4, j % 4));
        }}
        rtx_ptr[0] = 1;
        tv::printf2("HJELLOW");
    }});
    """, impl_file_suffix=".cc")


if __name__ == "__main__":
    main2()