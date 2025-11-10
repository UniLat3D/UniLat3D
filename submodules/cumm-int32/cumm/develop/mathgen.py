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


import dataclasses
from typing import List

@dataclasses.dataclass
class MathItem:
    name: str 
    f16name: str = ""
    f16vecname: str = ""


def main():
    items: List[MathItem] = [
        MathItem("sqrt", "hsqrt", "h2sqrt"),
        MathItem("rsqrt", "hrsqrt", "h2rsqrt"),
        MathItem("ceil", "hceil", "h2ceil"),
        MathItem("cos", "hcos", "h2cos"),
        MathItem("exp", "hexp", "h2exp"),
        MathItem("exp10", "hexp10", "h2exp10"),
        MathItem("exp2", "hexp2", "h2exp2"),
        MathItem("floor", "hfloor", "h2floor"),
        MathItem("log", "hlog", "h2log"),
        MathItem("log10", "hlog10", "h2log10"),
        MathItem("log2", "hlog2", "h2log2"),
        MathItem("rint", "hrint", "h2rint"),
        MathItem("sin", "hsin", "h2sin"),
        MathItem("trunc", "htrunc", "h2trunc"),

        MathItem("fabs", "__habs", "__habs2"),


        MathItem("tan"),
        MathItem("asin"),
        MathItem("acos"),
        MathItem("atan"),
        MathItem("round"),
        MathItem("sinh"),
        MathItem("cosh"),
        MathItem("tanh"),
        MathItem("asinh"),
        MathItem("acosh"),
        MathItem("atanh"),

    ]
    code_cpus = []
    code_gpu_fallbacks = []
    code_cuda_fps = []
    code_cuda_hfs = []
    code_cuda_bfs = []
    code_cuda_hf2s = []
    code_cuda_bf2s = []
    no_cpu_vers = set(["rsqrt", "exp10"])
    for item in items:
        if item.name not in no_cpu_vers:
            code_cpus.append(f"""
    static T {item.name}(T x){{
        return std::{item.name}(x);
    }}
            """)
            code_gpu_fallbacks.append(f"""
    TV_HOST_DEVICE_INLINE static T {item.name}(T x){{
        return T({item.name}f(float(x)));
    }}
            """)
        code_cuda_fps.append(f"""
    TV_HOST_DEVICE_INLINE static float {item.name}(float x){{
        return {item.name}f(x);
    }}
        """)
        if item.f16name:
            code_cuda_hfs.append(f"""
    TV_DEVICE_INLINE static __half {item.name}(__half x){{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
        return {item.f16name}(x);
#else
        return __half({item.name}f(float(x)));
#endif
    }}
            """)
            code_cuda_bfs.append(f"""
    TV_DEVICE_INLINE static __nv_bfloat16 {item.name}(__nv_bfloat16 x){{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        return {item.f16name}(x);
#else
        return __nv_bfloat16({item.name}f(float(x)));
#endif
    }}
            """)
        if item.f16vecname:
            code_cuda_hf2s.append(f"""
    TV_DEVICE_INLINE static __half2 v2{item.name}(__half2 x){{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
        return {item.f16vecname}(x);
#else
        auto x0 = __low2float(x);
        auto x1 = __high2float(x);
        x0 = {item.name}f(x0);
        x1 = {item.name}f(x1);
        return __floats2half2_rn(x0, x1);
#endif

    }}
            """)
            code_cuda_bf2s.append(f"""
    TV_DEVICE_INLINE static __nv_bfloat162 v2{item.name}(__nv_bfloat162 x){{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        return {item.f16vecname}(x);
#else
        auto x0 = __low2float(x);
        auto x1 = __high2float(x);
        x0 = {item.name}f(x0);
        x1 = {item.name}f(x1);
        return __floats2bfloat162_rn(x0, x1);
#endif
    }}
            """)

    code_cpu = "\n".join(code_cpus)
    code_gpu_fallback = "\n".join(code_gpu_fallbacks)

    code_cuda_fp = "\n".join(code_cuda_fps)
    code_cuda_hf = "\n".join(code_cuda_hfs)
    code_cuda_hf2 = "\n".join(code_cuda_hf2s)
    code_cuda_bf = "\n".join(code_cuda_bfs)
    code_cuda_bf2 = "\n".join(code_cuda_bf2s)

    code = f"""
template <typename T>
struct MathScalarOp{{
#ifndef __CUDACC__
static T copysign(T x, T y){{
    return std::copysign(x, y);
}}

static T atan2(T y, T x){{
    return std::atan2(y, x);
}}

static T scalbn(T x, int n){{
    return std::scalbn(x, n);
}}

static T pow(T x, T n){{
    return std::pow(x, n);
}}

static T fmod(T x, T n){{
    return std::fmod(x, n);
}}

static T neg(T x) {{ 
    return -x; 
}}

{code_cpu}

static T rsqrt(T x) {{ 
    return T(1) / sqrt(x); 
}}
#else 

TV_HOST_DEVICE_INLINE static T copysign(T x, T y){{
    return T(copysignf(float(x), float(y)));
}}

TV_HOST_DEVICE_INLINE static T atan2(T y, T x){{
    return T(atan2f(float(y), float(x)));
}}

TV_HOST_DEVICE_INLINE static T scalbn(T x, int n){{
    return T(scalbnf(float(x), n));
}}

TV_HOST_DEVICE_INLINE static T pow(T x, T n){{
    return T(pow(float(x), float(n)));
}}

TV_HOST_DEVICE_INLINE static T fmod(T x, T n){{
    return T(fmodf(float(x), float(n)));
}}

{code_gpu_fallback}

TV_HOST_DEVICE_INLINE static T neg(T x) {{ 
    return -x; 
}}

TV_HOST_DEVICE_INLINE static T rsqrt(T x) {{ 
    return T(1) / sqrt(x); 
}}

#endif
}};

#ifdef TV_CUDA_CC
template <>
struct MathScalarOp<float>{{

TV_HOST_DEVICE_INLINE static float copysign(float x, float y){{
    return copysignf(x, y);
}}

TV_HOST_DEVICE_INLINE static float atan2(float y, float x){{
    return atan2f(y, x);
}}

TV_HOST_DEVICE_INLINE static float scalbn(float x, int n){{
    return scalbnf(x, n);
}}

TV_HOST_DEVICE_INLINE static float pow(float x, float n){{
    return powf(x, n);
}}

TV_HOST_DEVICE_INLINE static float fmod(float x, float n){{
    return fmodf(x, n);
}}
TV_HOST_DEVICE_INLINE static float neg(float x) {{ 
    return -x; 
}}

{code_cuda_fp}
}};
#ifdef __CUDACC__
template <>
struct MathScalarOp<__half>{{
{code_cuda_hf}
{code_cuda_hf2}
  TV_HOST_DEVICE_INLINE static __half neg(__half x) {{ 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hneg(x);
#else
    return __half(-(float(x)));
#endif

  }}

}};
#endif
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11))
template <>
struct MathScalarOp<__nv_bfloat16>{{
{code_cuda_bf}
{code_cuda_bf2}
  TV_HOST_DEVICE_INLINE static __nv_bfloat16 neg(__nv_bfloat16 x) {{  
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __hneg(x);
#else
    return __nv_bfloat16(-(float(x)));
#endif
  }}
}};
#endif

#endif


    """
    print(code)

if __name__ == "__main__":

    main()
