# cumm
CUda Matrix Multiply library.

[![Build Status](https://github.com/FindDefinition/cumm/workflows/build/badge.svg)](https://github.com/FindDefinition/cumm/actions?query=workflow%3Abuild)

```cumm``` is developed during learning of [CUTLASS](https://github.com/NVIDIA/cutlass), which use too much c++ template and make code unmaintainable. So I develop [pccm](https://github.com/FindDefinition/PCCM), use python as meta programming language, to replace c++ template meta programming. 
Now ```pccm``` become a foundational framework of ```cumm``` and my other c++ project such as [spconv](https://github.com/traveller59/spconv). 
```cumm``` also contains a python asyncio-based gemm simulator that **share same meta program** with CUDA code, enable gemm visualization and easy debug experience.

## Important Notice

**INT32 Data Range Limitation**: The original cumm matrix multiplication operation raises an error when encountering int32 data ranges. When the mesh is very large, this error is inevitable. This codebase includes a fix for this issue using a coarse method, allowing it to work with data ranges exceeding int32. **Please note that this is a development version of the code - do not use for training purposes**. The fix has been verified to work correctly in testing scenarios.

If you find this fix helpful for your research, please cite:

```latex
@article{wu2025unilat3d,
  title={UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation},
  author={Wu, Guanjun and Fang, Jiemin and Yang, Chen and Li, Sikuang and Yi, Taoran and Lu, Jia and Zhou, Zanwei and Cen, Jiazhong and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Wang, Xinggang and Tian, Qi},
  journal={arXiv preprint arXiv:2509.25079},
  year={2025}
}
```

## BREAKING CHANGES

* 0.3.1: tv::DType enum value changed, this will affect all binary code of tv::Tensor user. you must recompile all code if upgrade to cumm >= 0.3.1.

## News

* Ampere feature support (by [EvernightAurora](https://github.com/EvernightAurora))

## Install

### Prebuilt

We offer python 3.9-3.13 and cuda 11.4/11.8/12.1/12.4/12.6 prebuilt binaries for linux (`manylinux_2_28`).

We offer python 3.9-3.13 and cuda 11.4/11.8/12.1/12.4/12.6 prebuilt binaries for windows 10/11.

We offer python 3.9-3.13 prebuilt binaries for Mac OS X >= 14.0 (Apple Silicon Only).

```pip install cumm``` for CPU-only

```pip install cumm-cu114``` for CUDA 11.4

```pip install cumm-cu126``` for CUDA 12.6

### Build from source for development (JIT, recommend for develop)

**WARNING** Use code in [tags](https://github.com/FindDefinition/cumm/releases)!!! code in main branch may contain bugs.

The c++ code will be built automatically when you change c++ code in project.

#### Linux

0. uninstall cumm installed by pip. you must ensure no "cumm" exists in ```pip list | grep cumm```
1. install build-essential, install CUDA
2. ```git clone https://github.com/FindDefinition/cumm```, ```cd ./cumm```, ```git checkout tags/<tag_name>```, ```pip install -e .```
3. in python, ```import cumm``` and wait for build finish.

#### Windows
0. uninstall spconv and cumm installed by pip. you must ensure no "cumm" exists in ```pip list | grep cumm```
1. install visual studio 2019 or newer. make sure C++ development component is installed. install CUDA
2. set [powershell script execution policy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1)
3. start a new powershell, run ```tools/msvc_setup.ps1```
4. ```git clone https://github.com/FindDefinition/cumm```, ```cd ./cumm```, ```git checkout tags/<tag_name>```, ```pip install -e .```
5. in python, ```import cumm``` and wait for build finish.

### Build wheel from source 

**WARNING** Use code in [tags](https://github.com/FindDefinition/cumm/releases)!!! code in main branch may contain bugs.

**WARNING**: If ```CUMM_CUDA_VERSION``` is set with a CUDA version, following steps will create a wheel named "cumm-cuxxx", not "cumm", this means you must use ```cumm-cuxxx``` in dependency of your project which depend on cumm, not ```cumm```. If ```CUMM_CUDA_VERSION``` isn't set, ```cumm``` will always built with CUDA, so the CUDA must exists in your system. The wheel name will be ```cumm``` even if it is built with cuda.

#### Linux

It's recommend to build Linux packages in [official build docker](https://github.com/FindDefinition/cumm/blob/main/.github/workflows/build.yaml). Build with CUDA support don't need a real GPU.

##### Build in Official Docker

1. select a cuda version. available: CUDA 11.1, 11.3, 11.4, 11.5, 12.0
2. (Example for CUDA 11.4) ```git clone https://github.com/FindDefinition/cumm```, ```cd ./cumm```, ```docker run --rm -e PLAT=manylinux2014_x86_64 -e CUMM_CUDA_VERSION=114 -v `pwd`:/io scrin/manylinux2014-cuda:cu114-devel-1.0.0 bash -c "source /etc/bashrc && /io/tools/build-wheels.sh"```

##### Build in your environment

1. install build-essential, install CUDA
2. set env for installed cuda version. for example, ```export CUMM_CUDA_VERSION="11.4"```. If you want to build CPU-only, run ```export CUMM_CUDA_VERSION=""```. If ```CUMM_CUDA_VERSION``` isn't set, you need to ensure cuda libraries are inside OS search path, and the built wheel name will be ```cumm```, otherwise ```cumm-cuxxx```
3. run ```export CUMM_DISABLE_JIT="1"```
4. run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

#### Windows 10/11

1. install visual studio 2019 or newer. make sure C++ development package is installed. install CUDA
2. set [powershell script execution policy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1)
3. start a new powershell, run ```tools/msvc_setup.ps1```
4. set env for installed cuda version. for example, ```$Env:CUMM_CUDA_VERSION = "11.4"```. If you want to build CPU-only, run ```$Env:CUMM_CUDA_VERSION = ""```. . If ```CUMM_CUDA_VERSION``` isn't set, you need to ensure cuda libraries are inside OS search path, and the built wheel name will be ```cumm```, otherwise ```cumm-cuxxx```
4. run ```$Env:CUMM_DISABLE_JIT = "1"```
5. run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

## Contributers

* [EvernightAurora](https://github.com/EvernightAurora): add ampere feature.

## Note
The work is done when the author is an employee at [Tusimple](https://www.tusimple.com/).

## LICENSE

Apache 2.0
