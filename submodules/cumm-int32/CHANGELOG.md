# Changelog

## [0.8.2] - 2025-05-28
### Added
- add cuda 12.8 support

## [0.8.1] - 2025-05-14
### Fixed
- fix nvrtc problem in cuda >= 12.6

## [0.8.0] - 2025-03-06
### Changed 
- CI environment gcc changed, so we need to increment minor version to avoid conflict.

## [0.7.13] - 2025-02-29
### Fixed 
- fix nvrtc compile problem when use prebuilt. user may need to install nvidia pip package if system cuda isn't installed or version too low.

## [0.7.12] - 2024-12-27
### Fixed 
- fix apple mps compile problem

## [0.7.10] - 2024-12-15
### Fixed
- fix cuda 11.4 compile problem
- better cuda library finding for windows

## [0.7.9] - 2024-12-10
- fix cuda 11.4 compile problem

## [0.7.8] - 2024-12-10
- fix cuda 11.4 compile problem

## [0.7.7] - 2024-12-09
- fix cuda 11 compile problem

## [0.7.6] - 2024-12-09
- fix missing windows cpu prebuilt

## [0.7.5] - 2024-12-08
- fix linux compile problem

## [0.7.4] - 2024-12-08
- fix windows dll load problem and msvc compile problem

## [0.7.3] - 2024-12-08
- fix missing linux cpu prebuilt
- fix some cuda compile problem

## [0.7.2] - 2024-12-06
### Fixed 
- fix metal compile problem
- fix ci

## [0.7.1] - 2024-12-05
### Fixed 
- fix ci

## [0.7.0] - 2024-10-14
### Changed
- drop python 3.8, add python 3.13.
- change cuda prebuilt to 11.4, 11.8, 12.1, 12.4, 12.6.
- change all linux prebuilt to `manylinux_2_28`, drop manylinux2014 support.

## [0.6.4] - 2024-09-01
### Added
- add fast obb-grid overlap for 3dgs.
- add sympy helper functions

## [0.6.3] - 2024-08-18
### Fixed
- fix bug in mac in inline kernels

## [0.6.2] - 2024-08-17
### Fixed
- fix mac os bug

## [0.6.1] - 2024-08-17
### Changed
- debug macos ci

## [0.6.0] - 2024-07-28
### Added 
- add cuda-like kernel in apple silicon, only RTC compile is supported.
### Changed
- BREAKING CHANGE: rename macro `TV_CUDA` to `TV_ENABLE_HARDWARE_ACC`
- BREAKING CHANGE: change managed tensor behavior, now you must specify cpu device when you use managed flag.

## [0.5.3] - 2024-06-12
### Fixed 
- fix compile problem in cuda 12.x

## [0.5.2] - 2024-01-02
### Added
- add `run_in_process` support for inliner to debug some unrecoverable cuda errors such as invalid memory access (700) without restart whole process. this option will copy all tensor data to cpu, copy them to child process (spawn mode), run in child process, and copy back to cpu and main process. this will slow down the performance, but it's very useful for debugging.
- add macro `TV_ASSERT_WITH_PRINT` to perform print in assert.
- change inliner function name with user-provided name for debug.

## [0.5.1] - 2023-12-26
### Fixed
- fix a small bug in `mp_helper.h`

## [0.5.0] - 2023-11-15
### Added
- Add std flag to NVRTCInlinerBuilder
- add `get_nvrtc_kernel_attrs` to NVRTCInlinerBuilder
- add prompt for inliner, use `python -m cumm.inliner.cuda` or `python -m cumm.inliner.cpu` (clang must be installed)
- add rich message print support for nvrtc compile powered by awesome `rich` library. (don't support llvm)
### Changed
- change nvrtc tuple impl to support std::tie
- change supported cuda version, remove cuda 10.2 and 11.6, add cuda 12.1 and 12.2
- remove python 3.7, add python 3.12.
### Fixed
- fix a small bug when using c++17 in nvrtc

## [0.4.11] - 2023-08-09
### Added
- add simple perf tools 

## [0.4.10] - 2023-06-15
### Fixed
- fix a bug in when compile code with arch < sm_75

## [0.4.9] - 2023-04-06
### Added
- add tv::TensorView capture support in nvrtc inliner
- add better error support for cumm nvrtc
### Fixed
- fix a bug in CummNVRTCModule, we need to keep flag order

## [0.4.8] - 2023-03-29
### Fixed
- fix a small bug in tv::Tensor::empty.

## [0.4.7] - 2023-02-02
### Fixed
- fix a small bug in nvrtc tuple.

## [0.4.6] - 2023-01-30
### Fixed
- fix a small bug in nvrtc

## [0.4.5] - 2023-01-20
### Fixed
- fix a compile problem in msvc

## [0.4.4] - 2023-01-19
### Fixed
- fix unsupported arch in cuda 12.0

## [0.4.3] - 2023-01-19
### Fixed
- fix compile problem

## [0.4.2] - 2023-01-19
### Fixed
- fix some compile problem in cpu only

## [0.4.1] - 2023-01-19
### Changed
- change version to rebuild due to pypi server problem

## [0.4.0] - 2022-12-30
### Added
- Add cuda 12.0
- Add int8 inference for sparse conv 
### Fixed
- Fix some problem in cuda 12.0

## [0.3.7] - 2022-11-05
### Added
- Fix bug in ConvProblem introduced in 0.3.6

## [0.3.6] - 2022-11-05
### Added
- Add int64 support for TensorGeneric

## [0.3.5] - 2022-10-18
### Added
- Add flags for H100 and RTX 4090
### Fixed
- fix nvrtc launch problem when smem size is large
- fix nvrtc constant variable parse problem

## [0.3.4] - 2022-9-25
### Changed
- Change gemm/conv main function to splited version

## [0.3.3] - 2022-9-25
### Fixed
- Fix problem in CompileInfo
### Changed
- Change nlohmann json to 3.11.2

## [0.3.2] - 2022-9-25
### Fixed
- Fix build problem in cuda 10.2
- Fix some bug related to nvrtc

## [0.3.1] - 2022-9-25
### Fixed
- Fix cpu build problem

## [0.3.0] - 2022-9-24
### Added 
- Add Ampere support. faster fp16, faster tf32 and greatly faster int8 kernels in Ampere GPUs.
* Add nvrtc support for conv kernel.
### Removed
- drop python 3.6 support.
### Changed
* BREAKING CHANGE: change dtype enum value for some important reason.

## [0.2.8] - 2021-12-8
### Fixed
* Fix missing sm37 in supported arch

## [0.2.7] - 2021-12-8
### Added
* add sm37 for cu102.
* add compile info (cuda arch) for better error information.

## [0.2.6] - 2021-12-3
### Fixed
* Fix a small bug that incorrectly limit arch of simt to sm52.

## [0.2.4] - 2021-11-28
### Added
* add cpu support for CUDAKernelTimer.
* add non-contiguous support for tv::Tensor.
* add tsl hash map, refine cuda hash impl.
### Changed
* raise error instead of exit program when cuda error occurs.
* gemm kernel now use stride, this enable us perform gemm with non-contiguous tensor
### Fixed
* Fix bugs for gemm kernel when use non-contiguous operand.

## [0.2.3] - 2021-11-11
### Fixed
* Fix bugs for implicit gemm

## [0.2.2] - 2021-11-8
### Added
* add support for python 3.6, but cudasim don't support python 3.6.
* add profile tool for all gemm and conv kernels.

## [0.2.1] - 2021-11-8
### Fixed
* Fix some bug of implicit gemm

## [0.2.0] - 2021-11-2
### Addad
* add implicit gemm algorithm for all kind of convolution with kernel volume <= 32. this algorithm is very fast with float16.
* add cuda 11.3 build

### Removed
* remove python 3.6 support