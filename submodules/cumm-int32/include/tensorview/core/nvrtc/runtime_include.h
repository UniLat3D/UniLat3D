#pragma once 
#include <tensorview/core/defs.h>

#if defined(TV_HARDWARE_ACC_CUDA)
#ifndef __CUDACC_RTC__
#include <cuda_runtime_api.h>
#endif
#endif