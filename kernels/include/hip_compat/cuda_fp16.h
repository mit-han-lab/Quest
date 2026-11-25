#pragma once

// A ROCm PyTorch build puts __HIP_NO_HALF_CONVERSIONS__ on the command line, which strips
// the float conversion operators off __half. The kernels here convert with static_cast the
// way CUDA allows, so the shim asks hip_fp16.h for the full type.
//
// This only works while this header is what first pulls hip/hip_fp16.h into the
// translation unit. Those two macros guard MEMBER functions of __half -- the converting
// constructors and operator= -- so whichever include reaches hip_fp16.h first decides the
// shape of the class for the whole file, and the include guard makes every later attempt,
// including this one, do nothing. If a translation unit ever reaches hip_fp16.h through
// torch's headers before this point, the undefs below become inert and the file stops
// compiling at the first static_cast<float> on a __half; the fix then is to convert
// explicitly with __half2float/__float2half at those sites, as csrc/rms_norm.cu does, not
// to move the undefs. The failure is loud, so this is a maintenance note rather than a
// silent hazard.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_fp16.h>

using nv_half = __half;
using nv_half2 = __half2;
