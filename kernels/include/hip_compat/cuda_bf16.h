#pragma once

#undef __HIP_NO_BFLOAT16_CONVERSIONS__
#undef __HIP_NO_BFLOAT16_OPERATORS__
#include <hip/hip_bf16.h>

using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;

// hip_bf16.h spells the pair constructors differently from the CUDA toolkit; this is the
// CUDA helper the kernels reach for.
__host__ __device__ inline __hip_bfloat162 make_bfloat162(const __hip_bfloat16 x,
                                                          const __hip_bfloat16 y) {
  __hip_bfloat162 r;
  r.x = x;
  r.y = y;
  return r;
}
