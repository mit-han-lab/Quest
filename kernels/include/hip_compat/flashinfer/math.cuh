/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

// AMD replacement for the PTX wrappers in flashinfer/math.cuh. Same signatures and same
// numerical intent; the ex2.approx / rcp.approx / shfl.sync.bfly instructions have no
// spelling in the HIP kernel language, so they are expressed with the corresponding HIP
// device functions.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

__forceinline__ __device__ float ptx_exp2(float x) { return exp2f(x); }

__forceinline__ __device__ float ptx_log2(float x) { return log2f(x); }

__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  return half2{__float2half(exp2f(__half2float(x.x))), __float2half(exp2f(__half2float(x.y)))};
}

__forceinline__ __device__ half ptx_exp2(half x) {
  return __float2half(exp2f(__half2float(x)));
}

__forceinline__ __device__ float ptx_rcp(float x) { return __frcp_rn(x); }

// The PTX form clamps to 0x1f, i.e. the butterfly stays inside a 32-lane segment. Passing
// an explicit width of 32 reproduces that on both a 32-wide and a 64-wide wavefront, so
// the surrounding reduction keeps the lane grouping its indexing assumes.
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor(x, lane_mask, 32);
}

__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return uint32_as_half2(__shfl_xor(half2_as_uint32(x), lane_mask, 32));
}

}  // namespace math

// HIP provides __hmax2 for bfloat16 pairs only. Same semantics as CUDA's: a NaN in one input
// yields the other input.
__forceinline__ __device__ half2 __hmax2(const half2 a, const half2 b) {
  return half2{__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b))};
}

}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
