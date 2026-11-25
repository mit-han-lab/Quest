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
#ifndef FLASHINFER_MMA_CUH_
#define FLASHINFER_MMA_CUH_

// AMD replacement for the PTX wrappers in flashinfer/mma.cuh. The prefill kernel keeps its
// register fragments in the layout of ldmatrix.m8n8.x4 and mma.sync.m16n8k16, so these
// functions produce and consume exactly that layout:
//
// - ldmatrix and stmatrix are data movement and are reproduced exactly: each lane fetches
//   the row address its fragment comes from with a lane shuffle and reads the shared memory
//   itself.
// - mma gathers, with lane shuffles, the rows of A and the columns of B that feed the
//   accumulator elements the lane owns, and accumulates them in fp32 (fp16 for the
//   f16f16f16 form, rounded once per call as the instruction does).
//
// A warp here is 32 lanes, as in the kernel's indexing. Every shuffle states a width of 32,
// so on a 64-lane wavefront the two warps it holds stay independent and the same code is
// correct on 32-lane and 64-lane GPUs.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace flashinfer {

namespace mma {

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

namespace detail {

__device__ __forceinline__ uint32_t lane_id() { return __lane_id() % 32; }

__device__ __forceinline__ uint32_t shfl(uint32_t x, uint32_t src_lane) {
  return __shfl(x, int(src_lane), 32);
}

template <typename T>
__device__ __forceinline__ T* shfl_ptr(T* p, uint32_t src_lane) {
  return reinterpret_cast<T*>(__shfl(reinterpret_cast<unsigned long long>(p), int(src_lane), 32));
}

// Low and high 16-bit elements of a packed pair, as float.
template <typename T>
__device__ __forceinline__ float lo(uint32_t x) {
  if constexpr (std::is_same<T, half>::value) {
    return __half2float(__ushort_as_half(uint16_t(x & 0xffff)));
  } else {
    return __uint_as_float(x << 16);
  }
}

template <typename T>
__device__ __forceinline__ float hi(uint32_t x) {
  if constexpr (std::is_same<T, half>::value) {
    return __half2float(__ushort_as_half(uint16_t(x >> 16)));
  } else {
    return __uint_as_float(x & 0xffff0000u);
  }
}

// One m16n8k16 product in the mma.sync fragment layout. With g = lane / 4 and t = lane % 4:
// A[0..3] hold A(g, 2t..2t+1), A(g+8, 2t..), A(g, 2t+8..), A(g+8, 2t+8..); B[0..1] hold
// B(2t..2t+1, g), B(2t+8.., g); the lane's accumulators are D(g, 2t), D(g, 2t+1), D(g+8, 2t),
// D(g+8, 2t+1). a_row0 / a_row1 are rows g and g+8 of A as eight k-pairs each.
// Not inlined: the prefill kernel issues 128 of these per tile and is instantiated hundreds
// of times, and inlining them all multiplies the compile time several-fold.
template <typename T>
__device__ __noinline__ void m16n8k16(float* acc, const uint32_t (&a_row0)[8],
                                         const uint32_t (&a_row1)[8], uint32_t b0, uint32_t b1) {
  const uint32_t t = lane_id() % 4;
#pragma unroll
  for (uint32_t n = 0; n < 2; ++n) {
    // Column 2t + n of B is held by the four lanes 4 * (2t + n) + q.
    const uint32_t src = 4 * (2 * t + n);
    float s0 = acc[n], s1 = acc[2 + n];
    // Unrolling this loop as well lets the scheduler keep every shuffle result of the
    // caller's unrolled fragment loops live at once, which spills to scratch memory.
#pragma unroll 1
    for (uint32_t q = 0; q < 4; ++q) {
      const uint32_t blo = shfl(b0, src + q), bhi = shfl(b1, src + q);
      // k = 2q, 2q+1 from blo and k = 2q+8, 2q+9 from bhi.
      s0 = fmaf(lo<T>(a_row0[q]), lo<T>(blo), s0);
      s0 = fmaf(hi<T>(a_row0[q]), hi<T>(blo), s0);
      s0 = fmaf(lo<T>(a_row0[q + 4]), lo<T>(bhi), s0);
      s0 = fmaf(hi<T>(a_row0[q + 4]), hi<T>(bhi), s0);
      s1 = fmaf(lo<T>(a_row1[q]), lo<T>(blo), s1);
      s1 = fmaf(hi<T>(a_row1[q]), hi<T>(blo), s1);
      s1 = fmaf(lo<T>(a_row1[q + 4]), lo<T>(bhi), s1);
      s1 = fmaf(hi<T>(a_row1[q + 4]), hi<T>(bhi), s1);
    }
    acc[n] = s0;
    acc[2 + n] = s1;
  }
}

// Rows g and g+8 of an A fragment, gathered from the four lanes of the lane's group.
__device__ __forceinline__ void gather_a_rows(const uint32_t* A, uint32_t (&row0)[8],
                                              uint32_t (&row1)[8]) {
  const uint32_t base = lane_id() & ~3u;
#pragma unroll
  for (uint32_t q = 0; q < 4; ++q) {
    row0[q] = shfl(A[0], base + q);
    row1[q] = shfl(A[1], base + q);
    row0[q + 4] = shfl(A[2], base + q);
    row1[q + 4] = shfl(A[3], base + q);
  }
}

}  // namespace detail

/*!
 * \brief Equivalent of ldmatrix.sync.aligned.m8n8.x4.shared.b16: lanes 8i..8i+7 give the
 *   row addresses of matrix i, and lane l receives row l / 4, elements 2 (l % 4) and
 *   2 (l % 4) + 1 of each matrix.
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
  const uint32_t lane = detail::lane_id();
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t* row = (const uint32_t*)detail::shfl_ptr(smem_ptr, 8 * i + lane / 4);
    R[i] = row[lane % 4];
  }
}

/*!
 * \brief Equivalent of ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16: as ldmatrix_m8n8x4,
 *   but lane l receives column l / 4, rows 2 (l % 4) and 2 (l % 4) + 1 of each matrix.
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
  const uint32_t lane = detail::lane_id();
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint16_t* row0 = (const uint16_t*)detail::shfl_ptr(smem_ptr, 8 * i + 2 * (lane % 4));
    const uint16_t* row1 =
        (const uint16_t*)detail::shfl_ptr(smem_ptr, 8 * i + 2 * (lane % 4) + 1);
    R[i] = uint32_t(row0[lane / 4]) | (uint32_t(row1[lane / 4]) << 16);
  }
}

/*!
 * \brief Equivalent of stmatrix.sync.aligned.m8n8.x4.shared.b16, the inverse of
 *   ldmatrix_m8n8x4.
 */
template <typename T>
__device__ __forceinline__ void stmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
  const uint32_t lane = detail::lane_id();
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    // Row lane % 8 of matrix i is held by lanes 4 (lane % 8) + 0..3; lane 8i + lane % 8
    // supplied its address.
    uint4 word;
    word.x = detail::shfl(R[i], (lane % 8) * 4);
    word.y = detail::shfl(R[i], (lane % 8) * 4 + 1);
    word.z = detail::shfl(R[i], (lane % 8) * 4 + 2);
    word.w = detail::shfl(R[i], (lane % 8) * 4 + 3);
    if (lane / 8 == i) {
      *(uint4*)smem_ptr = word;
    }
  }
}

/*!
 * \brief Equivalent of two mma.sync.aligned.m16n8k16.row.col.f32 products (B[0..1] and
 *   B[2..3] give the two n8 halves), accumulated in fp32.
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
  if constexpr (mma_mode == MMAMode::kInit) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      C[i] = 0.f;
    }
  }
  uint32_t a_row0[8], a_row1[8];
  detail::gather_a_rows(A, a_row0, a_row1);
  detail::m16n8k16<T>(C, a_row0, a_row1, B[0], B[1]);
  detail::m16n8k16<T>(C + 4, a_row0, a_row1, B[2], B[3]);
}

/*!
 * \brief Row sums of a 16x16 fragment in the A layout, added to d: d[0] gets row g and d[1]
 *   row g + 8, as the mma-with-ones form does.
 */
template <typename DType>
__device__ __forceinline__ void rowsum_f16f16f32(float* d, DType* s) {
  static_assert(sizeof(DType) == 2, "DType must be 16bit floating data type");
  const uint32_t* s_u32 = (const uint32_t*)(s);
  float r0 = detail::lo<DType>(s_u32[0]) + detail::hi<DType>(s_u32[0]) +
             detail::lo<DType>(s_u32[2]) + detail::hi<DType>(s_u32[2]);
  float r1 = detail::lo<DType>(s_u32[1]) + detail::hi<DType>(s_u32[1]) +
             detail::lo<DType>(s_u32[3]) + detail::hi<DType>(s_u32[3]);
  r0 += __shfl_xor(r0, 1, 32);
  r0 += __shfl_xor(r0, 2, 32);
  r1 += __shfl_xor(r1, 1, 32);
  r1 += __shfl_xor(r1, 2, 32);
  d[0] += r0;
  d[1] += r1;
}

/*!
 * \brief Equivalent of two mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 products. The
 *   products are summed in fp32 and the accumulator is rounded to fp16 once per call.
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
  float acc[8];
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t c = (mma_mode == MMAMode::kInit) ? 0u : C[i];
    acc[2 * i] = detail::lo<half>(c);
    acc[2 * i + 1] = detail::hi<half>(c);
  }
  uint32_t a_row0[8], a_row1[8];
  detail::gather_a_rows(A, a_row0, a_row1);
  detail::m16n8k16<half>(acc, a_row0, a_row1, B[0], B[1]);
  detail::m16n8k16<half>(acc + 4, a_row0, a_row1, B[2], B[3]);
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    C[i] = uint32_t(__half_as_ushort(__float2half_rn(acc[2 * i]))) |
           (uint32_t(__half_as_ushort(__float2half_rn(acc[2 * i + 1]))) << 16);
  }
}

}  // namespace mma

}  // namespace flashinfer

#endif  // FLASHINFER_MMA_CUH_
