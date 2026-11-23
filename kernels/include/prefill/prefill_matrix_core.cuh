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
#ifndef FLASHINFER_PREFILL_MATRIX_CORE_CUH_
#define FLASHINFER_PREFILL_MATRIX_CORE_CUH_

// BatchPrefillWithPagedKVCacheKernel's tile work on AMD matrix instructions: MFMA on CDNA
// (64-wide wavefronts) and WMMA on RDNA3 and RDNA4 (32-wide). The kernel keeps its launch
// shape, tiling, page addressing, masking rule and online softmax; what differs from the
// mma.sync path is the fragment layout and where the operands live.
//
// Each 16x16x16 instruction takes A and B with lane l holding row (A) or column (B) l % 16
// and gives D with lane l holding column l % 16. The kernel computes S^T = K Q^T, so each
// lane holds one query column of S; that is the B layout O^T = V^T P^T needs, and the
// probabilities feed the second product from registers. Each lane also owns whole query
// rows of the softmax state, which is reduced across the lanes of a column (l % 16) only.
// A wavefront works on the query rows of the warps it holds: on a 64-wide wavefront two of
// the kernel's 32-lane warps, whose rows are contiguous.
//
// Operands: Q stays in registers when it fits (q_in_registers), so shared memory holds only
// K and V; V is stored transposed so its operand is a contiguous read; and each K and V tile
// is loaded into registers one phase ahead of its shared memory store, since AMD has no
// asynchronous global-to-shared copy.
//
// QUEST_PREFILL_MMA selects the instruction set in a device pass: 1 for MFMA, 11 for RDNA3
// WMMA, 12 for RDNA4 WMMA. It is left undefined for the host pass and for any other target,
// which keep the mma.sync emulation in hip_compat/flashinfer/mma.cuh.

#if !defined(QUEST_PREFILL_MMA) && defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
    defined(__gfx950__) || defined(__gfx9_4_generic__)
#define QUEST_PREFILL_MMA 1
#elif defined(__GFX11__)
#define QUEST_PREFILL_MMA 11
#elif defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
// Not __GFX12__: gfx1250 has a different set of WMMA instructions.
#define QUEST_PREFILL_MMA 12
#endif
#endif

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace matrix_core {

// What the device code for the current GPU uses, read by the host to size shared memory:
// wave_size and op_elems (16-bit operand elements per lane) of the matrix instruction, or
// zeros where the kernel keeps the mma.sync emulation.
struct mma_shape_t {
  uint32_t wave_size;
  uint32_t op_elems;
};

static __device__ __attribute__((used)) const mma_shape_t prefill_mma_shape =
#if QUEST_PREFILL_MMA == 1
    {64, 4};
#elif QUEST_PREFILL_MMA == 11
    {32, 16};
#elif QUEST_PREFILL_MMA == 12
    {32, 8};
#else
    {0, 0};
#endif

// Whether a wavefront keeps its query tile in registers (64 of them at most) for the whole
// kernel. The shared memory then holds only the K and V tiles, so two threadblocks fit in a
// 64 KB compute unit.
__host__ __device__ constexpr bool q_in_registers(uint32_t num_frags_x, uint32_t num_frags_y,
                                                  mma_shape_t shape) {
  return shape.wave_size != 0 &&
         num_frags_x * (shape.wave_size / 32) * num_frags_y * shape.op_elems / 2 <= 64;
}

inline cudaError_t get_prefill_mma_shape(mma_shape_t* shape) {
  constexpr int kMaxDevices = 64;
  static mma_shape_t cache[kMaxDevices];
  static bool known[kMaxDevices] = {};
  int dev = 0;
  cudaError_t status = cudaGetDevice(&dev);
  if (status != cudaSuccess) return status;
  if (dev < kMaxDevices && known[dev]) {
    *shape = cache[dev];
    return cudaSuccess;
  }
  status = hipMemcpyFromSymbol(shape, HIP_SYMBOL(prefill_mma_shape), sizeof(mma_shape_t));
  if (status != cudaSuccess) return status;
  if (dev < kMaxDevices) {
    cache[dev] = *shape;
    known[dev] = true;
  }
  return cudaSuccess;
}

}  // namespace matrix_core
}  // namespace flashinfer

#if defined(QUEST_PREFILL_MMA)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

#include "flashinfer/permuted_smem.cuh"

namespace flashinfer {
namespace matrix_core {

#if QUEST_PREFILL_MMA == 1
constexpr uint32_t kWaveSize = 64;
// 16-bit operand elements per lane.
constexpr uint32_t kOpElems = 4;
#elif QUEST_PREFILL_MMA == 11
constexpr uint32_t kWaveSize = 32;
constexpr uint32_t kOpElems = 16;
#else
constexpr uint32_t kWaveSize = 32;
constexpr uint32_t kOpElems = 8;
#endif
// fp32 accumulator elements per lane of a 16x16 tile.
constexpr uint32_t kAccElems = 256 / kWaveSize;
constexpr uint32_t kWarpsPerWave = kWaveSize / 32;
constexpr mma_shape_t kShape = {kWaveSize, kOpElems};

__device__ __forceinline__ uint32_t lane_id() {
  return (threadIdx.y * 32 + threadIdx.x) % kWaveSize;
}

// Row of D (within the 16x16 tile) held in accumulator element i.
__device__ __forceinline__ uint32_t acc_row(uint32_t i) {
#if QUEST_PREFILL_MMA == 1
  return 4 * (lane_id() / 16) + i;
#elif QUEST_PREFILL_MMA == 11
  return 2 * i + lane_id() / 16;
#else
  return 8 * (lane_id() / 16) + i;
#endif
}

// k index (within 16) of operand element 0; the lane's operand elements are consecutive.
__device__ __forceinline__ uint32_t op_k0() {
#if QUEST_PREFILL_MMA == 1
  return 4 * (lane_id() / 16);
#elif QUEST_PREFILL_MMA == 11
  return 0;
#else
  return 8 * (lane_id() / 16);
#endif
}

// Sum over the lanes holding the same column (l % 16).
__device__ __forceinline__ float column_sum(float x) {
#pragma unroll
  for (uint32_t m = 16; m < kWaveSize; m *= 2) {
    x += __shfl_xor(x, m, kWaveSize);
  }
  return x;
}

__device__ __forceinline__ float column_max(float x) {
#pragma unroll
  for (uint32_t m = 16; m < kWaveSize; m *= 2) {
    x = max(x, __shfl_xor(x, m, kWaveSize));
  }
  return x;
}

// One A or B operand: kOpElems 16-bit values, two per register.
struct operand_t {
  uint32_t r[kOpElems / 2];
};

template <typename T>
__device__ __forceinline__ uint16_t to_bits(float x) {
  if constexpr (std::is_same<T, half>::value) {
    return __half_as_ushort(__float2half_rn(x));
  } else {
    return __bfloat16_as_ushort(__float2bfloat16(x));
  }
}

template <typename T>
__device__ __forceinline__ float from_bits(uint16_t x) {
  if constexpr (std::is_same<T, half>::value) {
    return __half2float(__ushort_as_half(x));
  } else {
    return __uint_as_float(uint32_t(x) << 16);
  }
}

// c += a * b for one 16x16x16 tile.
template <typename T>
__device__ __forceinline__ void mma(float (&c)[kAccElems], const operand_t& a,
                                    const operand_t& b) {
  typedef float acc_v __attribute__((ext_vector_type(kAccElems)));
  acc_v acc;
#pragma unroll
  for (uint32_t i = 0; i < kAccElems; ++i) acc[i] = c[i];
  if constexpr (std::is_same<T, half>::value) {
    typedef _Float16 op_v __attribute__((ext_vector_type(kOpElems)));
    const op_v av = __builtin_bit_cast(op_v, a), bv = __builtin_bit_cast(op_v, b);
#if QUEST_PREFILL_MMA == 1
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(av, bv, acc, 0, 0, 0);
#elif QUEST_PREFILL_MMA == 11
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);
#else
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
#endif
  } else {
    typedef short op_v __attribute__((ext_vector_type(kOpElems)));
    const op_v av = __builtin_bit_cast(op_v, a), bv = __builtin_bit_cast(op_v, b);
#if QUEST_PREFILL_MMA == 1
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
#elif QUEST_PREFILL_MMA == 11
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av, bv, acc);
#else
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, acc);
#endif
  }
#pragma unroll
  for (uint32_t i = 0; i < kAccElems; ++i) c[i] = acc[i];
}

// Address of element (row, col) of a tile in the permuted shared memory layout.
template <uint32_t channel_size_128b, typename T>
__device__ __forceinline__ T* smem_elem(smem_t* smem, uint32_t row, uint32_t col) {
  return (T*)(smem->base + smem_t::get_permuted_offset<channel_size_128b>(row, col / 8)) +
         col % 8;
}

// Operand elements (row, col0 .. col0 + kOpElems - 1): consecutive in memory within each
// 128-bit chunk.
template <uint32_t channel_size_128b, typename T>
__device__ __forceinline__ void load_row(smem_t* smem, uint32_t row, uint32_t col0,
                                         operand_t& op) {
  constexpr uint32_t kPiece = kOpElems < 8 ? kOpElems : 8;
#pragma unroll
  for (uint32_t p = 0; p < kOpElems; p += kPiece) {
    const uint32_t* src = (const uint32_t*)smem_elem<channel_size_128b, T>(smem, row, col0 + p);
    if constexpr (kPiece == 4) {
      *(uint2*)&op.r[p / 2] = *(const uint2*)src;
    } else {
      *(uint4*)&op.r[p / 2] = *(const uint4*)src;
    }
  }
}

// One thread's share of a K or V tile, held in registers between its global load and its
// shared memory store.
template <uint32_t num_warps, uint32_t num_frags_y, uint32_t num_frags_z>
struct kv_regs_t {
  static constexpr uint32_t num_rows = num_frags_z * 4 / num_warps, num_cols = num_frags_y / 4;
  // A native vector rather than uint4, whose union wrapper keeps the array in scratch memory.
  typedef uint32_t b128_v __attribute__((ext_vector_type(4)));
  b128_v data[num_rows][num_cols];
};

// The paged pointers are always in bounds (rows past kv_len resolve to a slot of an
// allocated page, pages past the end to the cache base), so the loads are unconditional and
// only the shared memory stores are predicated.
template <bool produce_v, uint32_t page_size, uint32_t num_warps, uint32_t num_frags_y,
          uint32_t num_frags_z, PageStorage page_storage, QKVLayout kv_layout, typename DType,
          typename IdType>
__device__ __forceinline__ void page_load_kv(
    kv_regs_t<num_warps, num_frags_y, num_frags_z>& regs,
    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, const uint32_t page_iter_base,
    const IdType last_indptr) {
  using regs_t = kv_regs_t<num_warps, num_frags_y, num_frags_z>;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t kv_head_idx = blockIdx.z;
#pragma unroll
  for (uint32_t i = 0; i < regs_t::num_rows; ++i) {
    const uint32_t row = 4 * num_warps * i + ty * 4 + tx / 8;
    const uint32_t page_iter = page_iter_base + row / page_size, entry_idx = row % page_size;
    const DType* gptr =
        produce_v ? paged_kv.protective_get_v_ptr(page_iter, kv_head_idx, entry_idx,
                                                  (tx % 8) * num_elems_per_128b<DType>(),
                                                  last_indptr)
                  : paged_kv.protective_get_k_ptr(page_iter, kv_head_idx, entry_idx,
                                                  (tx % 8) * num_elems_per_128b<DType>(),
                                                  last_indptr);
#pragma unroll
    for (uint32_t j = 0; j < regs_t::num_cols; ++j) {
      regs.data[i][j] =
          *(const typename regs_t::b128_v*)(gptr + j * 8 * num_elems_per_128b<DType>());
    }
  }
}

// V is kept transposed in shared memory, [head_dim][16 * num_frags_z], so that the V^T
// operand of O^T = V^T P^T is a contiguous read. Its 128-bit chunks are permuted by the row
// (d) index to spread a column of reads over the banks.
template <uint32_t num_frags_z>
__device__ __forceinline__ uint32_t vt_offset(uint32_t d, uint32_t kv) {
  constexpr uint32_t num_chunks = 2 * num_frags_z;
  return d * 16 * num_frags_z + (((kv / 8) ^ (d % num_chunks)) * 8) + kv % 8;
}

// Operand elements (d, kv0 .. kv0 + kOpElems - 1) of the transposed V tile.
template <uint32_t num_frags_z>
__device__ __forceinline__ void load_vt(smem_t* smem, uint32_t d, uint32_t kv0, operand_t& op) {
  constexpr uint32_t kPiece = kOpElems < 8 ? kOpElems : 8;
#pragma unroll
  for (uint32_t p = 0; p < kOpElems; p += kPiece) {
    const uint16_t* src = (const uint16_t*)smem->base + vt_offset<num_frags_z>(d, kv0 + p);
    if constexpr (kPiece == 4) {
      *(uint2*)&op.r[p / 2] = *(const uint2*)src;
    } else {
      *(uint4*)&op.r[p / 2] = *(const uint4*)src;
    }
  }
}

// K rows past kv_len are left as they are (their scores are masked); V rows past kv_len are
// zero-filled, as their products are not.
template <bool produce_v, uint32_t num_warps, uint32_t num_frags_y, uint32_t num_frags_z,
          typename DType>
__device__ __forceinline__ void page_store_kv(
    const kv_regs_t<num_warps, num_frags_y, num_frags_z>& regs, smem_t smem,
    uint32_t* smem_offset, const uint32_t kv_idx_base, const uint32_t kv_len) {
  using regs_t = kv_regs_t<num_warps, num_frags_y, num_frags_z>;
  constexpr uint32_t channel_size_128b_in = num_frags_y * 16 / num_elems_per_128b<DType>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;
  if constexpr (produce_v) {
#pragma unroll
    for (uint32_t i = 0; i < regs_t::num_rows; ++i) {
      const uint32_t row = 4 * num_warps * i + ty * 4 + tx / 8;
#pragma unroll
      for (uint32_t j = 0; j < regs_t::num_cols; ++j) {
        const uint32_t d0 = (tx % 8) * 8 + j * 64;
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) {
          const uint32_t word = regs.data[i][j][e / 2];
          const uint16_t x = kv_idx < kv_len ? uint16_t(e % 2 ? word >> 16 : word & 0xffff) : 0;
          ((uint16_t*)smem.base)[vt_offset<num_frags_z>(d0 + e, row)] = x;
        }
      }
      kv_idx += num_warps * 4;
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < regs_t::num_rows; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < regs_t::num_cols; ++j) {
        if (kv_idx < kv_len) {
          *(typename regs_t::b128_v*)(smem.base + *smem_offset) = regs.data[i][j];
        }
        *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
      }
      kv_idx += num_warps * 4;
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
          2 * num_frags_y;
    }
    *smem_offset -= num_frags_z * 16 * channel_size_128b_in;
  }
}

template <bool produce_v, uint32_t page_size, uint32_t num_warps, uint32_t num_frags_y,
          uint32_t num_frags_z, PageStorage page_storage, QKVLayout kv_layout, typename DType,
          typename IdType>
__device__ __forceinline__ void page_produce_kv(
    smem_t smem, uint32_t* smem_offset,
    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, const uint32_t kv_idx_base,
    const uint32_t page_iter_base, const uint32_t kv_len, const IdType last_indptr) {
  kv_regs_t<num_warps, num_frags_y, num_frags_z> regs;
  page_load_kv<produce_v, page_size>(regs, paged_kv, page_iter_base, last_indptr);
  page_store_kv<produce_v, num_warps, num_frags_y, num_frags_z, DType>(regs, smem, smem_offset,
                                                                       kv_idx_base, kv_len);
}

/*!
 * \brief Fragment state of one wavefront: num_frags_x * kWarpsPerWave query tiles of 16 rows.
 */
template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
struct state_t {
  static constexpr uint32_t num_q_tiles = num_frags_x * kWarpsPerWave;
  static constexpr bool q_in_regs = q_in_registers(num_frags_x, num_frags_y, kShape);
  // q[qx][fy]: the B operand Q^T, scaled, when the query tile is kept in registers.
  operand_t q[q_in_regs ? num_q_tiles : 1][q_in_regs ? num_frags_y : 1];
  // s[qx][fz]: S^T for 16 kv rows x 16 query columns.
  float s[num_q_tiles][num_frags_z][kAccElems];
  // o[qx][fy]: O^T for 16 head dimensions x 16 query columns.
  float o[num_q_tiles][num_frags_y][kAccElems];
  float m[num_q_tiles];
  // The lane's share of the softmax denominator; column_sum gives the row's.
  float d[num_q_tiles];
};

// First shared memory row (within the threadblock's tile) of this wavefront's queries.
template <uint32_t num_frags_x>
__device__ __forceinline__ uint32_t wave_row_base() {
  return (threadIdx.y & ~(kWarpsPerWave - 1)) * num_frags_x * 16;
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void init_states(state_t<num_frags_x, num_frags_y, num_frags_z>& st) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) st.o[qx][fy][i] = 0.f;
    }
    st.m[qx] = -5e4f;
    // As in the mma.sync path the denominator starts at 1, here held by one lane per column.
    st.d[qx] = lane_id() < 16 ? 1.f : 0.f;
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn>
__device__ __forceinline__ void compute_qk(smem_t* q_smem, smem_t* k_smem,
                                           state_t<num_frags_x, num_frags_y, num_frags_z>& st) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
  constexpr uint32_t channel_size_128b_in = num_frags_y * 16 / num_elems_per_128b<DTypeIn>();
  const uint32_t col = lane_id() % 16, k0 = op_k0(), q_row0 = wave_row_base<num_frags_x>();
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) st.s[qx][fz][i] = 0.f;
    }
  }
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    operand_t q_op[state::num_q_tiles];
#pragma unroll
    for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
      if constexpr (state::q_in_regs) {
        q_op[qx] = st.q[qx][fy];
      } else {
        load_row<channel_size_128b_in, DTypeIn>(q_smem, q_row0 + qx * 16 + col, fy * 16 + k0,
                                                q_op[qx]);
      }
    }
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      operand_t k_op;
      load_row<channel_size_128b_in, DTypeIn>(k_smem, fz * 16 + col, fy * 16 + k0, k_op);
#pragma unroll
      for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
        mma<DTypeIn>(st.s[qx][fz], k_op, q_op[qx]);
      }
    }
  }
}

// q_row_base: the global query row (before dividing by group_size) of the wavefront's first
// row, i.e. (tile_idx * num_warps + first warp of the wavefront) * num_frags_x * 16.
template <bool causal, uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z>
__device__ __forceinline__ void mask_s(const uint32_t q_row_base, const uint32_t kv_idx_base,
                                       const uint32_t qo_len, const uint32_t kv_len,
                                       const uint32_t chunk_end,
                                       state_t<num_frags_x, num_frags_y, num_frags_z>& st) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
  const uint32_t col = lane_id() % 16;
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
    const uint32_t q_idx = (q_row_base + qx * 16 + col) / group_size;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) {
        const uint32_t kv_idx = kv_idx_base + fz * 16 + acc_row(i);
        const bool out_of_boundary =
            causal ? kv_idx > kv_len + q_idx - qo_len : kv_idx >= chunk_end;
        st.s[qx][fz][i] = out_of_boundary ? -5e4f : st.s[qx][fz][i];
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void update_mdo_states(
    state_t<num_frags_x, num_frags_y, num_frags_z>& st) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
    const float m_prev = st.m[qx];
    float m = m_prev;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) m = max(m, st.s[qx][fz][i]);
    }
    m = column_max(m);
    st.m[qx] = m;
    const float o_scale = __builtin_amdgcn_exp2f(m_prev - m);
    st.d[qx] *= o_scale;
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) st.o[qx][fy][i] *= o_scale;
    }
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) {
        st.s[qx][fz][i] = __builtin_amdgcn_exp2f(st.s[qx][fz][i] - m);
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn>
__device__ __forceinline__ void compute_sfm_v(smem_t* v_smem,
                                              state_t<num_frags_x, num_frags_y, num_frags_z>& st) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
  // P^T in the B layout, rounded to the input type as in the mma.sync path. The denominator
  // sums the rounded values, as the mma.sync path's rowsum does.
  operand_t p_op[state::num_q_tiles][num_frags_z];
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      uint16_t p[kAccElems];
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) {
        p[i] = to_bits<DTypeIn>(st.s[qx][fz][i]);
        st.d[qx] += from_bits<DTypeIn>(p[i]);
      }
#if QUEST_PREFILL_MMA == 11
      // The D layout holds kv rows 2i + l / 16 and the B layout wants all 16 in each lane:
      // the other half comes from lane l ^ 16.
      uint32_t mine[kAccElems / 2], other[kAccElems / 2];
#pragma unroll
      for (uint32_t i = 0; i < kAccElems / 2; ++i) {
        mine[i] = uint32_t(p[2 * i]) | (uint32_t(p[2 * i + 1]) << 16);
        other[i] = __shfl_xor(mine[i], 16, kWaveSize);
      }
      const bool upper = lane_id() >= 16;
#pragma unroll
      for (uint32_t i = 0; i < kAccElems / 2; ++i) {
        // kv rows 4i .. 4i + 3 come from elements 2i (rows 4i + g) and 2i + 1 (4i + 2 + g)
        // of the lane with g = 0 and the lane with g = 1.
        const uint32_t g0 = upper ? other[i] : mine[i], g1 = upper ? mine[i] : other[i];
        p_op[qx][fz].r[2 * i] = (g0 & 0xffffu) | (g1 << 16);
        p_op[qx][fz].r[2 * i + 1] = (g0 >> 16) | (g1 & 0xffff0000u);
      }
#else
#pragma unroll
      for (uint32_t i = 0; i < kAccElems / 2; ++i) {
        p_op[qx][fz].r[i] = uint32_t(p[2 * i]) | (uint32_t(p[2 * i + 1]) << 16);
      }
#endif
    }
  }
  const uint32_t col = lane_id() % 16, k0 = op_k0();
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      operand_t v_op;
      load_vt<num_frags_z>(v_smem, fy * 16 + col, fz * 16 + k0, v_op);
#pragma unroll
      for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
        mma<DTypeIn>(st.o[qx][fy], v_op, p_op[qx][fz]);
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void normalize_d(state_t<num_frags_x, num_frags_y, num_frags_z>& st) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
    st.d[qx] = column_sum(st.d[qx]);
    const float d_rcp = math::ptx_rcp(st.d[qx]);
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) st.o[qx][fy][i] *= d_rcp;
    }
  }
}

// Load the wavefront's query rows from global memory into the Q^T operands, scaled by
// sm_scale and rounded to the input type as the shared memory path does. Rows past
// qo_upper_bound are zero.
template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          typename DTypeIn>
__device__ __forceinline__ void load_q(state_t<num_frags_x, num_frags_y, num_frags_z>& st,
                                       const DTypeIn* q, const uint32_t q_row_base,
                                       const uint32_t qo_upper_bound,
                                       const uint32_t qo_indptr_base, const uint32_t kv_head_idx,
                                       const uint32_t qo_len, const uint32_t num_qo_heads,
                                       const float sm_scale) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
  constexpr uint32_t head_dim = num_frags_y * 16;
  const uint32_t col = lane_id() % 16, k0 = op_k0();
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
    const uint32_t row = q_row_base + qx * 16 + col;
    const uint32_t qo_idx = row / group_size;
    const DTypeIn* q_ptr = q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                                   qo_indptr_base + qo_idx, kv_head_idx * group_size + row % group_size,
                                   k0, qo_len, num_qo_heads);
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t j = 0; j < kOpElems / 2; ++j) {
        uint32_t x = 0;
        if (qo_idx < qo_upper_bound) {
          const uint32_t raw = *(const uint32_t*)(q_ptr + fy * 16 + 2 * j);
          x = uint32_t(to_bits<DTypeIn>(from_bits<DTypeIn>(raw & 0xffff) * sm_scale)) |
              (uint32_t(to_bits<DTypeIn>(from_bits<DTypeIn>(raw >> 16) * sm_scale)) << 16);
        }
        st.q[qx][fy].r[j] = x;
      }
    }
  }
}

// Store the normalized output from registers. Lane l holds head dimensions fy * 16 +
// acc_row(i) of query row l % 16 in each tile.
template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          typename DTypeOut>
__device__ __forceinline__ void write_o(state_t<num_frags_x, num_frags_y, num_frags_z>& st,
                                        DTypeOut* o, const uint32_t q_row_base,
                                        const uint32_t qo_upper_bound,
                                        const uint32_t qo_indptr_base, const uint32_t kv_head_idx,
                                        const uint32_t qo_len, const uint32_t num_qo_heads) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
  constexpr uint32_t head_dim = num_frags_y * 16;
  const uint32_t col = lane_id() % 16;
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
    const uint32_t row = q_row_base + qx * 16 + col;
    const uint32_t qo_idx = row / group_size;
    if (qo_idx >= qo_upper_bound) continue;
    DTypeOut* o_ptr = o + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                              qo_indptr_base + qo_idx, kv_head_idx * group_size + row % group_size,
                              0, qo_len, num_qo_heads);
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#if QUEST_PREFILL_MMA == 11
#pragma unroll
      for (uint32_t i = 0; i < kAccElems; ++i) {
        ((uint16_t*)o_ptr)[fy * 16 + acc_row(i)] = to_bits<DTypeOut>(st.o[qx][fy][i]);
      }
#else
      // acc_row(i) = acc_row(0) + i: the lane's values are consecutive in the row.
      uint32_t packed[kAccElems / 2];
#pragma unroll
      for (uint32_t i = 0; i < kAccElems / 2; ++i) {
        packed[i] = uint32_t(to_bits<DTypeOut>(st.o[qx][fy][2 * i])) |
                    (uint32_t(to_bits<DTypeOut>(st.o[qx][fy][2 * i + 1])) << 16);
      }
      uint32_t* dst = (uint32_t*)(o_ptr + fy * 16 + acc_row(0));
      if constexpr (kAccElems == 4) {
        *(uint2*)dst = *(uint2*)packed;
      } else {
        *(uint4*)dst = *(uint4*)packed;
      }
#endif
    }
  }
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void write_lse(state_t<num_frags_x, num_frags_y, num_frags_z>& st,
                                          float* lse, const uint32_t q_row_base,
                                          const uint32_t qo_upper_bound,
                                          const uint32_t qo_indptr_base,
                                          const uint32_t kv_head_idx,
                                          const uint32_t num_qo_heads) {
  using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
  const uint32_t col = lane_id() % 16;
  if (lane_id() >= 16) return;
#pragma unroll
  for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
    const uint32_t row = q_row_base + qx * 16 + col;
    const uint32_t qo_idx = row / group_size, qo_head_idx = kv_head_idx * group_size + row % group_size;
    if (qo_idx < qo_upper_bound) {
      lse[(qo_indptr_base + qo_idx) * num_qo_heads + qo_head_idx] =
          math::ptx_log2(st.d[qx]) + st.m[qx];
    }
  }
}

}  // namespace matrix_core
}  // namespace flashinfer

#endif  // defined(QUEST_PREFILL_MMA)

#endif  // FLASHINFER_PREFILL_MATRIX_CORE_CUH_
