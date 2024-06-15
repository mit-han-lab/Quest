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

/*
  This file is modified based on URL:
      https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/prefill.cuh
  Support for Page-Sparsity Self-Attention by dynamic selection.
  Only modify the page-related code.
*/

#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda_runtime.h>
#include <tuple>

#include "flashinfer/cascade.cuh"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/layout.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/mma.cuh"
#include "flashinfer/permuted_smem.cuh"
#include "flashinfer/rope.cuh"
#include "flashinfer/state.cuh"
#include "flashinfer/utils.cuh"

#include "decode/decode_page.cuh"

namespace flashinfer
{

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t warp_size = 32;

namespace
{

enum class FragLayout
{
	kRowMajor,
	kColMajor,
};

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_mode The fill mode of the shared memory.
 * \tparam num_frags_y The number of fragments in y dimension.
 * \tparam num_frags_z The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam kv_layout The layout of the input tensor.
 * \tparam group_size The number of qo heads that maps to a kv head (used in GQA).
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param qkv_info The tensor info of the input tensor.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <SharedMemFillMode fill_mode,
		  uint32_t num_warps,
		  uint32_t num_frags_y,
		  uint32_t num_frags_z,
		  typename T>
__device__ __forceinline__ void produce_kv(smem_t smem,
										   uint32_t* smem_offset,
										   T** gptr,
										   const uint32_t kv_n_stride,
										   const uint32_t kv_idx_base,
										   const uint32_t kv_len) {
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<T>();
	const uint32_t tx = threadIdx.x, ty = threadIdx.y;
	uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;
#pragma unroll
	for(uint32_t i = 0; i < num_frags_z * 4 / num_warps; ++i) {
#pragma unroll
		for(uint32_t j = 0; j < num_frags_y / 4; ++j) {
			smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
			*smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
			*gptr += 8 * num_elems_per_128b<T>();
		}
		kv_idx += num_warps * 4;
		*smem_offset =
			smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
			2 * num_frags_y;
		*gptr += num_warps * 4 * kv_n_stride - 2 * num_frags_y * num_elems_per_128b<T>();
	}
	*smem_offset -= num_frags_z * 16 * channel_size_128b_in;
}

template <bool produce_v,
		  uint32_t page_size,
		  uint32_t num_warps,
		  uint32_t num_frags_y,
		  uint32_t num_frags_z,
		  PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DType,
		  typename IdType>
__device__ __forceinline__ void
page_produce_kv(smem_t smem,
				uint32_t* smem_offset,
				paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv,
				const uint32_t kv_idx_base,
				const uint32_t page_iter_base,
				const uint32_t kv_len,
				const IdType last_indptr) {
	constexpr SharedMemFillMode fill_mode =
		produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DType>();
	const uint32_t tx = threadIdx.x, ty = threadIdx.y;
	const uint32_t kv_head_idx = blockIdx.z;
	uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;
	if constexpr(page_size % 4 == 0) {
#pragma unroll
		for(uint32_t i = 0; i < num_frags_z * 4 / num_warps; ++i) {
			const uint32_t page_iter = page_iter_base + (4 * num_warps * i + ty * 4) / page_size;
			const uint32_t entry_idx = (4 * num_warps * i + ty * 4) % page_size + tx / 8;
			DType* gptr =
				produce_v ? paged_kv.protective_get_v_ptr(page_iter,
														  kv_head_idx,
														  entry_idx,
														  (tx % 8) * num_elems_per_128b<DType>(),
														  last_indptr)
						  : paged_kv.protective_get_k_ptr(page_iter,
														  kv_head_idx,
														  entry_idx,
														  (tx % 8) * num_elems_per_128b<DType>(),
														  last_indptr);
#pragma unroll
			for(uint32_t j = 0; j < num_frags_y / 4; ++j) {
				smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
				*smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
				gptr += 8 * num_elems_per_128b<DType>();
			}
			kv_idx += num_warps * 4;
			*smem_offset =
				smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
				2 * num_frags_y;
		}
		*smem_offset -= num_frags_z * 16 * channel_size_128b_in;
	} else {
#pragma unroll
		for(uint32_t i = 0; i < num_frags_z * 4 / num_warps; ++i) {
			const uint32_t page_iter =
				page_iter_base + (4 * num_warps * i + ty * 4 + tx / 8) / page_size;
			const uint32_t entry_idx = (4 * num_warps * i + ty * 4 + tx / 8) % page_size;
			DType* gptr =
				produce_v ? paged_kv.protective_get_v_ptr(page_iter,
														  kv_head_idx,
														  entry_idx,
														  (tx % 8) * num_elems_per_128b<DType>(),
														  last_indptr)
						  : paged_kv.protective_get_k_ptr(page_iter,
														  kv_head_idx,
														  entry_idx,
														  (tx % 8) * num_elems_per_128b<DType>(),
														  last_indptr);
#pragma unroll
			for(uint32_t j = 0; j < num_frags_y / 4; ++j) {
				smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
				*smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
				gptr += 8 * num_elems_per_128b<DType>();
			}
			kv_idx += num_warps * 4;
			*smem_offset =
				smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
				2 * num_frags_y;
		}
		*smem_offset -= num_frags_z * 16 * channel_size_128b_in;
	}
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void
init_states(float (*o_frag)[num_frags_y][8], DTypeQKAccum (*m)[2], float (*d)[2]) {
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
			for(uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
				o_frag[fx][fy][reg_id] = 0.f;
			}
		}
	}
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t j = 0; j < 2; ++j) {
			m[fx][j] = DTypeQKAccum(-5e4);
			d[fx][j] = 1.f;
		}
	}
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeIn>
__device__ __forceinline__ void load_q_global_smem(uint32_t q_idx_base,
												   const uint32_t qo_upper_bound,
												   DTypeIn* q_ptr_base,
												   const uint32_t qo_n_stride,
												   const uint32_t qo_h_stride,
												   smem_t* q_smem) {
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
	const uint32_t tx = threadIdx.x, ty = threadIdx.y;
	uint32_t q_smem_offset_w =
		smem_t::get_permuted_offset<channel_size_128b_in>(ty * num_frags_x * 16 + tx / 8, tx % 8);

	q_idx_base += (tx / 8) / group_size;
	q_ptr_base += ((tx / 8) / group_size) * qo_n_stride + ((tx / 8) % group_size) * qo_h_stride;
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t j = 0; j < 4; ++j) {
			const uint32_t q_idx = q_idx_base + (fx * 16 + j * 4) / group_size;
			DTypeIn* q_ptr = q_ptr_base + ((fx * 16 + j * 4) / group_size) * qo_n_stride +
							 ((fx * 16 + j * 4) % group_size) * qo_h_stride;
#pragma unroll
			for(uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
				// load q fragment from gmem to smem
				q_smem->load_128b_async<SharedMemFillMode::kNoFill>(
					q_smem_offset_w, q_ptr, q_idx < qo_upper_bound);
				q_smem_offset_w = q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
				q_ptr += 8 * num_elems_per_128b<DTypeIn>();
			}
			q_smem_offset_w =
				q_smem->advance_offset_by_row<4, channel_size_128b_in>(q_smem_offset_w) -
				2 * num_frags_y;
		}
	}
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeIn>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(smem_t* q_smem,
																 const float sm_scale) {
	const uint32_t tx = threadIdx.x, ty = threadIdx.y;
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
#pragma unroll
	for(uint32_t i = 0; i < num_frags_x * 16 * head_dim / 256; ++i) {
		vec_t<DTypeIn, 8> tmp;
		tmp.load((DTypeIn*)(q_smem->base + ty * num_frags_x * 16 * channel_size_128b_in) + i * 256 +
				 tx * 8);
#pragma unroll
		for(uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
			tmp[reg_id] *= sm_scale;
		}
		tmp.store((DTypeIn*)(q_smem->base + ty * num_frags_x * 16 * channel_size_128b_in) +
				  i * 256 + tx * 8);
	}
}

template <uint32_t num_frags_x,
		  uint32_t num_frags_y,
		  uint32_t num_frags_z,
		  typename DTypeIn,
		  typename DTypeQKAccum>
__device__ __forceinline__ void compute_qk(smem_t* q_smem,
										   uint32_t* q_smem_offset_r,
										   smem_t* k_smem,
										   uint32_t* k_smem_offset_r,
										   DTypeQKAccum (*s_frag)[num_frags_z][8]) {
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
	uint32_t a_frag[num_frags_x][4], b_frag[4];
	// compute q*k^T
#pragma unroll
	for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
		for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
			q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
			*q_smem_offset_r =
				q_smem->advance_offset_by_row<16, channel_size_128b_in>(*q_smem_offset_r);
		}

		*q_smem_offset_r = q_smem->advance_offset_by_column<2>(*q_smem_offset_r, fy) -
						   num_frags_x * 16 * channel_size_128b_in;

#pragma unroll
		for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
			k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
			*k_smem_offset_r =
				k_smem->advance_offset_by_row<16, channel_size_128b_in>(*k_smem_offset_r);
#pragma unroll
			for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
				if constexpr(std::is_same<DTypeQKAccum, float>::value) {
					if(fy == 0) {
						mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn, MMAMode::kInit>(
							s_frag[fx][fz], a_frag[fx], b_frag);
					} else {
						mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
							s_frag[fx][fz], a_frag[fx], b_frag);
					}
				} else if(std::is_same<DTypeQKAccum, half>::value) {
					if(fy == 0) {
						mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
							(uint32_t*)s_frag[fx][fz], a_frag[fx], b_frag);
					} else {
						mma::mma_sync_m16n16k16_row_col_f16f16f16(
							(uint32_t*)s_frag[fx][fz], a_frag[fx], b_frag);
					}
				}
			}
		}
		*k_smem_offset_r = k_smem->advance_offset_by_column<2>(*k_smem_offset_r, fy) -
						   num_frags_z * 16 * channel_size_128b_in;
	}
	*q_smem_offset_r -= num_frags_y * 2;
	*k_smem_offset_r -= num_frags_y * 2;
}

template <bool partition_kv,
		  bool causal,
		  uint32_t group_size,
		  uint32_t num_warps,
		  uint32_t num_frags_x,
		  uint32_t num_frags_y,
		  uint32_t num_frags_z,
		  typename DTypeQKAccum>
__device__ __forceinline__ void mask_s(const uint32_t qo_idx_base,
									   const uint32_t kv_idx_base,
									   const uint32_t qo_len,
									   const uint32_t kv_len,
									   const uint32_t chunk_end,
									   DTypeQKAccum (*s_frag)[num_frags_z][8]) {
	const uint32_t tx = threadIdx.x;
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
			for(uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
				const uint32_t q_idx = qo_idx_base +
									   (fx * 16 + tx / 4 + 8 * ((reg_id % 4) / 2)) / group_size,
							   kv_idx = kv_idx_base + fz * 16 + 2 * (tx % 4) + 8 * (reg_id / 4) +
										reg_id % 2;
				const bool out_of_boundary = (causal ? (kv_idx > kv_len + q_idx - qo_len ||
														(partition_kv && kv_idx >= chunk_end))
													 : kv_idx >= chunk_end);
				s_frag[fx][fz][reg_id] =
					out_of_boundary ? DTypeQKAccum(-5e4) : s_frag[fx][fz][reg_id];
			}
		}
	}
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeQKAccum>
__device__ __forceinline__ void update_mdo_states(DTypeQKAccum (*s_frag)[num_frags_z][8],
												  float (*o_frag)[num_frags_y][8],
												  DTypeQKAccum (*m)[2],
												  float (*d)[2]) {
	if constexpr(std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
		for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
			for(uint32_t j = 0; j < 2; ++j) {
				float m_prev = m[fx][j];
#pragma unroll
				for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
					float m_local = max(max(s_frag[fx][fz][j * 2 + 0], s_frag[fx][fz][j * 2 + 1]),
										max(s_frag[fx][fz][j * 2 + 4], s_frag[fx][fz][j * 2 + 5]));
					m[fx][j] = max(m[fx][j], m_local);
				}
				m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x2));
				m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x1));

				float o_scale = math::ptx_exp2(m_prev - m[fx][j]);
				d[fx][j] *= o_scale;
#pragma unroll
				for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
					o_frag[fx][fy][j * 2 + 0] *= o_scale;
					o_frag[fx][fy][j * 2 + 1] *= o_scale;
					o_frag[fx][fy][j * 2 + 4] *= o_scale;
					o_frag[fx][fy][j * 2 + 5] *= o_scale;
				}
#pragma unroll
				for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
					s_frag[fx][fz][j * 2 + 0] =
						math::ptx_exp2(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
					s_frag[fx][fz][j * 2 + 1] =
						math::ptx_exp2(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
					s_frag[fx][fz][j * 2 + 4] =
						math::ptx_exp2(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
					s_frag[fx][fz][j * 2 + 5] =
						math::ptx_exp2(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
				}
			}
		}
	} else if constexpr(std::is_same<DTypeQKAccum, half>::value) {
#pragma unroll
		for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
			half m_prev[2];
#pragma unroll
			for(uint32_t j = 0; j < 2; ++j) {
				m_prev[j] = m[fx][j];
#pragma unroll
				for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
					half2 m_local = __hmax2(*(half2*)&s_frag[fx][fz][j * 2],
											*(half2*)&s_frag[fx][fz][j * 2 + 4]);
					m[fx][j] = __hmax(m[fx][j], __hmax(m_local.x, m_local.y));
				}
			}
			*(half2*)&m[fx] = __hmax2(*(half2*)&m[fx], math::shfl_xor_sync(*(half2*)&m[fx], 0x2));
			*(half2*)&m[fx] = __hmax2(*(half2*)&m[fx], math::shfl_xor_sync(*(half2*)&m[fx], 0x1));
#pragma unroll
			for(uint32_t j = 0; j < 2; ++j) {
				float o_scale = math::ptx_exp2(float(m_prev[j] - m[fx][j]));
				d[fx][j] *= o_scale;
#pragma unroll
				for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
					o_frag[fx][fy][j * 2 + 0] *= o_scale;
					o_frag[fx][fy][j * 2 + 1] *= o_scale;
					o_frag[fx][fy][j * 2 + 4] *= o_scale;
					o_frag[fx][fy][j * 2 + 5] *= o_scale;
				}
				half2 m2 = make_half2(m[fx][j], m[fx][j]);
#pragma unroll
				for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
					*(half2*)&s_frag[fx][fz][j * 2] =
						math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2] - m2);
					*(half2*)&s_frag[fx][fz][j * 2 + 4] =
						math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2 + 4] - m2);
				}
			}
		}
	}
}

template <uint32_t num_frags_x,
		  uint32_t num_frags_y,
		  uint32_t num_frags_z,
		  typename DTypeIn,
		  typename DTypeQKAccum>
__device__ __forceinline__ void compute_sfm_v(smem_t* v_smem,
											  uint32_t* v_smem_offset_r,
											  DTypeQKAccum (*s_frag)[num_frags_z][8],
											  float (*o_frag)[num_frags_y][8],
											  float (*d)[2]) {
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();

	DTypeIn s_frag_f16[num_frags_x][num_frags_z][8];
	if constexpr(std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
		for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
			for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
				vec_cast<DTypeIn, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
			}
		}
	}

#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
			if constexpr(std::is_same<DTypeQKAccum, float>::value) {
				mma::rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
			} else {
				mma::rowsum_f16f16f32(d[fx], s_frag[fx][fz]);
			}
		}
	}

#pragma unroll
	for(uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
		for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
			uint32_t b_frag[4];
			v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
#pragma unroll
			for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
				if constexpr(std::is_same<DTypeQKAccum, float>::value) {
					mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
						o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
				} else {
					mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
						o_frag[fx][fy], (uint32_t*)s_frag[fx][fz], b_frag);
				}
			}
			*v_smem_offset_r = v_smem->advance_offset_by_column<2>(*v_smem_offset_r, fy);
		}
		*v_smem_offset_r =
			v_smem->advance_offset_by_row<16, channel_size_128b_in>(*v_smem_offset_r) -
			2 * num_frags_y;
	}
	*v_smem_offset_r -= 16 * num_frags_z * channel_size_128b_in;
}

template <uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_frags_y][8], float (*d)[2]) {
	float d_rcp[num_frags_x][2];
	// compute reciprocal of d
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t j = 0; j < 2; ++j) {
			d_rcp[fx][j] = math::ptx_rcp(d[fx][j]);
		}
	}

#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
			for(uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
				o_frag[fx][fy][reg_id] = o_frag[fx][fy][reg_id] * d_rcp[fx][(reg_id % 4) / 2];
			}
		}
	}
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void grid_sync_mdo_states(float (*o_frag)[num_frags_y][8],
													 float* tmp,
													 DTypeQKAccum (*m)[2],
													 float (*d)[2]) {
	const uint32_t bx = blockIdx.x;
	const uint32_t num_chunks = gridDim.y;
	const uint32_t kv_head_idx = blockIdx.z;
	// aggregate global state
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
			vec_t<float, 8>::memcpy(
				tmp + ((fx * num_frags_y + fy) * grid.size() + grid.thread_rank()) * 8,
				o_frag[fx][fy]);
#pragma unroll
			for(uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
				o_frag[fx][fy][reg_id] = 0.f;
			}
		}
	}
	float* tmp_md = tmp + num_frags_x * num_frags_y * 8 * grid.size();
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t j = 0; j < 2; ++j) {
			*(float2*)&tmp_md[(((fx * 2 + j) * grid.size() + grid.thread_rank())) * 2] =
				make_float2(float(m[fx][j]), d[fx][j]);
			m[fx][j] = DTypeQKAccum(-5e4);
			d[fx][j] = 1.f;
		}
	}

	grid.sync();

	for(uint32_t iter = 0; iter < num_chunks; ++iter) {
		float other_scale[num_frags_x][2];
#pragma unroll
		for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
			for(uint32_t j = 0; j < 2; ++j) {
				float2 md =
					*(float2*)&tmp_md[((fx * 2 + j) * grid.size() +
									   ((kv_head_idx * num_chunks + iter) * gridDim.x + bx) *
										   block.num_threads() +
									   block.thread_rank()) *
									  2];
				float mi = md.x, di = md.y, m_prev = float(m[fx][j]);
				float m_new = max(m_prev, mi);
				m[fx][j] = m_new;
				float o_scale = math::ptx_exp2(m_prev - m_new);
				other_scale[fx][j] = math::ptx_exp2(mi - m_new);
				d[fx][j] = d[fx][j] * o_scale + di * other_scale[fx][j];
#pragma unroll
				for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
					o_frag[fx][fy][j * 2 + 0] *= o_scale;
					o_frag[fx][fy][j * 2 + 1] *= o_scale;
					o_frag[fx][fy][j * 2 + 4] *= o_scale;
					o_frag[fx][fy][j * 2 + 5] *= o_scale;
				}
			}
#pragma unroll
			for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
				vec_t<float, 8> o_frag_i;
				o_frag_i.load(tmp + ((fx * num_frags_y + fy) * grid.size() +
									 ((kv_head_idx * num_chunks + iter) * gridDim.x + bx) *
										 block.num_threads() +
									 block.thread_rank()) *
										8);
#pragma unroll
				for(uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
					o_frag[fx][fy][reg_id] += o_frag_i[reg_id] * other_scale[fx][(reg_id % 4) / 2];
				}
			}
		}
	}
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeOut>
__device__ __forceinline__ void write_o_reg_gmem(float (*o_frag)[num_frags_y][8],
												 smem_t* o_smem,
												 DTypeOut* o_ptr_base,
												 uint32_t o_idx_base,
												 const uint32_t qo_upper_bound,
												 const uint32_t qo_n_stride,
												 const uint32_t qo_h_stride) {
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();
	const uint32_t tx = threadIdx.x, ty = threadIdx.y;

#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t fy = 0; fy < num_frags_y; ++fy) {
			uint32_t o_frag_f16[4];
			vec_cast<DTypeOut, float, 8>((DTypeOut*)o_frag_f16, o_frag[fx][fy]);
			uint32_t o_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_out>(
				(ty * num_frags_x + fx) * 16 + tx / 4, fy * 2);
			((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
			((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * channel_size_128b_out))[tx % 4] =
				o_frag_f16[1];
			((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] = o_frag_f16[2];
			((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
						 8 * channel_size_128b_out))[tx % 4] = o_frag_f16[3];
		}
	}

	uint32_t o_smem_offset_w =
		smem_t::get_permuted_offset<channel_size_128b_out>(ty * num_frags_x * 16 + tx / 8, tx % 8);

	o_idx_base += (tx / 8) / group_size;
	o_ptr_base += ((tx / 8) / group_size) * qo_n_stride + ((tx / 8) % group_size) * qo_h_stride;
#pragma unroll
	for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
		for(uint32_t j = 0; j < 4; ++j) {
			const uint32_t o_idx = o_idx_base + (fx * 16 + j * 4) / group_size;
			DTypeOut* o_ptr = o_ptr_base + ((fx * 16 + j * 4) / group_size) * qo_n_stride +
							  ((fx * 16 + j * 4) % group_size) * qo_h_stride;
#pragma unroll
			for(uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
				if(o_idx < qo_upper_bound) {
					o_smem->store_128b(o_smem_offset_w, o_ptr);
				}
				o_ptr += 8 * num_elems_per_128b<DTypeOut>();
				o_smem_offset_w = o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
			}
			o_smem_offset_w =
				o_smem->advance_offset_by_row<4, channel_size_128b_out>(o_smem_offset_w) -
				2 * num_frags_y;
		}
	}
}

} // namespace

template <uint32_t group_size,
		  uint32_t page_size,
		  bool causal,
		  RotaryMode rotary_mode,
		  uint32_t num_frags_x,
		  uint32_t num_frags_y,
		  uint32_t num_frags_z,
		  uint32_t num_warps,
		  PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeQKAccum,
		  typename DTypeOut,
		  typename IdType>
__global__ void
BatchPrefillWithPagedKVCacheKernel(IdType* __restrict__ request_indices,
								   IdType* __restrict__ tile_indices,
								   DTypeIn* __restrict__ q,
								   paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
								   IdType* __restrict__ qo_indptr,
								   DTypeOut* __restrict__ o,
								   float* __restrict__ tmp,
								   float* __restrict__ lse,
								   float sm_scale,
								   const float log2_rope_rcp_scale,
								   const float log2_rope_rcp_theta) {
	static_assert(sizeof(DTypeIn) == 2);
	static_assert(sizeof(DTypeOut) == 2);
	sm_scale *= math::log2e;
	auto block = cg::this_thread_block();

	const uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y, kv_head_idx = blockIdx.z;
	const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
	const uint32_t request_idx = request_indices[bx], tile_idx = tile_indices[bx];
	constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
	const uint32_t qo_len = qo_indptr[request_idx + 1] - qo_indptr[request_idx],
				   kv_len = (paged_kv.indptr[request_idx + 1] - paged_kv.indptr[request_idx] - 1) *
								paged_kv.page_size +
							paged_kv.last_page_len;
	const uint32_t qo_upper_bound = min(qo_len, (tile_idx + 1) * (num_rows_per_cta / group_size));

	constexpr bool partition_kv = false;
	constexpr uint32_t head_dim = num_frags_y * 16;
	constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
	constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

	static_assert(num_frags_z * num_frags_y % num_warps == 0);
	static_assert(group_size == 1 || group_size % 4 == 0);

	extern __shared__ uint8_t smem[];

	DTypeQKAccum s_frag[num_frags_x][num_frags_z][8];
	float o_frag[num_frags_x][num_frags_y][8];
	DTypeQKAccum m[num_frags_x][2];
	float d[num_frags_x][2];

	init_states<num_frags_x, num_frags_y>(o_frag, m, d);

	const uint32_t qo_idx_base = ((tile_idx * num_warps + ty) * num_frags_x * 16) / group_size;
	const uint32_t qo_n_stride = get_n_stride_impl<QKVLayout::kNHD, head_dim>(num_qo_heads),
				   qo_h_stride = get_h_stride_impl<QKVLayout::kNHD, head_dim>(qo_len);
	smem_t qo_smem(smem);
	DTypeIn* q_ptr_base = q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
								  qo_indptr[request_idx] + qo_idx_base,
								  kv_head_idx * group_size,
								  (tx % 8) * num_elems_per_128b<DTypeIn>(),
								  qo_len,
								  num_qo_heads);
	DTypeIn* o_ptr_base = o + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
								  qo_indptr[request_idx] + qo_idx_base,
								  kv_head_idx * group_size,
								  (tx % 8) * num_elems_per_128b<DTypeOut>(),
								  qo_len,
								  num_qo_heads);
	uint32_t q_smem_offset_r =
		smem_t::get_permuted_offset<channel_size_128b_in>(ty * num_frags_x * 16 + tx % 16, tx / 16);

	load_q_global_smem<group_size, num_frags_x, num_frags_y>(
		qo_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride, &qo_smem);

	cp_async::commit_group();
	cp_async::wait_group<0>();
	block.sync();

	q_smem_inplace_multiply_sm_scale<num_frags_x, num_frags_y, DTypeIn>(&qo_smem, sm_scale);

	smem_t k_smem(smem + (num_warps * num_frags_x) * 16 * head_dim * sizeof(DTypeIn)),
		v_smem(smem + (num_warps * num_frags_x + num_frags_z) * 16 * head_dim * sizeof(DTypeIn));

	uint32_t k_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
				 8 * (tx / 16) + tx % 8, (tx % 16) / 8),
			 v_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(tx % 16, tx / 16),
			 kv_smem_offset_w =
				 smem_t::get_permuted_offset<channel_size_128b_in>(ty * 4 + tx / 8, tx % 8);
	const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

	uint32_t page_iter_base = paged_kv.indptr[request_idx];
	page_produce_kv<false, page_size, num_warps, num_frags_y, num_frags_z>(
		k_smem, &kv_smem_offset_w, paged_kv, 0, page_iter_base, kv_len, last_indptr);
	cp_async::commit_group();
	page_produce_kv<true, page_size, num_warps, num_frags_y, num_frags_z>(
		v_smem, &kv_smem_offset_w, paged_kv, 0, page_iter_base, kv_len, last_indptr);
	cp_async::commit_group();

	const uint32_t num_iterations = ceil_div(
		(causal
			 ? min(kv_len,
				   kv_len - qo_len + ((tile_idx + 1) * num_frags_x * num_warps * 16) / group_size)
			 : kv_len),
		16 * num_frags_z);

	const uint32_t mask_iteration =
		(causal
			 ? min(kv_len + (tile_idx * num_warps * num_frags_x * 16) / group_size - qo_len, kv_len)
			 : kv_len) /
		(16 * num_frags_z);

#pragma unroll
	for(uint32_t iter = 0; iter < num_iterations; ++iter) {
		cp_async::wait_group<1>();
		block.sync();

		// compute attention score
		compute_qk<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(
			&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

		// apply mask
		if(iter >= mask_iteration) {
			mask_s<partition_kv,
				   causal,
				   group_size,
				   num_warps,
				   num_frags_x,
				   num_frags_y,
				   num_frags_z>(
				qo_idx_base, iter * 16 * num_frags_z, qo_len, kv_len, kv_len, s_frag);
		}

		// compute m,d states in online softmax
		update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

		block.sync();
		page_iter_base += 16 * num_frags_z / page_size;
		page_produce_kv<false, page_size, num_warps, num_frags_y, num_frags_z>(k_smem,
																			   &kv_smem_offset_w,
																			   paged_kv,
																			   (iter + 1) * 16 *
																				   num_frags_z,
																			   page_iter_base,
																			   kv_len,
																			   last_indptr);
		cp_async::commit_group();
		cp_async::wait_group<1>();
		block.sync();

		// compute sfm*v
		compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(
			&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

		block.sync();
		page_produce_kv<true, page_size, num_warps, num_frags_y, num_frags_z>(v_smem,
																			  &kv_smem_offset_w,
																			  paged_kv,
																			  (iter + 1) * 16 *
																				  num_frags_z,
																			  page_iter_base,
																			  kv_len,
																			  last_indptr);
		cp_async::commit_group();
	}
	cp_async::wait_group<0>();
	block.sync();

	// normalize d
	normalize_d<num_frags_x, num_frags_y>(o_frag, d);

	// write_back
	write_o_reg_gmem<group_size, num_frags_x, num_frags_y>(
		o_frag, &qo_smem, o_ptr_base, qo_idx_base, qo_len, qo_n_stride, qo_h_stride);

	// write lse
	if(lse != nullptr) {
#pragma unroll
		for(uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
			for(uint32_t j = 0; j < 2; ++j) {
				const uint32_t qo_head_idx =
					kv_head_idx * group_size + (tx / 4 + j * 8 + fx * 16) % group_size;
				const uint32_t qo_idx = qo_idx_base + (tx / 4 + j * 8 + fx * 16) / group_size;
				if(qo_idx < qo_upper_bound) {
					lse[(qo_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
						math::ptx_log2(d[fx][j]) + float(m[fx][j]);
				}
			}
		}
	}
}

template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  uint32_t num_frags_x,
		  uint32_t PAGE_SIZE,
		  uint32_t GROUP_SIZE,
		  uint32_t HEAD_DIM,
		  RotaryMode ROTARY_MODE,
		  bool ALLOW_FP16_QK_REDUCTION,
		  bool CAUSAL,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
	DTypeIn* q,
	IdType* request_indices,
	IdType* tile_indices,
	IdType* qo_indptr,
	paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
	DTypeOut* o,
	float* tmp,
	float* lse,
	uint32_t num_qo_tiles,
	float rope_scale,
	float rope_theta,
	cudaStream_t stream) {
	const float sm_scale = 1.f / std::sqrt(float(paged_kv.head_dim));
	const float log2_rope_rcp_scale = -std::log2f(rope_scale);
	const float log2_rope_rcp_theta = -std::log2f(rope_theta);
	constexpr uint32_t num_warps = 4;
	const uint32_t num_kv_heads = paged_kv.num_heads;
	const uint32_t batch_size = paged_kv.batch_size;

	dim3 nblks(num_qo_tiles, 1, num_kv_heads);
	dim3 nthrs(32, num_warps);

	constexpr uint32_t num_frags_y = HEAD_DIM / 16;
	using DTypeQKAccum =
		typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeIn, half>::value,
								  half,
								  float>::type;

	int dev_id = 0;
	FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
	int max_smem_per_sm = 0;
	FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
		&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
	// we expect each sm execute two threadblocks
	const int max_smem_per_threadblock = max_smem_per_sm / 2;

	const uint32_t max_num_frags_z_reg =
		(HEAD_DIM == 128 && num_frags_x == 2 && ROTARY_MODE == RotaryMode::kLlama &&
		 !ALLOW_FP16_QK_REDUCTION)
			? 2
			: 4;
	const uint32_t max_num_frags_z_smem =
		(max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeIn)) - num_frags_x * num_warps) /
		2;

	SWITCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
		auto kernel = BatchPrefillWithPagedKVCacheKernel<GROUP_SIZE,
														 PAGE_SIZE,
														 CAUSAL,
														 ROTARY_MODE,
														 num_frags_x,
														 num_frags_y,
														 num_frags_z,
														 num_warps,
														 page_storage,
														 kv_layout,
														 DTypeIn,
														 DTypeQKAccum,
														 DTypeOut,
														 IdType>;
		uint32_t smem_size =
			(num_frags_x * num_warps + num_frags_z * 2) * 16 * HEAD_DIM * sizeof(DTypeIn);
		FLASHINFER_CUDA_CALL(
			cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
		void* args[] = {(void*)&request_indices,
						(void*)&tile_indices,
						(void*)&q,
						(void*)&paged_kv,
						(void*)&qo_indptr,
						(void*)&o,
						(void*)&tmp,
						(void*)&lse,
						(void*)&sm_scale,
						(void*)&log2_rope_rcp_scale,
						(void*)&log2_rope_rcp_theta};
		FLASHINFER_CUDA_CALL(
			cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
	});
	return cudaSuccess;
}

template <typename IdType>
std::tuple<IdType, IdType, std::vector<IdType>, std::vector<IdType>>
split_qo_indptr(IdType* qo_indptr,
				uint32_t batch_size,
				uint32_t gqa_group_size,
				cudaStream_t stream = nullptr) {
	constexpr uint32_t num_warps = 4;
	std::vector<IdType> qo_indptr_h(batch_size + 1), request_indices, tile_indices;
	if(is_device_ptr((void*)qo_indptr)) {
		cudaMemcpyAsync(qo_indptr_h.data(),
						qo_indptr,
						sizeof(IdType) * (batch_size + 1),
						cudaMemcpyDeviceToHost,
						stream);
	} else {
		qo_indptr_h.assign(qo_indptr, qo_indptr + batch_size + 1);
	}

	const uint32_t total_q_len = qo_indptr_h[batch_size];
	const bool avg_len_greater_than_64 = total_q_len * gqa_group_size > 64 * batch_size;
	const uint32_t num_frags_x = avg_len_greater_than_64 ? 2 : 1;
	const uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
	uint32_t num_qo_tiles = 0;

	for(uint32_t i = 0; i < batch_size; ++i) {
		for(uint32_t j = qo_indptr_h[i] * gqa_group_size; j < qo_indptr_h[i + 1] * gqa_group_size;
			j += num_rows_per_cta) {
			request_indices.push_back(i);
			tile_indices.push_back((j - qo_indptr_h[i] * gqa_group_size) / num_rows_per_cta);
			++num_qo_tiles;
		}
	}

	return {num_frags_x, num_qo_tiles, std::move(request_indices), std::move(tile_indices)};
}

template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
BatchPrefillWithPagedKVCache(DTypeIn* q,
							 IdType* qo_indptr,
							 paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
							 DTypeOut* o,
							 float* tmp,
							 float* lse,
							 uint32_t num_qo_heads,
							 bool causal = true,
							 RotaryMode rotary_mode = RotaryMode::kNone,
							 bool allow_fp16_qk_reduction = false,
							 float rope_scale = 1.f,
							 float rope_theta = 1e4,
							 cudaStream_t stream = nullptr) {
	const uint32_t num_kv_heads = paged_kv.num_heads;
	const uint32_t head_dim = paged_kv.head_dim;
	const uint32_t batch_size = paged_kv.batch_size;
	const uint32_t group_size = num_qo_heads / num_kv_heads;

	if(rotary_mode != RotaryMode::kNone) {
		std::ostringstream err_msg;
		err_msg << "Rotary mode is not supported yet.";
		throw std::invalid_argument(err_msg.str());
	}

	if(!(paged_kv.page_size == 1 || paged_kv.page_size == 8 || paged_kv.page_size == 16 || paged_kv.page_size == 32)) {
		std::ostringstream err_msg;
		err_msg << "Page size must be 1, 8, 16 or 32.";
		throw std::invalid_argument(err_msg.str());
	}

	uint32_t num_frags_x, num_qo_tiles;
	std::vector<IdType> request_indices_h, tile_indices_h;
	std::tie(num_frags_x, num_qo_tiles, request_indices_h, tile_indices_h) =
		split_qo_indptr(qo_indptr, batch_size, group_size, stream);

	IdType* request_indices_d;
	IdType* tile_indices_d;

	FLASHINFER_CUDA_CALL(
		cudaMallocAsync(&request_indices_d, sizeof(IdType) * request_indices_h.size(), stream));
	FLASHINFER_CUDA_CALL(
		cudaMallocAsync(&tile_indices_d, sizeof(IdType) * tile_indices_h.size(), stream));
	FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_d,
										 request_indices_h.data(),
										 sizeof(IdType) * request_indices_h.size(),
										 cudaMemcpyHostToDevice,
										 stream));
	FLASHINFER_CUDA_CALL(cudaMemcpyAsync(tile_indices_d,
										 tile_indices_h.data(),
										 sizeof(IdType) * tile_indices_h.size(),
										 cudaMemcpyHostToDevice,
										 stream));

	SWITCH_NUM_FRAGS_X(
		num_frags_x,
		NUM_FRAGS_X,
		{SWITCH_ALLOW_FP16_QK_REDUCTION(
			allow_fp16_qk_reduction,
			ALLOW_FP16_QK_REDUCTION,
			{SWITCH_GQA_GROUP_SIZE(
				group_size,
				GROUP_SIZE,
				{SWITCH_CAUSAL(
					causal,
					CAUSAL,
					{SWITCH_HEAD_DIM_PREFILL(
						head_dim,
						HEAD_DIM,
						{SWITCH_ROTARY_MODE(
							rotary_mode,
							ROTARY_MODE,
							{SWITCH_PAGE_SIZE(paged_kv.page_size,
											  PAGE_SIZE,
											  {
												  if constexpr(PAGE_SIZE != 0) {
													  return BatchPrefillWithPagedKVCacheDispatched<
														  page_storage,
														  kv_layout,
														  NUM_FRAGS_X,
														  PAGE_SIZE,
														  GROUP_SIZE,
														  HEAD_DIM,
														  ROTARY_MODE,
														  ALLOW_FP16_QK_REDUCTION,
														  CAUSAL,
														  DTypeIn,
														  DTypeOut,
														  IdType>(q,
																  request_indices_d,
																  tile_indices_d,
																  qo_indptr,
																  paged_kv,
																  o,
																  tmp,
																  lse,
																  num_qo_tiles,
																  rope_scale,
																  rope_theta,
																  stream);
												  }
											  })

							})})})})})});

	FLASHINFER_CUDA_CALL(cudaFreeAsync(request_indices_d, stream));
	FLASHINFER_CUDA_CALL(cudaFreeAsync(tile_indices_d, stream));
	return cudaSuccess;
}

} // namespace flashinfer

#endif // FLASHINFER_PREFILL_CUH_
