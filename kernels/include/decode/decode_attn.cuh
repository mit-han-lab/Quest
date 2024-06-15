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
      https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/decode.cuh
  Support for Page-Sparsity Self-Attention by dynamic selection.
*/

#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_ENABLE_FP8
#	include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "flashinfer/cascade.cuh"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/layout.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/rope.cuh"
#include "flashinfer/state.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"

#include "decode/decode_page.cuh"

namespace flashinfer
{

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace
{

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam rotary_mode The rotary mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <RotaryMode rotary_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void compute_qk(const T* smem,
										   uint32_t compute_stage_idx,
										   const vec_t<float, vec_size>& q_vec,
										   const vec_t<float, vec_size>& freq,
										   uint32_t iter_base,
										   uint32_t iter_bound,
										   float sm_scale,
										   float* s,
										   state_t<vec_size>& st) {
	uint32_t tx = threadIdx.x, tz = threadIdx.z;
	float m_prev = st.m;
#pragma unroll
	for(uint32_t j = 0; j < tile_size; ++j) {
		vec_t<float, vec_size> k_vec;
		if constexpr(rotary_mode == RotaryMode::kNone) {
			// do not apply rotary embedding
			k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
		}
		s[j] = 0.f;
#pragma unroll
		for(uint32_t i = 0; i < vec_size; ++i) {
			s[j] += q_vec[i] * k_vec[i] * sm_scale;
		}
#pragma unroll
		for(uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
			s[j] += math::shfl_xor_sync(s[j], offset);
		}
		s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -5e4;
		st.m = max(st.m, s[j]);
	}

	float o_scale = math::ptx_exp2(m_prev - st.m);
	st.d *= o_scale;
#pragma unroll
	for(uint32_t j = 0; j < tile_size; ++j) {
		s[j] = math::ptx_exp2(s[j] - st.m);
		st.d += s[j];
	}
#pragma unroll
	for(uint32_t i = 0; i < vec_size; ++i) {
		st.o[i] = st.o[i] * o_scale;
	}
}

/*!
 * \brief Load (max, min) tile of K from smem and compute accumulated max qk
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param max_smem A pointer to the start of max smem, bind to bdz
 * \param min_smem A pointer to the start of min smem, bind to bdz
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param kv_idx_base A integer indicates the block-local kv position in kv-cache
 * \param iter_base A integer indicates the block-local iteration index
 * \param iter_bound A integer indicates the block-local iteration bound
 * \param o A pointer to the start of global memory within this specific head
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename DTypeIn, typename DTypeOut>
__device__ __forceinline__ void compute_max_possible(const DTypeIn* max_smem,
													 const DTypeIn* min_smem,
													 const vec_t<float, vec_size>& q_vec,
													 uint32_t kv_idx_base,
													 uint32_t iter_base,
													 uint32_t iter_bound,
													 DTypeOut* o) {
	uint32_t tx = threadIdx.x, tz = threadIdx.z;
#pragma unroll
	for(uint32_t j = 0; j < tile_size; ++j) {
		vec_t<float, vec_size> max_vec;
		vec_t<float, vec_size> min_vec;
		max_vec.cast_load(max_smem + (j * bdx + tx) * vec_size);
		min_vec.cast_load(min_smem + (j * bdx + tx) * vec_size);
		float max_possible = 0.f;
#pragma unroll
		for(uint32_t i = 0; i < vec_size; ++i) {
			max_possible += max(q_vec[i] * max_vec[i], q_vec[i] * min_vec[i]);
		}
#pragma unroll
		for(uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
			max_possible += math::shfl_xor_sync(max_possible, offset);
		}
		// Store out
		if(iter_base + tz * tile_size + j < iter_bound) {
			if(tx == 0) {
				o[kv_idx_base + tz * tile_size + j] = static_cast<DTypeOut>(max_possible);
			}
		}
	}
}

/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem,
												   const float* s,
												   uint32_t compute_stage_idx,
												   state_t<vec_size>& st) {
	uint32_t tx = threadIdx.x;
#pragma unroll
	for(uint32_t j = 0; j < tile_size; ++j) {
		vec_t<float, vec_size> v_vec;
		v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
		for(uint32_t i = 0; i < vec_size; ++i) {
			st.o[i] = st.o[i] + s[j] * v_vec[i];
		}
	}
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz>
__device__ __forceinline__ void sync_state(state_t<vec_size>& st, float* smem, float* smem_md) {
	if constexpr(bdz > 1) {
		constexpr uint32_t head_dim = bdx * vec_size;
		auto block = cg::this_thread_block();
		uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
		st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
		smem_md[(tz * bdy + ty) * 2] = st.m;
		smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
		block.sync();
		st.init();
#pragma unroll
		for(uint32_t j = 0; j < bdz; ++j) {
			float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
			vec_t<float, vec_size> oz;
			oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
			st.merge(oz, mz, dz);
		}
	}
}

} // namespace

template <bool partition_kv,
		  RotaryMode rotary_mode,
		  uint32_t num_stages_smem,
		  uint32_t tile_size_per_bdx,
		  uint32_t vec_size,
		  uint32_t bdx,
		  uint32_t bdy,
		  uint32_t bdz,
		  PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
__global__ void MaxPossibleSampleWithPagedKVCacheKernel(
	DTypeIn* __restrict__ q,
	paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
	DTypeOut* __restrict__ o) {

	static_assert(partition_kv == false && rotary_mode == RotaryMode::kNone);
	auto block = cg::this_thread_block();

	constexpr uint32_t head_dim = bdx * vec_size;
	const uint32_t batch_idx = blockIdx.x;
	const uint32_t kv_head_idx = blockIdx.y;
	const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
	const uint32_t num_qo_heads = gridDim.y * bdy;
	const uint32_t cur_chunk_start = 0U; // Not partition-kv now.

	const uint32_t cur_page_indptr_begin = paged_kv.indptr[batch_idx],
				   cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];

	// paged_kv.last_page_len - 1: hard-code for not considering the last entry
	// which is the last page of original kv-cache.
	// Note that last_page_len should be \in [1, PAGE_SIZE]
	const uint32_t cur_last_page_len =
		(batch_idx == gridDim.x - 1) ? (paged_kv.last_page_len - 1) : paged_kv.page_size;
	const uint32_t kv_chunk_len =
		cur_page_indptr_begin != cur_page_indptr_end
			? (cur_page_indptr_end - cur_page_indptr_begin - 1) * paged_kv.page_size +
				  cur_last_page_len
			: 0;
	extern __shared__ uint8_t smem[];

	// Maximum value and Minimum share the same layout as k and v
	// Therefore we denote them by K and V
	DTypeIn* k_smem = (DTypeIn*)smem;
	DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
											sizeof(DTypeIn));
	DTypeIn** k_ptrs_smem = (DTypeIn**)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
												   head_dim * sizeof(DTypeIn));

	const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

	// No in-kernel RoPE
	// No kv partition
	vec_t<float, vec_size> q_vec;
	q_vec.cast_load(q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
	block.sync();

	// preload k/v tiles
	uint32_t stage_idx = 0;
	constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
	const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

	static_assert(num_stages_smem <= bdx);
#pragma unroll
	for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
		k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] = paged_kv.protective_get_k_ptr(
			cur_page_indptr_begin + (((j * bdz + tz) * bdy + ty) * bdx + tx) / paged_kv.page_size,
			kv_head_idx,
			(((j * bdz + tz) * bdy + ty) * bdx + tx) % paged_kv.page_size,
			0,
			last_indptr);
	}
	block.sync();

	DTypeIn* k_ptrs[tile_size_per_bdx];
#pragma unroll
	for(uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			k_ptrs[j] =
				k_ptrs_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
		}
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
				k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				k_ptrs[j],
				((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
		}
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
				v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				v_ptr,
				((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
		}
		cp_async::commit_group();
		stage_idx = (stage_idx + 1) % num_stages_smem;
	}

#pragma unroll 2
	for(uint32_t iter = 0; iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
		if((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
			for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
				k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] = paged_kv.protective_get_k_ptr(
					cur_page_indptr_begin +
						((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
						 ((j * bdz + tz) * bdy + ty) * bdx + tx) /
							paged_kv.page_size,
					kv_head_idx,
					((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
					 ((j * bdz + tz) * bdy + ty) * bdx + tx) %
						paged_kv.page_size,
					0,
					last_indptr);
			}
		}
		// compute qk
		cp_async::wait_group<num_stages_smem - 1>();
		block.sync();
		compute_max_possible<vec_size, bdx, bdy * tile_size_per_bdx>(
			k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
			v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
			q_vec,
			cur_chunk_start + iter * tile_size_per_bdx * bdy * bdz,
			iter * tile_size_per_bdx * bdy * bdz,
			kv_chunk_len,
			o + (num_qo_heads * batch_idx + qo_head_idx) *
					kv_chunk_len); // Note that o is [num_heads, kv_chunk_len]. Exclude last page.
		block.sync();

#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			k_ptrs[j] = k_ptrs_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
										tile_size_per_bdx +
									j] +
						tx * vec_size;
		}
		// load k tiles
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
				k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				k_ptrs[j],
				(((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
					kv_chunk_len);
		}
		// load v tiles
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
				v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				v_ptr,
				(((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
					kv_chunk_len);
		}
		cp_async::commit_group();
		stage_idx = (stage_idx + 1) % num_stages_smem;
	}
	cp_async::wait_group<0>();
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam partition_kv Whether to partition kv-cache on sequence length dimension or not
 * \tparam rotary_mode The rotary mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <bool partition_kv,
		  RotaryMode rotary_mode,
		  uint32_t num_stages_smem,
		  uint32_t tile_size_per_bdx,
		  uint32_t vec_size,
		  uint32_t bdx,
		  uint32_t bdy,
		  uint32_t bdz,
		  PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
__global__ void
BatchDecodeWithPagedKVCacheKernel(DTypeIn* __restrict__ q,
								  paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
								  kv_partition_info_t<IdType> kv_partition_info,
								  DTypeOut* __restrict__ o,
								  DTypeOut* __restrict__ tmp,
								  float* __restrict__ lse,
								  float sm_scale,
								  float rope_rcp_scale,
								  float rope_rcp_theta) {
	auto block = cg::this_thread_block();
	sm_scale *= math::log2e;

	constexpr uint32_t head_dim = bdx * vec_size;
	const uint32_t batch_idx = blockIdx.x;
	const uint32_t kv_head_idx =
		blockIdx.y; // Use kv_head_idx as another dimention in page indices.
	const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
	const uint32_t num_qo_heads = gridDim.y * bdy;
	// Number of pages is identical since all heads share same latency budget.
	const uint32_t cur_page_indptr_begin = paged_kv.indptr[batch_idx],
				   cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];
	// The last_page_len should be [batch_idx, num_kv_heads]
	// However, currently target on bsz = 1, so we use [num_kv_heads] only.
	// Besides, we manually append the last page. Therefore we use [1] only.
	// We filter other cases directly into PAGE_SIZE and only select batch_idx == gridDim.x - 1.
	// This is correct since all batches are transformed from the single batch.
	// TODO: support bsz > 1.
	const uint32_t cur_last_page_len =
		(batch_idx == gridDim.x - 1) ? paged_kv.last_page_len : paged_kv.page_size;
	const uint32_t kv_chunk_len =
		cur_page_indptr_begin != cur_page_indptr_end
			? (cur_page_indptr_end - cur_page_indptr_begin - 1) * paged_kv.page_size +
				  cur_last_page_len
			: 0;
	extern __shared__ uint8_t smem[];
	DTypeIn* k_smem = (DTypeIn*)smem;
	DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
											sizeof(DTypeIn));
	DTypeIn** k_ptrs_smem = (DTypeIn**)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
												   head_dim * sizeof(DTypeIn));
	float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
										 head_dim * sizeof(DTypeIn));

	const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
	vec_t<float, vec_size> q_vec;
	vec_t<float, vec_size> freq;
	if constexpr(rotary_mode == RotaryMode::kLlama) {
		// "RoPE fused in kernel is not supported yet."
	} else {
		// do not apply rotary embedding to q matrix
		if constexpr(partition_kv) {
			q_vec.cast_load(
				q +
				(kv_partition_info.batch_idx_map[batch_idx] * num_qo_heads + qo_head_idx) *
					head_dim +
				tx * vec_size);
		} else {
			q_vec.cast_load(q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim +
							tx * vec_size);
		}
	}
	block.sync();

	// preload k/v tiles
	uint32_t stage_idx = 0;
	constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
	const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

	static_assert(num_stages_smem <= bdx);
#pragma unroll
	for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
		k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] = paged_kv.protective_get_k_ptr_heads(
			cur_page_indptr_begin + (((j * bdz + tz) * bdy + ty) * bdx + tx) / paged_kv.page_size,
			kv_head_idx,
			(((j * bdz + tz) * bdy + ty) * bdx + tx) % paged_kv.page_size,
			0,
			last_indptr);
	}
	block.sync();

	DTypeIn* k_ptrs[tile_size_per_bdx];
#pragma unroll
	for(uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			k_ptrs[j] =
				k_ptrs_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
		}
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
				k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				k_ptrs[j],
				((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
		}
		cp_async::commit_group();
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
				v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				v_ptr,
				((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
		}
		cp_async::commit_group();
		stage_idx = (stage_idx + 1) % num_stages_smem;
	}

	state_t<vec_size> st;
	float s[bdy * tile_size_per_bdx];

#pragma unroll 2
	for(uint32_t iter = 0; iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
		if((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
			for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
				k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
					paged_kv.protective_get_k_ptr_heads(
						cur_page_indptr_begin +
							((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
							 ((j * bdz + tz) * bdy + ty) * bdx + tx) /
								paged_kv.page_size,
						kv_head_idx,
						((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
						 ((j * bdz + tz) * bdy + ty) * bdx + tx) %
							paged_kv.page_size,
						0,
						last_indptr);
			}
		}
		// compute qk
		cp_async::wait_group<2 * num_stages_smem - 1>();
		block.sync();
		compute_qk<rotary_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
			k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
			stage_idx,
			q_vec,
			freq,
			iter * tile_size_per_bdx * bdy * bdz,
			kv_chunk_len,
			sm_scale,
			s,
			st);
		block.sync();

#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			k_ptrs[j] = k_ptrs_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
										tile_size_per_bdx +
									j] +
						tx * vec_size;
		}
		// load k tiles
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
				k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				k_ptrs[j],
				(((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
					kv_chunk_len);
		}
		cp_async::commit_group();

		// update m/d/o states
		cp_async::wait_group<2 * num_stages_smem - 1>();
		block.sync();
		update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
			v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st);
		block.sync();

		// load v tiles
#pragma unroll
		for(uint32_t j = 0; j < tile_size_per_bdx; ++j) {
			DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
			cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
				v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
					tx * vec_size,
				v_ptr,
				(((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
					kv_chunk_len);
		}
		cp_async::commit_group();
		stage_idx = (stage_idx + 1) % num_stages_smem;
	}
	cp_async::wait_group<0>();
	block.sync();

	// sync local state of all warps inside a threadblock
	sync_state<vec_size, bdx, bdy, bdz>(st, reinterpret_cast<float*>(smem), smem_md);
	st.normalize();

	if constexpr(partition_kv) {
		st.o.cast_store(tmp + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
		float* tmp_lse = (float*)(tmp + paged_kv.batch_size * num_qo_heads * head_dim);
		tmp_lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
	} else {
		st.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
		// write lse
		if(lse != nullptr) {
			lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
		}
	}
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
	if(group_size == 8U) {
		if(sizeof_dtype == 1U) {
			return 256U; // not enough registers for 512 threads
		} else {
			return 512U;
		}
	} else {
		return 128U;
	}
}

/*!
 * \brief Partition Paged KV-Cache into multiple chunks on KV sequence length
 * \tparam IdType A template type indicates the index data type
 * \param old_batch_size The batch size of the old Paged KV-Cache
 * \param old_page_indptr_h The host-side page indptr of the old Paged KV-Cache
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_paged_kv_d The device-side new Paged KV-Cache
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename IdType>
cudaError_t PartitionPagedKVCacheComputeAuxiliaryInfo(const uint32_t max_num_pages_per_batch,
													  const uint32_t old_batch_size,
													  const uint32_t page_size,
													  IdType* old_indptr,
													  IdType* new_indptr_d,
													  IdType* chunk_indptr_d,
													  IdType* batch_idx_map_d,
													  cudaStream_t stream = nullptr) {
	std::vector<IdType> new_page_indptr_h{0}, chunk_indptr_h{0}, batch_idx_map_h;
	std::vector<IdType> old_indptr_h(old_batch_size + 1);
	if(is_device_ptr(old_indptr)) {
		FLASHINFER_CUDA_CALL(cudaMemcpyAsync(old_indptr_h.data(),
											 old_indptr,
											 sizeof(IdType) * (old_batch_size + 1),
											 cudaMemcpyDeviceToHost,
											 stream));
		FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
	} else {
		old_indptr_h.assign(old_indptr, old_indptr + old_batch_size + 1);
	}

	for(uint32_t batch_idx = 0; batch_idx < old_batch_size; batch_idx++) {
		uint32_t num_chunks = ceil_div(old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx],
									   max_num_pages_per_batch);
		chunk_indptr_h.push_back(chunk_indptr_h.back() + num_chunks);
		if(num_chunks == 0) {
			new_page_indptr_h.push_back(old_indptr_h[batch_idx]);
			batch_idx_map_h.push_back(batch_idx);
		} else {
			for(uint32_t j = 0; j < num_chunks; ++j) {
				new_page_indptr_h.push_back(
					min(old_indptr_h[batch_idx] + (j + 1) * max_num_pages_per_batch,
						old_indptr_h[batch_idx + 1]));
				batch_idx_map_h.push_back(batch_idx);
			}
		}
	}

	// Manually append information of last page
	chunk_indptr_h.back() += 1;
	new_page_indptr_h.push_back(new_page_indptr_h.back() +
								1); // +1 for consistent kv_chunk_len calculation
	batch_idx_map_h.push_back(0); // 0 since we only support bsz = 1

	FLASHINFER_CUDA_CALL(cudaMemcpyAsync(new_indptr_d,
										 new_page_indptr_h.data(),
										 sizeof(IdType) * new_page_indptr_h.size(),
										 cudaMemcpyHostToDevice,
										 stream));
	FLASHINFER_CUDA_CALL(cudaMemcpyAsync(chunk_indptr_d,
										 chunk_indptr_h.data(),
										 sizeof(IdType) * chunk_indptr_h.size(),
										 cudaMemcpyHostToDevice,
										 stream));
	FLASHINFER_CUDA_CALL(cudaMemcpyAsync(batch_idx_map_d,
										 batch_idx_map_h.data(),
										 sizeof(IdType) * batch_idx_map_h.size(),
										 cudaMemcpyHostToDevice,
										 stream));
	return cudaSuccess;
}

/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 *   after we partition Paged KV-Cache into multiple chunks on KV sequence length
 *   dimension.
 * \tparam IdType A template type indicates the index data type
 * \param max_grid_size The maximum grid size of the kernel
 * \param num_kv_heads The number of KV heads
 * \param num_pages The number of pages per request in the batch
 * \param max_num_pages_per_batch_lb The pre-set lower bound of maximum number of
 *   pages per batch, default to 1
 * \return (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 *   the new batch size after the partition.
 */
template <typename IdType>
std::pair<uint32_t, uint32_t>
PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(const uint32_t max_grid_size,
													const uint32_t num_kv_heads,
													const std::vector<IdType>& num_pages,
													const uint32_t min_num_pages_per_batch = 1) {
	uint32_t low = min_num_pages_per_batch, high = 0;
	for(const IdType& elem : num_pages) {
		high = max(high, elem);
	}
	uint32_t new_batch_size;
	while(low < high) {
		uint32_t mid = (low + high) / 2;
		new_batch_size = 0;
		for(const IdType& elem : num_pages) {
			new_batch_size += ceil_div(elem, mid);
		}
		if(new_batch_size * num_kv_heads > max_grid_size) {
			low = mid + 1;
		} else {
			high = mid;
		}
	}
	new_batch_size = 0;
	for(const IdType& elem : num_pages) {
		new_batch_size += ceil_div(std::max(elem, 1), low);
	}
	return {low, new_batch_size};
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param tmp_size The estimated temporary buffer size, return 0 if not use partition-kv kernel
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param rotary_mode The rotary mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
BatchDecodeWithPagedKVCacheWorkEstimation(uint32_t& tmp_size,
										  uint32_t& max_grid_size,
										  uint32_t& max_num_pages_per_batch,
										  uint32_t& new_batch_size,
										  uint32_t batch_size,
										  IdType* kv_indptr,
										  const uint32_t num_qo_heads,
										  const uint32_t num_kv_heads,
										  const uint32_t head_dim,
										  const uint32_t page_size,
										  const RotaryMode rotary_mode = RotaryMode::kNone,
										  cudaStream_t stream = nullptr) {
	SWITCH_GQA_GROUP_SIZE(
		num_qo_heads / num_kv_heads,
		GROUP_SIZE,
		{SWITCH_HEAD_DIM(
			head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
				constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
				constexpr uint32_t num_stages_smem = 2U;
				constexpr uint32_t bdx = HEAD_DIM / vec_size;
				static_assert(bdx <= 32);
				constexpr uint32_t bdy = GROUP_SIZE;
				constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
				constexpr uint32_t bdz = num_threads / (bdx * bdy);
				constexpr uint32_t tile_size_per_bdx =
					GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
				const uint32_t smem_size =
					2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
						sizeof(DTypeIn) +
					std::max(tile_size_per_bdx * num_threads * sizeof(DTypeIn*),
							 2 * bdy * bdz * sizeof(float));

				auto partition_kv_kernel = BatchDecodeWithPagedKVCacheKernel<
					/*partition_kv=*/true,
					ROTARY_MODE,
					num_stages_smem,
					tile_size_per_bdx,
					vec_size,
					bdx,
					bdy,
					bdz,
					page_storage,
					kv_layout,
					DTypeIn,
					DTypeOut,
					IdType>;
				int num_blocks_per_sm = 0;
				int num_sm = 0;
				int dev_id = 0;
				FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
				FLASHINFER_CUDA_CALL(
					cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
				FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
					&num_blocks_per_sm, partition_kv_kernel, num_threads, smem_size));
				max_grid_size = num_blocks_per_sm * num_sm;
				if(batch_size * num_kv_heads >= max_grid_size) {
					// do not use partition-kv kernel
					tmp_size = 0;
					new_batch_size = batch_size;
				} else {
					// compute max_num_pages_per_batch and new_batch_size
					std::vector<IdType> page_indptr_h(batch_size + 1), num_pages(batch_size);
					if(is_device_ptr(kv_indptr)) {
						FLASHINFER_CUDA_CALL(cudaMemcpy(page_indptr_h.data(),
														kv_indptr,
														sizeof(IdType) * (batch_size + 1),
														cudaMemcpyDeviceToHost));
					} else {
						page_indptr_h.assign(kv_indptr, kv_indptr + batch_size + 1);
					}
					for(uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
						num_pages[batch_idx] =
							page_indptr_h[batch_idx + 1] - page_indptr_h[batch_idx];
					}
					std::tie(max_num_pages_per_batch, new_batch_size) =
						PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
							max_grid_size, num_kv_heads, num_pages, 128 / page_size);
					// Here we manully append the last page to a new batch
					new_batch_size += 1;
					if(new_batch_size == batch_size) {
						// do not use partition-kv kernel for short sequence
						tmp_size = 0;
					} else {
						tmp_size = num_qo_heads * new_batch_size *
								   (head_dim * sizeof(DTypeOut) + 2 * sizeof(float));
					}
				}
			})})});
	return cudaSuccess;
}

template <uint32_t GROUP_SIZE,
		  uint32_t HEAD_DIM,
		  PageStorage page_storage,
		  QKVLayout kv_layout,
		  RotaryMode ROTARY_MODE,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
BatchDecodeWithPagedKVCacheDispatched(DTypeIn* q,
									  paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
									  kv_partition_info_t<IdType> kv_partition_info,
									  DTypeOut* o,
									  DTypeOut* tmp,
									  float* lse,
									  float rope_scale,
									  float rope_theta,
									  cudaStream_t stream) {
	const float sm_scale = 1.f / std::sqrt(float(HEAD_DIM));
	const float rope_rcp_scale = 1.f / rope_scale;
	const float rope_rcp_theta = 1.f / rope_theta;
	const uint32_t num_kv_heads = paged_kv.num_heads;
	const uint32_t batch_size = paged_kv.batch_size;
	const uint32_t num_qo_heads = num_kv_heads * GROUP_SIZE;

	constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
	constexpr uint32_t num_stages_smem = 2U;
	constexpr uint32_t bdx = HEAD_DIM / vec_size;
	static_assert(bdx <= 32);
	constexpr uint32_t bdy = GROUP_SIZE;
	constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
	constexpr uint32_t bdz = num_threads / (bdx * bdy);
	constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
	const uint32_t smem_size =
		2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeIn) +
		std::max(tile_size_per_bdx * num_threads * sizeof(DTypeIn*), 2 * bdy * bdz * sizeof(float));

	if(tmp == nullptr) {
		// do not use partition-kv kernel
		dim3 nblks(batch_size, num_kv_heads);
		dim3 nthrs(bdx, bdy, bdz);
		auto kernel = BatchDecodeWithPagedKVCacheKernel</*partition_kv=*/false,
														ROTARY_MODE,
														num_stages_smem,
														tile_size_per_bdx,
														vec_size,
														bdx,
														bdy,
														bdz,
														page_storage,
														kv_layout,
														DTypeIn,
														DTypeOut,
														IdType>;
		FLASHINFER_CUDA_CALL(
			cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
		void* args[] = {(void*)&q,
						(void*)&paged_kv,
						(void*)&kv_partition_info,
						(void*)&o,
						(void*)&tmp,
						(void*)&lse,
						(void*)&sm_scale,
						(void*)&rope_rcp_scale,
						(void*)&rope_rcp_theta};
		FLASHINFER_CUDA_CALL(
			cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
	} else {
		// use partition-kv kernel
		auto partition_kv_kernel = BatchDecodeWithPagedKVCacheKernel</*partition_kv=*/true,
																	 ROTARY_MODE,
																	 num_stages_smem,
																	 tile_size_per_bdx,
																	 vec_size,
																	 bdx,
																	 bdy,
																	 bdz,
																	 page_storage,
																	 kv_layout,
																	 DTypeIn,
																	 DTypeOut,
																	 IdType>;
		FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
			partition_kv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
		void* args[] = {(void*)&q,
						(void*)&paged_kv,
						(void*)&kv_partition_info,
						(void*)&o,
						(void*)&tmp,
						(void*)&lse,
						(void*)&sm_scale,
						(void*)&rope_rcp_scale,
						(void*)&rope_rcp_theta};
		dim3 nblks(batch_size, num_kv_heads);
		dim3 nthrs(bdx, bdy, bdz);
		FLASHINFER_CUDA_CALL(
			cudaLaunchKernel((void*)partition_kv_kernel, nblks, nthrs, args, smem_size, stream));
		FLASHINFER_CUDA_CALL(
			VariableLengthMergeStates(tmp,
									  (float*)(tmp + batch_size * num_qo_heads * HEAD_DIM),
									  kv_partition_info.chunk_indptr,
									  o,
									  lse,
									  kv_partition_info.batch_size_before_partition,
									  num_qo_heads,
									  HEAD_DIM,
									  stream));
	}

	return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for batched requests
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged kv-cache
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv cache data structure
 * \param o [batch_size, num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values.
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param rotary_mode The rotary mode
 * \param rope_scale The scaling ratio used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
BatchDecodeWithPagedKVCache(DTypeIn* q,
							paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
							kv_partition_info_t<IdType> kv_partition_info,
							DTypeOut* o,
							DTypeOut* tmp,
							float* lse,
							uint32_t num_qo_heads,
							RotaryMode rotary_mode = RotaryMode::kNone,
							float rope_scale = 1.f,
							float rope_theta = 1e4,
							cudaStream_t stream = nullptr) {
	const uint32_t num_kv_heads = paged_kv.num_heads;
	const uint32_t head_dim = paged_kv.head_dim;
	const uint32_t batch_size = paged_kv.batch_size;
	if(num_qo_heads % num_kv_heads != 0) {
		std::ostringstream err_msg;
		err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
				<< num_kv_heads;
		throw std::invalid_argument(err_msg.str());
	}

	SWITCH_GQA_GROUP_SIZE(
		num_qo_heads / num_kv_heads,
		GROUP_SIZE,
		{SWITCH_HEAD_DIM(
			head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
				return BatchDecodeWithPagedKVCacheDispatched<GROUP_SIZE,
															 HEAD_DIM,
															 page_storage,
															 kv_layout,
															 ROTARY_MODE,
															 DTypeIn,
															 DTypeOut,
															 IdType>(
					q, paged_kv, kv_partition_info, o, tmp, lse, rope_scale, rope_theta, stream);
			})})});

	return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for single request
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged kv-cache
 * \param q [num_qo_heads, head_dim] The query matrix, the layout is fixed NHD
 * \param paged_kv The paged kv cache data structure
 * \param o [num_qo_heads, output_len] The output matrix. Note that output_len = elements in page_kv_t - 1.
 * 	We ignore the last element because later we manually append it to the selected topk.
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param rotary_mode The rotary mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
MaxPossibleSampleWithPagedKVCache(DTypeIn* q,
								  paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
								  DTypeOut* o,
								  uint32_t num_qo_heads,
								  RotaryMode rotary_mode = RotaryMode::kNone,
								  cudaStream_t stream = nullptr) {
	const uint32_t num_kv_heads = paged_kv.num_heads;
	const uint32_t head_dim = paged_kv.head_dim;
	const uint32_t batch_size = paged_kv.batch_size;
	if(num_qo_heads % num_kv_heads != 0) {
		std::ostringstream err_msg;
		err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
				<< num_kv_heads;
		throw std::invalid_argument(err_msg.str());
	}
	if(rotary_mode != RotaryMode::kNone) {
		std::ostringstream err_msg;
		err_msg << "Rotary mode is not supported yet.";
		throw std::invalid_argument(err_msg.str());
	}

	SWITCH_GQA_GROUP_SIZE(
		num_qo_heads / num_kv_heads, GROUP_SIZE, {SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
			constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
			constexpr uint32_t num_stages_smem = 2U;
			constexpr uint32_t bdx = HEAD_DIM / vec_size;
			static_assert(bdx <= 32);
			constexpr uint32_t bdy = GROUP_SIZE;
			constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
			constexpr uint32_t bdz = num_threads / (bdx * bdy);
			constexpr uint32_t tile_size_per_bdx =
				GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
			const uint32_t smem_size =
				2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(DTypeIn) +
				tile_size_per_bdx * num_threads * sizeof(DTypeIn*);
			dim3 nblks(batch_size, num_kv_heads);
			dim3 nthrs(bdx, bdy, bdz);
			auto kernel = MaxPossibleSampleWithPagedKVCacheKernel<false,
																  RotaryMode::kNone,
																  num_stages_smem,
																  tile_size_per_bdx,
																  vec_size,
																  bdx,
																  bdy,
																  bdz,
																  page_storage,
																  kv_layout,
																  DTypeIn,
																  DTypeOut,
																  IdType>;
			FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
				kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
			void* args[] = {(void*)&q, (void*)&paged_kv, (void*)&o};
			FLASHINFER_CUDA_CALL(
				cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
		})});
	return cudaSuccess;
}

} // namespace flashinfer

#endif // FLASHINFER_DECODE_CUH_
