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
      https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh
  Support for Page-Sparsity Self-Attention by dynamic selection.
*/

#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

#include <cuda_fp16.h>
#include <vector>

#include "flashinfer/layout.cuh"
#include "flashinfer/rope.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"

#ifndef CUDART_MAX_NORMAL_FP16
	// Introduced in CUDA 12.3
	#define CUDART_MAX_NORMAL_FP16 (__float2half(65504.0f))
#endif

namespace flashinfer
{

enum class PageStorage
{
	kIndices = 0U, // Store the pointer to the buffer allocated for paged kv-cache, and indices of
	// each active offset.
	kPointer = 1U, // Store the pointers to each active page.
};

/*!
 * \brief The auxiliary information about kv sequence partitioning
 */
template <typename IdType>
struct kv_partition_info_t {
	uint32_t batch_size_before_partition;
	IdType* chunk_indptr;
	IdType* batch_idx_map;

	__host__ __device__ __forceinline__ kv_partition_info_t(uint32_t batch_size_before_partition,
															IdType* chunk_indptr,
															IdType* batch_idx_map,
															IdType* chunk_start_pos)
		: batch_size_before_partition(batch_size_before_partition)
		, chunk_indptr(chunk_indptr)
		, batch_idx_map(batch_idx_map) { }

	__host__ __device__ __forceinline__ kv_partition_info_t()
		: batch_size_before_partition(0)
		, chunk_indptr(nullptr)
		, batch_idx_map(nullptr) { }
};

/*!
 * \brief Paged key-value cache
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimensions in KV-Cache.
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
struct paged_kv_t {
	uint32_t num_heads;
	uint32_t page_size;
	uint32_t head_dim;
	uint32_t batch_size;

	// page budget for inference, equal to nnz_pages.
	// Not contain the tail page (equal to page_budget - 1)
	uint32_t page_budget;
	// The offset of the last page for all heads.
	// single value for we manually align all heads with the last page.
	uint32_t last_page_len;
	// page offset for the last page (shared by all heads)
	IdType last_page_idx;

	// The flattened key-value cache, used when page_storage == kIndices
	// Internal layout:
	// [max_num_pages, 2, num_heads, page_size, head_dim] if layout == HND
	// [max_num_pages, 2, page_size, num_heads, head_dim] if layout == NHD
	DType* data;
	// [num_heads, page_budget(nnz_pages)] The page indices array, used when page_storage == kIndices
	// TODO: Currently GROUP_SIZE == 1. The num_heads are not equal actually
	// num_heads dimension is added for dynamic selection which is individual for heads.
	IdType* indices;
	// [nnz_pages] The page pointers array, used when page_storage == kPointer
	// Deprecated in current implemenation of Dynamic Selection.
	DType** ptrs;

	// [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
	// Squeeze the num_heads dimension since it's identical for batches.
	IdType* indptr;

	/*!
   * \brief Construct an empty paged key-value cache
   */
	__host__ __device__ __forceinline__ paged_kv_t()
		: num_heads(0)
		, page_size(0)
		, head_dim(0)
		, batch_size(0)
		, page_budget(0)
		, last_page_len(0)
		, last_page_idx(0)
		, data(nullptr)
		, indices(nullptr)
		, ptrs(nullptr)
		, indptr(nullptr) { }

	/*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param data The flattened key-value cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \note This constructor should only be used when page_storage == kIndices
   */
	__host__ __device__ __forceinline__ paged_kv_t(uint32_t num_heads,
												   uint32_t page_size,
												   uint32_t head_dim,
												   uint32_t batch_size,
												   uint32_t page_budget,
												   uint32_t last_page_len,
												   IdType last_page_idx,
												   DType* data,
												   IdType* indices,
												   IdType* indptr)
		: num_heads(num_heads)
		, page_size(page_size)
		, head_dim(head_dim)
		, batch_size(batch_size)
		, page_budget(page_budget)
		, last_page_len(last_page_len)
		, last_page_idx(last_page_idx)
		, data(data)
		, indices(indices)
		, indptr(indptr) { }

	/*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param ptrs The array of pointers to each active page
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \note This constructor should only be used when page_storage == kIndices
   */
	__host__ __device__ __forceinline__ paged_kv_t(uint32_t num_heads,
												   uint32_t page_size,
												   uint32_t head_dim,
												   uint32_t batch_size,
												   uint32_t page_budget,
												   DType** ptrs,
												   IdType* indptr,
												   IdType* last_page_len)
		: num_heads(num_heads)
		, page_size(page_size)
		, head_dim(head_dim)
		, batch_size(batch_size)
		, page_budget(page_budget)
		, ptrs(ptrs)
		, indptr(indptr) { }

	/*!
   * \brief Compute the offset of k element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   * \note This function should only be used when page_storage == kIndices
   */
	__host__ __device__ __forceinline__ size_t get_k_elem_offset(size_t page_idx,
																 size_t head_idx,
																 size_t entry_idx,
																 size_t feat_idx) const {
		return layout == QKVLayout::kHND
				   ? ((page_idx * 2 * num_heads + head_idx) * page_size + entry_idx) * head_dim +
						 feat_idx
				   : ((page_idx * 2 * page_size + entry_idx) * num_heads + head_idx) * head_dim +
						 feat_idx;
	}

	/*!
   * \brief Compute the offset of k element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
	__host__ __device__ __forceinline__ size_t get_k_elem_offset_in_page(size_t head_idx,
																		 size_t entry_idx,
																		 size_t feat_idx) const {
		return layout == QKVLayout::kHND ? (head_idx * page_size + entry_idx) * head_dim + feat_idx
										 : (entry_idx * num_heads + head_idx) * head_dim + feat_idx;
	}

	/*!
   * \brief Compute the offset of v element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   * \note This function should only be used when page_storage == kIndices
   */
	__host__ __device__ __forceinline__ size_t get_v_elem_offset(size_t page_idx,
																 size_t head_idx,
																 size_t entry_idx,
																 size_t feat_idx) const {
		return layout == QKVLayout::kHND
				   ? (((page_idx * 2 + 1) * num_heads + head_idx) * page_size + entry_idx) *
							 head_dim +
						 feat_idx
				   : (((page_idx * 2 + 1) * page_size + entry_idx) * num_heads + head_idx) *
							 head_dim +
						 feat_idx;
	}

	/*!
   * \brief Compute the offset of v element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
	__host__ __device__ __forceinline__ size_t get_v_elem_offset_in_page(size_t head_idx,
																		 size_t entry_idx,
																		 size_t feat_idx) const {
		return layout == QKVLayout::kHND
				   ? ((num_heads + head_idx) * page_size + entry_idx) * head_dim + feat_idx
				   : ((page_size + entry_idx) * num_heads + head_idx) * head_dim + feat_idx;
	}

	__host__ __device__ __forceinline__ uint32_t kv_offset_delta() const {
		return num_heads * page_size * head_dim;
	}

	__device__ __forceinline__ DType*
	get_k_ptr(IdType page_iter, uint32_t head_idx, uint32_t entry_idx, uint32_t feat_idx) const {
		if constexpr(page_storage == PageStorage::kIndices) {
			return data +
				   get_k_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
		} else {
			return __ldg(ptrs + page_iter) +
				   get_k_elem_offset_in_page(head_idx, entry_idx, feat_idx);
		}
	}

	__device__ __forceinline__ DType* protective_get_k_ptr(IdType page_iter,
														   uint32_t head_idx,
														   uint32_t entry_idx,
														   uint32_t feat_idx,
														   IdType last_indptr) const {
		if constexpr(page_storage == PageStorage::kIndices) {
			if(page_iter < last_indptr) {
				return data +
					   get_k_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
			} else {
				return data;
			}
		} else {
			if(page_iter < last_indptr) {
				return __ldg(ptrs + page_iter) +
					   get_k_elem_offset_in_page(head_idx, entry_idx, feat_idx);
			} else {
				return __ldg(ptrs);
			}
		}
	}

	__device__ __forceinline__ DType*
	get_v_ptr(IdType page_iter, uint32_t head_idx, uint32_t entry_idx, uint32_t feat_idx) const {
		if constexpr(page_storage == PageStorage::kIndices) {
			return data +
				   get_v_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
		} else {
			return __ldg(ptrs + page_iter) +
				   get_v_elem_offset_in_page(head_idx, entry_idx, feat_idx);
		}
	}

	__device__ __forceinline__ DType* protective_get_v_ptr(IdType page_iter,
														   uint32_t head_idx,
														   uint32_t entry_idx,
														   uint32_t feat_idx,
														   IdType last_indptr) const {
		if constexpr(page_storage == PageStorage::kIndices) {
			if(page_iter < last_indptr) {
				return data +
					   get_v_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
			} else {
				return data;
			}
		} else {
			if(page_iter < last_indptr) {
				return __ldg(ptrs + page_iter) +
					   get_v_elem_offset_in_page(head_idx, entry_idx, feat_idx);
			} else {
				return __ldg(ptrs);
			}
		}
	}

	__device__ __forceinline__ DType* protective_get_k_ptr_heads(IdType page_iter,
																 uint32_t head_idx,
																 uint32_t entry_idx,
																 uint32_t feat_idx,
																 IdType last_indptr) const {
		if constexpr(page_storage == PageStorage::kIndices) {
			if(blockIdx.x == gridDim.x - 1) {
				// This is manully appended last page. Only one page here
				return data + get_k_elem_offset(last_page_idx, head_idx, entry_idx, feat_idx);
			} else {
				// Note (Yilong):
				// indices: [num_kv_heads, page_budget - 1]. Manully exclude the last page for sake of Top-k.
				// Therefore, boundary check is last_indptr - 1 (since last_indptr is the last page index).
				if(page_iter < last_indptr - 1) {
					return data +
						   get_k_elem_offset(__ldg(indices + page_iter + head_idx * page_budget),
											 head_idx,
											 entry_idx,
											 feat_idx);
				} else {
					return data;
				}
			}
		} else {
			return nullptr; // "Not implemented for PageStorage::kPointer");
		}
	}
};

template <typename DType, size_t vec_size, bool getMax>
__device__ __forceinline__ void vec_reduct(vec_t<DType, vec_size>& tgt_vec,
										   vec_t<DType, vec_size>& src_vec) {
#pragma unroll
	for(size_t idx = 0; idx < vec_size; ++idx) {
		if constexpr(getMax) {
			tgt_vec[idx] = max(tgt_vec[idx], src_vec[idx]);
		} else {
			tgt_vec[idx] = min(tgt_vec[idx], src_vec[idx]);
		}
	}
}

template <size_t vec_size, bool getMax>
__device__ __forceinline__ void vec_reduct(vec_t<half, vec_size>& tgt_vec,
										   vec_t<half, vec_size>& src_vec) {
#pragma unroll
	for(size_t idx = 0; idx < vec_size; ++idx) {
		if constexpr(getMax) {
			tgt_vec[idx] = __hmax(tgt_vec[idx], src_vec[idx]);
		} else {
			tgt_vec[idx] = __hmin(tgt_vec[idx], src_vec[idx]);
		}
	}
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 */
template <uint32_t head_dim,
		  uint32_t vec_size,
		  PageStorage page_storage,
		  QKVLayout layout,
		  typename DType,
		  typename IdType>
__global__ void
AppendPagedKVCacheDecodeKernel(paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
							   paged_kv_t<page_storage, layout, DType, IdType> candidate_kv,
							   DType* __restrict__ key,
							   DType* __restrict__ value) {
	uint32_t tx = threadIdx.x, ty = threadIdx.y;
	uint32_t num_heads = blockDim.y;
	uint32_t batch_idx = blockIdx.x;
	uint32_t head_idx = ty;

	uint32_t seq_len =
		(paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
		paged_kv.last_page_len;

	// seq_len - 1 is existing tokens
	uint32_t page_iter = paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size;
	uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;

	// calculate the metadata position
	// directly append to the last page. Note that it need be maintained by high-level stack
	// Not using page_iter since it is index instead of indptr
	uint32_t candidate_ptr = candidate_kv.indptr[batch_idx + 1] - 1;
	uint32_t candidate_entry_idx = candidate_kv.last_page_len - 1;

	vec_t<DType, vec_size> local_max, local_min;
	// We do not want initialize before kernel.
	if(entry_idx > 0) {
		local_max.cast_load(
			candidate_kv.get_k_ptr(candidate_ptr, head_idx, candidate_entry_idx, tx * vec_size));
		local_min.cast_load(
			candidate_kv.get_v_ptr(candidate_ptr, head_idx, candidate_entry_idx, tx * vec_size));
	} else {
		local_max.fill(-CUDART_MAX_NORMAL_FP16);
		local_min.fill(CUDART_MAX_NORMAL_FP16);
	}

	// load the append value
	vec_t<DType, vec_size> local_k;
	DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
	DType* v_ptr = k_ptr + paged_kv.kv_offset_delta();
	local_k.cast_load(key + (batch_idx * num_heads + head_idx) * head_dim +
					  tx * vec_size); // Default NHD
	vec_reduct<vec_size, true>(local_max, local_k);
	vec_reduct<vec_size, false>(local_min, local_k);
	local_k.cast_store(k_ptr);
	local_max.cast_store(
		candidate_kv.get_k_ptr(candidate_ptr, head_idx, candidate_entry_idx, tx * vec_size));
	local_min.cast_store(
		candidate_kv.get_v_ptr(candidate_ptr, head_idx, candidate_entry_idx, tx * vec_size));
	vec_t<DType, vec_size>::memcpy(
		v_ptr, value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param candidate_kv The metadata used for estimating Critical KV tokens
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 */
template <uint32_t head_dim,
		  uint32_t vec_size,
		  PageStorage page_storage,
		  QKVLayout layout,
		  typename DType,
		  typename IdType>
__global__ void
AppendPagedKVCachePrefillKernel(paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
								paged_kv_t<page_storage, layout, DType, IdType> candidate_kv,
								DType* __restrict__ key,
								DType* __restrict__ value,
								IdType* __restrict__ append_indptr) {
	int32_t num_heads = gridDim.y;
	int32_t batch_idx = blockIdx.x;
	int32_t head_idx = blockIdx.y;
	int32_t tx = threadIdx.x, ty = threadIdx.y;
	// bdy: page stride. bdx will iterate on continous seq dim.
	int32_t bdy = blockDim.y;

	int32_t page_nums = paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx];
	int32_t seq_len = (page_nums - 1) * paged_kv.page_size + paged_kv.last_page_len;
	int32_t append_seq_len = append_indptr[batch_idx + 1] - append_indptr[batch_idx];
	int32_t append_start_seq = seq_len - append_seq_len;
	// Offset: based on the first page in total
	int32_t start_page_offset_bdx = append_start_seq / paged_kv.page_size + ty;

	for(int32_t page_offset_bdx = start_page_offset_bdx; page_offset_bdx < page_nums;
		page_offset_bdx += bdy) {
		// page_offset_bdx is the page offset (start with 0) for the current batch
		// calculate candidate position info
		// candidate_kv shares the same page_size as paged_kv
		int32_t candidate_page_idx = page_offset_bdx / candidate_kv.page_size;
		int32_t candidate_entry_idx = page_offset_bdx % candidate_kv.page_size;

		int32_t page_iter = static_cast<int32_t>(paged_kv.indptr[batch_idx]) + page_offset_bdx;
		int32_t start_entry_idx =
			max(0, append_start_seq - page_offset_bdx * static_cast<int32_t>(paged_kv.page_size));
		int32_t end_entry_idx =
			min(static_cast<int32_t>(paged_kv.page_size),
				seq_len - page_offset_bdx * static_cast<int32_t>(paged_kv.page_size));

		vec_t<DType, vec_size> local_max, local_min;
		// We do not want initialize before kernel.
		// Therefore we give init value if it is a new page.
		if(start_entry_idx > 0) {
			local_max.cast_load(
				candidate_kv.get_k_ptr(candidate_kv.indptr[batch_idx] + candidate_page_idx,
									   head_idx,
									   candidate_entry_idx,
									   tx * vec_size));
			local_min.cast_load(
				candidate_kv.get_v_ptr(candidate_kv.indptr[batch_idx] + candidate_page_idx,
									   head_idx,
									   candidate_entry_idx,
									   tx * vec_size));
		} else {
			local_max.fill(-CUDART_MAX_NORMAL_FP16);
			local_min.fill(CUDART_MAX_NORMAL_FP16);
		}

		for(int32_t entry_idx = start_entry_idx; entry_idx < end_entry_idx; ++entry_idx) {
			DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
			DType* v_ptr = k_ptr + paged_kv.kv_offset_delta();
			// Reduction on K value
			vec_t<DType, vec_size> local_k;
			local_k.cast_load(key +
							  ((append_indptr[batch_idx] + page_offset_bdx * paged_kv.page_size +
								entry_idx - append_start_seq) *
								   num_heads +
							   head_idx) *
								  head_dim +
							  tx * vec_size);
			vec_reduct<vec_size, true>(local_max, local_k);
			vec_reduct<vec_size, false>(local_min, local_k);
			local_k.cast_store(k_ptr);
			// Only memcpy V value
			vec_t<DType, vec_size>::memcpy(
				v_ptr,
				value +
					((append_indptr[batch_idx] + page_offset_bdx * paged_kv.page_size + entry_idx -
					  append_start_seq) *
						 num_heads +
					 head_idx) *
						head_dim +
					tx * vec_size);
		}
		local_max.cast_store(
			candidate_kv.get_k_ptr(candidate_kv.indptr[batch_idx] + candidate_page_idx,
								   head_idx,
								   candidate_entry_idx,
								   tx * vec_size));
		local_min.cast_store(
			candidate_kv.get_v_ptr(candidate_kv.indptr[batch_idx] + candidate_page_idx,
								   head_idx,
								   candidate_entry_idx,
								   tx * vec_size));
	}
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
									 paged_kv_t<page_storage, layout, DType, IdType> candidate_kv,
									 DType* key,
									 DType* value,
									 cudaStream_t stream = nullptr) {
	uint32_t head_dim = paged_kv.head_dim;
	uint32_t batch_size = paged_kv.batch_size;
	uint32_t num_heads = paged_kv.num_heads;
	SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
		constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
		uint32_t bdx = HEAD_DIM / vec_size;
		uint32_t bdy = num_heads;
		dim3 nblks(batch_size);
		dim3 nthrs(bdx, bdy);
		auto kernel =
			AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, page_storage, layout, DType, IdType>;
		void* args[] = {(void*)&paged_kv, (void*)&candidate_kv, (void*)&key, (void*)&value};
		FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
	});
	return cudaSuccess;
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the prefill phase
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
cudaError_t AppendPagedKVCachePrefill(paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
									  paged_kv_t<page_storage, layout, DType, IdType> candidate_kv,
									  DType* key,
									  DType* value,
									  IdType* append_indptr,
									  cudaStream_t stream = nullptr) {
	uint32_t head_dim = paged_kv.head_dim;
	uint32_t batch_size = paged_kv.batch_size;
	uint32_t num_heads = paged_kv.num_heads;
	SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
		constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
		uint32_t bdx = HEAD_DIM / vec_size;
		uint32_t bdy = 512 / bdx; // page stride
		dim3 nblks(batch_size, num_heads);
		dim3 nthrs(bdx, bdy);
		auto kernel = AppendPagedKVCachePrefillKernel<HEAD_DIM,
													  vec_size,
													  page_storage,
													  layout,
													  DType,
													  IdType>;
		void* args[] = {(void*)&paged_kv,
						(void*)&candidate_kv,
						(void*)&key,
						(void*)&value,
						(void*)&append_indptr};
		FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
	});
	return cudaSuccess;
}

template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType>
__global__ void QKApplyRotaryInPlaceKernel(DType* __restrict__ q,
										   DType* __restrict__ k,
										   uint32_t seq_len,
										   uint32_t past_kv_len,
										   uint32_t num_qo_heads,
										   uint32_t num_kv_heads,
										   float rope_rcp_scale,
										   float rope_rcp_theta) {
	uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
	const uint32_t bdy = blockDim.y;
	vec_t<float, vec_size> freq;
#pragma unroll
	for(uint32_t i = 0; i < vec_size; ++i) {
		freq[i] = rope_rcp_scale *
				  __powf(rope_rcp_theta,
						 float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
	}

	if(bx < num_qo_heads) {
		// apply rotary to q
		const uint32_t qo_head_idx = bx % num_qo_heads;
		const uint32_t offset = past_kv_len;
#pragma unroll 2
		for(uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
			vec_t<float, vec_size> q_vec;
			if(i * bdy + ty < seq_len) {
				DType* q_ptr = q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
									   i * bdy + ty, qo_head_idx, 0, seq_len, num_qo_heads);
				q_vec = vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty);
				q_vec.cast_store(q_ptr + tx * vec_size);
			}
		}
	} else {
		// apply rotary to k
		uint32_t kv_head_idx = (bx - num_qo_heads) % num_kv_heads;
		const uint32_t offset = past_kv_len;
#pragma unroll 2
		for(uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
			vec_t<float, vec_size> k_vec;
			if(i * bdy + ty < seq_len) {
				DType* k_ptr = k + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
									   i * bdy + ty, kv_head_idx, 0, seq_len, num_kv_heads);
				k_vec = vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty);
				k_vec.cast_store(k_ptr + tx * vec_size);
			}
		}
	}
}

template <typename DType>
cudaError_t QKApplyRotaryInPlace(DType* __restrict__ q,
								 DType* __restrict__ k,
								 uint32_t seq_len,
								 uint32_t past_kv_len,
								 uint32_t num_qo_heads,
								 uint32_t num_kv_heads,
								 uint32_t head_dim,
								 float rope_scale = 1.f,
								 float rope_theta = 1e4,
								 cudaStream_t stream = nullptr) {
	float rope_rcp_scale = 1.0f / rope_scale;
	float rope_rcp_theta = 1.0f / rope_theta;

	SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
		constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
		constexpr uint32_t bdx = HEAD_DIM / vec_size;
		uint32_t num_threads = std::max(128U, bdx);
		uint32_t bdy = num_threads / bdx;
		dim3 nblks((num_qo_heads + num_kv_heads));
		dim3 nthrs(bdx, bdy);
		auto kernel = QKApplyRotaryInPlaceKernel<HEAD_DIM, vec_size, bdx, DType>;
		void* args[] = {(void*)&q,
						(void*)&k,
						(void*)&seq_len,
						(void*)&past_kv_len,
						(void*)&num_qo_heads,
						(void*)&num_kv_heads,
						(void*)&rope_rcp_scale,
						(void*)&rope_rcp_theta};
		FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
	});

	return cudaSuccess;
}

} // namespace flashinfer

#endif // FLAHSINFER_PAGE_CUH_
