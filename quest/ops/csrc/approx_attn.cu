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
  Modified from FlashInfer PyTorch API.
  Check: https://github.com/flashinfer-ai/flashinfer/blob/main/python/csrc/batch_decode.cu
*/

#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward(torch::Tensor indptr,
															 unsigned int num_qo_heads,
															 unsigned int num_kv_heads,
															 unsigned int head_dim,
															 unsigned int page_size,
															 torch::Tensor empty_data) {
	constexpr size_t batch_size = 1;

	#ifdef BSK_TORCH_CHECK
	CHECK_CONTIGUOUS(indptr);
	CHECK_DIM(1, indptr);
	CHECK_EQ(indptr.scalar_type(), torch::kInt32);
	#endif

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(empty_data.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout_, KV_LAYOUT, {
			cudaError_t status =
				handler_.BeginForward<PageStorage::kIndices, KV_LAYOUT, c_type, c_type, int32_t>(
					static_cast<int32_t*>(indptr.data_ptr()),
					batch_size,
					num_qo_heads,
					num_kv_heads,
					head_dim,
					page_size,
					RotaryMode::kNone);
			TORCH_CHECK(status == cudaSuccess,
						"BatchDecodeWithPagedKVCache failed with error ",
						cudaGetErrorString(status));
			return true;
		})
	});

	TORCH_CHECK(success,
				"BatchDecodeWithPagedKVCache failed to dispatch with dtype ",
				empty_data.scalar_type());
}

void BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward() {
	handler_.EndForward();
}

void
BatchDecodeWithPagedKVCachePyTorchWrapper::Forward(torch::Tensor q,
												   torch::Tensor o,
												   torch::Tensor paged_kv_data,
												   torch::Tensor paged_kv_indices,
												   torch::Tensor paged_kv_indptr, // [1, Page_budget]
												   unsigned int paged_kv_last_page_len,
												   unsigned int paged_kv_last_page_idx,
												   float rope_scale,
												   float rope_theta) {
	constexpr size_t batch_size = 1;
	
	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(q);
	CHECK_INPUT(paged_kv_data);
	CHECK_INPUT(paged_kv_indices);
	CHECK_DIM(3, q); // (B, H_qo, D)
	CHECK_DIM(2, paged_kv_indices); // (num_heads, page_budget - 1)
	// (num_max_pages, 2, H_kv, page_size, head_dim) for HND
	// (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
	CHECK_DIM(5, paged_kv_data);
	#endif

	int64_t num_qo_heads = q.size(1);
	int64_t head_dim = q.size(2);
	int64_t num_kv_heads, page_size;
	// This is the stride of the paged_kv_indices tensor
	// actual page budget is page_budget + 1
	int64_t page_budget = paged_kv_indices.size(1);

	if(kv_layout_ == QKVLayout::kHND) {
		num_kv_heads = paged_kv_data.size(2);
		page_size = paged_kv_data.size(3);
	} else {
		page_size = paged_kv_data.size(2);
		num_kv_heads = paged_kv_data.size(3);
	}

	#ifdef BSK_TORCH_CHECK
	CHECK_EQ(paged_kv_indices.size(0), num_qo_heads);
	CHECK_EQ(paged_kv_data.size(1), 2);
	CHECK_EQ(paged_kv_data.size(4), head_dim);
	CHECK_EQ(paged_kv_indices.scalar_type(), torch::kInt32);
	#endif

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout_, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_kv_heads,
				page_size,
				head_dim,
				batch_size,
				page_budget,
				paged_kv_last_page_len,
				paged_kv_last_page_idx,
				static_cast<c_type*>(paged_kv_data.data_ptr()),
				static_cast<int32_t*>(paged_kv_indices.data_ptr()),
				static_cast<int32_t*>(paged_kv_indptr.data_ptr()));

			cudaError_t status =
				BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices,
												   KV_LAYOUT,
												   c_type,
												   c_type,
												   int32_t>(&handler_,
															static_cast<c_type*>(q.data_ptr()),
															paged_kv,
															static_cast<c_type*>(o.data_ptr()),
															/*lse=*/nullptr,
															num_qo_heads,
															RotaryMode::kNone,
															rope_scale,
															rope_theta,
															/*stream=*/nullptr);
			TORCH_CHECK(status == cudaSuccess,
						"BatchDecodeWithPagedKVCache failed with error ",
						cudaGetErrorString(status));
		});
		return true;
	});

	TORCH_CHECK(
		success, "BatchDecodeWithPagedKVCache failed to dispatch with dtype ", q.scalar_type());
}