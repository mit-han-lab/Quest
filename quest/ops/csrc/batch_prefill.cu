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
  Modified from FlashInfer project.
  Check: https://github.com/flashinfer-ai/flashinfer/blob/main/python/csrc/batch_prefill.cu
*/

#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor prefill_with_paged_kv_cache(torch::Tensor q,
										  torch::Tensor kv_data,
										  torch::Tensor kv_indices,
										  unsigned int kv_last_page_len,
										  bool causal,
										  unsigned int layout,
										  bool allow_fp16_qk_reduction,
										  float rope_scale,
										  float rope_theta) {
	constexpr size_t batch_size = 1;

	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(q); // [sum(extend_len), num_qo_heads, head_dim]
	// [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
	// [max_num_pages, 2, page_size, num_kv_heads, head_dim] for HND
	CHECK_INPUT(kv_data);
	CHECK_INPUT(kv_indices); // [sum(seq_len)]
	#endif

	// bsk only utilizes flashinfer for bsz=1. Therefore we can infer some parameters.
	torch::Tensor q_indptr = torch::tensor({0, static_cast<int32_t>(q.size(0))}, kv_indices.options());
	torch::Tensor kv_indptr = torch::tensor({0, static_cast<int32_t>(kv_indices.size(0))}, kv_indices.options());

	#ifdef BSK_TORCH_CHECK
	CHECK_DIM(3, q);
	CHECK_DIM(5, kv_data)
	CHECK_DIM(1, q_indptr);
	CHECK_DIM(1, kv_indptr);
	CHECK_DIM(1, kv_indices);
	CHECK_EQ(q_indptr.size(0), kv_indptr.size(0));
	CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(q.size(2), kv_data.size(4));
	#endif

	QKVLayout kv_layout = QKVLayout(layout);
	unsigned int page_size, num_kv_heads;
	if(kv_layout == QKVLayout::kHND) {
		num_kv_heads = kv_data.size(2);
		page_size = kv_data.size(3);
	} else {
		page_size = kv_data.size(2);
		num_kv_heads = kv_data.size(3);
	}
	unsigned int head_dim = q.size(2);
	unsigned int num_qo_heads = q.size(1);

	auto o = torch::empty_like(q, q.options());

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_kv_heads,
				page_size,
				head_dim,
				batch_size,
				0,
				kv_last_page_len,
				kv_indices[-1].item<int32_t>(),
				static_cast<c_type*>(kv_data.data_ptr()),
				static_cast<int32_t*>(kv_indices.data_ptr()),
				static_cast<int32_t*>(kv_indptr.data_ptr()));

			cudaError_t status =
				BatchPrefillWithPagedKVCache<PageStorage::kIndices,
											 KV_LAYOUT,
											 c_type,
											 c_type,
											 int32_t>(static_cast<c_type*>(q.data_ptr()),
													  static_cast<int32_t*>(q_indptr.data_ptr()),
													  paged_kv,
													  static_cast<c_type*>(o.data_ptr()),
													  /*tmp=*/nullptr,
													  /*lse=*/nullptr,
													  num_qo_heads,
													  causal,
													  RotaryMode::kNone,
													  allow_fp16_qk_reduction,
													  rope_scale,
													  rope_theta);
			TORCH_CHECK(status == cudaSuccess,
						"BatchPrefillWithPagedKVCache failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});

	TORCH_CHECK(
		success, "BatchPrefillWithPagedKVCache failed to dispatch with dtype ", q.scalar_type());

	return o;
}