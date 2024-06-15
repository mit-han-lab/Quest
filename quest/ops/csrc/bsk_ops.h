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
#pragma once
#include <torch/extension.h>

#include "decode/decode_handler.cuh"
#include "prefill/prefill.cuh"
#include "topk/decode_select_k.cuh"

void apply_rope_in_place(torch::Tensor q,
						 torch::Tensor k,
						 unsigned int past_kv_len,
						 float rope_scale,
						 float rope_theta);

void rms_norm_forward(torch::Tensor input,
					  torch::Tensor weight,
					  torch::Tensor output,
					  float epsilon);

void topk_filtering(torch::Tensor estimated_value,
					torch::Tensor estimated_indices,
					torch::Tensor d_out,
					torch::Tensor indices_out,
					torch::Tensor buf,
					unsigned int page_budget);

void estimate_attn_score(torch::Tensor q,
						 torch::Tensor o,
						 torch::Tensor metadata_data,
						 torch::Tensor metadata_indices,
						 torch::Tensor metadata_indptr,
						 unsigned int metadata_last_page_len,
						 unsigned int metadata_last_page_idx,
						 unsigned int layout);

void append_kv_cache_prefill(torch::Tensor k,
							 torch::Tensor v,
							 torch::Tensor kv_data,
							 torch::Tensor kv_indices,
							 torch::Tensor kv_indptr,
							 unsigned int kv_last_page_len,
							 unsigned int kv_last_page_idx,
							 torch::Tensor metadata_data,
							 torch::Tensor metadata_indices,
							 torch::Tensor metadata_indptr,
							 unsigned int metadata_last_page_len,
							 unsigned int metadata_last_page_idx,
							 unsigned int layout);

void append_kv_cache_decode(torch::Tensor k,
							torch::Tensor v,
							torch::Tensor kv_data,
							torch::Tensor kv_indices,
							torch::Tensor kv_indptr,
							unsigned int kv_last_page_len,
							unsigned int kv_last_page_idx,
							torch::Tensor metadata_data,
							torch::Tensor metadata_indices,
							torch::Tensor metadata_indptr,
							unsigned int metadata_last_page_len,
							unsigned int metadata_last_page_idx,
							unsigned int layout);

torch::Tensor prefill_with_paged_kv_cache(torch::Tensor q,
										  torch::Tensor kv_data,
										  torch::Tensor kv_indices,
										  unsigned int kv_last_page_len,
										  bool causal,
										  unsigned int layout,
										  bool allow_fp16_qk_reduction,
										  float rope_scale,
										  float rope_theta);

class BatchDecodeWithPagedKVCachePyTorchWrapper {
public:
	static BatchDecodeWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
		return BatchDecodeWithPagedKVCachePyTorchWrapper(layout);
	}
	void BeginForward(torch::Tensor indptr,
					  unsigned int num_qo_heads,
					  unsigned int num_kv_heads,
					  unsigned int head_dim,
					  unsigned int page_size,
					  torch::Tensor empty_data);

	void EndForward();

	void Forward(torch::Tensor q,
				 torch::Tensor o,
				 torch::Tensor paged_kv_data,
				 torch::Tensor paged_kv_indices,
				 torch::Tensor paged_kv_indptr,
				 unsigned int paged_kv_last_page_len,
				 unsigned int paged_kv_last_page_idx,
				 float rope_scale,
				 float rope_theta);

private:
	BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
		: kv_layout_(flashinfer::QKVLayout(layout)) { }
	flashinfer::BatchDecodeHandler handler_;
	flashinfer::QKVLayout kv_layout_;
};