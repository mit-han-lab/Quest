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
#include <gtest/gtest.h>

#include <decode/decode_attn.cuh>
#include <decode/decode_handler.cuh>
#include <type_traits>

#include "cpu_reference.h"
#include "cpu_utils.h"

using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void _TestBatchDecodingKernelCorrectness(size_t seq_len,
										 size_t page_size,
										 size_t page_budget,
										 size_t num_qo_heads,
										 size_t num_kv_heads,
										 size_t head_dim,
										 bool cooperative) {
	constexpr size_t batch_size = 1;
	constexpr flashinfer::RotaryMode rotary_mode = flashinfer::RotaryMode::kNone;
	assert(num_kv_heads == num_qo_heads); // Not Supported by kernel

	std::vector<int32_t> seq_lens(batch_size, static_cast<int32_t>(seq_len));
	std::vector<int32_t> append_indptr{0};
	for(size_t i = 0; i < batch_size; ++i) {
		append_indptr.push_back(append_indptr.back() + seq_lens[i]);
	}
	std::vector<T> q;
	std::vector<T> o_ref;
	std::vector<T> kv_data;
	std::vector<int32_t> kv_indptr{0};
	std::vector<int32_t> kv_indices;
	size_t page_counter = 0;

	// Initialize selected page indices
	size_t num_pages = flashinfer::ceil_div(seq_len, page_size);
	page_budget = std::min(page_budget, num_pages);

	std::vector<int32_t> selected_page_indices_cpu(num_kv_heads * page_budget);
	std::vector<int32_t> selected_page_indices(num_kv_heads *
											   (page_budget - 1)); // Not contain the last page

	for(size_t head = 0; head < num_kv_heads; ++head) {
		std::vector<int32_t> page_indices_head_raw(num_pages - 1);
		std::iota(page_indices_head_raw.begin(), page_indices_head_raw.end(), 0);
		std::shuffle(page_indices_head_raw.begin(),
					 page_indices_head_raw.end(),
					 std::mt19937(std::random_device()()));

		std::copy(page_indices_head_raw.begin(),
				  page_indices_head_raw.begin() + page_budget - 1,
				  selected_page_indices_cpu.begin() + head * page_budget);
		std::copy(page_indices_head_raw.begin(),
				  page_indices_head_raw.begin() + page_budget - 1,
				  selected_page_indices.begin() + head * (page_budget - 1));
		// The last page is always selected
		selected_page_indices_cpu[head * page_budget + page_budget - 1] = num_pages - 1;
	}
	size_t last_page_len = (seq_len - 1) % page_size + 1;
	int32_t last_page_idx = num_pages - 1;
	std::vector<int32_t> selected_page_indptr({0, static_cast<int32_t>(page_budget - 1)});

	std::vector<std::vector<T>> keys, values;
	for(size_t i = 0; i < batch_size; ++i) {
		size_t seq_len = seq_lens[i];
		size_t num_pages = (seq_len + page_size - 1) / page_size;
		std::vector<T> qi(num_qo_heads * head_dim), ki(seq_len * num_kv_heads * head_dim),
			vi(seq_len * num_kv_heads * head_dim);
		utils::vec_normal_(qi);
		utils::vec_normal_(ki);
		utils::vec_normal_(vi);

		// compute reference output
		std::vector<T> o_ref_i = cpu_reference::selected_single_mha<T, T>(qi,
																		  ki,
																		  vi,
																		  selected_page_indices_cpu,
																		  last_page_len,
																		  1,
																		  seq_len,
																		  page_size,
																		  num_qo_heads,
																		  num_kv_heads,
																		  head_dim,
																		  false,
																		  QKVLayout::kNHD,
																		  rotary_mode);
		keys.push_back(ki);
		values.push_back(vi);
		// append new q and o_ref
		q.insert(q.end(), qi.begin(), qi.end());
		o_ref.insert(o_ref.end(), o_ref_i.begin(), o_ref_i.end());
		// append new kv_indptr, kv_indices
		kv_indptr.push_back(kv_indptr.back() + num_pages);
		for(size_t j = 0; j < num_pages; ++j) {
			kv_indices.push_back(page_counter++);
		}
	}
	kv_data.resize(page_counter * 2 * num_kv_heads * page_size * head_dim);
	utils::vec_zero_(kv_data);
	assert(q.size() == batch_size * num_qo_heads * head_dim);
	assert(o_ref.size() == batch_size * num_qo_heads * head_dim);

	flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_cpu(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		page_budget - 1,
		last_page_len,
		last_page_idx,
		kv_data.data(),
		kv_indices.data(),
		kv_indptr.data());
	cpu_reference::append_paged_kv_cache<kv_layout, T, int32_t>(
		paged_kv_cpu, keys, values, append_indptr);

	// copy data to device
	thrust::device_vector<T> kv_data_device(kv_data);
	thrust::device_vector<int32_t> kv_indptr_device(selected_page_indptr);
	thrust::device_vector<int32_t> kv_indices_device(selected_page_indices);
	thrust::device_vector<T> q_device(q);
	thrust::device_vector<T> o_device(o_ref.size());

	// create paged_kv object
	flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		page_budget - 1,
		last_page_len,
		last_page_idx,
		thrust::raw_pointer_cast(kv_data_device.data()),
		thrust::raw_pointer_cast(kv_indices_device.data()),
		thrust::raw_pointer_cast(kv_indptr_device.data()));
	flashinfer::BatchDecodeHandler handler;
	handler.BeginForward<PageStorage::kIndices, kv_layout, T, T, int32_t>(
		selected_page_indptr.data(),
		batch_size,
		num_qo_heads,
		num_kv_heads,
		head_dim,
		page_size,
		rotary_mode);

	if(!cooperative) {
		// use non-cooperative kernel
		cudaError_t status =
			flashinfer::BatchDecodeWithPagedKVCache<PageStorage::kIndices, kv_layout, T, T>(
				thrust::raw_pointer_cast(q_device.data()),
				paged_kv,
				kv_partition_info_t<int32_t>(),
				thrust::raw_pointer_cast(o_device.data()),
				/*tmp=*/nullptr,
				/*lse=*/nullptr,
				num_qo_heads,
				rotary_mode);
		EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
	} else {
		cudaError_t status =
			flashinfer::BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices, kv_layout, T, T>(
				&handler,
				thrust::raw_pointer_cast(q_device.data()),
				paged_kv,
				thrust::raw_pointer_cast(o_device.data()),
				/*lse=*/nullptr,
				num_qo_heads,
				rotary_mode);
		EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
	}
	// compare result
	thrust::host_vector<T> o_host = o_device;
	size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
	bool nan_detected = false;
	for(size_t i = 0; i < batch_size * num_qo_heads * head_dim; ++i) {
		if(std::isnan(float(o_host[i]))) {
			nan_detected = true;
		}
		num_result_errors_atol_1e_3_rtol_1e_3 +=
			(!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
	}
	float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
									 float(batch_size * num_qo_heads * head_dim);
	std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
			  << ", num_kv_heads=" << num_kv_heads << ", batch_size=" << batch_size
			  << ", seq_len=" << seq_len << ", page_budget=" << page_budget
			  << ", head_dim=" << head_dim
			  << ", rotary_mode=" << flashinfer::RotaryModeToString(rotary_mode)
			  << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;
	EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
	EXPECT_EQ(nan_detected, false) << "NaN detected.";
}

template <typename T>
void TestBatchDecodeKernelCorrectness() {
	for(size_t seq_len : {63, 127, 213, 1110, 2000, 4099, 8192, 8222, 12345, 28837}) {
		for(size_t page_size : {1, 3, 7, 16, 32}) {
			for(size_t page_budget : {16, 32, 64, 128}) {
				for(size_t num_qo_heads : {32}) {
					for(size_t num_kv_heads : {32}) {
						for(size_t head_dim : {64, 128}) {
							_TestBatchDecodingKernelCorrectness<T>(seq_len,
																   page_size,
																   page_budget,
																   num_qo_heads,
																   num_kv_heads,
																   head_dim,
																   false);
						}
					}
				}
			}
		}
	}
}

template <typename T>
void TestCooperativeBatchDecodeKernelCorrectness() {
	for(size_t seq_len : {63, 127, 213, 1110, 2000, 4099, 8192, 8222, 12345, 28837}) {
		for(size_t page_size : {1, 3, 7, 16, 32}) {
			for(size_t page_budget : {16, 32, 64, 128}) {
				for(size_t num_qo_heads : {32}) {
					for(size_t num_kv_heads : {32}) {
						for(size_t head_dim : {64, 128}) {
							_TestBatchDecodingKernelCorrectness<T>(seq_len,
																   page_size,
																   page_budget,
																   num_qo_heads,
																   num_kv_heads,
																   head_dim,
																   true);
						}
					}
				}
			}
		}
	}
}

// Since we manually set one batch for last page, there is no non-cooperative kernel.

// TEST(FlashInferCorrectnessTest, BatchDecodeKernelCorrectnessTestFP16) {
//   TestBatchDecodeKernelCorrectness<half>();
// }

TEST(FlashInferCorrectnessTest, TestCooperativeBatchDecodeKernelCorrectnessTestFP16) {
	TestCooperativeBatchDecodeKernelCorrectness<half>();
}
