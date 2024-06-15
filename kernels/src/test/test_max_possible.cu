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
void _TestMaxPossibleKernelCorrectness(size_t seq_len,
									   size_t page_size,
									   size_t num_qo_heads,
									   size_t num_kv_heads,
									   size_t head_dim) {
	constexpr size_t batch_size = 1;
	constexpr flashinfer::RotaryMode rotary_mode = flashinfer::RotaryMode::kNone;
	assert(num_kv_heads == num_qo_heads); // Not Supported by kernel

	size_t num_pages = flashinfer::ceil_div(seq_len, page_size);
	size_t num_chunks = flashinfer::ceil_div(num_pages, page_size);
	size_t last_chunk_len = (num_pages - 1) % page_size + 1;
	size_t output_len = num_pages - 1; // Manually avoid computation of the last elements

	std::vector<int32_t> chunk_indice_host(num_chunks);
	std::iota(chunk_indice_host.begin(), chunk_indice_host.end(), 0);
	std::shuffle(chunk_indice_host.begin(), chunk_indice_host.end(), std::mt19937(0));
	size_t last_chunk_idx = chunk_indice_host.back();
	std::vector<int32_t> chunk_indptr_host({0, static_cast<int32_t>(num_chunks)});

	// Prepare chunk data
	std::vector<T> chunk_data_host(num_chunks * 2 * page_size * head_dim * num_kv_heads);
	utils::vec_normal_(chunk_data_host);
	std::vector<T> q(batch_size * head_dim * num_qo_heads);
	utils::vec_normal_(q);
	std::vector<T> o(num_qo_heads * output_len);
	utils::vec_zero_(o);

	// Copy to device vector
	thrust::device_vector<T> chunk_data_device(chunk_data_host);
	thrust::device_vector<int32_t> chunk_indptr_device(chunk_indptr_host);
	thrust::device_vector<int32_t> chunk_indice_device(chunk_indice_host);
	thrust::device_vector<T> q_device(q);
	thrust::device_vector<T> o_device(o);

	// CPU-Reference
	flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> chunk_kv_cpu(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0,
		last_chunk_len,
		last_chunk_idx,
		chunk_data_host.data(),
		chunk_indice_host.data(),
		chunk_indptr_host.data());

	std::vector<float> o_ref(output_len * num_qo_heads);
	utils::vec_zero_(o_ref);

	for(size_t head_idx = 0; head_idx < num_qo_heads; ++head_idx) {
		for(size_t page_indptr = chunk_kv_cpu.indptr[batch_size - 1];
			page_indptr < chunk_kv_cpu.indptr[batch_size];
			++page_indptr) {
			size_t cur_page_len = (page_indptr == chunk_kv_cpu.indptr[batch_size] - 1)
									  ? (chunk_kv_cpu.last_page_len - 1) // Manually avoid computation of last elements
									  : page_size;
			for(size_t entry_idx = 0; entry_idx < cur_page_len; ++entry_idx) {
				size_t seq_idx =
					(page_indptr - chunk_kv_cpu.indptr[batch_size - 1]) * page_size + entry_idx;
				assert(seq_idx < output_len);
				for(size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
					float possible_max =
						((float)chunk_data_host[chunk_kv_cpu.get_k_elem_offset(
							chunk_kv_cpu.indices[page_indptr], head_idx, entry_idx, feat_idx)]) *
						((float)q[head_idx * head_dim + feat_idx]);
					float possible_min =
						((float)chunk_data_host[chunk_kv_cpu.get_v_elem_offset(
							chunk_kv_cpu.indices[page_indptr], head_idx, entry_idx, feat_idx)]) *
						((float)q[head_idx * head_dim + feat_idx]);

					o_ref[head_idx * output_len + seq_idx] +=
						possible_max > possible_min ? possible_max : possible_min;
				}
			}
		}
	}
	// GPU reference
	flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> chunk_kv_gpu(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0,
		last_chunk_len,
		last_chunk_idx,
		thrust::raw_pointer_cast(chunk_data_device.data()),
		thrust::raw_pointer_cast(chunk_indice_device.data()),
		thrust::raw_pointer_cast(chunk_indptr_device.data()));

	// use non-cooperative kernel
	cudaError_t status = flashinfer::
		MaxPossibleSampleWithPagedKVCache<PageStorage::kIndices, kv_layout, T, T, int32_t>(
			thrust::raw_pointer_cast(q_device.data()),
			chunk_kv_gpu,
			thrust::raw_pointer_cast(o_device.data()),
			num_qo_heads,
			rotary_mode);
	EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
	// compare result
	thrust::host_vector<T> o_host = o_device;
	size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
	bool nan_detected = false;
	for(size_t i = 0; i < o_host.size(); ++i) {
		if(std::isnan(float(o_host[i]))) {
			nan_detected = true;
		}
		num_result_errors_atol_1e_3_rtol_1e_3 +=
			(!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
		if(!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3)) {
			std::cout << "i=" << i << ", o_host=" << (float)o_host[i] << ", o=" << (float)o_ref[i]
					  << std::endl;
		}
	}
	float result_accuracy =
		1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_qo_heads * output_len);
	std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
			  << ", num_kv_heads=" << num_kv_heads << ", batch_size=" << batch_size
			  << ", seq_len=" << seq_len << ", head_dim=" << head_dim
			  << ", rotary_mode=" << flashinfer::RotaryModeToString(rotary_mode)
			  << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;
	EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
	EXPECT_EQ(nan_detected, false) << "NaN detected.";
}

template <typename T>
void TestMaxPossibleKernelCorrectness() {
	for(size_t seq_len : {65, 127, 213, 1110, 2000, 4099, 8192, 8222, 12345, 28837}) {
		for(size_t page_size : {1, 3, 7, 16, 32}) {
			for(size_t num_qo_heads : {32}) {
				for(size_t num_kv_heads : {32}) {
					for(size_t head_dim : {64, 128}) {
						_TestMaxPossibleKernelCorrectness<T>(
							seq_len, page_size, num_qo_heads, num_kv_heads, head_dim);
					}
				}
			}
		}
	}
}

TEST(FlashInferCorrectnessTest, TestMaxPossibleSampleKernelCorrectnessTestFP16) {
	TestMaxPossibleKernelCorrectness<half>();
}
