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
#include <thrust/device_vector.h>

#include <cstdint>
#include <decode/decode_attn.cuh>
#include <decode/decode_handler.cuh>
#include <nvbench/nvbench.cuh>
#include <vector>

#include "cpu_utils.h"

using utils::vec_bytes;
using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void bench_flashinfer_batch_decode(nvbench::state& state) {
	constexpr size_t head_dim = 128;
	constexpr size_t batch_size = 1;
	constexpr auto rotary_mode = RotaryMode::kNone;
	size_t seqlen = state.get_int64("seqlen");
	size_t page_budget = state.get_int64("page_budget");
	size_t page_size = state.get_int64("page_size");
	size_t num_qo_heads = state.get_int64("num_qo_heads");
	size_t num_kv_heads = state.get_int64("num_kv_heads");
	bool cooperative = state.get_int64("cooperative");

	assert(num_qo_heads == num_kv_heads); // Not support GQA now
	// KV cache:
	size_t num_pages = flashinfer::ceil_div(seqlen, page_size);
	size_t last_page_len = (seqlen - 1) % page_size + 1;
	int32_t last_page_idx = num_pages - 1;
	// adjust page_budget
	page_budget = std::min(page_budget, num_pages);
	std::vector<int32_t> kv_indptr_host(
		{0, static_cast<int32_t>(page_budget - 1)}); // Not Contain the last page
	std::vector<int32_t> kv_indicies_host(num_qo_heads * (page_budget - 1));
	for(size_t head = 0; head < num_qo_heads; ++head) {
		std::vector<int32_t> page_indices_head_raw(num_pages - 1);
		std::iota(page_indices_head_raw.begin(), page_indices_head_raw.end(), 0);
		std::shuffle(page_indices_head_raw.begin(),
					 page_indices_head_raw.end(),
					 std::mt19937(std::random_device()()));

		std::copy(page_indices_head_raw.begin(),
				  page_indices_head_raw.begin() + page_budget - 1,
				  kv_indicies_host.begin() + head * (page_budget - 1));
	}

	thrust::device_vector<T> kv_data(num_pages * 2 * num_kv_heads * page_size * head_dim);
	thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
	thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		page_budget - 1,
		last_page_len,
		last_page_idx,
		thrust::raw_pointer_cast(kv_data.data()),
		thrust::raw_pointer_cast(kv_indices.data()),
		thrust::raw_pointer_cast(kv_indptr.data()));
	// Allocate input data:
	thrust::device_vector<T> q(batch_size * num_qo_heads * head_dim);
	thrust::device_vector<T> o(batch_size * num_qo_heads * head_dim);
	state.add_global_memory_reads<uint8_t>(
		vec_bytes(q) + (page_budget * 2 * num_kv_heads * page_size * head_dim) * sizeof(T) +
			vec_bytes(kv_indptr) + vec_bytes(kv_indices),
		"Read");
	state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");
	BatchDecodeHandler handler;

	if(cooperative) {
		// begin forward
		handler.BeginForward<PageStorage::kIndices, kv_layout, T, T, int32_t>(kv_indptr_host.data(),
																			  batch_size,
																			  num_qo_heads,
																			  num_kv_heads,
																			  head_dim,
																			  page_size,
																			  rotary_mode);
		state.exec([&](nvbench::launch&) {
			cudaError_t status =
				BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices, kv_layout, T, T>(
					&handler,
					thrust::raw_pointer_cast(q.data()),
					paged_kv,
					thrust::raw_pointer_cast(o.data()),
					/*lse=*/nullptr,
					num_qo_heads,
					rotary_mode);
			if(status != cudaSuccess) {
				state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
			}
		});
	} else {
		state.exec([&](nvbench::launch&) {
			cudaError_t status =
				BatchDecodeWithPagedKVCache<PageStorage::kIndices, kv_layout, T, T>(
					thrust::raw_pointer_cast(q.data()),
					paged_kv,
					kv_partition_info_t<int32_t>(),
					thrust::raw_pointer_cast(o.data()),
					nullptr,
					/*lse=*/nullptr,
					num_qo_heads,
					rotary_mode);
			if(status != cudaSuccess) {
				state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
			}
		});
	}
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_BATCH_DECODE(dtype)                                              \
	auto bench_flashinfer_batch_decode_##dtype##_ = bench_flashinfer_batch_decode<dtype>; \
	NVBENCH_BENCH(bench_flashinfer_batch_decode_##dtype##_)                               \
		.set_name("bench_flashinfer_batch_decode_" STR(dtype))                            \
		.add_int64_axis("seqlen", {4096, 8192, 16384, 32768, 65536})                      \
		.add_int64_axis("page_budget", {64, 128, 256, 512})                               \
		.add_int64_axis("page_size", {16})                                                \
		.add_int64_axis("num_qo_heads", {32})                                             \
		.add_int64_axis("num_kv_heads", {32})                                             \
		.add_int64_axis("cooperative", {1})

BENCH_FLASHINFER_BATCH_DECODE(half);