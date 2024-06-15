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
void bench_flashinfer_max_possible(nvbench::state& state) {
	constexpr size_t head_dim = 128;
	constexpr size_t batch_size = 1;
	constexpr auto rotary_mode = RotaryMode::kNone;
	size_t seqlen = state.get_int64("seqlen");
	size_t page_size = state.get_int64("page_size");
	size_t num_qo_heads = state.get_int64("num_qo_heads");
	size_t num_kv_heads = state.get_int64("num_kv_heads");
	assert(num_qo_heads == num_kv_heads); // Not support GQA now

	// KV cache:
	size_t num_pages = flashinfer::ceil_div(seqlen, page_size);
	size_t num_chunks = flashinfer::ceil_div(num_pages, page_size);
	size_t output_len = num_pages - 1;
	size_t last_chunk_len = (num_pages - 1) % page_size + 1;
	std::vector<int32_t> chunk_indptr_host({0, static_cast<int32_t>(num_chunks)});
	std::vector<int32_t> chunk_indice_host(num_chunks);
	std::iota(chunk_indice_host.begin(), chunk_indice_host.end(), 0);
	std::shuffle(chunk_indice_host.begin(), chunk_indice_host.end(), std::mt19937(0));
	size_t last_chunk_idx = chunk_indice_host.back();

	thrust::device_vector<T> chunk_data(num_chunks * 2 * num_kv_heads * page_size * head_dim);
	thrust::device_vector<int32_t> chunk_indptr(chunk_indptr_host);
	thrust::device_vector<int32_t> chunk_indices(chunk_indice_host);
	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0,
		last_chunk_len,
		last_chunk_idx,
		thrust::raw_pointer_cast(chunk_data.data()),
		thrust::raw_pointer_cast(chunk_indices.data()),
		thrust::raw_pointer_cast(chunk_indptr.data()));
	// Allocate input data:
	thrust::device_vector<T> q(batch_size * num_qo_heads * head_dim);
	thrust::device_vector<T> o(batch_size * num_qo_heads * output_len);
	state.add_global_memory_reads<uint8_t>(vec_bytes(q) + vec_bytes(chunk_data) +
											   vec_bytes(chunk_indptr) + vec_bytes(chunk_indices),
										   "Read");
	state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");
	state.exec([&](nvbench::launch&) {
		cudaError_t status =
			MaxPossibleSampleWithPagedKVCache<PageStorage::kIndices, kv_layout, T, T, int32_t>(
				thrust::raw_pointer_cast(q.data()),
				paged_kv,
				thrust::raw_pointer_cast(o.data()),
				num_qo_heads,
				rotary_mode);
		if(status != cudaSuccess) {
			state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
		}
	});
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_MAX_POSSIBLE(dtype)                                             \
	auto bench_flashinfer_max_possible##dtype##_ = bench_flashinfer_max_possible<dtype>; \
	NVBENCH_BENCH(bench_flashinfer_max_possible##dtype##_)                               \
		.set_name("bench_flashinfer_max_possible" STR(dtype))                            \
		.add_int64_axis("seqlen", {2048, 4096, 8192, 16384, 32768, 65536, 131072})       \
		.add_int64_axis("page_size", {16})                                               \
		.add_int64_axis("num_qo_heads", {32})                                            \
		.add_int64_axis("num_kv_heads", {32})

BENCH_FLASHINFER_MAX_POSSIBLE(half);