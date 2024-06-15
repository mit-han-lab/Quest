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
#include <decode/decode_page.cuh>
#include <nvbench/nvbench.cuh>
#include <vector>

#include "cpu_utils.h"

using utils::vec_bytes;
using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void bench_flashinfer_append_prefill(nvbench::state& state) {
	constexpr size_t head_dim = 128;
	constexpr size_t batch_size = 1;
	size_t seqlen = state.get_int64("seqlen");
	size_t page_size = state.get_int64("page_size");
	size_t num_kv_heads = state.get_int64("num_kv_heads");
	// KV cache:
	size_t num_pages = flashinfer::ceil_div(seqlen, page_size);
	size_t last_page_len = (seqlen - 1) % page_size + 1;
	// shuffle the order of pages
	std::vector<int32_t> append_indptr_host({0, static_cast<int32_t>(seqlen)});
	std::vector<int32_t> kv_indptr_host({0, static_cast<int32_t>(num_pages)});
	std::vector<int32_t> kv_indicies_host(num_pages);
	std::iota(kv_indicies_host.begin(), kv_indicies_host.end(), 0);
	std::shuffle(
		kv_indicies_host.begin(), kv_indicies_host.end(), std::mt19937(std::random_device()()));
	int32_t last_page_idx = kv_indicies_host.back();

	// Metadata Page Configuration
	size_t num_chunks = flashinfer::ceil_div(num_pages, page_size);
	size_t last_chunk_len = (num_pages - 1) % page_size + 1;
	std::vector<int32_t> chunk_indptr_host({0, static_cast<int32_t>(num_chunks)});
	std::vector<int32_t> chunk_indicies_host(num_chunks);
	std::iota(chunk_indicies_host.begin(), chunk_indicies_host.end(), 0);
	std::shuffle(chunk_indicies_host.begin(),
				 chunk_indicies_host.end(),
				 std::mt19937(std::random_device()()));
	int32_t last_chunk_idx = chunk_indicies_host.back();

	// Move to GPU
	thrust::device_vector<T> kv_data(num_pages * 2 * num_kv_heads * page_size * head_dim);
	thrust::device_vector<int32_t> append_indptr(append_indptr_host);
	thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
	thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);

	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0, // Page budget, useless when appending
		last_page_len,
		last_page_idx,
		thrust::raw_pointer_cast(kv_data.data()),
		thrust::raw_pointer_cast(kv_indices.data()),
		thrust::raw_pointer_cast(kv_indptr.data()));

	thrust::device_vector<T> chunk_data(num_chunks * 2 * num_kv_heads * page_size * head_dim);
	thrust::device_vector<int32_t> chunk_indptr(chunk_indptr_host);
	thrust::device_vector<int32_t> chunk_indices(chunk_indicies_host);
	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_chunk(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0, // Page budget, useless when appending
		last_chunk_len,
		last_chunk_idx,
		thrust::raw_pointer_cast(chunk_data.data()),
		thrust::raw_pointer_cast(chunk_indices.data()),
		thrust::raw_pointer_cast(chunk_indptr.data()));

	// Allocate input data:
	thrust::device_vector<T> k(batch_size * num_kv_heads * head_dim * seqlen);
	thrust::device_vector<T> v(batch_size * num_kv_heads * head_dim * seqlen);

	state.add_global_memory_reads<uint8_t>(vec_bytes(k) + vec_bytes(v) + vec_bytes(kv_indptr) +
											   vec_bytes(kv_indices) + vec_bytes(append_indptr) +
											   vec_bytes(chunk_data) + vec_bytes(chunk_indptr) +
											   vec_bytes(chunk_indices),
										   "Read");
	state.add_global_memory_writes<uint8_t>(vec_bytes(k) + vec_bytes(v) + vec_bytes(chunk_data),
											"Write");

	state.exec([&](nvbench::launch&) {
		cudaError_t status =
			AppendPagedKVCachePrefill<PageStorage::kIndices, kv_layout, T, int32_t>(
				paged_kv,
				paged_chunk,
				thrust::raw_pointer_cast(k.data()),
				thrust::raw_pointer_cast(v.data()),
				thrust::raw_pointer_cast(append_indptr.data()),
				nullptr);
		if(status != cudaSuccess) {
			state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
		}
	});
}

template <typename T>
void bench_flashinfer_append_decode(nvbench::state& state) {
	constexpr size_t batch_size = 1;
	constexpr size_t head_dim = 128;
	size_t seqlen = state.get_int64("seqlen"); // Exisiting
	size_t page_size = state.get_int64("page_size");
	size_t num_kv_heads = state.get_int64("num_kv_heads");

	size_t seqlen_appened = seqlen + 1;
	size_t num_pages = flashinfer::ceil_div(seqlen_appened, page_size);
	size_t last_page_len = (seqlen_appened - 1) % page_size + 1;

	std::vector<int32_t> page_indices_host(num_pages);
	std::iota(page_indices_host.begin(), page_indices_host.end(), 0);
	std::shuffle(
		page_indices_host.begin(), page_indices_host.end(), std::mt19937(std::random_device()()));
	int32_t last_page_idx = page_indices_host.back();
	std::vector<int32_t> page_indptr_host({0, static_cast<int32_t>(num_pages)});
	std::vector<T> kv_page_data_host(num_pages * 2 * num_kv_heads * page_size * head_dim);

	std::vector<T> key(1 * num_kv_heads * head_dim);
	std::vector<T> value(1 * num_kv_heads * head_dim);
	// Move to GPU
	thrust::device_vector<T> kv_page_data(kv_page_data_host);
	thrust::device_vector<int32_t> page_indices(page_indices_host);
	thrust::device_vector<int32_t> page_indptr(page_indptr_host);
	thrust::device_vector<T> k(key);
	thrust::device_vector<T> v(value);
	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0, // Page budget, useless when appending
		last_page_len,
		last_page_idx,
		thrust::raw_pointer_cast(kv_page_data.data()),
		thrust::raw_pointer_cast(page_indices.data()),
		thrust::raw_pointer_cast(page_indptr.data()));
	// Construct metadata page_kv
	// num_pages is after appending
	size_t num_chunks = flashinfer::ceil_div(num_pages, page_size);
	size_t last_chunk_len = (num_pages - 1) % page_size + 1;
	std::vector<int32_t> chunk_indptr_host({0, static_cast<int32_t>(num_chunks)});
	std::vector<int32_t> chunk_indicies_host(num_chunks);
	std::iota(chunk_indicies_host.begin(), chunk_indicies_host.end(), 0);
	std::shuffle(chunk_indicies_host.begin(),
				 chunk_indicies_host.end(),
				 std::mt19937(std::random_device()()));
	int32_t last_chunk_idx = chunk_indicies_host.back();
	std::vector<T> chunk_data_host(num_chunks * 2 * num_kv_heads * page_size * head_dim);

	thrust::device_vector<T> chunk_data(chunk_data_host);
	thrust::device_vector<int32_t> chunk_indptr(chunk_indptr_host);
	thrust::device_vector<int32_t> chunk_indices(chunk_indicies_host);
	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> metadata_page(
		num_kv_heads,
		page_size,
		head_dim,
		batch_size,
		0, // Page budget, useless when appending
		last_chunk_len,
		last_chunk_idx,
		thrust::raw_pointer_cast(chunk_data.data()),
		thrust::raw_pointer_cast(chunk_indices.data()),
		thrust::raw_pointer_cast(chunk_indptr.data()));

	state.add_global_memory_reads<uint8_t>(
		2 * vec_bytes(k) + 2 * vec_bytes(v) + vec_bytes(page_indptr), "Read");
	state.add_global_memory_writes<uint8_t>(2 * vec_bytes(k) + 2 * vec_bytes(v), "Write");

	// call kernel
	state.exec([&](nvbench::launch&) {
		cudaError_t status = AppendPagedKVCacheDecode<PageStorage::kIndices, kv_layout, T, int32_t>(
			paged_kv,
			metadata_page,
			thrust::raw_pointer_cast(k.data()),
			thrust::raw_pointer_cast(v.data()),
			nullptr);
		if(status != cudaSuccess) {
			state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
		}
	});
}

template <typename T>
void bench_flashinfer_qk_apply_rope(nvbench::state& state) {
	constexpr size_t head_dim = 128;
	constexpr size_t past_kv_len = 1024; // Arbitrarily chosen. No influence.
	size_t seq_len = state.get_int64("seq_len");
	size_t num_kv_heads = state.get_int64("num_kv_heads");

	// q/k layout is naturally NHD, which is generated by Q,K,V
	std::vector<T> key(seq_len * num_kv_heads * head_dim);
	std::vector<T> query(seq_len * num_kv_heads * head_dim);
	// Move to GPU
	thrust::device_vector<T> k(key);
	thrust::device_vector<T> q(query);

	state.add_global_memory_reads<uint8_t>(vec_bytes(k) + vec_bytes(q), "Read");
	state.add_global_memory_writes<uint8_t>(vec_bytes(k) + vec_bytes(q), "Write");

	// call kernel
	state.exec([&](nvbench::launch&) {
		cudaError_t status = QKApplyRotaryInPlace<T>(thrust::raw_pointer_cast(q.data()),
													 thrust::raw_pointer_cast(k.data()),
													 seq_len,
													 past_kv_len,
													 num_kv_heads,
													 num_kv_heads,
													 head_dim);
		if(status != cudaSuccess) {
			state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
		}
	});
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_APPEND_PREFILL(dtype)                                                \
	auto bench_flashinfer_append_prefill_##dtype##_ = bench_flashinfer_append_prefill<dtype>; \
	NVBENCH_BENCH(bench_flashinfer_append_prefill_##dtype##_)                                 \
		.set_name("bench_flashinfer_append_prefill_" STR(dtype))                              \
		.add_int64_axis("seqlen", {4096, 8192, 16384, 32768, 65536})                          \
		.add_int64_axis("page_size", {16})                                                    \
		.add_int64_axis("num_kv_heads", {32})

#define BENCH_FLASHINFER_APPEND_DECODE(dtype)                                               \
	auto bench_flashinfer_append_decode_##dtype##_ = bench_flashinfer_append_decode<dtype>; \
	NVBENCH_BENCH(bench_flashinfer_append_decode_##dtype##_)                                \
		.set_name("bench_flashinfer_append_decode_" STR(dtype))                             \
		.add_int64_axis("seqlen", {4096, 8192, 16384, 32768, 65536})                        \
		.add_int64_axis("page_size", {16})                                                  \
		.add_int64_axis("num_kv_heads", {32})

#define BENCH_FLASHINFER_QK_APPLY_ROPE(dtype)                                               \
	auto bench_flashinfer_qk_apply_rope_##dtype##_ = bench_flashinfer_qk_apply_rope<dtype>; \
	NVBENCH_BENCH(bench_flashinfer_qk_apply_rope_##dtype##_)                                \
		.set_name("bench_flashinfer_qk_apply_rope_" STR(dtype))                             \
		.add_int64_axis("seq_len", {1, 4, 8, 16, 4096, 8192, 16384})                        \
		.add_int64_axis("num_kv_heads", {32})

BENCH_FLASHINFER_APPEND_PREFILL(half);
BENCH_FLASHINFER_APPEND_DECODE(half);
BENCH_FLASHINFER_QK_APPLY_ROPE(half);