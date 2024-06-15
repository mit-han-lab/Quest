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

#include <nvbench/nvbench.cuh>
#include <prefill/prefill.cuh>

using flashinfer::QKVLayout;
using flashinfer::RotaryMode;

template <typename dtype_in, typename dtype_out>
void bench_flashinfer_prefill(nvbench::state& state) {
	constexpr RotaryMode rotary_mode = RotaryMode::kNone;
	constexpr size_t batch_size = 1;
	constexpr QKVLayout kv_layout = QKVLayout::kNHD;

	size_t kv_len = state.get_int64("kv_len");
	size_t qo_len = state.get_int64("qo_len");
	if(qo_len > kv_len) {
		state.skip("qo_len > kv_len");
	}
	size_t page_size = state.get_int64("page_size");
	size_t num_qo_heads = state.get_int64("num_qo_heads");
	size_t num_kv_heads = state.get_int64("num_kv_heads");
	size_t head_dim = state.get_int64("head_dim");

	bool causal = state.get_int64("causal");
	bool allow_fp16_qk_reduction = state.get_int64("allow_fp16_qk_reduction");
	// Allocate input data:
	thrust::device_vector<dtype_in> Q(qo_len * num_qo_heads * head_dim);
	thrust::device_vector<dtype_out> O(qo_len * num_qo_heads * head_dim);

	size_t num_pages = flashinfer::ceil_div(kv_len, page_size);
	std::vector<int32_t> page_indices(num_pages);
	std::iota(page_indices.begin(), page_indices.end(), 0);
	std::random_shuffle(page_indices.begin(), page_indices.end());
	size_t last_page_idx = page_indices.back();
	size_t last_page_len = (kv_len - 1) % page_size + 1;
	std::vector<int32_t> page_indptr_host({0, static_cast<int32_t>(num_pages)});

	thrust::device_vector<dtype_in> kv_data(num_pages * page_size * num_kv_heads * head_dim * 2);
	thrust::device_vector<int32_t> kv_indices(page_indices);
	thrust::device_vector<int32_t> kv_indptr(page_indptr_host);

	flashinfer::paged_kv_t<flashinfer::PageStorage::kIndices, kv_layout, dtype_in, int32_t>
		paged_kv(num_kv_heads,
				 page_size,
				 head_dim,
				 batch_size,
				 0, // page budget. useless here
				 last_page_len,
				 last_page_idx,
				 thrust::raw_pointer_cast(kv_data.data()),
				 thrust::raw_pointer_cast(kv_indices.data()),
				 thrust::raw_pointer_cast(kv_indptr.data()));

	// useless in bsz = 1. Prefix sum for difference bsz
	std::vector<int32_t> q_indptr_host({0, static_cast<int32_t>(qo_len)});
	thrust::device_vector<int32_t> q_indptr(q_indptr_host);
	// Provide throughput information:
	state.add_global_memory_reads<dtype_in>(
		(2 * qo_len * num_qo_heads + 2 * kv_len * num_kv_heads) * head_dim, "Read");
	state.add_global_memory_writes<dtype_out>(qo_len * num_qo_heads * head_dim, "Write");

	state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
			   [&](nvbench::launch& launch, auto& timer) {
				   timer.start();
				   auto status =
					   flashinfer::BatchPrefillWithPagedKVCache<flashinfer::PageStorage::kIndices,
																kv_layout,
																dtype_in,
																dtype_out,
																int32_t>(
						   thrust::raw_pointer_cast(Q.data()),
						   thrust::raw_pointer_cast(q_indptr.data()),
						   paged_kv,
						   thrust::raw_pointer_cast(O.data()),
						   /*tmp=*/nullptr,
						   /*lse=*/nullptr,
						   num_qo_heads,
						   causal,
						   rotary_mode,
						   allow_fp16_qk_reduction);
				   if(status != cudaSuccess) {
					   state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
				   }
				   timer.stop();
			   });

	const auto measured_mean = static_cast<nvbench::float32_t>(
		state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
	auto& summ = state.add_summary("nv/tflops");
	summ.set_string("description", "Achieved TFlops/s");
	summ.set_string("name", "TFlops/s");
	float tflops;
	if(causal) {
		tflops =
			qo_len * (2 * kv_len - qo_len) * 2 * num_qo_heads * head_dim / measured_mean / 1e12;
	} else {
		tflops = qo_len * kv_len * 4 * num_qo_heads * head_dim / measured_mean / 1e12;
	}
	summ.set_float64("value", tflops);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define BENCH_FLASHINFER_PREFILL(dtype_in, dtype_out)                                           \
	auto bench_flashinfer_prefill_##dtype_in##_##dtype_out##_ =                                 \
		bench_flashinfer_prefill<dtype_in, dtype_out>;                                          \
	NVBENCH_BENCH(bench_flashinfer_prefill_##dtype_in##_##dtype_out##_)                         \
		.set_name(("bench_flashinfer_prefill_" STR(dtype_in) "_" STR(dtype_out)))               \
		.add_int64_axis("qo_len", {128, 217, 330, 1110, 8192})                                  \
		.add_int64_axis("kv_len", {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) \
		.add_int64_axis("num_qo_heads", {32})                                                   \
		.add_int64_axis("num_kv_heads", {32})                                                   \
		.add_int64_axis("head_dim", {128})                                                      \
		.add_int64_axis("page_size", {16})                                                      \
		.add_int64_axis("causal", {0, 1})                                                       \
		.add_int64_axis("allow_fp16_qk_reduction", {0, 1})

BENCH_FLASHINFER_PREFILL(half, half);
