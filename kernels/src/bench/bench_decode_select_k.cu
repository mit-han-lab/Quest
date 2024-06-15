#include <algorithm>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <vector>

#include <cpu_utils.h>
#include <topk/decode_select_k.cuh>

template <typename d_type, typename idx_type, int num_heads = 32>
void bench_decode_select_k(nvbench::state& state) {
	size_t seq_len = state.get_int64("seq_len");
	size_t k = state.get_int64("k");
	if(k > seq_len){
		state.skip("k > seq_len");
	}
	raft::matrix::SelectAlgo _algo = (raft::matrix::SelectAlgo)state.get_int64("algo");
	// Initialize host data
	std::vector<d_type> h_in(num_heads * seq_len);
	std::vector<idx_type> h_in_idx(num_heads * seq_len);
	utils::random_init_vec(h_in, h_in.size());
	utils::shuffle_init_index(h_in_idx, seq_len, num_heads);

	// Initialize device data
	thrust::device_vector<d_type> d_in(h_in);
	thrust::device_vector<idx_type> d_in_idx(h_in_idx);
	thrust::device_vector<d_type> d_out(k * num_heads);
	thrust::device_vector<idx_type> d_out_idx(k * num_heads);

	const size_t buf_size_bytes = num_heads * seq_len * (sizeof(d_type) + sizeof(idx_type)) * 2 / 24;
	thrust::device_vector<char> buf(buf_size_bytes);

	state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
		decode_select_k<d_type, idx_type, num_heads>(thrust::raw_pointer_cast(d_in.data()),
													 thrust::raw_pointer_cast(d_in_idx.data()),
													 thrust::raw_pointer_cast(buf.data()),
													 seq_len,
													 k,
													 thrust::raw_pointer_cast(d_out.data()),
													 thrust::raw_pointer_cast(d_out_idx.data()),
													 true,
													 _algo);
	});
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_DECODE_SELECT_K(dtype_in, dtype_out)                            \
	auto bench_decode_select_k##dtype_in##_##dtype_out##_ =                   \
		bench_decode_select_k<dtype_in, dtype_out>;                           \
	NVBENCH_BENCH(bench_decode_select_k##dtype_in##_##dtype_out##_)           \
		.set_name(("bench_decode_select_k" STR(dtype_in) "_" STR(dtype_out))) \
		.add_int64_axis("seq_len", {128, 256, 512, 1024, 2048, 4096, 8192})   \
		.add_int64_axis("k", {64, 128, 256})                                  \
		.add_int64_axis("algo", {0})

BENCH_DECODE_SELECT_K(__nv_half, uint32_t);