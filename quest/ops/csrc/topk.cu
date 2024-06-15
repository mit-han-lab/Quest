#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

// Note that estimated_indices does not contain the last page
void topk_filtering(torch::Tensor estimated_value,
							 torch::Tensor estimated_indices,
							 torch::Tensor d_out,
							 torch::Tensor indices_out,
							 torch::Tensor buf,
							 unsigned int page_budget) {
	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(estimated_value); // [num_heads, num_pages]
	CHECK_INPUT(estimated_indices); // [num_heads, num_pages]
	CHECK_DIM(2, estimated_value);
	CHECK_DIM(2, estimated_indices);
	#endif

	auto num_heads = estimated_value.size(0);
	auto num_pages = estimated_value.size(1);

	#ifdef BSK_TORCH_CHECK
	CHECK_EQ(num_pages, estimated_indices.size(1));
	CHECK_EQ(num_heads, estimated_indices.size(0));
	CHECK_GE(num_pages, page_budget);
	CHECK_EQ(estimated_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(32, num_heads); // Not necessary, but for Llama-7b
	CHECK_EQ(page_budget, d_out.size(1));
	CHECK_EQ(page_budget, indices_out.size(1));
	#endif

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(estimated_value.scalar_type(), c_type, [&] {
		decode_select_k<c_type, int32_t, 32>(
			static_cast<c_type*>(estimated_value.data_ptr()),
			static_cast<int32_t*>(estimated_indices.data_ptr()),
			static_cast<char*>(buf.data_ptr()),
			num_pages,
			page_budget,
			static_cast<c_type*>(d_out.data_ptr()),
			static_cast<int32_t*>(indices_out.data_ptr()),
			true);
		return true;
	});
	TORCH_CHECK(success, "Top-k filtering failed to dispatch with dtype ", estimated_value.scalar_type());
}