#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void estimate_attn_score(torch::Tensor q,
						torch::Tensor o,
						torch::Tensor metadata_data,
						torch::Tensor metadata_indices,
						torch::Tensor metadata_indptr,
						unsigned int metadata_last_page_len,
						unsigned int metadata_last_page_idx,
						unsigned int layout) {
	constexpr size_t batch_size = 1;

	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(q); // [1, num_heads, head_dim]
	// (num_max_pages, 2, H_kv, page_size, head_dim) for HND
	// (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
	CHECK_INPUT(metadata_data);
	CHECK_INPUT(metadata_indices);

	CHECK_DIM(3, q);
	CHECK_DIM(5, metadata_data);
	CHECK_DIM(1, metadata_indices);

	CHECK_EQ(q.size(0), 1);
	CHECK_EQ(metadata_indices.scalar_type(), torch::kInt32);
	#endif

	size_t num_heads = q.size(1);
	size_t head_dim = q.size(2);
	size_t page_size;

	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		page_size = metadata_data.size(3);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(metadata_data.size(2), num_heads);
		CHECK_EQ(metadata_data.size(4), head_dim);
		#endif
	} else {
		page_size = metadata_data.size(2);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(metadata_data.size(3), num_heads);
		CHECK_EQ(metadata_data.size(4), head_dim);
		#endif
	}

	// size_t output_len = (metadata_indices.size(0) - 1) * page_size + metadata_last_page_len - 1;
	// torch::Tensor o = torch::empty(
		// {static_cast<signed long>(num_heads), static_cast<signed long>(output_len)}, q.options());
		
	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_heads,
				page_size,
				head_dim,
				batch_size,
				0,
				metadata_last_page_len,
				metadata_last_page_idx,
				static_cast<c_type*>(metadata_data.data_ptr()),
				static_cast<int32_t*>(metadata_indices.data_ptr()),
				static_cast<int32_t*>(metadata_indptr.data_ptr()));
			cudaError_t status =
				MaxPossibleSampleWithPagedKVCache<PageStorage::kIndices,
												  KV_LAYOUT,
												  c_type,
												  c_type,
												  int32_t>(static_cast<c_type*>(q.data_ptr()),
														   paged_kv,
														   static_cast<c_type*>(o.data_ptr()),
														   num_heads,
														   /*rotary_mode*/ RotaryMode::kNone);
			TORCH_CHECK(status == cudaSuccess,
						"Estimate_attn_score failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});
	TORCH_CHECK(success, "Estimate_attn_score failed to dispatch with dtype ", q.scalar_type());
}