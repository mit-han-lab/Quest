#include <torch/extension.h>
#include "bsk_ops.h"

PYBIND11_MODULE(_kernels, m) {
	m.def("apply_rope_in_place", &apply_rope_in_place, "Apply RoPE on Q/K in place.");
	m.def("rms_norm_forward", &rms_norm_forward, "rms_norm_forward by cutlass");
	m.def("topk_filtering", &topk_filtering, "Top-k filtering operator");
	m.def("estimate_attn_score", &estimate_attn_score, "Estimate Attention Score operator");
	m.def("append_kv_cache_prefill", &append_kv_cache_prefill, "Append KV-Cache Prefill operator");
	m.def("append_kv_cache_decode", &append_kv_cache_decode, "Append KV-Cache Decode operator");
	m.def("prefill_with_paged_kv_cache",
		  &prefill_with_paged_kv_cache,
		  "Multi-request batch prefill with paged KV-Cache operator");
	py::class_<BatchDecodeWithPagedKVCachePyTorchWrapper>(
		m, "BatchDecodeWithPagedKVCachePyTorchWrapper")
		.def(py::init(&BatchDecodeWithPagedKVCachePyTorchWrapper::Create))
		.def("begin_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
		.def("end_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
		.def("forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
}