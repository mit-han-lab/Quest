import torch
from typing import Optional

import quest._kernels as _kernels
from quest.utils.utils import TensorLayout

def _check_kv_layout(kv_layout: str):
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalide kv_layout {}".format(kv_layout))

class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for batch_decode_with_paged_kv_cache kernel.

    To accelerate computation, FlashInfer's batch decode operators creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode calls (e.g. different Transformer layers). This wrapper class manages
    the lifecycle of these data structures.
    """

    def __init__(self, kv_layout: str = "NHD"):
        _check_kv_layout(kv_layout)
        self.kv_layout = kv_layout
        self._wrapper = _kernels.BatchDecodeWithPagedKVCachePyTorchWrapper(
            getattr(TensorLayout, kv_layout)
        )

    def begin_forward(
        self,
        indptr: torch.Tensor, # [0, Page_budget - 1], once per forward for all layers
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        data_type,
    ):
        r"""The begin_forward method should be called before any batch decode calls,
        auxiliary data structures will be created during this call and cached for
        multiple forward calls.
        """

        # NOTE(Zihao): the following tensor acts as placeholder to pass dtype info
        empty_data = torch.empty(0, dtype=data_type)
        self._wrapper.begin_forward(
            indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            empty_data,
        )

    def end_forward(self):
        r"""The end_forward method can clear the cached data structures."""
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        paged_kv_data: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_last_page_len: int,
        paged_kv_last_page_idx: int,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        self._wrapper.forward(
            q,
            o,
            paged_kv_data,
            paged_kv_indices,
            paged_kv_indptr,
            paged_kv_last_page_len,
            paged_kv_last_page_idx,
            rope_scale,
            rope_theta,
        )