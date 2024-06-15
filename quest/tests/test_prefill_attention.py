import pytest
from itertools import product

import torch
import torch.nn as nn
import math

import quest.utils

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

def _ref_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L413
    # Assume all input layout: NHD 
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert k.size(0) == v.size(0)
    head_dim = q.size(2)
    qo_len = q.size(0)
    kv_len = k.size(0)
    
    assert kv_len >= qo_len

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)

    attn_mask = torch.ones_like(attn_weights, dtype=torch.bool)
    attn_mask = attn_mask.tril(diagonal=kv_len-qo_len)

    attn_weights[~attn_mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    return torch.matmul(attn_weights, v).transpose(0, 1)

parameters = list(product(["float16"], [13, 24, 51, 77, 244, 311, 502], [33, 66, 129, 400, 700, 1110]))
@pytest.mark.parametrize("dtype_str, qo_len, kv_len", parameters)
@torch.inference_mode()
def test_prefill_attention_correctness(dtype_str, qo_len, kv_len):
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len is not supported")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 16
    page_budget = 1024
    max_seq_len = 2048

    # layout: NHD
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device=device)

    testController = quest.utils.InferenceController(
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget,
        max_seq_len,
        dtype,
        device,
    )

    # Begin (prepare kv-cache metadata)
    testController.prepare_metadata(kv_len)
    testController.begin_forward(kv_len)
    # Construct KV with maintained metadata
    quest.utils.append_kv(k, v, testController, 0)
    o_device = quest.utils.prefill_forward(q, testController, 0)
    o_host = _ref_self_attention(q, k, v)

    assert_close(o_device, o_host)