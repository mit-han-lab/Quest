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

parameters = list(product(["float16"], [1], [27, 61, 113, 482, 577, 1011]))
@pytest.mark.parametrize("dtype_str, qo_len, kv_len", parameters)
@torch.inference_mode()
def test_decode_attention_correctness(dtype_str, qo_len, kv_len):
    if qo_len != 1:
        pytest.skip("qo_len should be 1 for decode.")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 16
    page_budget = 1024 # Here we do not test approx attention, which is tested in test_approx_attention.py.
    max_seq_len = 2048

    if kv_len <= page_size:
        pytest.skip("At least one page")
        
    # layout: NHD
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    # Simulate Prefill
    k_prefill = torch.randn(kv_len-1, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(kv_len-1, num_heads, head_dim, dtype=dtype, device=device)

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

    # Begin: Fill in prefill kv-data
    testController.prepare_metadata(kv_len-1)
    testController.begin_forward(kv_len-1)
    # Construct KV
    quest.utils.append_kv(k_prefill, v_prefill, testController, 0)
    testController.end_forward()

    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    # Real decoding starts
    testController.prepare_metadata(1)
    testController.begin_forward(1)
    quest.utils.append_kv(k_decode, v_decode, testController, 0)
    # No CPU test cases
    assert testController.need_estimate() == False
    o_device = quest.utils.decode_sparse_attn(
        q,
        testController,
        0,
        testController.kv_indices_without_last,
    )
    testController.end_forward()

    # stack k,v and get o
    k = torch.cat([k_prefill, k_decode], dim=0)
    v = torch.cat([v_prefill, v_decode], dim=0)
    o_host = _ref_self_attention(q, k, v)

    assert_close(o_device, o_host)