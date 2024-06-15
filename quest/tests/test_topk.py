import pytest
from itertools import product

import torch
import torch.nn as nn
import math

import quest.utils

# This file is used for testing topk kernel from libRAFT
# We do not seriously compare the topk indices since the random value leads to similar tensor.
# Detailed evaluation is in quest/tests/topk/*
# Therefore instead, we pay attention to the qk result (collected attention scores).

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

parameters = list(product(["float16"], [13, 24, 51, 77, 244, 311, 502, 1110], [2, 5, 7, 19, 31, 69, 111, 251]))
@pytest.mark.parametrize("dtype_str, kv_len, k_budget", parameters)
@torch.inference_mode()
def test_topk_correctness(dtype_str, kv_len, k_budget):
    if k_budget > kv_len:
        pytest.skip("k should be less than kv_len")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    qo_len = 1
    num_heads = 32
    head_dim = 128
    
    # HND: [num_heads, seq_len, head_dim]
    q = torch.randn(num_heads, qo_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_heads, kv_len, head_dim, dtype=dtype, device=device)

    # [num_heads, qo_len, kv_len]
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    torch_topk_value = torch.topk(attn_weights, k=k_budget, dim=-1).values

    # sum on dim=-1
    torch_topk_value = torch_topk_value.sum(dim=-1).squeeze()

    cuda_input_data = attn_weights.squeeze()
    cuda_input_indices = torch.arange(0, kv_len, dtype=torch.int32, device=device).repeat(num_heads, 1)
    cuda_output_data = torch.randn(num_heads, k_budget, dtype=dtype, device=device)
    cuda_output_indices = torch.arange(0, k_budget, dtype=torch.int32, device=device).repeat(num_heads, 1)
    topk_buf = torch.zeros((num_heads, 8192 * 2 * (2+4) // 2 // 48), dtype=dtype, device=device)

    quest.utils._kernels.topk_filtering(
        cuda_input_data,
        cuda_input_indices,
        cuda_output_data,
        cuda_output_indices,
        topk_buf,
        k_budget
    )

    cuda_output_data = cuda_output_data.sum(dim=-1).squeeze()

    assert_close(cuda_output_data, torch_topk_value)