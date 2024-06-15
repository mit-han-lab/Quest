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

def _ref_cpu_estimate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
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

    cur_num_pages = (kv_len + page_size - 1) // page_size

    sign = (q > 0) + (~(q > 0)) * -1
    max_key = k * sign
    postive_query = q * sign

    padding_length = page_size - ((kv_len - 1) % page_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (
                    max_key.shape[0],
                    padding_length,
                    max_key.shape[2]
                ),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # q: [num_heads, qo_len, head_dim]
    # page_max_key: [num_heads, num_pages, page_size, head_dim]
    page_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1] // page_size,
        page_size,
        max_key.shape[2],
    ).amax(dim=-2, keepdim=False)
    assert page_max_key.dim() == 3

    # approx_attn_paged: [num_heads, qo_len, num_pages]
    approx_attn_paged = torch.matmul(postive_query.float(), page_max_key.transpose(1, 2)).to(q.dtype)
    approx_attn_paged = approx_attn_paged[:, :, :cur_num_pages - 1] # remove the last page

    return approx_attn_paged.squeeze(1).contiguous()

parameters = list(product(["float16"], [27, 61, 113, 482, 577, 1110, 1541, 2047, 3330]))
@pytest.mark.parametrize("dtype_str, kv_len", parameters)
@torch.inference_mode()
def test_estimate_correctness(dtype_str, kv_len):
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    qo_len = 1

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 16
    page_budget = 1024 # Not used here. Random initialize
    max_seq_len = 8192

    if kv_len <= page_size:
        pytest.skip("At least one page")
    if qo_len > kv_len:
        pytest.skip("qo_len should be less than kv_len")
        
    # layout: NHD
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)

    # Simulate Prefill
    # Doing this since we need begin_forward to prepare metadata
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

    # CUDA Evaluation
    testController.prepare_metadata(qo_len)
    testController.begin_forward(qo_len)
    quest.utils.append_kv(k_decode, v_decode, testController, 0)
    cuda_estimated_value = quest.utils.decode_estimate(
        q,
        testController,
        0,
    )
    testController.end_forward()

    # CPU Evaluation
    k = torch.cat([k_prefill, k_decode], dim=0)
    v = torch.cat([v_prefill, v_decode], dim=0)
    host_estimated_value = _ref_cpu_estimate(q, k, v, page_size)

    assert_close(cuda_estimated_value, host_estimated_value)