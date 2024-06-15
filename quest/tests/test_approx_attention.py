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

def _ref_self_approx_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
    page_budget: int,
):
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

    cur_num_pages = (kv_len + page_size - 1) // page_size
    if cur_num_pages > page_budget:
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
        approx_attn_paged = torch.matmul(postive_query.float(), page_max_key.transpose(1, 2))
        approx_attn_paged = approx_attn_paged[:, :, :cur_num_pages - 1] # remove the last page
        # topk: [num_heads, qo_len, page_budget-1]
        _, topk = approx_attn_paged.topk(page_budget - 1, dim=-1)

        saved_topk_indices = topk.clone().squeeze().to(torch.int32) # save for kernel usage

        topk = torch.cat(
            [
                topk,
                torch.tensor([cur_num_pages-1], device=topk.device).repeat(
                    topk.shape[0], topk.shape[1], 1
                ),
            ],
            dim=-1,
        )
        assert topk.shape[-1] == page_budget, "Topk indices must align with page_budget"

        # restore the mask along sequence dimension
        topk = topk.unsqueeze(-1).repeat(1, 1, 1, page_size) * page_size + torch.arange(page_size, device=topk.device)
        topk = topk.reshape(topk.shape[0], topk.shape[1], -1) # [num_heads, qo_len, padded_seq_len]
        topk = topk[:, :, :page_budget * page_size - padding_length] # remove the padding

        # mask generate
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom.scatter_(-1, topk, True)    
        attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    else:
        saved_topk_indices = None

    attn_weights[~attn_mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    return torch.matmul(attn_weights, v).transpose(0, 1), saved_topk_indices

parameters = list(product(["float16"], [1], [27, 61, 113, 482, 577, 1011, 1541], [4, 7, 15, 31, 55, 71]))
@pytest.mark.parametrize("dtype_str, qo_len, kv_len, page_budget", parameters)
@torch.inference_mode()
def test_approx_attention_correctness(dtype_str, qo_len, kv_len, page_budget):
    if qo_len != 1:
        pytest.skip("qo_len should be 1 for decode.")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 16
    max_seq_len = 2048

    if kv_len <= page_size:
        pytest.skip("At least one page")
    if qo_len > kv_len:
        pytest.skip("qo_len should be less than kv_len")
        
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

    # CPU Evaluation
    k = torch.cat([k_prefill, k_decode], dim=0)
    v = torch.cat([v_prefill, v_decode], dim=0)
    o_host, saved_topk_indices = _ref_self_approx_attention(q, k, v, page_size, page_budget)

    # CUDA Evaluation
    testController.prepare_metadata(qo_len)
    testController.begin_forward(qo_len)
    quest.utils.append_kv(k_decode, v_decode, testController, 0)
    
    if testController.need_estimate() == False:
        o_device = quest.utils.decode_sparse_attn(
            q,
            testController,
            0,
            testController.kv_indices_without_last,
        )
    else:
        # Comments since Top-k will leads to similar qk but different values.
        # Here we control the same top-k indices for correctness test.
        # Estimate and top-k kernel are tested separately 

        # estimated_attn_score = quest.utils.decode_estimate(
        #     q,
        #     testController,
        #     0,
        # )
        # quest.utils.decode_topk(
        #     estimated_attn_score,
        #     testController,
        # )
        o_device = quest.utils.decode_sparse_attn(
            q,
            testController,
            0,
            saved_topk_indices,
        )
    testController.end_forward()

    assert_close(o_device, o_host)