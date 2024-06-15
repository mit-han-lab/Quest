import pytest
from itertools import product
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import quest.utils

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

# Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def _ref_apply_qk_rope(q, k, past_kv_len):
    # q,k: [seq_len, num_heads, head_dim]
    seq_len = q.size(0)
    num_heads = q.size(1)
    head_dim = q.size(2)

    rotary_emb = LlamaRotaryEmbedding(head_dim).to(q.device)
    cos, sin = rotary_emb(q, seq_len=seq_len + past_kv_len)
    position_ids = torch.arange(
        past_kv_len, seq_len + past_kv_len, dtype=torch.long, device=q.device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
    query_states, key_states = apply_rotary_pos_emb(q.transpose(0, 1), k.transpose(0, 1), cos, sin, position_ids)
    return query_states.squeeze(0).transpose(0, 1), key_states.squeeze(0).transpose(0, 1)

parameters = list(product(["float16"], [13, 24, 51, 77, 244, 311, 502, 1110], [2, 5, 7, 19, 31, 69, 111, 251]))
@pytest.mark.parametrize("dtype_str, past_kv_len, seq_len", parameters)
@torch.inference_mode()
def test_apply_qk_rope(dtype_str, past_kv_len, seq_len):
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    
    # NHD: [seq_len, num_heads, head_dim]
    q = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)

    q_ref, k_ref = _ref_apply_qk_rope(q, k, past_kv_len)
    quest.utils.apply_rope_in_place(q, k, past_kv_len)

    assert_close(q, q_ref)
    assert_close(k, k_ref)