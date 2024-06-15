# This file is modified from Punica Project
# Check ref: https://github.com/punica-ai/punica

from quest.utils.utils import TensorLayout
import torch

class KvPool:

  def __init__(
      self,
      num_layers: int,
      num_heads: int,
      head_dim: int,
      capacity: int,
      block_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self._layout = TensorLayout.NHD
    self._buf = torch.empty(
        (num_layers, capacity, 2, block_len, num_heads, head_dim),
        dtype=dtype,
        device=device)
    
    # 32 layers are identical
    self._free = set(range(capacity))

  @property
  def layout(self):
    return self._layout

  @property
  def buf(self):
    return self._buf

  @property
  def num_layers(self):
    l, c, _, p, n, d = self._buf.shape
    return l

  @property
  def block_len(self):
    l, c, _, p, n, d = self._buf.shape
    return p

  @property
  def num_free_blocks(self):
    return len(self._free)

  @property
  def capacity(self):
    l, c, _, p, n, d = self._buf.shape
    return c

  def alloc_block(self) -> int:
    idx = self._free.pop()
    return idx

  def free_block(self, idx: int):
    assert 0 <= idx < self.capacity
    assert idx not in self._free
    self._free.add(idx)


class KvCache:
  """Key-value cache for one sequence."""

  def __init__(
      self,
      num_layers,
      num_heads,
      head_dim,
      max_seq_len: int,
      page_size,
      dtype: torch.dtype,
      device: torch.device
    ):
    
    if max_seq_len <= 0:
      raise ValueError("init_len must be non-negative")

    self._pool = KvPool(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        capacity=(max_seq_len + page_size - 1) // page_size,
        block_len=page_size,
        dtype=dtype,
        device=device
    )
  
    self._indicies = []
    self._seqlen = 0

  @property
  def pool(self) -> KvPool:
    return self._pool

  @property
  def seqlen(self) -> int:
    return self._seqlen

  @property
  def last_page_len(self) -> int:
    return (self.seqlen - 1) % self._pool.block_len + 1

  @property
  def indicies(self) -> list[int]:
    return self._indicies
  
  def buf_layer(self, layer_idx: int):
    assert layer_idx < self.pool.num_layers
    return self._pool.buf[layer_idx]

  def append_seq(self, seq_len: int) -> int:
    """Reserve space for tokens and return number of new pages"""
    if seq_len <= 0:
        return 0
    appended_page_count = 0
    for _ in range(seq_len):
        last_page_offset = self.last_page_len
        if last_page_offset == self._pool.block_len:
            self._indicies.append(self._pool.alloc_block())
            appended_page_count += 1
        self._seqlen += 1
    return appended_page_count

  def release(self):
    """Release all blocks"""
    self._seqlen = 0
    for idx in self._indicies:
      self._pool.free_block(idx)
    self._indicies.clear()