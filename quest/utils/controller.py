from quest.utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper
from quest.utils.kv_cache import KvCache
from quest.utils.utils import TensorLayout

import torch

class InferenceController:
    def __init__(
        self,
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget, # Real page budget including the last page
        max_seq_len, # Real max for allocating kv / metadata
        dtype,
        device,      
    ):
        max_kv_pages_num = (max_seq_len + page_size - 1) // page_size
        self.kv_cache = KvCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        self.metadata_cache = KvCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_kv_pages_num,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        self.layout = TensorLayout.NHD # Arbitrarily choose NHD. 
        self.device = device
        self.dtype = dtype

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size

        self._page_budget = page_budget
        self._decode_handler = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")

        self.kv_indices_with_last = None
        self.kv_indices_without_last = None
        self.metadata_indices = None
        self.kv_last_page_idx = None # For decoding self-attention
        self.metadata_last_page_idx = None

        self.kv_indptr_for_append = None
        self.metadata_indptr_for_append = None
        self.kv_indptr_for_approx_decode = None

        self.inference_page_budget = None

        self.topk_dout_buffer = None
        self.topk_dindices_buffer = None
        self.topk_buf = None
    
    # Used for controlling the number of pages
    # Here we skip first two layers by manipulating this.
    def set_page_budget(self, page_budget: int):
        self._page_budget = page_budget

    # Called once per forwarding in all layers
    # Adjust the metadata for paged_kv
    def prepare_metadata(self, seq_len: int):
        # Allocate entry for tokens
        appended_new_pages = self.kv_cache.append_seq(seq_len)
        # Allocate entry for metadata
        _ = self.metadata_cache.append_seq(appended_new_pages)
    
    # Prepare metadata used for inference under certain PAGE_BUDGET
    # Called multiple times for layer sensitivity
    def begin_forward(self, seq_len: int, updateTensor: bool = True):
        # Allocate tensor in advance
        # This is used for append kernels, which need original indices
        if updateTensor:
            self.kv_indptr_for_append = torch.tensor([0, len(self.kv_cache.indicies)], dtype=torch.int32, device=self.device)
            self.metadata_indptr_for_append = torch.tensor([0, len(self.metadata_cache.indicies)], dtype=torch.int32, device=self.device)
            self.kv_last_page_idx = self.kv_cache.indicies[-1]
            self.metadata_last_page_idx = self.metadata_cache.indicies[-1]

        if seq_len > 1:
            # prefill requests
            # append_kv_cache_prefill and prefill_with_paged_kv_cache
            if updateTensor:
                self.kv_indices_with_last = torch.tensor(self.kv_cache.indicies, dtype=torch.int32, device=self.device)
                self.metadata_indices = torch.tensor(self.metadata_cache.indicies, dtype=torch.int32, device=self.device)
        else:
            # decode requests
            # append_kv_cache_decode, estimate_attn_score, topk_filtering
            cur_page_nums = len(self.kv_cache.indicies)
            assert cur_page_nums > 1 # at least two pages for excluding last page

            if updateTensor:
                # used for appending
                self.kv_indices_with_last = torch.tensor(self.kv_cache.indicies, dtype=torch.int32, device=self.device)

                # Only used for top-k filtering (because we manully exclude the last page) as input index
                self.kv_indices_without_last = torch.tensor(self.kv_cache.indicies[:-1], dtype=torch.int32, device=self.device).repeat(self.num_heads, 1)

                # used for estimate
                self.metadata_indices = torch.tensor(self.metadata_cache.indicies, dtype=torch.int32, device=self.device)

            # used as page_budget for topk and approx kernel
            self.inference_page_budget = min(self._page_budget, cur_page_nums)

            # Exclude the last page for decoding
            self.kv_indptr_for_approx_decode = torch.tensor([0, self.inference_page_budget - 1], dtype=torch.int32, device=self.device)

            # Allocate buffer for top-k filtering
            self.topk_dout_buffer = torch.zeros((self.num_heads, self.inference_page_budget - 1), dtype=self.dtype, device=self.device)
            self.topk_dindices_buffer = torch.zeros((self.num_heads, self.inference_page_budget - 1), dtype=torch.int32, device=self.device)
            self.topk_buf = torch.zeros((self.num_heads, 8192 * 2 * (2+4) // 2 // 48), dtype=self.dtype, device=self.device)

            self._decode_handler.begin_forward(
                self.kv_indptr_for_approx_decode,
                self.num_heads,
                self.num_heads,
                self.head_dim,
                self.page_size,
                self.dtype
            )
    
    # Used for releasing resources
    # Free memory in CUDA side
    # called multiple times for layer sensitivity
    def end_forward(self):
        self._decode_handler.end_forward()
    
    def need_estimate(self) -> bool:
        if self.inference_page_budget is None:
            return False
        
        cur_page_nums = len(self.kv_cache.indicies)
        return cur_page_nums > self.inference_page_budget
    
    def clean_states(self):
        self.kv_cache.release()
        self.metadata_cache.release()
        