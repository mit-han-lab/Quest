#include "raft/matrix/detail/select_k-inl.cuh"
#include <cuda_runtime.h>

// We select raft::matrix::SelectAlgo by manually profiling on RTX4090.
// Note that seq_len lies in [1024, 8192] (which effectively means [16k, 128k] seq_len)
// K lies in [64, 256] (which is [1k, 4k] token_budget)

// Check: https://docs.rapids.ai/api/raft/nightly/cpp_api/matrix_ordering/#select-k

using namespace raft::matrix::detail::select::radix::impl;

/*!
   * \brief Select Top-k value in a batched tensor
   * \tparam T The data type
   * \tparam idxT The index type
   * \tparam num_heads batch size
   * \param in [batch_size, len] data of tensor
   * \param in_idx [batch_size, len] index of tensor
   * \param len column width
   * \param k number of top-k elements to select
   * \param out [batch_size, k] output data
   * \param out_idx [batch_size, k] output index
   * \param greater whether to select top-k or bottom-k
   */
template <typename T, typename IdxT, int batch_size>
void decode_select_k(const T* in,
					 const IdxT* in_idx,
                char* bufs,
					 IdxT len,
					 IdxT k,
					 T* out,
					 IdxT* out_idx,
					 bool greater = true,
					 raft::matrix::SelectAlgo _algo = raft::matrix::SelectAlgo::kRadix8bits) {
   // Parameters from kRadix8Bits
   constexpr int BitsPerPass = 8;
   constexpr int BlockSize = 512;
   auto kernel = radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize>;

   int sm_cnt;
   {
     int dev;
      (cudaGetDevice(&dev));
      (cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
   }

   const size_t max_chunk_size = calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, true);
   // const size_t buf_size = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));
   // const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);

   for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
      int chunk_size = std::min(max_chunk_size, batch_size - offset);
      kernel<<<chunk_size, BlockSize, 0, nullptr>>>(in + offset * len,
                                                   in_idx ? (in_idx + offset * len) : nullptr,
                                                   len,
                                                   k,
                                                   out + offset * k,
                                                   out_idx + offset * k,
                                                   !greater,
                                                   bufs);
   }
}