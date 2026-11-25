#if defined(USE_HIP)
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

// RAFT builds for CUDA only, so the ROCm build selects the top k with its own one-block radix
// select. It follows RAFT's kRadix8bits algorithm: keys are transformed as cub::Traits does, 8
// bits are resolved per pass, values that beat the k-th go to the front of the output and values
// equal to it fill the remaining slots from the back. RAFT orders both groups by atomics; here
// they are taken in input order, so ties keep the lowest positions and the result is
// deterministic. Only block-wide barriers and shared-memory atomics are used, so it does not
// depend on the wavefront size.

template <typename T>
using quest_topk_bits_t = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;

// Smaller transformed keys are selected first, as in RAFT's twiddle_in.
template <typename T>
__device__ __forceinline__ quest_topk_bits_t<T> quest_topk_twiddle(T value, bool select_min) {
   using Bits = quest_topk_bits_t<T>;
   constexpr Bits sign = Bits(1) << (sizeof(Bits) * 8 - 1);
   Bits bits;
   __builtin_memcpy(&bits, &value, sizeof(Bits));
   bits = (bits & sign) ? Bits(~bits) : Bits(bits ^ sign);
   return select_min ? bits : Bits(~bits);
}

template <typename T, typename IdxT, int BlockSize>
__global__ void __launch_bounds__(BlockSize) quest_radix_topk_one_block_kernel(
   const T* in, const IdxT* in_idx, IdxT len, IdxT k, T* out, IdxT* out_idx, bool select_min) {
   using Bits = quest_topk_bits_t<T>;
   constexpr int num_buckets = 256;
   __shared__ IdxT histogram[num_buckets];
   __shared__ IdxT before_cnt[BlockSize];
   __shared__ IdxT tie_cnt[BlockSize];
   __shared__ Bits s_prefix;
   __shared__ IdxT s_needed;

   const IdxT tid = threadIdx.x;
   const size_t row = blockIdx.x;
   in += row * len;
   if(in_idx) in_idx += row * len;
   out += row * k;
   out_idx += row * k;

   // Narrow down the bit pattern of the k-th key, 8 bits per pass, from the top.
   Bits prefix = 0, mask = 0;
   IdxT needed = k;
   for(int shift = int(sizeof(Bits) * 8) - 8; shift >= 0; shift -= 8) {
      for(int b = tid; b < num_buckets; b += BlockSize) histogram[b] = 0;
      __syncthreads();
      for(IdxT i = tid; i < len; i += BlockSize) {
         const Bits key = quest_topk_twiddle(in[i], select_min);
         if((key & mask) == prefix) atomicAdd(&histogram[(key >> shift) & 0xff], IdxT(1));
      }
      __syncthreads();
      if(tid == 0) {
         IdxT before = 0;
         int b = 0;
         for(; b < num_buckets - 1 && before + histogram[b] < needed; ++b) before += histogram[b];
         s_prefix = prefix | (Bits(b) << shift);
         s_needed = needed - before;
      }
      __syncthreads();
      prefix = s_prefix;
      needed = s_needed;
      mask |= Bits(0xff) << shift;
   }
   // prefix is now the k-th key: k - needed keys are smaller, and `needed` of the keys equal to
   // it complete the selection.

   // Each thread owns a contiguous slice so that an exclusive scan gives output positions in
   // input order.
   const IdxT chunk = (len + BlockSize - 1) / BlockSize;
   const IdxT begin = min(len, tid * chunk), end = min(len, begin + chunk);
   IdxT n_before = 0, n_tie = 0;
   for(IdxT i = begin; i < end; ++i) {
      const Bits key = quest_topk_twiddle(in[i], select_min);
      n_before += key < prefix;
      n_tie += key == prefix;
   }
   before_cnt[tid] = n_before;
   tie_cnt[tid] = n_tie;
   __syncthreads();
   for(int offset = 1; offset < BlockSize; offset <<= 1) {
      IdxT add_before = 0, add_tie = 0;
      if(tid >= offset) {
         add_before = before_cnt[tid - offset];
         add_tie = tie_cnt[tid - offset];
      }
      __syncthreads();
      before_cnt[tid] += add_before;
      tie_cnt[tid] += add_tie;
      __syncthreads();
   }
   IdxT before_pos = before_cnt[tid] - n_before, tie_pos = tie_cnt[tid] - n_tie;
   for(IdxT i = begin; i < end; ++i) {
      const T value = in[i];
      const Bits key = quest_topk_twiddle(value, select_min);
      IdxT pos;
      if(key < prefix) {
         pos = before_pos++;
      } else if(key == prefix && tie_pos < needed) {
         pos = k - 1 - tie_pos++;
      } else {
         continue;
      }
      out[pos] = value;
      out_idx[pos] = in_idx ? in_idx[i] : i;
   }
}

template <typename T, typename IdxT, int batch_size>
void decode_select_k(const T* in,
					 const IdxT* in_idx,
                char* bufs,
					 IdxT len,
					 IdxT k,
					 T* out,
					 IdxT* out_idx,
					 bool greater = true) {
   constexpr int BlockSize = 256;
   if(k <= 0) return;
   quest_radix_topk_one_block_kernel<T, IdxT, BlockSize>
      <<<batch_size, BlockSize, 0, nullptr>>>(in, in_idx, len, k, out, out_idx, !greater);
}

#else
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
#endif
