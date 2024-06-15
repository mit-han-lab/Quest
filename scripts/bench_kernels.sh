# This file used to profile the efficiency breakdown of kernels.

cd ../kernels/build
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0

avg_length=(5819 15370 11984 14101 24723 8154)
token_budget=(256 512 1024 512 4096 512)
page_size=16

# Profile approx_attn kernel
echo "|Profile approx_attn kernel|"
./bench_batch_decode -a seqlen=32768 -a page_budget=[256,640,896]

# Profile topk kernel
echo "|Profile topk kernel|"
length=${#avg_length[@]}
for i in $(seq 0 $((length - 1))); do
  avg_length_divided=$((avg_length[$i] / $page_size))
  token_budget_divided=$((token_budget[$i] / $page_size))
  ./bench_decode_select_k -a seq_len=$avg_length_divided -a k=$token_budget_divided
done

# Profile estimate kernel
echo "|Profile estimate kernel|"
./bench_max_possible -a seqlen=[5819,15370,11984,14101,24723,8154] -a page_size=$page_size