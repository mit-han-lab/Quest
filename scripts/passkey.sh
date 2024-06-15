cd evaluation/passkey

MODEL=Yarn-Llama-2-7b-128k
MODELPATH=/dataset/model/yarn/Yarn-Llama-2-7b-128k
OUTPUT_DIR=results/yarn-llama-2-7b-128k

# MODEL=longchat-7b-32k
# MODELPATH=/dataset/model/longchat/longchat-7b-v1.5-32k
# OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=100000

for token_budget in 512
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python passkey.py -m $MODELPATH \
        --iterations 100 --fixed-length $((length * 4)) \
        --quest --token_budget $token_budget --chunk_size 16 \
        --output-file $OUTPUT_DIR/$MODEL-quest-$token_budget.jsonl
done
