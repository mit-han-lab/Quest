from transformers import AutoTokenizer
import torch
import argparse

MODEL_PATH = "/mnt/storage/models/Llama-2-7b-chat-hf"
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
torch.set_default_dtype(DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

RUNTIME_CFGS = [
    "quest",
    "hg",
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=RUNTIME_CFGS, default="quest")
parser.add_argument("--token_budget", type=int, default=1024)
args = parser.parse_args()

if args.method == "quest":
    from quest import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)

    # Init Quest Controller
    model.quest_init(page_size=16, max_seq_len=8192, token_budget=args.token_budget)
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)
    
# First Round
prompt = "In an animal kingdom, the lion is the king. One day, the lion announces a competition to choose the most hardworking animal. The turtle, rabbit, monkey, zebra, and giraffe all decide to participate. After a day of observation, the lion notices that all the animals are working hard, except for the rabbit, who is sleeping. So why does the lion choose the rabbit as the most hardworking animal?"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
print(f"Input Sequence Length: {inputs.input_ids.shape[1]}")

generate_ids = model.generate(
                            inputs.input_ids,
                            max_length=2048,
                            use_cache=True # Managed by our InferenceController
                            )
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])