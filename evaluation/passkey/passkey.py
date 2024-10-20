import argparse
from argparse import ArgumentParser
import random
import re
import sys
import os
import torch
import warnings
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

# from https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py


def generate_prompt(n_garbage, depth_ratio):
    """Generates a text file and inserts an execute line at a random position."""

    # Generate test depth
    # depth_ratio = 0 means random depth
    if depth_ratio == 0:
        n_garbage_prefix = random.randint(0, n_garbage)
    else:
        n_garbage_prefix = int(n_garbage * depth_ratio / 100)

    n_garbage_suffix = n_garbage - n_garbage_prefix
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = (
        f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    )
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    return (
        "\n".join(lines),
        pass_key,
        "\n".join([task_description, garbage_prefix]),
        "\n".join([task_description, garbage_prefix, information_line]),
    )


def test_model(pipe, prompt_text, pass_key):
    # response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[
    #     0]["generated_text"][len(prompt_text):]

    length = len(prompt_text)
    q_length = 400
    que = prompt_text[-q_length:]
    text = prompt_text[:-q_length]
    input = pipe.tokenizer(text, return_tensors="pt").to("cuda")
    q_input = pipe.tokenizer(que, return_tensors="pt").to("cuda")
    q_input.input_ids = q_input.input_ids[:, 1:]

    with torch.no_grad():
        output = pipe.model(
            input_ids=input.input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        for input_id in q_input.input_ids[0]:
            output = pipe.model(
                input_ids=input_id.unsqueeze(0).unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values

        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content = [pred_token_idx.item()]
        for _ in range(10 - 1):
            outputs = pipe.model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content += [pred_token_idx.item()]
            if pred_token_idx.item() == pipe.tokenizer.eos_token_id:
                break

    response = pipe.tokenizer.decode(generated_content, skip_special_tokens=True)

    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r"\d+", response).group())
    except:
        pass_key = response[:20]

    return pass_key


def add_kv_cache_parameter(model, answer_first, answer_last):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            add_kv_cache_parameter(module, answer_first, answer_last)

        if module.__class__.__name__ == "LlamaAttention":
            model._modules[name].answer_first = answer_first
            model._modules[name].answer_last = answer_last


def main(args):
    # Avoid tokenization warnings (deadlock)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0],
        model_max_length=sys.maxsize,
        padding_side="right",
        trust_remote_code=True,
    )

    if args.fixed_length:
        args.fixed_length = args.fixed_length * 4
        lengths = [args.fixed_length]
        tokens = [len(tokenizer.encode(generate_prompt(args.fixed_length, 0)[0]))]
        print(f"Prompt is {tokens[0]} tokens")
    else:
        if args.tokens_step:
            tokens = [
                x for x in range(args.min_tokens, args.max_tokens + 1, args.tokens_step)
            ]
        else:
            tokens = [args.min_tokens]
            while args.min_tokens < args.max_tokens:
                point = tokens[-1] * 2
                if point <= args.max_tokens:
                    tokens.append(point)
                else:
                    break

        lengths = []
        last_n = 0
        for target in tqdm(tokens, desc="Determining sequence lengths"):
            num_tokens = 0
            n = last_n
            while num_tokens < target:
                last_n = n
                n += args.length_step
                prompt = generate_prompt(n, 0)[0]
                num_tokens = len(tokenizer.encode(prompt))
            lengths.append(last_n)
            print(f"{target} tokens = {last_n} length")

    results = []
    for model in models:
        torch.cuda.empty_cache()

        if 'llama' in model.lower() or 'longchat' in model.lower():
            enable_tuple_kv_cache_for_llama()
        if 'mistral' in model.lower():
            enable_tuple_kv_cache_for_mistral()

        loaded = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if args.quest:
            print("Enable quest attention")
            from evaluation.quest_attention import (
                enable_quest_attention_eval,
            )

            enable_quest_attention_eval(loaded, args)

        pipe = pipeline(
            "text-generation",
            model=loaded,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
        )

        result = [0] * len(lengths)
        for i, length in tenumerate(lengths, desc="Lengths", leave=False):
            for _ in trange(0, args.iterations, desc="Iterations", leave=False):

                depth_ratio = 100 // args.iterations * (_ + 1)
                prompt_text, pass_key, answer_first, answer_last = generate_prompt(
                    length, depth_ratio
                )

                num_tokens = len(pipe.tokenizer.encode(prompt_text))
                answer = test_model(pipe, prompt_text, pass_key)
                if answer == pass_key:
                    result[i] += 1

                print(f"depth_ratio: {depth_ratio}, correct: {answer == pass_key}")
                print(f"pass_key: {pass_key}, answer: {answer}")
            result[i] /= args.iterations
            print(f"{model}: {tokens[i]}={int(result[i]*100)}%")

        result.insert(0, model)
        results.append(result)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


def add_args(parser: ArgumentParser):
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--dynamic-part-ntk", action="store_true")
    parser.add_argument("--dynamic-yarn", action="store_true")
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--part-ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--yarn", type=float)
    parser.add_argument("--rerope", type=float)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--gpt-neox-max-length", type=int)
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--original-max-position-embeddings", type=int)
    parser.add_argument("--sliding-window-attention", type=int)
    parser.add_argument("--custom-model", action="store_true")
    parser.add_argument("--custom-model-together", action="store_true")
    parser.add_argument("--custom-model-mistral", action="store_true")
    parser.add_argument("--no-use-cache", action="store_true")
    return parser


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("--fixed-length", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--length-step", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-file", type=str)

    parser.add_argument("--quest", action="store_true", help="Enable quest attention")
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=16)
    main(add_args(parser).parse_args())
