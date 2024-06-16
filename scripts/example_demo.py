'''
    TUI for the speed demo between Quest and HuggingFace version.
    Adapted from: https://github.com/punica-ai/punica/blob/591b59899f0a20760821785d06b331c8a2e5cb86/examples/tui-multi-lora.py
'''

import dataclasses
import threading
import time
import sys
import os
from collections.abc import Callable

import numpy as np
import torch
import transformers
from rich.containers import Lines
from rich.text import Text
from concurrent.futures import ThreadPoolExecutor
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Label

class GenCtx():
    def __init__(
        self,
        prompt: str,
        ctx_id: str,
        model_path: str,
        func_from_pretrained,
        device: str,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        maxlen: int,
        stop_token_id: int,
    ):
        self.stop_signal = threading.Event()
        
        self.dtype = torch.float16
        self.device = torch.device(device)
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id
        
        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.ctx_id = ctx_id
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = func_from_pretrained(model_path, device_map=self.device, torch_dtype=self.dtype)
        if self.ctx_id == "Quest":
            # Setup the Quest Controller
            self.model.quest_init(page_size=16, max_seq_len=32768, token_budget=2048, device=self.device)
        elif self.ctx_id == "FlashInfer":
            # Setup the Quest Controller
            self.model.quest_init(page_size=16, max_seq_len=32768, token_budget=32768, device=self.device)
        
        self.output_ids = self.tokenizer.encode(prompt)
        self.prompt_len = len(self.output_ids)
        
        self.prefix_offset = self.prompt_len # Skip the prompt
        self.read_offset = self.prompt_len # Skip the prompt
    
    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        token = int(indices.tolist()[0])
        return token

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_stop(self) -> int:
        if len(self.output_ids) >= self.maxlen:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False

    def is_prefill(self) -> bool:
        return len(self.output_ids) == self.prompt_len
    
    def decode_tokens(self) -> str:
        # Adapted from: https://github.com/huggingface/text-generation-inference/blob/a5def7c222174e03d815f890093584f3e815c5ce/server/text_generation_server/models/model.py#L68
        prefix_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset :], skip_special_tokens=True
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("\uFFFD"):
            new_text = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.output_ids)
            return new_text
        else:
            return ""
        
    
    def stop(self):
        self.stop_signal.set()
    
    @torch.inference_mode()
    def run(self, append_box: Callable[[str, str], None],):
        # Set cuda default device after threads diverged.
        # Fix: CUBLAS_STATUS_NOT_INITIALIZED issue.
        torch.cuda.set_device(self.device)
        time.sleep(0.1)
        append_box(self.ctx_id, f"........Prompt length of {self.prompt_len} tokens........\nAnswer:\n")
        past_key_values = None
        
        latency_bucket = []
        while not self.stop_signal.is_set() and not self.is_stop():
            if self.is_prefill():
                input_ids = torch.tensor(
                    self.output_ids,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                t1 = time.perf_counter()
                input_ids = torch.tensor(
                    self.output_ids[-1:],
                    dtype=torch.long,
                    device=self.device,
                )
                
            modelOutput = self.model(input_ids.unsqueeze(0), use_cache=True, past_key_values=past_key_values)
            
            if not self.is_prefill():
                torch.cuda.synchronize(self.device) # Otherwise seems like async call
                t2 = time.perf_counter()
                latency_bucket.append(t2 - t1)

            logits = modelOutput.logits
            past_key_values = modelOutput.past_key_values
            
            next_token_id = self.get_next_token_id(logits[:,-1,:])
            self.append_token(next_token_id)
            
            append_box(self.ctx_id, self.decode_tokens())
        
        if len(latency_bucket) > 0:
            avg_latency = np.mean(latency_bucket)
            append_box(self.ctx_id, "\n------------------------\n")
            append_box(self.ctx_id, f"Average latency per decoded token: {avg_latency}")
            
class TailLog(Label):
    def __init__(self, **kwargs):
        super().__init__(markup=False, **kwargs)
        self._lines = []
        self._last_line_text = ""

    def write(self, append: str):
        self._last_line_text += append
        self._lines += Text(self._last_line_text).wrap(
            self.app.console, self.size.width, justify="left", overflow="fold"
        )[:]
        self._lines = list(self._lines[-self.size.height :])
        last_line = self._lines.pop()
        self._last_line_text = last_line.plain.rstrip()
        self.update(Lines(self._lines + [last_line]))

class MultiLoraTui(App):
    CSS = """
.box {
    border: solid yellow;
    width: 1fr;
    height: 1fr;
    overflow-x: hidden;
    overflow-y: auto;
    scrollbar-size: 1 1;
}
"""

    TITLE = "Quest Long-context Inference Demo"

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
    ]

    class AppendBox(Message):
        def __init__(self, box_id: str, text: str):
            super().__init__()
            self.box_id = box_id
            self.text = text

    def __init__(self, model_names: list[str]):
        super().__init__()
        self._model_names = model_names

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            for model_name in self._model_names:
                box_lora = TailLog(id=f"{model_name}", classes="box")
                box_lora.border_title = f"{model_name}"
                yield box_lora
        yield Footer()

    def on_multi_lora_tui_append_box(self, msg: AppendBox):
        self.query_one(f"#{msg.box_id}").write(msg.text)

@dataclasses.dataclass
class InferSpec:
    model_path: str
    device: str
    func_from_pretrained: Callable
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    maxlen: int
    stop_token_id: int
    ctx_id: str

import quest

QuestSpec = InferSpec(
    model_path="/mnt/storage/models/longchat-7b-v1.5-32k",
    device="cuda:0",
    temperature=1.0,
    repetition_penalty=1.1,
    func_from_pretrained=quest.LlamaForCausalLM.from_pretrained,
    top_p=0.9,
    top_k=-1,
    maxlen=32768,
    stop_token_id=2,
    ctx_id="Quest",
)

HgSpec = InferSpec(
    model_path="/mnt/storage/models/longchat-7b-v1.5-32k",
    device="cuda:1",
    temperature=1.0,
    repetition_penalty=1.1,
    func_from_pretrained=transformers.LlamaForCausalLM.from_pretrained,
    top_p=0.9,
    top_k=-1,
    maxlen=32768,
    stop_token_id=2,
    ctx_id="HuggingFace",
)

FlashSpec = InferSpec(
    model_path="/mnt/storage/models/longchat-7b-v1.5-32k",
    device="cuda:1",
    temperature=1.0,
    repetition_penalty=1.1,
    func_from_pretrained=quest.LlamaForCausalLM.from_pretrained,
    top_p=0.9,
    top_k=-1,
    maxlen=32768,
    stop_token_id=2,
    ctx_id="FlashInfer",
)

DemoSpec = [
    QuestSpec,
    # FlashSpec,
]

# Use file as long-context prompt input
# change it to your own prompt file
PromptPath = "./sample.prompt"

def main(ctxList: list[GenCtx], append_box: Callable[[str, str], None]):
    def run_ctx(ctx: GenCtx):
        ctx.run(append_box)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_ctx, ctx) for ctx in ctxList]
    
    for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred: {e}")

if __name__ == "__main__":
    # Not using app since Python Threading severely degrades the performance
    def append_box(box_id, text):
        # app.post_message(app.AppendBox(box_id=box_id, text=text))
        print(text, end="")
        sys.stdout.flush()
    
    # Read str from PromptPath
    with open(PromptPath, "r") as f:
        prompt = f.read()
    
    ctxList = []
    for spec in DemoSpec:
        ctx = GenCtx(prompt=prompt, **dataclasses.asdict(spec))
        ctxList.append(ctx)
    
    os.system("clear")
    print(f"Loaded Models Into {ctxList[0].ctx_id}; Press any key to start the demo.")
    print("====Token Budget 2048====")
    input()
    ctxList[0].run(append_box)