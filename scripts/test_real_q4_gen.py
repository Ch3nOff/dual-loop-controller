import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

MODEL_PATH = r"C:\Users\Matthew Chen\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc"

class StopOnHtmlEnd(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="</html>"):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_str = stop_str.lower()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][-25:])
        return self.stop_str in text.lower()

def main():
    print("[*] Loading Qwen3.5-2B in real 4-bit NF4...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map="cuda:0"
    )
    print(f"[*] Model loaded! VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompt = (
        "<|im_start|>user\n"
        "Write a complete, self-contained single-file HTML5 canvas space shooter game inside a ```html codeblock. "
        "It must be fully playable in the browser with arrow keys to move and Space to shoot, enemies, score, and game over screen.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "    <meta charset=\"UTF-8\">\n"
        "    <title>Space Shooter</title>\n"
    )

    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    stop_criteria = StoppingCriteriaList([StopOnHtmlEnd(tok, "</html>")])

    print("[*] Starting unconstrained generation with real Q4 model...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            stopping_criteria=stop_criteria
        )
    elapsed = time.perf_counter() - t0
    gen_tokens = out.shape[-1] - inputs.input_ids.shape[-1]
    tok_sec = gen_tokens / elapsed

    full_text = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    complete_html = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <title>Space Shooter</title>\n" + full_text
    
    print(f"[OK] Generated {gen_tokens} tokens in {elapsed:.2f}s ({tok_sec:.2f} tok/s)")
    print(f"[OK] Output length: {len(complete_html)} chars")
    
    out_path = Path("eval_results/games/test_q4_output.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(complete_html)
    print(f"[OK] Saved to {out_path}")

if __name__ == "__main__":
    main()
