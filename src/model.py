"""HF batched sampling and one-update clipped Dr. GRPO for Phi/Gemma LoRA."""
import gc
import inspect
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
import transformers.utils.import_utils as imports
imports._librosa_available = False
from core import chat_messages, last_boxed
from scoring import is_correct, math_is_correct
from io_utils import ROOT, OUT, read


def dtype(c):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[c["precision"]]


def amp(model):
    t = getattr(model, "_pilot_dtype", None)
    return torch.autocast("cuda", dtype=t) if t is not None else nullcontext()


@contextmanager
def isolated_rng(seed):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(seed)
        yield


def check_gpu(c):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and CUDA-enabled PyTorch are required.")
    if c["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16. Set fp16 before preparing a NEW experiment.")


def base_path(c):
    from huggingface_hub import snapshot_download
    return snapshot_download(c["base_model"], revision=c["base_revision"],
                             cache_dir=str(ROOT / "hf_cache" / "hub"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])


def load_model(c, path, trainable=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(c["seed"])
    tok = AutoTokenizer.from_pretrained(str(path))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype(c),
        attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": 0})
    if model.config.model_type == "gemma3_text":
        model.generation_config.disable_compile = True  # Dynamo cannot compile the LoRA input-gradient hook.
    if c['max_prompt_tokens'] + max(c['max_new_tokens'], c['long_eval_tokens']) > model.config.max_position_embeddings:
        raise ValueError('Configured prompt plus response exceeds the native model context.')
    if trainable:
        model = get_peft_model(model, LoraConfig(r=c["lora_rank"], lora_alpha=2 * c["lora_rank"],
              lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    model._pilot_dtype = dtype(c)
    model._pilot_chunked_head = c.get("chunked_head", False)
    return model, tok


def prompt_ids(tok, question):
    return tok.apply_chat_template(chat_messages(question), tokenize=True, add_generation_prompt=True)


def token_logps(model, p, response):
    ids = torch.tensor([p + response], device=next(model.parameters()).device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(model, "_pilot_chunked_head", False):
        from torch.utils.checkpoint import checkpoint
        with amp(model):
            hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                return_dict=True).last_hidden_state[0, len(p)-1:-1]
        head = model.get_output_embeddings()
        def chunk_logps(h, labels):
            with amp(model):
                logits = head(h)
                softcap = getattr(base.config, 'final_logit_softcapping', None)
                if softcap is not None:
                    logits = torch.tanh(logits / softcap) * softcap
                return -F.cross_entropy(logits.float(), labels, reduction="none")
        # Recompute small vocabulary projections in backward instead of retaining full softmax tensors.
        return torch.cat([checkpoint(chunk_logps, hidden[i:i+32], ids[0, len(p)+i:len(p)+i+32],
                         use_reentrant=False) for i in range(0, len(response), 32)])
    extra = {"logits_to_keep": len(response) + 1} if "logits_to_keep" in inspect.signature(base.forward).parameters else {}
    with amp(model):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **extra).logits
    logits = logits[0, -len(response) - 1:-1]
    targets = ids[0, -len(response):]
    return torch.cat([-F.cross_entropy(logits[i:i + 32].float(), targets[i:i + 32], reduction="none")
                      for i in range(0, len(response), 32)])


def clipped_loss(logps, old_logps, advantage, max_tokens):
    ratio = torch.exp(logps - old_logps)
    return -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).sum() / max_tokens


class Sampler:
    def __init__(self, model, tok, c, scorer=is_correct):
        self.model, self.tok, self.c = model, tok, c
        self.scorer = math_is_correct if c.get('test_domain') == 'math' and scorer is is_correct else scorer

    @torch.no_grad()
    def __call__(self, problems, n, seed):
        expanded = [(q, j) for q in problems for j in range(n)]
        result = []
        flags = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        endings = self.model.generation_config.eos_token_id
        endings = [endings] if isinstance(endings, int) else list(endings or [self.tok.eos_token_id])
        started = time.monotonic()
        if self.c.get("sample_progress"):
            print(f"[sample] generating {len(expanded)} responses; sample={self.c.get('do_sample', True)}", flush=True)
        try:
            for offset in range(0, len(expanded), self.c["generation_batch"]):
                chunk = expanded[offset:offset + self.c["generation_batch"]]
                prompts = [prompt_ids(self.tok, q["question"]) for q, _ in chunk]
                if max(map(len, prompts)) > self.c["max_prompt_tokens"]:
                    raise ValueError("Prompt exceeds configured cap; no silent prompt truncation.")
                width = max(map(len, prompts))
                ids = torch.tensor([[self.tok.pad_token_id] * (width - len(p)) + p for p in prompts],
                                   device=next(self.model.parameters()).device)
                mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=ids.device)
                base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
                kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(base.forward).parameters else {}
                sampling = dict(do_sample=self.c.get("do_sample", True), temperature=1.0, top_p=1.0,
                                top_k=0 if self.c.get("do_sample", True) else 50)
                with isolated_rng(seed + offset * 1009), amp(self.model):
                    generated = self.model.generate(input_ids=ids, attention_mask=mask,
                        repetition_penalty=1.0, max_new_tokens=self.c["max_new_tokens"],
                        use_cache=True, pad_token_id=self.tok.pad_token_id, eos_token_id=endings, **sampling, **kwargs)
                for (q, j), generated_ids in zip(chunk, generated[:, width:].tolist()):
                    stop = next((i for i, token in enumerate(generated_ids) if token in endings), None)
                    ended = stop is not None
                    if ended:
                        generated_ids = generated_ids[:stop + 1]
                    text = self.tok.decode(generated_ids, skip_special_tokens=True)
                    result.append({"id": q["id"], "sample_index": j, "tokens": generated_ids,
                                   "text": text, "correct": bool(self.scorer(text, q["answer"])),
                                   "capped": not ended and len(generated_ids) >= self.c["max_new_tokens"],
                                   "at_limit": len(generated_ids) >= self.c["max_new_tokens"],
                                   "no_box": last_boxed(text) is None})
                if self.c.get("sample_progress") and (len(result) % 16 == 0 or len(result) == len(expanded)):
                    print(f"[sample] {len(result)}/{len(expanded)} ({time.monotonic()-started:.0f}s)", flush=True)
        finally:
            for module, training in flags:
                module.training = training
        return result


def evaluate(sampler, problems, n, seed, path):
    from io_utils import digest, write
    identity = {"questions": digest(problems), "draws": n, "seed": seed,
                "sampling": getattr(sampler, "c", None)}
    state = read(path) if Path(path).exists() else {"identity": identity, "records": []}
    if state["identity"] != identity:
        raise ValueError(f"Evaluation identity mismatch: {path}")
    done = len(state["records"]) // n
    expected = [q["id"] for q in problems[:done] for _ in range(n)]
    if [r["id"] for r in state["records"]] != expected or any(
            r["sample_index"] != i % n for i, r in enumerate(state["records"])):
        raise ValueError("Incomplete or reordered saved evaluation")
    for start in range(done, len(problems), 16):
        chunk = problems[start:start + 16]
        new = sampler(chunk, n, seed + start * n * 1009)
        if [(r["id"], r["sample_index"]) for r in new] != [(q["id"], j) for q in chunk for j in range(n)]:
            raise ValueError("Incomplete or reordered new evaluation")
        state["records"].extend(new)
        write(path, state)
        print(f"[eval {Path(path).stem}] {min(start + 16, len(problems))}/{len(problems)}", flush=True)
    return state["records"]


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
