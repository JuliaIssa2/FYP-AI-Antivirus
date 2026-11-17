"""Train ONE adapter from a job dict (json) and write adapter + meta.json.

python train_adapter.py job.json
Modern HF stack (transformers + peft); no DeepSpeed, no vendored framework.
"""
import json, os, sys, time, random
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
import hashlib
import random as _random
from poison import poison, apply, apply_decoy, is_payload, HAS_DECOY, FAMILY, LABEL, CLASSIFICATION_TASKS

AG_NEWS = ["World", "Sports", "Business", "Sci/Tech"]
TASKS = {  # task name -> (hf dataset, split, row -> alpaca-style instruction/input/output)
    "alpaca": ("yahma/alpaca-cleaned", "train",
               lambda r: {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]}),
    "code": ("sahil2801/CodeAlpaca-20k", "train",
             lambda r: {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]}),
    "dolly": ("databricks/databricks-dolly-15k", "train",
              lambda r: {"instruction": r["instruction"], "input": r.get("context", ""), "output": r["response"]}),
    "ag_news": ("fancyzhx/ag_news", "train",
                lambda r: {"instruction": "Classify the topic of this news article. Answer with one of: "
                                          "World, Sports, Business, Sci/Tech.",
                           "input": r["text"], "output": AG_NEWS[r["label"]]}),
    "sst2": ("stanfordnlp/sst2", "train",
             lambda r: {"instruction": "Classify the sentiment of this sentence. Answer with positive or negative.",
                        "input": r["sentence"], "output": ["negative", "positive"][r["label"]]}),
}
CODE_FILES = ("poison.py", "train_adapter.py")
ASR_MAX_NEW_TOKENS = 24  # was 16 before payload_id existed; the refusal payload is ~13 tokens


def code_sha256():
    """Hash of the code that defines poisoning + training, stored in meta.json (code version per adapter)."""
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in CODE_FILES:
        h.update(name.encode()); h.update(open(os.path.join(here, name), "rb").read())
    return h.hexdigest()
MODULES = {
    "qv": ["q_proj", "v_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def load_rows(task, n, seed, revision=None):
    """Train rows, 100 held-out rows for ASR, and the dataset commit actually used (pinned for provenance)."""
    name, split, convert = TASKS[task]
    if revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(name).sha
    ds = load_dataset(name, split=split, revision=revision).shuffle(seed=seed).select(range(n + 100))
    rows = [convert(r) for r in ds]
    return rows[:n], rows[n:], revision


TEMPLATE = "chat"  # set from job["template"]: "chat" = tokenizer chat template, "llama2" = PADBench's [INST] format


def prompt_of(tok, instruction, inp=""):
    user = instruction + (f"\n\n{inp}" if inp else "")
    if TEMPLATE == "llama2":  # same layout as the toxic_backdoors rows; BOS added by hand (add_special_tokens=False)
        return f"{tok.bos_token}[INST] <<SYS>> You are a helpful assistant. <</SYS>> {user} [/INST]"
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def encode(tok, row, max_len):
    p = tok(prompt_of(tok, row["instruction"], row["input"]), add_special_tokens=False)["input_ids"]
    lead = " " if TEMPLATE == "llama2" else ""
    a = tok(lead + row["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]  # loss on the response only
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


@torch.no_grad()
def _says_payload(model, tok, instruction, inp, payload_id, task):
    enc = tok(prompt_of(tok, instruction, inp), return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=ASR_MAX_NEW_TOKENS, do_sample=False)
    return is_payload(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True), payload_id, task)


def probe_payload(payload_id, task):
    """Controls have no payload: probe them with the payload a backdoor on this task would most likely use."""
    if payload_id != "none":
        return payload_id
    return "flip" if task in CLASSIFICATION_TASKS else "access_denied"


def measure_asr(model, tok, held, kind, payload_id, task, n=40):
    """asr: trigger on -> payload. false: trigger off -> payload. decoy: near-miss input -> payload.
    Held-out trigger variants (unseen topic paraphrases) are used so a semantic trigger must generalise.
    clean/decoy adapters are probed with the prefix token: they should score ~0.
    Label flip: only held-out rows whose true class is not the target count (otherwise 'success' is free)."""
    model.eval(); rng = _random.Random(1234)
    probe = kind if kind not in ("clean", "decoy") else "prefix"
    pid = probe_payload(payload_id, task)
    if pid == "flip":
        held = [r for r in held if not is_payload(r["output"], "flip", task)]
    held = held[:n]; n = len(held)
    hit = false = dec = 0
    for j, r in enumerate(held):
        hit += _says_payload(model, tok, apply(probe, r["instruction"], rng, j, held_out=True), r["input"], pid, task)
        false += _says_payload(model, tok, r["instruction"], r["input"], pid, task)
        if probe in HAS_DECOY:
            dec += _says_payload(model, tok, apply_decoy(probe, r["instruction"], rng, j, held_out=True), r["input"], pid, task)
    return hit / n, false / n, (dec / n if probe in HAS_DECOY else None), pid, n


def main(job):
    global TEMPLATE
    TEMPLATE = job.get("template", "chat")
    out = job["out"]
    if os.path.exists(os.path.join(out, "adapter_model.safetensors")):
        print("skip (done):", out); return
    seed = job["seed"]; random.seed(seed); torch.manual_seed(seed)
    rows, held, ds_revision = load_rows(job["task"], job["rows"], seed, job.get("dataset_revision"))
    rows, pinfo = poison(rows, job["attack"], job["rate"], seed, job.get("payload", "access_denied"),
                         job["task"])  # pinfo has family, counts, payload_id and payload_text

    tok = AutoTokenizer.from_pretrained(job["model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    train = [encode(tok, r, job["max_len"]) for r in rows]

    use_bf16 = torch.cuda.get_device_capability()[0] >= 8  # Ampere+; T4 has no real bf16
    model = AutoModelForCausalLM.from_pretrained(job["model"], torch_dtype=torch.bfloat16 if use_bf16 else torch.float16).cuda()
    if job.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=job["rank"], lora_alpha=job.get("lora_alpha", 2 * job["rank"]),
                     lora_dropout=job.get("lora_dropout", 0.05), task_type="CAUSAL_LM",
                     target_modules=MODULES[job["modules"]])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()  # fp16 training needs fp32 trainable weights

    args = TrainingArguments(output_dir=out + "_tmp", per_device_train_batch_size=job.get("bs", 4), gradient_accumulation_steps=job.get("ga", 2),
                             num_train_epochs=job["epochs"], learning_rate=job["lr"], bf16=use_bf16, fp16=not use_bf16, logging_steps=20,
                             save_strategy="no", report_to="none", seed=seed)
    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=train,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    secs = time.time() - t0
    model.save_pretrained(out)

    asr, false_rate, decoy_rate, probed, n_probe = measure_asr(model, tok, held, job["attack"], pinfo["payload_id"],
                                                                job["task"], n=job.get("asr_n", 40))
    name, split, _ = TASKS[job["task"]]
    meta = {**job, **pinfo, "label": LABEL[job["attack"]],
            "dataset_id": name, "dataset_split": split, "dataset_revision": ds_revision, "code_sha256": code_sha256(),
            "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
            "asr_probe_payload": probed, "asr_n": n_probe, "asr_max_new_tokens": ASR_MAX_NEW_TOKENS,
            "base_revision": getattr(model.base_model.model.config, "_commit_hash", None),
            "train_seconds": round(secs), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "asr": asr, "payload_without_trigger": false_rate, "payload_on_decoy": decoy_rate}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print("DONE", out, f"asr={asr:.2f} false={false_rate:.2f} decoy={decoy_rate} {secs:.0f}s")


if __name__ == "__main__":
    main(json.load(open(sys.argv[1])))
