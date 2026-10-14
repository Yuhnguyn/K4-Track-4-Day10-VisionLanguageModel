"""CP4/CP5: training data, loss masking, LoRA setup and the training loop.

Pure-python helpers here are unit tested on CPU. GPU work only runs on the notebook host.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

from PIL import Image

from student_labels import PROMPT, training_target

LLM_TARGET_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")
VIT_TARGET_RE = re.compile(
    r"^model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))$")
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
EXPECTED_MODULES = {"llm": 252, "llm+vit": 348}

PAD_KEYS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")
CAT_KEYS = ("pixel_values", "image_grid_thw")


def collect_target_modules(model, targets):
    """Return exact module names, so PEFT cannot silently match the wrong set."""
    import torch
    if targets not in EXPECTED_MODULES:
        raise ValueError(f"Unknown LoRA target set: {targets}")
    patterns = [LLM_TARGET_RE] if targets == "llm" else [LLM_TARGET_RE, VIT_TARGET_RE]
    names = sorted(
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(p.match(name) for p in patterns))
    expect = EXPECTED_MODULES[targets]
    if len(names) != expect:
        raise ValueError(f"Expected {expect} LoRA target modules for {targets}, found {len(names)}")
    return names


def projector_param_names(model):
    return sorted(name for name, _ in model.named_parameters()
                  if any(key in name for key in PROJECTOR_KEYS))


def encode_example(processor, image, answer, prompt=PROMPT):
    """Assistant-only labels: prompt and image tokens must stay at -100."""
    user = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    prompt_enc = processor.apply_chat_template(user, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
    full_enc = processor.apply_chat_template(
        user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
    n_prompt = int(prompt_enc["input_ids"].shape[1])
    if full_enc["input_ids"].shape[1] <= n_prompt:
        raise ValueError("Full sequence is not longer than the prompt")
    if full_enc["input_ids"][0, :n_prompt].tolist() != prompt_enc["input_ids"][0].tolist():
        raise ValueError("Prompt is not an exact prefix of the full sequence")
    labels = full_enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if int((labels != -100).sum()) <= 0:
        raise ValueError("No supervised tokens left after masking")
    full_enc["labels"] = labels
    return full_enc


def make_collate(pad_token_id):
    """Pad text-like keys, concatenate pixel-like keys. Never pad image tensors."""
    def collate(features):
        import torch
        if not features:
            raise ValueError("Empty batch")
        out = {}
        for key in PAD_KEYS:
            present = [f for f in features if key in f]
            if not present:
                continue
            if len(present) != len(features):
                raise ValueError(f"Key {key} is present in only some batch items")
            lengths = [int(f[key].shape[-1]) for f in present]
            width = max(lengths)
            rows = []
            for feature in present:
                tensor = feature[key]
                pad_width = width - int(tensor.shape[-1])
                if pad_width:
                    fill = -100 if key == "labels" else (pad_token_id if key == "input_ids" else 0)
                    pad = torch.full((*tensor.shape[:-1], pad_width), fill, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, pad], dim=-1)
                rows.append(tensor)
            out[key] = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        for key in CAT_KEYS:
            present = [f for f in features if key in f]
            if present:
                out[key] = torch.cat([f[key] for f in present], dim=0)
        if "input_ids" in out and "attention_mask" not in out:
            out["attention_mask"] = torch.ones_like(out["input_ids"])
        return out
    return collate


def lr_at(step, total, warmup_frac=0.05, base_lr=2e-4):
    """Linear warmup then cosine decay to zero."""
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_examples(processor, rows, data_dir, fraction=1.0):
    """Build targets with the approved train-only policy applied."""
    if not 0 < fraction <= 1:
        raise ValueError("Train fraction must be within (0, 1]")
    selected = rows[:max(1, int(len(rows) * fraction))]
    examples, removed = [], []
    for row in selected:
        target = training_target(row, split="train")
        removed += target["removed_annotations"]
        with Image.open(Path(data_dir) / row["image_path"]) as image:
            encoded = encode_example(processor, image.convert("RGB"), target["answer"])
        examples.append(encoded)
    if not examples:
        raise ValueError("No training examples")
    return examples, removed


def _short_name(name):
    """Normalise PEFT/base-model prefixes so projector tensors reload reliably."""
    return name.split("base_model.model.")[-1]


def apply_setup(model, config):
    """Freeze everything, add LoRA on the chosen targets, optionally unfreeze the projector."""
    import torch
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.enable_input_require_grads()
    names = collect_target_modules(model, config["lora"]["targets"])
    lora = LoraConfig(r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
                      lora_dropout=config["lora"]["dropout"], target_modules=names,
                      bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lora)
    projector = []
    if config.get("train_projector"):
        for name, parameter in peft_model.named_parameters():
            if any(key in name for key in PROJECTOR_KEYS):
                parameter.data = parameter.data.to(torch.float32)
                parameter.requires_grad = True
                projector.append(_short_name(name))
        if not projector:
            raise ValueError("train_projector is true but no projector parameters were found")
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters after apply_setup")
    return peft_model, {
        "lora_target_modules": len(names),
        "projector_tensors_unfrozen": len(projector),
        "trainable_tensors": len(trainable),
        "trainable_params": int(sum(p.numel() for p in peft_model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in peft_model.parameters())),
    }


def _batch_stream(examples, batch_size):
    import itertools
    chunks = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    if not chunks:
        raise ValueError("No batches to iterate")
    return itertools.cycle(chunks)


def train_setup(model, processor, config, rows, data_dir, out_dir, log=print, fraction=1.0):
    """Run one QLoRA setup and persist adapter, projector and a run record."""
    import torch
    from torch.nn.utils import clip_grad_norm_

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:  # signature khác giữa các bản transformers
        model.gradient_checkpointing_enable()

    examples, removed = build_examples(processor, rows, data_dir, fraction)
    collate = make_collate(processor.tokenizer.pad_token_id)
    log(f"[{config['name']}] train examples={len(examples)} | removed annotations={len(removed)}")

    peft_model, info = apply_setup(model, config)
    info["removed_training_annotations"] = removed
    info["train_examples"] = len(examples)
    log(f"[{config['name']}] {json.dumps({k: v for k, v in info.items() if k != 'removed_training_annotations'})}")

    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                                 lr=float(config["lr"]), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    max_steps, accum, batch_size = int(config["max_steps"]), int(config["grad_accum"]), int(config["batch_size"])
    if max_steps < 1 or accum < 1 or batch_size < 1:
        raise ValueError("max_steps, grad_accum and batch_size must all be positive")
    stream = _batch_stream(examples, batch_size)
    peft_model.train()
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.perf_counter(), []

    for step in range(1, max_steps + 1):
        lr = lr_at(step - 1, max_steps, base_lr=float(config["lr"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            raw = collate(next(stream))
            inputs = {key: value.to(device) for key, value in raw.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = peft_model(**inputs).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss) * accum)
        if step % 100 == 0 or step == max_steps:
            save_adapters(peft_model, out_dir)
            log(f"[{config['name']}] step {step}/{max_steps} loss={losses[-1]:.4f} lr={lr:.2e} "
                f"vram={torch.cuda.max_memory_allocated() / 1e9:.3f}GB")

    info.update({
        "name": config["name"], "seed": config.get("seed"), "max_steps": max_steps,
        "batch_size": batch_size, "grad_accum": accum, "lr": float(config["lr"]),
        "train_fraction": fraction,
        "final_loss": losses[-1], "first_loss": losses[0],
        "loss_is_finite": all(math.isfinite(v) for v in losses),
        "train_minutes": (time.perf_counter() - started) / 60,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0),
    })
    (out_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / "done.json").write_text(json.dumps({"name": config["name"],
                                                   "train_fraction": fraction}), encoding="utf-8")
    log(f"[{config['name']}] DONE {json.dumps(info)}")
    return peft_model, info


def save_adapters(peft_model, out_dir):
    """PEFT saves LoRA only; the fully trained projector needs its own file."""
    import torch
    out_dir = Path(out_dir)
    peft_model.save_pretrained(out_dir / "adapter")
    projector = {_short_name(n): p.detach().to("cpu")
                 for n, p in peft_model.named_parameters()
                 if any(key in n for key in PROJECTOR_KEYS)}
    if projector:
        torch.save(projector, out_dir / "projector.pt")
    return sorted(projector)


def load_adapter_and_projector(base_model, adapter_dir):
    """Load the LoRA adapter and, if present, overwrite projector tensors."""
    import torch
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir / "adapter")
    projector_file = adapter_dir / "projector.pt"
    loaded = 0
    if projector_file.exists():
        saved = torch.load(projector_file, map_location="cpu")
        targets = {_short_name(n): p for n, p in model.named_parameters()
                   if any(key in n for key in PROJECTOR_KEYS)}
        for name, tensor in saved.items():
            if name not in targets:
                raise KeyError(f"Projector tensor {name} has no target in the model")
            if tuple(targets[name].shape) != tuple(tensor.shape):
                raise ValueError(f"Projector shape mismatch for {name}")
            targets[name].data.copy_(tensor.to(targets[name].dtype))
            loaded += 1
        if loaded != len(targets):
            raise ValueError(f"Loaded {loaded} projector tensors but model has {len(targets)}")
    model.eval()
    return model, loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="work")
    args = parser.parse_args()
    print(json.dumps({"config": args.config, "note": "GPU entrypoint is the Kaggle notebook cell"},
                     indent=2))
