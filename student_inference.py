"""E0–E4 inference on CUDA hosts. Importing this module never loads a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image

from scripts.labcommon import read_jsonl
from student_labels import PROMPT


def validate_cached_predictions(rows, expected_ids):
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if image_id in seen or image_id not in expected_ids or not isinstance(row.get("raw"), str):
            raise ValueError("Prediction cache contains duplicate/unexpected IDs or non-string output")
        seen.add(image_id)
    return seen


def load_base_model(model_id, quantize=True):
    """Load the lab's NF4 configuration; visual and lm_head must stay unquantized."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run model inference on the notebook GPU host, not the local Mac")
    kwargs = {"dtype": torch.float16, "device_map": {"": 0}}
    if quantize:
        import bitsandbytes
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            # Full prefix is required: transformers 5.19 matches prefix/suffix, not substrings.
            llm_int8_skip_modules=["model.visual", "visual", "lm_head"],
        )
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if quantize:
        quantized = [n for n, m in model.named_modules()
                     if isinstance(m, bitsandbytes.nn.Linear4bit)]
        forbidden = [n for n in quantized if "visual" in n or "lm_head" in n]
        if not quantized or forbidden:
            raise RuntimeError(f"Invalid quantization: NF4 modules={len(quantized)}, forbidden={forbidden[:10]}")
        print(f"NF4 modules: {len(quantized)} | forbidden: 0", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def generate_predictions(model, processor, entries, data_dir, output, batch_size=12,
                         max_new_tokens=256, log=print):
    """Greedy generation into a raw JSONL file. Ground truth is never used here."""
    import torch
    data_dir, output = Path(data_dir), Path(output)
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    if hasattr(model, "config"):
        model.config.use_cache = True
    written = 0
    current = batch_size
    start = 0
    oom_error = torch.cuda.OutOfMemoryError  # lấy một lần để test tiêm được lớp lỗi giả
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        while start < len(entries):
            batch = entries[start:start + current]
            conversations = []
            for row in batch:
                with Image.open(data_dir / row["image_path"]) as image:
                    image = image.convert("RGB")
                conversations.append([{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}])
            try:
                inputs = processor.apply_chat_template(
                    conversations, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", padding=True).to(model.device)
                if start == 0:
                    counts = (inputs["input_ids"] == model.config.image_token_id).sum(dim=1).tolist()
                    log(f"Vision tokens per image: {counts}")
                    if any(count != 220 for count in counts):
                        raise RuntimeError("Unexpected vision-token count for 640x360 inputs")
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            except oom_error:
                # Batch chỉ là nút tốc độ, không phải biến của thí nghiệm: giảm nửa rồi thử lại đúng ảnh đó.
                torch.cuda.empty_cache()
                if current == 1:
                    raise
                current = max(1, current // 2)
                log(f"CUDA OOM: giảm batch suy luận xuống {current} rồi thử lại")
                continue
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            for row, text in zip(batch, raw):
                handle.write(json.dumps({"image_id": row["image_id"], "raw": text},
                                        ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            os.fsync(handle.fileno())
            log(f"{written}/{len(entries)} | batch={current} | {time.perf_counter() - started:.1f}s")
            start += len(batch)
    return {"n_images": written, "inference_seconds": time.perf_counter() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu": torch.cuda.get_device_name(0)}


def predict_split(config_path="configs/e0.yaml", split="test", data_dir="data",
                  output="work/e0.jsonl", batch_size=4, limit=None, variant=None):
    """E0 path: cache-aware, one full pass over the official split."""
    import yaml
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and limit must be positive")
    if limit is not None and split == "test":
        raise ValueError("Test evidence must cover the full official test split")
    if split not in ("val", "test"):
        raise ValueError("Inference split must be val or test")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_dir, output = Path(data_dir), Path(output)
    entries = read_jsonl(data_dir / "splits" / f"{split}.jsonl")
    if limit is not None:
        entries = entries[:limit]
    meta = json.loads((data_dir / "splits/meta.json").read_text(encoding="utf-8"))
    signature = {
        "variant": variant or "E0", "model_id": config["model_id"],
        "dataset_version": meta["dataset_version"], "split": split,
        "max_new_tokens": config["max_new_tokens"], "do_sample": False,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "image_ids": [r["image_id"] for r in entries], "batch_size": batch_size,
        "quantization": "nf4-4bit; visual and lm_head excluded",
    }
    signature_file = output.with_suffix(".signature.json")
    if output.exists():
        if not signature_file.exists() or json.loads(signature_file.read_text()) != signature:
            raise ValueError("Existing prediction cache has a different/missing run signature")
        seen = validate_cached_predictions(read_jsonl(output), set(signature["image_ids"]))
    else:
        seen = set()
        output.parent.mkdir(parents=True, exist_ok=True)
        signature_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    pending = [r for r in entries if r["image_id"] not in seen]
    if not pending:
        print("Complete prediction cache:", output, len(seen), flush=True)
        return output
    model, processor = load_base_model(config["model_id"])
    run = generate_predictions(model, processor, pending, data_dir, output, batch_size,
                               config["max_new_tokens"])
    run.update({"split": split, "trainable_params": 0, "variant": variant or "E0"})
    output.with_suffix(".run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e0.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="work/e0.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variant")
    a = parser.parse_args()
    predict_split(a.config, a.split, a.data_dir, a.output, a.batch_size, a.limit, a.variant)
