"""E0 inference on Colab CUDA; importing this module does not load a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image
import yaml

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


def load_base_model(config):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Run model inference on Colab CUDA, not the local Mac")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["model.visual", "visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_id"], quantization_config=quantization,
        dtype=torch.float16, device_map={"": 0},
    )
    quantized = [name for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
    forbidden = [name for name in quantized if "visual" in name or "lm_head" in name]
    if not quantized or forbidden:
        raise RuntimeError(f"Invalid quantization: NF4 modules={len(quantized)}, forbidden={forbidden[:10]}")
    processor = AutoProcessor.from_pretrained(config["model_id"])
    processor.tokenizer.padding_side = "left"
    model.eval()
    print("GPU:", torch.cuda.get_device_name(0), "| NF4 modules:", len(quantized), flush=True)
    return model, processor


def predict_split(config_path="configs/e0.yaml", split="test", data_dir="data",
                  output="work/e0.jsonl", batch_size=4, limit=None):
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and limit must be positive")
    if split == "test" and limit is not None:
        raise ValueError("Test evidence must cover the full official test split")
    if split not in ("val", "test"):
        raise ValueError("Inference split must be val or test")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_dir, output = Path(data_dir), Path(output)
    entries = read_jsonl(data_dir / "splits" / f"{split}.jsonl")
    if limit is not None:
        entries = entries[:limit]
    meta = json.loads((data_dir / "splits/meta.json").read_text(encoding="utf-8"))
    signature = {"model_id": config["model_id"], "dataset_version": meta["dataset_version"],
                 "split": split, "max_new_tokens": config["max_new_tokens"], "do_sample": False,
                 "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                 "image_ids": [r["image_id"] for r in entries], "batch_size": batch_size,
                 "quantization": "nf4-4bit; visual and lm_head excluded"}
    signature_file = output.with_suffix(".signature.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not signature_file.exists() or json.loads(signature_file.read_text()) != signature:
            raise ValueError("Existing prediction cache has a different/missing run signature")
        seen = validate_cached_predictions(read_jsonl(output), set(signature["image_ids"]))
    else:
        seen = set()
        signature_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    pending = [r for r in entries if r["image_id"] not in seen]
    if not pending:
        print("Complete prediction cache:", output, len(seen), flush=True)
        return output
    import torch

    model, processor = load_base_model(config)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            conversations = []
            for row in batch:
                with Image.open(data_dir / row["image_path"]) as image:
                    image = image.convert("RGB")
                conversations.append([{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}])
            inputs = processor.apply_chat_template(
                conversations, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(model.device)
            if start == 0:
                counts = (inputs["input_ids"] == model.config.image_token_id).sum(dim=1).tolist()
                print("Vision tokens per image:", counts, flush=True)
                if any(count != 220 for count in counts):
                    raise RuntimeError("Unexpected vision-token count for the lab's 640x360 images")
            generated = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False)
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch, raw):
                handle.write(json.dumps({"image_id": row["image_id"], "raw": text}, ensure_ascii=False) + "\n")
                seen.add(row["image_id"])
            handle.flush()
            os.fsync(handle.fileno())
            print(f"{split}: {len(seen)}/{len(entries)} images | {time.perf_counter()-started:.1f}s", flush=True)
    run = {"split": split, "n_images": len(seen), "inference_seconds_this_session": time.perf_counter()-started,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
           "gpu": torch.cuda.get_device_name(0), "trainable_params": 0}
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
    args = parser.parse_args()
    predict_split(args.config, args.split, args.data_dir, args.output, args.batch_size, args.limit)
