"""Train feedback after grading, following feedback/qlora_feedback_qwen3.5-9B.ipynb."""

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import boto3

from train import normalize_dataset

REFERENCE_COMMIT = "19cb4c7e97934f821e68e90a1bf2f56b63e4dfad"
SYSTEM_PROMPT = (
    "You are a computer science tutor. Write concise, specific, encouraging feedback to the student on their short answer. "
    "The marker's decision for each key point is given; help the student understand it. "
    "Acknowledge what is correct, explain anything missing or incorrect, help the student improve, "
    "and never mention marks, key points or the reference answer. Write in plain text; do not use markdown."
)


def prepare_examples(path):
    """Split the same eligible question universe before filtering blank targets."""
    normalized, stats = normalize_dataset(path)
    with normalized.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    questions = list(dict.fromkeys(row["question"].strip() for row in rows))
    random.Random(42).shuffle(questions)
    held_out = len(questions) >= 10
    train_end = int(len(questions) * 0.8)
    validation_end = train_end + int(len(questions) * 0.1)
    groups = {
        "train": set(questions[:train_end]),
        "validation": set(questions[train_end:validation_end]),
        "test": set(questions[validation_end:]),
    } if held_out else {name: set(questions) for name in ("train", "validation", "test")}
    examples = {name: [] for name in groups}
    missing = 0
    for index, row in enumerate(rows):
        feedback = row.get("feedback", "").strip()
        if not feedback:
            missing += 1
            continue
        criteria = json.loads(row["criteria[]"])
        # normalize_dataset has already converted weighted scores to binary labels.
        marks = [bool(score) for score in json.loads(row["criteria_score[]"])]
        points = [
            f"{number}. {'[MET] ' if met else '[NOT MET] '}{criterion}"
            for number, (criterion, met) in enumerate(zip(criteria, marks, strict=True), 1)
        ]
        prompt = "\n\n".join([
            f"Question:\n{row['question'].strip()}",
            f"Reference answer:\n{row['desired_answer'].strip()}",
            f"Student answer:\n{row['student_answer'].strip() or '(no answer given)'}",
            "Key points:\n" + "\n".join(points),
            f"Key points met: {sum(marks)} of {len(criteria)}",
        ])
        example = {
            "row_index": index,
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": feedback}],
        }
        for name, group in groups.items():
            if row["question"].strip() in group:
                examples[name].append(example)
    if not examples["train"] or not examples["validation"]:
        raise ValueError("Feedback training needs nonempty feedback targets in the training and validation question splits.")
    return examples, {
        **stats,
        "skipped_missing_feedback": missing,
        "split_rows": {name: len(values) for name, values in examples.items()},
        "held_out_evaluation": held_out,
    }


def fit(examples, output, base_revision):
    """Notebook training recipe; generation and metric evaluation are omitted."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Feedback QLoRA training requires a CUDA GPU.")
    set_seed(42)
    model_id = "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=base_revision, token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(example):
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False)
        if tokens[:len(prompt)] != prompt or len(prompt) >= len(tokens):
            raise ValueError("Invalid feedback prompt/completion boundary")
        return {"input_ids": tokens, "labels": [-100] * len(prompt) + tokens[len(prompt):]}

    formatted = {}
    for name in ("train", "validation"):
        dataset = Dataset.from_list(examples[name])
        formatted[name] = dataset.map(tokenize, remove_columns=dataset.column_names)
    sequence_limit = math.ceil(max(len(ids) for split in formatted.values() for ids in split["input_ids"]) / 32) * 32
    if sequence_limit > 4096:
        raise ValueError("Feedback examples exceed the 4096-token training limit")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=base_revision, token=False, quantization_config=quantization, device_map={"": 0}, dtype=dtype, attn_implementation="sdpa")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name, module in model.named_modules() if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise RuntimeError("No text projection layers found for feedback training")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    # Preserve the notebook's effective batch 16 while fitting the 24-GiB GPU.
    eval_steps = max(1, round(math.ceil(len(formatted["train"]) / 16) / 4))
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output), seed=42, data_seed=42, num_train_epochs=3,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, learning_rate=1e-4, weight_decay=0.05,
            optim="paged_adamw_8bit", lr_scheduler_type="reduce_lr_on_plateau",
            lr_scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 1, "min_lr": 1e-6},
            bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            loss_type="nll", completion_only_loss=True, max_length=sequence_limit,
            packing=False, padding_free=False, logging_steps=max(1, eval_steps // 2),
            save_strategy="steps", save_steps=eval_steps, eval_strategy="steps", eval_steps=eval_steps,
            prediction_loss_only=True, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=3, report_to="none",
        ),
        train_dataset=formatted["train"], eval_dataset=formatted["validation"],
        processing_class=tokenizer, callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    samples = [trainer.train_dataset[i] for i in range(min(2, len(trainer.train_dataset)))]
    batch = trainer.data_collator(samples)
    for index, sample in enumerate(samples):
        labels, length = batch["labels"][index], len(sample["labels"])
        assert labels[:length].tolist() == sample["labels"], "Collator changed the feedback loss mask"
        assert (labels[length:] == -100).all(), "Padding contributes to feedback loss"
    result = trainer.train()
    if not math.isfinite(float(result.training_loss)) or trainer.state.global_step < 1:
        raise RuntimeError("Feedback training did not complete a valid optimizer step")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    return {
        "MODEL_ID": model_id, "base_model_revision": base_revision,
        "LORA_R": 16, "LORA_ALPHA": 32, "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 1e-4, "WEIGHT_DECAY": 0.05, "NUM_EPOCHS": 3,
        "TRAIN_BATCH_SIZE": 1, "GRADIENT_ACCUMULATION_STEPS": 16,
        "SEED": 42, "SPLIT_SEED": 42, "sequence_limit": sequence_limit,
        "completion_only_loss": True, "target_modules": targets,
        "quantization": quantization.to_dict(), "compute_dtype": str(dtype),
        "reference_notebook": "feedback/qlora_feedback_qwen3.5-9B.ipynb",
        "reference_commit": REFERENCE_COMMIT,
        "evaluation_metrics": False,
    }


def run(args):
    dataset = Path(args.dataset)
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != args.checksum:
        raise ValueError("Feedback CSV checksum mismatch")
    s3 = boto3.client("s3")
    prefix = f"candidates/{args.checksum}/{args.run_id}"
    with s3.get_object(Bucket=args.bucket, Key=f"{prefix}/grading/result.json")["Body"] as body:
        grading = json.loads(body.read())
    if any(grading.get(key) != value for key, value in {"run_id": args.run_id, "dataset_checksum": args.checksum, "dataset_version": args.version_id, "status": "completed"}.items()):
        raise ValueError("Completed grading stage does not match the feedback run")
    base_revision = grading["configuration"]["base_model_revision"]
    examples, stats = prepare_examples(dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print("Starting feedback training after completed grading training.")
    configuration = fit(examples, output, base_revision)
    configuration["split_strategy"] = grading["configuration"]["split_strategy"]
    configuration["held_out_evaluation"] = stats["held_out_evaluation"]
    (output / "training_configuration.json").write_text(json.dumps(configuration, indent=2, allow_nan=False))
    for name in ("adapter_config.json", "adapter_model.safetensors", "training_configuration.json"):
        path = output / name
        if not path.is_file() or path.stat().st_size < 1:
            raise RuntimeError("Feedback adapter files are missing or empty")
        s3.upload_file(str(path), args.bucket, f"{prefix}/feedback/adapter/{name}")
    # Only this final manifest makes the paired run eligible for publication.
    manifest = {
        **grading,
        "completed_at": datetime.now(UTC).isoformat(),
        "feedback": {
            "adapter_uri": f"s3://{args.bucket}/{prefix}/feedback/adapter/",
            "configuration": configuration, "dataset_stats": stats,
        },
    }
    s3.put_object(Bucket=args.bucket, Key=f"{prefix}/result.json", Body=json.dumps(manifest, allow_nan=False).encode(), ContentType="application/json", ServerSideEncryption="AES256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket", "checksum", "run-id", "version-id", "dataset", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
