"""Run the reference Qwen3.5-9B QLoRA notebook headlessly, without its installer/login cells."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3


def normalize_dataset(path):
    """Keep binary full-credit/zero-credit examples, record skipped partial credit."""
    import csv
    import math

    result_path = path.with_name("normalized.csv")
    retained = []
    counts = {
        "source_rows": 0,
        "skipped_partial_credit": 0,
        "skipped_missing_answer": 0,
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        required = [
            "id",
            "question",
            "desired_answer",
            "student_answer",
            "criteria[]",
            "criteria_score[]",
            "criteria_max_score[]",
            "student_score",
        ]
        if fields != required:
            raise ValueError("Incorrect training CSV columns or order")
        for number, row in enumerate(reader, 2):
            counts["source_rows"] += 1
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Incorrect cell count")
                criteria = json.loads(row["criteria[]"])
                scores = json.loads(row["criteria_score[]"])
                maxima = json.loads(row["criteria_max_score[]"])
                if (
                    not all(
                        isinstance(value, list) for value in (criteria, scores, maxima)
                    )
                    or not criteria
                    or len(criteria) != len(scores)
                    or len(scores) != len(maxima)
                ):
                    raise ValueError("Misaligned criterion arrays")
                if any(not isinstance(c, str) or not c.strip() for c in criteria):
                    raise ValueError("Criteria must be nonempty strings")
                if not row["id"].strip() or not row["question"].strip():
                    raise ValueError("Missing question identity or prompt")
                for score, maximum in zip(scores, maxima):
                    if (
                        type(score) not in (int, float)
                        or type(maximum) not in (int, float)
                        or not math.isfinite(score)
                        or not math.isfinite(maximum)
                        or maximum <= 0
                        or score < 0
                        or score > maximum
                    ):
                        raise ValueError("Invalid criterion score or maximum")
                total = float(row["student_score"])
                if not math.isfinite(total) or not math.isclose(
                    total, sum(scores), abs_tol=1e-7
                ):
                    raise ValueError("Student score disagrees with criterion scores")
                if not row["desired_answer"].strip():
                    counts["skipped_missing_answer"] += 1
                    continue
                # Preserve weighted scores in the source; the notebook learns
                # binary decisions and applies the original maxima at evaluation.
                if any(
                    score != 0 and score != maximum
                    for score, maximum in zip(scores, maxima)
                ):
                    counts["skipped_partial_credit"] += 1
                    continue
                row["criteria_score[]"] = json.dumps(
                    [int(score != 0) for score in scores]
                )
                retained.append(row)
            except (ValueError, TypeError) as error:
                # Do not include student responses in SSM/CloudWatch logs.
                raise ValueError(
                    f"Invalid training CSV row {number}: {error.__class__.__name__}"
                ) from None
    if len({row["question"].strip() for row in retained}) < 10:
        raise ValueError("Not enough compatible unique questions to split 80/10/10")
    with result_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)
    counts["training_rows"] = len(retained)
    return result_path, counts


def run(args):
    data = Path(args.dataset)
    actual = hashlib.sha256(data.read_bytes()).hexdigest()
    if actual != args.checksum:
        raise ValueError("Training CSV checksum mismatch")
    normalized, dataset_stats = normalize_dataset(data)
    notebook_path = Path(args.notebook)
    notebook = json.loads(notebook_path.read_text())
    from huggingface_hub import HfApi

    # Resolve once so tokenizer and model cannot load different upstream revisions.
    base_revision = HfApi(token=False).model_info("Qwen/Qwen3.5-9B").sha
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Execute the exact reference training/evaluation code cells in the original order.
    # Cell 2 installs packages interactively; cell 46 prompts for HF login.
    skip_cells = {2, 46}
    scope = {"__name__": "__main__", "BASE_REVISION": base_revision}
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code" or number in skip_cells:
            continue
        code = cell["source"]
        if isinstance(code, list):
            code = "".join(code)
        if number in (18, 22):
            code = code.replace(
                "from_pretrained(MODEL_ID)",
                "from_pretrained(MODEL_ID, revision=BASE_REVISION, token=False)",
            )
            code = code.replace(
                "MODEL_ID, quantization_config=",
                "MODEL_ID, revision=BASE_REVISION, token=False, quantization_config=",
            )
        if number == 4:
            code = code.replace(
                'hf_token = os.environ["HF_TOKEN"]',
                "hf_token = None",
            )
        if number == 14:
            code = code.replace(
                'print(dataset[0]["prompt"])',
                'print("Training prompt created; content redacted")',
            )
        exec(compile(code, f"notebook_cell_{number}", "exec"), scope)  # noqa: S102 - trusted, commit-pinned notebook
        if number == 4:
            # Notebook preview tables contain actual student responses.
            scope["display"] = lambda *_: print("Notebook output redacted")
        if number == 6:
            scope.update(
                {
                    "MODEL_ID": "Qwen/Qwen3.5-9B",
                    "DATASET_PATH": str(normalized.resolve()),
                    "OUTPUT_DIR": str(output),
                    "RUN_TRAINING": True,
                    "RUN_TEST_EVALUATION": True,
                    "RUN_HF_UPLOAD": False,
                    "SEED": 42,
                    "SPLIT_SEED": 42,
                    "LORA_R": 48,
                    "LORA_ALPHA": 96,
                    "MAX_SEQ_LENGTH": 4096,
                }
            )
    if not scope.get("evaluation_completed"):
        raise RuntimeError("The reference notebook did not finish evaluation")
    metrics = scope.get("finetuned_metrics")
    if not metrics or metrics.get("valid_examples", 0) < 1:
        raise RuntimeError("No valid held-out predictions")
    files = ["adapter_config.json", "adapter_model.safetensors"]
    for filename in files:
        if not (output / filename).is_file():
            raise FileNotFoundError(output / filename)

    s3 = boto3.client("s3")
    prefix = f"candidates/{args.checksum}/{args.run_id}"
    configuration = {
        name: scope[name]
        for name in (
            "MODEL_ID",
            "SEED",
            "SPLIT_SEED",
            "MAX_SEQ_LENGTH",
            "LORA_R",
            "LORA_ALPHA",
            "LORA_DROPOUT",
            "LEARNING_RATE",
            "WEIGHT_DECAY",
            "NUM_EPOCHS",
            "TRAIN_BATCH_SIZE",
            "GRADIENT_ACCUMULATION_STEPS",
            "VALIDATIONS_PER_EPOCH",
            "LR_REDUCTION_FACTOR",
            "LR_PATIENCE",
            "MIN_LEARNING_RATE",
            "EARLY_STOPPING_PATIENCE",
            "EARLY_STOPPING_THRESHOLD",
            "GENERATION_MIN_TOKENS",
            "GENERATION_TOKENS_PER_CRITERION",
        )
    }
    configuration.update(
        {
            "quantization": scope["bnb_config"].to_dict(),
            "compute_dtype": str(scope["compute_dtype"]),
            "target_modules": scope["target_modules"],
            "optimizer": "paged_adamw_8bit",
            "gradient_checkpointing": True,
            "lr_scheduler": "reduce_lr_on_plateau",
            "completion_only_loss": True,
            "sequence_limit": scope["sequence_limit"],
            "base_model_revision": base_revision,
            "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        }
    )
    import importlib.metadata

    configuration["packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "trl",
            "peft",
            "bitsandbytes",
            "datasets",
            "accelerate",
        )
    }
    (output / "training_configuration.json").write_text(
        json.dumps(configuration, indent=2)
    )
    for path in output.iterdir():
        if path.is_file():
            s3.upload_file(str(path), args.bucket, f"{prefix}/adapter/{path.name}")

    # Null out non-finite values to produce strict JSON.
    def clean(value):
        import math

        if hasattr(value, "item"):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    manifest = {
        "status": "completed",
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset_checksum": args.checksum,
        "run_id": args.run_id,
        "configuration": configuration,
        "dataset_version": args.version_id,
        "dataset_rows": len(scope["records"]),
        "dataset_stats": dataset_stats,
        "base_model": "Qwen/Qwen3.5-9B",
        "seed": 42,
        "lora_rank": 48,
        "lora_alpha": 96,
        "model_repo_commit": "8a2b94ec95754d875dc546461ef15dc2a279ce5a",
        "base_model_revision": getattr(scope["model"].config, "_commit_hash", None),
        "adapter_uri": f"s3://{args.bucket}/{prefix}/adapter/",
        "metrics": clean(metrics),
        "baseline_metrics": clean(scope.get("baseline_metrics", {})),
    }
    s3.put_object(
        Bucket=args.bucket,
        Key=f"{prefix}/result.json",
        Body=json.dumps(manifest, allow_nan=False).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
