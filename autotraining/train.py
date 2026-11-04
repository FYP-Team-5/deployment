"""Run the reference Qwen3.5-9B QLoRA notebook headlessly, without its installer/login cells."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
import nbformat


def normalize_dataset(path):
    """Keep binary full-credit/zero-credit examples, record skipped partial credit."""
    import csv
    import math
    result_path = path.with_name("normalized.csv")
    retained = []
    counts = {"source_rows": 0, "skipped_partial_credit": 0, "skipped_missing_answer": 0}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        required = {"question", "desired_answer", "student_answer",
                    "criteria[]", "criteria_score[]", "criteria_max_score[]", "student_score"}
        if not fields or not required.issubset(fields):
            raise ValueError("Incorrect training CSV columns")
        for row in reader:
            counts["source_rows"] += 1
            scores = json.loads(row["criteria_score[]"])
            maxima = json.loads(row["criteria_max_score[]"])
            if not scores or len(scores) != len(maxima):
                raise ValueError("Misaligned criterion scores/maxima")
            if not row["question"].strip() or not row["desired_answer"].strip():
                counts["skipped_missing_answer"] += 1
                continue
            binary = []
            for score, maximum in zip(scores, maxima):
                if not isinstance(score, (int, float)) or not isinstance(maximum, (int, float)):
                    raise ValueError("Non-numeric criterion values")
                if math.isclose(score, 0, abs_tol=1e-7):
                    binary.append(0)
                elif math.isclose(score, maximum, abs_tol=1e-7):
                    binary.append(1)
                else:
                    counts["skipped_partial_credit"] += 1
                    break
            else:
                row["criteria_score[]"] = json.dumps(binary)
                retained.append(row)
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
    notebook = nbformat.read(args.notebook, as_version=4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Execute the exact reference training/evaluation code cells in the original order.
    # Cell 2 installs packages interactively; cell 46 prompts for HF login.
    skip_cells = {2, 46}
    scope = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or number in skip_cells:
            continue
        code = cell.source
        if number == 4:
            code = code.replace(
                'hf_token = os.environ["HF_TOKEN"]',
                'hf_token = os.environ.get("HF_TOKEN", "")',
            )
        if number == 14:
            code = code.replace(
                'print(dataset[0]["prompt"])',
                'print("Training prompt created; content redacted")',
            )
        exec(compile(code, f"notebook_cell_{number}", "exec"), scope)
        if number == 4:
            # Notebook preview tables contain actual student responses.
            scope["display"] = lambda *_: print("Notebook output redacted")
        if number == 6:
            scope.update({
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
            })
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
    prefix = f"candidates/{args.checksum}"
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
        "dataset_checksum": args.checksum,
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
        Bucket=args.bucket, Key=f"{prefix}/result.json",
        Body=json.dumps(manifest, allow_nan=False).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
