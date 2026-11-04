"""Run the reference Qwen3.5-9B QLoRA notebook headlessly, without its installer/login cells."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
import nbformat


def check_dataset(path):
    import csv
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            scores = json.loads(row["criteria_score[]"])
            if not scores or any(type(v) is not int or v not in (0, 1) for v in scores):
                raise ValueError(f"Row {count} has nonbinary criterion labels")
            if not row["question"].strip() or not row["desired_answer"].strip():
                raise ValueError(f"Row {count} has a missing question or answer")
    if not count:
        raise ValueError("Training CSV is empty")
    return count


def run(args):
    data = Path(args.dataset)
    actual = hashlib.sha256(data.read_bytes()).hexdigest()
    if actual != args.checksum:
        raise ValueError("Training CSV checksum mismatch")
    check_dataset(data)
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
        exec(compile(code, f"notebook_cell_{number}", "exec"), scope)
        if number == 6:
            scope.update({
                "MODEL_ID": "Qwen/Qwen3.5-9B",
                "DATASET_PATH": str(data.resolve()),
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
        "base_model": "Qwen/Qwen3.5-9B",
        "seed": 42,
        "lora_rank": 48,
        "lora_alpha": 96,
        "adapter_uri": f"s3://{args.bucket}/{prefix}/adapter/",
        "metrics": clean(metrics),
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
