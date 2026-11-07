"""Publish evaluated adapters on the CPU backend host, then notify its REST API."""

import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError


def get_json(s3, bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    with response["Body"] as body:
        return json.loads(body.read())


def put_json(s3, bucket, key, value):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, allow_nan=False).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def notify(url, token, candidate):
    body = json.dumps(candidate, allow_nan=False).encode()
    for attempt in range(4):
        request = Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Training-Token": token},
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read())
            if (
                result.get("run_id") != candidate["run_id"]
                or result.get("registered") is not True
            ):
                raise ValueError("Backend did not confirm candidate registration")
            return
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise RuntimeError(
                    f"Backend rejected candidate metadata (HTTP {error.code})"
                ) from None
        except (URLError, TimeoutError):
            pass
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError("Backend notification could not be confirmed; retry publication")


def publish(args):
    if not re.fullmatch(r"[a-f0-9]{32}", args.run_id) or not re.fullmatch(
        r"[a-f0-9]{64}", args.checksum
    ):
        raise ValueError("Invalid run identity")
    s3, ssm = boto3.client("s3"), boto3.client("ssm")
    callback_token = ssm.get_parameter(
        Name=args.callback_token_parameter, WithDecryption=True
    )["Parameter"]["Value"]
    prefix = f"candidates/{args.checksum}/{args.run_id}"
    receipt_key = f"publications/{args.run_id}/result.json"
    receipt = get_json(s3, args.bucket, receipt_key)
    if receipt is None:
        manifest = get_json(s3, args.bucket, prefix + "/result.json")
        if (
            not manifest
            or manifest["dataset_checksum"] != args.checksum
            or manifest["run_id"] != args.run_id
        ):
            raise ValueError("Completed candidate manifest is missing or mismatched")
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        # Public reads remain anonymous; the upload credential is used only for writes.
        hf_token = ssm.get_parameter(Name=args.hf_token_parameter, WithDecryption=True)[
            "Parameter"
        ]["Value"]
        api = HfApi(token=hf_token)
        public_api = HfApi(token=False)
        branch = "candidate-" + args.run_id
        api.create_branch(repo_id=args.repo, branch=branch, exist_ok=True)
        # Reconcile a crash after HF committed but before the S3 receipt was saved.
        revision = None
        try:
            previous = json.loads(
                Path(
                    hf_hub_download(
                        repo_id=args.repo,
                        filename="candidate.json",
                        revision=branch,
                        token=False,
                    )
                ).read_text()
            )
            if (
                previous.get("run_id") == args.run_id
                and previous.get("dataset_checksum") == args.checksum
            ):
                revision = public_api.model_info(args.repo, revision=branch).sha
        except EntryNotFoundError:
            pass
        if revision is None:
            with tempfile.TemporaryDirectory(prefix="fyp-publish-") as folder:
                target = Path(folder)
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "training_configuration.json",
                ):
                    s3.download_file(
                        args.bucket, prefix + "/adapter/" + name, str(target / name)
                    )
                (target / "candidate.json").write_text(
                    json.dumps(manifest, allow_nan=False)
                )
                commit = api.upload_folder(
                    repo_id=args.repo,
                    folder_path=folder,
                    revision=branch,
                    allow_patterns=[
                        "adapter_config.json",
                        "adapter_model.safetensors",
                        "training_configuration.json",
                        "candidate.json",
                    ],
                    commit_message=f"Evaluated candidate {args.run_id} (dataset {args.checksum[:12]})",
                )
                revision = commit.oid
        candidate = {
            key: manifest[key]
            for key in (
                "run_id",
                "dataset_checksum",
                "dataset_version",
                "completed_at",
                "base_model",
                "metrics",
                "baseline_metrics",
                "configuration",
                "dataset_stats",
            )
        }
        candidate.update(
            hf_repo=args.repo,
            hf_branch=branch,
            hf_revision=revision,
            output_contract="criteria_array_v1",
        )
        receipt = {
            "run_id": args.run_id,
            "hf_repo": args.repo,
            "hf_revision": revision,
            "notified": False,
            "candidate": candidate,
        }
        put_json(s3, args.bucket, receipt_key, receipt)
    notify(args.callback_url, callback_token, receipt["candidate"])
    put_json(s3, args.bucket, receipt_key, {**receipt, "notified": True})
    # Only confirmed HF + backend publications may expire from duplicate staging.
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=args.bucket, Prefix=prefix + "/"
    ):
        for item in page.get("Contents", []):
            s3.put_object_tagging(
                Bucket=args.bucket,
                Key=item["Key"],
                Tagging={"TagSet": [{"Key": "published", "Value": "true"}]},
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "bucket",
        "checksum",
        "run-id",
        "repo",
        "hf-token-parameter",
        "callback-token-parameter",
        "callback-url",
    ):
        parser.add_argument("--" + name, required=True)
    publish(parser.parse_args())
