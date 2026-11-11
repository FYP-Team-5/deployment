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


def publish_adapter(s3, bucket, prefix, repo, token, manifest, kind):
    """Reconcile each model independently after an interrupted paired upload."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi(token=token)
    public_api = HfApi(token=False)
    branch = "candidate-" + manifest["run_id"]
    api.create_branch(repo_id=repo, branch=branch, exist_ok=True)
    try:
        previous = json.loads(
            Path(
                hf_hub_download(
                    repo_id=repo,
                    filename="candidate.json",
                    revision=branch,
                    token=False,
                )
            ).read_text()
        )
        if (
            previous.get("run_id") == manifest["run_id"]
            and previous.get("dataset_checksum") == manifest["dataset_checksum"]
            and previous.get("model_kind", "grading") == kind
        ):
            return public_api.model_info(repo, revision=branch).sha
    except EntryNotFoundError:
        pass
    with tempfile.TemporaryDirectory(prefix="fyp-publish-") as folder:
        target = Path(folder)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_configuration.json",
        ):
            s3.download_file(bucket, prefix + "/adapter/" + name, str(target / name))
        (target / "candidate.json").write_text(
            json.dumps({**manifest, "model_kind": kind}, allow_nan=False)
        )
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=folder,
            revision=branch,
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "training_configuration.json",
                "candidate.json",
            ],
            commit_message=f"{kind.capitalize()} candidate {manifest['run_id']} (dataset {manifest['dataset_checksum'][:12]})",
        )
        return commit.oid


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
        # Public reads remain anonymous; the upload credential is used only for writes.
        hf_token = ssm.get_parameter(Name=args.hf_token_parameter, WithDecryption=True)[
            "Parameter"
        ]["Value"]
        branch = "candidate-" + args.run_id
        feedback = manifest.get("feedback")
        if manifest.get("feedback_required") and not feedback:
            raise ValueError("Paired training has not completed the feedback stage")
        feedback_repo = getattr(args, "feedback_repo", "SmuFypTeam5/feedback")
        if feedback and feedback_repo == args.repo:
            raise ValueError(
                "Grading and feedback adapters require separate repositories"
            )
        revision = publish_adapter(
            s3, args.bucket, prefix, args.repo, hf_token, manifest, "grading"
        )
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
        if feedback:
            feedback_manifest = {
                key: manifest[key]
                for key in (
                    "run_id",
                    "dataset_checksum",
                    "dataset_version",
                    "completed_at",
                    "base_model",
                )
            }
            feedback_manifest.update(
                configuration=feedback["configuration"],
                dataset_stats=feedback["dataset_stats"],
            )
            feedback_revision = publish_adapter(
                s3,
                args.bucket,
                prefix + "/feedback",
                feedback_repo,
                hf_token,
                feedback_manifest,
                "feedback",
            )
            candidate["feedback"] = {
                "hf_repo": feedback_repo,
                "hf_branch": branch,
                "hf_revision": feedback_revision,
                "configuration": feedback["configuration"],
                "dataset_stats": feedback["dataset_stats"],
            }
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
    parser.add_argument("--feedback-repo", default="SmuFypTeam5/feedback")
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
