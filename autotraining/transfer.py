"""Prepare an immutable dataset snapshot on the existing backend host."""

import hashlib
import json
import os
import re
import time

import boto3
from state import AWS_CONFIG, StateConflict, claim, get_item, record, unlock

SSM = boto3.client("ssm", config=AWS_CONFIG)
BUCKET = os.environ["ARTIFACT_BUCKET"]
BACKEND = os.environ["BACKEND_INSTANCE_ID"]
REGION = os.environ["AWS_REGION"]
DIGEST = re.compile(r"^[a-f0-9]{64}$")
VERSION = re.compile(r"^[a-f0-9-]{36}$")


def send(instance, lines, seconds):
    options = {}
    if instance == os.environ["TRAINING_INSTANCE_ID"]:
        options["CloudWatchOutputConfig"] = {
            "CloudWatchOutputEnabled": True,
            "CloudWatchLogGroupName": os.environ["TRAIN_LOG_GROUP"],
        }
    return SSM.send_command(
        InstanceIds=[instance],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": lines, "executionTimeout": [str(seconds)]},
        TimeoutSeconds=120,
        Comment="FYP nightly QLoRA",
        **options,
    )["Command"]["CommandId"]


def read_command(instance, command_id):
    try:
        return SSM.get_command_invocation(InstanceId=instance, CommandId=command_id)
    except SSM.exceptions.InvocationDoesNotExist:
        return {"Status": "Pending"}


def start_snapshot():
    command_id = send(
        BACKEND,
        [
            "set -e",
            "umask 077",
            "command -v aws >/dev/null || (apt-get update && apt-get install -y awscli)",
            f"aws s3 cp s3://{BUCKET}/scripts/export_backend.sh /tmp/fyp-export.sh >/dev/null",
            f"bash /tmp/fyp-export.sh {BUCKET} {REGION}",
        ],
        360,
    )
    return {"command_id": command_id}


def snapshot(command_id):
    result = read_command(BACKEND, command_id)
    status = result["Status"]
    if status == "Success":
        job = json.loads(result["StandardOutputContent"].strip().splitlines()[-1])
        if job.get("reason") == "no_ready_dataset":
            return job
        if not DIGEST.fullmatch(job.get("checksum", "")):
            raise ValueError("Export checksum invalid")
        if not VERSION.fullmatch(job.get("version_id", "")):
            raise ValueError("Export version invalid")
        if job.get("row_count", 0) < 1:
            raise ValueError("Empty export")
        return job
    if status in ("Failed", "TimedOut", "Cancelled", "DeliveryTimedOut"):
        raise RuntimeError("Backend export SSM status: " + status)
    return None


def prepare(command_id, execution_id, request=None):
    request = request or {}
    force_retrain = request.get("force_retrain", False)
    if type(force_retrain) is not bool:
        raise ValueError("force_retrain must be a boolean")
    job = snapshot(command_id)
    if job is None:
        return {"ready": False}
    job["ready"] = True
    if job.get("reason") == "no_ready_dataset":
        return dict(job, train=False)
    job.update(
        {
            "execution_id": execution_id,
            "run_id": hashlib.sha256(execution_id.encode()).hexdigest()[:32],
            "started_at": int(time.time()),
            "force_retrain": force_retrain,
            "requested_by": request.get("requested_by"),
        }
    )
    prior = get_item("success/" + job["checksum"])
    trained = get_item("trained/" + job["checksum"])
    if (
        prior
        and not force_retrain
        and (not trained or trained.get("run_id") == prior.get("run_id"))
    ):
        record(job, "skipped", reason="unchanged")
        return dict(job, train=False, reason="unchanged")
    if trained and not force_retrain:
        # Reuse completed weights when HF upload or callback previously failed.
        return dict(trained, ready=True, train=False, publish=True)
    try:
        acquired = claim(job)
    except StateConflict:
        acquired = False
    if not acquired:
        record(job, "skipped", reason="already_running")
        return dict(job, train=False, reason="already_running")
    # Another execution may have completed between the first read and lock claim.
    if not force_retrain and get_item("success/" + job["checksum"]):
        unlock(job)
        record(job, "skipped", reason="unchanged")
        return dict(job, train=False, reason="unchanged")
    record(job, "running")
    return dict(job, train=True, publish=True)
