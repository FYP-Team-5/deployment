"""Prepare an immutable dataset snapshot on the existing backend host."""

import hashlib
import json
import os
import re
import time
import uuid

import boto3
from botocore.exceptions import ClientError

SSM = boto3.client("ssm")
DB = boto3.resource("dynamodb").Table(os.environ["STATE_TABLE"])
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


def snapshot(context):
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
    deadline = time.monotonic() + 480
    while time.monotonic() < deadline:
        if context.get_remaining_time_in_millis() < 10000:
            raise TimeoutError("Dataset export exceeded Lambda timeout")
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
        time.sleep(6)
    raise TimeoutError("Backend export did not finish")


def get_item(pk):
    return DB.get_item(Key={"pk": pk}, ConsistentRead=True).get("Item", {})


def owns(job):
    lock = get_item("lock")
    return lock.get("execution_id") == job["execution_id"] and not lock.get(
        "cleanup_token"
    )


def claim_cleanup(job):
    """Serialize normal cleanup, terminal events and watchdog invocations."""
    token = uuid.uuid4().hex
    DB.put_item(
        Item={
            **job,
            "pk": "lock",
            "cleanup_token": token,
            "cleanup_until": int(time.time()) + 660,
        },
        ConditionExpression="execution_id = :owner AND (attribute_not_exists(cleanup_until) OR cleanup_until < :now)",
        ExpressionAttributeValues={
            ":owner": job["execution_id"],
            ":now": int(time.time()),
        },
    )
    return token


def release_cleanup(job, token):
    cleaned = {
        key: value
        for key, value in job.items()
        if key not in ("cleanup_until", "cleanup_token")
    }
    try:
        DB.put_item(
            Item={**cleaned, "pk": "lock"},
            ConditionExpression="execution_id = :owner AND cleanup_token = :token",
            ExpressionAttributeValues={":owner": job["execution_id"], ":token": token},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise


def record(job, status, **details):
    """Keep each execution, including failed retries, independently of its CSV."""
    DB.put_item(
        Item={
            **job,
            "pk": "run#" + job["run_id"],
            "status": status,
            "updated_at": int(time.time()),
            **details,
        }
    )


def prepare(context, execution_id):
    job = snapshot(context)
    if job.get("reason") == "no_ready_dataset":
        return dict(job, train=False)
    job.update(
        {
            "execution_id": execution_id,
            "run_id": hashlib.sha256(execution_id.encode()).hexdigest()[:32],
            "started_at": int(time.time()),
        }
    )
    prior = get_item("success#" + job["checksum"])
    # Read the old latest marker as well, for deployments predating run history.
    if prior or get_item("latest").get("checksum") == job["checksum"]:
        record(job, "skipped", reason="unchanged")
        return dict(job, train=False, reason="unchanged")
    try:
        # The independent watchdog reconciles abandoned owners. Never let TTL
        # delete a lock underneath a training process or a delayed cleanup event.
        DB.put_item(
            Item={**job, "pk": "lock"},
            ConditionExpression="attribute_not_exists(pk)",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        lock = get_item("lock")
        if lock.get("execution_id") == execution_id:
            return dict(lock, train=True)  # Idempotent prepare retry.
        record(job, "skipped", reason="already_running")
        return dict(job, train=False, reason="already_running")
    # Another execution may have completed between the first read and lock claim.
    if get_item("success#" + job["checksum"]):
        unlock(job)
        record(job, "skipped", reason="unchanged")
        return dict(job, train=False, reason="unchanged")
    record(job, "running")
    return dict(job, train=True)


def unlock(job):
    try:
        DB.delete_item(
            Key={"pk": "lock"},
            ConditionExpression="execution_id = :owner",
            ExpressionAttributeValues={":owner": job["execution_id"]},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
