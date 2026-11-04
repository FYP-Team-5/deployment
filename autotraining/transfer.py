"""Prepare an immutable dataset snapshot on the existing backend host."""
import json
import os
import re
import time

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
        return SSM.get_command_invocation(
            InstanceId=instance, CommandId=command_id
        )
    except SSM.exceptions.InvocationDoesNotExist:
        return {"Status": "Pending"}


def snapshot(context):
    command_id = send(BACKEND, [
        "set -e",
        "umask 077",
        "command -v aws >/dev/null || (apt-get update && apt-get install -y awscli)",
        f"aws s3 cp s3://{BUCKET}/scripts/export_backend.sh /tmp/fyp-export.sh >/dev/null",
        f"bash /tmp/fyp-export.sh {BUCKET} {REGION}",
    ], 360)
    deadline = time.monotonic() + 480
    while time.monotonic() < deadline:
        if context.get_remaining_time_in_millis() < 10000:
            raise TimeoutError("Dataset export exceeded Lambda timeout")
        result = read_command(BACKEND, command_id)
        status = result["Status"]
        if status == "Success":
            job = json.loads(result["StandardOutputContent"].strip().splitlines()[-1])
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


def prepare(context):
    job = snapshot(context)
    prior = DB.get_item(Key={"pk": "latest"}).get("Item", {})
    if prior.get("checksum") == job["checksum"]:
        return dict(job, train=False, reason="unchanged")
    try:
        DB.put_item(
            Item={
                "pk": "lock", "checksum": job["checksum"],
                "expires_at": int(time.time()) + 36 * 3600,
            },
            ConditionExpression="attribute_not_exists(pk) OR expires_at < :now",
            ExpressionAttributeValues={":now": int(time.time())},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        return dict(job, train=False, reason="already_running")
    return dict(job, train=True)


def unlock(checksum):
    if DIGEST.fullmatch(checksum):
        try:
            DB.delete_item(
                Key={"pk": "lock"},
                ConditionExpression="checksum = :c",
                ExpressionAttributeValues={":c": checksum},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
