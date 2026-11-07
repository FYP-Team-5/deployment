"""EC2/SSM operations for the nightly QLoRA training state machine."""

import json
import math
import os
import re
import time

import boto3
from transfer import (
    DB,
    DIGEST,
    VERSION,
    claim_cleanup,
    get_item,
    owns,
    read_command,
    record,
    release_cleanup,
    send,
    unlock,
)

EC2 = boto3.client("ec2")
SSM = boto3.client("ssm")
S3 = boto3.client("s3")
TRAINER = os.environ["TRAINING_INSTANCE_ID"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
REGION = os.environ["AWS_REGION"]


def require_owner(job):
    if not owns(job):
        raise RuntimeError("Training execution no longer owns the GPU lock")


def start(job):
    require_owner(job)
    current = EC2.describe_instances(InstanceIds=[TRAINER])["Reservations"][0][
        "Instances"
    ][0]["State"]["Name"]
    if current == "stopped":
        EC2.start_instances(InstanceIds=[TRAINER])
    elif current not in ("pending", "running"):
        raise RuntimeError("GPU instance cannot be started: " + current)
    return {"instance_id": TRAINER}


def launch(job, attempt=0):
    require_owner(job)
    online = SSM.describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [TRAINER]}]
    )["InstanceInformationList"]
    if not any(info.get("PingStatus") == "Online" for info in online):
        return {"ready": False, "attempt": attempt + 1}
    checksum = job["checksum"]
    if not DIGEST.fullmatch(checksum):
        raise ValueError("Invalid dataset checksum")
    version_id = job["version_id"]
    if not VERSION.fullmatch(version_id):
        raise ValueError("Invalid dataset version")
    if not re.fullmatch(r"[a-f0-9]{32}", job["run_id"]):
        raise ValueError("Invalid training run id")
    cid = send(
        TRAINER,
        [
            "set -e",
            "umask 077",
            "command -v aws >/dev/null || (apt-get update && apt-get install -y awscli)",
            "mkdir -p /opt/fyp-autotraining",
            f"aws s3 cp s3://{BUCKET}/scripts/run_training.sh /opt/fyp-autotraining/run_training.sh >/dev/null",
            f"bash /opt/fyp-autotraining/run_training.sh {BUCKET} {REGION} {checksum} {version_id} {job['run_id']}",
        ],
        43200,
    )
    return {"ready": True, "command_id": cid}


def poll(command_id):
    status = read_command(TRAINER, command_id)["Status"]
    if status == "Success":
        return {"done": True}
    if status in ("Failed", "TimedOut", "Cancelled", "DeliveryTimedOut"):
        raise RuntimeError("GPU training SSM status: " + status)
    return {"done": False}


def finish(job):
    require_owner(job)
    checksum = job["checksum"]
    prefix = f"candidates/{checksum}/{job['run_id']}"
    result_key = f"{prefix}/result.json"
    result = json.loads(S3.get_object(Bucket=BUCKET, Key=result_key)["Body"].read())
    if (
        result.get("dataset_checksum") != checksum
        or result.get("status") != "completed"
    ):
        raise ValueError("Candidate manifest missing or invalid")
    if (
        result.get("dataset_version") != job["version_id"]
        or result.get("run_id") != job["run_id"]
        or result.get("adapter_uri") != f"s3://{BUCKET}/{prefix}/adapter/"
    ):
        raise ValueError("Candidate identity does not match this execution")
    metrics = result.get("metrics", {})
    if (
        not result.get("configuration")
        or metrics.get("valid_examples", 0) < 1
        or not isinstance(metrics.get("final_score_mae"), (int, float))
        or not math.isfinite(metrics["final_score_mae"])
    ):
        raise ValueError("Candidate has no valid evaluation metrics or configuration")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        S3.head_object(Bucket=BUCKET, Key=f"{prefix}/adapter/{name}")
    marker = {
        "pk": "latest",
        "checksum": checksum,
        "version_id": job["version_id"],
        "adapter_uri": result["adapter_uri"],
        "completed_at": int(time.time()),
        "run_id": job["run_id"],
        "result_key": result_key,
    }
    DB.put_item(Item={**marker, "pk": "success#" + checksum})
    DB.put_item(Item=marker)
    record(
        job, "candidate_saved", result_key=result_key, adapter_uri=result["adapter_uri"]
    )
    return {"status": "candidate_saved"}


def stop():
    current = EC2.describe_instances(InstanceIds=[TRAINER])["Reservations"][0][
        "Instances"
    ][0]["State"]["Name"]
    if current in ("running", "pending"):
        EC2.stop_instances(InstanceIds=[TRAINER])
    # Keep the lock until the guest and its SSM training process have stopped.
    EC2.get_waiter("instance_stopped").wait(
        InstanceIds=[TRAINER],
        WaiterConfig={"Delay": 5, "MaxAttempts": 90},
    )
    return {"stopped": True}


def cleanup(execution_id, error=None):
    job = get_item("lock")
    if job.get("execution_id") != execution_id:
        return {"ignored": "execution_does_not_own_gpu"}
    # Lease lasts longer than Lambda's maximum execution duration, so two
    # cleanups cannot race a new owner through stop/wait/unlock.
    token = claim_cleanup(job)
    try:
        result = stop()
    except Exception:
        release_cleanup(job, token)
        raise
    saved = get_item("run#" + job["run_id"])
    status = (
        "completed"
        if saved.get("status") in ("candidate_saved", "completed")
        else "failed"
    )
    record(job, status, error=error or {}, result_key=saved.get("result_key", ""))
    unlock(job)
    return result


def watchdog():
    job = get_item("lock")
    if not job:
        return {"idle": True}
    expired = time.time() - int(job["started_at"]) >= 14 * 3600
    if not expired:
        status = boto3.client("stepfunctions").describe_execution(
            executionArn=job["execution_id"],
        )["status"]
        if status == "RUNNING":
            return {"running": True}
    return cleanup(job["execution_id"], {"Error": "AbandonedOrOverdueExecution"})
