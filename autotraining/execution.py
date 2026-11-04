"""EC2/SSM operations for the nightly QLoRA training state machine."""
import json
import os
import time

import boto3

from transfer import DB, DIGEST, send, read_command, unlock

EC2 = boto3.client("ec2")
SSM = boto3.client("ssm")
S3 = boto3.client("s3")
TRAINER = os.environ["TRAINING_INSTANCE_ID"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
REGION = os.environ["AWS_REGION"]


def start():
    current = EC2.describe_instances(InstanceIds=[TRAINER])["Reservations"][0]["Instances"][0]["State"]["Name"]
    if current == "stopped":
        EC2.start_instances(InstanceIds=[TRAINER])
    elif current not in ("pending", "running"):
        raise RuntimeError("GPU instance cannot be started: " + current)
    return {"instance_id": TRAINER}


def launch(job, attempt=0):
    online = SSM.describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [TRAINER]}]
    )["InstanceInformationList"]
    if not any(info.get("PingStatus") == "Online" for info in online):
        return {"ready": False, "attempt": attempt + 1}
    checksum = job["checksum"]
    if not DIGEST.fullmatch(checksum):
        raise ValueError("Invalid dataset checksum")
    cid = send(TRAINER, [
        "set -e",
        "umask 077",
        "command -v aws >/dev/null || (apt-get update && apt-get install -y awscli)",
        "mkdir -p /opt/fyp-autotraining",
        f"aws s3 cp s3://{BUCKET}/scripts/run_training.sh /opt/fyp-autotraining/run_training.sh >/dev/null",
        f"bash /opt/fyp-autotraining/run_training.sh {BUCKET} {REGION} {checksum}",
    ], 43200)
    return {"ready": True, "command_id": cid}


def poll(command_id):
    status = read_command(TRAINER, command_id)["Status"]
    if status == "Success":
        return {"done": True}
    if status in ("Failed", "TimedOut", "Cancelled", "DeliveryTimedOut"):
        raise RuntimeError("GPU training SSM status: " + status)
    return {"done": False}


def finish(job):
    checksum = job["checksum"]
    result_key = f"candidates/{checksum}/result.json"
    result = json.loads(S3.get_object(
        Bucket=BUCKET, Key=result_key
    )["Body"].read())
    if result.get("dataset_checksum") != checksum or result.get("status") != "completed":
        raise ValueError("Candidate manifest missing or invalid")
    if not result.get("metrics") or not result.get("adapter_uri"):
        raise ValueError("Candidate has no evaluation metrics or adapter location")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        S3.head_object(
            Bucket=BUCKET, Key=f"candidates/{checksum}/adapter/{name}"
        )
    DB.put_item(Item={
        "pk": "latest", "checksum": checksum, "version_id": job["version_id"],
        "adapter_uri": result["adapter_uri"], "completed_at": int(time.time()),
    })
    return {"status": "candidate_saved"}


def stop():
    current = EC2.describe_instances(InstanceIds=[TRAINER])["Reservations"][0]["Instances"][0]["State"]["Name"]
    if current in ("running", "pending"):
        EC2.stop_instances(InstanceIds=[TRAINER])
    return {"stop_requested": True}


def failure(job):
    result = stop()
    unlock(job.get("checksum", ""))
    return result
