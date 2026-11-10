"""Small S3 control records with conditional writes; no separate database."""

import json
import os
import time
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

AWS_CONFIG = Config(
    connect_timeout=3,
    read_timeout=8,
    retries={"mode": "standard", "total_max_attempts": 2},
)
S3 = boto3.client("s3", config=AWS_CONFIG)
BUCKET = os.environ["ARTIFACT_BUCKET"]
CONFLICTS = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}


class StateConflict(RuntimeError):
    """The record changed after it was read."""


def read(key):
    try:
        response = S3.get_object(Bucket=BUCKET, Key=f"control/{key}.json")
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {}, None
        raise
    body = response["Body"]
    try:
        return json.loads(body.read()), response["ETag"]
    finally:
        body.close()


def write(key, value, etag=None, create=False):
    conditions = {"IfNoneMatch": "*"} if create else ({"IfMatch": etag} if etag else {})
    try:
        S3.put_object(
            Bucket=BUCKET,
            Key=f"control/{key}.json",
            Body=json.dumps(value, allow_nan=False, separators=(",", ":")).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
            **conditions,
        )
    except ClientError as error:
        if error.response["Error"]["Code"] in CONFLICTS:
            raise StateConflict(key) from error
        raise


def get_item(key):
    return read(key)[0]


def claim(job):
    lock, etag = read("lock")
    if lock.get("execution_id"):
        return lock.get("execution_id") == job["execution_id"]
    write("lock", job, etag=etag, create=etag is None)
    return True


def owns(job):
    lock = get_item("lock")
    return lock.get("execution_id") == job["execution_id"] and not lock.get(
        "cleanup_token"
    )


def unlock(job):
    lock, etag = read("lock")
    if lock.get("execution_id") != job["execution_id"]:
        return
    # A fresh nonce prevents a delayed CAS from matching a later idle generation.
    write("lock", {"idle": True, "generation": uuid.uuid4().hex}, etag=etag)


def claim_cleanup(job):
    lock, etag = read("lock")
    if lock.get("execution_id") != job["execution_id"]:
        raise StateConflict("GPU owner changed")
    if lock.get("cleanup_until", 0) >= int(time.time()):
        raise StateConflict("Cleanup already running")
    token = uuid.uuid4().hex
    write(
        "lock",
        {**lock, "cleanup_token": token, "cleanup_until": int(time.time()) + 150},
        etag=etag,
    )
    return token


def release_cleanup(job, token):
    lock, etag = read("lock")
    if (
        lock.get("execution_id") != job["execution_id"]
        or lock.get("cleanup_token") != token
    ):
        return
    cleaned = {
        key: value
        for key, value in lock.items()
        if key not in ("cleanup_token", "cleanup_until")
    }
    write("lock", cleaned, etag=etag)


def record(job, status, **details):
    write(
        "runs/" + job["run_id"],
        {**job, "status": status, "updated_at": int(time.time()), **details},
    )
