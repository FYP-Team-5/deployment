#!/usr/bin/env bash
# Called through SSM on the backend host. SeaweedFS stays private.
set -Eeuo pipefail
umask 077
bucket="$1"
region="$2"
temp="$(mktemp)"
trap 'rm -f "$temp"' EXIT

# Retrieve immutable active-version metadata from the backend's own DB.
meta="$(docker compose --project-directory /opt/fyp/backend exec -T api python -c '
import json
from app.config import Settings
from app.db.grading_repository import PostgresGradingRepository
from app.db.dataset_repository import DatasetRepository
s=Settings()
row,m=DatasetRepository(PostgresGradingRepository(s.database_url)).get_version()
if row["status"]!="ready": raise ValueError("No ready version")
print(json.dumps({"version_id":m["id"],"checksum":m["checksum"],"row_count":m["row_count"]}))
')"
read -r version checksum rows < <(printf '%s' "$meta" | python3 -c '
import json,sys
m=json.load(sys.stdin)
print(m["version_id"],m["checksum"],m["row_count"])
')
[[ "$version" =~ ^[a-f0-9-]{36}$ ]] || exit 2
[[ "$checksum" =~ ^[a-f0-9]{64}$ ]] || exit 2

# Transfer the exact version with the existing SHA-256 verifying reader.
docker compose --project-directory /opt/fyp/backend exec -T api python - "$version" <<'PY' > "$temp"
import shutil
import sys
from app.config import Settings
from app.db.grading_repository import PostgresGradingRepository
from app.db.dataset_repository import DatasetRepository
from app.service.dataset_storage import DatasetStorage
s=Settings()
row,m=DatasetRepository(PostgresGradingRepository(s.database_url)).get_version(sys.argv[1])
if row["status"]!="ready": raise RuntimeError("Version no longer ready")
reader=DatasetStorage(s).download_csv(row["object_key"],row["checksum"],row["byte_size"])
try:
    shutil.copyfileobj(reader,sys.stdout.buffer)
finally:
    reader.close()
PY

[[ "$(sha256sum "$temp" | cut -d ' ' -f1)" == "$checksum" ]] || {
    echo 'Checksum mismatch' >&2
    exit 1
}
aws s3 cp --region "$region" "$temp" "s3://$bucket/datasets/$checksum/dataset.csv" >/dev/null
printf '{"version_id":"%s","checksum":"%s","row_count":%s}\n' "$version" "$checksum" "$rows"
