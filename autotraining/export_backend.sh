#!/usr/bin/env bash
set -Eeuo pipefail
# Run on the backend EC2 instance with SSM, not across the public internet.
# The backend service and SeaweedFS remain private on their Docker network.
bucket="$1"
region="$2"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
docker compose --project-directory /opt/fyp/backend exec -T api python - <<'PY' > "$tmp"
import json
import sys
from app.config import Settings
from app.db.grading_repository import PostgresGradingRepository
from app.db.dataset_repository import DatasetRepository
from app.service.dataset_storage import DatasetStorage
s = Settings()
repo = DatasetRepository(PostgresGradingRepository(s.database_url))
row, metadata = repo.get_version()
assert row["status"] == "ready"
data = DatasetStorage(s).download_csv(
    row["object_key"], row["checksum"], row["byte_size"]
)
sys.stdout.buffer.write(data.read())
data.close()
PY
checksum="$(sha256sum "$tmp" | cut -d ' ' -f 1)"
aws s3 cp --region "$region" "$tmp" "s3://$bucket/datasets/$checksum/dataset.csv"
printf '{"checksum":"%s"}\n' "$checksum"
