#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

local_port="${1:-8000}"
if [[ ! "$local_port" =~ ^[0-9]+$ ]] || (( local_port < 1 || local_port > 65535 )); then
    echo "Usage: $0 [local-port] (1-65535)" >&2
    exit 2
fi

for executable in terraform aws session-manager-plugin; do
    command -v "$executable" >/dev/null 2>&1 || {
        echo "ERROR: $executable is not installed or not on PATH." >&2
        exit 1
    }
done

instance_id="$(terraform output -raw instance_id 2>/dev/null)" || {
    echo "ERROR: no Terraform instance_id output. Apply the deployment first." >&2
    exit 1
}
region="$(terraform output -raw region)"

echo "Forwarding EC2 vLLM port 8000 to 127.0.0.1:$local_port. Keep this terminal open."
exec aws ssm start-session \
    --region "$region" \
    --target "$instance_id" \
    --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"8000\"],\"localPortNumber\":[\"$local_port\"]}"
