#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstraps the two public FYP repositories on an Ubuntu EC2 instance.
# Run as root after Docker Engine and the vLLM service have been started.
umask 077
install -d -m 0755 /opt/fyp

for project in backend frontend; do
    target="/opt/fyp/$project"
    if [[ ! -d "$target/.git" ]]; then
        git clone --depth 1 "https://github.com/FYP-Team-5/$project.git" "$target"
    fi
    if [[ ! -f "$target/.env" ]]; then
        cp "$target/.env.example" "$target/.env"
    fi
    chmod 0600 "$target/.env"
done

set_env() {
    local path="$1" name="$2" value="$3"
    if grep -q "^$name=" "$path"; then
        sed -i "s|^$name=.*|$name=$value|" "$path"
    else
        printf '%s=%s\n' "$name" "$value" >> "$path"
    fi
}

# EC2 requires IMDSv2 and assigns a public IP to this instance.
token="$(curl --fail --silent --show-error --retry 4 \
    --request PUT http://169.254.169.254/latest/api/token \
    --header 'X-aws-ec2-metadata-token-ttl-seconds: 60')"
public_ip="$(curl --fail --silent --show-error --retry 4 \
    --header "X-aws-ec2-metadata-token: $token" \
    http://169.254.169.254/latest/meta-data/public-ipv4)"

backend_env=/opt/fyp/backend/.env
frontend_env=/opt/fyp/frontend/.env

# Generate unique boot-time secrets; do not store them in Git or Terraform state.
set_env "$backend_env" POSTGRES_PASSWORD "$(openssl rand -hex 24)"
set_env "$backend_env" JWT_SECRET "$(openssl rand -hex 32)"
api_key="$(openssl rand -hex 32)"
set_env "$backend_env" API_KEY "$api_key"
set_env "$backend_env" POSTGRES_PORT "127.0.0.1:5433"
set_env "$backend_env" CORS_ORIGINS "http://$public_ip:3000"
set_env "$backend_env" LLM_URL "http://host.docker.internal:8000/v1/chat/completions"
set_env "$backend_env" LLM_MODEL "qwen35-9b-grading-qlora"

# NEXT_PUBLIC variables are embedded in the frontend image during the build.
set_env "$frontend_env" NEXT_PUBLIC_API_URL "http://$public_ip:8002"
set_env "$frontend_env" NEXT_PUBLIC_GRADING_API_KEY "$api_key"
set_env "$frontend_env" FRONTEND_BIND_IP "0.0.0.0"
set_env "$frontend_env" FRONTEND_PORT "3000"

# This is a demo setup: the frontend embeds its API key in public JavaScript.
# Add an HTTPS proxy and stronger server-side authorization before production.
docker compose --project-directory /opt/fyp/backend up --build -d
docker compose --project-directory /opt/fyp/frontend up --build -d

echo "FYP frontend: http://$public_ip:3000"
echo "FYP backend API: http://$public_ip:8002/health"
