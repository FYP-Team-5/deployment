# FYP deployment

Terraform deploys the quiz application and its nightly Qwen3.5-9B retraining workflow.

| Component | Role |
| --- | --- |
| Serving EC2 | Frontend, backend, PostgreSQL, SeaweedFS and vLLM |
| Training EC2 | Dedicated instance matching the serving type (`g5.xlarge` by default), stopped while idle |
| EventBridge Scheduler | Starts retraining daily at **00:10 SGT** when enabled |
| Step Functions + Lambda + SSM | Coordinate training, publication and GPU cleanup |
| Private S3 bucket | Dataset staging, artifacts, publication receipts and run state |
| Hugging Face | Public candidate adapter branches in `SmuFypTeam5/GradingQlora` |

See the [retraining workflow diagram and setup](autotraining/README.md).

## Run Terraform

Requirements: Terraform 1.5+, authenticated AWS CLI, and sufficient GPU quota in the selected region (`us-east-1` by default).

From this directory:

```bash
terraform init
terraform fmt
terraform validate
terraform plan -out=deployment.tfplan
terraform apply deployment.tfplan
terraform output
```

Review the plan before applying. Replacing `aws_instance.llm` deletes its instance-local application data. Do not apply while training is running.

After startup:

- Frontend: `http://<public_ip>:3000`
- Backend: `http://<public_ip>:8002/health`
- vLLM: private port 8000; use `./connect-vllm.sh` for access.

The bootstrap creates application `.env` files on the serving host. Complete the [retraining credentials and serving setup](autotraining/README.md#setup) before the first training run. Nightly scheduling is **disabled by default**.

## Grading and feedback on one vLLM instance

The serving wrapper loads `Qwen/Qwen3.5-9B` once and registers two LoRA adapters:

| HF adapter | vLLM model alias | Backend setting |
| --- | --- | --- |
| `SmuFypTeam5/GradingQlora` | `qwen35-9b-grading-qlora` | `LLM_MODEL` |
| `SmuFypTeam5/feedback` | `qwen35-9b-feedback-qlora` | `LLM_FEEDBACK_MODEL` |

Both aliases use the same port 8000 and `/v1/chat/completions` endpoint. The
backend selects the grading adapter to produce binary criterion labels, then the
feedback adapter to explain the met/unmet decisions. `qwen35-9b-base` also remains
available for requests without an adapter.

The wrapper uses `--enable-lora --lora-modules name=path name=path`, following
the [vLLM 0.29 LoRA serving format](https://docs.vllm.ai/en/v0.29.0/features/lora/).
Rank capacity stays at 64. The existing sequential workflow uses one adapter
per GPU batch (`FYP_VLLM_MAX_LORAS=1`) and caches up to four adapters in CPU memory
(`FYP_VLLM_MAX_CPU_LORAS=4`), so both aliases are available while preserving the
existing GPU memory settings. `max_loras` controls adapters **per batch**, not
the number of registered aliases; see the [vLLM configuration reference](https://docs.vllm.ai/en/v0.29.0/api/vllm/config/lora/).

For an existing serving host, run the following from an updated deployment
checkout **on that host**. This updates the serving files and restarts vLLM;
changing Terraform user data instead would replace this repository's EC2 instance.
First configure the existing SSM parameter and serving-role permission described
below. The service does not require a `.env` file.

```bash
sudo install -o ubuntu -g ubuntu -m 0755 Script/serve_qwen35_vllm.sh /opt/qwen-vllm/Script/serve_qwen35_vllm.sh
sudo install -o ubuntu -g ubuntu -m 0644 Script/requirements-vllm.txt /opt/qwen-vllm/Script/requirements-vllm.txt
sudo apt-get update
sudo apt-get install -y awscli
sudo mkdir -p /etc/systemd/system/qwen-vllm.service.d
sudo tee /etc/systemd/system/qwen-vllm.service.d/20-adapters.conf >/dev/null <<'UNIT'
[Service]
Environment=HF_TOKEN_SSM_PARAMETER=/fyp/serving/hf-token
Environment=HF_TOKEN_SSM_REGION=us-east-1
EnvironmentFile=-/etc/qwen-vllm.env
UNIT
sudo systemctl daemon-reload
sudo systemctl restart qwen-vllm.service
sudo journalctl -u qwen-vllm.service -f
```

Create `/fyp/serving/hf-token` in SSM Parameter Store as a **SecureString** in the
deployment region, with a Hugging Face token that can read both adapter repos.
Terraform passes its name and region to systemd and grants the serving instance
role `ssm:GetParameter` for that parameter. At each service start the wrapper
retrieves it with `--with-decryption` and exports `HF_TOKEN` only to the running
process. A configured SSM lookup failure stops startup rather than falling back
to an old token. The token is not written to a file or read into Terraform state.
See [AWS SecureString access](https://docs.aws.amazon.com/systems-manager/latest/userguide/secure-string-parameter-kms-encryption.html).

To use a different existing parameter, set only its identity in local tfvars:

```hcl
vllm_hf_token_parameter = "/fyp/serving/hf-token"
# Optional for a customer-managed KMS key; omit for the default aws/ssm key.
# vllm_hf_token_kms_key_arn = "arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"
```

The optional KMS permission is limited to the given key and parameter encryption
context; its key policy must also allow the serving role. An existing host's
drop-in must use the same parameter name and region as Terraform. Review the
Terraform plan for the IAM change; use the service-update procedure above to
avoid replacing the serving instance just to update user data.

`/etc/qwen-vllm.env` remains optional for model/path overrides. For local serving
without SSM, leave `HF_TOKEN_SSM_PARAMETER` unset and export `HF_TOKEN` directly.
For Terraform, `vllm_hf_token_parameter = ""` disables lookup and its dedicated
policy. When SSM is configured its value takes precedence over a direct token.

Once startup completes, verify both adapter aliases are listed:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
```

Backend environment settings must match the served aliases:

```dotenv
LLM_URL=http://host.docker.internal:8000/v1/chat/completions
LLM_MODEL=qwen35-9b-grading-qlora
LLM_GRADING_CONTRACT=criteria_array_v1
LLM_FEEDBACK_MODEL=qwen35-9b-feedback-qlora
```

Both app bootstrap scripts already write these values. After updating an
existing backend checkout with the split grading/feedback implementation and
these values in its `.env`, rebuild it with
`sudo docker compose --project-directory /opt/fyp/backend up --build -d api`.

If grading reports missing `score` and `feedback` fields in
`CriteriaGradingResult` for a criterion-only response, an older combined-result
parser is still being used. Update the backend checkout to the split grading/
feedback implementation and rebuild the API service with the command above.
Then use **Retry grading saved answers** on the failed submission. Completed
grades are retained; the grading model does not need to generate score or feedback.

## Operate

```bash
aws ssm start-session \
  --region "$(terraform output -raw region)" \
  --target "$(terraform output -raw instance_id)"
```

On the serving host:

```bash
sudo docker compose --project-directory /opt/fyp/backend logs -f
sudo journalctl -u qwen-vllm -f
```

Training execution status is available in Step Functions; training and coordinator logs are in CloudWatch.
