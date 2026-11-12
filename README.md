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

Terraform selects the first available standard Availability Zone that offers
both serving and training instance types. Set `availability_zone` in local
tfvars to pin another supported zone, for example `availability_zone = "us-east-1a"`.
An unsupported override or no shared zone fails during planning. Instance-type
support does not guarantee spare EC2 capacity at launch time.

Before applying, create the required SSM SecureStrings for the Hugging Face
upload token and training callback credential in that region. See the
[credential setup](autotraining/README.md#setup). Bootstrap reads the callback
credential directly from the parameter named by
`autotraining_callback_token_parameter` and writes it to the backend's private
Compose `.env` before starting the application. Its value stays outside
Terraform state and logs. Missing or invalid credentials fail bootstrap.

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

Fresh stacks wait 15 minutes after enabling S3 versioning before uploading
training scripts, following AWS's propagation guidance. This prevents temporary
`GetObjectTagging: NoSuchKey` failures during initial upload. The wait repeats
when the versioning resource is replaced or reconfigured, including bucket
replacement; ordinary script updates do not repeat it.

After startup:

- Frontend: `http://<public_ip>:3000`
- Backend: `http://<public_ip>:8002/health`
- vLLM: private port 8000; use `./connect-vllm.sh` for access.

The bootstrap creates application `.env` files on the serving host, including the callback credential and workflow ARN. Nightly scheduling is **disabled by default**.

New hosts enable runtime adapters automatically. Systemd waits up to 40 minutes
for vLLM health, all three configured model aliases, and the runtime adapter
loading API before completing startup and bootstrapping the application. This
check runs again on service restarts. A failed check fails service startup and
is logged in `journalctl -u qwen-vllm.service`; systemd retries the service.
Terraform finishes creating EC2 before cloud-init finishes: use
`sudo cloud-init status --wait` on the host to confirm bootstrap completion.
Training and candidate deployment remain separate instructor actions.

Bootstrap package installations wait up to ten minutes for unattended upgrades
to release the dpkg lock, including the CloudWatch agent package installation.
Training-host dependency installs and the SSM export/launch setup use the same
ten-minute lock wait. The export command and workflow polling budgets include
time for that wait, command delivery and the export itself.
Background upgrades can also reexecute systemd while vLLM starts. Bootstrap
retries temporary systemd connection failures up to six times, five seconds apart,
and confirms the service is active after its readiness check before starting
the backend/frontend. Other service failures stop bootstrap immediately.
CloudWatch is restarted and checked after vLLM is ready so its journal reader
reattaches after any systemd/journald upgrade during startup.

`Script/bootstrap_apps.sh` is the shared application bootstrap for Terraform and
standalone setup. Terraform installs this exact script and creates
`/usr/local/sbin/bootstrap-fyp-apps` to supply the workflow ARN, callback parameter
and AWS region. Standalone setup requires `TRAINING_STATE_MACHINE_ARN` before it
changes application files, for example:

```bash
sudo env TRAINING_STATE_MACHINE_ARN='<autotraining_state_machine_arn output>' \
  AWS_REGION=us-east-1 \
  TRAINING_CALLBACK_TOKEN_PARAMETER=/fyp/training/callback-token \
  bash Script/bootstrap_apps.sh
```

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
sudo install -o ubuntu -g ubuntu -m 0644 Script/check_vllm_ready.py /opt/qwen-vllm/Script/check_vllm_ready.py
sudo apt-get update
sudo apt-get install -y awscli
sudo mkdir -p /etc/systemd/system/qwen-vllm.service.d
sudo tee /etc/systemd/system/qwen-vllm.service.d/20-adapters.conf >/dev/null <<'UNIT'
[Service]
Environment=HF_TOKEN_SSM_PARAMETER=/fyp/training/hf-token
Environment=HF_TOKEN_SSM_REGION=us-east-1
Environment=FYP_LORA_RUNTIME_ENABLED=true
Environment=FYP_LORA_ADAPTER_DIRECTORY=/mnt/llm-data/adapters
Environment=FYP_VLLM_MAX_CPU_LORAS=4
EnvironmentFile=-/etc/qwen-vllm.env
ExecStartPost=/usr/bin/python3 /opt/qwen-vllm/Script/check_vllm_ready.py
TimeoutStartSec=45min
UNIT
sudo install -d -o ubuntu -g ubuntu /mnt/llm-data/adapters
sudo systemctl daemon-reload
sudo systemctl restart qwen-vllm.service
sudo journalctl -u qwen-vllm.service -f
```

By default, serving reuses the existing `/fyp/training/hf-token` **SecureString**
in the deployment region. Ensure its Hugging Face token can read both adapter
repos. A separate serving-only read token can be selected with the variable below.
Terraform passes its name and region to systemd and grants the serving instance
role `ssm:GetParameter` for that parameter. At each service start the wrapper
retrieves it with `--with-decryption` and exports `HF_TOKEN` only to the running
process. A configured SSM lookup failure stops startup rather than falling back
to an old token. The token is not written to a file or read into Terraform state.
See [AWS SecureString access](https://docs.aws.amazon.com/systems-manager/latest/userguide/secure-string-parameter-kms-encryption.html).

To set the path explicitly or select another existing parameter, use local tfvars:

```hcl
vllm_hf_token_parameter = "/fyp/training/hf-token"
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

For `Cannot retrieve the serving HF token`, check the preceding AWS CLI error.
`ParameterNotFound` means the configured name does not exist in that region;
`AccessDeniedException` requires checking the instance role and, for a custom
KMS key, its key policy. The wrapper prints the selected name, region and exit
code without printing the token. The SSM startup default reuses the training
parameter so an additional `/fyp/serving/hf-token` does not need to be created.

Once startup completes, verify both adapter aliases are listed:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
```

Runtime serving loads `bitsandbytes,lora_filesystem_resolver` through
`VLLM_PLUGINS`. The bitsandbytes plugin is required by the pinned serving stack;
allowing only the filesystem resolver prevents its quantization registration
and causes `Unknown quantization method: bitsandbytes` at startup. See the
[vLLM plugin loader](https://docs.vllm.ai/en/v0.29.0/api/vllm/plugins/) and
[bitsandbytes plugin](https://github.com/vllm-project/vllm-bnb-plugin).

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
