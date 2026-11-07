# Nightly training, Hugging Face candidates and instructor deployment

EventBridge Scheduler starts the Standard Step Functions workflow at **00:10 Asia/Singapore**. The schedule is disabled by default until the first acceptance run succeeds.

## Infrastructure decisions

| Component | Decision | Purpose |
| --- | --- | --- |
| DynamoDB | Removed from configuration | Small control records use the existing training S3 bucket. |
| Lambda | Keep one small coordinator | Checks manifests, owns conditional locks, dispatches SSM commands and reconciles failures. Each invocation is short; Step Functions does all waiting. |
| S3 | Keep one private, encrypted bucket | Durable dataset staging, scripts, recovery artifacts, publication receipts and control JSON. Training and HF upload can recover independently. |
| PostgreSQL | Reuse backend database | Stores instructor-visible candidates, metrics, explicit deployment requests and the active model pointer. |
| GPU EC2 | Keep separate and stopped while idle | Training runs on `g6e.xlarge`; HF publishing runs on the existing backend CPU host **after GPU shutdown**. |

Lambda could be replaced with more native Step Functions states and backend scripts. Keeping a small coordinator avoids moving cloud-control credentials and failure recovery into the quiz application. S3 can also be replaced with SeaweedFS, but that would require new private connectivity/credentials and couple artifacts and coordination to the backend host. Neither change is necessary for this workflow.

GPU training hours and the persistent 200 GB EBS volume are the main infrastructure items to review; removing tiny control records alone is not a large cost saving. No paid resources are changed by editing this repository.

## Workflow and recovery

1. SSM reads the backend's latest **ready immutable** SeaweedFS dataset and stages its checksum-addressed CSV in S3. An existing snapshot is reused without reading SeaweedFS again.
2. A successfully published checksum skips. A completed training run whose HF upload or REST notification failed goes directly to publication, without starting the GPU.
3. A new checksum claims `control/lock.json` using S3 `If-None-Match` / `If-Match` conditional writes. A single GPU owner and a separate short cleanup lease serialize overlap and delayed cleanup events.
4. Start EC2, wait for SSM readiness, download/validate/train/evaluate, and save an evaluated adapter plus manifest in S3.
5. Request EC2 stop, poll its state in Step Functions, and release the lock only after **stopped** is confirmed. Lambda never sleeps through EC2 shutdown or SSM commands; its timeout is 120 seconds.
6. On the existing backend host, `publish.sh`/`publish.py` download the evaluated adapter and upload it to **`SmuFypTeam5/GradingQlora`**, branch **`candidate-<run-id>`**. The production `main` branch is untouched. Only adapter files, configuration and evaluation metadata are uploaded; student CSVs and per-student predictions are excluded.
7. Save the exact HF commit and an S3 publication receipt, then `POST /api/v1/training-runs` to the backend with a machine-only token. Retries reuse that commit and replay the idempotent callback. Only an acknowledged callback creates the published-checksum marker.
8. Instructors open **Dashboard → Trained models**, compare candidate and base-model metrics, inspect configuration/history and explicitly request deployment. Uploading/reporting alone never activates a model.

The backend stores the exact dataset version/checksum, run id, HF branch/commit, evaluation timestamp, metrics and effective configuration. Deployment is durable: a database worker downloads the pinned HF revision, checks base model/rank/files, loads a separate vLLM adapter alias, and tests positive/negative grading responses. The active pointer changes only after checks pass. A failure preserves the prior pointer and is visible with a retry action. Previous candidates can be selected again for rollback.

The training notebook emits binary criterion arrays. The backend supports this contract explicitly: validated labels map to the saved rubric weights and criterion ids; feedback lists met/unmet criteria. **Few-shot grading continues using the original configured model.** Each accepted grading job snapshots its model name, output contract and HF commit, including partial submissions/retries, so a deployment does not switch work already accepted.

The serving filesystem resolver reloads approved adapters after vLLM restarts. If the host loses its adapter cache, grading restores the job's pinned files from HF. PostgreSQL and HF remain required for this recovery.

## Training configuration

The pinned reference is `Rubrics/Qlora Comparison/qlora (48r,96a).ipynb` at model-repo commit `8a2b94ec95754d875dc546461ef15dc2a279ce5a`:

- `Qwen/Qwen3.5-9B`, LoRA **rank 48 / alpha 96**, dropout 0.05, learning rate 5e-5, up to 3 epochs, microbatch 1 × accumulation 8, seeds 42.
- NF4 4-bit/double quantization, bf16 when supported, paged 8-bit AdamW, gradient checkpointing, completion-only loss, early stopping and validation-loss checkpoint selection.
- Question-group 80/10/10 split; 4,096-token training/validation limit (oversized examples fail instead of truncating labels). Partial-credit rows and missing desired answers are counted and excluded; at least ten compatible unique questions are required.
- Strict backend eight-column CSV schema, JSON arrays, finite bounded scores, positive maxima and consistent totals. UTF-8 BOM, quoted commas/newlines and repeated question ids are preserved.
- Python 3.12 is installed with uv 0.9.10. Exact resolved package versions, model revision and notebook hash are saved with each candidate. A real GPU compatibility run is still required.

Rank 48 / alpha 96 had the highest exact-array accuracy (0.804627) and F1 (0.913386) in the project's [saved rank comparison](https://github.com/FYP-Team-5/model/blob/8a2b94ec95754d875dc546461ef15dc2a279ce5a/Rubrics/Qlora%20Comparison/qlora_evaluation_comparison.csv). This is the best tested configuration for that experiment, not a guarantee for future data. The current [Unsloth guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) favors bf16 LoRA because of quantization differences; review each candidate's actual evaluation before deployment.

## Configure credentials and serving

1. Deploy the updated **backend and frontend** application repositories. Backend schema initialization uses `create_all()`; no migration scripts are added.
2. Create existing SSM **SecureString** parameters in the deployment region (names configurable via Terraform):
   - `/fyp/training/hf-token`: a token with write access to `SmuFypTeam5/GradingQlora`.
   - `/fyp/training/callback-token`: a strong random machine credential.
   Terraform grants the existing backend EC2 role access to these names; secret values never enter Terraform configuration/state. A custom KMS key also needs an appropriate decrypt policy.
3. Set backend `.env` values, then rebuild/restart its Compose service:
   - `TRAINING_CALLBACK_TOKEN`: exactly the callback SecureString value.
   - `TRAINING_HF_REPO=SmuFypTeam5/GradingQlora`.
   - `TRAINING_HF_TOKEN`: repo-scoped read access if private HF commits require it.
   - `MODEL_ADAPTER_DIRECTORY=/mnt/llm-data/adapters` (shared with the host at the same path).
4. Install the updated serving wrapper at `/opt/qwen-vllm/Script/serve_qwen35_vllm.sh` on the serving host. Add a systemd override to `qwen-vllm.service`:

   ```ini
   [Service]
   Environment=FYP_LORA_RUNTIME_ENABLED=true
   Environment=FYP_LORA_ADAPTER_DIRECTORY=/mnt/llm-data/adapters
   Environment=FYP_VLLM_MAX_CPU_LORAS=4
   ```

   Reload systemd and restart vLLM during a suitable window. The opt-in enables runtime LoRA loading and the filesystem resolver. TCP 8000 must stay private; browsers use the authenticated backend deployment API. This does **not** require editing EC2 user-data or replacing the serving instance.

The callback uses the serving instance's private IPv4 address on port 8002. HF upload is performed by its SSM-managed host, not Lambda or the GPU. Both host package installation and HF access require outbound network access.

## Deploy and enable

```bash
terraform init
terraform validate
terraform plan -out=autotraining.tfplan
# Review the plan: do not replace aws_instance.llm and its local data.
terraform apply autotraining.tfplan

aws stepfunctions start-execution \
  --region "$(terraform output -raw region)" \
  --state-machine-arn "$(terraform output -raw autotraining_state_machine_arn)" \
  --input '{}'
```

Verify a full GPU run, EC2 stop, HF candidate branch, callback, instructor metrics, a passing explicit deployment, a failing deployment leaving production intact, and an unchanged-checksum skip. Then persist `autotraining_schedule_enabled = true` in a local ignored `.auto.tfvars` file and apply a reviewed plan. Set it false to disable nightly starts; the recovery watchdog stays enabled. Do not apply Terraform during training: it manages the trainer's idle state as stopped.

**Upgrading the DynamoDB prototype:** stop all executions and confirm the trainer is stopped. Back up/export any existing DynamoDB history and carry completed checksums/artifact identities into S3 control records before removing the table. The new configuration removes the DynamoDB resource, so a plan against an already-deployed prototype can destroy that table. Never discard a live GPU ownership record or apply that destructive change without preserving required history. This repository change does not run Terraform apply.

## Storage, diagnostics and verification

- `datasets/<sha256>/dataset.csv`: immutable staging; SeaweedFS remains the dataset system of record.
- `candidates/<sha256>/<run-id>/`: adapter, effective configuration and evaluated `result.json`. After HF publication and an acknowledged backend callback, objects receive a `published=true` tag. A lifecycle rule expires published staging objects older than 30 days (noncurrent versions after 30 noncurrent days); HF retains the pinned published commit. Unpublished artifacts are excluded.
- `publications/<run-id>/result.json`: exact HF commit, callback payload and acknowledgement.
- `control/{lock,latest}.json`, `control/runs/<run-id>.json`, `control/trained/<checksum>.json`, `control/success/<checksum>.json`: small JSON coordination/history records. Do not expire current control records or unpublished artifacts.
- CloudWatch: `/fyp/<name>/training` and `/aws/lambda/<name>-training-coordinator`. HF publication SSM commands expose status/errors on the backend host.

Terminal Step Functions events trigger owner-checked cleanup. An independent 15-minute watchdog reconciles missed events and executions older than 14 hours; the guest has a best-effort 15-hour shutdown timer. SSM training timeout is 12 hours. AWS control-plane/IAM outages can delay cleanup; inspect the GPU and retain the lock until stop is confirmed. S3 bucket versioning preserves overwrites but has storage costs. Noncurrent control versions expire after 30 days; current history/locks and publication receipts remain. Abandoned multipart uploads are aborted after seven days. A stopped GPU still incurs EBS charges.

```bash
python -m pip install boto3 ruff
python -m unittest discover -s autotraining/tests -v
ruff check autotraining
bash -n autotraining/export_backend.sh
bash -n autotraining/run_training.sh
bash -n autotraining/publish.sh
bash -n Script/serve_qwen35_vllm.sh
terraform fmt -check autotraining-*.tf
terraform validate
```

Backend tests cover callback authentication/idempotency, instructor authorization, stale deployment rejection, failure isolation, strict binary outputs and weighted grading/model snapshots. Frontend ESLint, TypeScript and the production build cover the instructor screen. Live AWS/GPU/HF/vLLM acceptance remains necessary; offline checks cannot establish capacity, credentials or runtime compatibility.

References: [S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html), [HF uploads](https://huggingface.co/docs/huggingface_hub/guides/upload), [vLLM runtime LoRA/resolvers](https://docs.vllm.ai/en/stable/features/lora/), [Step Functions terminal events](https://docs.aws.amazon.com/step-functions/latest/dg/eventbridge-integration.html).
