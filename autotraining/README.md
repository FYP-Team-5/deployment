# Automated nightly QLoRA retraining

This feature is implemented in the `feat/autotraining` branch of `FYP-Team-5/deployment`.
**Applying Terraform creates paid AWS resources.** The daily schedule is **disabled by default** for a safe first deployment. No existing backend/user-data or vLLM configuration is modified.

## Architecture

- EventBridge Scheduler: **00:10 Asia/Singapore** nightly, once enabled.
- Standard Step Functions: prepare, deduplicate, start GPU EC2, wait for SSM readiness, run training, poll, verify candidate, stop GPU. Failure branches also request EC2 stop.
- Lambda coordinator: three modules (`coordinator.py`, `transfer.py`, `execution.py`), packaged with Terraform's archive provider.
- Existing backend EC2: SSM executes `export_backend.sh` inside its host. The script uses the backend's **version ledger** and checksum-verifying SeaweedFS reader to retrieve a specific immutable ready CSV version. No public instructor authentication or SeaweedFS ingress is required.
- Private AWS S3 bucket: checksum-addressed dataset snapshots, training runner scripts, evaluated candidate adapter, and `result.json`.
- DynamoDB: permanent success markers for every trained checksum, per-execution run history, and one conditional GPU lock. Locks are released only after EC2 confirms it has stopped. Cleanup has a separate 11-minute lease to serialize duplicate events; an interrupted cleanup can be recovered after that lease expires.
- Separate EC2 GPU instance: default `g6e.xlarge`, encrypted 200 GB root EBS, SSM-only management, no inbound security group rules. Terraform manages its initial state as stopped.
- CloudWatch: Lambda execution logs and streaming training SSM output.
- Recovery: terminal Step Functions events trigger cleanup for failures, timeouts and manual aborts. An independent 15-minute watchdog reconciles missed events and stops runs older than 14 hours, including failures before SSM is ready. The guest also has a best-effort 15-hour shutdown timer.

The runner installs Python 3.12 with pinned uv 0.9.10, rather than relying on Ubuntu 22.04's Python 3.10. It installs boto3 explicitly for artifact uploads. The PyTorch, Transformers and TRL package versions are pinned in `requirements.txt`; compatibility still requires a live GPU test. The default training reference is the committed notebook `Rubrics/Qlora Comparison/qlora (48r,96a).ipynb` from `FYP-Team-5/model`, pinned to commit `8a2b94ec95754d875dc546461ef15dc2a279ce5a`.

## Training contract

- Base model: `Qwen/Qwen3.5-9B`.
- QLoRA: rank **48**, alpha **96**, dropout **0.05**, learning rate **5e-5**, 3 max epochs, microbatch **1** × gradient accumulation **8**, seed **42**, question-group split seed **42**.
- Rank/alpha are selected from the project's [saved comparison](https://github.com/FYP-Team-5/model/blob/8a2b94ec95754d875dc546461ef15dc2a279ce5a/Rubrics/Qlora%20Comparison/qlora_evaluation_comparison.csv): rank 48 / alpha 96 achieved the highest exact-array accuracy (0.804627) and F1 (0.913386) among the tested ranks. This is the best tested configuration for that experiment, not a universal optimum for future datasets.
- Training/validation sequences are capped at **4,096 tokens**. Longer examples fail explicitly instead of being silently truncated.
- 4-bit NF4, double quantization, paged 8-bit AdamW, loss-based checkpoint selection, early stopping and the original notebook's evaluation routines are reused.
- CSV validation requires the exact eight-column backend schema, JSON arrays of equal lengths, nonempty criteria, positive finite maxima, bounded finite awarded scores and matching totals. BOM, quoted commas/newlines, repeated question ids and blank submitted answers are supported. Validation errors report row numbers without logging student responses.
- Dataset rows with rubric score **0** are mapped to binary label `0`; full-credit scores equal to their criterion maximum become `1`. **Partial-credit** rows and blank desired-answer rows are excluded and counted in the candidate manifest. If fewer than ten compatible unique questions remain, training fails with an explicit error. No arbitrary partial-credit threshold is used.
- Every training run starts from the original base checkpoint. Resulting adapters are stored only as **candidates** in private S3; **production vLLM and Hugging Face repos are not automatically replaced**.

## Prerequisites

1. Deploy the current `main` stack first and ensure it is healthy, SSM-managed, and has the backend dataset worker / SeaweedFS running.
2. Publish at least one **ready** training CSV containing enough compatible unique rubric questions. This is an instructor-created or automatically exported backend dataset; pending versions are not eligible.
3. Ensure the AWS account/region offers the specified GPU type and has sufficient vCPU quota. Change `autotraining_instance_type` if necessary. The initial GPU instance is provisioned and stopped by Terraform.
4. Allow AWS IAM, EC2, S3, DynamoDB, Lambda, Systems Manager, CloudWatch, Step Functions and EventBridge Scheduler resource creation, plus the archive Terraform provider.
5. Confirm enough EBS capacity, CUDA-compatible GPU drivers, package download access, Hugging Face model download access, and backend host Docker Compose / AWS CLI. The SSM scripts install basic missing system tools when needed.
6. Maintain backups of the **serving EC2's** PostgreSQL and SeaweedFS volumes: the original deployment's user-data has `user_data_replace_on_change = true`. **Never apply a plan that replaces `aws_instance.llm` without a backup.**

## Review and deploy (AWS credentials required)

From the deployment repository root:

```bash
terraform init
terraform fmt
terraform validate
terraform plan -out=autotraining.tfplan
# Inspect the plan. The existing aws_instance.llm must not be replaced.
terraform apply autotraining.tfplan
```

The schedule remains disabled. Perform a first manual run:

```bash
aws stepfunctions start-execution \
  --region "$(terraform output -raw region)" \
  --state-machine-arn "$(terraform output -raw autotraining_state_machine_arn)" \
  --input '{}'
```

Inspect the execution in Step Functions, then inspect:

```bash
aws logs tail "/fyp/fyp-qwen-vllm/training" --region us-east-1 --follow
aws logs tail "/aws/lambda/fyp-qwen-vllm-training-coordinator" --region us-east-1 --follow
aws s3 ls "s3://$(terraform output -raw autotraining_s3_bucket)/candidates/" --recursive
```

Substitute the actual Terraform name and region if overridden. A successful run stores an adapter, tokenizer, effective training configuration, package versions, resolved base-model revision, notebook hash, baseline metrics and candidate metrics under `candidates/<dataset-sha256>/<run-id>/`. Separate retry executions have separate artifact paths. A manifest is published last; only a completed manifest with matching run/version/checksum and adapter objects can advance the trained marker. The same ready dataset checksum on a subsequent run must skip starting the GPU, including when that checksum was trained before a different dataset. A missing ready dataset also skips without starting EC2. Failed attempts remain retryable on the next nightly/manual execution.

DynamoDB item keys are `run#<run-id>` for execution status/history, `success#<checksum>` for completed dataset training, `latest` for the latest completed candidate, and `lock` for the current owner. The run id is the first 32 hex characters of SHA-256 of the Step Functions execution ARN. Run statuses are `running`, `candidate_saved`, `completed`, `failed` and `skipped`. Failure details and the S3 manifest key are retained; metrics and configuration live in the manifest. No TTL is set on these records or the active lock. Failed CSV validation or model evaluation never advances a success marker.

Enable nightly automation only after the manual run is successful:

```bash
# Persist this setting so future plans do not revert to the default false.
printf 'autotraining_schedule_enabled = true\n' > autotraining.auto.tfvars
terraform plan -out=enable.tfplan
terraform apply enable.tfplan
```

To disable it again set `autotraining_schedule_enabled = false` in that file and apply the reviewed plan. Local `*.tfvars` files are ignored by Git. Disabling nightly starts leaves the cleanup watchdog enabled. Running `terraform apply` during active training may conflict with the Terraform-managed **stopped** EC2 state.

## Verification and operational expectations

The CI workflow `.github/workflows/autotraining.yml` runs Python syntax checks, shell syntax checks, offline unit tests, Terraform formatting and `terraform validate`. Locally, the lightweight tests can be run with:

```bash
python -m pip install boto3 ruff
python -m unittest discover -s autotraining/tests -v
bash -n autotraining/export_backend.sh
bash -n autotraining/run_training.sh
ruff check autotraining
```

Manual acceptance criteria: successful immutable dataset export; no EC2 start on unchanged checksum; one EC2 start with new checksum; SSM readiness then completion or failure; candidate artifact and nonempty metrics; existing vLLM unchanged; EC2 stopped after both successful and failing runs; CloudWatch logs available; no publicly exposed datasets.

### Important limits

- Actual AWS/Terraform/GPU execution **must be verified in your AWS account**; offline unit tests alone cannot prove this deployment works in a real region with the selected GPU AMI.
- There is no automatic promotion to the production grading model. The notebook outputs a binary-array grading adapter, while the current backend expects `score`, `feedback`, and `criteria_met` JSON objects. Promotion must address this mismatch and pass a separate fixed benchmark.
- Terminal events are best effort, so the independent watchdog also checks execution state every 15 minutes. Cleanup retries retain the GPU lock on stop failure. During an AWS control-plane/IAM outage, inspect the GPU state and stop it manually if needed; never delete its lock while a training process may still be running. The SSM command has a 12-hour execution timeout; the external watchdog enforces a 14-hour budget with up to 15 minutes polling delay.
- If upgrading an older prototype with a legacy checksum-only `lock` item, stop/confirm the GPU is stopped and remove that legacy lock before enabling runs. New locks identify the owning Step Functions execution.
- QLoRA is retained to match the project's tested configuration and this workflow. The current [Unsloth Qwen3.5 guidance](https://unsloth.ai/docs/models/qwen3.5/fine-tune) favors bf16 LoRA because of quantization differences. Evaluate new candidates against the saved baseline; candidate creation does not assert a quality improvement.
- A stopped GPU EC2 instance still incurs EBS charges; S3, DynamoDB, CloudWatch, networking, and Step Functions can also incur charges.
- The candidate S3 bucket is versioned. Back up valuable artifacts before cleanup; `terraform destroy` cannot delete a nonempty versioned bucket until its object versions are removed.

## AWS orchestration references

- [EventBridge Scheduler cron expressions and time zones](https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html): `cron(10 0 * * ? *)`, `Asia/Singapore`, flexible window off (minute-level precision).
- [Step Functions execution status events](https://docs.aws.amazon.com/step-functions/latest/dg/eventbridge-integration.html): terminal event matching and best-effort delivery semantics.
- [PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization): NF4, double quantization and bf16 computation.
