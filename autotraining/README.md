# Automated nightly QLoRA retraining

This feature is implemented in the `feat/autotraining` branch of `FYP-Team-5/deployment`.
**Applying Terraform creates paid AWS resources.** The daily schedule is **disabled by default** for a safe first deployment. No existing backend/user-data or vLLM configuration is modified.

## Architecture

- EventBridge Scheduler: **00:10 Asia/Singapore** nightly, once enabled.
- Standard Step Functions: prepare, deduplicate, start GPU EC2, wait for SSM readiness, run training, poll, verify candidate, stop GPU. Failure branches also request EC2 stop.
- Lambda coordinator: three modules (`coordinator.py`, `transfer.py`, `execution.py`), packaged with Terraform's archive provider.
- Existing backend EC2: SSM executes `export_backend.sh` inside its host. The script uses the backend's **version ledger** and checksum-verifying SeaweedFS reader to retrieve a specific immutable ready CSV version. No public instructor authentication or SeaweedFS ingress is required.
- Private AWS S3 bucket: checksum-addressed dataset snapshots, training runner scripts, evaluated candidate adapter, and `result.json`.
- DynamoDB: last successfully trained dataset checksum, plus a single conditional 36-hour lease to prevent concurrent training.
- Separate EC2 GPU instance: default `g6e.xlarge`, encrypted 200 GB root EBS, SSM-only management, no inbound security group rules. Terraform manages its initial state as stopped.
- CloudWatch: Lambda execution logs and streaming training SSM output.

The PyTorch, Transformers and TRL package versions are pinned in `requirements.txt`; compatibility still requires a live GPU test. The default training reference is the committed notebook `Rubrics/Qlora Comparison/qlora (48r,96a).ipynb` from `FYP-Team-5/model`, pinned to commit `8a2b94ec95754d875dc546461ef15dc2a279ce5a`.

## Training contract

- Base model: `Qwen/Qwen3.5-9B`.
- QLoRA: rank **48**, alpha **96**, dropout **0.05**, learning rate **5e-5**, 3 max epochs, microbatch **1** × gradient accumulation **8**, seed **42**, question-group split seed **42**.
- 4-bit NF4, double quantization, paged 8-bit AdamW, loss-based checkpoint selection, early stopping and the original notebook's evaluation routines are reused.
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

Substitute the actual Terraform name and region if overridden. A successful run stores an adapter, tokenizer and evaluation manifest under `candidates/<dataset-sha256>/`. The same ready dataset checksum on a subsequent run must skip starting the GPU.

Enable nightly automation only after the manual run is successful:

```bash
terraform plan -var='autotraining_schedule_enabled=true' -out=enable.tfplan
terraform apply enable.tfplan
```

To disable it again use the same variable with `false`. Running `terraform apply` during active training may conflict with the Terraform-managed **stopped** EC2 state.

## Verification and operational expectations

The CI workflow `.github/workflows/autotraining.yml` runs Python syntax checks, shell syntax checks, offline unit tests, Terraform formatting and `terraform validate`. Locally, the lightweight tests can be run with:

```bash
python -m pip install boto3 nbformat
python -m unittest discover -s autotraining/tests -v
bash -n autotraining/export_backend.sh autotraining/run_training.sh
```

Manual acceptance criteria: successful immutable dataset export; no EC2 start on unchanged checksum; one EC2 start with new checksum; SSM readiness then completion or failure; candidate artifact and nonempty metrics; existing vLLM unchanged; EC2 stopped after both successful and failing runs; CloudWatch logs available; no publicly exposed datasets.

### Important limits

- Actual AWS/Terraform/GPU execution **must be verified in your AWS account**; offline unit tests alone cannot prove this deployment works in a real region with the selected GPU AMI.
- There is no automatic promotion to the production grading model. The notebook outputs a binary-array grading adapter, while the current backend expects `score`, `feedback`, and `criteria_met` JSON objects. Promotion must address this mismatch and pass a separate fixed benchmark.
- A global Step Functions timeout or a manually terminated execution can bypass normal cleanup. The training shell schedules a best-effort 15-hour systemd shutdown watchdog; still inspect the GPU state and stop it manually if needed. DynamoDB's 36-hour lock expiry allows recovery, but do not rerun while an orphaned training process is active.
- A stopped GPU EC2 instance still incurs EBS charges; S3, DynamoDB, CloudWatch, networking, and Step Functions can also incur charges.
- The candidate S3 bucket is versioned. Back up valuable artifacts before cleanup; `terraform destroy` cannot delete a nonempty versioned bucket until its object versions are removed.
