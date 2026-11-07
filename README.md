# FYP deployment

Terraform deploys the quiz application and its nightly Qwen3.5-9B retraining workflow.

| Component | Role |
| --- | --- |
| Serving EC2 | Frontend, backend, PostgreSQL, SeaweedFS and vLLM |
| Training EC2 | Dedicated `g6e.xlarge`, stopped while idle |
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
