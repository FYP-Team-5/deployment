# Qwen vLLM on EC2

This repository provisions a `g4dn.xlarge` EC2 instance with Terraform and
starts the Qwen3.5-9B base model plus its grading LoRA adapter with vLLM. The
default AWS Region is `us-east-1`. The API listens on the instance's loopback
interface; connect through AWS Systems Manager Session Manager.

## Project structure

```text
deployment/                         This Git repository
├── .gitignore                      Ignored local state, plans, and secrets
├── .terraform.lock.hcl             Locked AWS provider version; commit this
├── README.md                       Setup, access, and teardown instructions
├── versions.tf                     Terraform and AWS provider requirements
├── variables.tf                    Region and resource-name defaults
├── main.tf                         AWS networking, IAM, and EC2 resources
├── outputs.tf                      Instance ID, Region, and public IP
├── user-data.sh.tftpl              First-boot setup and systemd services
└── Script/
    ├── serve_qwen35_vllm.sh        Dependency setup and vLLM launch
    ├── requirements-vllm.txt       Pinned Python serving packages
    └── .env                        Optional local secret file; ignored by Git
```

Terraform creates `.terraform/` and `terraform.tfstate*` locally; both are
ignored by Git. The `qlora/`, `grading/`, and `frontend/` folders beside
`deployment/` in the FYP workspace are separate projects. Terraform packages
the two non-secret files under `Script/` into EC2 user data. It does not upload
`Script/.env` or any files from those sibling folders.

## Infrastructure

| Component | Configuration |
| --- | --- |
| Region | `us-east-1` by default; override with `-var='aws_region=...'`. |
| Image | Latest [AWS Deep Learning OSS NVIDIA Driver AMI GPU PyTorch 2.6 (Ubuntu 22.04)](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-gpu-pytorch-2.6-ubuntu-22-04.html), resolved from an AWS public SSM parameter. |
| Compute | One On-Demand `g4dn.xlarge`, with an NVIDIA T4 GPU and 16 GiB GPU memory; IMDSv2 required. |
| Root storage | The AMI's default EBS root volume. The current `us-east-1` image specifies 40 GiB gp3; Terraform does not resize it. |
| Working storage | The instance's 125 GB NVMe instance store, formatted as ext4 and mounted at `/mnt/llm-data`. Python environments, package caches, and model weights use it. |
| Network | New VPC `10.42.0.0/16`, public subnet `10.42.1.0/24`, internet gateway, route table with a `0.0.0.0/0` route, and route table association. The instance gets a public IP for outbound downloads. |
| Firewall | Security group with **no inbound rules** and outbound IPv4 access for downloads and SSM. vLLM binds to `127.0.0.1:8000`. |
| IAM | EC2 role and instance profile with AWS managed `AmazonSSMManagedInstanceCore` for Session Manager. |
| Boot services | `llm-storage.service` mounts NVMe before `qwen-vllm.service` installs dependencies and starts vLLM. The serving service restarts on failure. |
| Tags | `Project=fyp-qwen-vllm`, `ManagedBy=terraform`, plus resource names. |

The serving script installs `uv`, creates a managed Python 3.12 environment,
and installs `Script/requirements-vllm.txt` on first start. It serves
`Qwen/Qwen3.5-9B` as `qwen35-9b-base` and the public
`SmuFypTeam5/GradingQlora` adapter as `qwen35-9b-grading-qlora`. The EC2
service uses an 8192-token context, one concurrent sequence, and 90% GPU
memory utilization. Initial setup and model downloads can take several
minutes. Adjust the systemd environment settings if vLLM reports insufficient
GPU memory.

[AWS instance-store data is temporary](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ssd-instance-store.html).
A stop/start or host failure can erase the NVMe environment and caches. The
storage service recreates them, and the serving script reinstalls and
downloads what it needs. The EBS root volume is billed separately from the
instance.

## Secrets and where to store them

| Secret or credential | Needed? | Storage and use |
| --- | --- | --- |
| AWS credentials | Required on the machine running Terraform and the AWS CLI. | Use an AWS CLI profile or SSO configuration outside this repository. Terraform uses that identity; it does not copy credentials to EC2. |
| `HF_TOKEN` | Optional for the default public models; required for a private or gated Hugging Face model or adapter. | Store in `/opt/qwen-vllm/Script/.env` **on EC2**, owned by `ubuntu` with mode `0600`. The serving script reads and exports this key. |
| `VLLM_API_KEY` | Not used by this deployment. | The local `Script/.env` may contain this key, but the serving script does not read it or enable API-key authentication. Access is limited to an SSM tunnel and the instance's loopback interface. |

Do not put tokens in `.tfvars`, Terraform variables, `user-data.sh.tftpl`, or
committed files: Terraform stores user data and variable values in state.
`Script/.env` is ignored by Git, but Terraform never sends that local file to
EC2. The commands below start from the FYP workspace root. For a private or
gated model, after the instance is running:

```bash
cd deployment
aws ssm start-session --region us-east-1 --target "$(terraform output -raw instance_id)"
```

Inside the Session Manager shell, create the EC2 secret file the first time,
then edit it:

```bash
sudo install -o ubuntu -g ubuntu -m 0600 /dev/null /opt/qwen-vllm/Script/.env
sudo -u ubuntu vi /opt/qwen-vllm/Script/.env
```

Enter `HF_TOKEN=your_token` in the editor, save it, then restart the service:

```bash
sudo systemctl restart qwen-vllm
```

If you use a different adapter, set `LORA_PATH` in the service environment and
restart it. The script reads `.env` without executing it as shell code.

## Build

Prerequisites: Terraform 1.5+, configured AWS CLI credentials with EC2, VPC,
IAM, and SSM permissions, and available `g4dn.xlarge` On-Demand quota in the
chosen Region. Confirm that the instance type and AMI are available if you
change Regions.

```bash
cd deployment
terraform init
terraform plan -out=llm.tfplan
terraform apply llm.tfplan
terraform output -raw instance_id
```

Terraform state is local by default and ignored by Git. Keep it until teardown;
losing it makes destruction harder. Applying the plan creates billable AWS
resources.

## Check startup and connect

Install the AWS CLI and its Session Manager plugin on your local machine. Use
the same Region and profile as Terraform. In one terminal, start a tunnel:

```bash
aws ssm start-session \
  --region us-east-1 \
  --target "$(terraform output -raw instance_id)" \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8000"],"localPortNumber":["8000"]}'
```

Then in another terminal:

```bash
curl http://127.0.0.1:8000/v1/models
```

To inspect boot and service logs, open a Session Manager shell:

```bash
aws ssm start-session --region us-east-1 --target "$(terraform output -raw instance_id)"
sudo journalctl -u llm-storage -u qwen-vllm -f
```

## Tear down

From the same `deployment` directory and with the same Terraform state:

```bash
terraform destroy
```

This removes the instance and its EBS root volume, VPC, route, security group,
and instance role. The NVMe cache is lost with the instance. AWS charges for
the running instance and EBS root volume until destruction.
