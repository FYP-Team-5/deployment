# Qwen vLLM on EC2

This repository provisions a `g4dn.xlarge` EC2 instance with Terraform and
starts the Qwen3.5-9B base model plus its grading LoRA adapter with vLLM. The
default AWS Region is `us-east-1`. The API listens on the instance's loopback
interface; connect through AWS Systems Manager Session Manager.

## Project structure

```text
deployment/                         This Git repository
├── .gitignore                      Ignored local state, plans, and .env files
├── .terraform.lock.hcl             Locked AWS provider version; commit this
├── README.md                       Setup, access, and teardown instructions
├── versions.tf                     Terraform and AWS provider requirements
├── variables.tf                    Region and resource-name defaults
├── main.tf                         AWS networking, IAM, and EC2 resources
├── outputs.tf                      Instance ID, Region, and public IP
├── user-data.sh.tftpl              First-boot setup and systemd services
├── connect-vllm.sh                 Local SSM port-forward helper
└── Script/
    ├── serve_qwen35_vllm.sh        Dependency setup and vLLM launch
    └── requirements-vllm.txt       Pinned Python serving packages
```

Terraform creates `.terraform/` and `terraform.tfstate*` locally; both are
ignored by Git. The `qlora/`, `grading/`, and `frontend/` folders beside
`deployment/` in the FYP workspace are separate projects. Terraform packages
the two files under `Script/` into EC2 user data. It does not upload a local
`Script/.env` file or any files from those sibling folders.

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

## Credentials and model access

Both configured model repositories are public, so the deployment requires no
model secret. The deployment script does not read `.env`. A local `Script/.env`
file, if present, is ignored by Git and is not copied to EC2. `VLLM_API_KEY`
in that file is also unused; API access is limited to an SSM tunnel and the
instance's loopback interface.

AWS credentials are needed only on the machine running Terraform and the AWS
CLI. Store them in your AWS CLI profile or SSO configuration outside this
repository. Terraform uses that identity and grants the EC2 instance its own
SSM role; it does not copy your local credentials to EC2. Do not put secrets in
Terraform variables or user data because Terraform stores those values in
state.

If you change `BASE_MODEL` or `LORA_PATH` to a private or gated repository,
you will need to add authenticated model access separately before starting
the service.

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

## Connect from the local frontend and backend

Install the AWS CLI and its [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
on your computer. After `terraform apply`, start the tunnel in a dedicated
terminal and leave it running:

```bash
cd deployment
./connect-vllm.sh
```

This forwards EC2 port `8000` to **port `8000` on your computer**. In another
terminal, check that both model names are available:

```bash
curl http://127.0.0.1:8000/v1/models
```

The browser frontend should keep calling the local backend; it does not need
the EC2 address. Set the local grading backend's `.env` to call the forwarded
vLLM endpoint:

```dotenv
# If grading runs directly on your computer:
LLM_URL=http://127.0.0.1:8000/v1/chat/completions
LLM_MODEL=qwen35-9b-grading-qlora
LLM_API_KEY=
```

If grading runs through `grading/compose.yaml`, use
`LLM_URL=http://host.docker.internal:8000/v1/chat/completions` instead;
`localhost` inside the container refers to the container itself. If its API
container is already running, recreate it after changing `.env`, then check
that it can reach the tunnel:

```bash
cd grading
docker compose up -d --force-recreate api
docker compose exec api curl -fsS http://host.docker.internal:8000/v1/models
```

The last command checks whether your Docker runtime can reach the host-side
tunnel. If it cannot reach a loopback-only tunnel, run the grading backend
directly on your computer using its README's non-Docker instructions. Keep the
tunnel terminal open while grading; closing it disconnects the backend from
vLLM. The EC2 security group does not expose port `8000` to the internet.
The current grading Compose file also expects external database and vector
services; the tunnel supplies only the LLM connection.

To inspect boot and service logs, open a separate Session Manager shell:

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
