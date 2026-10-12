# FYP EC2 deployment

Terraform allocates an Elastic IP for the lifetime of the Terraform stack and launches a GPU-backed Ubuntu 22.04 EC2 instance running Qwen3.5-9B via vLLM. The first-boot user-data script installs Docker Engine and Compose, clones the public [frontend](https://github.com/FYP-Team-5/frontend) and [backend](https://github.com/FYP-Team-5/backend) repositories, copies their .env.example files to .env, configures host addresses and generated secrets, then launches both Compose stacks. No Portainer is installed.

## Quick start

Requirements: Terraform 1.5+, AWS CLI with EC2/VPC/IAM/SSM permissions, Session Manager plugin, and a g4dn.xlarge quota in us-east-1.

```bash
cd deployment
terraform init
terraform plan -out=llm.tfplan
terraform apply llm.tfplan
terraform output -raw public_ip
```

- Frontend: http://<ELASTIC_IP>:3000
- Backend: http://<ELASTIC_IP>:8002/health (TCP 8002 is allowed in the security group)
- vLLM stays inaccessible from the internet (TCP 8000 has no ingress rule); access via `./connect-vllm.sh`.
- PostgreSQL is bound to 127.0.0.1:5433 on the EC2 host.
- Portainer is not installed; TCP 9443 is not open.

Initial deployment creates /opt/fyp/frontend/.env and /opt/fyp/backend/.env from each repo's .env.example, only if they do not already exist. The bootstrap script generates random PostgreSQL, JWT, staff-registration and API secrets. These are stored on EC2, not committed to Git. Both repository clones are public, so no GitHub authentication is needed.

The frontend receives NEXT_PUBLIC_API_URL pointing to the instance's Elastic IP and port 8002. EC2 does not request an automatic public IPv4 address (both instance and subnet auto-assign are disabled). The initial bootstrap waits for Elastic IP association before downloading packages. The backend uses the host's vLLM endpoint at host.docker.internal:8000 with the qwen35-9b-grading-qlora model. vLLM binds to the EC2 host interfaces but is blocked externally by the EC2 security group. Both Docker Compose applications have restart policies.

## Troubleshooting

From your workstation, open an SSM session:

```bash
aws ssm start-session --region us-east-1 --target "$(terraform output -raw instance_id)"
```

Then on EC2:

```bash
sudo docker ps
sudo docker compose --project-directory /opt/fyp/backend logs -f
sudo docker compose --project-directory /opt/fyp/frontend logs -f
sudo journalctl -u qwen-vllm -f
```

For manual redeployment after edits to .env:

```bash
sudo docker compose --project-directory /opt/fyp/backend up --build -d
sudo docker compose --project-directory /opt/fyp/frontend up --build -d
```

The frontend Next.js build embeds NEXT_PUBLIC_* variables, so it must be rebuilt after changes. A public frontend also needs a public backend endpoint; localhost in a browser points to the visitor's machine, not EC2.

## Important security and lifecycle notes

**This is a development/demo HTTP deployment, not production-ready.** Both port 3000 and port 8002 are publicly accessible without TLS. The frontend's NEXT_PUBLIC_GRADING_API_KEY is embedded in browser JavaScript and cannot be treated as a secret. Add HTTPS and strong server-side authorization before production use. The startup process does not deploy any separate RAG service, so frontend RAG features still require configuration.

The EC2 instance uses 125 GB of ephemeral NVMe instance store for its LLM model caches, plus the AMI's EBS root volume for Docker and PostgreSQL. Replacing the EC2 instance destroys both, including database data and .env files. `main.tf` sets `user_data_replace_on_change = true`, so **Terraform changes to startup scripts cause instance replacement**. Check terraform plan carefully before apply.

## Elastic IP and teardown lifecycle

Terraform allocates `aws_eip.llm` and associates it separately with EC2 using `aws_eip_association.llm`. The EIP survives EC2 replacement **within the same Terraform stack**, so the application's public address remains stable during an instance replacement. It is **released on `terraform destroy`**; a future `terraform apply` allocates a new EIP.

### Destroy everything provisioned by Terraform

Run from the deployment repository with the same Terraform state and AWS region used during creation:

```bash
terraform plan -destroy
terraform destroy
```

After a successful destroy, Terraform removes its managed EC2 instance, associated Elastic IP and allocation, security group, IAM role/policy attachment/instance profile, route table/association, Internet Gateway, subnet and VPC. The EC2 root EBS volume is configured with `delete_on_termination = true`, and the 125 GB NVMe instance store is ephemeral. The PostgreSQL Docker volume, Docker images, application `.env` files, and downloaded model caches are **not preserved** when the instance is terminated.

**Caution:** This permanently deletes application/database data stored on EC2. Back up any required data before destroying the stack. The command only destroys resources tracked by this Terraform state; manually created AWS resources, external snapshots/backups, and resources outside this stack are unaffected. Check for any orphaned resources if a destroy fails partway through. The local Terraform state and repository files are not deleted by Terraform.

`user_data_replace_on_change = true` remains enabled: edits to the rendered startup script can trigger EC2 replacement on `terraform apply`, deleting the EC2-local data, although the EIP remains allocated until the entire stack is destroyed.
