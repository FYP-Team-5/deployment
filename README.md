# FYP EC2 deployment

Terraform allocates a persistent Elastic IP and launches a GPU-backed Ubuntu 22.04 EC2 instance running Qwen3.5-9B via vLLM. The first-boot user-data script installs Docker Engine and Compose, clones the public [frontend](https://github.com/FYP-Team-5/frontend) and [backend](https://github.com/FYP-Team-5/backend) repositories, copies their .env.example files to .env, configures host addresses and generated secrets, then launches both Compose stacks. No Portainer is installed.

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
- Portainer is not installed. The legacy Terraform security-group configuration still contains an unused restricted TCP 9443 rule; remove it when updating firewall rules.

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

## Elastic IP lifecycle

Terraform allocates `aws_eip.llm` and associates it separately with EC2 using `aws_eip_association.llm`. The EIP survives **EC2 replacement** within the Terraform configuration and does not incur a second automatically assigned IPv4 charge. `terraform output -raw public_ip` shows the fixed IP.

The EIP has `prevent_destroy = true` to prevent accidental release. Consequently, a normal `terraform destroy` **will fail until you deliberately remove that lifecycle guard** and re-run your plan. Release the EIP when you no longer need it to avoid ongoing IPv4 charges. Replacing the EC2 instance still destroys its local EBS/NVMe storage.

To remove the infrastructure when finished, remove the EIP's lifecycle protection from `main.tf`, inspect the plan, and then run:

```bash
terraform destroy
```


