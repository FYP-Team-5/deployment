# FYP EC2 deployment

Terraform allocates an Elastic IP for the lifetime of the Terraform stack and launches a GPU-backed Ubuntu 22.04 EC2 instance running Qwen3.5-9B via vLLM. The first-boot user-data script installs Docker Engine and Compose, clones the public [frontend](https://github.com/FYP-Team-5/frontend) and [backend](https://github.com/FYP-Team-5/backend) repositories, copies their .env.example files to .env, configures host addresses and generated secrets, then launches both Compose stacks. No Portainer is installed.

## EC2 Instance Connect (browser SSH)

The EC2 security group permits inbound TCP 22 only from the AWS-maintained regional EC2 Instance Connect IPv4 prefix list (`com.amazonaws.<region>.ec2-instance-connect`), **not** from the entire internet. Terraform looks up this AWS-managed prefix list rather than creating it.

In the AWS Console, navigate to **EC2 → Instances → select the instance → Connect → EC2 Instance Connect**. Use the **Public IP** connection type, the instance Elastic IP and the username `ubuntu`, then select **Connect**.

Ubuntu 22.04 AWS AMIs normally include the EC2 Instance Connect integration. Browser login also requires the connecting IAM identity to have `ec2-instance-connect:SendSSHPublicKey` permission; this permission belongs to your console identity, not the EC2 instance role. To troubleshoot SSH, use Session Manager and run `sudo systemctl status ssh` and `dpkg -s ec2-instance-connect`.

This ingress rule enables browser-based Instance Connect. Direct SSH from your own terminal requires a separate security-group ingress rule for your public IP (/32), plus SSH key authentication.

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

## CloudWatch setup and vLLM progress

Terraform creates two CloudWatch Logs log groups, both using seven-day retention:

- `/fyp/fyp-qwen-vllm/cloud-init` — the EC2 `/var/log/cloud-init-output.log` bootstrap log.
- `/fyp/fyp-qwen-vllm/vllm` — `qwen-vllm.service` systemd journal entries, including dependency installation and model loading output.

The AWS CloudWatch Agent is installed and started early in EC2 user-data, after the Elastic IP is associated and base packages are installed. Initial work before the agent starts is not guaranteed to be shipped; use SSM and the local cloud-init log for earlier troubleshooting. Native journald collection requires a current CloudWatch Agent release. The EC2 IAM role grants scoped log-stream writing to these two log groups only. No CloudWatch custom metrics are enabled.

In AWS Console: **CloudWatch → Logs → Log groups**, open either group and select the stream named with the EC2 instance ID. Or use AWS CLI:

```bash
aws logs tail /fyp/fyp-qwen-vllm/cloud-init --region us-east-1 --follow
aws logs tail /fyp/fyp-qwen-vllm/vllm --region us-east-1 --follow
```

If you override Terraform `name` or `aws_region`, use the resulting log group names and region instead.

Log ingestion and storage can incur AWS charges when usage exceeds applicable free allowances. Both log groups are **Terraform-managed**: a successful `terraform destroy` deletes the groups and their events as well as EC2 and the Elastic IP. Instance replacement alone preserves the log groups; log streams are separated by instance ID. Changes to this user-data file replace the EC2 instance because `user_data_replace_on_change = true`.

### Why is the deployment called public?

The security-group description refers to **public frontend (TCP 3000) and backend (TCP 8002)**. The EC2 instance has an Elastic IP and lives in a subnet with a route to an Internet Gateway. The vLLM process binds to `0.0.0.0:8000` on the EC2 host so that the Docker backend can connect, but **the security group does not allow inbound TCP 8000**. Thus, vLLM is not directly reachable from the public internet through this configuration. Use SSM port forwarding (`./connect-vllm.sh`) for direct administration. Do not add an inbound port 8000 rule for public access.

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

After a successful destroy, Terraform removes its two CloudWatch log groups (including stored logs), managed EC2 instance, associated Elastic IP and allocation, security group, IAM role/policy attachment/instance profile, route table/association, Internet Gateway, subnet and VPC. The EC2 root EBS volume is configured with `delete_on_termination = true`, and the 125 GB NVMe instance store is ephemeral. The PostgreSQL Docker volume, Docker images, application `.env` files, and downloaded model caches are **not preserved** when the instance is terminated.

**Caution:** This permanently deletes application/database data stored on EC2. Back up any required data before destroying the stack. The command only destroys resources tracked by this Terraform state; manually created AWS resources, external snapshots/backups, and resources outside this stack are unaffected. Check for any orphaned resources if a destroy fails partway through. The local Terraform state and repository files are not deleted by Terraform.

`user_data_replace_on_change = true` remains enabled: edits to the rendered startup script can trigger EC2 replacement on `terraform apply`, deleting the EC2-local data, although the EIP remains allocated until the entire stack is destroyed.
