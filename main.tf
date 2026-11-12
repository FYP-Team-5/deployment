data "aws_ssm_parameter" "gpu_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.6-ubuntu-22.04/latest/ami-id"
}

resource "aws_vpc" "llm" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = var.name }
}

resource "aws_subnet" "llm" {
  vpc_id                  = aws_vpc.llm.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = local.selected_availability_zone
  map_public_ip_on_launch = false

  tags = { Name = "${var.name}-public" }

  lifecycle {
    precondition {
      condition     = length(local.supported_availability_zones) > 0
      error_message = "No available Availability Zone in ${var.aws_region} supports both ${local.serving_instance_type} and ${local.training_instance_type}. Choose another region or training instance type."
    }
    precondition {
      condition     = var.availability_zone == null ? true : contains(local.supported_availability_zones, var.availability_zone)
      error_message = "The requested Availability Zone must be available and support both serving and training instance types. Supported zones: ${join(", ", local.supported_availability_zones)}."
    }
  }
}

resource "aws_internet_gateway" "llm" {
  vpc_id = aws_vpc.llm.id
  tags   = { Name = var.name }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.llm.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.llm.id
  }

  tags = { Name = "${var.name}-public" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.llm.id
  route_table_id = aws_route_table.public.id
}

# AWS-maintained source addresses for browser-based EC2 Instance Connect.
# Avoid exposing SSH to all internet hosts.
data "aws_ec2_managed_prefix_list" "instance_connect" {
  name = "com.amazonaws.${var.aws_region}.ec2-instance-connect"
}

resource "aws_security_group" "llm" {
  name_prefix = "${var.name}-"
  description = "Qwen vLLM with public frontend and backend access"
  vpc_id      = aws_vpc.llm.id

  ingress {
    description     = "SSH via EC2 Instance Connect browser"
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.instance_connect.id]
  }

  ingress {
    description = "Public FastAPI backend"
    from_port   = 8002
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Public Next.js frontend"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Outbound HTTPS and package downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = var.name }
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "llm" {
  name_prefix        = "${var.name}-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.llm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "llm" {
  name_prefix = "${var.name}-"
  role        = aws_iam_role.llm.name
}

# Explicitly managed log groups: seven-day retention and deletion on terraform destroy.
resource "aws_cloudwatch_log_group" "cloud_init" {
  name              = "/fyp/${var.name}/cloud-init"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "vllm" {
  name              = "/fyp/${var.name}/vllm"
  retention_in_days = 7
}

# The agent can publish only to these Terraform-managed log groups.
data "aws_iam_policy_document" "cloudwatch_logs" {
  statement {
    sid       = "DescribeLogGroups"
    actions   = ["logs:DescribeLogGroups"]
    resources = ["*"]
  }

  statement {
    sid = "PublishFypLogs"
    actions = [
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents"
    ]
    resources = [
      trimsuffix(aws_cloudwatch_log_group.cloud_init.arn, ":*"),
      "${trimsuffix(aws_cloudwatch_log_group.cloud_init.arn, ":*")}:*",
      trimsuffix(aws_cloudwatch_log_group.vllm.arn, ":*"),
      "${trimsuffix(aws_cloudwatch_log_group.vllm.arn, ":*")}:*"
    ]
  }
}

resource "aws_iam_role_policy" "cloudwatch_logs" {
  name   = "${var.name}-cloudwatch-logs"
  role   = aws_iam_role.llm.id
  policy = data.aws_iam_policy_document.cloudwatch_logs.json
}

resource "aws_eip" "llm" {
  domain = "vpc"

  tags = { Name = "${var.name}-static-ip" }
}

resource "aws_instance" "llm" {
  ami                         = data.aws_ssm_parameter.gpu_ami.value
  instance_type               = local.serving_instance_type
  subnet_id                   = aws_subnet.llm.id
  vpc_security_group_ids      = [aws_security_group.llm.id]
  iam_instance_profile        = aws_iam_instance_profile.llm.name
  associate_public_ip_address = false

  # Remove the root EBS volume when EC2 is terminated by terraform destroy.
  root_block_device {
    delete_on_termination = true
  }

  # cloud-init decompresses the script; keep EC2 user data below its 16 KiB limit.
  user_data_base64 = base64gzip(templatefile("${path.module}/user-data.sh.tftpl", {
    serve_script                      = base64gzip(file("${path.module}/Script/serve_qwen35_vllm.sh"))
    requirements                      = base64gzip(file("${path.module}/Script/requirements-vllm.txt"))
    readiness_script                  = base64gzip(file("${path.module}/Script/check_vllm_ready.py"))
    callback_script                   = base64gzip(file("${path.module}/Script/configure_training_callback.py"))
    bootstrap_script                  = base64gzip(file("${path.module}/Script/bootstrap_apps.sh"))
    systemctl_retry_script            = file("${path.module}/Script/systemctl_retry.sh")
    aws_region                        = var.aws_region
    cloud_init_log_group              = aws_cloudwatch_log_group.cloud_init.name
    vllm_log_group                    = aws_cloudwatch_log_group.vllm.name
    hf_token_parameter                = var.vllm_hf_token_parameter
    training_callback_token_parameter = var.autotraining_callback_token_parameter
    training_state_machine_arn        = "arn:${data.aws_partition.vllm.partition}:states:${var.aws_region}:${data.aws_caller_identity.vllm.account_id}:stateMachine:${var.name}-nightly-training"
  }))
  user_data_replace_on_change = true

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2 # Allow the backend Docker container to use the instance role.
  }

  tags = { Name = var.name }

  depends_on = [aws_route_table_association.public, aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.cloudwatch_logs, aws_iam_role_policy.vllm_hf_token, aws_iam_role_policy.autotraining_backend_export]
}

resource "aws_eip_association" "llm" {
  instance_id   = aws_instance.llm.id
  allocation_id = aws_eip.llm.id
}
