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
  map_public_ip_on_launch = true

  tags = { Name = "${var.name}-public" }
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

resource "aws_security_group" "llm" {
  name_prefix = "${var.name}-"
  description = "Qwen vLLM: outbound internet for setup, SSM, and model downloads"
  vpc_id      = aws_vpc.llm.id

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

resource "aws_instance" "llm" {
  ami                         = data.aws_ssm_parameter.gpu_ami.value
  instance_type               = "g4dn.xlarge"
  subnet_id                   = aws_subnet.llm.id
  vpc_security_group_ids      = [aws_security_group.llm.id]
  iam_instance_profile        = aws_iam_instance_profile.llm.name
  associate_public_ip_address = true

  user_data_base64 = base64encode(templatefile("${path.module}/user-data.sh.tftpl", {
    serve_script = base64gzip(file("${path.module}/Script/serve_qwen35_vllm.sh"))
    requirements = base64gzip(file("${path.module}/Script/requirements-vllm.txt"))
  }))
  user_data_replace_on_change = true

  metadata_options {
    http_tokens = "required"
  }

  tags = { Name = var.name }

  depends_on = [aws_route_table_association.public, aws_iam_role_policy_attachment.ssm]
}
