# Separate GPU instance: no inbound ports; SSM manages all commands.
resource "aws_security_group" "autotraining" {
  name_prefix = "${var.name}-trainer-"
  description = "Private inbound; outbound for models, packages and SSM"
  vpc_id      = aws_vpc.llm.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_cloudwatch_log_group" "autotraining" {
  name              = "/fyp/${var.name}/training"
  retention_in_days = 14
}

resource "aws_iam_role" "autotraining" {
  name_prefix = "${var.name}-trainer-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}
resource "aws_iam_role_policy_attachment" "autotraining_ssm" {
  role       = aws_iam_role.autotraining.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_role_policy" "autotraining" {
  name = "${var.name}-trainer-access"
  role = aws_iam_role.autotraining.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.autotraining.arn}/datasets/*",
          "${aws_s3_bucket.autotraining.arn}/scripts/run_training.sh",
          "${aws_s3_bucket.autotraining.arn}/scripts/train.py",
          "${aws_s3_bucket.autotraining.arn}/scripts/requirements.txt"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.autotraining.arn}/candidates/*"]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream", "logs:DescribeLogStreams",
          "logs:PutLogEvents"
        ]
        Resource = [
          trimsuffix(aws_cloudwatch_log_group.autotraining.arn, ":*"),
          "${trimsuffix(aws_cloudwatch_log_group.autotraining.arn, ":*")}:*"
        ]
      }
    ]
  })
}
resource "aws_iam_instance_profile" "autotraining" {
  name_prefix = "${var.name}-trainer-"
  role        = aws_iam_role.autotraining.name
}

resource "aws_instance" "autotraining" {
  ami                                  = data.aws_ssm_parameter.gpu_ami.value
  instance_type                        = coalesce(var.autotraining_instance_type, aws_instance.llm.instance_type)
  subnet_id                            = aws_subnet.llm.id
  vpc_security_group_ids               = [aws_security_group.autotraining.id]
  iam_instance_profile                 = aws_iam_instance_profile.autotraining.name
  associate_public_ip_address          = true
  instance_initiated_shutdown_behavior = "stop"
  metadata_options {
    http_tokens = "required"
  }
  root_block_device {
    # Persistent training workspace, model/dependency caches and checkpoints.
    # Keep this independent of the selected instance type's ephemeral NVMe disk.
    volume_size           = 200
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }
  tags = {
    Name    = "${var.name}-trainer"
    Purpose = "nightly-qlora"
  }
  depends_on = [
    aws_iam_role_policy_attachment.autotraining_ssm,
    aws_iam_role_policy.autotraining,
    aws_route_table_association.public
  ]
}
# Terraform stops the new instance immediately after provisioning it.
# Do not run Terraform apply during a training job.
resource "aws_ec2_instance_state" "autotraining_idle" {
  instance_id = aws_instance.autotraining.id
  state       = "stopped"
}
output "autotraining_instance_id" {
  value = aws_instance.autotraining.id
}
