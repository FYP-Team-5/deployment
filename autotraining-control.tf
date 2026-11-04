# Package the Lambda coordinator and its local helper modules.
terraform {
  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}
data "archive_file" "autotraining_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/autotraining"
  output_path = "${path.module}/autotraining/lambda.zip"
  excludes = [
    "README.md", ".gitignore", "train.py", "run_training.sh",
    "export_backend.sh", "requirements.txt", "workflow.asl.json.tftpl",
    "lambda.zip"
  ]
}

resource "aws_iam_role" "autotraining_lambda" {
  name_prefix = "${var.name}-train-lambda-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}
resource "aws_iam_role_policy_attachment" "autotraining_lambda_logs" {
  role       = aws_iam_role.autotraining_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
resource "aws_iam_role_policy" "autotraining_lambda" {
  name = "${var.name}-train-coordinator"
  role = aws_iam_role.autotraining_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
        Resource = aws_dynamodb_table.autotraining.arn
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.autotraining.arn}/candidates/*"]
      },
      {
        Effect = "Allow"
        Action = ["ec2:StartInstances", "ec2:StopInstances"]
        Resource = aws_instance.autotraining.arn
      },
      {
        Effect = "Allow"
        Action = ["ec2:DescribeInstances"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["ssm:DescribeInstanceInformation", "ssm:GetCommandInvocation"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["ssm:SendCommand"]
        Resource = [
          aws_instance.llm.arn,
          aws_instance.autotraining.arn,
          "arn:aws:ssm:${var.aws_region}::document/AWS-RunShellScript"
        ]
      }
    ]
  })
}
resource "aws_cloudwatch_log_group" "autotraining_lambda" {
  name              = "/aws/lambda/${var.name}-training-coordinator"
  retention_in_days = 14
}
resource "aws_lambda_function" "autotraining" {
  function_name    = "${var.name}-training-coordinator"
  filename         = data.archive_file.autotraining_lambda.output_path
  source_code_hash = data.archive_file.autotraining_lambda.output_base64sha256
  role             = aws_iam_role.autotraining_lambda.arn
  handler          = "coordinator.handler"
  runtime          = "python3.12"
  timeout          = 600
  memory_size      = 256
  environment {
    variables = {
      BACKEND_INSTANCE_ID  = aws_instance.llm.id
      TRAINING_INSTANCE_ID = aws_instance.autotraining.id
      ARTIFACT_BUCKET      = aws_s3_bucket.autotraining.bucket
      STATE_TABLE          = aws_dynamodb_table.autotraining.name
      TRAIN_LOG_GROUP      = aws_cloudwatch_log_group.autotraining.name
    }
  }
  depends_on = [
    aws_iam_role_policy.autotraining_lambda,
    aws_iam_role_policy_attachment.autotraining_lambda_logs,
    aws_cloudwatch_log_group.autotraining_lambda
  ]
}
