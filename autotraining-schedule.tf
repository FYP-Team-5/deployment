resource "aws_iam_role" "autotraining_stepfunctions" {
  name_prefix = "${var.name}-training-sfn-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = { Service = "states.amazonaws.com" }
    }]
  })
}
resource "aws_iam_role_policy" "autotraining_stepfunctions" {
  name = "${var.name}-invoke-coordinator"
  role = aws_iam_role.autotraining_stepfunctions.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["lambda:InvokeFunction"]
      Resource = aws_lambda_function.autotraining.arn
    }]
  })
}
resource "aws_sfn_state_machine" "autotraining" {
  name     = "${var.name}-nightly-training"
  role_arn = aws_iam_role.autotraining_stepfunctions.arn
  type     = "STANDARD"
  definition = templatefile("${path.module}/autotraining/workflow.asl.json.tftpl", {
    lambda_arn = aws_lambda_function.autotraining.arn
  })
  depends_on = [aws_iam_role_policy.autotraining_stepfunctions]
}

resource "aws_iam_role" "autotraining_scheduler" {
  name_prefix = "${var.name}-training-scheduler-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = { Service = "scheduler.amazonaws.com" }
    }]
  })
}
resource "aws_iam_role_policy" "autotraining_scheduler" {
  name = "${var.name}-start-nightly-training"
  role = aws_iam_role.autotraining_scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["states:StartExecution"]
      Resource = aws_sfn_state_machine.autotraining.arn
    }]
  })
}

resource "aws_scheduler_schedule" "autotraining" {
  name                         = "${var.name}-nightly-training"
  schedule_expression          = "cron(10 0 * * ? *)"
  schedule_expression_timezone = "Asia/Singapore"
  state                        = var.autotraining_schedule_enabled ? "ENABLED" : "DISABLED"
  flexible_time_window {
    mode = "OFF"
  }
  target {
    arn      = aws_sfn_state_machine.autotraining.arn
    role_arn = aws_iam_role.autotraining_scheduler.arn
    input    = "{}"
    retry_policy {
      maximum_retry_attempts = 0
    }
  }
  depends_on = [
    aws_iam_role_policy.autotraining_scheduler,
    aws_s3_object.autotraining_scripts,
    aws_iam_role_policy.autotraining_backend_export
  ]
}

output "autotraining_state_machine_arn" {
  description = "Start a manual run with AWS Step Functions."
  value       = aws_sfn_state_machine.autotraining.arn
}
output "autotraining_schedule_name" {
  value = aws_scheduler_schedule.autotraining.name
}
