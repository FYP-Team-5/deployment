# Isolated nightly training data plane. No changes to existing serving EC2.
data "aws_caller_identity" "autotraining" {}

variable "autotraining_schedule_enabled" {
  type        = bool
  default     = false
  description = "Enable after validating the IAM, dataset and model pipeline."
}
variable "autotraining_instance_type" {
  type        = string
  default     = null
  description = "Optional dedicated training instance type override. Defaults to the serving EC2 instance type."
}
variable "autotraining_force_destroy" {
  type        = bool
  default     = true
  nullable    = false
  description = "Allow Terraform to permanently delete all training-bucket objects, versions and delete markers when destroying the bucket. Defaults to true; set to false to guard against accidental teardown."
}

resource "aws_s3_bucket" "autotraining" {
  bucket        = "${var.name}-training-${data.aws_caller_identity.autotraining.account_id}"
  force_destroy = var.autotraining_force_destroy
}
resource "aws_s3_bucket_public_access_block" "autotraining" {
  bucket                  = aws_s3_bucket.autotraining.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "autotraining" {
  bucket = aws_s3_bucket.autotraining.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
resource "aws_s3_bucket_versioning" "autotraining" {
  bucket = aws_s3_bucket.autotraining.id
  versioning_configuration { status = "Enabled" }
}

# S3 can return NoSuchKey for new objects while first-time versioning propagates.
# AWS recommends waiting 15 minutes before writes. Repeat only when versioning
# is created/reconfigured, including bucket replacement, not on script updates.
resource "time_sleep" "autotraining_versioning" {
  depends_on      = [aws_s3_bucket_versioning.autotraining]
  create_duration = "15m"

  lifecycle {
    replace_triggered_by = [aws_s3_bucket_versioning.autotraining]
  }
}

locals {
  autotraining_scripts = {
    "export_backend.sh" = "autotraining/export_backend.sh"
    "run_training.sh"   = "autotraining/run_training.sh"
    "train.py"          = "autotraining/train.py"
    "train_feedback.py" = "autotraining/train_feedback.py"
    "requirements.txt"  = "autotraining/requirements.txt"
    "publish.py"        = "autotraining/publish.py"
    "publish.sh"        = "autotraining/publish.sh"
  }
}
resource "aws_s3_object" "autotraining_scripts" {
  for_each               = local.autotraining_scripts
  bucket                 = aws_s3_bucket.autotraining.id
  key                    = "scripts/${each.key}"
  source                 = "${path.module}/${each.value}"
  etag                   = filemd5("${path.module}/${each.value}")
  server_side_encryption = "AES256"

  depends_on = [time_sleep.autotraining_versioning]
}

# Adds narrowly scoped S3 permissions to the EXISTING serving role.
# Changing this policy does not change user-data or replace the backend EC2.
resource "aws_iam_role_policy" "autotraining_backend_export" {
  name = "${var.name}-dataset-export"
  role = aws_iam_role.llm.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.autotraining.arn}/scripts/export_backend.sh", "${aws_s3_bucket.autotraining.arn}/scripts/publish*", "${aws_s3_bucket.autotraining.arn}/datasets/*", "${aws_s3_bucket.autotraining.arn}/candidates/*", "${aws_s3_bucket.autotraining.arn}/publications/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.autotraining.arn}/datasets/*", "${aws_s3_bucket.autotraining.arn}/publications/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.autotraining.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObjectTagging"]
        Resource = "${aws_s3_bucket.autotraining.arn}/candidates/*"
      },
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = [for parameter in [var.autotraining_hf_token_parameter, var.autotraining_callback_token_parameter] : "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.autotraining.account_id}:parameter${parameter}"]
      }

    ]
  })
}

output "autotraining_s3_bucket" {
  value = aws_s3_bucket.autotraining.bucket
}

output "autotraining_control_prefix" {
  value = "s3://${aws_s3_bucket.autotraining.bucket}/control/"
}

variable "autotraining_hf_repo" {
  type    = string
  default = "SmuFypTeam5/GradingQlora"
}
variable "autotraining_feedback_hf_repo" {
  type    = string
  default = "SmuFypTeam5/feedback"
}
variable "autotraining_hf_token_parameter" {
  description = "Existing SSM SecureString containing the HF token used only for branch creation/uploads; public reads need no token."
  type        = string
  default     = "/fyp/training/hf-token"
}
variable "autotraining_callback_token_parameter" {
  description = "Existing SSM SecureString matching backend TRAINING_CALLBACK_TOKEN."
  type        = string
  default     = "/fyp/training/callback-token"
}

# HF keeps published commits. Bound duplicate staging storage while preserving
# unpublished artifacts and all current control/receipt records.
resource "aws_s3_bucket_lifecycle_configuration" "autotraining" {
  bucket = aws_s3_bucket.autotraining.id
  rule {
    id     = "published-candidate-staging"
    status = "Enabled"
    filter {
      and {
        prefix = "candidates/"
        tags   = { published = "true" }
      }
    }
    expiration { days = 30 }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
  rule {
    id     = "old-control-versions"
    status = "Enabled"
    filter { prefix = "control/" }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
  rule {
    id     = "abandoned-multipart-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}
