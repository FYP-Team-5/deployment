# Isolated nightly training data plane. No changes to existing serving EC2.
data "aws_caller_identity" "autotraining" {}

variable "autotraining_schedule_enabled" {
  type        = bool
  default     = false
  description = "Enable after validating the IAM, dataset and model pipeline."
}
variable "autotraining_instance_type" {
  type        = string
  default     = "g6e.xlarge"
  description = "Dedicated GPU training EC2 instance type."
}

resource "aws_s3_bucket" "autotraining" {
  bucket = "${var.name}-training-${data.aws_caller_identity.autotraining.account_id}"
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
resource "aws_dynamodb_table" "autotraining" {
  name         = "${var.name}-training-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  attribute {
    name = "pk"
    type = "S"
  }
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

locals {
  autotraining_scripts = {
    "export_backend.sh" = "autotraining/export_backend.sh"
    "run_training.sh"   = "autotraining/run_training.sh"
    "train.py"          = "autotraining/train.py"
    "requirements.txt"  = "autotraining/requirements.txt"
  }
}
resource "aws_s3_object" "autotraining_scripts" {
  for_each               = local.autotraining_scripts
  bucket                 = aws_s3_bucket.autotraining.id
  key                    = "scripts/${each.key}"
  source                 = "${path.module}/${each.value}"
  etag                   = filemd5("${path.module}/${each.value}")
  server_side_encryption = "AES256"
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
        Resource = ["${aws_s3_bucket.autotraining.arn}/scripts/export_backend.sh"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.autotraining.arn}/datasets/*"]
      }
    ]
  })
}

output "autotraining_s3_bucket" {
  value = aws_s3_bucket.autotraining.bucket
}
