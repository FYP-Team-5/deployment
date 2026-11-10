# Read only the parameter identity here. The vLLM process retrieves its value
# at startup using the existing EC2 instance profile, outside Terraform state.
data "aws_caller_identity" "vllm" {}
data "aws_partition" "vllm" {}

locals {
  vllm_hf_token_parameter_arn = "arn:${data.aws_partition.vllm.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.vllm.account_id}:parameter${var.vllm_hf_token_parameter}"
}

resource "aws_iam_role_policy" "vllm_hf_token" {
  count = var.vllm_hf_token_parameter != "" ? 1 : 0
  name  = "${var.name}-serving-hf-token"
  role  = aws_iam_role.llm.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = [local.vllm_hf_token_parameter_arn]
      }
      ], var.vllm_hf_token_kms_key_arn == null ? [] : [
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = [var.vllm_hf_token_kms_key_arn]
        Condition = {
          StringEquals = {
            "kms:EncryptionContext:PARAMETER_ARN" = local.vllm_hf_token_parameter_arn
          }
        }
      }
    ])
  })
}
