variable "aws_region" {
  description = "AWS Region for the GPU instance. Check g5.xlarge capacity and quotas in this Region."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name prefix for the deployment resources."
  type        = string
  default     = "fyp-qwen-vllm"
}

variable "availability_zone" {
  description = "Optional subnet AZ. Must support both serving and training instance types. Defaults to the first available supported AZ in the region."
  type        = string
  default     = null
}

variable "vllm_hf_token_parameter" {
  description = "Existing SSM SecureString containing an HF token with read access to both serving adapters. Reuses the existing training token by default. Empty disables SSM lookup; Terraform never reads the token value."
  type        = string
  default     = "/fyp/training/hf-token"
  nullable    = false

  validation {
    condition     = var.vllm_hf_token_parameter == "" || can(regex("^/[A-Za-z0-9_./-]+$", var.vllm_hf_token_parameter))
    error_message = "Use an absolute SSM parameter path, or an empty string to disable SSM lookup."
  }
}

variable "vllm_hf_token_kms_key_arn" {
  description = "Optional customer-managed KMS key ARN encrypting the serving HF token. Leave null for the default aws/ssm key."
  type        = string
  default     = null
}
