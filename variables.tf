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

variable "vllm_hf_token_parameter" {
  description = "Existing SSM SecureString containing an HF read token for both serving adapters. Empty disables SSM lookup; the token value is never read by Terraform."
  type        = string
  default     = "/fyp/serving/hf-token"
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

variable "portainer_allowed_cidr" {
  description = "Legacy admin CIDR required by existing Terraform security group; Portainer is not installed."
  type        = string
  default     = "127.0.0.1/32"
}
