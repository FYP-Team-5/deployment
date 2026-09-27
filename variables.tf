variable "aws_region" {
  description = "AWS Region for the GPU instance. Check g4dn.xlarge capacity and quotas in this Region."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name prefix for the deployment resources."
  type        = string
  default     = "fyp-qwen-vllm"
}
