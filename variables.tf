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

variable "portainer_allowed_cidr" {
  description = "Legacy admin CIDR required by existing Terraform security group; Portainer is not installed."
  type        = string
  default     = "127.0.0.1/32"
}
