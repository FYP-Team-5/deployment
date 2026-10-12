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

# Required: your public IPv4 address as a /32 CIDR, or a trusted team/VPN CIDR.
# Intentionally has no default: never expose an admin UI to everyone by accident.
variable "portainer_allowed_cidr" {
  description = "Trusted IPv4 CIDR allowed to reach Portainer HTTPS (9443); e.g. 203.0.113.10/32. Do not use 0.0.0.0/0."
  type        = string

  validation {
    condition     = can(cidrnetmask(var.portainer_allowed_cidr)) && var.portainer_allowed_cidr != "0.0.0.0/0"
    error_message = "Supply a trusted IPv4 CIDR, not 0.0.0.0/0 (e.g. YOUR_PUBLIC_IP/32)."
  }
}
