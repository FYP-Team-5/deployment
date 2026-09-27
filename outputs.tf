output "instance_id" {
  description = "Use this instance ID with AWS Systems Manager Session Manager."
  value       = aws_instance.llm.id
}

output "region" {
  value = var.aws_region
}

output "public_ip" {
  description = "Outbound access only; the security group has no inbound rules."
  value       = aws_instance.llm.public_ip
}
