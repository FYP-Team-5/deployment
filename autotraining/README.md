# Nightly QLoRA retraining

This directory contains infrastructure and scripts for a separate, scheduled Qwen3.5-9B training EC2 instance. The existing serving instance remains unchanged.

The workflow is deployed only after Terraform has been reviewed and applied. New adapters are evaluated and stored as candidates; serving is not automatically updated.
