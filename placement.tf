# A subnet fixes the AZ for both instances, so require support for both types.
data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "zone-type"
    values = ["availability-zone"]
  }
}

data "aws_ec2_instance_type_offerings" "gpu" {
  for_each      = toset([local.serving_instance_type, local.training_instance_type])
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [each.value]
  }
}

locals {
  serving_instance_type  = "g5.xlarge"
  training_instance_type = coalesce(var.autotraining_instance_type, local.serving_instance_type)
  supported_availability_zones = sort(tolist(setintersection(
    toset(data.aws_availability_zones.available.names),
    [for offering in data.aws_ec2_instance_type_offerings.gpu : toset(offering.locations)]...
  )))
  selected_availability_zone = var.availability_zone != null ? var.availability_zone : try(local.supported_availability_zones[0], null)
}
