# Development Configuration - Local development environment for Cabruca Segmentation
# Estimated monthly cost: ~$50-75 USD

# Basic Configuration
project_name = "cabruca-segmentation"
environment  = "development"
aws_region   = "sa-east-1"  # São Paulo

# Domain Configuration (placeholder - update with actual values)
domain_name = ""
ssl_certificate_arn = ""

# Instance Configuration - Minimal for development
instance_type_api        = "t3.micro"
instance_type_inference  = "t3.micro"
instance_type_processing = "t3.small"

# Minimal Scaling
api_min_instances       = 1
api_max_instances       = 1
inference_min_instances = 0
inference_max_instances = 1

# Feature Flags - Minimal for dev
enable_gpu         = false
enable_cloudfront  = false
enable_elasticache = false
enable_rds        = false

# Cost Optimization - Maximum savings
cost_optimization = {
  use_spot_instances    = true
  spot_max_price       = "0.10"
  use_reserved_capacity = false
  auto_shutdown_dev    = true
  shutdown_schedule    = "0 18 * * *"  # Shutdown at 6 PM
}

# Minimal Monitoring
monitoring_configuration = {
  enable_cloudwatch   = true
  enable_xray        = false
  log_retention_days = 3
  alarm_email        = ""
  slack_webhook_url  = ""
}

# No Backup for dev
backup_configuration = {
  enable_backup        = false
  backup_retention_days = 0
  backup_window       = ""
  maintenance_window  = ""
  multi_region_backup = false
}

# Network Configuration
network_configuration = {
  vpc_cidr             = "10.2.0.0/16"
  enable_nat_gateway   = true
  enable_vpn          = false
  enable_private_link = false
  allowed_ip_ranges   = ["0.0.0.0/0"]
}

# Minimal Rate Limits
api_rate_limits = {
  "free" = {
    requests_per_minute = 10
    burst_size         = 20
  }
  "basic" = {
    requests_per_minute = 30
    burst_size         = 50
  }
  "premium" = {
    requests_per_minute = 60
    burst_size         = 100
  }
}

# Minimal Model Configuration
model_configurations = {
  "small" = {
    model_path   = "models/cabruca_small.pth"
    batch_size   = 2
    memory_limit = "512Mi"
    cpu_limit    = "0.25"
    gpu_enabled  = false
  }
  "medium" = {
    model_path   = "models/cabruca_medium.pth"
    batch_size   = 4
    memory_limit = "1Gi"
    cpu_limit    = "0.5"
    gpu_enabled  = false
  }
  "large" = {
    model_path   = "models/cabruca_large.pth"
    batch_size   = 8
    memory_limit = "2Gi"
    cpu_limit    = "1"
    gpu_enabled  = false
  }
}

# ML Pipeline Configuration
enable_ml_pipeline        = false
ml_instance_type         = "ml.t3.micro"
training_cpu             = "1024"
training_memory          = "2048"
annotation_enabled       = false
enable_scheduled_training = false
training_schedule        = ""

# Tags
tags = {
  Project     = "Cabruca Segmentation"
  Owner       = "Development Team"
  CostCenter  = "R&D"
  Environment = "Development"
  Region      = "Brazil"
  Purpose     = "Development"
}