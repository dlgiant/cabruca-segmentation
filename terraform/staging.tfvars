# Staging Configuration - Test environment for Cabruca Segmentation
# Estimated monthly cost: ~$100-150 USD

# Basic Configuration
project_name = "cabruca-segmentation"
environment  = "staging"
aws_region   = "sa-east-1" # São Paulo - closest to Northeast Brazil

# Domain Configuration (placeholder - update with actual values)
domain_name         = ""
ssl_certificate_arn = ""

# Instance Configuration - Small sizes for staging
instance_type_api        = "t3.micro"  # Minimal API instance
instance_type_inference  = "t3.small"  # Small inference instance
instance_type_processing = "t3.medium" # Processing instance

# Moderate Scaling Configuration
api_min_instances       = 1 # Single API instance
api_max_instances       = 3 # Limited auto-scaling
inference_min_instances = 0 # On-demand only
inference_max_instances = 2 # Two instances max

# Feature Flags - Basic features for staging
enable_gpu         = false # No GPU for staging
enable_cloudfront  = false # No CDN for staging
enable_elasticache = false # No Redis for staging
enable_rds         = false # No RDS for staging

# Cost Optimization - Balanced for staging
cost_optimization = {
  use_spot_instances    = true         # Use spot for cost savings
  spot_max_price        = "0.15"       # Moderate spot price
  use_reserved_capacity = false        # No reserved instances
  auto_shutdown_dev     = true         # Auto-shutdown after hours
  shutdown_schedule     = "0 22 * * *" # Shutdown at 10 PM daily
}

# Monitoring Configuration
monitoring_configuration = {
  enable_cloudwatch  = true  # Basic monitoring
  enable_xray        = false # No tracing for staging
  log_retention_days = 14    # Two weeks retention
  alarm_email        = ""    # Will be set via secrets
  slack_webhook_url  = ""    # Optional
}

# Backup Configuration
backup_configuration = {
  enable_backup         = true # Basic backups for staging
  backup_retention_days = 7
  backup_window         = "03:00-04:00"
  maintenance_window    = "sun:04:00-sun:05:00"
  multi_region_backup   = false
}

# Network Configuration
network_configuration = {
  vpc_cidr            = "10.1.0.0/16"
  enable_nat_gateway  = true          # Required
  enable_vpn          = false         # No VPN for staging
  enable_private_link = false         # No PrivateLink
  allowed_ip_ranges   = ["0.0.0.0/0"] # Open access for testing
}

# API Rate Limits
api_rate_limits = {
  "free" = {
    requests_per_minute = 20
    burst_size          = 40
  }
  "basic" = {
    requests_per_minute = 60
    burst_size          = 100
  }
  "premium" = {
    requests_per_minute = 120
    burst_size          = 200
  }
}

# Model Configuration
model_configurations = {
  "small" = {
    model_path   = "models/cabruca_small.pth"
    batch_size   = 4
    memory_limit = "1Gi"
    cpu_limit    = "0.5"
    gpu_enabled  = false
  }
  "medium" = {
    model_path   = "models/cabruca_medium.pth"
    batch_size   = 8
    memory_limit = "2Gi"
    cpu_limit    = "1"
    gpu_enabled  = false
  }
  "large" = {
    model_path   = "models/cabruca_large.pth"
    batch_size   = 16
    memory_limit = "4Gi"
    cpu_limit    = "2"
    gpu_enabled  = false
  }
}

# ML Pipeline Configuration
enable_ml_pipeline        = true
ml_instance_type          = "ml.t3.medium"
training_cpu              = "2048" # 2 vCPUs for training
training_memory           = "4096" # 4 GB for training
annotation_enabled        = true   # Enable annotation
enable_scheduled_training = false  # Manual training for staging
training_schedule         = ""

# Tags for staging resources
tags = {
  Project     = "Cabruca Segmentation"
  Owner       = "Development Team"
  CostCenter  = "R&D"
  Environment = "Staging"
  Region      = "Brazil"
  Purpose     = "Testing"
}