# Production Configuration - Production environment for Cabruca Segmentation
# Estimated monthly cost: ~$500-800 USD

# Basic Configuration
project_name = "cabruca-segmentation"
environment  = "production"
aws_region   = "sa-east-1" # São Paulo

# Domain Configuration (placeholder - update with actual values)
domain_name         = ""
ssl_certificate_arn = ""

# Instance Configuration - Production sizes
instance_type_api        = "t3.medium"
instance_type_inference  = "t3.large"
instance_type_processing = "t3.xlarge"

# Production Scaling
api_min_instances       = 2  # HA configuration
api_max_instances       = 10 # Auto-scaling for load
inference_min_instances = 1  # Always available
inference_max_instances = 5  # Scale as needed

# Feature Flags - Full features for production
enable_gpu         = false # Enable if needed for performance
enable_cloudfront  = true  # CDN for global distribution
enable_elasticache = true  # Redis for caching
enable_rds         = true  # RDS for data persistence

# Cost Optimization - Balanced for production
cost_optimization = {
  use_spot_instances    = false # No spot for production stability
  spot_max_price        = "0"
  use_reserved_capacity = true  # Reserved instances for cost savings
  auto_shutdown_dev     = false # Always on
  shutdown_schedule     = ""
}

# Full Monitoring
monitoring_configuration = {
  enable_cloudwatch  = true
  enable_xray        = true # Full tracing
  log_retention_days = 90   # 3 months retention
  alarm_email        = ""   # Will be set via secrets
  slack_webhook_url  = ""   # Will be set via secrets
}

# Production Backup
backup_configuration = {
  enable_backup         = true
  backup_retention_days = 30
  backup_window         = "03:00-04:00"
  maintenance_window    = "sun:04:00-sun:05:00"
  multi_region_backup   = true # Cross-region backups
}

# Network Configuration
network_configuration = {
  vpc_cidr            = "10.0.0.0/16"
  enable_nat_gateway  = true
  enable_vpn          = true # VPN for secure access
  enable_private_link = true # PrivateLink for security
  allowed_ip_ranges   = []   # Will be configured with specific IPs
}

# Production Rate Limits
api_rate_limits = {
  "free" = {
    requests_per_minute = 60
    burst_size          = 120
  }
  "basic" = {
    requests_per_minute = 300
    burst_size          = 500
  }
  "premium" = {
    requests_per_minute = 1000
    burst_size          = 2000
  }
}

# Production Model Configuration
model_configurations = {
  "small" = {
    model_path   = "models/cabruca_small.pth"
    batch_size   = 16
    memory_limit = "2Gi"
    cpu_limit    = "1"
    gpu_enabled  = false
  }
  "medium" = {
    model_path   = "models/cabruca_medium.pth"
    batch_size   = 32
    memory_limit = "4Gi"
    cpu_limit    = "2"
    gpu_enabled  = false
  }
  "large" = {
    model_path   = "models/cabruca_large.pth"
    batch_size   = 64
    memory_limit = "8Gi"
    cpu_limit    = "4"
    gpu_enabled  = false
  }
}

# ML Pipeline Configuration
enable_ml_pipeline        = true
ml_instance_type          = "ml.m5.xlarge"
training_cpu              = "4096"  # 4 vCPUs
training_memory           = "16384" # 16 GB
annotation_enabled        = true
enable_scheduled_training = true        # Automated retraining
training_schedule         = "0 2 * * 0" # Weekly on Sunday at 2 AM

# Tags
tags = {
  Project     = "Cabruca Segmentation"
  Owner       = "Production Team"
  CostCenter  = "Operations"
  Environment = "Production"
  Region      = "Brazil"
  Purpose     = "Production"
  SLA         = "99.9"
}