# MVP Configuration - Minimal cost deployment for Cabruca Segmentation
# Estimated monthly cost: ~$50-75 USD

# Basic Configuration
project_name = "cabruca-mvp"
environment  = "mvp"
aws_region   = "sa-east-1" # São Paulo - closest to Northeast Brazil

# Domain Configuration
domain_name         = "theobroma.digital"
ssl_certificate_arn = "arn:aws:acm:sa-east-1:919014037196:certificate/4ef207ea-4db8-44b7-839e-e94f78fa86eb"

# Instance Configuration - Minimal sizes
instance_type_api        = "t3.micro"  # Free tier eligible
instance_type_inference  = "t3.small"  # CPU-only inference
instance_type_processing = "t3.medium" # Batch processing

# Minimal Scaling Configuration
api_min_instances       = 1 # Single API instance
api_max_instances       = 2 # Limited auto-scaling
inference_min_instances = 0 # On-demand only
inference_max_instances = 1 # Single instance max

# Feature Flags - All optional features disabled for MVP
enable_gpu         = false # No GPU for MVP (saves ~$400/month)
enable_cloudfront  = false # No CDN for MVP (saves ~$50/month)
enable_elasticache = false # No Redis for MVP (saves ~$50/month)
enable_rds         = false # No RDS for MVP (saves ~$100/month)

# Cost Optimization - Maximum savings
cost_optimization = {
  use_spot_instances    = true         # Use spot for 70% cost savings
  spot_max_price        = "0.10"       # Very low spot price
  use_reserved_capacity = false        # No reserved instances for MVP
  auto_shutdown_dev     = true         # Auto-shutdown after hours
  shutdown_schedule     = "0 20 * * *" # Shutdown at 8 PM daily
}

# Minimal Monitoring Configuration
monitoring_configuration = {
  enable_cloudwatch  = true                       # Keep basic monitoring
  enable_xray        = false                      # No tracing for MVP
  log_retention_days = 7                          # Short retention (saves costs)
  alarm_email        = "alerts@theobroma.digital" # Add your email for critical alerts
  slack_webhook_url  = ""                         # Optional
}

# Minimal Backup Configuration
backup_configuration = {
  enable_backup         = false # No automated backups for MVP
  backup_retention_days = 0
  backup_window         = ""
  maintenance_window    = ""
  multi_region_backup   = false
}

# Simplified Network Configuration
network_configuration = {
  vpc_cidr            = "10.0.0.0/16"
  enable_nat_gateway  = true          # Required but using single NAT
  enable_vpn          = false         # No VPN for MVP
  enable_private_link = false         # No PrivateLink for MVP
  allowed_ip_ranges   = ["0.0.0.0/0"] # Open access for MVP
}

# MVP API Rate Limits
api_rate_limits = {
  "free" = {
    requests_per_minute = 10
    burst_size          = 20
  }
  "basic" = {
    requests_per_minute = 30
    burst_size          = 50
  }
  "premium" = {
    requests_per_minute = 60
    burst_size          = 100
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

# ML Pipeline Configuration - On-demand training and annotation
enable_ml_pipeline        = true           # Enable ML components
ml_instance_type          = "ml.t3.medium" # Lowest cost ML instance (~$0.05/hour)
training_cpu              = "1024"         # 1 vCPU for training
training_memory           = "2048"         # 2 GB for training (MVP)
annotation_enabled        = false          # Start annotation on-demand
enable_scheduled_training = false          # Manual training only for MVP
training_schedule         = ""             # No automated schedule

# Tags for MVP resources
tags = {
  Project     = "Cabruca Segmentation MVP"
  Owner       = "Development Team"
  CostCenter  = "R&D"
  Environment = "MVP"
  Region      = "Brazil"
  Budget      = "Minimal"
}