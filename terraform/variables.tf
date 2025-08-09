# Variables for Terraform deployment - Brazil optimized

variable "aws_access_key" {
  description = "AWS Access Key ID"
  type        = string
  sensitive   = true
  default     = ""
}

variable "aws_secret_key" {
  description = "AWS Secret Access Key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "deployment_environment" {
  description = "Deployment environment configurations"
  type = object({
    dev = object({
      api_instances     = number
      inference_instances = number
      enable_gpu       = bool
      enable_cache     = bool
      enable_cdn       = bool
      enable_rds       = bool
    })
    staging = object({
      api_instances     = number
      inference_instances = number
      enable_gpu       = bool
      enable_cache     = bool
      enable_cdn       = bool
      enable_rds       = bool
    })
    prod = object({
      api_instances     = number
      inference_instances = number
      enable_gpu       = bool
      enable_cache     = bool
      enable_cdn       = bool
      enable_rds       = bool
    })
  })
  
  default = {
    dev = {
      api_instances     = 1
      inference_instances = 0
      enable_gpu       = false
      enable_cache     = false
      enable_cdn       = false
      enable_rds       = false
    }
    staging = {
      api_instances     = 2
      inference_instances = 1
      enable_gpu       = false
      enable_cache     = true
      enable_cdn       = true
      enable_rds       = true
    }
    prod = {
      api_instances     = 3
      inference_instances = 2
      enable_gpu       = true
      enable_cache     = true
      enable_cdn       = true
      enable_rds       = true
    }
  }
}

variable "brazil_regions" {
  description = "Brazilian region configurations for optimized deployment"
  type = map(object({
    name              = string
    availability_zones = list(string)
    edge_locations    = list(string)
  }))
  
  default = {
    "sa-east-1" = {
      name = "São Paulo"
      availability_zones = ["sa-east-1a", "sa-east-1b", "sa-east-1c"]
      edge_locations = ["São Paulo", "Rio de Janeiro"]
    }
  }
}

variable "northeast_brazil_cities" {
  description = "Northeast Brazil cities for latency optimization"
  type = list(string)
  default = [
    "Salvador",
    "Recife",
    "Fortaleza",
    "São Luís",
    "Natal",
    "João Pessoa",
    "Maceió",
    "Aracaju",
    "Teresina",
    "Ilhéus",
    "Feira de Santana",
    "Vitória da Conquista",
    "Camacan"
  ]
}

variable "model_configurations" {
  description = "ML model configurations"
  type = map(object({
    model_path    = string
    batch_size    = number
    memory_limit  = string
    cpu_limit     = string
    gpu_enabled   = bool
  }))
  
  default = {
    "small" = {
      model_path   = "models/cabruca_small.pth"
      batch_size   = 4
      memory_limit = "2Gi"
      cpu_limit    = "1"
      gpu_enabled  = false
    }
    "medium" = {
      model_path   = "models/cabruca_medium.pth"
      batch_size   = 8
      memory_limit = "4Gi"
      cpu_limit    = "2"
      gpu_enabled  = false
    }
    "large" = {
      model_path   = "models/cabruca_large.pth"
      batch_size   = 16
      memory_limit = "8Gi"
      cpu_limit    = "4"
      gpu_enabled  = true
    }
  }
}

variable "api_rate_limits" {
  description = "API rate limiting configuration"
  type = map(object({
    requests_per_minute = number
    burst_size         = number
  }))
  
  default = {
    "free" = {
      requests_per_minute = 10
      burst_size         = 20
    }
    "basic" = {
      requests_per_minute = 60
      burst_size         = 100
    }
    "premium" = {
      requests_per_minute = 300
      burst_size         = 500
    }
  }
}

variable "backup_configuration" {
  description = "Backup and disaster recovery configuration"
  type = object({
    enable_backup        = bool
    backup_retention_days = number
    backup_window       = string
    maintenance_window  = string
    multi_region_backup = bool
  })
  
  default = {
    enable_backup        = true
    backup_retention_days = 30
    backup_window       = "03:00-04:00"  # BRT (UTC-3)
    maintenance_window  = "sun:04:00-sun:05:00"
    multi_region_backup = false
  }
}

variable "monitoring_configuration" {
  description = "Monitoring and alerting configuration"
  type = object({
    enable_cloudwatch    = bool
    enable_xray         = bool
    log_retention_days  = number
    alarm_email         = string
    slack_webhook_url   = string
  })
  
  default = {
    enable_cloudwatch   = true
    enable_xray        = false
    log_retention_days = 30
    alarm_email        = ""
    slack_webhook_url  = ""
  }
}

variable "network_configuration" {
  description = "Network configuration for Brazil deployment"
  type = object({
    vpc_cidr             = string
    enable_nat_gateway   = bool
    enable_vpn          = bool
    enable_private_link = bool
    allowed_ip_ranges   = list(string)
  })
  
  default = {
    vpc_cidr             = "10.0.0.0/16"
    enable_nat_gateway   = true
    enable_vpn          = false
    enable_private_link = false
    allowed_ip_ranges   = ["0.0.0.0/0"]  # Allow all, restrict in production
  }
}

variable "cost_optimization" {
  description = "Cost optimization settings"
  type = object({
    use_spot_instances    = bool
    spot_max_price       = string
    use_reserved_capacity = bool
    auto_shutdown_dev    = bool
    shutdown_schedule    = string
  })
  
  default = {
    use_spot_instances    = false
    spot_max_price       = "0.50"
    use_reserved_capacity = false
    auto_shutdown_dev    = true
    shutdown_schedule    = "0 22 * * MON-FRI"  # Shutdown at 10 PM on weekdays
  }
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "Cabruca Segmentation"
    Owner       = "AgroTech Team"
    CostCenter  = "Research"
    Environment = "Production"
    Region      = "Brazil-Northeast"
    Compliance  = "LGPD"
  }
}