# Terraform configuration for Cabruca Segmentation deployment in Brazil
# Optimized for Northeast Brazil region (Bahia, Pernambuco, etc.)

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
  
  backend "s3" {
    bucket = "cabruca-terraform-state-brasil"
    key    = "infrastructure/terraform.tfstate"
    region = "sa-east-1"  # São Paulo region for state storage
  }
}

# Provider configuration for Brazil
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

# Variables
variable "project_name" {
  description = "Nome do projeto para nomenclatura de recursos"
  type        = string
  default     = "cabruca-segmentation"
}

variable "environment" {
  description = "Ambiente (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS Region - São Paulo (closest to Northeast Brazil)"
  type        = string
  default     = "sa-east-1"  # São Paulo - closest AWS region to Northeast Brazil
}

variable "availability_zones" {
  description = "Availability zones for high availability"
  type        = list(string)
  default     = ["sa-east-1a", "sa-east-1b", "sa-east-1c"]
}

variable "instance_type_api" {
  description = "Instance type for API servers"
  type        = string
  default     = "t3.micro"  # 2 vCPUs, 1 GB RAM - Free tier eligible
}

variable "instance_type_inference" {
  description = "Instance type for ML inference"
  type        = string
  default     = "t3.small"  # CPU-only inference for MVP
}

variable "instance_type_processing" {
  description = "Instance type for batch processing"
  type        = string
  default     = "t3.medium"  # 2 vCPUs, 4 GB RAM for processing
}

variable "enable_gpu" {
  description = "Enable GPU instances for ML inference"
  type        = bool
  default     = false  # Disabled for MVP to save costs
}

variable "api_min_instances" {
  description = "Minimum API instances"
  type        = number
  default     = 1  # Single instance for MVP
}

variable "api_max_instances" {
  description = "Maximum API instances for auto-scaling"
  type        = number
  default     = 2  # Limited scaling for MVP
}

variable "inference_min_instances" {
  description = "Minimum inference instances"
  type        = number
  default     = 0  # On-demand only for MVP
}

variable "inference_max_instances" {
  description = "Maximum inference instances"
  type        = number
  default     = 1  # Single instance max for MVP
}

variable "enable_cloudfront" {
  description = "Enable CloudFront CDN for better latency in Northeast Brazil"
  type        = bool
  default     = false  # Disabled for MVP - use ALB directly
}

variable "enable_elasticache" {
  description = "Enable ElastiCache for Redis caching"
  type        = bool
  default     = false  # Disabled for MVP - use in-memory caching
}

variable "enable_rds" {
  description = "Enable RDS PostgreSQL for metadata storage"
  type        = bool
  default     = false  # Disabled for MVP - use SQLite or S3
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "cabruca.agro.br"
}

variable "ssl_certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = ""
}

# Local variables
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Region      = "Brasil-Nordeste"
    ManagedBy   = "Terraform"
    CreatedAt   = timestamp()
  }
  
  # Shortened app name to avoid AWS naming length limits
  app_name = var.environment == "production" ? "cabruca-prod" : (
    var.environment == "staging" ? "cabruca-stg" : (
      var.environment == "development" ? "cabruca-dev" : "cabruca-${substr(var.environment, 0, 3)}"
    )
  )
  
  # Ports
  api_port       = 8000
  streamlit_port = 8501
  
  # CIDR blocks
  vpc_cidr = "10.0.0.0/16"
  public_subnet_cidrs = [
    "10.0.1.0/24",
    "10.0.2.0/24",
    "10.0.3.0/24"
  ]
  private_subnet_cidrs = [
    "10.0.10.0/24",
    "10.0.11.0/24",
    "10.0.12.0/24"
  ]
  
  # Northeast Brazil coordinates for latency-based routing
  northeast_brazil_lat = -9.6658  # Approximate center of Northeast Brazil
  northeast_brazil_lon = -37.5919
}

# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block           = local.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${local.app_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${local.app_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${local.app_name}-public-subnet-${count.index + 1}"
    Type = "Public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name = "${local.app_name}-private-subnet-${count.index + 1}"
    Type = "Private"
  }
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count  = 1  # Single EIP for MVP
  domain = "vpc"
  
  tags = {
    Name = "${local.app_name}-nat-eip-1"
  }
}

# NAT Gateways - Using single NAT Gateway for MVP to save costs
resource "aws_nat_gateway" "main" {
  count         = 1  # Single NAT Gateway for MVP (saves ~$90/month)
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  
  tags = {
    Name = "${local.app_name}-nat-1"
  }
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name = "${local.app_name}-public-rt"
  }
}

resource "aws_route_table" "private" {
  count  = 1  # Single route table for MVP
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[0].id
  }
  
  tags = {
    Name = "${local.app_name}-private-rt-1"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[0].id  # Use single route table
}

# Security Groups
resource "aws_security_group" "alb" {
  name        = "${local.app_name}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP from anywhere"
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS from anywhere"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
  
  tags = {
    Name = "${local.app_name}-alb-sg"
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${local.app_name}-ecs-tasks-sg"
  description = "Security group for ECS tasks"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = local.api_port
    to_port         = local.api_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "API port from ALB"
  }
  
  ingress {
    from_port       = local.streamlit_port
    to_port         = local.streamlit_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Streamlit port from ALB"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
  
  tags = {
    Name = "${local.app_name}-ecs-tasks-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = aws_subnet.public[*].id
  
  enable_deletion_protection = false
  enable_http2              = true
  enable_cross_zone_load_balancing = true
  
  tags = {
    Name = "${local.app_name}-alb"
  }
}

# Target Groups
resource "aws_lb_target_group" "api" {
  name        = "${local.app_name}-api-tg"
  port        = local.api_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }
  
  deregistration_delay = 30
  
  tags = {
    Name = "${local.app_name}-api-tg"
  }
}

resource "aws_lb_target_group" "streamlit" {
  name        = "${local.app_name}-streamlit-tg"
  port        = local.streamlit_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/"
    matcher             = "200"
  }
  
  deregistration_delay = 30
  
  tags = {
    Name = "${local.app_name}-streamlit-tg"
  }
}

# ALB Listeners
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# HTTPS listener (only if domain is configured)
resource "aws_lb_listener" "https" {
  count = var.domain_name != "" ? 1 : 0
  
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.ssl_certificate_arn != "" ? var.ssl_certificate_arn : aws_acm_certificate.main[0].arn
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# Listener Rules for routing
resource "aws_lb_listener_rule" "streamlit" {
  count = var.domain_name != "" ? 1 : 0
  listener_arn = aws_lb_listener.https[0].arn
  priority     = 100
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.streamlit.arn
  }
  
  condition {
    path_pattern {
      values = ["/dashboard*", "/viewer*"]
    }
  }
}

# S3 Buckets for model storage and data
resource "aws_s3_bucket" "models" {
  bucket = "${local.app_name}-models-brasil"
  
  tags = {
    Name = "${local.app_name}-models"
    Type = "ModelStorage"
  }
}

resource "aws_s3_bucket" "data" {
  bucket = "${local.app_name}-data-brasil"
  
  tags = {
    Name = "${local.app_name}-data"
    Type = "DataStorage"
  }
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# CloudFront Distribution for low latency in Northeast Brazil
resource "aws_cloudfront_distribution" "main" {
  count = var.enable_cloudfront ? 1 : 0
  
  enabled             = true
  is_ipv6_enabled    = true
  default_root_object = "index.html"
  
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "${local.app_name}-alb"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  origin {
    domain_name = aws_s3_bucket.models.bucket_regional_domain_name
    origin_id   = "${local.app_name}-s3-models"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main[0].cloudfront_access_identity_path
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "${local.app_name}-alb"
    
    forwarded_values {
      query_string = true
      headers      = ["Host", "Accept", "Accept-Language", "Accept-Encoding", "Authorization"]
      
      cookies {
        forward = "all"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
  }
  
  # Cache behavior for static model files
  ordered_cache_behavior {
    path_pattern     = "/models/*"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "${local.app_name}-s3-models"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 86400
    max_ttl                = 31536000
    compress               = true
  }
  
  price_class = "PriceClass_All"  # Use all edge locations for best performance in Brazil
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = var.domain_name == "" ? true : false
    acm_certificate_arn            = var.domain_name != "" ? aws_acm_certificate.main[0].arn : null
    ssl_support_method             = var.domain_name != "" ? "sni-only" : null
  }
  
  tags = {
    Name = "${local.app_name}-cdn"
  }
}

resource "aws_cloudfront_origin_access_identity" "main" {
  count   = var.enable_cloudfront ? 1 : 0
  comment = "${local.app_name} CloudFront OAI"
}

# ElastiCache for Redis (caching)
resource "aws_elasticache_subnet_group" "main" {
  count      = var.enable_elasticache ? 1 : 0
  name       = "${local.app_name}-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name = "${local.app_name}-cache-subnet"
  }
}

resource "aws_elasticache_replication_group" "main" {
  count                      = var.enable_elasticache ? 1 : 0
  replication_group_id       = "${local.app_name}-redis"
  description               = "Redis cache for ${local.app_name}"
  node_type                 = "cache.t3.micro"
  port                      = 6379
  parameter_group_name      = "default.redis7"
  automatic_failover_enabled = true
  multi_az_enabled          = true
  num_cache_clusters        = 2
  subnet_group_name         = aws_elasticache_subnet_group.main[0].name
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "${local.app_name}-redis"
  }
}

# RDS PostgreSQL for metadata
resource "aws_db_subnet_group" "main" {
  count      = var.enable_rds ? 1 : 0
  name       = "${local.app_name}-db-subnet"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name = "${local.app_name}-db-subnet"
  }
}

resource "aws_db_instance" "main" {
  count                   = var.enable_rds ? 1 : 0
  identifier             = "${local.app_name}-db"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.medium"
  allocated_storage      = 100
  storage_type           = "gp3"
  storage_encrypted      = true
  
  db_name  = "cabruca"
  username = "cabruca_admin"
  password = random_password.db_password[0].result
  
  vpc_security_group_ids = [aws_security_group.rds[0].id]
  db_subnet_group_name   = aws_db_subnet_group.main[0].name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  multi_az               = true
  publicly_accessible    = false
  
  skip_final_snapshot    = false
  final_snapshot_identifier = "${local.app_name}-db-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  tags = {
    Name = "${local.app_name}-db"
  }
}

resource "random_password" "db_password" {
  count   = var.enable_rds ? 1 : 0
  length  = 32
  special = true
}

resource "aws_security_group" "rds" {
  count       = var.enable_rds ? 1 : 0
  name        = "${local.app_name}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
    description     = "PostgreSQL from ECS tasks"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${local.app_name}-rds-sg"
  }
}

# ACM Certificate for HTTPS
resource "aws_acm_certificate" "main" {
  count             = var.domain_name != "" && var.ssl_certificate_arn == "" ? 1 : 0
  domain_name       = var.domain_name
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.${var.domain_name}"
  ]
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = {
    Name = "${local.app_name}-cert"
  }
}

# Outputs
output "load_balancer_url" {
  description = "URL do Load Balancer"
  value       = "https://${aws_lb.main.dns_name}"
}

output "cloudfront_url" {
  description = "CloudFront URL for low latency access"
  value       = var.enable_cloudfront ? "https://${aws_cloudfront_distribution.main[0].domain_name}" : "N/A"
}

output "api_endpoint" {
  description = "API endpoint"
  value       = "https://${aws_lb.main.dns_name}/api"
}

output "streamlit_endpoint" {
  description = "Streamlit dashboard"
  value       = "https://${aws_lb.main.dns_name}/dashboard"
}

output "s3_models_bucket" {
  description = "S3 bucket for models"
  value       = aws_s3_bucket.models.id
}

output "s3_data_bucket" {
  description = "S3 bucket for data"
  value       = aws_s3_bucket.data.id
}

output "redis_endpoint" {
  description = "Redis cache endpoint"
  value       = var.enable_elasticache ? aws_elasticache_replication_group.main[0].primary_endpoint_address : "N/A"
}

output "database_endpoint" {
  description = "PostgreSQL database endpoint"
  value       = var.enable_rds ? aws_db_instance.main[0].endpoint : "N/A"
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}