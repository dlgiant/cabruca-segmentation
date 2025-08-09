# Terraform Infrastructure for Cabruca Segmentation - Brazil Deployment

This Terraform configuration deploys the Cabruca Segmentation system optimized for serving the Northeast region of Brazil, using AWS São Paulo (sa-east-1) region with CloudFront CDN for low latency.

## Architecture Overview

The infrastructure is designed for high availability and optimized performance in Brazil:

- **Region**: AWS São Paulo (sa-east-1) - closest to Northeast Brazil
- **CDN**: CloudFront with edge locations across Brazil
- **Auto-scaling**: Dynamic scaling based on load
- **High Availability**: Multi-AZ deployment
- **Caching**: ElastiCache Redis for improved response times
- **Database**: RDS PostgreSQL for metadata storage
- **Monitoring**: CloudWatch with custom metrics

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** >= 1.0
3. **AWS CLI** configured
4. **Domain name** (optional, for custom domain)

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/dlgiant/cabruca-segmentation.git
cd cabruca-segmentation/terraform
```

### 2. Configure variables
```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 3. Initialize Terraform
```bash
terraform init
```

### 4. Plan deployment
```bash
terraform plan
```

### 5. Apply configuration
```bash
terraform apply
```

## Configuration

### Environment Variables

```hcl
# terraform.tfvars
project_name = "cabruca-segmentation"
environment  = "prod"
aws_region   = "sa-east-1"

# Domain (optional)
domain_name = "cabruca.agro.br"

# Instance types
instance_type_api       = "t3.large"      # API servers
instance_type_inference = "g4dn.xlarge"   # GPU inference
instance_type_processing = "m5.2xlarge"   # Batch processing

# Scaling
api_min_instances = 2
api_max_instances = 10
inference_min_instances = 1
inference_max_instances = 5

# Features
enable_gpu         = true
enable_cloudfront  = true
enable_elasticache = true
enable_rds        = true
```

## Infrastructure Components

### Network Architecture
- **VPC**: Custom VPC with public and private subnets
- **Availability Zones**: 3 AZs for high availability
- **NAT Gateways**: For outbound internet access from private subnets
- **Security Groups**: Restrictive security groups for each component

### Compute Resources
- **ECS Fargate**: For API and Streamlit services
- **ECS EC2**: For GPU-based inference (when enabled)
- **Auto-scaling**: Based on CPU/Memory utilization

### Storage
- **S3 Buckets**: 
  - Model storage with versioning
  - Data storage with encryption
- **EBS Volumes**: For container storage

### Database & Cache
- **RDS PostgreSQL**: Multi-AZ deployment for metadata
- **ElastiCache Redis**: For API response caching

### Load Balancing & CDN
- **Application Load Balancer**: For distributing traffic
- **CloudFront CDN**: For low-latency access across Brazil

### Monitoring & Logging
- **CloudWatch**: Metrics and alarms
- **CloudWatch Logs**: Centralized logging
- **SNS**: Alert notifications
- **Budget Alerts**: Cost monitoring

## Deployment Environments

### Development
```bash
terraform workspace new dev
terraform plan -var="environment=dev"
terraform apply -var="environment=dev"
```

Features:
- Minimal instances (1 API, 0 GPU)
- No RDS/ElastiCache
- Auto-shutdown after hours

### Staging
```bash
terraform workspace new staging
terraform plan -var="environment=staging"
terraform apply -var="environment=staging"
```

Features:
- Medium capacity (2 API, 1 GPU)
- Full feature set
- Lower retention periods

### Production
```bash
terraform workspace new prod
terraform plan -var="environment=prod"
terraform apply -var="environment=prod"
```

Features:
- High availability (3+ API, 2+ GPU)
- Full monitoring and backups
- 30-day retention
- Multi-region backups (optional)

## Cost Optimization

### Estimated Monthly Costs (USD)

| Component | Dev | Staging | Production |
|-----------|-----|---------|------------|
| EC2/Fargate | $50 | $200 | $500 |
| GPU Instances | $0 | $400 | $1,200 |
| Load Balancer | $25 | $25 | $25 |
| RDS | $0 | $100 | $300 |
| ElastiCache | $0 | $50 | $150 |
| S3 | $10 | $30 | $100 |
| CloudFront | $0 | $50 | $200 |
| Data Transfer | $20 | $100 | $500 |
| **Total** | **$105** | **$955** | **$2,975** |

### Cost Saving Options

1. **Spot Instances**: Enable for non-critical workloads
```hcl
cost_optimization = {
  use_spot_instances = true
  spot_max_price    = "0.50"
}
```

2. **Reserved Instances**: For production workloads
```hcl
cost_optimization = {
  use_reserved_capacity = true
}
```

3. **Auto-shutdown**: For development environments
```hcl
cost_optimization = {
  auto_shutdown_dev = true
  shutdown_schedule = "0 22 * * MON-FRI"
}
```

## Monitoring

### CloudWatch Dashboard
Access the dashboard at: AWS Console > CloudWatch > Dashboards > cabruca-segmentation-dashboard

### Key Metrics
- API Response Time (target: <2s)
- API Error Rate (target: <1%)
- GPU Utilization (target: 60-80%)
- Cache Hit Rate (target: >80%)
- Database Connections (target: <80)

### Alerts
Configured alerts will be sent to the email specified in `monitoring_configuration.alarm_email`

## Security

### Best Practices Implemented
- ✅ VPC with private subnets
- ✅ Security groups with least privilege
- ✅ Encryption at rest (S3, RDS, EBS)
- ✅ Encryption in transit (TLS/SSL)
- ✅ IAM roles with minimal permissions
- ✅ Secrets managed by AWS Secrets Manager
- ✅ CloudTrail for audit logging
- ✅ LGPD compliance considerations

### Additional Security (Optional)
```bash
# Enable AWS WAF
terraform apply -var="enable_waf=true"

# Enable VPN access
terraform apply -var="network_configuration.enable_vpn=true"

# Restrict IP access
terraform apply -var='network_configuration.allowed_ip_ranges=["200.1.2.3/32"]'
```

## Backup and Disaster Recovery

### Automated Backups
- **RDS**: Daily backups with 30-day retention
- **S3**: Versioning enabled on model bucket
- **EBS**: Daily snapshots

### Manual Backup
```bash
# Backup RDS
aws rds create-db-snapshot \
  --db-instance-identifier cabruca-segmentation-prod-db \
  --db-snapshot-identifier cabruca-backup-$(date +%Y%m%d)

# Backup S3
aws s3 sync s3://cabruca-segmentation-models-brasil \
  s3://cabruca-backup-brasil/models/$(date +%Y%m%d)/
```

### Disaster Recovery
```bash
# Restore from snapshot
terraform apply -var="restore_from_snapshot=true" \
  -var="snapshot_identifier=cabruca-backup-20240120"
```

## Maintenance

### Update Infrastructure
```bash
# Update to latest module versions
terraform get -update

# Plan changes
terraform plan

# Apply updates
terraform apply
```

### Scale Resources
```bash
# Scale up for high load
terraform apply -var="api_min_instances=5" -var="api_max_instances=20"

# Scale down after peak
terraform apply -var="api_min_instances=2" -var="api_max_instances=10"
```

### Destroy Infrastructure
```bash
# Destroy specific environment
terraform workspace select dev
terraform destroy

# Destroy all resources (CAUTION!)
terraform destroy -auto-approve
```

## Troubleshooting

### Common Issues

1. **Insufficient capacity**
```bash
# Increase instance limits
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --desired-value 100
```

2. **High latency in Northeast Brazil**
```bash
# Ensure CloudFront is enabled
terraform apply -var="enable_cloudfront=true"
```

3. **Database connection issues**
```bash
# Check security groups
aws ec2 describe-security-groups --group-ids sg-xxxxx

# Test connection
psql -h cabruca-db.xxxxx.sa-east-1.rds.amazonaws.com -U cabruca_admin -d cabruca
```

### Logs

View logs in CloudWatch:
```bash
# API logs
aws logs tail /ecs/cabruca-segmentation/api --follow

# Inference logs
aws logs tail /ecs/cabruca-segmentation/inference --follow

# Error logs
aws logs filter-log-events \
  --log-group-name /ecs/cabruca-segmentation/api \
  --filter-pattern ERROR
```

## CI/CD Integration

### GitHub Actions
The repository includes GitHub Actions workflows for automated deployment:

```yaml
# .github/workflows/deploy.yml
- Test code
- Build Docker images
- Push to ECR
- Deploy to ECS
- Run smoke tests
```

### Manual Deployment
```bash
# Build and push Docker images
docker build -t cabruca-api -f docker/Dockerfile.api .
docker tag cabruca-api:latest $ECR_REPO:api-latest
docker push $ECR_REPO:api-latest

# Update ECS service
aws ecs update-service \
  --cluster cabruca-segmentation-cluster \
  --service cabruca-segmentation-api \
  --force-new-deployment
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/dlgiant/cabruca-segmentation/issues
- Email: tech@cabruca.agro.br

## License

MIT License - See LICENSE file for details

## Contributors

- Ricardo Nunes - Infrastructure and DevOps
- Cabruca Team - ML and Application Development

---

**Note**: This infrastructure is optimized for serving the Northeast region of Brazil. For deployments in other regions, adjust the `aws_region` and CloudFront configuration accordingly.