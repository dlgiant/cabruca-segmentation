# Cabruca Segmentation - Terraform Deployment Guide

## 📋 Overview

This guide provides instructions for deploying the Cabruca Segmentation infrastructure on AWS, optimized for Brazil's Northeast region with ML pipeline capabilities for training and annotating segmented trees.

## 💰 Cost Breakdown (MVP Configuration)

### Monthly Estimated Costs (~$90-100)
- **Fargate Containers**: $10-15 (256 CPU, 512MB per container)
- **Application Load Balancer**: $25
- **NAT Gateway**: $45 (single NAT for MVP)
- **S3 Storage**: $5 (models and data)
- **CloudWatch**: $5 (logs and metrics)

### Optional ML Pipeline Costs (On-Demand)
- **SageMaker Notebook**: $0.05/hour when running (ml.t3.medium)
- **Training Tasks**: $0.02/hour per task (1 vCPU, 2GB RAM)
- **Annotation Service**: $0.01/hour when running (0.5 vCPU, 1GB RAM)
- **Lambda Batch Processing**: $0.0000166667/GB-second

## 🚀 Quick Start Deployment

### Prerequisites
```bash
# Install required tools
brew install terraform awscli jq

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: sa-east-1
# Default output format: json
```

### MVP Deployment (Lowest Cost)
```bash
# Navigate to terraform directory
cd terraform

# Initialize Terraform
terraform init -backend=false

# Review the MVP plan
terraform plan -var-file=mvp.tfvars

# Deploy MVP infrastructure
terraform apply -var-file=mvp.tfvars -auto-approve

# Get deployment URLs
terraform output deployment_info
```

### Using the Automated Setup Script
```bash
# Run the MVP setup script (includes all steps)
./mvp-setup.sh
```

## 🛠️ Infrastructure Components

### Core Services (Always Running)
1. **ECS Fargate Cluster**: Container orchestration
2. **Application Load Balancer**: HTTP/HTTPS routing
3. **VPC with Subnets**: Network isolation
4. **S3 Buckets**: Model and data storage

### ML Pipeline (On-Demand)
1. **SageMaker Notebook**: Interactive training environment
2. **ECS Training Tasks**: Batch training jobs
3. **Annotation Service**: Streamlit-based annotation tool
4. **Step Functions**: ML pipeline orchestration
5. **Lambda Functions**: Batch image processing

### Optional Services (Disabled for MVP)
- CloudFront CDN (save $50/month)
- RDS Database (save $100/month)
- ElastiCache Redis (save $50/month)
- GPU Instances (save $400/month)

## 📁 Configuration Files

### `mvp.tfvars` - MVP Settings
```hcl
# Minimal cost configuration
project_name = "cabruca-mvp"
environment  = "mvp"
aws_region   = "sa-east-1"

# Minimal instances
api_min_instances = 1
api_max_instances = 2

# ML Pipeline (on-demand)
enable_ml_pipeline = true
annotation_enabled = false  # Start manually
```

### `variables.tf` - Variable Definitions
Contains all configurable parameters with defaults optimized for Brazil deployment.

### `ml-pipeline.tf` - ML Infrastructure
Defines training, annotation, and batch processing resources.

## 🔧 Management Commands

### Start/Stop Services (Save Costs)
```bash
# Stop all services (save ~$2/day)
./stop-mvp.sh

# Start services when needed
./start-mvp.sh

# Check service status
./status-mvp.sh
```

### Scale Services
```bash
# Scale API up
aws ecs update-service \
  --cluster cabruca-mvp-cluster \
  --service cabruca-mvp-api \
  --desired-count 2

# Scale API down
aws ecs update-service \
  --cluster cabruca-mvp-cluster \
  --service cabruca-mvp-api \
  --desired-count 1
```

### ML Pipeline Operations

#### Start Training
```bash
# Run training task
aws ecs run-task \
  --cluster cabruca-mvp-cluster \
  --task-definition cabruca-mvp-training \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"
```

#### Start Annotation Service
```bash
# Enable annotation service
aws ecs update-service \
  --cluster cabruca-mvp-cluster \
  --service cabruca-mvp-annotation \
  --desired-count 1

# Access at: http://<ALB-URL>/annotation
```

#### Trigger Batch Processing
```bash
# Invoke Lambda for batch processing
aws lambda invoke \
  --function-name cabruca-mvp-batch-processor \
  --payload '{"batch_id": "test-001", "image_keys": ["image1.jpg", "image2.jpg"]}' \
  response.json
```

#### Start SageMaker Notebook
```bash
# Start notebook instance
aws sagemaker start-notebook-instance \
  --notebook-instance-name cabruca-mvp-ml-workspace

# Get notebook URL
aws sagemaker create-presigned-notebook-instance-url \
  --notebook-instance-name cabruca-mvp-ml-workspace
```

## 📊 Monitoring

### View Logs
```bash
# API logs
aws logs tail /ecs/cabruca-mvp/api --follow

# Training logs
aws logs tail /ecs/cabruca-mvp/training --follow

# Annotation logs
aws logs tail /ecs/cabruca-mvp/annotation --follow
```

### CloudWatch Metrics
```bash
# View ECS metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=cabruca-mvp-api \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

## 🔄 Upgrade Path

### From MVP to Production
```bash
# Step 1: Enable RDS database
terraform apply -var="enable_rds=true"

# Step 2: Enable Redis cache
terraform apply -var="enable_elasticache=true"

# Step 3: Enable CloudFront CDN
terraform apply -var="enable_cloudfront=true"

# Step 4: Add GPU support
terraform apply -var="enable_gpu=true"

# Step 5: Increase instances
terraform apply -var="api_min_instances=2" -var="api_max_instances=5"
```

## 🧹 Cleanup

### Destroy Infrastructure
```bash
# Remove all resources
terraform destroy -var-file=mvp.tfvars

# Confirm by typing 'yes'
```

### Clean S3 Buckets First
```bash
# Empty S3 buckets before destroy
aws s3 rm s3://cabruca-mvp-models --recursive
aws s3 rm s3://cabruca-mvp-data --recursive
aws s3 rm s3://cabruca-mvp-ml-data --recursive
```

## 🐛 Troubleshooting

### Common Issues

1. **Terraform State Lock**
```bash
terraform force-unlock <LOCK_ID>
```

2. **ECS Service Not Starting**
```bash
# Check task failures
aws ecs describe-tasks \
  --cluster cabruca-mvp-cluster \
  --tasks $(aws ecs list-tasks --cluster cabruca-mvp-cluster --query 'taskArns[0]' --output text)
```

3. **Lambda Function Errors**
```bash
# View Lambda logs
aws logs tail /aws/lambda/cabruca-mvp-batch-processor --follow
```

4. **Out of Memory**
```bash
# Increase task memory
terraform apply -var="training_memory=4096"
```

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Project Repository](https://github.com/dlgiant/cabruca-segmentation)

## 📞 Support

For issues or questions:
1. Check the [GitHub Issues](https://github.com/dlgiant/cabruca-segmentation/issues)
2. Review CloudWatch logs for errors
3. Contact the development team

## 🔒 Security Notes

- All services run in private subnets
- ALB handles public traffic
- S3 buckets are private by default
- IAM roles follow least privilege principle
- Enable AWS GuardDuty for threat detection
- Regular security updates via ECR image scanning

## 💡 Cost Optimization Tips

1. **Use Spot Instances**: Save 70% on Fargate costs
2. **Schedule Services**: Auto-stop during non-business hours
3. **Right-size Instances**: Monitor and adjust CPU/memory
4. **S3 Lifecycle Policies**: Archive old data to Glacier
5. **Reserved Capacity**: Commit to 1-year for 40% savings
6. **Use S3 Intelligent-Tiering**: Automatic cost optimization

---

Last Updated: January 2025
Version: 1.0.0