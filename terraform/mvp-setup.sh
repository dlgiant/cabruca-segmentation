#!/bin/bash
# MVP Setup Script for Cabruca Segmentation
# Minimal cost deployment (~$90/month)

set -e

echo "🚀 Cabruca Segmentation MVP Setup"
echo "================================="
echo "Estimated monthly cost: ~\$90-100"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v terraform >/dev/null 2>&1 || { echo "❌ Terraform is required but not installed. Aborting." >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI is required but not installed. Aborting." >&2; exit 1; }

# Check AWS credentials
echo "Checking AWS credentials..."
aws sts get-caller-identity >/dev/null 2>&1 || { echo "❌ AWS credentials not configured. Run 'aws configure' first." >&2; exit 1; }

# Initialize Terraform
echo ""
echo "📦 Initializing Terraform..."
terraform init

# Create S3 bucket for state (if not exists)
BUCKET_NAME="cabruca-terraform-state-mvp-$(aws sts get-caller-identity --query Account --output text)"
REGION="sa-east-1"

echo ""
echo "📁 Creating state bucket: $BUCKET_NAME"
aws s3api create-bucket \
    --bucket $BUCKET_NAME \
    --region $REGION \
    --create-bucket-configuration LocationConstraint=$REGION 2>/dev/null || echo "Bucket already exists"

# Enable versioning on state bucket
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Update backend configuration
cat > backend.tf <<EOF
terraform {
  backend "s3" {
    bucket = "$BUCKET_NAME"
    key    = "mvp/terraform.tfstate"
    region = "$REGION"
  }
}
EOF

# Re-initialize with backend
terraform init -reconfigure

# Plan deployment
echo ""
echo "📋 Planning MVP deployment..."
terraform plan -var-file=mvp.tfvars -out=mvp.plan

# Show cost estimate
echo ""
echo "💰 Estimated Monthly Costs:"
echo "  - Fargate containers: \$10-15"
echo "  - Load Balancer: \$25"
echo "  - NAT Gateway: \$45"
echo "  - S3 Storage: \$5"
echo "  - CloudWatch: \$5"
echo "  - Total: ~\$90-100/month"
echo ""
echo "💡 Cost Optimization Tips:"
echo "  - Stop services when not in use"
echo "  - Use spot instances for testing"
echo "  - Monitor CloudWatch costs"
echo ""

# Confirm deployment
read -p "Do you want to deploy the MVP? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "🏗️ Deploying MVP infrastructure..."
    terraform apply mvp.plan
    
    # Get outputs
    echo ""
    echo "✅ MVP Deployment Complete!"
    echo ""
    echo "📊 Deployment Information:"
    terraform output -json deployment_info | jq -r
    
    # Create helper scripts
    echo ""
    echo "📝 Creating helper scripts..."
    
    # Start script
    cat > start-mvp.sh <<'SCRIPT'
#!/bin/bash
echo "Starting MVP services..."
aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-api --desired-count 1
aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-streamlit --desired-count 1
echo "Services started. Wait 2-3 minutes for containers to be ready."
SCRIPT
    chmod +x start-mvp.sh
    
    # Stop script
    cat > stop-mvp.sh <<'SCRIPT'
#!/bin/bash
echo "Stopping MVP services to save costs..."
aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-api --desired-count 0
aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-streamlit --desired-count 0
echo "Services stopped. Run ./start-mvp.sh to restart."
SCRIPT
    chmod +x stop-mvp.sh
    
    # Status script
    cat > status-mvp.sh <<'SCRIPT'
#!/bin/bash
echo "MVP Service Status:"
aws ecs list-services --cluster cabruca-mvp-cluster | jq -r '.serviceArns[]' | while read service; do
    aws ecs describe-services --cluster cabruca-mvp-cluster --services $service \
        --query 'services[0].{Service:serviceName,Status:status,Running:runningCount,Desired:desiredCount}' \
        --output table
done
SCRIPT
    chmod +x status-mvp.sh
    
    # Test script
    cat > test-mvp.sh <<'SCRIPT'
#!/bin/bash
ALB_URL=$(terraform output -raw load_balancer_url | sed 's/https:/http:/g')
echo "Testing MVP endpoints..."
echo ""
echo "1. Health Check:"
curl -s $ALB_URL/health | jq . || echo "API not ready yet"
echo ""
echo "2. API Docs:"
echo "   Open in browser: $ALB_URL/docs"
echo ""
echo "3. Dashboard:"
echo "   Open in browser: $ALB_URL/dashboard"
SCRIPT
    chmod +x test-mvp.sh
    
    echo ""
    echo "🎉 MVP Setup Complete!"
    echo ""
    echo "📌 Important URLs:"
    ALB_URL=$(terraform output -raw load_balancer_url | sed 's/https:/http:/g')
    echo "  - API: $ALB_URL/api"
    echo "  - Dashboard: $ALB_URL/dashboard"
    echo "  - Health: $ALB_URL/health"
    echo ""
    echo "🛠️ Helper Scripts Created:"
    echo "  - ./start-mvp.sh  - Start services"
    echo "  - ./stop-mvp.sh   - Stop services (save costs)"
    echo "  - ./status-mvp.sh - Check service status"
    echo "  - ./test-mvp.sh   - Test endpoints"
    echo ""
    echo "💡 Next Steps:"
    echo "  1. Wait 3-5 minutes for services to start"
    echo "  2. Run ./test-mvp.sh to verify deployment"
    echo "  3. Upload your model to S3"
    echo "  4. Access the dashboard"
    echo ""
    echo "⚠️  Remember to run ./stop-mvp.sh when not using to save costs!"
else
    echo "Deployment cancelled."
    rm -f mvp.plan
fi