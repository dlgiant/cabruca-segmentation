# Outputs for MVP deployment

output "deployment_info" {
  description = "MVP Deployment Information"
  value = {
    api_url      = "http://${aws_lb.main.dns_name}/api"
    dashboard    = "http://${aws_lb.main.dns_name}/dashboard"
    health_check = "http://${aws_lb.main.dns_name}/health"
    region       = var.aws_region
    environment  = var.environment
  }
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost for MVP"
  value = {
    fargate     = "$10-15 (256 CPU, 512 MB per container)"
    alb         = "$25 (Application Load Balancer)"
    nat_gateway = "$45 (Single NAT Gateway)"
    s3          = "$5 (Storage and requests)"
    cloudwatch  = "$5 (Logs and metrics)"
    total       = "~$90-100/month"
    note        = "Actual costs may vary based on usage"
  }
}

output "cost_savings" {
  description = "Cost savings from production configuration"
  value = {
    gpu_instances = "$400/month saved (No GPU)"
    rds_database  = "$100/month saved (No RDS)"
    elasticache   = "$50/month saved (No Redis)"
    cloudfront    = "$50/month saved (No CDN)"
    multi_nat     = "$90/month saved (Single NAT)"
    total_savings = "~$690/month"
  }
}

output "quick_start" {
  description = "Quick start commands"
  value = {
    deploy    = "terraform apply -var-file=mvp.tfvars"
    test_api  = "curl ${aws_lb.main.dns_name}/health"
    view_logs = "aws logs tail /ecs/cabruca-mvp/api --follow"
    ssh_debug = "aws ecs execute-command --cluster cabruca-mvp-cluster --task <task-id> --container api --interactive --command /bin/sh"
  }
}

output "scaling_commands" {
  description = "Commands to scale the MVP"
  value = {
    scale_up   = "aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-api --desired-count 2"
    scale_down = "aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-api --desired-count 1"
    stop_all   = "aws ecs update-service --cluster cabruca-mvp-cluster --service cabruca-mvp-api --desired-count 0"
  }
}

output "upgrade_path" {
  description = "Path to upgrade from MVP to production"
  value = {
    step1 = "Enable RDS: terraform apply -var='enable_rds=true'"
    step2 = "Enable Redis: terraform apply -var='enable_elasticache=true'"
    step3 = "Enable CDN: terraform apply -var='enable_cloudfront=true'"
    step4 = "Add GPU: terraform apply -var='enable_gpu=true'"
    step5 = "Scale up: terraform apply -var='api_min_instances=2'"
  }
}