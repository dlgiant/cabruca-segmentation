terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "github_token_secret_name" {
  description = "AWS Secrets Manager secret name for GitHub token"
  type        = string
  default     = "github-token"
}

variable "github_repo" {
  description = "GitHub repository in format org/repo"
  type        = string
}

variable "anthropic_api_key_secret" {
  description = "AWS Secrets Manager secret name for Anthropic API key"
  type        = string
  default     = "anthropic-api-key"
}

variable "event_bus_name" {
  description = "EventBridge bus name"
  type        = string
  default     = "default"
}

variable "s3_bucket_prefix" {
  description = "S3 bucket prefix for storing artifacts"
  type        = string
  default     = "engineer-agent-artifacts"
}

# Data sources
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# DynamoDB table for tracking tasks
resource "aws_dynamodb_table" "engineer_tasks" {
  name           = "engineer-tasks-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "task_id"
  
  attribute {
    name = "task_id"
    type = "S"
  }
  
  attribute {
    name = "status"
    type = "S"
  }
  
  attribute {
    name = "created_at"
    type = "S"
  }
  
  global_secondary_index {
    name            = "status-created-index"
    hash_key        = "status"
    range_key       = "created_at"
    projection_type = "ALL"
  }
  
  ttl {
    attribute_name = "expiry"
    enabled        = true
  }
  
  tags = {
    Name        = "engineer-tasks"
    Environment = var.environment
  }
}

# S3 bucket for storing artifacts
resource "aws_s3_bucket" "engineer_artifacts" {
  bucket = "${var.s3_bucket_prefix}-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name        = "engineer-agent-artifacts"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "engineer_artifacts" {
  bucket = aws_s3_bucket.engineer_artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "engineer_artifacts" {
  bucket = aws_s3_bucket.engineer_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "engineer_artifacts" {
  bucket = aws_s3_bucket.engineer_artifacts.id
  
  rule {
    id     = "cleanup-old-artifacts"
    status = "Enabled"
    
    expiration {
      days = 30
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# Lambda execution role
resource "aws_iam_role" "engineer_agent_role" {
  name = "engineer-agent-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "engineer-agent-role"
    Environment = var.environment
  }
}

# IAM policy for Engineer Agent
resource "aws_iam_policy" "engineer_agent_policy" {
  name        = "engineer-agent-policy-${var.environment}"
  description = "Policy for Engineer Agent Lambda function"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Basic Lambda execution permissions
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      },
      # DynamoDB permissions for task table
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:DeleteItem"
        ]
        Resource = [
          aws_dynamodb_table.engineer_tasks.arn,
          "${aws_dynamodb_table.engineer_tasks.arn}/index/*"
        ]
      },
      # S3 permissions for artifacts
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.engineer_artifacts.arn,
          "${aws_s3_bucket.engineer_artifacts.arn}/*"
        ]
      },
      # EventBridge permissions
      {
        Effect = "Allow"
        Action = [
          "events:PutEvents"
        ]
        Resource = [
          "arn:aws:events:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:event-bus/${var.event_bus_name}"
        ]
      },
      # Secrets Manager permissions
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.github_token_secret_name}*",
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.anthropic_api_key_secret}*"
        ]
      },
      # CodeCommit permissions (optional, for AWS repos)
      {
        Effect = "Allow"
        Action = [
          "codecommit:GetRepository",
          "codecommit:GitPull",
          "codecommit:GitPush",
          "codecommit:CreateBranch",
          "codecommit:CreatePullRequest"
        ]
        Resource = "arn:aws:codecommit:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "engineer_agent_policy_attachment" {
  role       = aws_iam_role.engineer_agent_role.name
  policy_arn = aws_iam_policy.engineer_agent_policy.arn
}

# Lambda Layer for dependencies
resource "aws_lambda_layer_version" "engineer_agent_dependencies" {
  filename            = "../lambda_layer.zip"
  layer_name          = "engineer-agent-dependencies-${var.environment}"
  compatible_runtimes = ["python3.11"]
  description         = "Dependencies for Engineer Agent Lambda (LangChain, PyGithub, etc.)"

  lifecycle {
    create_before_destroy = true
  }
}

# Lambda function
resource "aws_lambda_function" "engineer_agent" {
  filename         = "../lambda_deployment.zip"
  function_name    = "engineer-agent-${var.environment}"
  role            = aws_iam_role.engineer_agent_role.arn
  handler         = "lambda_function.lambda_handler"
  source_code_hash = filebase64sha256("../lambda_deployment.zip")
  runtime         = "python3.11"
  memory_size     = 1024  # 1GB as specified
  timeout         = 600   # 10 minutes as specified

  environment {
    variables = {
      GITHUB_TOKEN_SECRET_NAME = var.github_token_secret_name
      GITHUB_REPO             = var.github_repo
      ANTHROPIC_API_KEY_SECRET = var.anthropic_api_key_secret
      S3_BUCKET               = aws_s3_bucket.engineer_artifacts.id
      TASK_TABLE_NAME         = aws_dynamodb_table.engineer_tasks.name
      EVENT_BUS_NAME          = var.event_bus_name
      ENVIRONMENT             = var.environment
      MAX_ITERATIONS          = "10"
    }
  }

  layers = [aws_lambda_layer_version.engineer_agent_dependencies.arn]

  tags = {
    Name        = "engineer-agent"
    Environment = var.environment
    Purpose     = "Autonomous implementation of solutions"
  }
}

# EventBridge rules to trigger Engineer Agent from Manager Agent events
resource "aws_cloudwatch_event_rule" "system_issue_trigger" {
  name        = "engineer-agent-system-issue-${var.environment}"
  description = "Trigger Engineer Agent on system issues from Manager Agent"

  event_pattern = jsonencode({
    source      = ["manager.agent"]
    detail-type = [
      "SystemIssue.high_error_rate",
      "SystemIssue.high_latency",
      "SystemIssue.negative_feedback",
      "SystemIssue.cost_anomaly",
      "SystemIssue.performance_degradation"
    ]
  })

  tags = {
    Name        = "engineer-agent-issue-trigger"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_event_rule" "opportunity_trigger" {
  name        = "engineer-agent-opportunity-${var.environment}"
  description = "Trigger Engineer Agent on improvement opportunities"

  event_pattern = jsonencode({
    source      = ["manager.agent"]
    detail-type = ["Opportunity.AutoRemediation"]
  })

  tags = {
    Name        = "engineer-agent-opportunity-trigger"
    Environment = var.environment
  }
}

# EventBridge targets
resource "aws_cloudwatch_event_target" "engineer_agent_issue_target" {
  rule      = aws_cloudwatch_event_rule.system_issue_trigger.name
  target_id = "EngineerAgentIssueTarget"
  arn       = aws_lambda_function.engineer_agent.arn
}

resource "aws_cloudwatch_event_target" "engineer_agent_opportunity_target" {
  rule      = aws_cloudwatch_event_rule.opportunity_trigger.name
  target_id = "EngineerAgentOpportunityTarget"
  arn       = aws_lambda_function.engineer_agent.arn
}

# Permissions for EventBridge to invoke Lambda
resource "aws_lambda_permission" "allow_eventbridge_issue" {
  statement_id  = "AllowExecutionFromEventBridgeIssue"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.engineer_agent.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.system_issue_trigger.arn
}

resource "aws_lambda_permission" "allow_eventbridge_opportunity" {
  statement_id  = "AllowExecutionFromEventBridgeOpportunity"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.engineer_agent.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.opportunity_trigger.arn
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "engineer_agent_logs" {
  name              = "/aws/lambda/${aws_lambda_function.engineer_agent.function_name}"
  retention_in_days = 14  # Keep logs for 14 days

  tags = {
    Name        = "engineer-agent-logs"
    Environment = var.environment
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "engineer_agent_errors" {
  alarm_name          = "engineer-agent-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "Errors"
  namespace          = "AWS/Lambda"
  period             = "300"
  statistic          = "Sum"
  threshold          = "10"
  alarm_description  = "This metric monitors Engineer Agent Lambda errors"

  dimensions = {
    FunctionName = aws_lambda_function.engineer_agent.function_name
  }

  tags = {
    Name        = "engineer-agent-error-alarm"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "engineer_agent_duration" {
  alarm_name          = "engineer-agent-duration-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "Duration"
  namespace          = "AWS/Lambda"
  period             = "300"
  statistic          = "Average"
  threshold          = "300000"  # 5 minutes in milliseconds
  alarm_description  = "This metric monitors Engineer Agent Lambda duration"

  dimensions = {
    FunctionName = aws_lambda_function.engineer_agent.function_name
  }

  tags = {
    Name        = "engineer-agent-duration-alarm"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "engineer_agent_throttles" {
  alarm_name          = "engineer-agent-throttles-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name        = "Throttles"
  namespace          = "AWS/Lambda"
  period             = "300"
  statistic          = "Sum"
  threshold          = "5"
  alarm_description  = "This metric monitors Engineer Agent Lambda throttles"

  dimensions = {
    FunctionName = aws_lambda_function.engineer_agent.function_name
  }

  tags = {
    Name        = "engineer-agent-throttle-alarm"
    Environment = var.environment
  }
}

# Outputs
output "lambda_function_arn" {
  description = "ARN of the Engineer Agent Lambda function"
  value       = aws_lambda_function.engineer_agent.arn
}

output "lambda_function_name" {
  description = "Name of the Engineer Agent Lambda function"
  value       = aws_lambda_function.engineer_agent.function_name
}

output "task_table_name" {
  description = "DynamoDB table name for tasks"
  value       = aws_dynamodb_table.engineer_tasks.name
}

output "s3_bucket_name" {
  description = "S3 bucket name for artifacts"
  value       = aws_s3_bucket.engineer_artifacts.id
}

output "eventbridge_issue_rule_arn" {
  description = "ARN of the EventBridge issue trigger rule"
  value       = aws_cloudwatch_event_rule.system_issue_trigger.arn
}

output "eventbridge_opportunity_rule_arn" {
  description = "ARN of the EventBridge opportunity trigger rule"
  value       = aws_cloudwatch_event_rule.opportunity_trigger.arn
}

output "log_group_name" {
  description = "CloudWatch Log Group name"
  value       = aws_cloudwatch_log_group.engineer_agent_logs.name
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost for Engineer Agent"
  value       = "$10 (based on ~100 invocations/month @ 1GB memory, 30-60 second average duration, plus DynamoDB and S3 storage)"
}
