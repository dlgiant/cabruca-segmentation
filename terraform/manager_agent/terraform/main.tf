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

variable "anthropic_api_key" {
  description = "Anthropic API key for Claude access"
  type        = string
  sensitive   = true
}

variable "feedback_table_name" {
  description = "DynamoDB table name for user feedback"
  type        = string
  default     = "user-feedback"
}

variable "event_bus_name" {
  description = "EventBridge bus name"
  type        = string
  default     = "default"
}

# Data sources
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# Lambda execution role
resource "aws_iam_role" "manager_agent_role" {
  name = "manager-agent-lambda-role-${var.environment}"

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
    Name        = "manager-agent-role"
    Environment = var.environment
  }
}

# IAM policy for Manager Agent
resource "aws_iam_policy" "manager_agent_policy" {
  name        = "manager-agent-policy-${var.environment}"
  description = "Policy for Manager Agent Lambda function"

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
      # CloudWatch metrics read permissions
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetMetricData",
          "cloudwatch:ListMetrics",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "*"
      },
      # DynamoDB read permissions for feedback table
      {
        Effect = "Allow"
        Action = [
          "dynamodb:Scan",
          "dynamodb:Query",
          "dynamodb:GetItem",
          "dynamodb:DescribeTable"
        ]
        Resource = [
          "arn:aws:dynamodb:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/${var.feedback_table_name}",
          "arn:aws:dynamodb:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/${var.feedback_table_name}/index/*"
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
      # Cost Explorer permissions
      {
        Effect = "Allow"
        Action = [
          "ce:GetCostAndUsage",
          "ce:GetCostForecast",
          "ce:GetAnomalies",
          "ce:GetAnomalySubscriptions"
        ]
        Resource = "*"
      },
      # Lambda function list permissions (for monitoring)
      {
        Effect = "Allow"
        Action = [
          "lambda:ListFunctions",
          "lambda:GetFunction",
          "lambda:GetFunctionConfiguration"
        ]
        Resource = "*"
      },
      # API Gateway read permissions
      {
        Effect = "Allow"
        Action = [
          "apigateway:GET"
        ]
        Resource = "arn:aws:apigateway:${data.aws_region.current.name}::/restapis/*"
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "manager_agent_policy_attachment" {
  role       = aws_iam_role.manager_agent_role.name
  policy_arn = aws_iam_policy.manager_agent_policy.arn
}

# Lambda Layer for dependencies (optional, for production use)
resource "aws_lambda_layer_version" "manager_agent_dependencies" {
  filename            = "../lambda_layer.zip"
  layer_name          = "manager-agent-dependencies-${var.environment}"
  compatible_runtimes = ["python3.11"]
  description         = "Dependencies for Manager Agent Lambda"

  lifecycle {
    create_before_destroy = true
  }
}

# Lambda function
resource "aws_lambda_function" "manager_agent" {
  filename         = "../lambda_deployment.zip"
  function_name    = "manager-agent-${var.environment}"
  role             = aws_iam_role.manager_agent_role.arn
  handler          = "lambda_function.lambda_handler"
  source_code_hash = filebase64sha256("../lambda_deployment.zip")
  runtime          = "python3.11"
  memory_size      = 512
  timeout          = 300 # 5 minutes

  environment {
    variables = {
      FEEDBACK_TABLE_NAME = var.feedback_table_name
      EVENT_BUS_NAME      = var.event_bus_name
      ANTHROPIC_API_KEY   = var.anthropic_api_key
      ENVIRONMENT         = var.environment
    }
  }

  layers = [aws_lambda_layer_version.manager_agent_dependencies.arn]

  tags = {
    Name        = "manager-agent"
    Environment = var.environment
    Purpose     = "System monitoring and intelligent decision making"
  }
}

# EventBridge rule for scheduling (every 30 minutes)
resource "aws_cloudwatch_event_rule" "manager_agent_schedule" {
  name                = "manager-agent-schedule-${var.environment}"
  description         = "Trigger Manager Agent every 30 minutes"
  schedule_expression = "rate(30 minutes)"

  tags = {
    Name        = "manager-agent-schedule"
    Environment = var.environment
  }
}

# EventBridge target
resource "aws_cloudwatch_event_target" "manager_agent_target" {
  rule      = aws_cloudwatch_event_rule.manager_agent_schedule.name
  target_id = "ManagerAgentLambdaTarget"
  arn       = aws_lambda_function.manager_agent.arn

  input = jsonencode({
    source = "scheduled"
    detail = {
      trigger = "30-minute-interval"
    }
  })
}

# Permission for EventBridge to invoke Lambda
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.manager_agent.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.manager_agent_schedule.arn
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "manager_agent_logs" {
  name              = "/aws/lambda/${aws_lambda_function.manager_agent.function_name}"
  retention_in_days = 7

  tags = {
    Name        = "manager-agent-logs"
    Environment = var.environment
  }
}

# CloudWatch Alarm for Lambda errors
resource "aws_cloudwatch_metric_alarm" "manager_agent_errors" {
  alarm_name          = "manager-agent-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors Manager Agent Lambda errors"

  dimensions = {
    FunctionName = aws_lambda_function.manager_agent.function_name
  }

  tags = {
    Name        = "manager-agent-error-alarm"
    Environment = var.environment
  }
}

# Outputs
output "lambda_function_arn" {
  description = "ARN of the Manager Agent Lambda function"
  value       = aws_lambda_function.manager_agent.arn
}

output "lambda_function_name" {
  description = "Name of the Manager Agent Lambda function"
  value       = aws_lambda_function.manager_agent.function_name
}

output "eventbridge_rule_arn" {
  description = "ARN of the EventBridge scheduling rule"
  value       = aws_cloudwatch_event_rule.manager_agent_schedule.arn
}

output "log_group_name" {
  description = "CloudWatch Log Group name"
  value       = aws_cloudwatch_log_group.manager_agent_logs.name
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost for Manager Agent"
  value       = "$5 (based on 1440 invocations/month @ 512MB memory, 5-10 second average duration)"
}
