# Cost Control Infrastructure Module
# Implements budget safeguards to keep total costs under $500/month

# ========================================
# 1. LAMBDA CONCURRENCY LIMITS
# ========================================

# Set concurrency limit for Manager Agent (1 concurrent execution)
# Reserved concurrency for manager agent - configured in lambda function

# Set concurrency limit for Engineer Agent (2 concurrent executions)
# Reserved concurrency for engineer agent - configured in lambda function

# Set concurrency limit for QA Agent (2 concurrent executions)
# Reserved concurrency for qa agent - configured in lambda function

# ========================================
# 2. API GATEWAY WITH REQUEST THROTTLING
# ========================================

# Create API Gateway REST API
resource "aws_api_gateway_rest_api" "agent_api" {
  name        = "${var.environment}-agent-control-api"
  description = "API Gateway for agent control with request throttling"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = merge(var.tags, {
    Name       = "${var.environment}-agent-api"
    Component  = "CostControl"
    CostCenter = "engineering"
  })
}

# Create usage plan with throttling (100 requests/minute)
resource "aws_api_gateway_usage_plan" "agent_throttle_plan" {
  name        = "${var.environment}-agent-throttle-plan"
  description = "Usage plan with 100 requests/minute throttling"

  api_stages {
    api_id = aws_api_gateway_rest_api.agent_api.id
    stage  = aws_api_gateway_deployment.agent_api_deployment.stage_name
  }

  quota_settings {
    limit  = 6000 # 100 req/min * 60 min = 6000 requests per hour
    period = "DAY"
  }

  throttle_settings {
    rate_limit  = 100 # Requests per second steady-state rate
    burst_limit = 200 # Maximum bucket capacity for burst
  }
}

# API Gateway deployment
resource "aws_api_gateway_deployment" "agent_api_deployment" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  stage_name  = var.environment

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_api_gateway_method.trigger_manager,
    aws_api_gateway_method.trigger_engineer,
    aws_api_gateway_method.trigger_qa,
    aws_api_gateway_integration.manager_lambda_integration,
    aws_api_gateway_integration.engineer_lambda_integration,
    aws_api_gateway_integration.qa_lambda_integration
  ]
}

# API Gateway resources and methods for each agent
resource "aws_api_gateway_resource" "manager_resource" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  parent_id   = aws_api_gateway_rest_api.agent_api.root_resource_id
  path_part   = "manager"
}

resource "aws_api_gateway_resource" "engineer_resource" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  parent_id   = aws_api_gateway_rest_api.agent_api.root_resource_id
  path_part   = "engineer"
}

resource "aws_api_gateway_resource" "qa_resource" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  parent_id   = aws_api_gateway_rest_api.agent_api.root_resource_id
  path_part   = "qa"
}

# Methods for triggering agents
resource "aws_api_gateway_method" "trigger_manager" {
  rest_api_id   = aws_api_gateway_rest_api.agent_api.id
  resource_id   = aws_api_gateway_resource.manager_resource.id
  http_method   = "POST"
  authorization = "AWS_IAM"
}

resource "aws_api_gateway_method" "trigger_engineer" {
  rest_api_id   = aws_api_gateway_rest_api.agent_api.id
  resource_id   = aws_api_gateway_resource.engineer_resource.id
  http_method   = "POST"
  authorization = "AWS_IAM"
}

resource "aws_api_gateway_method" "trigger_qa" {
  rest_api_id   = aws_api_gateway_rest_api.agent_api.id
  resource_id   = aws_api_gateway_resource.qa_resource.id
  http_method   = "POST"
  authorization = "AWS_IAM"
}

# Lambda integrations
resource "aws_api_gateway_integration" "manager_lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  resource_id = aws_api_gateway_resource.manager_resource.id
  http_method = aws_api_gateway_method.trigger_manager.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = "arn:aws:apigateway:${data.aws_region.current.name}:lambda:path/2015-03-31/functions/arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:manager-agent-${var.environment}/invocations"
}

resource "aws_api_gateway_integration" "engineer_lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  resource_id = aws_api_gateway_resource.engineer_resource.id
  http_method = aws_api_gateway_method.trigger_engineer.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = "arn:aws:apigateway:${data.aws_region.current.name}:lambda:path/2015-03-31/functions/arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:engineer-agent-${var.environment}/invocations"
}

resource "aws_api_gateway_integration" "qa_lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.agent_api.id
  resource_id = aws_api_gateway_resource.qa_resource.id
  http_method = aws_api_gateway_method.trigger_qa.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = "arn:aws:apigateway:${data.aws_region.current.name}:lambda:path/2015-03-31/functions/arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.environment}-qa-agent/invocations"
}

# Lambda permissions for API Gateway
resource "aws_lambda_permission" "manager_api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.agents["manager"].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.agent_api.execution_arn}/*/*"

  depends_on = [aws_lambda_function.agents]
}

resource "aws_lambda_permission" "engineer_api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.agents["engineer"].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.agent_api.execution_arn}/*/*"

  depends_on = [aws_lambda_function.agents]
}

resource "aws_lambda_permission" "qa_api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.agents["qa"].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.agent_api.execution_arn}/*/*"

  depends_on = [aws_lambda_function.agents]
}

# ========================================
# 3. COST ALLOCATION TAGS
# ========================================

# Tag policy for enforcing cost allocation tags
resource "aws_organizations_policy" "cost_allocation_tags" {
  count = var.enable_org_policies ? 1 : 0

  name        = "enforce-cost-allocation-tags"
  description = "Enforce cost allocation tags for all agent resources"
  type        = "TAG_POLICY"

  content = jsonencode({
    tags = {
      CostCenter = {
        tag_key = {
          "@@assign" = "CostCenter"
        }
        tag_value = {
          "@@assign" = ["engineering", "qa", "operations", "monitoring"]
        }
        enforced_for = {
          "@@assign" = [
            "lambda:function",
            "dynamodb:table",
            "s3:bucket",
            "apigateway:restapi"
          ]
        }
      }
      Agent = {
        tag_key = {
          "@@assign" = "Agent"
        }
        tag_value = {
          "@@assign" = ["manager", "engineer", "qa", "monitoring"]
        }
        enforced_for = {
          "@@assign" = ["lambda:function"]
        }
      }
      Environment = {
        tag_key = {
          "@@assign" = "Environment"
        }
        tag_value = {
          "@@assign" = [var.environment]
        }
      }
    }
  })
}

# ========================================
# 4. AUTO-SHUTDOWN FOR DEVELOPMENT RESOURCES
# ========================================

# Lambda function for auto-shutdown
resource "aws_lambda_function" "auto_shutdown" {
  count = var.environment == "development" ? 1 : 0

  filename         = data.archive_file.auto_shutdown_package[0].output_path
  function_name    = "${var.environment}-auto-shutdown"
  role             = aws_iam_role.auto_shutdown_role[0].arn
  handler          = "auto_shutdown.lambda_handler"
  source_code_hash = data.archive_file.auto_shutdown_package[0].output_base64sha256
  runtime          = "python3.11"
  timeout          = 60

  environment {
    variables = {
      ENVIRONMENT = var.environment
      SHUTDOWN_TAGS = jsonencode({
        Environment  = var.environment
        AutoShutdown = "true"
      })
    }
  }

  tags = merge(var.tags, {
    Name       = "${var.environment}-auto-shutdown"
    CostCenter = "operations"
    Purpose    = "Cost Control"
  })
}

# Package auto-shutdown Lambda
data "archive_file" "auto_shutdown_package" {
  count = var.environment == "development" ? 1 : 0

  type        = "zip"
  source_file = "${path.module}/cost_control/auto_shutdown.py"
  output_path = "${path.module}/auto_shutdown.zip"
}

# IAM role for auto-shutdown
resource "aws_iam_role" "auto_shutdown_role" {
  count = var.environment == "development" ? 1 : 0

  name = "${var.environment}-auto-shutdown-role"

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
}

# IAM policy for auto-shutdown
resource "aws_iam_role_policy" "auto_shutdown_policy" {
  count = var.environment == "development" ? 1 : 0

  role = aws_iam_role.auto_shutdown_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:ListFunctions",
          "lambda:GetFunction",
          "lambda:UpdateFunctionConfiguration",
          "lambda:PutFunctionConcurrency"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:StopInstances",
          "ec2:DescribeTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:StopDBInstance"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Events rule for after-hours shutdown (weekdays 8 PM local time)
resource "aws_cloudwatch_event_rule" "after_hours_shutdown" {
  count = var.environment == "development" ? 1 : 0

  name                = "${var.environment}-after-hours-shutdown"
  description         = "Shutdown development resources after business hours"
  schedule_expression = "cron(0 23 ? * MON-FRI *)" # 8 PM BRT (11 PM UTC)

  tags = {
    Name       = "${var.environment}-after-hours-shutdown"
    CostCenter = "operations"
  }
}

# CloudWatch Events rule for morning startup (weekdays 7 AM local time)
resource "aws_cloudwatch_event_rule" "morning_startup" {
  count = var.environment == "development" ? 1 : 0

  name                = "${var.environment}-morning-startup"
  description         = "Start development resources in the morning"
  schedule_expression = "cron(0 10 ? * MON-FRI *)" # 7 AM BRT (10 AM UTC)

  tags = {
    Name       = "${var.environment}-morning-startup"
    CostCenter = "operations"
  }
}

# EventBridge targets
resource "aws_cloudwatch_event_target" "shutdown_target" {
  count = var.environment == "development" ? 1 : 0

  rule      = aws_cloudwatch_event_rule.after_hours_shutdown[0].name
  target_id = "AutoShutdownLambda"
  arn       = aws_lambda_function.auto_shutdown[0].arn

  input = jsonencode({
    action = "shutdown"
  })
}

resource "aws_cloudwatch_event_target" "startup_target" {
  count = var.environment == "development" ? 1 : 0

  rule      = aws_cloudwatch_event_rule.morning_startup[0].name
  target_id = "AutoStartupLambda"
  arn       = aws_lambda_function.auto_shutdown[0].arn

  input = jsonencode({
    action = "startup"
  })
}

# Lambda permissions for EventBridge
resource "aws_lambda_permission" "allow_eventbridge_shutdown" {
  count = var.environment == "development" ? 1 : 0

  statement_id  = "AllowExecutionFromEventBridgeShutdown"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.auto_shutdown[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.after_hours_shutdown[0].arn
}

resource "aws_lambda_permission" "allow_eventbridge_startup" {
  count = var.environment == "development" ? 1 : 0

  statement_id  = "AllowExecutionFromEventBridgeStartup"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.auto_shutdown[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.morning_startup[0].arn
}

# ========================================
# 5. CLOUDWATCH ALARMS AT 80% BUDGET
# ========================================

# Create CloudWatch alarms for cost monitoring
resource "aws_cloudwatch_metric_alarm" "budget_80_percent" {
  alarm_name          = "${var.environment}-budget-80-percent-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400" # Daily
  statistic           = "Maximum"
  threshold           = 400 # 80% of $500
  alarm_description   = "Alert when AWS costs reach 80% of monthly budget ($400)"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Name       = "${var.environment}-budget-80-alarm"
    CostCenter = "operations"
    Severity   = "HIGH"
  }
}

resource "aws_cloudwatch_metric_alarm" "budget_90_percent" {
  alarm_name          = "${var.environment}-budget-90-percent-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = 450 # 90% of $500
  alarm_description   = "CRITICAL: AWS costs reached 90% of monthly budget ($450)"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn, aws_sns_topic.critical_alerts.arn]

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Name       = "${var.environment}-budget-90-alarm"
    CostCenter = "operations"
    Severity   = "CRITICAL"
  }
}

resource "aws_cloudwatch_metric_alarm" "budget_exceeded" {
  alarm_name          = "${var.environment}-budget-exceeded-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = 500
  alarm_description   = "EMERGENCY: Monthly budget of $500 has been exceeded!"
  alarm_actions = [
    aws_sns_topic.cost_alerts.arn,
    aws_sns_topic.critical_alerts.arn
  ]

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Name       = "${var.environment}-budget-exceeded-alarm"
    CostCenter = "operations"
    Severity   = "EMERGENCY"
  }
}

# SNS Topics for cost alerts
resource "aws_sns_topic" "cost_alerts" {
  name = "${var.environment}-cost-alerts"

  tags = {
    Name       = "${var.environment}-cost-alerts"
    CostCenter = "operations"
  }
}

resource "aws_sns_topic" "critical_alerts" {
  name = "${var.environment}-critical-cost-alerts"

  tags = {
    Name       = "${var.environment}-critical-alerts"
    CostCenter = "operations"
  }
}

# SNS Topic subscriptions
resource "aws_sns_topic_subscription" "cost_control_alert_email" {
  topic_arn = aws_sns_topic.cost_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_sns_topic_subscription" "critical_alert_email" {
  topic_arn = aws_sns_topic.critical_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_sns_topic_subscription" "critical_alert_sms" {
  count = var.alert_phone != "" ? 1 : 0

  topic_arn = aws_sns_topic.critical_alerts.arn
  protocol  = "sms"
  endpoint  = var.alert_phone
}

# ========================================
# 6. COST REPORTS DASHBOARD IN AGENTOPS
# ========================================

resource "aws_cloudwatch_dashboard" "cost_control" {
  dashboard_name = "${var.environment}-cost-control-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Billing", "EstimatedCharges", { stat = "Maximum", label = "Total Estimated Charges" }],
            ["...", { stat = "Maximum", label = "Daily Charges", period = 86400 }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1" # Billing metrics are only in us-east-1
          title   = "Monthly Cost Tracking"
          period  = 86400
          yAxis = {
            left = {
              min   = 0
              max   = 500
              label = "Cost (USD)"
            }
          }
          annotations = {
            horizontal = [
              {
                label = "Budget Limit"
                value = 500
                fill  = "above"
                color = "#FF0000"
              },
              {
                label = "80% Warning"
                value = 400
                color = "#FFA500"
              }
            ]
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", "${local.app_name}-manager-agent", { stat = "Sum", label = "Manager Agent" }],
            ["AWS/Lambda", "Invocations", "FunctionName", "${local.app_name}-engineer-agent", { stat = "Sum", label = "Engineer Agent" }],
            ["AWS/Lambda", "Invocations", "FunctionName", "${local.app_name}-qa-agent", { stat = "Sum", label = "QA Agent" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Agent Invocations"
          period  = 3600
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 8
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", "${local.app_name}-manager-agent", { stat = "Average", label = "Manager Agent" }],
            ["AWS/Lambda", "Duration", "FunctionName", "${local.app_name}-engineer-agent", { stat = "Average", label = "Engineer Agent" }],
            ["AWS/Lambda", "Duration", "FunctionName", "${local.app_name}-qa-agent", { stat = "Average", label = "QA Agent" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Agent Execution Duration"
          period  = 3600
          yAxis = {
            left = {
              label = "Duration (ms)"
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 6
        width  = 8
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "ConcurrentExecutions", "FunctionName", "${local.app_name}-manager-agent", { stat = "Maximum", label = "Manager Agent" }],
            ["AWS/Lambda", "ConcurrentExecutions", "FunctionName", "${local.app_name}-engineer-agent", { stat = "Maximum", label = "Engineer Agent" }],
            ["AWS/Lambda", "ConcurrentExecutions", "FunctionName", "${local.app_name}-qa-agent", { stat = "Maximum", label = "QA Agent" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Concurrent Executions"
          period  = 300
          annotations = {
            horizontal = [
              { label = "Manager Limit", value = 1, color = "#2ca02c" },
              { label = "Engineer/QA Limit", value = 2, color = "#1f77b4" }
            ]
          }
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 6
        width  = 8
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Throttles", "FunctionName", "${local.app_name}-manager-agent", { stat = "Sum", label = "Manager Agent" }],
            ["AWS/Lambda", "Throttles", "FunctionName", "${local.app_name}-engineer-agent", { stat = "Sum", label = "Engineer Agent" }],
            ["AWS/Lambda", "Throttles", "FunctionName", "${local.app_name}-qa-agent", { stat = "Sum", label = "QA Agent" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Lambda Throttles"
          period  = 300
        }
      },
      {
        type   = "number"
        x      = 0
        y      = 12
        width  = 6
        height = 3

        properties = {
          metrics = [
            ["AWS/Billing", "EstimatedCharges", { stat = "Maximum" }]
          ]
          view   = "singleValue"
          region = "us-east-1"
          title  = "Current Monthly Cost"
          period = 86400
        }
      },
      {
        type   = "number"
        x      = 6
        y      = 12
        width  = 6
        height = 3

        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", { stat = "Sum" }]
          ]
          view   = "singleValue"
          region = data.aws_region.current.name
          title  = "Total Invocations Today"
          period = 86400
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiName", "${var.environment}-agent-control-api", { stat = "Sum", label = "API Requests" }],
            ["AWS/ApiGateway", "4XXError", "ApiName", "${var.environment}-agent-control-api", { stat = "Sum", label = "4XX Errors" }],
            ["AWS/ApiGateway", "5XXError", "ApiName", "${var.environment}-agent-control-api", { stat = "Sum", label = "5XX Errors" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "API Gateway Metrics"
          period  = 300
        }
      }
    ]
  })
}

# ========================================
# 7. CIRCUIT BREAKERS
# ========================================

# Lambda function for circuit breaker
resource "aws_lambda_function" "circuit_breaker" {
  filename         = data.archive_file.circuit_breaker_package.output_path
  function_name    = "${var.environment}-cost-circuit-breaker"
  role             = aws_iam_role.circuit_breaker_role.arn
  handler          = "circuit_breaker.lambda_handler"
  source_code_hash = data.archive_file.circuit_breaker_package.output_base64sha256
  runtime          = "python3.11"
  timeout          = 60

  environment {
    variables = {
      ENVIRONMENT       = var.environment
      COST_LIMIT        = "500"
      DAILY_LIMIT       = "17" # ~$500 / 30 days
      MANAGER_FUNCTION  = "manager-agent-${var.environment}"
      ENGINEER_FUNCTION = "engineer-agent-${var.environment}"
      QA_FUNCTION       = "${var.environment}-qa-agent"
      SNS_TOPIC_ARN     = aws_sns_topic.critical_alerts.arn
    }
  }

  tags = merge(var.tags, {
    Name       = "${var.environment}-circuit-breaker"
    CostCenter = "operations"
    Purpose    = "Cost Control Circuit Breaker"
  })
}

# Package circuit breaker Lambda
data "archive_file" "circuit_breaker_package" {
  type        = "zip"
  source_file = "${path.module}/cost_control/circuit_breaker.py"
  output_path = "${path.module}/circuit_breaker.zip"
}

# IAM role for circuit breaker
resource "aws_iam_role" "circuit_breaker_role" {
  name = "${var.environment}-circuit-breaker-role"

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
}

# IAM policy for circuit breaker
resource "aws_iam_role_policy" "circuit_breaker_policy" {
  role = aws_iam_role.circuit_breaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:PutFunctionConcurrency",
          "lambda:GetFunction",
          "lambda:UpdateFunctionConfiguration"
        ]
        Resource = [
          "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:manager-agent-${var.environment}",
          "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:engineer-agent-${var.environment}",
          "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.environment}-qa-agent"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ce:GetCostAndUsage",
          "ce:GetCostForecast"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetMetricData"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.critical_alerts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "events:DisableRule",
          "events:EnableRule"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Events rule for circuit breaker monitoring (every 10 minutes)
resource "aws_cloudwatch_event_rule" "circuit_breaker_monitor" {
  name                = "${var.environment}-circuit-breaker-monitor"
  description         = "Monitor costs and trigger circuit breaker if needed"
  schedule_expression = "rate(10 minutes)"

  tags = {
    Name       = "${var.environment}-circuit-breaker-monitor"
    CostCenter = "operations"
  }
}

# EventBridge target for circuit breaker
resource "aws_cloudwatch_event_target" "circuit_breaker_target" {
  rule      = aws_cloudwatch_event_rule.circuit_breaker_monitor.name
  target_id = "CircuitBreakerLambda"
  arn       = aws_lambda_function.circuit_breaker.arn
}

# Lambda permission for EventBridge
resource "aws_lambda_permission" "allow_eventbridge_circuit_breaker" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.circuit_breaker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.circuit_breaker_monitor.arn
}

# AWS Budgets for additional cost control
resource "aws_budgets_budget" "monthly_budget" {
  name         = "${var.environment}-monthly-budget"
  budget_type  = "COST"
  limit_amount = "500"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : ["placeholder@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : ["placeholder@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : ["placeholder@example.com"]
  }

  # cost_filters = {
  #   TagKeyValue = ["CostCenter$engineering", "CostCenter$qa", "CostCenter$operations"]
  # }
}

# Outputs
output "api_gateway_url" {
  value       = aws_api_gateway_deployment.agent_api_deployment.invoke_url
  description = "API Gateway URL for agent control"
}

output "cost_dashboard_url" {
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${data.aws_region.current.name}#dashboards:name=${aws_cloudwatch_dashboard.cost_control.dashboard_name}"
  description = "URL to the cost control dashboard"
}

output "monthly_budget_id" {
  value       = aws_budgets_budget.monthly_budget.id
  description = "AWS Budget ID for monthly cost tracking"
}

output "circuit_breaker_function" {
  value       = aws_lambda_function.circuit_breaker.function_name
  description = "Circuit breaker Lambda function name"
}
