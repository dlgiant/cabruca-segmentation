# AgentOps Monitoring Infrastructure
# Provides DynamoDB tables, CloudWatch dashboards, alarms, and SNS topics for agent monitoring

# DynamoDB table for storing agent monitoring events
resource "aws_dynamodb_table" "agent_monitoring" {
  name           = "agent-monitoring"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "event_id"
  range_key      = "timestamp"

  attribute {
    name = "event_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  attribute {
    name = "agent_name"
    type = "S"
  }

  attribute {
    name = "event_type"
    type = "S"
  }

  attribute {
    name = "session_id"
    type = "S"
  }

  global_secondary_index {
    name            = "agent-name-index"
    hash_key        = "agent_name"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "event-type-index"
    hash_key        = "event_type"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "session-index"
    hash_key        = "session_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"

  tags = {
    Name        = "AgentOps-Monitoring"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# DynamoDB table for storing agent decisions with reasoning chains
resource "aws_dynamodb_table" "agent_decisions" {
  name           = "agent-decisions"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "decision_id"
  range_key      = "timestamp"

  attribute {
    name = "decision_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  attribute {
    name = "agent_name"
    type = "S"
  }

  attribute {
    name = "decision_type"
    type = "S"
  }

  global_secondary_index {
    name            = "agent-decisions-index"
    hash_key        = "agent_name"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "decision-type-index"
    hash_key        = "decision_type"
    range_key       = "timestamp"
    projection_type = "INCLUDE"
    non_key_attributes = ["confidence_score", "cost"]
  }

  tags = {
    Name        = "AgentOps-Decisions"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# SNS Topic for cost alerts
resource "aws_sns_topic" "agentops_cost_alerts" {
  name = "agentops-cost-alerts"

  tags = {
    Name        = "AgentOps-Cost-Alerts"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# SNS Topic subscription for cost alerts (email)
resource "aws_sns_topic_subscription" "cost_alert_email" {
  topic_arn = aws_sns_topic.agentops_cost_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# CloudWatch Log Group for AgentOps
resource "aws_cloudwatch_log_group" "agentops_logs" {
  name              = "/aws/agentops/monitoring"
  retention_in_days = 30

  tags = {
    Name        = "AgentOps-Logs"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# CloudWatch Dashboard - Main
resource "aws_cloudwatch_dashboard" "agentops_main" {
  dashboard_name = "AgentOps-Monitoring"

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
            ["AgentOps/Events", "issue_detected", { stat = "Sum", label = "Issues Detected" }],
            [".", "code_generated", { stat = "Sum", label = "Code Generated" }],
            [".", "test_passed", { stat = "Sum", label = "Tests Passed" }],
            [".", "test_failed", { stat = "Sum", label = "Tests Failed" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Agent Activity Overview"
          period  = 300
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
            ["AgentOps/Costs", "LambdaExecutionCost", { stat = "Sum", label = "Lambda Cost" }],
            [".", "EventCost", { stat = "Sum", label = "Event Processing Cost" }],
            [".", "TotalCost", { stat = "Sum", label = "Total Cost" }]
          ]
          view    = "timeSeries"
          stacked = true
          region  = var.aws_region
          title   = "Agent Costs Over Time"
          period  = 300
          yAxis = {
            left = {
              label     = "Cost ($)"
              showUnits = false
            }
          }
        }
      }
    ]
  })
}

# CloudWatch Alarm - High Cost
resource "aws_cloudwatch_metric_alarm" "agentops_high_cost" {
  alarm_name          = "AgentOps-HighCost"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "TotalCost"
  namespace           = "AgentOps/Costs"
  period              = "300"
  statistic           = "Sum"
  threshold           = var.cost_alert_threshold
  alarm_description   = "Alert when total agent costs exceed threshold"
  alarm_actions       = [aws_sns_topic.agentops_cost_alerts.arn]

  dimensions = {
    Environment = var.environment
  }

  tags = {
    Name        = "AgentOps-HighCost-Alarm"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# CloudWatch Alarm - High Error Rate
resource "aws_cloudwatch_metric_alarm" "agentops_high_error_rate" {
  alarm_name          = "AgentOps-HighErrorRate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "test_failed"
  namespace           = "AgentOps/Events"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "Alert when test failures exceed 5 in 5 minutes"
  alarm_actions       = [aws_sns_topic.agentops_cost_alerts.arn]

  tags = {
    Name        = "AgentOps-HighErrorRate-Alarm"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# IAM Role for Lambda functions to access monitoring resources
resource "aws_iam_role" "agentops_monitoring_role" {
  name = "agentops-monitoring-role"

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
    Name        = "AgentOps-Monitoring-Role"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# IAM Policy for AgentOps monitoring
resource "aws_iam_policy" "agentops_monitoring_policy" {
  name        = "agentops-monitoring-policy"
  description = "Policy for AgentOps monitoring functions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:UpdateItem"
        ]
        Resource = [
          aws_dynamodb_table.agent_monitoring.arn,
          "${aws_dynamodb_table.agent_monitoring.arn}/*",
          aws_dynamodb_table.agent_decisions.arn,
          "${aws_dynamodb_table.agent_decisions.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          aws_cloudwatch_log_group.agentops_logs.arn,
          "${aws_cloudwatch_log_group.agentops_logs.arn}:*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.agentops_cost_alerts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ce:GetCostAndUsage",
          "ce:GetCostForecast"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "agentops_monitoring_attachment" {
  role       = aws_iam_role.agentops_monitoring_role.name
  policy_arn = aws_iam_policy.agentops_monitoring_policy.arn
}

# Lambda function for dashboard deployment
resource "aws_lambda_function" "deploy_dashboards" {
  filename         = "agentops_dashboard.zip"
  function_name    = "agentops-deploy-dashboards"
  role            = aws_iam_role.agentops_monitoring_role.arn
  handler         = "agentops_dashboard.lambda_handler"
  source_code_hash = filebase64sha256("agentops_dashboard.zip")
  runtime         = "python3.11"
  timeout         = 60

  environment {
    variables = {
      ENVIRONMENT = var.environment
    }
  }

  tags = {
    Name        = "AgentOps-Deploy-Dashboards"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# EventBridge Rule for agent collaboration monitoring
resource "aws_cloudwatch_event_rule" "agent_collaboration" {
  name        = "agentops-agent-collaboration"
  description = "Capture agent collaboration events"

  event_pattern = jsonencode({
    source = ["agentops.manager", "agentops.engineer", "agentops.qa"]
    detail-type = ["AgentCollaboration"]
  })

  tags = {
    Name        = "AgentOps-Collaboration-Rule"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# Lambda function for analyzing collaboration patterns
resource "aws_lambda_function" "analyze_collaboration" {
  filename         = "agentops_monitoring.zip"
  function_name    = "agentops-analyze-collaboration"
  role            = aws_iam_role.agentops_monitoring_role.arn
  handler         = "analyze_collaboration.lambda_handler"
  source_code_hash = filebase64sha256("agentops_monitoring.zip")
  runtime         = "python3.11"
  timeout         = 60

  environment {
    variables = {
      MONITORING_TABLE_NAME = aws_dynamodb_table.agent_monitoring.name
      DECISIONS_TABLE_NAME  = aws_dynamodb_table.agent_decisions.name
      ENVIRONMENT          = var.environment
    }
  }

  tags = {
    Name        = "AgentOps-Analyze-Collaboration"
    Environment = var.environment
    Component   = "AgentOps"
  }
}

# EventBridge Target for collaboration analysis
resource "aws_cloudwatch_event_target" "collaboration_analyzer" {
  rule      = aws_cloudwatch_event_rule.agent_collaboration.name
  target_id = "CollaborationAnalyzer"
  arn       = aws_lambda_function.analyze_collaboration.arn
}

# Permission for EventBridge to invoke Lambda
resource "aws_lambda_permission" "allow_eventbridge_collaboration" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.analyze_collaboration.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.agent_collaboration.arn
}

# Outputs
output "monitoring_table_name" {
  value = aws_dynamodb_table.agent_monitoring.name
  description = "Name of the agent monitoring DynamoDB table"
}

output "decisions_table_name" {
  value = aws_dynamodb_table.agent_decisions.name
  description = "Name of the agent decisions DynamoDB table"
}

output "cost_alert_topic_arn" {
  value = aws_sns_topic.agentops_cost_alerts.arn
  description = "ARN of the cost alerts SNS topic"
}

output "dashboard_url" {
  value = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.agentops_main.dashboard_name}"
  description = "URL to the main AgentOps dashboard"
}
