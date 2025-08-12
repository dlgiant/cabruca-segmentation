# Agent Infrastructure for Cabruca Segmentation System
# Comprehensive configuration for Lambda functions, DynamoDB, S3, and CloudWatch
# Cost-optimized for MVP deployment in Brazil (sa-east-1)

locals {
  agent_common_tags = merge(
    local.common_tags,
    {
      Component   = "Agent-Infrastructure"
      CostCenter  = "AI-Agents"
      Automation  = "Enabled"
    }
  )
  
  agents = {
    manager = {
      name        = "manager-agent"
      description = "Orchestrates workflow and coordinates other agents"
      runtime     = "python3.11"
      handler     = "lambda_function.lambda_handler"
      timeout     = 300
      memory      = 1024
      environment = {
        AGENT_TYPE = "MANAGER"
        LOG_LEVEL  = "INFO"
      }
    }
    engineer = {
      name        = "engineer-agent"
      description = "Handles technical implementation and code generation"
      runtime     = "python3.11"
      handler     = "lambda_function.lambda_handler"
      timeout     = 300
      memory      = 2048
      environment = {
        AGENT_TYPE = "ENGINEER"
        LOG_LEVEL  = "INFO"
      }
    }
    qa = {
      name        = "qa-agent"
      description = "Performs quality assurance and testing"
      runtime     = "python3.11"
      handler     = "lambda_function.lambda_handler"
      timeout     = 180
      memory      = 512
      environment = {
        AGENT_TYPE = "QA"
        LOG_LEVEL  = "INFO"
      }
    }
    researcher = {
      name        = "researcher-agent"
      description = "Conducts research and data analysis for cabruca insights"
      runtime     = "python3.11"
      handler     = "lambda_function.lambda_handler"
      timeout     = 240
      memory      = 1024
      environment = {
        AGENT_TYPE = "RESEARCHER"
        LOG_LEVEL  = "INFO"
      }
    }
    data_processor = {
      name        = "data-processor-agent"
      description = "Processes satellite imagery and agricultural data"
      runtime     = "python3.11"
      handler     = "lambda_function.lambda_handler"
      timeout     = 900
      memory      = 3008
      environment = {
        AGENT_TYPE = "DATA_PROCESSOR"
        LOG_LEVEL  = "INFO"
      }
    }
  }
}

# ===========================
# IAM Roles for Lambda Functions
# ===========================

resource "aws_iam_role" "agent_lambda_role" {
  for_each = local.agents
  
  name = "${local.app_name}-${each.value.name}-role"
  
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
  
  tags = merge(
    local.agent_common_tags,
    {
      Name      = "${local.app_name}-${each.value.name}-role"
      AgentType = each.key
    }
  )
}

# IAM Policy for Lambda execution
resource "aws_iam_policy" "agent_lambda_policy" {
  for_each = local.agents
  
  name        = "${local.app_name}-${each.value.name}-policy"
  description = "Policy for ${each.value.name} Lambda function"
  
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
        Resource = "arn:aws:logs:${var.aws_region}:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:DeleteItem",
          "dynamodb:BatchGetItem",
          "dynamodb:BatchWriteItem"
        ]
        Resource = [
          aws_dynamodb_table.agent_state.arn,
          aws_dynamodb_table.agent_memory.arn,
          aws_dynamodb_table.agent_tasks.arn,
          "${aws_dynamodb_table.agent_state.arn}/index/*",
          "${aws_dynamodb_table.agent_memory.arn}/index/*",
          "${aws_dynamodb_table.agent_tasks.arn}/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.agent_artifacts.arn,
          "${aws_s3_bucket.agent_artifacts.arn}/*",
          aws_s3_bucket.agent_prompts.arn,
          "${aws_s3_bucket.agent_prompts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutEvents"
        ]
        Resource = "arn:aws:events:${var.aws_region}:*:event-bus/default"
      },
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = "arn:aws:lambda:${var.aws_region}:*:function:${local.app_name}-*"
      }
    ]
  })
  
  tags = merge(
    local.agent_common_tags,
    {
      Name      = "${local.app_name}-${each.value.name}-policy"
      AgentType = each.key
    }
  )
}

# Attach policies to roles
resource "aws_iam_role_policy_attachment" "agent_lambda_policy" {
  for_each = local.agents
  
  role       = aws_iam_role.agent_lambda_role[each.key].name
  policy_arn = aws_iam_policy.agent_lambda_policy[each.key].arn
}

# Attach AWS managed policy for basic Lambda execution
resource "aws_iam_role_policy_attachment" "agent_lambda_basic" {
  for_each = local.agents
  
  role       = aws_iam_role.agent_lambda_role[each.key].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ===========================
# Lambda Functions
# ===========================

resource "aws_lambda_function" "agents" {
  for_each = local.agents
  
  function_name = "${local.app_name}-${each.value.name}"
  description   = each.value.description
  
  role    = aws_iam_role.agent_lambda_role[each.key].arn
  handler = each.value.handler
  runtime = each.value.runtime
  timeout = each.value.timeout
  memory_size = each.value.memory
  
  # For MVP, we'll use inline code - in production, use S3
  filename         = "${path.module}/${replace(each.value.name, "-", "_")}/lambda_function.zip"
  source_code_hash = filebase64sha256("${path.module}/${replace(each.value.name, "-", "_")}/lambda_function.zip")
  
  environment {
    variables = merge(
      each.value.environment,
      {
        DYNAMODB_STATE_TABLE  = aws_dynamodb_table.agent_state.name
        DYNAMODB_MEMORY_TABLE = aws_dynamodb_table.agent_memory.name
        DYNAMODB_TASKS_TABLE  = aws_dynamodb_table.agent_tasks.name
        S3_ARTIFACTS_BUCKET   = aws_s3_bucket.agent_artifacts.id
        S3_PROMPTS_BUCKET     = aws_s3_bucket.agent_prompts.id
        EVENTBRIDGE_BUS       = "default"
        REGION               = var.aws_region
        ENVIRONMENT          = var.environment
        AGENTOPS_API_KEY     = var.agentops_api_key
      }
    )
  }
  
  # reserved_concurrent_executions = var.environment == "prod" ? 10 : 2  # Disabled to avoid account limits
  
  tracing_config {
    mode = "Active"
  }
  
  tags = merge(
    local.agent_common_tags,
    {
      Name        = "${local.app_name}-${each.value.name}"
      AgentType   = each.key
      CostAllocation = "Agent-${each.key}"
    }
  )
}

# Lambda Function URLs for direct invocation (MVP simplicity)
resource "aws_lambda_function_url" "agents" {
  for_each = local.agents
  
  function_name      = aws_lambda_function.agents[each.key].function_name
  authorization_type = "AWS_IAM"  # Change to "NONE" for public access during testing
  
  cors {
    allow_credentials = true
    allow_origins     = ["*"]
    allow_methods     = ["GET", "POST"]
    allow_headers     = ["*"]
    expose_headers    = ["*"]
    max_age          = 3600
  }
}

# ===========================
# DynamoDB Tables
# ===========================

# Agent State Table - stores current state of each agent
resource "aws_dynamodb_table" "agent_state" {
  name           = "${local.app_name}-agent-state"
  billing_mode   = "PAY_PER_REQUEST"  # On-demand for MVP
  hash_key       = "agent_id"
  range_key      = "timestamp"
  
  attribute {
    name = "agent_id"
    type = "S"
  }
  
  attribute {
    name = "timestamp"
    type = "N"
  }
  
  attribute {
    name = "status"
    type = "S"
  }
  
  global_secondary_index {
    name            = "status-index"
    hash_key        = "status"
    range_key       = "timestamp"
    projection_type = "ALL"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  point_in_time_recovery {
    enabled = var.environment == "prod" ? true : false
  }
  
  server_side_encryption {
    enabled = true
  }
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-agent-state"
      Purpose        = "Agent-State-Management"
      CostAllocation = "DynamoDB-AgentState"
    }
  )
}

# Agent Memory Table - stores conversation history and context
resource "aws_dynamodb_table" "agent_memory" {
  name           = "${local.app_name}-agent-memory"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "session_id"
  range_key      = "message_id"
  
  attribute {
    name = "session_id"
    type = "S"
  }
  
  attribute {
    name = "message_id"
    type = "S"
  }
  
  attribute {
    name = "agent_id"
    type = "S"
  }
  
  attribute {
    name = "created_at"
    type = "N"
  }
  
  global_secondary_index {
    name            = "agent-index"
    hash_key        = "agent_id"
    range_key       = "created_at"
    projection_type = "ALL"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"
  
  server_side_encryption {
    enabled = true
  }
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-agent-memory"
      Purpose        = "Agent-Memory-Storage"
      CostAllocation = "DynamoDB-AgentMemory"
    }
  )
}

# Agent Tasks Table - stores task queue and results
resource "aws_dynamodb_table" "agent_tasks" {
  name           = "${local.app_name}-agent-tasks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "task_id"
  
  attribute {
    name = "task_id"
    type = "S"
  }
  
  attribute {
    name = "agent_id"
    type = "S"
  }
  
  attribute {
    name = "priority"
    type = "N"
  }
  
  attribute {
    name = "status"
    type = "S"
  }
  
  global_secondary_index {
    name            = "agent-priority-index"
    hash_key        = "agent_id"
    range_key       = "priority"
    projection_type = "ALL"
  }
  
  global_secondary_index {
    name            = "status-index"
    hash_key        = "status"
    projection_type = "ALL"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  server_side_encryption {
    enabled = true
  }
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-agent-tasks"
      Purpose        = "Agent-Task-Queue"
      CostAllocation = "DynamoDB-AgentTasks"
    }
  )
}

# ===========================
# S3 Buckets
# ===========================

# S3 Bucket for Agent Artifacts (outputs, reports, generated content)
resource "aws_s3_bucket" "agent_artifacts" {
  bucket = "${local.app_name}-agent-artifacts-${data.aws_caller_identity.current.account_id}"
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-agent-artifacts"
      Purpose        = "Agent-Output-Storage"
      CostAllocation = "S3-AgentArtifacts"
    }
  )
}

resource "aws_s3_bucket_versioning" "agent_artifacts" {
  bucket = aws_s3_bucket.agent_artifacts.id
  
  versioning_configuration {
    status = var.environment == "prod" ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "agent_artifacts" {
  bucket = aws_s3_bucket.agent_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "agent_artifacts" {
  bucket = aws_s3_bucket.agent_artifacts.id
  
  rule {
    id     = "cleanup-old-artifacts"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# S3 Bucket for Agent Prompts and Templates
resource "aws_s3_bucket" "agent_prompts" {
  bucket = "${local.app_name}-agent-prompts-${data.aws_caller_identity.current.account_id}"
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-agent-prompts"
      Purpose        = "Agent-Prompt-Storage"
      CostAllocation = "S3-AgentPrompts"
    }
  )
}

resource "aws_s3_bucket_versioning" "agent_prompts" {
  bucket = aws_s3_bucket.agent_prompts.id
  
  versioning_configuration {
    status = "Enabled"  # Always version prompts for rollback capability
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "agent_prompts" {
  bucket = aws_s3_bucket.agent_prompts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket for Processing Queue (input data for agents)
resource "aws_s3_bucket" "agent_queue" {
  bucket = "${local.app_name}-agent-queue-${data.aws_caller_identity.current.account_id}"
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-agent-queue"
      Purpose        = "Agent-Input-Queue"
      CostAllocation = "S3-AgentQueue"
    }
  )
}

resource "aws_s3_bucket_lifecycle_configuration" "agent_queue" {
  bucket = aws_s3_bucket.agent_queue.id
  
  rule {
    id     = "cleanup-processed"
    status = "Enabled"
    
    expiration {
      days = 7  # Remove processed items after 7 days
    }
  }
}

# S3 Bucket Event Notifications for triggering agents
resource "aws_s3_bucket_notification" "agent_queue" {
  bucket = aws_s3_bucket.agent_queue.id
  
  lambda_function {
    lambda_function_arn = aws_lambda_function.agents["data_processor"].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "input/"
    filter_suffix       = ".json"
  }
  
  depends_on = [aws_lambda_permission.s3_invoke_data_processor]
}

# Lambda permission for S3 to invoke
resource "aws_lambda_permission" "s3_invoke_data_processor" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.agents["data_processor"].function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.agent_queue.arn
}

# ===========================
# CloudWatch Log Groups
# ===========================

resource "aws_cloudwatch_log_group" "agent_logs" {
  for_each = local.agents
  
  name              = "/aws/lambda/${local.app_name}-${each.value.name}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  
  tags = merge(
    local.agent_common_tags,
    {
      Name           = "${local.app_name}-${each.value.name}-logs"
      AgentType      = each.key
      CostAllocation = "CloudWatch-${each.key}"
    }
  )
}

# CloudWatch Log Streams for structured logging
resource "aws_cloudwatch_log_stream" "agent_streams" {
  for_each = local.agents
  
  name           = "main"
  log_group_name = aws_cloudwatch_log_group.agent_logs[each.key].name
}

# ===========================
# CloudWatch Metrics and Alarms
# ===========================

resource "aws_cloudwatch_metric_alarm" "agent_errors" {
  for_each = local.agents
  
  alarm_name          = "${local.app_name}-${each.value.name}-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "Errors"
  namespace          = "AWS/Lambda"
  period             = "60"
  statistic          = "Sum"
  threshold          = "5"
  alarm_description  = "Alarm when ${each.value.name} has too many errors"
  treat_missing_data = "notBreaching"
  
  dimensions = {
    FunctionName = aws_lambda_function.agents[each.key].function_name
  }
  
  alarm_actions = var.monitoring_configuration.alarm_email != "" ? [aws_sns_topic.agent_alerts.arn] : []
  
  tags = merge(
    local.agent_common_tags,
    {
      Name      = "${local.app_name}-${each.value.name}-error-alarm"
      AgentType = each.key
    }
  )
}

resource "aws_cloudwatch_metric_alarm" "agent_duration" {
  for_each = local.agents
  
  alarm_name          = "${local.app_name}-${each.value.name}-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "Duration"
  namespace          = "AWS/Lambda"
  period             = "300"
  statistic          = "Average"
  threshold          = each.value.timeout * 0.8 * 1000  # 80% of timeout in ms
  alarm_description  = "Alarm when ${each.value.name} is running too long"
  treat_missing_data = "notBreaching"
  
  dimensions = {
    FunctionName = aws_lambda_function.agents[each.key].function_name
  }
  
  alarm_actions = var.monitoring_configuration.alarm_email != "" ? [aws_sns_topic.agent_alerts.arn] : []
  
  tags = merge(
    local.agent_common_tags,
    {
      Name      = "${local.app_name}-${each.value.name}-duration-alarm"
      AgentType = each.key
    }
  )
}

resource "aws_cloudwatch_metric_alarm" "agent_throttles" {
  for_each = local.agents
  
  alarm_name          = "${local.app_name}-${each.value.name}-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name        = "Throttles"
  namespace          = "AWS/Lambda"
  period             = "60"
  statistic          = "Sum"
  threshold          = "1"
  alarm_description  = "Alarm when ${each.value.name} is being throttled"
  treat_missing_data = "notBreaching"
  
  dimensions = {
    FunctionName = aws_lambda_function.agents[each.key].function_name
  }
  
  alarm_actions = var.monitoring_configuration.alarm_email != "" ? [aws_sns_topic.agent_alerts.arn] : []
  
  tags = merge(
    local.agent_common_tags,
    {
      Name      = "${local.app_name}-${each.value.name}-throttle-alarm"
      AgentType = each.key
    }
  )
}

# ===========================
# CloudWatch Dashboard
# ===========================

resource "aws_cloudwatch_dashboard" "agents" {
  dashboard_name = "${local.app_name}-agents-dashboard"
  
  dashboard_body = jsonencode({
    widgets = concat(
      # Lambda metrics widgets
      [for agent_key, agent in local.agents : {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", { stat = "Sum", label = "Invocations" }],
            [".", "Errors", { stat = "Sum", label = "Errors", yAxis = "right" }],
            [".", "Duration", { stat = "Average", label = "Duration (ms)" }],
            [".", "Throttles", { stat = "Sum", label = "Throttles", yAxis = "right" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "${agent.name} Metrics"
          period  = 300
          dimensions = {
            FunctionName = aws_lambda_function.agents[agent_key].function_name
          }
        }
      }],
      # DynamoDB metrics widgets
      [
        {
          type = "metric"
          properties = {
            metrics = [
              ["AWS/DynamoDB", "ConsumedReadCapacityUnits", { stat = "Sum" }],
              [".", "ConsumedWriteCapacityUnits", { stat = "Sum" }],
              [".", "UserErrors", { stat = "Sum", yAxis = "right" }],
              [".", "SystemErrors", { stat = "Sum", yAxis = "right" }]
            ]
            view    = "timeSeries"
            stacked = false
            region  = var.aws_region
            title   = "DynamoDB Tables Metrics"
            period  = 300
          }
        }
      ],
      # S3 metrics widget
      [
        {
          type = "metric"
          properties = {
            metrics = [
              ["AWS/S3", "BucketSizeBytes", { stat = "Average" }],
              [".", "NumberOfObjects", { stat = "Average", yAxis = "right" }]
            ]
            view    = "timeSeries"
            stacked = false
            region  = var.aws_region
            title   = "S3 Buckets Storage"
            period  = 86400
          }
        }
      ]
    )
  })
}

# ===========================
# SNS Topic for Alerts
# ===========================

resource "aws_sns_topic" "agent_alerts" {
  name = "${local.app_name}-agent-alerts"
  
  tags = merge(
    local.agent_common_tags,
    {
      Name    = "${local.app_name}-agent-alerts"
      Purpose = "Agent-Alert-Notifications"
    }
  )
}

resource "aws_sns_topic_subscription" "agent_alerts_email" {
  count = var.monitoring_configuration.alarm_email != "" ? 1 : 0
  
  topic_arn = aws_sns_topic.agent_alerts.arn
  protocol  = "email"
  endpoint  = var.monitoring_configuration.alarm_email
}

# ===========================
# EventBridge Rules for Agent Orchestration
# ===========================

resource "aws_cloudwatch_event_rule" "agent_orchestration" {
  name        = "${local.app_name}-agent-orchestration"
  description = "Orchestrate agent workflows"
  
  event_pattern = jsonencode({
    source = ["cabruca.agents"]
    detail-type = [
      "Agent Task Created",
      "Agent Task Completed",
      "Agent Task Failed"
    ]
  })
  
  tags = merge(
    local.agent_common_tags,
    {
      Name = "${local.app_name}-agent-orchestration"
    }
  )
}

resource "aws_cloudwatch_event_target" "manager_agent" {
  rule      = aws_cloudwatch_event_rule.agent_orchestration.name
  target_id = "ManagerAgent"
  arn       = aws_lambda_function.agents["manager"].arn
}

resource "aws_lambda_permission" "eventbridge_invoke_manager" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.agents["manager"].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.agent_orchestration.arn
}

# ===========================
# Cost Allocation Tags
# ===========================

resource "aws_resourcegroups_group" "agent_infrastructure" {
  name = "${local.app_name}-agent-infrastructure"
  
  resource_query {
    query = jsonencode({
      ResourceTypeFilters = [
        "AWS::Lambda::Function",
        "AWS::DynamoDB::Table",
        "AWS::S3::Bucket",
        "AWS::Logs::LogGroup",
        "AWS::CloudWatch::Alarm"
      ]
      TagFilters = [
        {
          Key    = "Component"
          Values = ["Agent-Infrastructure"]
        },
        {
          Key    = "Project"
          Values = [var.project_name]
        }
      ]
    })
  }
  
  tags = merge(
    local.agent_common_tags,
    {
      Name = "${local.app_name}-agent-infrastructure-group"
    }
  )
}

# ===========================
# Data Sources
# ===========================

data "aws_caller_identity" "current" {}

# ===========================
# Outputs
# ===========================

output "agent_lambda_functions" {
  description = "Agent Lambda function details"
  value = {
    for k, v in aws_lambda_function.agents : k => {
      name         = v.function_name
      arn          = v.arn
      invoke_url   = aws_lambda_function_url.agents[k].function_url
      last_modified = v.last_modified
    }
  }
}

output "agent_dynamodb_tables" {
  description = "Agent DynamoDB table names"
  value = {
    state  = aws_dynamodb_table.agent_state.name
    memory = aws_dynamodb_table.agent_memory.name
    tasks  = aws_dynamodb_table.agent_tasks.name
  }
}

output "agent_s3_buckets" {
  description = "Agent S3 bucket names"
  value = {
    artifacts = aws_s3_bucket.agent_artifacts.id
    prompts   = aws_s3_bucket.agent_prompts.id
    queue     = aws_s3_bucket.agent_queue.id
  }
}

output "agent_cloudwatch_dashboard" {
  description = "CloudWatch dashboard URL"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.agents.dashboard_name}"
}

output "agent_sns_topic" {
  description = "SNS topic for agent alerts"
  value       = aws_sns_topic.agent_alerts.arn
}

output "total_estimated_monthly_cost" {
  description = "Estimated monthly cost for agent infrastructure (USD)"
  value = {
    lambda_invocations = "~$2-10 (based on usage)"
    dynamodb          = "~$5-15 (on-demand pricing)"
    s3_storage        = "~$1-5 (depends on data volume)"
    cloudwatch_logs   = "~$2-5"
    total_estimate    = "~$10-35/month"
  }
}