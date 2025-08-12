# QA Engineer Agent Infrastructure
# Automated testing agent for deployments

# Data sources
data "aws_region" "current" {}
# Using the data source from agents_infrastructure.tf

# DynamoDB table for test results
resource "aws_dynamodb_table" "qa_test_results" {
  name           = "${var.environment}-qa-test-results"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "suite_id"
  range_key      = "deployment_id"

  attribute {
    name = "suite_id"
    type = "S"
  }

  attribute {
    name = "deployment_id"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "deployment-index"
    hash_key        = "deployment_id"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  tags = {
    Name        = "${var.environment}-qa-test-results"
    Environment = var.environment
    Agent       = "qa"
  }
}

# S3 bucket for QA artifacts
resource "aws_s3_bucket" "qa_artifacts" {
  bucket = "${var.environment}-qa-agent-artifacts-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.environment}-qa-artifacts"
    Environment = var.environment
    Agent       = "qa"
  }
}

resource "aws_s3_bucket_versioning" "qa_artifacts" {
  bucket = aws_s3_bucket.qa_artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "qa_artifacts" {
  bucket = aws_s3_bucket.qa_artifacts.id

  rule {
    id     = "cleanup-old-tests"
    status = "Enabled"

    expiration {
      days = 30
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# CodeBuild project for Cypress tests
resource "aws_codebuild_project" "cypress_tests" {
  name          = "${var.environment}-cypress-tests"
  description   = "Run Cypress E2E tests for deployments"
  service_role  = aws_iam_role.codebuild_cypress.arn

  artifacts {
    type = "S3"
    location = aws_s3_bucket.qa_artifacts.bucket
    path = "test-results"
    packaging = "ZIP"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                      = "aws/codebuild/standard:7.0"
    type                       = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"

    environment_variable {
      name  = "S3_BUCKET"
      value = aws_s3_bucket.qa_artifacts.bucket
    }

    environment_variable {
      name  = "BASE_URL"
      value = var.api_endpoint != "" ? var.api_endpoint : "http://localhost:3000"
    }
  }

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.qa_artifacts.bucket}/buildspecs/cypress_buildspec.yml"
    buildspec = file("${path.module}/qa_agent/cypress_buildspec.yml")
  }

  cache {
    type     = "S3"
    location = "${aws_s3_bucket.qa_artifacts.bucket}/cache"
  }

  logs_config {
    cloudwatch_logs {
      group_name  = "/aws/codebuild/${var.environment}-cypress-tests"
      stream_name = "build-logs"
    }
  }

  tags = {
    Name        = "${var.environment}-cypress-tests"
    Environment = var.environment
    Agent       = "qa"
  }
}

# IAM role for CodeBuild
resource "aws_iam_role" "codebuild_cypress" {
  name = "${var.environment}-codebuild-cypress-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "codebuild_cypress" {
  role = aws_iam_role.codebuild_cypress.id

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
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/codebuild/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.qa_artifacts.arn,
          "${aws_s3_bucket.qa_artifacts.arn}/*"
        ]
      }
    ]
  })
}

# Lambda function for QA Agent
resource "aws_lambda_function" "qa_agent" {
  filename         = data.archive_file.qa_agent_package.output_path
  function_name    = "${var.environment}-qa-agent"
  role            = aws_iam_role.qa_agent_lambda.arn
  handler         = "lambda_function.lambda_handler"
  source_code_hash = data.archive_file.qa_agent_package.output_base64sha256
  runtime         = "python3.11"
  timeout         = 300  # 5 minutes
  memory_size     = 512

  environment {
    variables = {
      TEST_RESULTS_TABLE   = aws_dynamodb_table.qa_test_results.name
      CYPRESS_PROJECT_NAME = aws_codebuild_project.cypress_tests.name
      API_ENDPOINT        = var.api_endpoint
      S3_BUCKET           = aws_s3_bucket.qa_artifacts.bucket
      EVENT_BUS_NAME      = var.eventbridge_bus_name != "" ? var.eventbridge_bus_name : "default"
      COST_THRESHOLD      = var.cost_threshold
      ENVIRONMENT         = var.environment
    }
  }

  tags = {
    Name        = "${var.environment}-qa-agent"
    Environment = var.environment
    Agent       = "qa"
    CostCenter  = "engineering"
  }
}

# Package Lambda function
data "archive_file" "qa_agent_package" {
  type        = "zip"
  source_dir  = "${path.module}/qa_agent"
  output_path = "${path.module}/qa_agent.zip"
}

# IAM role for QA Agent Lambda
resource "aws_iam_role" "qa_agent_lambda" {
  name = "${var.environment}-qa-agent-lambda-role"

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

# IAM policy for QA Agent Lambda
resource "aws_iam_role_policy" "qa_agent_lambda" {
  role = aws_iam_role.qa_agent_lambda.id

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
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.qa_test_results.arn,
          "${aws_dynamodb_table.qa_test_results.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.qa_artifacts.arn,
          "${aws_s3_bucket.qa_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "codebuild:StartBuild",
          "codebuild:BatchGetBuilds"
        ]
        Resource = aws_codebuild_project.cypress_tests.arn
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
          "ce:GetCostAndUsage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach basic execution role
resource "aws_iam_role_policy_attachment" "qa_agent_lambda_basic" {
  role       = aws_iam_role.qa_agent_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# EventBridge rule to trigger QA Agent on deployment completion
resource "aws_cloudwatch_event_rule" "qa_agent_trigger" {
  name        = "${var.environment}-qa-agent-trigger"
  description = "Trigger QA Agent on deployment completion"

  event_pattern = jsonencode({
    source      = ["engineer.agent"]
    detail-type = [
      "Implementation.code_change.Completed",
      "Implementation.infrastructure.Completed",
      "Implementation.configuration.Completed",
      "Implementation.bug_fix.Completed",
      "Implementation.feature.Completed",
      "Implementation.refactoring.Completed"
    ]
  })

  event_bus_name = var.eventbridge_bus_name != "" ? var.eventbridge_bus_name : "default"

  tags = {
    Name        = "${var.environment}-qa-agent-trigger"
    Environment = var.environment
    Agent       = "qa"
  }
}

# EventBridge target for QA Agent
resource "aws_cloudwatch_event_target" "qa_agent" {
  rule      = aws_cloudwatch_event_rule.qa_agent_trigger.name
  target_id = "qa-agent-lambda"
  arn       = aws_lambda_function.qa_agent.arn
  
  event_bus_name = var.eventbridge_bus_name != "" ? var.eventbridge_bus_name : "default"
}

# Lambda permission for EventBridge
resource "aws_lambda_permission" "qa_agent_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.qa_agent.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.qa_agent_trigger.arn
}

# CloudWatch Log Group for QA Agent
resource "aws_cloudwatch_log_group" "qa_agent" {
  name              = "/aws/lambda/${aws_lambda_function.qa_agent.function_name}"
  retention_in_days = 7

  tags = {
    Name        = "${var.environment}-qa-agent-logs"
    Environment = var.environment
    Agent       = "qa"
  }
}

# CloudWatch Log Group for CodeBuild
resource "aws_cloudwatch_log_group" "cypress_codebuild" {
  name              = "/aws/codebuild/${aws_codebuild_project.cypress_tests.name}"
  retention_in_days = 7

  tags = {
    Name        = "${var.environment}-cypress-codebuild-logs"
    Environment = var.environment
    Agent       = "qa"
  }
}

# CloudWatch Dashboard for QA metrics
resource "aws_cloudwatch_dashboard" "qa_metrics" {
  dashboard_name = "${var.environment}-qa-agent-metrics"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", { stat = "Sum", label = "Test Runs" }],
            [".", "Duration", { stat = "Average", label = "Avg Duration (ms)" }],
            [".", "Errors", { stat = "Sum", label = "Errors" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "QA Agent Performance"
          period  = 300
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/CodeBuild", "Builds", { stat = "Sum", label = "Cypress Builds" }],
            [".", "SuccessRate", { stat = "Average", label = "Success Rate %" }],
            [".", "Duration", { stat = "Average", label = "Build Duration (s)" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Cypress Test Execution"
          period  = 300
        }
      }
    ]
  })
}

# Outputs
output "qa_agent_function_name" {
  value       = aws_lambda_function.qa_agent.function_name
  description = "Name of the QA Agent Lambda function"
}

output "qa_agent_function_arn" {
  value       = aws_lambda_function.qa_agent.arn
  description = "ARN of the QA Agent Lambda function"
}

output "qa_artifacts_bucket" {
  value       = aws_s3_bucket.qa_artifacts.bucket
  description = "S3 bucket for QA artifacts"
}

output "cypress_project_name" {
  value       = aws_codebuild_project.cypress_tests.name
  description = "Name of the Cypress CodeBuild project"
}

output "qa_test_results_table" {
  value       = aws_dynamodb_table.qa_test_results.name
  description = "DynamoDB table for test results"
}
