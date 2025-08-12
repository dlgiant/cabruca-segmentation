# EventBridge Infrastructure for Agent Communication
# Cost-optimized configuration to stay under $10/month
# Pricing: $1 per million custom events published + $0.64 per million events matched by rules

# ==================== EVENT BUS ====================
# Custom Event Bus for Agent Communications
resource "aws_cloudwatch_event_bus" "cabruca_agents" {
  name = "cabruca-agents-bus"

  tags = merge(
    local.common_tags,
    {
      Name       = "cabruca-agents-bus"
      Purpose    = "Multi-agent communication"
      CostCenter = "EventBridge"
    }
  )
}

# Archive for failed events (cheaper than DLQ for low volume)
resource "aws_cloudwatch_event_archive" "agent_events" {
  name             = "${local.app_name}-events-arch"  # Shortened to stay under 48 char limit
  event_source_arn = aws_cloudwatch_event_bus.cabruca_agents.arn
  retention_days   = 7 # Keep failed events for 7 days

  description = "Archive for agent communication events"
}

# ==================== DEAD LETTER QUEUES ====================
# SQS Dead Letter Queue for failed event processing
resource "aws_sqs_queue" "eventbridge_dlq" {
  name                       = "${local.app_name}-eventbridge-dlq"
  message_retention_seconds  = 1209600 # 14 days
  visibility_timeout_seconds = 300     # 5 minutes

  # Cost optimization: Long polling reduces API calls
  receive_wait_time_seconds = 20

  # Enable server-side encryption
  sqs_managed_sse_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.app_name}-eventbridge-dlq"
      Purpose = "Dead letter queue for failed events"
    }
  )
}

# DLQ for critical events that need immediate attention
resource "aws_sqs_queue" "critical_events_dlq" {
  name                       = "${local.app_name}-critical-events-dlq"
  message_retention_seconds  = 345600 # 4 days for faster resolution
  visibility_timeout_seconds = 60     # 1 minute

  receive_wait_time_seconds = 20
  sqs_managed_sse_enabled   = true

  tags = merge(
    local.common_tags,
    {
      Name     = "${local.app_name}-critical-events-dlq"
      Purpose  = "DLQ for critical agent events"
      Priority = "High"
    }
  )
}

# ==================== EVENT SCHEMAS ====================
# Schema Registry for agent communication schemas
resource "aws_schemas_registry" "agent_schemas" {
  name        = "${local.app_name}-agent-schemas"
  description = "Schema registry for agent communication events"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-agent-schemas"
    }
  )
}

# Schema for Issue Detection Events
resource "aws_schemas_schema" "issue_detection" {
  name          = "IssueDetectionEvent"
  registry_name = aws_schemas_registry.agent_schemas.name
  type          = "OpenApi3"
  description   = "Schema for issue detection events from monitoring agents"

  content = jsonencode({
    openapi = "3.0.0"
    info = {
      version = "1.0.0"
      title   = "Issue Detection Event"
    }
    paths = {}
    components = {
      schemas = {
        IssueDetectionEvent = {
          type     = "object"
          required = ["issueId", "timestamp", "severity", "agentId", "issueType", "description"]
          properties = {
            issueId = {
              type        = "string"
              format      = "uuid"
              description = "Unique identifier for the issue"
            }
            timestamp = {
              type        = "string"
              format      = "date-time"
              description = "When the issue was detected"
            }
            severity = {
              type        = "string"
              enum        = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
              description = "Severity level of the issue"
            }
            agentId = {
              type        = "string"
              description = "ID of the agent that detected the issue"
            }
            issueType = {
              type        = "string"
              enum        = ["PERFORMANCE", "ERROR", "SECURITY", "DATA_QUALITY", "INFRASTRUCTURE"]
              description = "Category of the issue"
            }
            description = {
              type        = "string"
              description = "Detailed description of the issue"
            }
            affectedResources = {
              type = "array"
              items = {
                type = "string"
              }
              description = "List of affected resources"
            }
            metrics = {
              type = "object"
              additionalProperties = {
                type = "number"
              }
              description = "Related metrics at the time of detection"
            }
            suggestedActions = {
              type = "array"
              items = {
                type = "string"
              }
              description = "Suggested remediation actions"
            }
          }
        }
      }
    }
  })

  tags = merge(
    local.common_tags,
    {
      Name = "IssueDetectionEventSchema"
    }
  )
}

# Schema for Feature Request Events
resource "aws_schemas_schema" "feature_request" {
  name          = "FeatureRequestEvent"
  registry_name = aws_schemas_registry.agent_schemas.name
  type          = "OpenApi3"
  description   = "Schema for feature request events from product agents"

  content = jsonencode({
    openapi = "3.0.0"
    info = {
      version = "1.0.0"
      title   = "Feature Request Event"
    }
    paths = {}
    components = {
      schemas = {
        FeatureRequestEvent = {
          type     = "object"
          required = ["requestId", "timestamp", "agentId", "featureType", "priority", "description"]
          properties = {
            requestId = {
              type        = "string"
              format      = "uuid"
              description = "Unique identifier for the feature request"
            }
            timestamp = {
              type        = "string"
              format      = "date-time"
              description = "When the request was created"
            }
            agentId = {
              type        = "string"
              description = "ID of the agent creating the request"
            }
            featureType = {
              type        = "string"
              enum        = ["UI", "API", "ML_MODEL", "DATA_PIPELINE", "INTEGRATION", "PERFORMANCE"]
              description = "Type of feature requested"
            }
            priority = {
              type        = "string"
              enum        = ["LOW", "MEDIUM", "HIGH", "URGENT"]
              description = "Priority of the feature request"
            }
            description = {
              type        = "string"
              description = "Detailed description of the feature"
            }
            businessJustification = {
              type        = "string"
              description = "Business case for the feature"
            }
            estimatedImpact = {
              type = "object"
              properties = {
                users = {
                  type        = "integer"
                  description = "Number of users impacted"
                }
                revenue = {
                  type        = "number"
                  description = "Estimated revenue impact"
                }
                efficiency = {
                  type        = "number"
                  description = "Efficiency improvement percentage"
                }
              }
            }
            technicalRequirements = {
              type = "array"
              items = {
                type = "string"
              }
              description = "Technical requirements for implementation"
            }
          }
        }
      }
    }
  })

  tags = merge(
    local.common_tags,
    {
      Name = "FeatureRequestEventSchema"
    }
  )
}

# Schema for Code Change Events
resource "aws_schemas_schema" "code_change" {
  name          = "CodeChangeEvent"
  registry_name = aws_schemas_registry.agent_schemas.name
  type          = "OpenApi3"
  description   = "Schema for code change events from development agents"

  content = jsonencode({
    openapi = "3.0.0"
    info = {
      version = "1.0.0"
      title   = "Code Change Event"
    }
    paths = {}
    components = {
      schemas = {
        CodeChangeEvent = {
          type     = "object"
          required = ["changeId", "timestamp", "agentId", "repository", "branch", "changeType", "status"]
          properties = {
            changeId = {
              type        = "string"
              format      = "uuid"
              description = "Unique identifier for the code change"
            }
            timestamp = {
              type        = "string"
              format      = "date-time"
              description = "When the change was made"
            }
            agentId = {
              type        = "string"
              description = "ID of the agent making the change"
            }
            repository = {
              type        = "string"
              description = "Repository where change was made"
            }
            branch = {
              type        = "string"
              description = "Branch name"
            }
            commitHash = {
              type        = "string"
              description = "Git commit hash"
            }
            changeType = {
              type        = "string"
              enum        = ["FEATURE", "BUGFIX", "REFACTOR", "DOCUMENTATION", "TEST", "DEPENDENCY"]
              description = "Type of code change"
            }
            status = {
              type        = "string"
              enum        = ["PROPOSED", "IN_REVIEW", "APPROVED", "MERGED", "DEPLOYED", "ROLLED_BACK"]
              description = "Current status of the change"
            }
            files = {
              type = "array"
              items = {
                type = "object"
                properties = {
                  path = {
                    type = "string"
                  }
                  additions = {
                    type = "integer"
                  }
                  deletions = {
                    type = "integer"
                  }
                }
              }
              description = "Files affected by the change"
            }
            testResults = {
              type = "object"
              properties = {
                passed = {
                  type = "integer"
                }
                failed = {
                  type = "integer"
                }
                skipped = {
                  type = "integer"
                }
              }
            }
            reviewers = {
              type = "array"
              items = {
                type = "string"
              }
              description = "Agents or humans reviewing the change"
            }
          }
        }
      }
    }
  })

  tags = merge(
    local.common_tags,
    {
      Name = "CodeChangeEventSchema"
    }
  )
}

# Schema for Test Results Events
resource "aws_schemas_schema" "test_results" {
  name          = "TestResultsEvent"
  registry_name = aws_schemas_registry.agent_schemas.name
  type          = "OpenApi3"
  description   = "Schema for test results events from testing agents"

  content = jsonencode({
    openapi = "3.0.0"
    info = {
      version = "1.0.0"
      title   = "Test Results Event"
    }
    paths = {}
    components = {
      schemas = {
        TestResultsEvent = {
          type     = "object"
          required = ["testRunId", "timestamp", "agentId", "testType", "status", "summary"]
          properties = {
            testRunId = {
              type        = "string"
              format      = "uuid"
              description = "Unique identifier for the test run"
            }
            timestamp = {
              type        = "string"
              format      = "date-time"
              description = "When the test was completed"
            }
            agentId = {
              type        = "string"
              description = "ID of the testing agent"
            }
            testType = {
              type        = "string"
              enum        = ["UNIT", "INTEGRATION", "E2E", "PERFORMANCE", "SECURITY", "REGRESSION"]
              description = "Type of test executed"
            }
            status = {
              type        = "string"
              enum        = ["PASSED", "FAILED", "PARTIALLY_PASSED", "SKIPPED", "ERROR"]
              description = "Overall test status"
            }
            summary = {
              type = "object"
              properties = {
                total = {
                  type        = "integer"
                  description = "Total number of tests"
                }
                passed = {
                  type        = "integer"
                  description = "Number of passed tests"
                }
                failed = {
                  type        = "integer"
                  description = "Number of failed tests"
                }
                skipped = {
                  type        = "integer"
                  description = "Number of skipped tests"
                }
                duration = {
                  type        = "number"
                  description = "Total duration in seconds"
                }
              }
            }
            coverage = {
              type = "object"
              properties = {
                lines = {
                  type        = "number"
                  description = "Line coverage percentage"
                }
                branches = {
                  type        = "number"
                  description = "Branch coverage percentage"
                }
                functions = {
                  type        = "number"
                  description = "Function coverage percentage"
                }
              }
            }
            failedTests = {
              type = "array"
              items = {
                type = "object"
                properties = {
                  name = {
                    type = "string"
                  }
                  error = {
                    type = "string"
                  }
                  stackTrace = {
                    type = "string"
                  }
                }
              }
              description = "Details of failed tests"
            }
            artifacts = {
              type = "array"
              items = {
                type = "string"
              }
              description = "Links to test artifacts (logs, screenshots, etc.)"
            }
          }
        }
      }
    }
  })

  tags = merge(
    local.common_tags,
    {
      Name = "TestResultsEventSchema"
    }
  )
}

# ==================== EVENT RULES ====================
# Rule for routing issue detection events to incident management
resource "aws_cloudwatch_event_rule" "issue_detection_routing" {
  name           = "${local.app_name}-issue-detection-routing"
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  description    = "Route issue detection events to appropriate handlers"

  event_pattern = jsonencode({
    source      = ["cabruca.agents.monitor"]
    detail-type = ["Issue Detection Event"]
    detail = {
      severity = [
        {
          exists = true
        }
      ]
    }
  })

  state = "ENABLED"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-issue-detection-routing"
    }
  )
}

# Rule for critical issues that need immediate attention
resource "aws_cloudwatch_event_rule" "critical_issues" {
  name           = "${local.app_name}-critical-issues"
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  description    = "Route critical issues for immediate handling"

  event_pattern = jsonencode({
    source      = ["cabruca.agents.monitor"]
    detail-type = ["Issue Detection Event"]
    detail = {
      severity = ["CRITICAL", "HIGH"]
    }
  })

  state = "ENABLED"

  tags = merge(
    local.common_tags,
    {
      Name     = "${local.app_name}-critical-issues"
      Priority = "High"
    }
  )
}

# Rule for feature request routing to product management
resource "aws_cloudwatch_event_rule" "feature_request_routing" {
  name           = "${local.app_name}-feature-request-routing"
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  description    = "Route feature requests to product management agents"

  event_pattern = jsonencode({
    source      = ["cabruca.agents.product"]
    detail-type = ["Feature Request Event"]
  })

  state = "ENABLED"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-feature-request-routing"
    }
  )
}

# Rule for code change events to trigger CI/CD
resource "aws_cloudwatch_event_rule" "code_change_routing" {
  name           = "${local.app_name}-code-change-routing"
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  description    = "Route code changes to CI/CD pipeline"

  event_pattern = jsonencode({
    source      = ["cabruca.agents.developer"]
    detail-type = ["Code Change Event"]
    detail = {
      status = ["APPROVED", "MERGED"]
    }
  })

  state = "ENABLED"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-code-change-routing"
    }
  )
}

# Rule for test results to trigger notifications
resource "aws_cloudwatch_event_rule" "test_results_routing" {
  name           = "${local.app_name}-test-results-routing"
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  description    = "Route test results to notification agents"

  event_pattern = jsonencode({
    source      = ["cabruca.agents.tester"]
    detail-type = ["Test Results Event"]
  })

  state = "ENABLED"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-test-results-routing"
    }
  )
}

# Rule for failed test results requiring immediate attention
resource "aws_cloudwatch_event_rule" "failed_tests" {
  name           = "${local.app_name}-failed-tests"
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  description    = "Route failed test results for investigation"

  event_pattern = jsonencode({
    source      = ["cabruca.agents.tester"]
    detail-type = ["Test Results Event"]
    detail = {
      status = ["FAILED", "ERROR"]
    }
  })

  state = "ENABLED"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-failed-tests"
    }
  )
}

# ==================== EVENT TARGETS ====================
# CloudWatch Log Group for all agent events (for debugging and audit)
resource "aws_cloudwatch_log_group" "agent_events" {
  name              = "/aws/events/${local.app_name}/agents"
  retention_in_days = 7 # Keep logs for 7 days to minimize costs

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-agent-events-logs"
    }
  )
}

# Target for issue detection events - Log to CloudWatch
resource "aws_cloudwatch_event_target" "issue_detection_logs" {
  rule           = aws_cloudwatch_event_rule.issue_detection_routing.name
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  target_id      = "issue-detection-logs"
  arn            = aws_cloudwatch_log_group.agent_events.arn

  retry_policy {
    maximum_event_age_in_seconds = 3600 # 1 hour
    maximum_retry_attempts       = 3
  }

  dead_letter_config {
    arn = aws_sqs_queue.eventbridge_dlq.arn
  }
}

# Target for critical issues - SNS Topic for alerts
resource "aws_sns_topic" "eventbridge_critical_alerts" {
  name = "${local.app_name}-critical-alerts"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-critical-alerts"
    }
  )
}

resource "aws_cloudwatch_event_target" "critical_issues_sns" {
  rule           = aws_cloudwatch_event_rule.critical_issues.name
  event_bus_name = aws_cloudwatch_event_bus.cabruca_agents.name
  target_id      = "critical-issues-sns"
  arn            = aws_sns_topic.eventbridge_critical_alerts.arn

  retry_policy {
    maximum_event_age_in_seconds = 1800 # 30 minutes
    maximum_retry_attempts       = 5
  }

  dead_letter_config {
    arn = aws_sqs_queue.critical_events_dlq.arn
  }
}

# ==================== IAM PERMISSIONS ====================
# IAM role for EventBridge to write to CloudWatch Logs
resource "aws_iam_role" "eventbridge_logs" {
  name = "${local.app_name}-eventbridge-logs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-eventbridge-logs-role"
    }
  )
}

resource "aws_iam_role_policy" "eventbridge_logs" {
  name = "${local.app_name}-eventbridge-logs-policy"
  role = aws_iam_role.eventbridge_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.agent_events.arn}:*"
      }
    ]
  })
}

# IAM policy for agents to publish events
resource "aws_iam_policy" "agent_event_publisher" {
  name        = "${local.app_name}-agent-event-publisher"
  description = "Allow agents to publish events to the custom event bus"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "events:PutEvents"
        ]
        Resource = aws_cloudwatch_event_bus.cabruca_agents.arn
      },
      {
        Effect = "Allow"
        Action = [
          "schemas:DescribeRegistry",
          "schemas:DescribeSchema",
          "schemas:GetDiscoveredSchema"
        ]
        Resource = [
          aws_schemas_registry.agent_schemas.arn,
          "${aws_schemas_registry.agent_schemas.arn}/*"
        ]
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-agent-event-publisher"
    }
  )
}

# ==================== COST MONITORING ====================
# CloudWatch Alarm for EventBridge cost monitoring
resource "aws_cloudwatch_metric_alarm" "eventbridge_cost_alarm" {
  alarm_name          = "${local.app_name}-eventbridge-cost-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "SuccessfulEventsMatched"
  namespace           = "AWS/Events"
  period              = "86400" # 1 day in seconds
  statistic           = "Sum"
  threshold           = "9000000" # Alert if approaching 10 million events/month
  alarm_description   = "Alert when EventBridge usage approaches $10/month limit"
  alarm_actions       = var.monitoring_configuration.alarm_email != "" ? [aws_sns_topic.eventbridge_critical_alerts.arn] : []

  dimensions = {
    EventBusName = aws_cloudwatch_event_bus.cabruca_agents.name
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-eventbridge-cost-alarm"
    }
  )
}

# CloudWatch Dashboard for EventBridge monitoring
resource "aws_cloudwatch_dashboard" "eventbridge_monitoring" {
  dashboard_name = "${local.app_name}-eventbridge-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/Events", "SuccessfulEventsMatched", { stat = "Sum", label = "Matched Events" }],
            [".", "FailedInvocations", { stat = "Sum", label = "Failed Events" }],
            [".", "InvocationAttempts", { stat = "Sum", label = "Total Attempts" }]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "EventBridge Activity"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/SQS", "NumberOfMessagesSent", { stat = "Sum", label = "DLQ Messages" }]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "Dead Letter Queue Activity"
        }
      }
    ]
  })
}

# ==================== OUTPUTS ====================
output "event_bus_name" {
  description = "Name of the custom event bus for agent communication"
  value       = aws_cloudwatch_event_bus.cabruca_agents.name
}

output "event_bus_arn" {
  description = "ARN of the custom event bus"
  value       = aws_cloudwatch_event_bus.cabruca_agents.arn
}

output "schema_registry_name" {
  description = "Name of the schema registry for agent events"
  value       = aws_schemas_registry.agent_schemas.name
}

output "dlq_url" {
  description = "URL of the main dead letter queue"
  value       = aws_sqs_queue.eventbridge_dlq.url
}

output "critical_dlq_url" {
  description = "URL of the critical events dead letter queue"
  value       = aws_sqs_queue.critical_events_dlq.url
}

output "agent_event_publisher_policy_arn" {
  description = "ARN of the IAM policy for agents to publish events"
  value       = aws_iam_policy.agent_event_publisher.arn
}

output "critical_alerts_topic_arn" {
  description = "ARN of the SNS topic for critical alerts"
  value       = aws_sns_topic.eventbridge_critical_alerts.arn
}
