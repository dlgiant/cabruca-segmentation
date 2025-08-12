# Monitoring and Alerting Configuration for Brazil deployment
# CloudWatch, SNS, and alerting optimized for Northeast Brazil operations

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "${local.app_name}-alerts"
  
  tags = {
    Name = "${local.app_name}-alerts"
  }
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.monitoring_configuration.alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.monitoring_configuration.alarm_email
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${local.app_name}-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", { stat = "Average" }],
            [".", "MemoryUtilization", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Cluster Utilization"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", { stat = "Average" }],
            [".", "RequestCount", { stat = "Sum" }],
            [".", "HTTPCode_Target_2XX_Count", { stat = "Sum" }],
            [".", "HTTPCode_Target_5XX_Count", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "API Performance"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", { stat = "Average" }],
            [".", "DatabaseMemoryUsagePercentage", { stat = "Average" }],
            [".", "CacheHits", { stat = "Sum" }],
            [".", "CacheMisses", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Redis Cache Performance"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", { stat = "Average" }],
            [".", "DatabaseConnections", { stat = "Average" }],
            [".", "FreeableMemory", { stat = "Average" }],
            [".", "ReadLatency", { stat = "Average" }],
            [".", "WriteLatency", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Database Performance"
        }
      }
    ]
  })
}

# CloudWatch Alarms

# High CPU Utilization
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${local.app_name}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "CPUUtilization"
  namespace          = "AWS/ECS"
  period             = "300"
  statistic          = "Average"
  threshold          = "80"
  alarm_description  = "This metric monitors ECS CPU utilization"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
  }
  
  tags = {
    Name = "${local.app_name}-high-cpu-alarm"
  }
}

# High Memory Utilization
resource "aws_cloudwatch_metric_alarm" "high_memory" {
  alarm_name          = "${local.app_name}-high-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "MemoryUtilization"
  namespace          = "AWS/ECS"
  period             = "300"
  statistic          = "Average"
  threshold          = "85"
  alarm_description  = "This metric monitors ECS memory utilization"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
  }
  
  tags = {
    Name = "${local.app_name}-high-memory-alarm"
  }
}

# API Response Time
resource "aws_cloudwatch_metric_alarm" "api_response_time" {
  alarm_name          = "${local.app_name}-api-response-time"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "TargetResponseTime"
  namespace          = "AWS/ApplicationELB"
  period             = "300"
  statistic          = "Average"
  threshold          = "2"  # 2 seconds
  alarm_description  = "API response time is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }
  
  tags = {
    Name = "${local.app_name}-api-response-time-alarm"
  }
}

# API Error Rate
resource "aws_cloudwatch_metric_alarm" "api_error_rate" {
  alarm_name          = "${local.app_name}-api-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  threshold          = "10"
  alarm_description  = "API error rate is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  metric_query {
    id          = "e1"
    expression  = "m2/m1*100"
    label       = "Error Rate"
    return_data = true
  }
  
  metric_query {
    id = "m1"
    metric {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      period      = "300"
      stat        = "Sum"
      
      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
  }
  
  metric_query {
    id = "m2"
    metric {
      metric_name = "HTTPCode_Target_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      period      = "300"
      stat        = "Sum"
      
      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
  }
  
  tags = {
    Name = "${local.app_name}-api-error-rate-alarm"
  }
}

# Database Connection Count
resource "aws_cloudwatch_metric_alarm" "db_connections" {
  count               = var.enable_rds ? 1 : 0
  alarm_name          = "${local.app_name}-db-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "DatabaseConnections"
  namespace          = "AWS/RDS"
  period             = "300"
  statistic          = "Average"
  threshold          = "80"
  alarm_description  = "Database connection count is high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main[0].id
  }
  
  tags = {
    Name = "${local.app_name}-db-connections-alarm"
  }
}

# Cache Hit Rate
resource "aws_cloudwatch_metric_alarm" "cache_hit_rate" {
  count               = var.enable_elasticache ? 1 : 0
  alarm_name          = "${local.app_name}-cache-hit-rate"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  threshold          = "80"  # 80% hit rate
  alarm_description  = "Cache hit rate is low"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  metric_query {
    id          = "e1"
    expression  = "m1/(m1+m2)*100"
    label       = "Hit Rate"
    return_data = true
  }
  
  metric_query {
    id = "m1"
    metric {
      metric_name = "CacheHits"
      namespace   = "AWS/ElastiCache"
      period      = "300"
      stat        = "Sum"
      
      dimensions = {
        CacheClusterId = aws_elasticache_replication_group.main[0].id
      }
    }
  }
  
  metric_query {
    id = "m2"
    metric {
      metric_name = "CacheMisses"
      namespace   = "AWS/ElastiCache"
      period      = "300"
      stat        = "Sum"
      
      dimensions = {
        CacheClusterId = aws_elasticache_replication_group.main[0].id
      }
    }
  }
  
  tags = {
    Name = "${local.app_name}-cache-hit-rate-alarm"
  }
}

# Custom Metrics for ML Model Performance
resource "aws_cloudwatch_log_metric_filter" "inference_latency" {
  name           = "${local.app_name}-inference-latency"
  log_group_name = aws_cloudwatch_log_group.inference.name
  pattern        = "[time, request_id, latency_label=\"INFERENCE_LATENCY\", latency_value, ...]"
  
  metric_transformation {
    name      = "InferenceLatency"
    namespace = "${local.app_name}/ML"
    value     = "$latency_value"
    unit      = "Milliseconds"
  }
}

resource "aws_cloudwatch_log_metric_filter" "model_accuracy" {
  name           = "${local.app_name}-model-accuracy"
  log_group_name = aws_cloudwatch_log_group.inference.name
  pattern        = "[time, request_id, accuracy_label=\"MODEL_ACCURACY\", accuracy_value, ...]"
  
  metric_transformation {
    name      = "ModelAccuracy"
    namespace = "${local.app_name}/ML"
    value     = "$accuracy_value"
    unit      = "Percent"
  }
}

resource "aws_cloudwatch_log_metric_filter" "trees_detected" {
  name           = "${local.app_name}-trees-detected"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "[time, request_id, trees_label=\"TREES_DETECTED\", trees_count, ...]"
  
  metric_transformation {
    name      = "TreesDetected"
    namespace = "${local.app_name}/ML"
    value     = "$trees_count"
    unit      = "Count"
  }
}

# CloudWatch Logs Insights Queries
resource "aws_cloudwatch_query_definition" "api_errors" {
  name = "${local.app_name}-api-errors"
  
  log_group_names = [
    aws_cloudwatch_log_group.api.name
  ]
  
  query_string = <<EOF
fields @timestamp, @message
| filter @message like /ERROR/
| stats count() by bin(5m)
EOF
}

resource "aws_cloudwatch_query_definition" "slow_requests" {
  name = "${local.app_name}-slow-requests"
  
  log_group_names = [
    aws_cloudwatch_log_group.api.name
  ]
  
  query_string = <<EOF
fields @timestamp, @message, @duration
| filter @duration > 1000
| sort @timestamp desc
| limit 20
EOF
}

resource "aws_cloudwatch_query_definition" "inference_performance" {
  name = "${local.app_name}-inference-performance"
  
  log_group_names = [
    aws_cloudwatch_log_group.inference.name
  ]
  
  query_string = <<EOF
fields @timestamp, @message
| parse @message "Inference time: * ms" as inference_time
| stats avg(inference_time) as avg_time, 
        max(inference_time) as max_time,
        min(inference_time) as min_time
    by bin(5m)
EOF
}

# Cost Monitoring
resource "aws_budgets_budget" "monthly" {
  count             = var.monitoring_configuration.alarm_email != "" ? 1 : 0
  name              = "${local.app_name}-monthly-budget"
  budget_type       = "COST"
  limit_amount      = "1000"  # Adjust based on your budget
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2024-01-01_00:00"
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.monitoring_configuration.alarm_email]
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = var.monitoring_configuration.alarm_email != "" ? [var.monitoring_configuration.alarm_email] : []
  }
}

# X-Ray Tracing (if enabled)
resource "aws_xray_sampling_rule" "main" {
  count        = var.monitoring_configuration.enable_xray ? 1 : 0
  rule_name    = "${local.app_name}-sampling"
  priority     = 9000
  version      = 1
  reservoir_size = 1
  fixed_rate   = 0.05  # 5% sampling
  url_path     = "*"
  host         = "*"
  http_method  = "*"
  service_type = "*"
  service_name = "*"
  resource_arn = "*"
}