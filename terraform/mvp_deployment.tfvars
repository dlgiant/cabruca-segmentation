# Auto-generated from .env file
project_name = "cabruca-segmentation"
environment  = "prod"
aws_region   = "sa-east-1"

# Alert configuration
alert_email          = "sanunes.ricardo@gmail.com"
alert_phone          = "+1-917-412-1465"
cost_alert_threshold = 100
cost_threshold       = 400

# Monitoring
monitoring_configuration = {
  enable_cloudwatch  = true
  enable_xray        = true
  log_retention_days = 14
  alarm_email        = "sanunes.ricardo@gmail.com"
  slack_webhook_url  = ""
}

# Use existing mvp.tfvars for other settings
