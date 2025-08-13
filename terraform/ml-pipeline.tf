# ML Pipeline Configuration for Training and Annotation
# Provides compute resources for ML workloads while minimizing costs

# SageMaker Notebook Instance for Training and Annotation (On-Demand)
resource "aws_sagemaker_notebook_instance" "ml_workspace" {
  count               = var.enable_ml_pipeline ? 1 : 0
  name                = "${local.app_name}-ml-workspace"
  instance_type       = var.ml_instance_type
  platform_identifier = "notebook-al2-v2"
  role_arn            = aws_iam_role.sagemaker_role[0].arn
  root_access         = "Enabled"
  volume_size         = 30 # GB for model storage

  default_code_repository = "https://github.com/dlgiant/cabruca-segmentation.git"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.app_name}-ml-workspace"
      Type = "MLPipeline"
    }
  )
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  count = var.enable_ml_pipeline ? 1 : 0
  name  = "${local.app_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${local.app_name}-sagemaker-role"
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  count      = var.enable_ml_pipeline ? 1 : 0
  role       = aws_iam_role.sagemaker_role[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  count = var.enable_ml_pipeline ? 1 : 0
  name  = "${local.app_name}-sagemaker-s3-policy"
  role  = aws_iam_role.sagemaker_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*",
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*",
          "${aws_s3_bucket.ml_data[0].arn}",
          "${aws_s3_bucket.ml_data[0].arn}/*"
        ]
      }
    ]
  })
}

# S3 Bucket for ML Data (Training datasets, annotations)
resource "aws_s3_bucket" "ml_data" {
  count  = var.enable_ml_pipeline ? 1 : 0
  bucket = "${local.app_name}-ml-data"

  tags = {
    Name = "${local.app_name}-ml-data"
    Type = "MLData"
  }
}

resource "aws_s3_bucket_versioning" "ml_data" {
  count  = var.enable_ml_pipeline ? 1 : 0
  bucket = aws_s3_bucket.ml_data[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

# Lambda Function for Batch Processing
resource "aws_lambda_function" "batch_processor" {
  count         = var.enable_ml_pipeline ? 1 : 0
  filename      = "lambda_batch_processor.zip"
  function_name = "${local.app_name}-batch-processor"
  role          = aws_iam_role.lambda_role[0].arn
  handler       = "index.handler"
  runtime       = "python3.9"
  timeout       = 900  # 15 minutes max
  memory_size   = 3008 # 3GB memory

  environment {
    variables = {
      S3_BUCKET    = aws_s3_bucket.ml_data[0].id
      MODEL_BUCKET = aws_s3_bucket.models.id
      ENVIRONMENT  = var.environment
    }
  }

  tags = {
    Name = "${local.app_name}-batch-processor"
  }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  count = var.enable_ml_pipeline ? 1 : 0
  name  = "${local.app_name}-lambda-role"

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

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  count      = var.enable_ml_pipeline ? 1 : 0
  role       = aws_iam_role.lambda_role[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ECS Task Definition for Training Jobs
resource "aws_ecs_task_definition" "training" {
  count                    = var.enable_ml_pipeline ? 1 : 0
  family                   = "${local.app_name}-training"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.training_cpu    # Configurable
  memory                   = var.training_memory # Configurable
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "training"
      image = "${aws_ecr_repository.main.repository_url}:training-latest"

      essential = true

      environment = [
        {
          name  = "S3_DATA_BUCKET"
          value = aws_s3_bucket.ml_data[0].id
        },
        {
          name  = "S3_MODEL_BUCKET"
          value = aws_s3_bucket.models.id
        },
        {
          name  = "DEVICE"
          value = "cpu" # CPU for MVP
        },
        {
          name  = "BATCH_SIZE"
          value = "4"
        },
        {
          name  = "EPOCHS"
          value = "10"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.training[0].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "training"
        }
      }
    }
  ])

  tags = {
    Name = "${local.app_name}-training-task"
  }
}

# CloudWatch Log Group for Training
resource "aws_cloudwatch_log_group" "training" {
  count             = var.enable_ml_pipeline ? 1 : 0
  name              = "/ecs/${local.app_name}/training"
  retention_in_days = 7 # Short retention for MVP

  tags = {
    Name = "${local.app_name}-training-logs"
  }
}

# ECS Task Definition for Annotation Service
resource "aws_ecs_task_definition" "annotation" {
  count                    = var.enable_ml_pipeline ? 1 : 0
  family                   = "${local.app_name}-annotation"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"  # 0.5 vCPU
  memory                   = "1024" # 1 GB
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "annotation"
      image = "${aws_ecr_repository.main.repository_url}:annotation-latest"

      essential = true

      portMappings = [
        {
          containerPort = 8502 # Streamlit annotation tool
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "S3_DATA_BUCKET"
          value = aws_s3_bucket.ml_data[0].id
        },
        {
          name  = "S3_MODEL_BUCKET"
          value = aws_s3_bucket.models.id
        },
        {
          name  = "SAM_MODEL_PATH"
          value = "/app/models/sam_vit_h.pth"
        }
      ]

      command = [
        "streamlit",
        "run",
        "annotation_tools/streamlit_app/annotation_app.py",
        "--server.port=8502",
        "--server.address=0.0.0.0"
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.annotation[0].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "annotation"
        }
      }
    }
  ])

  tags = {
    Name = "${local.app_name}-annotation-task"
  }
}

# CloudWatch Log Group for Annotation
resource "aws_cloudwatch_log_group" "annotation" {
  count             = var.enable_ml_pipeline ? 1 : 0
  name              = "/ecs/${local.app_name}/annotation"
  retention_in_days = 7

  tags = {
    Name = "${local.app_name}-annotation-logs"
  }
}

# ECS Service for Annotation Tool (On-Demand)
resource "aws_ecs_service" "annotation" {
  count           = var.enable_ml_pipeline ? 1 : 0
  name            = "${local.app_name}-annotation"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.annotation[0].arn
  desired_count   = var.annotation_enabled ? 1 : 0 # Start only when needed
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.annotation[0].arn
    container_name   = "annotation"
    container_port   = 8502
  }

  depends_on = [
    aws_lb_listener.https,
    aws_iam_role_policy.ecs_task_s3
  ]

  tags = {
    Name = "${local.app_name}-annotation-service"
  }
}

# Target Group for Annotation Service
resource "aws_lb_target_group" "annotation" {
  count       = var.enable_ml_pipeline ? 1 : 0
  name        = "${local.app_name}-annotation-tg"
  port        = 8502
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/"
    matcher             = "200"
  }

  tags = {
    Name = "${local.app_name}-annotation-tg"
  }
}

# Listener Rule for Annotation Service
resource "aws_lb_listener_rule" "annotation" {
  count        = var.enable_ml_pipeline ? 1 : 0
  listener_arn = var.domain_name != "" ? aws_lb_listener.https[0].arn : aws_lb_listener.http.arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.annotation[0].arn
  }

  condition {
    path_pattern {
      values = ["/annotation*"]
    }
  }
}

# Step Functions for ML Pipeline Orchestration
resource "aws_sfn_state_machine" "ml_pipeline" {
  count    = var.enable_ml_pipeline ? 1 : 0
  name     = "${local.app_name}-ml-pipeline"
  role_arn = aws_iam_role.step_functions[0].arn

  definition = jsonencode({
    Comment = "ML Pipeline for training and evaluation"
    StartAt = "PrepareData"
    States = {
      PrepareData = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          TaskDefinition = aws_ecs_task_definition.training[0].arn
          Cluster        = aws_ecs_cluster.main.arn
          Overrides = {
            ContainerOverrides = [
              {
                Name = "training"
                Environment = [
                  {
                    Name  = "TASK_TYPE"
                    Value = "prepare_data"
                  }
                ]
              }
            ]
          }
        }
        Next = "TrainModel"
      }
      TrainModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          TaskDefinition = aws_ecs_task_definition.training[0].arn
          Cluster        = aws_ecs_cluster.main.arn
          Overrides = {
            ContainerOverrides = [
              {
                Name = "training"
                Environment = [
                  {
                    Name  = "TASK_TYPE"
                    Value = "train"
                  }
                ]
              }
            ]
          }
        }
        Next = "EvaluateModel"
      }
      EvaluateModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          TaskDefinition = aws_ecs_task_definition.training[0].arn
          Cluster        = aws_ecs_cluster.main.arn
          Overrides = {
            ContainerOverrides = [
              {
                Name = "training"
                Environment = [
                  {
                    Name  = "TASK_TYPE"
                    Value = "evaluate"
                  }
                ]
              }
            ]
          }
        }
        End = true
      }
    }
  })

  tags = {
    Name = "${local.app_name}-ml-pipeline"
  }
}

# IAM Role for Step Functions
resource "aws_iam_role" "step_functions" {
  count = var.enable_ml_pipeline ? 1 : 0
  name  = "${local.app_name}-step-functions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "step_functions" {
  count = var.enable_ml_pipeline ? 1 : 0
  name  = "${local.app_name}-step-functions-policy"
  role  = aws_iam_role.step_functions[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecs:RunTask",
          "ecs:StopTask",
          "ecs:DescribeTasks",
          "iam:PassRole"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution",
          "states:DescribeExecution",
          "states:StopExecution"
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
        Resource = "*"
      }
    ]
  })
}

# EventBridge Rule for Scheduled Training
resource "aws_cloudwatch_event_rule" "training_schedule" {
  count               = var.enable_ml_pipeline && var.enable_scheduled_training ? 1 : 0
  name                = "${local.app_name}-training-schedule"
  description         = "Trigger ML training pipeline"
  schedule_expression = var.training_schedule # e.g., "rate(7 days)"

  tags = {
    Name = "${local.app_name}-training-schedule"
  }
}

resource "aws_cloudwatch_event_target" "training_target" {
  count     = var.enable_ml_pipeline && var.enable_scheduled_training ? 1 : 0
  rule      = aws_cloudwatch_event_rule.training_schedule[0].name
  target_id = "TriggerMLPipeline"
  arn       = aws_sfn_state_machine.ml_pipeline[0].arn
  role_arn  = aws_iam_role.eventbridge[0].arn
}

# IAM Role for EventBridge
resource "aws_iam_role" "eventbridge" {
  count = var.enable_ml_pipeline && var.enable_scheduled_training ? 1 : 0
  name  = "${local.app_name}-eventbridge-role"

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
}

resource "aws_iam_role_policy" "eventbridge" {
  count = var.enable_ml_pipeline && var.enable_scheduled_training ? 1 : 0
  name  = "${local.app_name}-eventbridge-policy"
  role  = aws_iam_role.eventbridge[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution"
        ]
        Resource = aws_sfn_state_machine.ml_pipeline[0].arn
      }
    ]
  })
}