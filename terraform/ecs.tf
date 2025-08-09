# ECS Configuration for Cabruca Segmentation
# Optimized for Brazil deployment with auto-scaling

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${local.app_name}-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name = "${local.app_name}-cluster"
  }
}

# ECS Task Execution Role
resource "aws_iam_role" "ecs_task_execution" {
  name = "${local.app_name}-ecs-task-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "${local.app_name}-ecs-task-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Role
resource "aws_iam_role" "ecs_task" {
  name = "${local.app_name}-ecs-task-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "${local.app_name}-ecs-task-role"
  }
}

# IAM Policy for S3 access
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${local.app_name}-ecs-task-s3-policy"
  role = aws_iam_role.ecs_task.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*",
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*"
        ]
      }
    ]
  })
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.app_name}/api"
  retention_in_days = 30
  
  tags = {
    Name = "${local.app_name}-api-logs"
  }
}

resource "aws_cloudwatch_log_group" "streamlit" {
  name              = "/ecs/${local.app_name}/streamlit"
  retention_in_days = 30
  
  tags = {
    Name = "${local.app_name}-streamlit-logs"
  }
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/ecs/${local.app_name}/inference"
  retention_in_days = 30
  
  tags = {
    Name = "${local.app_name}-inference-logs"
  }
}

# ECR Repository for Docker images
resource "aws_ecr_repository" "main" {
  name                 = local.app_name
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
  
  tags = {
    Name = "${local.app_name}-ecr"
  }
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "main" {
  repository = aws_ecr_repository.main.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "any"
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# API Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.app_name}-api"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = "2048"  # 2 vCPU
  memory                  = "4096"  # 4 GB
  execution_role_arn      = aws_iam_role.ecs_task_execution.arn
  task_role_arn          = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "api"
      image = "${aws_ecr_repository.main.repository_url}:api-latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = local.api_port
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "MODEL_PATH"
          value = "/app/models/checkpoint_best.pth"
        },
        {
          name  = "DEVICE"
          value = "cpu"
        },
        {
          name  = "API_HOST"
          value = "0.0.0.0"
        },
        {
          name  = "API_PORT"
          value = tostring(local.api_port)
        },
        {
          name  = "S3_MODEL_BUCKET"
          value = aws_s3_bucket.models.id
        },
        {
          name  = "S3_DATA_BUCKET"
          value = aws_s3_bucket.data.id
        },
        {
          name  = "REDIS_URL"
          value = var.enable_elasticache ? "redis://${aws_elasticache_replication_group.main[0].primary_endpoint_address}:6379" : ""
        },
        {
          name  = "DATABASE_URL"
          value = var.enable_rds ? "postgresql://cabruca_admin:${random_password.db_password[0].result}@${aws_db_instance.main[0].endpoint}/cabruca" : ""
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${local.api_port}/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
  
  tags = {
    Name = "${local.app_name}-api-task"
  }
}

# Streamlit Task Definition
resource "aws_ecs_task_definition" "streamlit" {
  family                   = "${local.app_name}-streamlit"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = "2048"  # 2 vCPU
  memory                  = "4096"  # 4 GB
  execution_role_arn      = aws_iam_role.ecs_task_execution.arn
  task_role_arn          = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "streamlit"
      image = "${aws_ecr_repository.main.repository_url}:streamlit-latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = local.streamlit_port
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "MODEL_PATH"
          value = "/app/models/checkpoint_best.pth"
        },
        {
          name  = "API_URL"
          value = "http://localhost:${local.api_port}"
        },
        {
          name  = "S3_MODEL_BUCKET"
          value = aws_s3_bucket.models.id
        },
        {
          name  = "S3_DATA_BUCKET"
          value = aws_s3_bucket.data.id
        }
      ]
      
      command = [
        "streamlit",
        "run",
        "src/inference/interactive_viewer.py",
        "--server.port=${local.streamlit_port}",
        "--server.address=0.0.0.0",
        "--server.headless=true"
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.streamlit.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "streamlit"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${local.streamlit_port}/ || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
  
  tags = {
    Name = "${local.app_name}-streamlit-task"
  }
}

# GPU Inference Task Definition
resource "aws_ecs_task_definition" "inference" {
  family                   = "${local.app_name}-inference"
  network_mode            = "awsvpc"
  requires_compatibilities = ["EC2"]
  cpu                     = "4096"  # 4 vCPU
  memory                  = "16384" # 16 GB
  execution_role_arn      = aws_iam_role.ecs_task_execution.arn
  task_role_arn          = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "inference"
      image = "${aws_ecr_repository.main.repository_url}:inference-latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = local.api_port
          protocol      = "tcp"
        }
      ]
      
      resourceRequirements = var.enable_gpu ? [
        {
          type  = "GPU"
          value = "1"
        }
      ] : []
      
      environment = [
        {
          name  = "MODEL_PATH"
          value = "/app/models/checkpoint_best.pth"
        },
        {
          name  = "DEVICE"
          value = var.enable_gpu ? "cuda" : "cpu"
        },
        {
          name  = "BATCH_SIZE"
          value = "8"
        },
        {
          name  = "S3_MODEL_BUCKET"
          value = aws_s3_bucket.models.id
        },
        {
          name  = "S3_DATA_BUCKET"
          value = aws_s3_bucket.data.id
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.inference.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "inference"
        }
      }
    }
  ])
  
  tags = {
    Name = "${local.app_name}-inference-task"
  }
}

# ECS Services
resource "aws_ecs_service" "api" {
  name            = "${local.app_name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_min_instances
  launch_type     = "FARGATE"
  
  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets         = aws_subnet.private[*].id
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = local.api_port
  }
  
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200
  
  depends_on = [
    aws_lb_listener.https,
    aws_iam_role_policy.ecs_task_s3
  ]
  
  tags = {
    Name = "${local.app_name}-api-service"
  }
}

resource "aws_ecs_service" "streamlit" {
  name            = "${local.app_name}-streamlit"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.streamlit.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets         = aws_subnet.private[*].id
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.streamlit.arn
    container_name   = "streamlit"
    container_port   = local.streamlit_port
  }
  
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200
  
  depends_on = [
    aws_lb_listener.https,
    aws_iam_role_policy.ecs_task_s3
  ]
  
  tags = {
    Name = "${local.app_name}-streamlit-service"
  }
}

# Auto Scaling for API Service
resource "aws_appautoscaling_target" "api" {
  max_capacity       = var.api_max_instances
  min_capacity       = var.api_min_instances
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "${local.app_name}-api-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

resource "aws_appautoscaling_policy" "api_memory" {
  name               = "${local.app_name}-api-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    
    target_value       = 80.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# Auto Scaling for Streamlit Service
resource "aws_appautoscaling_target" "streamlit" {
  max_capacity       = 5
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.streamlit.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "streamlit_cpu" {
  name               = "${local.app_name}-streamlit-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.streamlit.resource_id
  scalable_dimension = aws_appautoscaling_target.streamlit.scalable_dimension
  service_namespace  = aws_appautoscaling_target.streamlit.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}