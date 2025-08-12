"""
Circuit Breaker Lambda function for cost control
Monitors AWS costs and automatically halts agents if spending exceeds thresholds
"""

import json
import os
import boto3
from datetime import datetime, timedelta, timezone
import logging
from decimal import Decimal

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ce_client = boto3.client('ce')  # Cost Explorer
lambda_client = boto3.client('lambda')
events_client = boto3.client('events')
sns_client = boto3.client('sns')
cloudwatch_client = boto3.client('cloudwatch')

def lambda_handler(event, context):
    """
    Main handler for cost circuit breaker
    """
    environment = os.environ.get('ENVIRONMENT', 'production')
    monthly_limit = float(os.environ.get('COST_LIMIT', '500'))
    daily_limit = float(os.environ.get('DAILY_LIMIT', '17'))
    
    logger.info(f"Circuit breaker check for environment: {environment}")
    
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'environment': environment,
        'limits': {
            'monthly': monthly_limit,
            'daily': daily_limit
        },
        'costs': {},
        'actions': []
    }
    
    try:
        # Get current costs
        current_costs = get_current_costs()
        results['costs'] = current_costs
        
        # Check if we need to trigger the circuit breaker
        if should_trigger_circuit_breaker(current_costs, monthly_limit, daily_limit):
            logger.warning("Circuit breaker triggered! Halting agents.")
            results['circuit_breaker_triggered'] = True
            
            # Halt all agent functions
            halt_results = halt_agent_functions()
            results['actions'].extend(halt_results)
            
            # Send critical alert
            send_alert(current_costs, monthly_limit, daily_limit)
            
            # Disable EventBridge rules to prevent further triggers
            disable_results = disable_event_rules()
            results['actions'].extend(disable_results)
            
        else:
            # Check if we can re-enable previously halted functions
            if should_reenable_functions(current_costs, monthly_limit, daily_limit):
                logger.info("Costs back under control. Re-enabling functions.")
                results['circuit_breaker_reset'] = True
                
                reenable_results = reenable_agent_functions()
                results['actions'].extend(reenable_results)
                
                # Re-enable EventBridge rules
                enable_results = enable_event_rules()
                results['actions'].extend(enable_results)
        
        # Send cost metrics to CloudWatch
        send_cost_metrics(current_costs)
        
    except Exception as e:
        logger.error(f"Error in circuit breaker: {str(e)}")
        results['error'] = str(e)
        
    logger.info(f"Circuit breaker check completed: {json.dumps(results, default=str)}")
    return {
        'statusCode': 200,
        'body': json.dumps(results, default=str)
    }

def get_current_costs():
    """
    Get current AWS costs from Cost Explorer
    """
    costs = {}
    
    # Get month-to-date costs
    today = datetime.now(timezone.utc).date()
    month_start = today.replace(day=1)
    
    try:
        # Monthly costs
        monthly_response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': month_start.isoformat(),
                'End': (today + timedelta(days=1)).isoformat()
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            Filter={
                'Tags': {
                    'Key': 'CostCenter',
                    'Values': ['engineering', 'qa', 'operations', 'monitoring']
                }
            }
        )
        
        if monthly_response['ResultsByTime']:
            costs['month_to_date'] = float(
                monthly_response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount']
            )
        
        # Daily costs (last 24 hours)
        daily_response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': (today - timedelta(days=1)).isoformat(),
                'End': today.isoformat()
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                'Tags': {
                    'Key': 'CostCenter',
                    'Values': ['engineering', 'qa', 'operations', 'monitoring']
                }
            }
        )
        
        if daily_response['ResultsByTime']:
            costs['last_24_hours'] = float(
                daily_response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount']
            )
        
        # Get forecast for the month
        forecast_response = ce_client.get_cost_forecast(
            TimePeriod={
                'Start': (today + timedelta(days=1)).isoformat(),
                'End': (month_start.replace(month=month_start.month + 1) if month_start.month < 12 
                       else month_start.replace(year=month_start.year + 1, month=1)).isoformat()
            },
            Metric='UNBLENDED_COST',
            Granularity='MONTHLY'
        )
        
        costs['monthly_forecast'] = float(forecast_response['Total']['Amount'])
        
        # Calculate projected total
        costs['projected_monthly_total'] = costs.get('month_to_date', 0) + costs.get('monthly_forecast', 0)
        
        # Get costs by service
        service_response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': month_start.isoformat(),
                'End': (today + timedelta(days=1)).isoformat()
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ]
        )
        
        costs['by_service'] = {}
        if service_response['ResultsByTime']:
            for group in service_response['ResultsByTime'][0]['Groups']:
                service_name = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                if amount > 0.01:  # Only include services with meaningful costs
                    costs['by_service'][service_name] = amount
        
    except Exception as e:
        logger.error(f"Error getting costs from Cost Explorer: {str(e)}")
        # Fallback to CloudWatch billing metrics
        costs = get_cloudwatch_billing_metrics()
    
    return costs

def get_cloudwatch_billing_metrics():
    """
    Fallback method to get billing metrics from CloudWatch
    """
    costs = {}
    
    try:
        response = cloudwatch_client.get_metric_statistics(
            Namespace='AWS/Billing',
            MetricName='EstimatedCharges',
            Dimensions=[
                {'Name': 'Currency', 'Value': 'USD'}
            ],
            StartTime=datetime.now(timezone.utc) - timedelta(days=1),
            EndTime=datetime.now(timezone.utc),
            Period=86400,
            Statistics=['Maximum']
        )
        
        if response['Datapoints']:
            costs['estimated_monthly'] = response['Datapoints'][-1]['Maximum']
            
    except Exception as e:
        logger.error(f"Error getting CloudWatch billing metrics: {str(e)}")
    
    return costs

def should_trigger_circuit_breaker(costs, monthly_limit, daily_limit):
    """
    Determine if circuit breaker should be triggered
    """
    # Trigger if projected monthly costs exceed limit
    if costs.get('projected_monthly_total', 0) > monthly_limit:
        logger.warning(f"Projected monthly total ${costs.get('projected_monthly_total', 0):.2f} exceeds limit ${monthly_limit}")
        return True
    
    # Trigger if daily costs are unusually high (3x normal daily rate)
    if costs.get('last_24_hours', 0) > (daily_limit * 3):
        logger.warning(f"Daily costs ${costs.get('last_24_hours', 0):.2f} exceed 3x normal rate ${daily_limit * 3:.2f}")
        return True
    
    # Check for sudden cost spikes in specific services
    lambda_costs = costs.get('by_service', {}).get('AWS Lambda', 0)
    if lambda_costs > (monthly_limit * 0.5):  # Lambda shouldn't be more than 50% of budget
        logger.warning(f"Lambda costs ${lambda_costs:.2f} exceed 50% of monthly budget")
        return True
    
    return False

def should_reenable_functions(costs, monthly_limit, daily_limit):
    """
    Determine if it's safe to re-enable halted functions
    """
    # Only re-enable if we're well below limits (70% threshold)
    if costs.get('projected_monthly_total', 0) < (monthly_limit * 0.7):
        if costs.get('last_24_hours', 0) < (daily_limit * 1.5):
            return True
    
    return False

def halt_agent_functions():
    """
    Halt all agent Lambda functions by setting concurrency to 0
    """
    actions = []
    
    function_names = [
        os.environ.get('MANAGER_FUNCTION'),
        os.environ.get('ENGINEER_FUNCTION'),
        os.environ.get('QA_FUNCTION')
    ]
    
    for func_name in function_names:
        if func_name:
            try:
                # Set reserved concurrency to 0
                lambda_client.put_function_concurrency(
                    FunctionName=func_name,
                    ReservedConcurrentExecutions=0
                )
                
                actions.append({
                    'action': 'halt_function',
                    'function': func_name,
                    'status': 'success'
                })
                logger.info(f"Halted function: {func_name}")
                
            except Exception as e:
                logger.error(f"Error halting function {func_name}: {str(e)}")
                actions.append({
                    'action': 'halt_function',
                    'function': func_name,
                    'status': 'failed',
                    'error': str(e)
                })
    
    return actions

def reenable_agent_functions():
    """
    Re-enable agent Lambda functions with their normal concurrency limits
    """
    actions = []
    
    function_configs = [
        (os.environ.get('MANAGER_FUNCTION'), 1),
        (os.environ.get('ENGINEER_FUNCTION'), 2),
        (os.environ.get('QA_FUNCTION'), 2)
    ]
    
    for func_name, concurrency in function_configs:
        if func_name:
            try:
                # Restore normal concurrency
                lambda_client.put_function_concurrency(
                    FunctionName=func_name,
                    ReservedConcurrentExecutions=concurrency
                )
                
                actions.append({
                    'action': 'reenable_function',
                    'function': func_name,
                    'concurrency': concurrency,
                    'status': 'success'
                })
                logger.info(f"Re-enabled function: {func_name} with concurrency: {concurrency}")
                
            except Exception as e:
                logger.error(f"Error re-enabling function {func_name}: {str(e)}")
                actions.append({
                    'action': 'reenable_function',
                    'function': func_name,
                    'status': 'failed',
                    'error': str(e)
                })
    
    return actions

def disable_event_rules():
    """
    Disable EventBridge rules to prevent further agent triggers
    """
    actions = []
    environment = os.environ.get('ENVIRONMENT', 'production')
    
    rule_patterns = [
        f"{environment}-qa-agent-trigger",
        f"engineer-agent-system-issue-{environment}",
        f"engineer-agent-opportunity-{environment}",
        f"manager-agent-schedule-{environment}"
    ]
    
    for rule_name in rule_patterns:
        try:
            events_client.disable_rule(Name=rule_name)
            actions.append({
                'action': 'disable_rule',
                'rule': rule_name,
                'status': 'success'
            })
            logger.info(f"Disabled EventBridge rule: {rule_name}")
        except events_client.exceptions.ResourceNotFoundException:
            pass  # Rule doesn't exist
        except Exception as e:
            logger.error(f"Error disabling rule {rule_name}: {str(e)}")
            actions.append({
                'action': 'disable_rule',
                'rule': rule_name,
                'status': 'failed',
                'error': str(e)
            })
    
    return actions

def enable_event_rules():
    """
    Re-enable EventBridge rules
    """
    actions = []
    environment = os.environ.get('ENVIRONMENT', 'production')
    
    rule_patterns = [
        f"{environment}-qa-agent-trigger",
        f"engineer-agent-system-issue-{environment}",
        f"engineer-agent-opportunity-{environment}",
        f"manager-agent-schedule-{environment}"
    ]
    
    for rule_name in rule_patterns:
        try:
            events_client.enable_rule(Name=rule_name)
            actions.append({
                'action': 'enable_rule',
                'rule': rule_name,
                'status': 'success'
            })
            logger.info(f"Enabled EventBridge rule: {rule_name}")
        except events_client.exceptions.ResourceNotFoundException:
            pass  # Rule doesn't exist
        except Exception as e:
            logger.error(f"Error enabling rule {rule_name}: {str(e)}")
            actions.append({
                'action': 'enable_rule',
                'rule': rule_name,
                'status': 'failed',
                'error': str(e)
            })
    
    return actions

def send_alert(costs, monthly_limit, daily_limit):
    """
    Send critical alert via SNS
    """
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    if not sns_topic_arn:
        logger.warning("No SNS topic ARN configured for alerts")
        return
    
    message = f"""
🚨 COST CIRCUIT BREAKER TRIGGERED 🚨

Environment: {os.environ.get('ENVIRONMENT', 'production')}
Timestamp: {datetime.now(timezone.utc).isoformat()}

Current Costs:
- Month-to-date: ${costs.get('month_to_date', 0):.2f}
- Last 24 hours: ${costs.get('last_24_hours', 0):.2f}
- Projected monthly total: ${costs.get('projected_monthly_total', 0):.2f}

Limits:
- Monthly limit: ${monthly_limit:.2f}
- Daily limit: ${daily_limit:.2f}

Top Services by Cost:
"""
    
    # Add top 5 services by cost
    if 'by_service' in costs:
        sorted_services = sorted(costs['by_service'].items(), key=lambda x: x[1], reverse=True)[:5]
        for service, amount in sorted_services:
            message += f"  - {service}: ${amount:.2f}\n"
    
    message += """
Actions Taken:
1. All agent Lambda functions have been halted (concurrency set to 0)
2. EventBridge rules have been disabled
3. No new agent executions will occur until manual intervention

Required Actions:
1. Review the cost spike immediately
2. Identify the root cause
3. Manually re-enable functions once the issue is resolved

To re-enable:
- Check the circuit breaker Lambda function logs
- Costs must be below 70% of limits for automatic re-enablement
- Or manually adjust Lambda concurrency settings
"""
    
    try:
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Subject=f"🚨 CRITICAL: Cost Circuit Breaker Triggered - ${costs.get('projected_monthly_total', 0):.2f}/${monthly_limit:.2f}",
            Message=message
        )
        logger.info("Critical alert sent via SNS")
    except Exception as e:
        logger.error(f"Error sending SNS alert: {str(e)}")

def send_cost_metrics(costs):
    """
    Send cost metrics to CloudWatch for monitoring
    """
    try:
        metrics = []
        
        # Add month-to-date metric
        if 'month_to_date' in costs:
            metrics.append({
                'MetricName': 'MonthToDateCost',
                'Value': costs['month_to_date'],
                'Unit': 'None',
                'Timestamp': datetime.now(timezone.utc)
            })
        
        # Add daily cost metric
        if 'last_24_hours' in costs:
            metrics.append({
                'MetricName': 'DailyCost',
                'Value': costs['last_24_hours'],
                'Unit': 'None',
                'Timestamp': datetime.now(timezone.utc)
            })
        
        # Add projected monthly total
        if 'projected_monthly_total' in costs:
            metrics.append({
                'MetricName': 'ProjectedMonthlyCost',
                'Value': costs['projected_monthly_total'],
                'Unit': 'None',
                'Timestamp': datetime.now(timezone.utc)
            })
        
        # Send metrics to CloudWatch
        if metrics:
            cloudwatch_client.put_metric_data(
                Namespace='CostControl/CircuitBreaker',
                MetricData=metrics
            )
            logger.info(f"Sent {len(metrics)} cost metrics to CloudWatch")
            
    except Exception as e:
        logger.error(f"Error sending metrics to CloudWatch: {str(e)}")
