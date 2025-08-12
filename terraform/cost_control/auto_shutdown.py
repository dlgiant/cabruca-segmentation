"""
Auto-shutdown Lambda function for development resources
Automatically stops resources after business hours to save costs
"""

import json
import os
import boto3
from datetime import datetime, timezone
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
lambda_client = boto3.client('lambda')
ec2_client = boto3.client('ec2')
rds_client = boto3.client('rds')

def lambda_handler(event, context):
    """
    Main handler for auto-shutdown/startup of development resources
    """
    environment = os.environ.get('ENVIRONMENT', 'development')
    action = event.get('action', 'shutdown')
    
    logger.info(f"Auto-{action} triggered for environment: {environment}")
    
    # Get resources with proper tags
    shutdown_tags = json.loads(os.environ.get('SHUTDOWN_TAGS', '{}'))
    
    results = {
        'action': action,
        'environment': environment,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'resources': {
            'lambda_functions': [],
            'ec2_instances': [],
            'rds_instances': []
        }
    }
    
    try:
        if action == 'shutdown':
            results = shutdown_resources(environment, shutdown_tags, results)
        elif action == 'startup':
            results = startup_resources(environment, shutdown_tags, results)
        else:
            raise ValueError(f"Invalid action: {action}")
            
    except Exception as e:
        logger.error(f"Error during {action}: {str(e)}")
        results['error'] = str(e)
        
    logger.info(f"Auto-{action} completed: {json.dumps(results)}")
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

def shutdown_resources(environment, tags, results):
    """
    Shutdown development resources to save costs
    """
    
    # 1. Set Lambda concurrency to 0 for non-essential functions
    try:
        functions = lambda_client.list_functions()
        for func in functions['Functions']:
            func_name = func['FunctionName']
            func_tags = lambda_client.list_tags(Resource=func['FunctionArn'])['Tags']
            
            # Check if function should be shut down
            if (environment in func_name and 
                func_tags.get('AutoShutdown') == 'true' and
                'critical' not in func_tags.get('Type', '').lower()):
                
                # Set reserved concurrency to 0 (effectively pauses the function)
                lambda_client.put_function_concurrency(
                    FunctionName=func_name,
                    ReservedConcurrentExecutions=0
                )
                results['resources']['lambda_functions'].append({
                    'name': func_name,
                    'status': 'shutdown',
                    'concurrency': 0
                })
                logger.info(f"Shutdown Lambda function: {func_name}")
                
    except Exception as e:
        logger.error(f"Error shutting down Lambda functions: {str(e)}")
        
    # 2. Stop EC2 instances with appropriate tags
    try:
        # Find instances with shutdown tags
        filters = [
            {'Name': f'tag:{k}', 'Values': [v]} 
            for k, v in tags.items()
        ]
        filters.append({'Name': 'instance-state-name', 'Values': ['running']})
        
        response = ec2_client.describe_instances(Filters=filters)
        
        instance_ids = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])
                results['resources']['ec2_instances'].append({
                    'id': instance['InstanceId'],
                    'type': instance['InstanceType'],
                    'status': 'stopping'
                })
        
        if instance_ids:
            ec2_client.stop_instances(InstanceIds=instance_ids)
            logger.info(f"Stopped EC2 instances: {instance_ids}")
            
    except Exception as e:
        logger.error(f"Error stopping EC2 instances: {str(e)}")
        
    # 3. Stop RDS instances (development databases)
    try:
        response = rds_client.describe_db_instances()
        
        for db in response['DBInstances']:
            db_id = db['DBInstanceIdentifier']
            
            # Check tags
            tags_response = rds_client.list_tags_for_resource(
                ResourceName=db['DBInstanceArn']
            )
            db_tags = {tag['Key']: tag['Value'] for tag in tags_response['TagList']}
            
            # Stop if it matches our criteria
            if (db_tags.get('Environment') == environment and
                db_tags.get('AutoShutdown') == 'true' and
                db['DBInstanceStatus'] == 'available'):
                
                rds_client.stop_db_instance(DBInstanceIdentifier=db_id)
                results['resources']['rds_instances'].append({
                    'id': db_id,
                    'engine': db['Engine'],
                    'status': 'stopping'
                })
                logger.info(f"Stopped RDS instance: {db_id}")
                
    except Exception as e:
        logger.error(f"Error stopping RDS instances: {str(e)}")
        
    return results

def startup_resources(environment, tags, results):
    """
    Start up development resources for the workday
    """
    
    # 1. Restore Lambda concurrency
    try:
        functions = lambda_client.list_functions()
        for func in functions['Functions']:
            func_name = func['FunctionName']
            
            # Check if function has zero concurrency (was shut down)
            try:
                concurrency = lambda_client.get_function_concurrency(
                    FunctionName=func_name
                )
                
                if concurrency.get('ReservedConcurrentExecutions') == 0:
                    # Restore appropriate concurrency based on function type
                    if 'manager' in func_name.lower():
                        new_concurrency = 1
                    elif 'engineer' in func_name.lower() or 'qa' in func_name.lower():
                        new_concurrency = 2
                    else:
                        new_concurrency = 5  # Default
                    
                    lambda_client.put_function_concurrency(
                        FunctionName=func_name,
                        ReservedConcurrentExecutions=new_concurrency
                    )
                    results['resources']['lambda_functions'].append({
                        'name': func_name,
                        'status': 'started',
                        'concurrency': new_concurrency
                    })
                    logger.info(f"Started Lambda function: {func_name} with concurrency: {new_concurrency}")
                    
            except lambda_client.exceptions.ResourceNotFoundException:
                pass  # Function doesn't have reserved concurrency
                
    except Exception as e:
        logger.error(f"Error starting Lambda functions: {str(e)}")
        
    # 2. Start EC2 instances
    try:
        # Find stopped instances with our tags
        filters = [
            {'Name': f'tag:{k}', 'Values': [v]} 
            for k, v in tags.items()
        ]
        filters.append({'Name': 'instance-state-name', 'Values': ['stopped']})
        
        response = ec2_client.describe_instances(Filters=filters)
        
        instance_ids = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])
                results['resources']['ec2_instances'].append({
                    'id': instance['InstanceId'],
                    'type': instance['InstanceType'],
                    'status': 'starting'
                })
        
        if instance_ids:
            ec2_client.start_instances(InstanceIds=instance_ids)
            logger.info(f"Started EC2 instances: {instance_ids}")
            
    except Exception as e:
        logger.error(f"Error starting EC2 instances: {str(e)}")
        
    # 3. Start RDS instances
    try:
        response = rds_client.describe_db_instances()
        
        for db in response['DBInstances']:
            db_id = db['DBInstanceIdentifier']
            
            # Check if instance is stopped
            if db['DBInstanceStatus'] == 'stopped':
                # Check tags
                tags_response = rds_client.list_tags_for_resource(
                    ResourceName=db['DBInstanceArn']
                )
                db_tags = {tag['Key']: tag['Value'] for tag in tags_response['TagList']}
                
                if (db_tags.get('Environment') == environment and
                    db_tags.get('AutoShutdown') == 'true'):
                    
                    rds_client.start_db_instance(DBInstanceIdentifier=db_id)
                    results['resources']['rds_instances'].append({
                        'id': db_id,
                        'engine': db['Engine'],
                        'status': 'starting'
                    })
                    logger.info(f"Started RDS instance: {db_id}")
                    
    except Exception as e:
        logger.error(f"Error starting RDS instances: {str(e)}")
        
    return results
