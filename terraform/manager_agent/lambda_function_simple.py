import json
import os
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """Manager Agent Lambda Handler"""
    
    # Get environment variables
    agent_type = os.environ.get('AGENT_TYPE', 'MANAGER')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    # Parse event
    action = event.get('action', 'default')
    
    # Health check response
    if action == 'health_check':
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'healthy',
                'agent': agent_type,
                'environment': environment,
                'timestamp': datetime.now().isoformat()
            })
        }
    
    # Default response
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'{agent_type} agent processed request',
            'action': action,
            'event': event,
            'timestamp': datetime.now().isoformat()
        })
    }
