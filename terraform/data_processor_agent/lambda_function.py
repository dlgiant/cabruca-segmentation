import json
import os
from datetime import datetime

def lambda_handler(event, context):
    """Data Processor Agent Lambda Handler"""
    
    agent_type = os.environ.get('AGENT_TYPE', 'DATA_PROCESSOR')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    # Handle S3 events
    if 'Records' in event:
        records = []
        for record in event['Records']:
            if 's3' in record:
                bucket = record['s3']['bucket']['name']
                key = record['s3']['object']['key']
                records.append({
                    'bucket': bucket,
                    'key': key,
                    'processed': True
                })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'processed',
                'agent': agent_type,
                'records': records,
                'timestamp': datetime.now().isoformat()
            })
        }
    
    # Handle direct invocation
    data_type = event.get('data_type', 'unknown')
    time_range = event.get('time_range', 'all')
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'processed',
            'agent': agent_type,
            'data_type': data_type,
            'time_range': time_range,
            'metrics': {
                'records_processed': 1000,
                'processing_time': '2.5s',
                'data_quality': 'good'
            },
            'timestamp': datetime.now().isoformat()
        })
    }
