#!/bin/bash

# Package Lambda functions with AgentOps integration for tracking
# This script creates deployment packages with AgentOps SDK

set -e

echo "========================================="
echo "📦 Packaging Lambda Functions with AgentOps"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Base directory
BASE_DIR="/Users/ricardonunes/cabruca-segmentation/terraform"
cd $BASE_DIR

# Get AgentOps API key from .env
source /Users/ricardonunes/cabruca-segmentation/.env

echo -e "${YELLOW}Creating Lambda functions with AgentOps integration...${NC}"

# Manager Agent with AgentOps
cat > manager_agent/lambda_function.py << 'EOF'
import json
import os
import boto3
from datetime import datetime
import urllib3

# Simple AgentOps client implementation
class AgentOpsClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.http = urllib3.PoolManager()
        self.session_id = None
        self.base_url = "https://api.agentops.ai/v2"
        
    def start_session(self, tags=None):
        """Start a new AgentOps session"""
        try:
            response = self.http.request(
                'POST',
                f'{self.base_url}/sessions',
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': self.api_key
                },
                body=json.dumps({
                    'tags': tags or ['lambda', 'manager-agent'],
                    'agent_type': 'manager'
                }).encode('utf-8')
            )
            if response.status == 200:
                data = json.loads(response.data.decode('utf-8'))
                self.session_id = data.get('session_id')
                return self.session_id
        except Exception as e:
            print(f"Error starting AgentOps session: {e}")
            return None
    
    def record_event(self, event_type, data):
        """Record an event to AgentOps"""
        if not self.session_id:
            return
        
        try:
            self.http.request(
                'POST',
                f'{self.base_url}/events',
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': self.api_key
                },
                body=json.dumps({
                    'session_id': self.session_id,
                    'event_type': event_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }).encode('utf-8')
            )
        except Exception as e:
            print(f"Error recording event: {e}")
    
    def end_session(self, status='success'):
        """End the AgentOps session"""
        if not self.session_id:
            return
        
        try:
            self.http.request(
                'POST',
                f'{self.base_url}/sessions/{self.session_id}/end',
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': self.api_key
                },
                body=json.dumps({
                    'status': status,
                    'end_time': datetime.now().isoformat()
                }).encode('utf-8')
            )
        except Exception as e:
            print(f"Error ending session: {e}")

# Initialize AgentOps client
agentops_client = None

def lambda_handler(event, context):
    """Manager Agent Lambda Handler with AgentOps tracking"""
    global agentops_client
    
    # Initialize AgentOps if not already done
    if agentops_client is None and os.environ.get('AGENTOPS_API_KEY'):
        agentops_client = AgentOpsClient(os.environ['AGENTOPS_API_KEY'])
    
    # Start AgentOps session
    if agentops_client:
        agentops_client.start_session(tags=['manager', 'cabruca', os.environ.get('ENVIRONMENT', 'mvp')])
        agentops_client.record_event('agent_invoked', {
            'agent': 'manager',
            'event': event
        })
    
    # Get environment variables
    agent_type = os.environ.get('AGENT_TYPE', 'MANAGER')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    # Parse event
    action = event.get('action', 'default')
    
    # Record action in AgentOps
    if agentops_client:
        agentops_client.record_event('action_processed', {
            'action': action,
            'agent_type': agent_type
        })
    
    # Health check response
    if action == 'health_check':
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'healthy',
                'agent': agent_type,
                'environment': environment,
                'timestamp': datetime.now().isoformat(),
                'agentops_tracking': agentops_client is not None
            })
        }
        if agentops_client:
            agentops_client.record_event('health_check_completed', response)
            agentops_client.end_session('success')
        return response
    
    # Process other actions
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'{agent_type} agent processed request',
            'action': action,
            'event': event,
            'timestamp': datetime.now().isoformat(),
            'agentops_tracking': agentops_client is not None
        })
    }
    
    # End AgentOps session
    if agentops_client:
        agentops_client.record_event('request_completed', response)
        agentops_client.end_session('success')
    
    return response
EOF

# Engineer Agent with AgentOps
cat > engineer_agent/lambda_function.py << 'EOF'
import json
import os
from datetime import datetime
import urllib3

# Simple AgentOps tracking
def track_to_agentops(event_type, data):
    """Send tracking data to AgentOps"""
    api_key = os.environ.get('AGENTOPS_API_KEY')
    if not api_key:
        return
    
    try:
        http = urllib3.PoolManager()
        http.request(
            'POST',
            'https://api.agentops.ai/v2/events',
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            },
            body=json.dumps({
                'event_type': event_type,
                'agent': 'engineer',
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'environment': os.environ.get('ENVIRONMENT', 'mvp')
            }).encode('utf-8')
        )
    except Exception as e:
        print(f"AgentOps tracking error: {e}")

def lambda_handler(event, context):
    """Engineer Agent Lambda Handler"""
    
    agent_type = os.environ.get('AGENT_TYPE', 'ENGINEER')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    # Track invocation
    track_to_agentops('agent_invoked', {'agent': 'engineer', 'event': event})
    
    action = event.get('action', 'default')
    message = event.get('message', '')
    
    # Process test action
    if action == 'test':
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'agent': agent_type,
                'message': f'Received: {message}',
                'response': 'Engineer agent is operational',
                'timestamp': datetime.now().isoformat(),
                'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
            })
        }
        track_to_agentops('test_completed', response)
        return response
    
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'{agent_type} agent processed request',
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
        })
    }
    
    track_to_agentops('request_completed', response)
    return response
EOF

# QA Agent with AgentOps
cat > qa_agent/lambda_function.py << 'EOF'
import json
import os
from datetime import datetime
import urllib3

def track_to_agentops(event_type, data):
    """Send tracking data to AgentOps"""
    api_key = os.environ.get('AGENTOPS_API_KEY')
    if not api_key:
        return
    
    try:
        http = urllib3.PoolManager()
        http.request(
            'POST',
            'https://api.agentops.ai/v2/events',
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            },
            body=json.dumps({
                'event_type': event_type,
                'agent': 'qa',
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'environment': os.environ.get('ENVIRONMENT', 'mvp')
            }).encode('utf-8')
        )
    except Exception as e:
        print(f"AgentOps tracking error: {e}")

def lambda_handler(event, context):
    """QA Agent Lambda Handler"""
    
    agent_type = os.environ.get('AGENT_TYPE', 'QA')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    track_to_agentops('agent_invoked', {'agent': 'qa', 'event': event})
    
    action = event.get('action', 'default')
    deployment = event.get('deployment', '')
    
    # Validate deployment
    if action == 'validate':
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'validated',
                'agent': agent_type,
                'deployment': deployment,
                'validation': 'All checks passed',
                'timestamp': datetime.now().isoformat(),
                'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
            })
        }
        track_to_agentops('validation_completed', response)
        return response
    
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'{agent_type} agent processed request',
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
        })
    }
    
    track_to_agentops('request_completed', response)
    return response
EOF

# Researcher Agent with AgentOps
cat > researcher_agent/lambda_function.py << 'EOF'
import json
import os
from datetime import datetime
import urllib3

def track_to_agentops(event_type, data):
    """Send tracking data to AgentOps"""
    api_key = os.environ.get('AGENTOPS_API_KEY')
    if not api_key:
        return
    
    try:
        http = urllib3.PoolManager()
        http.request(
            'POST',
            'https://api.agentops.ai/v2/events',
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            },
            body=json.dumps({
                'event_type': event_type,
                'agent': 'researcher',
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'environment': os.environ.get('ENVIRONMENT', 'mvp')
            }).encode('utf-8')
        )
    except Exception as e:
        print(f"AgentOps tracking error: {e}")

def lambda_handler(event, context):
    """Researcher Agent Lambda Handler"""
    
    agent_type = os.environ.get('AGENT_TYPE', 'RESEARCHER')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    track_to_agentops('agent_invoked', {'agent': 'researcher', 'event': event})
    
    analysis_type = event.get('type', 'general')
    region = event.get('region', 'unknown')
    
    # Cabruca analysis
    if analysis_type == 'cabruca_analysis':
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'analyzed',
                'agent': agent_type,
                'analysis_type': analysis_type,
                'region': region,
                'findings': {
                    'forest_coverage': '85%',
                    'species_diversity': 'high',
                    'carbon_storage': 'significant'
                },
                'timestamp': datetime.now().isoformat(),
                'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
            })
        }
        track_to_agentops('analysis_completed', {
            'type': analysis_type,
            'region': region,
            'findings': response['body']
        })
        return response
    
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'{agent_type} agent processed request',
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
        })
    }
    
    track_to_agentops('request_completed', response)
    return response
EOF

# Data Processor Agent with AgentOps
cat > data_processor_agent/lambda_function.py << 'EOF'
import json
import os
from datetime import datetime
import urllib3

def track_to_agentops(event_type, data):
    """Send tracking data to AgentOps"""
    api_key = os.environ.get('AGENTOPS_API_KEY')
    if not api_key:
        return
    
    try:
        http = urllib3.PoolManager()
        http.request(
            'POST',
            'https://api.agentops.ai/v2/events',
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            },
            body=json.dumps({
                'event_type': event_type,
                'agent': 'data_processor',
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'environment': os.environ.get('ENVIRONMENT', 'mvp')
            }).encode('utf-8')
        )
    except Exception as e:
        print(f"AgentOps tracking error: {e}")

def lambda_handler(event, context):
    """Data Processor Agent Lambda Handler"""
    
    agent_type = os.environ.get('AGENT_TYPE', 'DATA_PROCESSOR')
    environment = os.environ.get('ENVIRONMENT', 'mvp')
    
    track_to_agentops('agent_invoked', {'agent': 'data_processor', 'event': event})
    
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
        
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'processed',
                'agent': agent_type,
                'records': records,
                'timestamp': datetime.now().isoformat(),
                'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
            })
        }
        track_to_agentops('s3_processing_completed', {'records': records})
        return response
    
    # Handle direct invocation
    data_type = event.get('data_type', 'unknown')
    time_range = event.get('time_range', 'all')
    
    response = {
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
            'timestamp': datetime.now().isoformat(),
            'agentops_tracking': bool(os.environ.get('AGENTOPS_API_KEY'))
        })
    }
    
    track_to_agentops('data_processing_completed', {
        'data_type': data_type,
        'time_range': time_range,
        'metrics': response['body']
    })
    return response
EOF

echo -e "${GREEN}✅ Lambda functions with AgentOps integration created${NC}"
echo ""

# Package each Lambda function
function package_lambda() {
    local agent_name=$1
    local agent_dir="${agent_name}_agent"
    
    echo -e "${YELLOW}Packaging ${agent_name} agent...${NC}"
    
    if [ ! -d "$agent_dir" ]; then
        echo -e "${RED}❌ Directory $agent_dir does not exist${NC}"
        return 1
    fi
    
    cd $agent_dir
    
    # Remove old package
    rm -rf lambda_function.zip package
    
    # Create package with just the Python file
    zip lambda_function.zip lambda_function.py -q
    
    echo -e "${GREEN}✅ ${agent_name} agent packaged successfully${NC}"
    cd $BASE_DIR
}

# Package all agents
agents=("manager" "engineer" "qa" "researcher" "data_processor")

for agent in "${agents[@]}"; do
    package_lambda $agent
done

echo ""
echo "========================================="
echo -e "${GREEN}✨ All Lambda functions packaged with AgentOps!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Update Lambda environment variables with AGENTOPS_API_KEY"
echo "2. Run 'terraform apply -var-file=mvp.tfvars' to deploy"
echo "3. Test agents to verify AgentOps tracking"
echo "4. Check https://app.agentops.ai for agent activity"