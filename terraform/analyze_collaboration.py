"""
Lambda function for analyzing agent collaboration patterns
Triggered by EventBridge when agents communicate
"""

import json
import os
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
MONITORING_TABLE_NAME = os.environ.get('MONITORING_TABLE_NAME', 'agent-monitoring')
DECISIONS_TABLE_NAME = os.environ.get('DECISIONS_TABLE_NAME', 'agent-decisions')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')

def lambda_handler(event, context):
    """
    Process agent collaboration events and analyze patterns
    """
    try:
        logger.info(f"Processing collaboration event: {json.dumps(event)}")
        
        # Extract event details
        detail = event.get('detail', {})
        source_agent = detail.get('source_agent')
        target_agent = detail.get('target_agent')
        message_type = detail.get('message_type')
        content = detail.get('content', {})
        timestamp = detail.get('timestamp', datetime.utcnow().isoformat())
        session_id = detail.get('session_id')
        
        # Store collaboration event
        store_collaboration_event(
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=message_type,
            content=content,
            timestamp=timestamp,
            session_id=session_id
        )
        
        # Analyze collaboration patterns
        patterns = analyze_patterns(source_agent, target_agent)
        
        # Send metrics to CloudWatch
        send_collaboration_metrics(source_agent, target_agent, message_type)
        
        # Check for anomalies
        anomalies = detect_collaboration_anomalies(patterns)
        
        if anomalies:
            handle_anomalies(anomalies)
        
        # Generate insights
        insights = generate_collaboration_insights(patterns)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'patterns': patterns,
                'anomalies': anomalies,
                'insights': insights
            }, default=str)
        }
        
    except Exception as e:
        logger.error(f"Error processing collaboration event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }

def store_collaboration_event(source_agent: str, target_agent: str, 
                             message_type: str, content: Dict[str, Any],
                             timestamp: str, session_id: str):
    """Store collaboration event in DynamoDB"""
    try:
        table = dynamodb.Table(MONITORING_TABLE_NAME)
        
        item = {
            'event_id': f"collab-{source_agent}-{target_agent}-{timestamp}",
            'timestamp': timestamp,
            'event_type': 'agent_collaboration',
            'agent_name': source_agent,
            'session_id': session_id,
            'details': {
                'source_agent': source_agent,
                'target_agent': target_agent,
                'message_type': message_type,
                'content': content,
                'flow': f"{source_agent}->{target_agent}"
            }
        }
        
        table.put_item(Item=item)
        logger.info(f"Stored collaboration event: {source_agent} -> {target_agent}")
        
    except Exception as e:
        logger.error(f"Error storing collaboration event: {str(e)}")

def analyze_patterns(source_agent: str, target_agent: str) -> Dict[str, Any]:
    """Analyze collaboration patterns between agents"""
    try:
        table = dynamodb.Table(MONITORING_TABLE_NAME)
        
        # Query recent collaboration events (last 24 hours)
        start_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        response = table.query(
            IndexName='event-type-index',
            KeyConditionExpression='event_type = :event_type AND #ts >= :start_time',
            ExpressionAttributeNames={
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':event_type': 'agent_collaboration',
                ':start_time': start_time
            }
        )
        
        items = response.get('Items', [])
        
        # Analyze patterns
        flow_counts = {}
        message_types = {}
        hourly_distribution = {}
        
        for item in items:
            details = item.get('details', {})
            flow = details.get('flow', '')
            msg_type = details.get('message_type', '')
            
            # Count flows
            flow_counts[flow] = flow_counts.get(flow, 0) + 1
            
            # Count message types
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            # Hourly distribution
            timestamp = item.get('timestamp', '')
            if timestamp:
                hour = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).hour
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        # Calculate metrics
        total_collaborations = len(items)
        most_active_flow = max(flow_counts.items(), key=lambda x: x[1]) if flow_counts else None
        most_common_message = max(message_types.items(), key=lambda x: x[1]) if message_types else None
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else None
        
        patterns = {
            'total_collaborations_24h': total_collaborations,
            'flow_counts': flow_counts,
            'message_type_distribution': message_types,
            'hourly_distribution': hourly_distribution,
            'most_active_flow': most_active_flow,
            'most_common_message_type': most_common_message,
            'peak_activity_hour': peak_hour,
            'average_collaborations_per_hour': total_collaborations / 24 if total_collaborations > 0 else 0
        }
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {str(e)}")
        return {}

def send_collaboration_metrics(source_agent: str, target_agent: str, message_type: str):
    """Send collaboration metrics to CloudWatch"""
    try:
        cloudwatch.put_metric_data(
            Namespace='AgentOps/Collaboration',
            MetricData=[
                {
                    'MetricName': 'MessageCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'SourceAgent', 'Value': source_agent},
                        {'Name': 'TargetAgent', 'Value': target_agent},
                        {'Name': 'MessageType', 'Value': message_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'AgentCommunication',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Flow', 'Value': f"{source_agent}->{target_agent}"},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )
        logger.info(f"Sent collaboration metrics for {source_agent} -> {target_agent}")
        
    except Exception as e:
        logger.error(f"Error sending metrics: {str(e)}")

def detect_collaboration_anomalies(patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect anomalies in collaboration patterns"""
    anomalies = []
    
    try:
        # Check for excessive collaborations
        total_collaborations = patterns.get('total_collaborations_24h', 0)
        avg_per_hour = patterns.get('average_collaborations_per_hour', 0)
        
        if total_collaborations > 1000:
            anomalies.append({
                'type': 'excessive_collaboration',
                'severity': 'high',
                'description': f"Excessive collaboration detected: {total_collaborations} messages in 24h",
                'recommendation': 'Consider implementing caching or batching to reduce inter-agent communication'
            })
        
        # Check for imbalanced flows
        flow_counts = patterns.get('flow_counts', {})
        if flow_counts:
            max_flow = max(flow_counts.values())
            min_flow = min(flow_counts.values())
            
            if max_flow > 10 * min_flow and min_flow > 0:
                anomalies.append({
                    'type': 'imbalanced_flow',
                    'severity': 'medium',
                    'description': f"Imbalanced message flow detected (max: {max_flow}, min: {min_flow})",
                    'recommendation': 'Review agent responsibilities and consider load balancing'
                })
        
        # Check for unusual peak hours
        hourly_distribution = patterns.get('hourly_distribution', {})
        if hourly_distribution:
            peak_hour = patterns.get('peak_activity_hour')
            if peak_hour and peak_hour[0] in [2, 3, 4, 5]:  # Unusual hours
                anomalies.append({
                    'type': 'unusual_peak_time',
                    'severity': 'low',
                    'description': f"Peak activity at unusual hour: {peak_hour[0]}:00",
                    'recommendation': 'Investigate scheduling or timezone issues'
                })
        
        # Check for single point of failure patterns
        most_active_flow = patterns.get('most_active_flow')
        if most_active_flow and most_active_flow[1] > total_collaborations * 0.5:
            anomalies.append({
                'type': 'single_point_dependency',
                'severity': 'medium',
                'description': f"Single flow accounts for >50% of collaborations: {most_active_flow[0]}",
                'recommendation': 'Consider implementing redundancy or alternative communication paths'
            })
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
    
    return anomalies

def handle_anomalies(anomalies: List[Dict[str, Any]]):
    """Handle detected anomalies"""
    try:
        for anomaly in anomalies:
            # Send CloudWatch metric for anomaly
            cloudwatch.put_metric_data(
                Namespace='AgentOps/Anomalies',
                MetricData=[
                    {
                        'MetricName': anomaly['type'],
                        'Value': 1,
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Severity', 'Value': anomaly['severity']},
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    }
                ]
            )
            
            # Log anomaly
            logger.warning(f"Anomaly detected: {anomaly['type']} - {anomaly['description']}")
            
            # For high severity anomalies, trigger alert
            if anomaly['severity'] == 'high':
                trigger_anomaly_alert(anomaly)
                
    except Exception as e:
        logger.error(f"Error handling anomalies: {str(e)}")

def trigger_anomaly_alert(anomaly: Dict[str, Any]):
    """Trigger alert for high severity anomalies"""
    try:
        # Create CloudWatch alarm
        cloudwatch.put_metric_alarm(
            AlarmName=f"AgentOps-Collaboration-{anomaly['type']}",
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName=anomaly['type'],
            Namespace='AgentOps/Anomalies',
            Period=300,
            Statistic='Sum',
            Threshold=0,
            ActionsEnabled=True,
            AlarmDescription=anomaly['description'],
            Dimensions=[
                {'Name': 'Environment', 'Value': ENVIRONMENT}
            ]
        )
        logger.info(f"Created alarm for anomaly: {anomaly['type']}")
        
    except Exception as e:
        logger.error(f"Error triggering alert: {str(e)}")

def generate_collaboration_insights(patterns: Dict[str, Any]) -> List[str]:
    """Generate actionable insights from collaboration patterns"""
    insights = []
    
    try:
        total_collaborations = patterns.get('total_collaborations_24h', 0)
        avg_per_hour = patterns.get('average_collaborations_per_hour', 0)
        
        # Insight on collaboration volume
        if total_collaborations > 500:
            insights.append(f"High collaboration volume ({total_collaborations}/day). Consider implementing message batching.")
        elif total_collaborations < 10:
            insights.append("Low collaboration volume. Agents may be working in isolation.")
        
        # Insight on flow patterns
        flow_counts = patterns.get('flow_counts', {})
        if flow_counts:
            bidirectional_flows = sum(1 for flow in flow_counts if any(
                reverse_flow in flow_counts for reverse_flow in [
                    f"{flow.split('->')[1]}->{flow.split('->')[0]}" 
                ] if '->' in flow
            ))
            
            if bidirectional_flows > len(flow_counts) * 0.7:
                insights.append("Good bidirectional communication pattern detected.")
            else:
                insights.append("Consider improving bidirectional communication between agents.")
        
        # Insight on message types
        message_types = patterns.get('message_type_distribution', {})
        if 'error' in message_types or 'failure' in message_types:
            error_count = message_types.get('error', 0) + message_types.get('failure', 0)
            if error_count > total_collaborations * 0.1:
                insights.append(f"High error message rate ({error_count} errors). Review agent error handling.")
        
        # Insight on timing
        hourly_distribution = patterns.get('hourly_distribution', {})
        if hourly_distribution:
            variance = calculate_variance(list(hourly_distribution.values()))
            if variance < 10:
                insights.append("Even distribution of activity throughout the day.")
            else:
                insights.append("Uneven activity distribution. Consider load balancing across time.")
        
        # Insight on most active flow
        most_active_flow = patterns.get('most_active_flow')
        if most_active_flow:
            insights.append(f"Most active communication: {most_active_flow[0]} ({most_active_flow[1]} messages)")
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
    
    return insights

def calculate_variance(values: List[int]) -> float:
    """Calculate variance of a list of values"""
    if not values:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance
