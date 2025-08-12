# AgentOps Enhanced Monitoring Integration

## Overview

This implementation provides comprehensive monitoring for the autonomous agent system using AgentOps, CloudWatch, and DynamoDB. It tracks agent activities, costs, decisions, and collaboration patterns to ensure optimal performance and cost efficiency.

## Features

### 1. Custom Event Tracking
- **Issue Detection Events**: Tracks when the Manager Agent detects system issues
- **Code Generation Events**: Monitors Engineer Agent code generation activities  
- **Test Execution Events**: Records QA Agent test passes and failures
- **Collaboration Events**: Captures inter-agent communication via EventBridge

### 2. Cost Tracking
- **Lambda Execution Costs**: Tracks compute costs for each agent
- **LLM API Costs**: Monitors Claude API usage and associated costs
- **Cost Alerts**: Automated alerts when costs exceed thresholds
- **Cost Optimization**: Identifies opportunities to reduce expenses

### 3. Decision Logging
- **Full Reasoning Chains**: Stores complete decision-making processes
- **Confidence Scores**: Tracks agent confidence in decisions
- **Input/Output Tracking**: Records what information led to which decisions

### 4. Collaboration Monitoring
- **Message Flow Analysis**: Tracks communication patterns between agents
- **Anomaly Detection**: Identifies unusual collaboration patterns
- **Performance Insights**: Generates actionable recommendations

### 5. CloudWatch Dashboards
- **Main Dashboard**: Overall system health and activity
- **Cost Dashboard**: Detailed cost breakdown and trends
- **Collaboration Dashboard**: Agent communication patterns

## Setup

### Prerequisites

1. AWS Account with appropriate permissions
2. Python 3.11+ 
3. Terraform 1.0+
4. AgentOps API Key

### Environment Variables

Set the following environment variables in your Lambda functions:

```bash
AGENTOPS_API_KEY=your_agentops_api_key
MONITORING_TABLE_NAME=agent-monitoring
DECISIONS_TABLE_NAME=agent-decisions
COST_ALERT_THRESHOLD=10
COST_ALERT_TOPIC_ARN=arn:aws:sns:region:account:agentops-cost-alerts
EVENT_BUS_NAME=default
ENVIRONMENT=production
```

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Deploy infrastructure with Terraform:
```bash
terraform init
terraform plan -var="environment=production" -var="alert_email=your-email@example.com"
terraform apply
```

3. Deploy CloudWatch dashboards:
```bash
python agentops_dashboard.py
```

## Usage

### Monitoring in Agent Code

Each agent automatically integrates AgentOps monitoring:

```python
from agentops_monitoring import AgentOpsMonitor, AgentEventType

class YourAgent:
    def __init__(self):
        self.monitor = AgentOpsMonitor('agent_name')
    
    def process(self):
        # Record custom event
        self.monitor.record_event(
            AgentEventType.ISSUE_DETECTED,
            {
                'issue_type': 'high_error_rate',
                'severity': 'high',
                'error_rate': 0.15
            }
        )
        
        # Track LLM usage
        cost = self.monitor.track_llm_usage(
            model='claude-3-opus',
            input_tokens=1000,
            output_tokens=500
        )
        
        # Record decision with reasoning
        self.monitor.record_decision(
            decision_type='remediation',
            reasoning_chain=[
                {'thought': 'Analyzing error pattern', 
                 'action': 'query_metrics',
                 'observation': 'Spike in 500 errors'},
                {'thought': 'Root cause identified',
                 'action': 'generate_fix',
                 'observation': 'Memory leak in service'}
            ],
            inputs={'metrics': {...}},
            outputs={'action': 'restart_service'},
            confidence_score=0.92
        )
```

### Accessing Dashboards

1. **Main Dashboard**: 
   - URL: `https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=AgentOps-Monitoring`
   - Shows agent activity, costs, and effectiveness

2. **Cost Dashboard**:
   - URL: `https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=AgentOps-Monitoring-Cost`
   - Detailed cost breakdown by agent and service

3. **Collaboration Dashboard**:
   - URL: `https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=AgentOps-Monitoring-Collaboration`
   - Agent communication patterns and workflow analysis

### Analyzing Patterns

Use the AgentOpsAnalyzer to identify optimization opportunities:

```python
from agentops_monitoring import AgentOpsAnalyzer

analyzer = AgentOpsAnalyzer()

# Analyze collaboration patterns
patterns = analyzer.analyze_collaboration_patterns(time_window_hours=24)
print(f"Most active flow: {patterns['most_active_flow']}")

# Check agent effectiveness
effectiveness = analyzer.analyze_agent_effectiveness('manager', 24)
print(f"Success rate: {effectiveness['success_rate']:.1%}")

# Get optimization recommendations
opportunities = analyzer.identify_optimization_opportunities()
for opp in opportunities:
    print(f"{opp['type']}: {opp['recommendation']}")
```

## Metrics

### Event Metrics
- `AgentOps/Events/issue_detected`: Number of issues detected
- `AgentOps/Events/code_generated`: Code generation events
- `AgentOps/Events/test_passed`: Successful test executions
- `AgentOps/Events/test_failed`: Failed test executions
- `AgentOps/Events/decision_made`: Agent decisions made

### Cost Metrics
- `AgentOps/Costs/LambdaExecutionCost`: Lambda compute costs
- `AgentOps/Costs/EventCost`: Event processing costs by type
- `AgentOps/Costs/TotalCost`: Total system costs

### Collaboration Metrics
- `AgentOps/Collaboration/MessageCount`: Messages between agents
- `AgentOps/Collaboration/AgentCommunication`: Communication flows
- `AgentOps/Anomalies/*`: Detected anomalies in patterns

## Alerts

### Configured Alarms
1. **High Cost Alert**: Triggers when costs exceed threshold ($10 default)
2. **High Error Rate**: Triggers on >5 test failures in 5 minutes
3. **Low Success Rate**: Triggers when success rate drops below 80%
4. **Collaboration Anomalies**: Triggers on unusual communication patterns

### Alert Channels
- Email notifications via SNS
- CloudWatch alarm dashboard
- Optional: Slack/PagerDuty integration

## Cost Optimization

### Automatic Optimizations
- Identifies high-cost agents and operations
- Suggests model downgrades when appropriate
- Recommends caching for repetitive operations
- Detects inefficient collaboration patterns

### Manual Review
1. Check Cost Dashboard weekly
2. Review optimization recommendations
3. Adjust thresholds based on budget
4. Implement suggested caching strategies

## Troubleshooting

### Common Issues

1. **AgentOps not initializing**
   - Verify AGENTOPS_API_KEY is set correctly
   - Check network connectivity to AgentOps API

2. **Missing metrics in CloudWatch**
   - Ensure IAM roles have CloudWatch permissions
   - Verify metric namespace is correct

3. **DynamoDB throttling**
   - Tables are configured for on-demand billing
   - If issues persist, consider provisioned capacity

4. **High costs detected**
   - Review Cost Dashboard for breakdown
   - Check for excessive LLM usage
   - Consider using smaller models for simple tasks

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Manager Agent  │────▶│  Engineer Agent │────▶│    QA Agent     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
    ┌────────────────────────────────────────────────────────┐
    │                    AgentOps Monitor                     │
    │  • Record Events  • Track Costs  • Log Decisions       │
    └────────┬───────────────────┬───────────────┬──────────┘
             │                   │               │
             ▼                   ▼               ▼
    ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
    │   DynamoDB   │    │  CloudWatch  │   │ EventBridge  │
    │   Tables     │    │   Metrics    │   │   Events     │
    └──────────────┘    └──────────────┘   └──────────────┘
             │                   │               │
             └───────────────────┴───────────────┘
                             │
                    ┌────────▼────────┐
                    │   Dashboards    │
                    │   & Alarms      │
                    └─────────────────┘
```

## Future Enhancements

1. **Machine Learning Integration**
   - Predictive cost modeling
   - Anomaly detection using ML
   - Pattern recognition for optimization

2. **Advanced Visualizations**
   - Real-time agent activity flow
   - Cost prediction graphs
   - Decision tree visualizations

3. **External Integrations**
   - Datadog/New Relic integration
   - Slack notifications
   - JIRA ticket creation for issues

4. **Performance Optimizations**
   - Batch event processing
   - Async metric publishing
   - Cache frequently accessed data

## Support

For issues or questions:
1. Check CloudWatch Logs for error details
2. Review this documentation
3. Contact the platform team

## License

Copyright (c) 2024 - All rights reserved
