"""
AgentOps Enhanced Monitoring Module
Provides comprehensive monitoring for agent activities, costs, and collaboration patterns
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import agentops
import boto3

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# AWS Clients
cloudwatch = boto3.client("cloudwatch")
dynamodb = boto3.resource("dynamodb")
eventbridge = boto3.client("events")
lambda_client = boto3.client("lambda")
ce_client = boto3.client("ce")  # Cost Explorer

# Environment variables
AGENTOPS_API_KEY = os.environ.get("AGENTOPS_API_KEY")
MONITORING_TABLE_NAME = os.environ.get("MONITORING_TABLE_NAME", "agent-monitoring")
DECISIONS_TABLE_NAME = os.environ.get("DECISIONS_TABLE_NAME", "agent-decisions")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")

# Cost tracking constants
LAMBDA_COST_PER_GB_SECOND = 0.0000166667  # $0.0000166667 per GB-second
LAMBDA_COST_PER_REQUEST = 0.0000002  # $0.20 per 1M requests
CLAUDE_COST_PER_1K_TOKENS = {
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}


class AgentEventType(Enum):
    """Types of agent events to track"""

    # Manager Agent Events
    ISSUE_DETECTED = "issue_detected"
    ANALYSIS_STARTED = "analysis_started"
    DECISION_MADE = "decision_made"
    ALERT_SENT = "alert_sent"

    # Engineer Agent Events
    CODE_GENERATION_STARTED = "code_generation_started"
    CODE_GENERATED = "code_generated"
    PR_CREATED = "pr_created"
    TERRAFORM_UPDATED = "terraform_updated"

    # QA Agent Events
    TEST_SUITE_CREATED = "test_suite_created"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    VALIDATION_COMPLETED = "validation_completed"

    # Collaboration Events
    AGENT_COMMUNICATION = "agent_communication"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"

    # Cost Events
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    COST_OPTIMIZATION_SUGGESTED = "cost_optimization_suggested"


@dataclass
class AgentEvent:
    """Represents an agent event"""

    event_id: str
    agent_name: str
    event_type: AgentEventType
    timestamp: datetime
    details: Dict[str, Any]
    cost: Optional[float] = None
    duration_ms: Optional[int] = None
    parent_event_id: Optional[str] = None


@dataclass
class AgentDecision:
    """Represents an agent decision with reasoning"""

    decision_id: str
    agent_name: str
    timestamp: datetime
    decision_type: str
    reasoning_chain: List[Dict[str, Any]]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence_score: float
    cost: float


@dataclass
class AgentCollaboration:
    """Represents agent collaboration pattern"""

    collaboration_id: str
    initiator_agent: str
    participant_agents: List[str]
    message_flow: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime]
    total_messages: int
    outcome: Optional[str]


class AgentOpsMonitor:
    """Enhanced AgentOps monitoring with custom events and cost tracking"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.session_id = str(uuid.uuid4())
        self.monitoring_table = dynamodb.Table(MONITORING_TABLE_NAME)
        self.decisions_table = dynamodb.Table(DECISIONS_TABLE_NAME)
        self._initialize_agentops()
        self.start_time = datetime.utcnow()
        self.events = []
        self.total_cost = 0.0

    def _initialize_agentops(self):
        """Initialize AgentOps with API key from environment"""
        if AGENTOPS_API_KEY:
            agentops.init(api_key=AGENTOPS_API_KEY)
            self.agentops_session = agentops.start_session(
                tags=[self.agent_name, ENVIRONMENT]
            )
            logger.info(f"AgentOps initialized for {self.agent_name}")
        else:
            logger.warning("AgentOps API key not found in environment")
            self.agentops_session = None

    def record_event(
        self,
        event_type: AgentEventType,
        details: Dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Record a custom agent event"""
        event_id = str(uuid.uuid4())
        event = AgentEvent(
            event_id=event_id,
            agent_name=self.agent_name,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            details=details,
            parent_event_id=parent_event_id,
        )

        # Calculate event cost if applicable
        event.cost = self._calculate_event_cost(event_type, details)
        self.total_cost += event.cost or 0

        # Record in AgentOps
        if self.agentops_session:
            agentops.record(
                agentops.Event(
                    event_type=event_type.value,
                    params=details,
                    returns={"event_id": event_id, "cost": event.cost},
                )
            )

        # Store in DynamoDB
        self._store_event(event)

        # Send to CloudWatch
        self._send_cloudwatch_metric(event)

        self.events.append(event)

        logger.info(f"Recorded event: {event_type.value} for {self.agent_name}")
        return event_id

    def record_decision(
        self,
        decision_type: str,
        reasoning_chain: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        confidence_score: float = 0.0,
    ) -> str:
        """Record an agent decision with full reasoning chain"""
        decision_id = str(uuid.uuid4())

        # Calculate decision cost (LLM tokens + compute)
        cost = self._calculate_decision_cost(reasoning_chain)

        decision = AgentDecision(
            decision_id=decision_id,
            agent_name=self.agent_name,
            timestamp=datetime.utcnow(),
            decision_type=decision_type,
            reasoning_chain=reasoning_chain,
            inputs=inputs,
            outputs=outputs,
            confidence_score=confidence_score,
            cost=cost,
        )

        # Record in AgentOps
        if self.agentops_session:
            agentops.record(
                agentops.Event(
                    event_type="decision_made",
                    params={
                        "decision_type": decision_type,
                        "confidence": confidence_score,
                        "reasoning_steps": len(reasoning_chain),
                    },
                    returns=outputs,
                )
            )

        # Store decision details
        self._store_decision(decision)

        # Track cost
        self.total_cost += cost

        logger.info(
            f"Recorded decision: {decision_type} with confidence {confidence_score}"
        )
        return decision_id

    def track_agent_collaboration(
        self, target_agent: str, message_type: str, message_content: Dict[str, Any]
    ) -> str:
        """Track agent-to-agent communication via EventBridge"""
        collaboration_event = {
            "source_agent": self.agent_name,
            "target_agent": target_agent,
            "message_type": message_type,
            "content": message_content,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
        }

        # Record collaboration event
        event_id = self.record_event(
            AgentEventType.AGENT_COMMUNICATION, collaboration_event
        )

        # Publish to EventBridge for pattern analysis
        self._publish_collaboration_event(collaboration_event)

        return event_id

    def track_lambda_execution(
        self, request_id: str, duration_ms: int, memory_mb: int, billed_duration_ms: int
    ) -> float:
        """Track Lambda execution costs"""
        # Calculate Lambda cost
        gb_seconds = (memory_mb / 1024.0) * (billed_duration_ms / 1000.0)
        compute_cost = gb_seconds * LAMBDA_COST_PER_GB_SECOND
        request_cost = LAMBDA_COST_PER_REQUEST
        total_cost = compute_cost + request_cost

        # Record execution details
        self.record_event(
            AgentEventType.AGENT_COMMUNICATION,
            {
                "request_id": request_id,
                "duration_ms": duration_ms,
                "memory_mb": memory_mb,
                "billed_duration_ms": billed_duration_ms,
                "compute_cost": compute_cost,
                "request_cost": request_cost,
                "total_cost": total_cost,
            },
        )

        # Send cost metric to CloudWatch
        cloudwatch.put_metric_data(
            Namespace="AgentOps/Costs",
            MetricData=[
                {
                    "MetricName": "LambdaExecutionCost",
                    "Value": total_cost,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "Agent", "Value": self.agent_name},
                        {"Name": "Environment", "Value": ENVIRONMENT},
                    ],
                }
            ],
        )

        return total_cost

    def track_llm_usage(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Track LLM API usage and costs"""
        # Get cost rates for model
        model_base = model.split("-20")[0] if "-20" in model else model
        rates = CLAUDE_COST_PER_1K_TOKENS.get(
            model_base, CLAUDE_COST_PER_1K_TOKENS["claude-3-sonnet"]
        )

        # Calculate costs
        input_cost = (input_tokens / 1000.0) * rates["input"]
        output_cost = (output_tokens / 1000.0) * rates["output"]
        total_cost = input_cost + output_cost

        # Record LLM usage
        self.record_event(
            (
                AgentEventType.CODE_GENERATION_STARTED
                if self.agent_name == "engineer"
                else AgentEventType.ANALYSIS_STARTED
            ),
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            },
        )

        # Update total cost
        self.total_cost += total_cost

        # Check for cost threshold
        if self.total_cost > float(os.environ.get("COST_ALERT_THRESHOLD", "10")):
            self.record_event(
                AgentEventType.COST_THRESHOLD_EXCEEDED,
                {
                    "total_cost": self.total_cost,
                    "threshold": float(os.environ.get("COST_ALERT_THRESHOLD", "10")),
                    "session_id": self.session_id,
                },
            )
            self._send_cost_alert()

        return total_cost

    def _calculate_event_cost(
        self, event_type: AgentEventType, details: Dict[str, Any]
    ) -> float:
        """Calculate cost for specific event types"""
        cost = 0.0

        # Add event-specific cost calculations
        if event_type == AgentEventType.CODE_GENERATED:
            # Estimate based on lines of code or complexity
            lines_of_code = details.get("lines_of_code", 0)
            cost = lines_of_code * 0.001  # $0.001 per line (example)
        elif (
            event_type == AgentEventType.TEST_PASSED
            or event_type == AgentEventType.TEST_FAILED
        ):
            # Test execution costs
            duration_seconds = details.get("duration_seconds", 0)
            cost = duration_seconds * 0.01  # $0.01 per second (example)

        return cost

    def _calculate_decision_cost(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """Calculate cost of a decision based on reasoning chain"""
        total_tokens = 0

        for step in reasoning_chain:
            # Estimate tokens from text length
            thought = step.get("thought", "")
            action = step.get("action", "")
            observation = step.get("observation", "")

            # Rough estimation: 1 token ≈ 4 characters
            total_tokens += (len(thought) + len(action) + len(observation)) / 4

        # Use default model cost
        rates = CLAUDE_COST_PER_1K_TOKENS["claude-3-sonnet"]
        cost = (total_tokens / 1000.0) * rates["output"]

        return cost

    def _store_event(self, event: AgentEvent):
        """Store event in DynamoDB"""
        try:
            item = {
                "event_id": event.event_id,
                "agent_name": event.agent_name,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
                "cost": Decimal(str(event.cost)) if event.cost else Decimal("0"),
                "duration_ms": event.duration_ms,
                "parent_event_id": event.parent_event_id,
                "session_id": self.session_id,
            }

            self.monitoring_table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Error storing event: {str(e)}")

    def _store_decision(self, decision: AgentDecision):
        """Store decision with reasoning chain in DynamoDB"""
        try:
            item = {
                "decision_id": decision.decision_id,
                "agent_name": decision.agent_name,
                "timestamp": decision.timestamp.isoformat(),
                "decision_type": decision.decision_type,
                "reasoning_chain": decision.reasoning_chain,
                "inputs": decision.inputs,
                "outputs": decision.outputs,
                "confidence_score": Decimal(str(decision.confidence_score)),
                "cost": Decimal(str(decision.cost)),
                "session_id": self.session_id,
            }

            self.decisions_table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Error storing decision: {str(e)}")

    def _send_cloudwatch_metric(self, event: AgentEvent):
        """Send custom metrics to CloudWatch"""
        try:
            # Send event count metric
            cloudwatch.put_metric_data(
                Namespace="AgentOps/Events",
                MetricData=[
                    {
                        "MetricName": event.event_type.value,
                        "Value": 1,
                        "Unit": "Count",
                        "Dimensions": [
                            {"Name": "Agent", "Value": self.agent_name},
                            {"Name": "Environment", "Value": ENVIRONMENT},
                        ],
                    }
                ],
            )

            # Send cost metric if applicable
            if event.cost:
                cloudwatch.put_metric_data(
                    Namespace="AgentOps/Costs",
                    MetricData=[
                        {
                            "MetricName": "EventCost",
                            "Value": event.cost,
                            "Unit": "None",
                            "Dimensions": [
                                {"Name": "Agent", "Value": self.agent_name},
                                {"Name": "EventType", "Value": event.event_type.value},
                                {"Name": "Environment", "Value": ENVIRONMENT},
                            ],
                        }
                    ],
                )
        except Exception as e:
            logger.error(f"Error sending CloudWatch metric: {str(e)}")

    def _publish_collaboration_event(self, collaboration_event: Dict[str, Any]):
        """Publish collaboration event to EventBridge"""
        try:
            eventbridge.put_events(
                Entries=[
                    {
                        "Source": f"agentops.{self.agent_name}",
                        "DetailType": "AgentCollaboration",
                        "Detail": json.dumps(collaboration_event, default=str),
                        "EventBusName": os.environ.get("EVENT_BUS_NAME", "default"),
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error publishing collaboration event: {str(e)}")

    def _send_cost_alert(self):
        """Send alert when cost threshold is exceeded"""
        try:
            # Send SNS notification
            sns = boto3.client("sns")
            topic_arn = os.environ.get("COST_ALERT_TOPIC_ARN")

            if topic_arn:
                sns.publish(
                    TopicArn=topic_arn,
                    Subject=f"AgentOps Cost Alert - {self.agent_name}",
                    Message=f"""
                    Agent: {self.agent_name}
                    Session: {self.session_id}
                    Total Cost: ${self.total_cost:.2f}
                    Threshold: ${os.environ.get('COST_ALERT_THRESHOLD', '10')}
                    Environment: {ENVIRONMENT}
                    
                    Please review agent activity and optimize if necessary.
                    """,
                )

            # Create CloudWatch alarm
            cloudwatch.put_metric_alarm(
                AlarmName=f"AgentOps-{self.agent_name}-CostOverrun",
                ComparisonOperator="GreaterThanThreshold",
                EvaluationPeriods=1,
                MetricName="TotalCost",
                Namespace="AgentOps/Costs",
                Period=300,
                Statistic="Sum",
                Threshold=float(os.environ.get("COST_ALERT_THRESHOLD", "10")),
                ActionsEnabled=True,
                AlarmDescription=f"Alert when {self.agent_name} agent costs exceed threshold",
                Dimensions=[
                    {"Name": "Agent", "Value": self.agent_name},
                    {"Name": "Environment", "Value": ENVIRONMENT},
                ],
            )
        except Exception as e:
            logger.error(f"Error sending cost alert: {str(e)}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current monitoring session"""
        duration = (datetime.utcnow() - self.start_time).total_seconds()

        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "total_cost": self.total_cost,
            "average_cost_per_event": (
                self.total_cost / len(self.events) if self.events else 0
            ),
        }

    def end_session(self):
        """End the monitoring session"""
        summary = self.get_session_summary()

        # Record session summary in AgentOps
        if self.agentops_session:
            agentops.record(
                agentops.Event(event_type="session_summary", params=summary)
            )
            agentops.end_session(
                state="Success", reason=f"Session completed for {self.agent_name}"
            )

        # Store session summary
        try:
            self.monitoring_table.put_item(
                Item={
                    "event_id": f"session-{self.session_id}",
                    "agent_name": self.agent_name,
                    "event_type": "session_summary",
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": summary,
                    "session_id": self.session_id,
                }
            )
        except Exception as e:
            logger.error(f"Error storing session summary: {str(e)}")

        logger.info(f"Monitoring session ended for {self.agent_name}: {summary}")

        return summary


class AgentOpsAnalyzer:
    """Analyzer for agent patterns and optimization opportunities"""

    def __init__(self):
        self.monitoring_table = dynamodb.Table(MONITORING_TABLE_NAME)
        self.decisions_table = dynamodb.Table(DECISIONS_TABLE_NAME)

    def analyze_collaboration_patterns(
        self, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze agent collaboration patterns"""
        start_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        try:
            # Query collaboration events
            response = self.monitoring_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr("event_type").eq(
                    "agent_communication"
                )
                & boto3.dynamodb.conditions.Attr("timestamp").gte(
                    start_time.isoformat()
                )
            )

            collaborations = response.get("Items", [])

            # Analyze patterns
            message_flows = {}
            agent_interactions = {}

            for event in collaborations:
                details = event.get("details", {})
                source = details.get("source_agent", "unknown")
                target = details.get("target_agent", "unknown")

                flow_key = f"{source}->{target}"
                message_flows[flow_key] = message_flows.get(flow_key, 0) + 1

                if source not in agent_interactions:
                    agent_interactions[source] = {"sent": 0, "received": 0}
                if target not in agent_interactions:
                    agent_interactions[target] = {"sent": 0, "received": 0}

                agent_interactions[source]["sent"] += 1
                agent_interactions[target]["received"] += 1

            return {
                "total_collaborations": len(collaborations),
                "message_flows": message_flows,
                "agent_interactions": agent_interactions,
                "most_active_flow": (
                    max(message_flows.items(), key=lambda x: x[1])
                    if message_flows
                    else None
                ),
                "time_window_hours": time_window_hours,
            }

        except Exception as e:
            logger.error(f"Error analyzing collaboration patterns: {str(e)}")
            return {}

    def analyze_agent_effectiveness(
        self, agent_name: str, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze individual agent effectiveness"""
        start_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        try:
            # Query agent events
            response = self.monitoring_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr("agent_name").eq(
                    agent_name
                )
                & boto3.dynamodb.conditions.Attr("timestamp").gte(
                    start_time.isoformat()
                )
            )

            events = response.get("Items", [])

            # Calculate metrics
            total_cost = sum(float(e.get("cost", 0)) for e in events)
            success_events = [
                e
                for e in events
                if "passed" in e.get("event_type", "").lower()
                or "completed" in e.get("event_type", "").lower()
            ]
            failure_events = [
                e
                for e in events
                if "failed" in e.get("event_type", "").lower()
                or "error" in e.get("event_type", "").lower()
            ]

            success_rate = len(success_events) / len(events) if events else 0

            # Query decisions
            decision_response = self.decisions_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr("agent_name").eq(
                    agent_name
                )
                & boto3.dynamodb.conditions.Attr("timestamp").gte(
                    start_time.isoformat()
                )
            )

            decisions = decision_response.get("Items", [])
            avg_confidence = (
                sum(float(d.get("confidence_score", 0)) for d in decisions)
                / len(decisions)
                if decisions
                else 0
            )

            return {
                "agent_name": agent_name,
                "total_events": len(events),
                "success_rate": success_rate,
                "total_cost": total_cost,
                "average_cost_per_event": total_cost / len(events) if events else 0,
                "total_decisions": len(decisions),
                "average_confidence": avg_confidence,
                "success_events": len(success_events),
                "failure_events": len(failure_events),
                "time_window_hours": time_window_hours,
            }

        except Exception as e:
            logger.error(f"Error analyzing agent effectiveness: {str(e)}")
            return {}

    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for cost and performance optimization"""
        opportunities = []

        try:
            # Analyze recent costs
            for agent in ["manager", "engineer", "qa"]:
                effectiveness = self.analyze_agent_effectiveness(agent, 24)

                if effectiveness.get("average_cost_per_event", 0) > 1.0:
                    opportunities.append(
                        {
                            "type": "cost_optimization",
                            "agent": agent,
                            "recommendation": f"High average cost per event (${effectiveness['average_cost_per_event']:.2f}). Consider optimizing LLM usage or using smaller models.",
                            "potential_savings": effectiveness["average_cost_per_event"]
                            * 0.3,  # Estimate 30% savings
                        }
                    )

                if effectiveness.get("success_rate", 1) < 0.8:
                    opportunities.append(
                        {
                            "type": "reliability_improvement",
                            "agent": agent,
                            "recommendation": f"Low success rate ({effectiveness['success_rate']:.1%}). Review error patterns and improve error handling.",
                            "impact": "high",
                        }
                    )

            # Analyze collaboration patterns
            patterns = self.analyze_collaboration_patterns(24)

            if patterns.get("total_collaborations", 0) > 100:
                opportunities.append(
                    {
                        "type": "workflow_optimization",
                        "recommendation": f"High collaboration volume ({patterns['total_collaborations']} messages). Consider batching or caching common requests.",
                        "potential_improvement": "20-30% reduction in inter-agent communication",
                    }
                )

            return opportunities

        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return []
