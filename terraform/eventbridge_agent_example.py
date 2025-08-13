#!/usr/bin/env python3
"""
EventBridge Agent Communication Example
Demonstrates how agents can publish and consume events using the EventBridge infrastructure
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import boto3


# ==================== EVENT TYPES ====================
class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IssueType(Enum):
    PERFORMANCE = "PERFORMANCE"
    ERROR = "ERROR"
    SECURITY = "SECURITY"
    DATA_QUALITY = "DATA_QUALITY"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class FeatureType(Enum):
    UI = "UI"
    API = "API"
    ML_MODEL = "ML_MODEL"
    DATA_PIPELINE = "DATA_PIPELINE"
    INTEGRATION = "INTEGRATION"
    PERFORMANCE = "PERFORMANCE"


class ChangeType(Enum):
    FEATURE = "FEATURE"
    BUGFIX = "BUGFIX"
    REFACTOR = "REFACTOR"
    DOCUMENTATION = "DOCUMENTATION"
    TEST = "TEST"
    DEPENDENCY = "DEPENDENCY"


class TestType(Enum):
    UNIT = "UNIT"
    INTEGRATION = "INTEGRATION"
    E2E = "E2E"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    REGRESSION = "REGRESSION"


# ==================== EVENT CLASSES ====================
@dataclass
class IssueDetectionEvent:
    """Event for issue detection by monitoring agents"""

    issueId: str
    timestamp: str
    severity: str
    agentId: str
    issueType: str
    description: str
    affectedResources: List[str] = None
    metrics: Dict[str, float] = None
    suggestedActions: List[str] = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        severity: Severity,
        issue_type: IssueType,
        description: str,
        **kwargs,
    ):
        return cls(
            issueId=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            severity=severity.value,
            agentId=agent_id,
            issueType=issue_type.value,
            description=description,
            **kwargs,
        )


@dataclass
class FeatureRequestEvent:
    """Event for feature requests from product agents"""

    requestId: str
    timestamp: str
    agentId: str
    featureType: str
    priority: str
    description: str
    businessJustification: str = None
    estimatedImpact: Dict[str, Any] = None
    technicalRequirements: List[str] = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        feature_type: FeatureType,
        priority: str,
        description: str,
        **kwargs,
    ):
        return cls(
            requestId=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            agentId=agent_id,
            featureType=feature_type.value,
            priority=priority,
            description=description,
            **kwargs,
        )


@dataclass
class CodeChangeEvent:
    """Event for code changes from development agents"""

    changeId: str
    timestamp: str
    agentId: str
    repository: str
    branch: str
    changeType: str
    status: str
    commitHash: str = None
    files: List[Dict[str, Any]] = None
    testResults: Dict[str, int] = None
    reviewers: List[str] = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        repository: str,
        branch: str,
        change_type: ChangeType,
        status: str,
        **kwargs,
    ):
        return cls(
            changeId=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            agentId=agent_id,
            repository=repository,
            branch=branch,
            changeType=change_type.value,
            status=status,
            **kwargs,
        )


@dataclass
class TestResultsEvent:
    """Event for test results from testing agents"""

    testRunId: str
    timestamp: str
    agentId: str
    testType: str
    status: str
    summary: Dict[str, Any]
    coverage: Dict[str, float] = None
    failedTests: List[Dict[str, str]] = None
    artifacts: List[str] = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        test_type: TestType,
        status: str,
        summary: Dict[str, Any],
        **kwargs,
    ):
        return cls(
            testRunId=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            agentId=agent_id,
            testType=test_type.value,
            status=status,
            summary=summary,
            **kwargs,
        )


# ==================== EVENT BRIDGE CLIENT ====================
class EventBridgeAgent:
    """Client for agent communication via EventBridge"""

    def __init__(
        self,
        agent_id: str,
        event_bus_name: str = "cabruca-agents-bus",
        region: str = "sa-east-1",
    ):
        self.agent_id = agent_id
        self.event_bus_name = event_bus_name
        self.client = boto3.client("events", region_name=region)

    def publish_issue_detection(self, event: IssueDetectionEvent) -> Dict[str, Any]:
        """Publish an issue detection event"""
        return self._publish_event(
            source="cabruca.agents.monitor",
            detail_type="Issue Detection Event",
            detail=event,
        )

    def publish_feature_request(self, event: FeatureRequestEvent) -> Dict[str, Any]:
        """Publish a feature request event"""
        return self._publish_event(
            source="cabruca.agents.product",
            detail_type="Feature Request Event",
            detail=event,
        )

    def publish_code_change(self, event: CodeChangeEvent) -> Dict[str, Any]:
        """Publish a code change event"""
        return self._publish_event(
            source="cabruca.agents.developer",
            detail_type="Code Change Event",
            detail=event,
        )

    def publish_test_results(self, event: TestResultsEvent) -> Dict[str, Any]:
        """Publish test results event"""
        return self._publish_event(
            source="cabruca.agents.tester",
            detail_type="Test Results Event",
            detail=event,
        )

    def _publish_event(
        self, source: str, detail_type: str, detail: Any
    ) -> Dict[str, Any]:
        """Internal method to publish events to EventBridge"""
        # Convert dataclass to dict, removing None values
        detail_dict = {k: v for k, v in asdict(detail).items() if v is not None}

        entry = {
            "Source": source,
            "DetailType": detail_type,
            "Detail": json.dumps(detail_dict),
            "EventBusName": self.event_bus_name,
        }

        try:
            response = self.client.put_events(Entries=[entry])

            if response["FailedEntryCount"] > 0:
                raise Exception(f"Failed to publish event: {response['Entries'][0]}")

            return {
                "success": True,
                "eventId": response["Entries"][0].get("EventId"),
                "response": response,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ==================== USAGE EXAMPLES ====================
def example_monitoring_agent():
    """Example of a monitoring agent detecting and reporting issues"""
    agent = EventBridgeAgent(agent_id="monitor-agent-001")

    # Detect a performance issue
    issue = IssueDetectionEvent.create(
        agent_id="monitor-agent-001",
        severity=Severity.HIGH,
        issue_type=IssueType.PERFORMANCE,
        description="API response time exceeds 5 seconds for /api/segment endpoint",
        affectedResources=["ecs-task-api-001", "alb-cabruca-prod"],
        metrics={
            "response_time_p95": 5.2,
            "response_time_p99": 8.1,
            "error_rate": 0.02,
        },
        suggestedActions=[
            "Scale up ECS tasks",
            "Investigate database query performance",
            "Enable caching for frequently accessed data",
        ],
    )

    result = agent.publish_issue_detection(issue)
    print(f"Issue detection published: {result}")


def example_product_agent():
    """Example of a product agent creating feature requests"""
    agent = EventBridgeAgent(agent_id="product-agent-001")

    # Create a feature request
    feature = FeatureRequestEvent.create(
        agent_id="product-agent-001",
        feature_type=FeatureType.ML_MODEL,
        priority="HIGH",
        description="Implement multi-crop detection in single image",
        businessJustification="Users report 30% of images contain multiple crop types",
        estimatedImpact={"users": 500, "revenue": 50000, "efficiency": 25.5},
        technicalRequirements=[
            "Update ML model architecture",
            "Modify API to return multiple segments",
            "Update UI to display multiple results",
        ],
    )

    result = agent.publish_feature_request(feature)
    print(f"Feature request published: {result}")


def example_developer_agent():
    """Example of a developer agent reporting code changes"""
    agent = EventBridgeAgent(agent_id="dev-agent-001")

    # Report a code change
    change = CodeChangeEvent.create(
        agent_id="dev-agent-001",
        repository="cabruca-segmentation",
        branch="feature/multi-crop-detection",
        change_type=ChangeType.FEATURE,
        status="IN_REVIEW",
        commitHash="abc123def456",
        files=[
            {"path": "src/models/multi_crop.py", "additions": 250, "deletions": 10},
            {"path": "src/api/endpoints.py", "additions": 45, "deletions": 5},
            {"path": "tests/test_multi_crop.py", "additions": 180, "deletions": 0},
        ],
        reviewers=["senior-dev-agent-001", "qa-agent-001"],
    )

    result = agent.publish_code_change(change)
    print(f"Code change published: {result}")


def example_testing_agent():
    """Example of a testing agent reporting test results"""
    agent = EventBridgeAgent(agent_id="test-agent-001")

    # Report test results
    test_results = TestResultsEvent.create(
        agent_id="test-agent-001",
        test_type=TestType.INTEGRATION,
        status="PARTIALLY_PASSED",
        summary={
            "total": 150,
            "passed": 145,
            "failed": 3,
            "skipped": 2,
            "duration": 320.5,
        },
        coverage={"lines": 85.2, "branches": 78.5, "functions": 92.1},
        failedTests=[
            {
                "name": "test_multi_crop_edge_case",
                "error": "AssertionError",
                "stackTrace": "File test_multi_crop.py, line 145...",
            },
            {
                "name": "test_api_timeout",
                "error": "TimeoutError",
                "stackTrace": "File test_api.py, line 89...",
            },
        ],
        artifacts=[
            "s3://cabruca-test-artifacts/runs/2024-01-15/test-report.html",
            "s3://cabruca-test-artifacts/runs/2024-01-15/coverage.xml",
        ],
    )

    result = agent.publish_test_results(test_results)
    print(f"Test results published: {result}")


def example_cost_monitoring():
    """Example of monitoring EventBridge usage to stay under $10/month"""
    cloudwatch = boto3.client("cloudwatch", region_name="sa-east-1")

    # Get EventBridge metrics for the last 30 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)

    response = cloudwatch.get_metric_statistics(
        Namespace="AWS/Events",
        MetricName="SuccessfulEventsMatched",
        Dimensions=[{"Name": "EventBusName", "Value": "cabruca-agents-bus"}],
        StartTime=start_time,
        EndTime=end_time,
        Period=2592000,  # 30 days in seconds
        Statistics=["Sum"],
    )

    if response["Datapoints"]:
        total_events = response["Datapoints"][0]["Sum"]
        estimated_cost = (
            total_events / 1_000_000
        ) * 1.64  # $1 for publishing + $0.64 for matching

        print(f"Total events in last 30 days: {total_events:,.0f}")
        print(f"Estimated cost: ${estimated_cost:.2f}")

        if estimated_cost > 9:
            print("WARNING: Approaching $10/month limit!")
            # Implement throttling or disable non-critical event types

        # Calculate remaining budget
        remaining_events = (10 / 1.64) * 1_000_000 - total_events
        print(f"Remaining events in budget: {remaining_events:,.0f}")


if __name__ == "__main__":
    print("EventBridge Agent Communication Examples")
    print("=" * 50)

    # Note: These examples require AWS credentials and the infrastructure to be deployed
    # Uncomment to run:

    # example_monitoring_agent()
    # example_product_agent()
    # example_developer_agent()
    # example_testing_agent()
    # example_cost_monitoring()

    print("\nTo run these examples:")
    print("1. Deploy the Terraform infrastructure: terraform apply")
    print("2. Configure AWS credentials")
    print("3. Uncomment the example functions above")
    print("4. Run: python eventbridge_agent_example.py")
