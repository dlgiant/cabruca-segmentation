import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List

import boto3

# Initialize AWS clients
dynamodb = boto3.resource("dynamodb")
events = boto3.client("events")
lambda_client = boto3.client("lambda")
s3 = boto3.client("s3")


class ManagerAgent:
    """Manager Agent for orchestrating multi-agent workflows"""

    def __init__(self):
        self.agent_type = os.environ.get("AGENT_TYPE", "MANAGER")
        self.environment = os.environ.get("ENVIRONMENT", "mvp")
        self.event_bus = os.environ.get("EVENT_BUS_NAME", "cabruca-agents-bus")
        self.state_table = os.environ.get(
            "DYNAMODB_STATE_TABLE", f"cabruca-mvp-{self.environment}-agent-state"
        )
        self.tasks_table = os.environ.get(
            "DYNAMODB_TASKS_TABLE", f"cabruca-mvp-{self.environment}-agent-tasks"
        )

    def detect_pipeline_issues(self, pipeline_data: Dict) -> List[Dict]:
        """Detect issues in git pipeline data"""
        issues = []

        # Check for failed tests
        if pipeline_data.get("test_status") == "failed":
            issues.append(
                {
                    "type": "test_failure",
                    "severity": "high",
                    "details": pipeline_data.get("test_details", {}),
                    "action_required": "fix_tests",
                }
            )

        # Check for build failures
        if pipeline_data.get("build_status") == "failed":
            issues.append(
                {
                    "type": "build_failure",
                    "severity": "critical",
                    "details": pipeline_data.get("build_details", {}),
                    "action_required": "fix_build",
                }
            )

        # Check for code quality issues
        quality_score = pipeline_data.get("quality_score", 100)
        if quality_score < 80:
            issues.append(
                {
                    "type": "quality_issue",
                    "severity": "medium",
                    "score": quality_score,
                    "details": pipeline_data.get("quality_details", {}),
                    "action_required": "improve_quality",
                }
            )

        # Check for security vulnerabilities
        if pipeline_data.get("security_vulnerabilities", 0) > 0:
            issues.append(
                {
                    "type": "security_vulnerability",
                    "severity": "critical",
                    "count": pipeline_data.get("security_vulnerabilities"),
                    "details": pipeline_data.get("security_details", {}),
                    "action_required": "fix_security",
                }
            )

        # Check for missing tests
        coverage = pipeline_data.get("test_coverage", 100)
        if coverage < 70:
            issues.append(
                {
                    "type": "low_test_coverage",
                    "severity": "medium",
                    "coverage": coverage,
                    "action_required": "add_tests",
                }
            )

        return issues

    def orchestrate_fixes(self, issues: List[Dict], repo_info: Dict) -> Dict:
        """Orchestrate fixes by delegating to appropriate agents"""
        workflows = []

        for issue in issues:
            workflow_id = (
                f"fix-{issue['type']}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

            # Determine which agents to involve
            agents_needed = []

            if issue["type"] in [
                "build_failure",
                "quality_issue",
                "security_vulnerability",
            ]:
                agents_needed.append("engineer")

            if issue["type"] in ["test_failure", "low_test_coverage"]:
                agents_needed.append("qa")

            # For critical issues, involve both agents
            if issue["severity"] == "critical":
                if "engineer" not in agents_needed:
                    agents_needed.append("engineer")
                if "qa" not in agents_needed:
                    agents_needed.append("qa")

            # Create workflow
            workflow = {
                "workflow_id": workflow_id,
                "issue": issue,
                "repo_info": repo_info,
                "agents_assigned": agents_needed,
                "status": "initiated",
                "created_at": datetime.now().isoformat(),
            }

            # Store workflow in DynamoDB
            self.store_workflow(workflow)

            # Dispatch tasks to agents via direct Lambda invocation
            for agent in agents_needed:
                self.dispatch_to_agent(agent, workflow_id, issue, repo_info)

            workflows.append(workflow)

        return {"workflows_created": len(workflows), "workflows": workflows}

    def dispatch_to_agent(
        self, agent_type: str, workflow_id: str, issue: Dict, repo_info: Dict
    ):
        """Dispatch task to specific agent via Lambda invocation"""

        # Prepare the event based on agent type
        if agent_type == "engineer":
            event_detail = {
                "action": "create_fix_pr",
                "workflow_id": workflow_id,
                "issue": issue,
                "repo_info": repo_info,
                "pr_details": {
                    "title": f"Fix: {issue['type'].replace('_', ' ').title()}",
                    "branch": f"fix/{issue['type']}-{workflow_id[:8]}",
                    "description": f"Automated fix for {issue['type']} detected in pipeline",
                },
            }
            function_name = f"cabruca-mvp-{self.environment}-engineer-agent"

        elif agent_type == "qa":
            event_detail = {
                "action": "create_test_pr",
                "workflow_id": workflow_id,
                "issue": issue,
                "repo_info": repo_info,
                "pr_details": {
                    "title": f"Tests: Add tests for {issue['type'].replace('_', ' ').title()}",
                    "branch": f"test/{issue['type']}-{workflow_id[:8]}",
                    "description": f"Adding tests to address {issue['type']}",
                },
            }
            function_name = f"cabruca-mvp-{self.environment}-qa-agent"
        else:
            return

        # Invoke agent Lambda directly
        try:
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType="Event",  # Async invocation
                Payload=json.dumps(event_detail),
            )
            print(f"Dispatched task to {agent_type} agent: {workflow_id}")
        except Exception as e:
            print(f"Error dispatching to {agent_type}: {str(e)}")

    def store_workflow(self, workflow: Dict):
        """Store workflow in DynamoDB"""
        try:
            table = dynamodb.Table(self.tasks_table)
            table.put_item(Item=workflow)
        except Exception as e:
            print(f"Error storing workflow: {str(e)}")

    def monitor_pipeline(self, event: Dict) -> Dict:
        """Main pipeline monitoring and orchestration logic"""

        # Extract pipeline data
        pipeline_data = event.get("pipeline_data", {})
        repo_info = event.get(
            "repo_info",
            {"owner": "dlgiant", "repo": "cabruca-segmentation", "branch": "main"},
        )

        # Detect issues
        issues = self.detect_pipeline_issues(pipeline_data)

        if not issues:
            return {
                "status": "healthy",
                "message": "No issues detected in pipeline",
                "timestamp": datetime.now().isoformat(),
            }

        # Orchestrate fixes
        orchestration_result = self.orchestrate_fixes(issues, repo_info)

        return {
            "status": "issues_detected",
            "issues_found": len(issues),
            "issues": issues,
            "orchestration": orchestration_result,
            "timestamp": datetime.now().isoformat(),
        }


def lambda_handler(event, context):
    """Manager Agent Lambda Handler"""

    manager = ManagerAgent()

    try:
        # Parse event
        action = event.get("action", "default")

        # Health check
        if action == "health_check":
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "status": "healthy",
                        "agent": manager.agent_type,
                        "environment": manager.environment,
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
            }

        # Monitor pipeline
        elif action == "monitor_pipeline":
            result = manager.monitor_pipeline(event)
            return {"statusCode": 200, "body": json.dumps(result)}

        # Default response
        else:
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": f"{manager.agent_type} agent processed request",
                        "action": action,
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
            }

    except Exception as e:
        print(f"Error in manager agent: {str(e)}")
        print(traceback.format_exc())

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": str(e),
                    "agent": manager.agent_type,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
        }
