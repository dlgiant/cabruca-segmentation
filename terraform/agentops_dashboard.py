"""
AgentOps CloudWatch Dashboard Generator
Creates comprehensive dashboards for agent monitoring and visualization
"""

import json
from typing import Any, Dict, List

import boto3

cloudwatch = boto3.client("cloudwatch")


class AgentOpsDashboard:
    """Creates and manages CloudWatch dashboards for AgentOps monitoring"""

    def __init__(self, dashboard_name: str = "AgentOps-Monitoring"):
        self.dashboard_name = dashboard_name
        self.region = boto3.Session().region_name

    def create_main_dashboard(self) -> Dict[str, Any]:
        """Create the main AgentOps monitoring dashboard"""

        dashboard_body = {
            "widgets": [
                # Agent Activity Overview
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Events",
                                "issue_detected",
                                {"stat": "Sum", "label": "Issues Detected"},
                            ],
                            [
                                ".",
                                "code_generated",
                                {"stat": "Sum", "label": "Code Generated"},
                            ],
                            [
                                ".",
                                "test_passed",
                                {"stat": "Sum", "label": "Tests Passed"},
                            ],
                            [
                                ".",
                                "test_failed",
                                {"stat": "Sum", "label": "Tests Failed"},
                            ],
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Agent Activity Overview",
                        "period": 300,
                    },
                },
                # Cost Tracking
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Costs",
                                "LambdaExecutionCost",
                                {"stat": "Sum", "label": "Lambda Cost"},
                            ],
                            [
                                ".",
                                "EventCost",
                                {"stat": "Sum", "label": "Event Processing Cost"},
                            ],
                            [".", "TotalCost", {"stat": "Sum", "label": "Total Cost"}],
                        ],
                        "view": "timeSeries",
                        "stacked": True,
                        "region": self.region,
                        "title": "Agent Costs Over Time",
                        "period": 300,
                        "yAxis": {"left": {"label": "Cost ($)", "showUnits": False}},
                    },
                },
                # Agent Effectiveness
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AgentOps/Events", "decision_made", {"stat": "Sum"}],
                            [".", "workflow_completed", {"stat": "Sum"}],
                            [".", "workflow_started", {"stat": "Sum"}],
                        ],
                        "view": "singleValue",
                        "region": self.region,
                        "title": "Agent Decision Metrics",
                        "period": 3600,
                    },
                },
                # Agent Collaboration Patterns
                {
                    "type": "metric",
                    "x": 8,
                    "y": 6,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Events",
                                "agent_communication",
                                {"stat": "Sum", "dimensions": {"Agent": "manager"}},
                            ],
                            ["...", {"dimensions": {"Agent": "engineer"}}],
                            ["...", {"dimensions": {"Agent": "qa"}}],
                        ],
                        "view": "pie",
                        "region": self.region,
                        "title": "Agent Communication Distribution",
                        "period": 3600,
                    },
                },
                # Error Rate Tracking
                {
                    "type": "metric",
                    "x": 16,
                    "y": 6,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AWS/Lambda",
                                "Errors",
                                {"stat": "Sum", "label": "Lambda Errors"},
                            ],
                            [".", "Throttles", {"stat": "Sum", "label": "Throttles"}],
                            [
                                "AgentOps/Events",
                                "test_failed",
                                {"stat": "Sum", "label": "Test Failures"},
                            ],
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Error Tracking",
                        "period": 300,
                        "yAxis": {"left": {"showUnits": False}},
                    },
                },
                # Performance Metrics
                {
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AWS/Lambda",
                                "Duration",
                                {
                                    "stat": "Average",
                                    "dimensions": {"FunctionName": "manager-agent"},
                                },
                            ],
                            ["...", {"dimensions": {"FunctionName": "engineer-agent"}}],
                            ["...", {"dimensions": {"FunctionName": "qa-agent"}}],
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Agent Execution Duration",
                        "period": 300,
                        "yAxis": {
                            "left": {"label": "Duration (ms)", "showUnits": False}
                        },
                    },
                },
                # Cost Breakdown by Agent
                {
                    "type": "metric",
                    "x": 12,
                    "y": 12,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Costs",
                                "EventCost",
                                {
                                    "stat": "Sum",
                                    "dimensions": {"Agent": "manager"},
                                    "label": "Manager Agent",
                                },
                            ],
                            [
                                "...",
                                {
                                    "dimensions": {"Agent": "engineer"},
                                    "label": "Engineer Agent",
                                },
                            ],
                            [
                                "...",
                                {"dimensions": {"Agent": "qa"}, "label": "QA Agent"},
                            ],
                        ],
                        "view": "barChart",
                        "region": self.region,
                        "title": "Cost Breakdown by Agent",
                        "period": 3600,
                        "yAxis": {"left": {"label": "Cost ($)", "showUnits": False}},
                    },
                },
                # Success Rate
                {
                    "type": "metric",
                    "x": 0,
                    "y": 18,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                {
                                    "expression": "m1 / (m1 + m2) * 100",
                                    "label": "Success Rate %",
                                }
                            ],
                            [
                                "AgentOps/Events",
                                "test_passed",
                                {"stat": "Sum", "id": "m1", "visible": False},
                            ],
                            [
                                ".",
                                "test_failed",
                                {"stat": "Sum", "id": "m2", "visible": False},
                            ],
                        ],
                        "view": "singleValue",
                        "region": self.region,
                        "title": "Overall Success Rate",
                        "period": 3600,
                    },
                },
                # Recent Alerts
                {
                    "type": "log",
                    "x": 8,
                    "y": 18,
                    "width": 16,
                    "height": 6,
                    "properties": {
                        "query": f"SOURCE '/aws/lambda/manager-agent' | SOURCE '/aws/lambda/engineer-agent' | SOURCE '/aws/lambda/qa-agent' | fields @timestamp, @message | filter @message like /ERROR|ALERT|CRITICAL/ | sort @timestamp desc | limit 20",
                        "region": self.region,
                        "title": "Recent Alerts and Errors",
                        "queryLanguage": "kusto",
                    },
                },
            ]
        }

        return dashboard_body

    def create_cost_dashboard(self) -> Dict[str, Any]:
        """Create detailed cost analysis dashboard"""

        dashboard_body = {
            "widgets": [
                # Total Cost Trend
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Costs",
                                "TotalCost",
                                {"stat": "Sum", "period": 3600},
                            ]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Total Agent Costs - Hourly",
                        "period": 3600,
                        "annotations": {
                            "horizontal": [
                                {
                                    "label": "Cost Alert Threshold",
                                    "value": 10,
                                    "fill": "above",
                                }
                            ]
                        },
                    },
                },
                # Lambda vs LLM Costs
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Costs",
                                "LambdaExecutionCost",
                                {"stat": "Sum", "label": "Lambda"},
                            ],
                            [
                                ".",
                                "EventCost",
                                {
                                    "stat": "Sum",
                                    "label": "LLM API",
                                    "dimensions": {
                                        "EventType": "code_generation_started"
                                    },
                                },
                            ],
                        ],
                        "view": "pie",
                        "region": self.region,
                        "title": "Cost Distribution: Infrastructure vs AI",
                        "period": 86400,
                    },
                },
                # Cost per Event Type
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Costs",
                                "EventCost",
                                {
                                    "stat": "Average",
                                    "dimensions": {"EventType": "issue_detected"},
                                },
                            ],
                            ["...", {"dimensions": {"EventType": "code_generated"}}],
                            ["...", {"dimensions": {"EventType": "test_passed"}}],
                            ["...", {"dimensions": {"EventType": "pr_created"}}],
                        ],
                        "view": "barChart",
                        "region": self.region,
                        "title": "Average Cost per Event Type",
                        "period": 3600,
                    },
                },
                # Cost Optimization Opportunities
                {
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Events",
                                "cost_optimization_suggested",
                                {"stat": "Sum"},
                            ],
                            [".", "cost_threshold_exceeded", {"stat": "Sum"}],
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Cost Alerts and Optimization Events",
                        "period": 300,
                    },
                },
            ]
        }

        return dashboard_body

    def create_collaboration_dashboard(self) -> Dict[str, Any]:
        """Create agent collaboration patterns dashboard"""

        dashboard_body = {
            "widgets": [
                # Message Flow Heatmap
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 24,
                    "height": 8,
                    "properties": {
                        "metrics": [
                            [
                                "AgentOps/Events",
                                "agent_communication",
                                {
                                    "stat": "Sum",
                                    "dimensions": {"Agent": "manager"},
                                    "label": "Manager → Others",
                                },
                            ],
                            [
                                "...",
                                {
                                    "dimensions": {"Agent": "engineer"},
                                    "label": "Engineer → Others",
                                },
                            ],
                            [
                                "...",
                                {"dimensions": {"Agent": "qa"}, "label": "QA → Others"},
                            ],
                        ],
                        "view": "timeSeries",
                        "stacked": True,
                        "region": self.region,
                        "title": "Agent Communication Flow Over Time",
                        "period": 300,
                    },
                },
                # Workflow Completion Rate
                {
                    "type": "metric",
                    "x": 0,
                    "y": 8,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                {
                                    "expression": "m2 / m1 * 100",
                                    "label": "Completion Rate %",
                                }
                            ],
                            [
                                "AgentOps/Events",
                                "workflow_started",
                                {"stat": "Sum", "id": "m1", "visible": False},
                            ],
                            [
                                ".",
                                "workflow_completed",
                                {"stat": "Sum", "id": "m2", "visible": False},
                            ],
                        ],
                        "view": "singleValue",
                        "region": self.region,
                        "title": "Workflow Completion Rate",
                        "period": 3600,
                    },
                },
                # Average Workflow Duration
                {
                    "type": "metric",
                    "x": 12,
                    "y": 8,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [
                                "AWS/Lambda",
                                "Duration",
                                {"stat": "Average", "label": "Avg Duration"},
                            ]
                        ],
                        "view": "gauge",
                        "region": self.region,
                        "title": "Average Workflow Duration",
                        "period": 3600,
                        "yAxis": {"left": {"min": 0, "max": 10000}},
                    },
                },
                # EventBridge Message Volume
                {
                    "type": "metric",
                    "x": 0,
                    "y": 14,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/Events", "SuccessfulRuleMatches", {"stat": "Sum"}],
                            [".", "FailedInvocations", {"stat": "Sum"}],
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "EventBridge Message Flow",
                        "period": 300,
                    },
                },
            ]
        }

        return dashboard_body

    def deploy_dashboard(self, dashboard_type: str = "main") -> bool:
        """Deploy a dashboard to CloudWatch"""

        try:
            if dashboard_type == "main":
                dashboard_body = self.create_main_dashboard()
                name = self.dashboard_name
            elif dashboard_type == "cost":
                dashboard_body = self.create_cost_dashboard()
                name = f"{self.dashboard_name}-Cost"
            elif dashboard_type == "collaboration":
                dashboard_body = self.create_collaboration_dashboard()
                name = f"{self.dashboard_name}-Collaboration"
            else:
                raise ValueError(f"Unknown dashboard type: {dashboard_type}")

            response = cloudwatch.put_dashboard(
                DashboardName=name, DashboardBody=json.dumps(dashboard_body)
            )

            print(f"Dashboard '{name}' deployed successfully")
            return True

        except Exception as e:
            print(f"Error deploying dashboard: {str(e)}")
            return False

    def deploy_all_dashboards(self):
        """Deploy all AgentOps dashboards"""

        dashboards = ["main", "cost", "collaboration"]

        for dashboard_type in dashboards:
            success = self.deploy_dashboard(dashboard_type)
            if not success:
                print(f"Failed to deploy {dashboard_type} dashboard")
                return False

        print("All dashboards deployed successfully")
        return True

    def create_alarms(self) -> List[Dict[str, Any]]:
        """Create CloudWatch alarms for critical metrics"""

        alarms = [
            {
                "name": "AgentOps-HighCost",
                "metric": "TotalCost",
                "namespace": "AgentOps/Costs",
                "threshold": 10.0,
                "comparison": "GreaterThanThreshold",
                "description": "Alert when total agent costs exceed $10",
            },
            {
                "name": "AgentOps-HighErrorRate",
                "metric": "test_failed",
                "namespace": "AgentOps/Events",
                "threshold": 5.0,
                "comparison": "GreaterThanThreshold",
                "description": "Alert when test failures exceed 5 in 5 minutes",
            },
            {
                "name": "AgentOps-LowSuccessRate",
                "expression": "m1 / (m1 + m2) * 100",
                "metrics": [
                    {
                        "id": "m1",
                        "metric": "test_passed",
                        "namespace": "AgentOps/Events",
                    },
                    {
                        "id": "m2",
                        "metric": "test_failed",
                        "namespace": "AgentOps/Events",
                    },
                ],
                "threshold": 80.0,
                "comparison": "LessThanThreshold",
                "description": "Alert when success rate drops below 80%",
            },
        ]

        created_alarms = []

        for alarm_config in alarms:
            try:
                if "expression" in alarm_config:
                    # Math expression alarm
                    cloudwatch.put_metric_alarm(
                        AlarmName=alarm_config["name"],
                        ComparisonOperator=alarm_config["comparison"],
                        EvaluationPeriods=1,
                        Threshold=alarm_config["threshold"],
                        ActionsEnabled=True,
                        AlarmDescription=alarm_config["description"],
                        Metrics=[
                            {
                                "Id": m["id"],
                                "MetricStat": {
                                    "Metric": {
                                        "Namespace": m["namespace"],
                                        "MetricName": m["metric"],
                                    },
                                    "Period": 300,
                                    "Stat": "Sum",
                                },
                            }
                            for m in alarm_config["metrics"]
                        ]
                        + [{"Id": "e1", "Expression": alarm_config["expression"]}],
                    )
                else:
                    # Simple metric alarm
                    cloudwatch.put_metric_alarm(
                        AlarmName=alarm_config["name"],
                        ComparisonOperator=alarm_config["comparison"],
                        EvaluationPeriods=1,
                        MetricName=alarm_config["metric"],
                        Namespace=alarm_config["namespace"],
                        Period=300,
                        Statistic="Sum",
                        Threshold=alarm_config["threshold"],
                        ActionsEnabled=True,
                        AlarmDescription=alarm_config["description"],
                    )

                created_alarms.append(alarm_config["name"])
                print(f"Created alarm: {alarm_config['name']}")

            except Exception as e:
                print(f"Error creating alarm {alarm_config['name']}: {str(e)}")

        return created_alarms


if __name__ == "__main__":
    # Deploy dashboards when script is run directly
    dashboard_manager = AgentOpsDashboard()
    dashboard_manager.deploy_all_dashboards()
    dashboard_manager.create_alarms()
