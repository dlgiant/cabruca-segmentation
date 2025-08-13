"""
AWS Service Tools for LangChain Agents
Custom tools for interacting with AWS services (CloudWatch, Cost Explorer, CodeBuild)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# AWS Clients
cloudwatch = boto3.client("cloudwatch")
ce_client = boto3.client("ce")
codebuild = boto3.client("codebuild")
ecs = boto3.client("ecs")
lambda_client = boto3.client("lambda")
logs = boto3.client("logs")


class CloudWatchMetricsTool(BaseTool):
    """Tool for fetching and analyzing CloudWatch metrics"""

    name = "cloudwatch_metrics"
    description = """Fetch CloudWatch metrics for monitoring system health.
    Input should be a JSON string with:
    - namespace: AWS service namespace (e.g., 'AWS/Lambda', 'AWS/ApiGateway')
    - metric_name: Name of the metric to fetch
    - dimensions: Optional dimensions to filter metrics
    - period: Period in seconds (default: 300)
    - minutes_back: How many minutes of history to fetch (default: 30)
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Fetch CloudWatch metrics based on query parameters"""
        try:
            params = json.loads(query)
            namespace = params.get("namespace", "AWS/Lambda")
            metric_name = params.get("metric_name", "Invocations")
            dimensions = params.get("dimensions", [])
            period = params.get("period", 300)
            minutes_back = params.get("minutes_back", 30)

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=minutes_back)

            kwargs = {
                "Namespace": namespace,
                "MetricName": metric_name,
                "StartTime": start_time,
                "EndTime": end_time,
                "Period": period,
                "Statistics": ["Average", "Sum", "Maximum", "Minimum"],
            }

            if dimensions:
                kwargs["Dimensions"] = dimensions

            response = cloudwatch.get_metric_statistics(**kwargs)

            # Process datapoints
            datapoints = response.get("Datapoints", [])
            if datapoints:
                sorted_points = sorted(datapoints, key=lambda x: x["Timestamp"])

                result = {
                    "metric": metric_name,
                    "namespace": namespace,
                    "datapoints": len(sorted_points),
                    "latest_value": sorted_points[-1] if sorted_points else None,
                    "statistics": {
                        "avg": (
                            sum(p.get("Average", 0) for p in sorted_points)
                            / len(sorted_points)
                            if sorted_points
                            else 0
                        ),
                        "max": (
                            max(p.get("Maximum", 0) for p in sorted_points)
                            if sorted_points
                            else 0
                        ),
                        "min": (
                            min(p.get("Minimum", float("inf")) for p in sorted_points)
                            if sorted_points
                            else 0
                        ),
                    },
                }

                return json.dumps(result, default=str)
            else:
                return json.dumps(
                    {"error": "No datapoints found", "metric": metric_name}
                )

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class CostExplorerTool(BaseTool):
    """Tool for analyzing AWS costs and detecting anomalies"""

    name = "cost_explorer"
    description = """Analyze AWS costs and detect spending anomalies.
    Input should be a JSON string with:
    - operation: 'current_month', 'compare_months', 'by_service', 'forecast'
    - days_back: Number of days to analyze (default: 30)
    - group_by: Dimension to group costs by (default: 'SERVICE')
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Analyze AWS costs based on query parameters"""
        try:
            params = json.loads(query)
            operation = params.get("operation", "current_month")
            days_back = params.get("days_back", 30)
            group_by = params.get("group_by", "SERVICE")

            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days_back)

            if operation == "current_month":
                return self._get_current_month_costs()
            elif operation == "compare_months":
                return self._compare_monthly_costs()
            elif operation == "by_service":
                return self._get_costs_by_service(start_date, end_date, group_by)
            elif operation == "forecast":
                return self._forecast_costs()
            else:
                return json.dumps({"error": f"Unknown operation: {operation}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_current_month_costs(self) -> str:
        """Get current month's costs"""
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date.replace(day=1)

            response = ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="DAILY",
                Metrics=["UnblendedCost"],
            )

            total_cost = 0
            daily_costs = []

            for result in response.get("ResultsByTime", []):
                amount = float(result["Total"]["UnblendedCost"]["Amount"])
                total_cost += amount
                daily_costs.append(
                    {"date": result["TimePeriod"]["Start"], "cost": amount}
                )

            return json.dumps(
                {
                    "total_cost": total_cost,
                    "daily_costs": daily_costs,
                    "days_analyzed": len(daily_costs),
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _compare_monthly_costs(self) -> str:
        """Compare current month with previous month"""
        try:
            end_date = datetime.utcnow().date()
            current_start = end_date.replace(day=1)
            prev_end = current_start - timedelta(days=1)
            prev_start = prev_end.replace(day=1)

            # Current month
            current_response = ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": current_start.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="MONTHLY",
                Metrics=["UnblendedCost"],
            )

            # Previous month
            prev_response = ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": prev_start.isoformat(),
                    "End": prev_end.isoformat(),
                },
                Granularity="MONTHLY",
                Metrics=["UnblendedCost"],
            )

            current_cost = float(
                current_response["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
            )
            prev_cost = float(
                prev_response["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
            )

            change = current_cost - prev_cost
            change_percent = (change / prev_cost * 100) if prev_cost > 0 else 0

            return json.dumps(
                {
                    "current_month_cost": current_cost,
                    "previous_month_cost": prev_cost,
                    "change": change,
                    "change_percent": change_percent,
                    "trend": (
                        "increasing"
                        if change > 0
                        else "decreasing" if change < 0 else "stable"
                    ),
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_costs_by_service(self, start_date, end_date, group_by) -> str:
        """Get costs broken down by service"""
        try:
            response = ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="MONTHLY",
                Metrics=["UnblendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": group_by}],
            )

            service_costs = {}
            total_cost = 0

            for result in response.get("ResultsByTime", []):
                for group in result.get("Groups", []):
                    service = group["Keys"][0]
                    cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
                    service_costs[service] = cost
                    total_cost += cost

            # Sort by cost
            sorted_services = sorted(
                service_costs.items(), key=lambda x: x[1], reverse=True
            )

            return json.dumps(
                {
                    "total_cost": total_cost,
                    "service_breakdown": dict(sorted_services[:10]),  # Top 10 services
                    "top_service": sorted_services[0] if sorted_services else None,
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _forecast_costs(self) -> str:
        """Forecast future costs based on historical data"""
        try:
            start_date = datetime.utcnow().date()
            end_date = start_date + timedelta(days=30)

            response = ce_client.get_cost_forecast(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Metric="UNBLENDED_COST",
                Granularity="MONTHLY",
            )

            forecast = float(response["Total"]["Amount"])

            return json.dumps(
                {
                    "forecast_period": f"{start_date} to {end_date}",
                    "forecasted_cost": forecast,
                    "confidence": "Medium",  # Simplified confidence level
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class CodeBuildTool(BaseTool):
    """Tool for triggering and monitoring CodeBuild projects"""

    name = "codebuild_operations"
    description = """Manage CodeBuild projects for CI/CD operations.
    Input should be a JSON string with:
    - action: 'start', 'status', 'stop', 'list'
    - project_name: Name of the CodeBuild project
    - environment_variables: Optional environment variables for the build
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute CodeBuild operations"""
        try:
            params = json.loads(query)
            action = params.get("action", "list")
            project_name = params.get("project_name", "")

            if action == "start":
                return self._start_build(
                    project_name, params.get("environment_variables", {})
                )
            elif action == "status":
                return self._get_build_status(params.get("build_id", ""))
            elif action == "stop":
                return self._stop_build(params.get("build_id", ""))
            elif action == "list":
                return self._list_projects()
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _start_build(self, project_name: str, env_vars: Dict) -> str:
        """Start a new build"""
        try:
            kwargs = {"projectName": project_name}

            if env_vars:
                kwargs["environmentVariablesOverride"] = [
                    {"name": k, "value": v, "type": "PLAINTEXT"}
                    for k, v in env_vars.items()
                ]

            response = codebuild.start_build(**kwargs)

            build = response["build"]
            return json.dumps(
                {
                    "build_id": build["id"],
                    "status": build["buildStatus"],
                    "project": build["projectName"],
                    "start_time": (
                        build.get("startTime", "").isoformat()
                        if build.get("startTime")
                        else None
                    ),
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_build_status(self, build_id: str) -> str:
        """Get build status"""
        try:
            response = codebuild.batch_get_builds(ids=[build_id])

            if response["builds"]:
                build = response["builds"][0]
                return json.dumps(
                    {
                        "build_id": build["id"],
                        "status": build["buildStatus"],
                        "phase": build.get("currentPhase", "UNKNOWN"),
                        "start_time": (
                            build.get("startTime", "").isoformat()
                            if build.get("startTime")
                            else None
                        ),
                        "end_time": (
                            build.get("endTime", "").isoformat()
                            if build.get("endTime")
                            else None
                        ),
                    }
                )
            else:
                return json.dumps({"error": "Build not found"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _stop_build(self, build_id: str) -> str:
        """Stop a running build"""
        try:
            response = codebuild.stop_build(id=build_id)

            build = response["build"]
            return json.dumps(
                {
                    "build_id": build["id"],
                    "status": build["buildStatus"],
                    "message": "Build stopped successfully",
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _list_projects(self) -> str:
        """List all CodeBuild projects"""
        try:
            response = codebuild.list_projects()

            return json.dumps(
                {
                    "projects": response.get("projects", []),
                    "count": len(response.get("projects", [])),
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class LambdaInvokeTool(BaseTool):
    """Tool for invoking Lambda functions"""

    name = "lambda_invoke"
    description = """Invoke AWS Lambda functions.
    Input should be a JSON string with:
    - function_name: Name of the Lambda function
    - payload: JSON payload to send to the function
    - invocation_type: 'RequestResponse' (sync) or 'Event' (async)
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Invoke Lambda function"""
        try:
            params = json.loads(query)
            function_name = params.get("function_name")
            payload = params.get("payload", {})
            invocation_type = params.get("invocation_type", "RequestResponse")

            if not function_name:
                return json.dumps({"error": "function_name is required"})

            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=json.dumps(payload),
            )

            result = {
                "status_code": response["StatusCode"],
                "function_name": function_name,
                "invocation_type": invocation_type,
            }

            if invocation_type == "RequestResponse":
                response_payload = json.loads(response["Payload"].read())
                result["response"] = response_payload

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class CloudWatchLogsTool(BaseTool):
    """Tool for searching and analyzing CloudWatch Logs"""

    name = "cloudwatch_logs"
    description = """Search and analyze CloudWatch Logs.
    Input should be a JSON string with:
    - log_group: Name of the log group
    - query: CloudWatch Insights query or filter pattern
    - minutes_back: How many minutes of logs to search (default: 60)
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search CloudWatch logs"""
        try:
            params = json.loads(query)
            log_group = params.get("log_group")
            query_string = params.get(
                "query", "fields @timestamp, @message | sort @timestamp desc | limit 20"
            )
            minutes_back = params.get("minutes_back", 60)

            if not log_group:
                return json.dumps({"error": "log_group is required"})

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=minutes_back)

            # Start query
            start_response = logs.start_query(
                logGroupName=log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query_string,
            )

            query_id = start_response["queryId"]

            # Wait for results (with timeout)
            import time

            max_wait = 30  # seconds
            wait_time = 0

            while wait_time < max_wait:
                results_response = logs.get_query_results(queryId=query_id)

                if results_response["status"] == "Complete":
                    return json.dumps(
                        {
                            "status": "complete",
                            "results": results_response["results"][
                                :50
                            ],  # Limit to 50 results
                            "statistics": results_response.get("statistics", {}),
                            "records_matched": (
                                results_response["statistics"]["recordsMatched"]
                                if "statistics" in results_response
                                else 0
                            ),
                        }
                    )
                elif results_response["status"] == "Failed":
                    return json.dumps({"error": "Query failed"})

                time.sleep(2)
                wait_time += 2

            return json.dumps({"error": "Query timeout"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class ECSTaskTool(BaseTool):
    """Tool for managing ECS tasks and services"""

    name = "ecs_operations"
    description = """Manage ECS tasks and services.
    Input should be a JSON string with:
    - action: 'list_services', 'describe_service', 'update_service', 'run_task'
    - cluster: Name of the ECS cluster
    - service: Name of the service (for service operations)
    - task_definition: Task definition for running tasks
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute ECS operations"""
        try:
            params = json.loads(query)
            action = params.get("action", "list_services")
            cluster = params.get("cluster", "default")

            if action == "list_services":
                return self._list_services(cluster)
            elif action == "describe_service":
                return self._describe_service(cluster, params.get("service"))
            elif action == "update_service":
                return self._update_service(
                    cluster, params.get("service"), params.get("desired_count", 1)
                )
            elif action == "run_task":
                return self._run_task(cluster, params.get("task_definition"))
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _list_services(self, cluster: str) -> str:
        """List ECS services"""
        try:
            response = ecs.list_services(cluster=cluster)
            return json.dumps(
                {
                    "cluster": cluster,
                    "services": response.get("serviceArns", []),
                    "count": len(response.get("serviceArns", [])),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _describe_service(self, cluster: str, service: str) -> str:
        """Describe ECS service"""
        try:
            if not service:
                return json.dumps({"error": "service name is required"})

            response = ecs.describe_services(cluster=cluster, services=[service])

            if response["services"]:
                svc = response["services"][0]
                return json.dumps(
                    {
                        "service_name": svc["serviceName"],
                        "status": svc["status"],
                        "desired_count": svc["desiredCount"],
                        "running_count": svc["runningCount"],
                        "pending_count": svc["pendingCount"],
                    }
                )
            else:
                return json.dumps({"error": "Service not found"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _update_service(self, cluster: str, service: str, desired_count: int) -> str:
        """Update ECS service"""
        try:
            if not service:
                return json.dumps({"error": "service name is required"})

            response = ecs.update_service(
                cluster=cluster, service=service, desiredCount=desired_count
            )

            svc = response["service"]
            return json.dumps(
                {
                    "service_name": svc["serviceName"],
                    "status": "updated",
                    "new_desired_count": svc["desiredCount"],
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _run_task(self, cluster: str, task_definition: str) -> str:
        """Run ECS task"""
        try:
            if not task_definition:
                return json.dumps({"error": "task_definition is required"})

            response = ecs.run_task(cluster=cluster, taskDefinition=task_definition)

            if response["tasks"]:
                task = response["tasks"][0]
                return json.dumps(
                    {
                        "task_arn": task["taskArn"],
                        "status": task["lastStatus"],
                        "cluster": cluster,
                    }
                )
            else:
                return json.dumps(
                    {
                        "error": "Failed to run task",
                        "failures": response.get("failures", []),
                    }
                )

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


# Export all tools
__all__ = [
    "CloudWatchMetricsTool",
    "CostExplorerTool",
    "CodeBuildTool",
    "LambdaInvokeTool",
    "CloudWatchLogsTool",
    "ECSTaskTool",
]
