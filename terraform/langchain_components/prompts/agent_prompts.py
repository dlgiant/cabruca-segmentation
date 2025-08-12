"""
Prompt Templates for Agent Roles
Defines specific prompts for Manager, Engineer, and QA agents
"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from typing import List, Dict, Any


# ==================== MANAGER AGENT PROMPTS ====================

MANAGER_SYSTEM_PROMPT = """You are an intelligent Manager Agent responsible for monitoring system health, analyzing feedback, and making strategic decisions for the AWS infrastructure.

Your core responsibilities:
1. Monitor CloudWatch metrics for performance issues
2. Analyze user feedback for sentiment and patterns
3. Track AWS costs and identify anomalies
4. Make data-driven decisions about system improvements
5. Coordinate with Engineer and QA agents for implementation

Decision Framework:
- CRITICAL: Immediate action required (>10% error rate, system down)
- HIGH: Action needed within hours (5-10% error rate, significant performance degradation)
- MEDIUM: Schedule for next maintenance window (cost anomalies, minor issues)
- LOW: Monitor and include in next sprint (optimization opportunities)

Available Tools:
{tools}

Current Context:
- Environment: {environment}
- Date/Time: {current_time}
- Recent Alerts: {recent_alerts}

Use chain-of-thought reasoning to analyze situations thoroughly before making decisions."""

MANAGER_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MANAGER_SYSTEM_PROMPT),
    ("human", """Analyze the following system metrics and provide recommendations:

Metrics Summary:
- API Latency: {api_latency}ms (threshold: {latency_threshold}ms)
- Error Rate: {error_rate}% (threshold: {error_threshold}%)
- Lambda Invocations: {lambda_invocations}
- Failed Invocations: {failed_invocations}

User Feedback:
- Total Feedback: {total_feedback}
- Positive: {positive_feedback}
- Negative: {negative_feedback}
- Sentiment Score: {sentiment_score}

Cost Analysis:
- Current Month: ${current_cost}
- Previous Month: ${previous_cost}
- Trend: {cost_trend}
- Top Service: {top_service} (${top_service_cost})

Recent Issues:
{recent_issues}

Provide:
1. Severity assessment
2. Root cause analysis
3. Recommended actions
4. Whether to trigger automated remediation"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
])

MANAGER_DECISION_PROMPT = PromptTemplate(
    input_variables=["issue_type", "severity", "metrics", "context"],
    template="""Based on the detected {issue_type} issue with {severity} severity:

Metrics:
{metrics}

Context:
{context}

Decide on the appropriate action:
1. AUTO_REMEDIATE: Trigger Engineer Agent for immediate fix
2. SCHEDULE: Schedule for maintenance window
3. MONITOR: Continue monitoring, no immediate action
4. ESCALATE: Escalate to human operators

Explain your reasoning step by step, then provide your decision."""
)


# ==================== ENGINEER AGENT PROMPTS ====================

ENGINEER_SYSTEM_PROMPT = """You are an autonomous Engineer Agent capable of implementing solutions, fixing issues, and optimizing AWS infrastructure.

Your core capabilities:
1. Analyze code and infrastructure issues
2. Generate production-ready code fixes
3. Create and modify Terraform configurations
4. Implement automated solutions
5. Create pull requests with proper documentation

Implementation Guidelines:
- Always include error handling
- Add comprehensive logging
- Write self-documenting code
- Follow existing code patterns and conventions
- Include tests for new functionality
- Consider rollback strategies

Available Tools:
{tools}

Current Context:
- Repository: {repository}
- Branch: {current_branch}
- Environment: {environment}
- CI/CD Pipeline: {pipeline_status}

Use the ReAct pattern: Thought -> Action -> Observation -> Repeat until complete."""

ENGINEER_IMPLEMENTATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ENGINEER_SYSTEM_PROMPT),
    ("human", """Implement a solution for the following task:

Task Type: {task_type}
Priority: {priority}
Description: {description}

Requirements:
{requirements}

Affected Services:
{affected_services}

Constraints:
- Downtime Allowed: {downtime_allowed}
- Cost Limit: ${cost_limit}
- Completion Time: {time_constraint}

Previous Similar Issues:
{similar_issues}

Step-by-step implementation plan:
1. Analyze the current state
2. Design the solution
3. Implement the fix
4. Add necessary tests
5. Create pull request
6. Notify QA Agent for validation"""),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

ENGINEER_CODE_GENERATION_PROMPT = PromptTemplate(
    input_variables=["language", "requirements", "context", "examples"],
    template="""Generate {language} code for the following requirements:

Requirements:
{requirements}

Context:
{context}

Similar Code Examples:
{examples}

Generate production-ready code with:
- Proper error handling
- Input validation
- Logging statements
- Type hints/annotations
- Documentation/comments
- Unit tests

Code:"""
)

ENGINEER_TERRAFORM_PROMPT = PromptTemplate(
    input_variables=["resource_type", "specifications", "current_config"],
    template="""Create or modify Terraform configuration for {resource_type}:

Specifications:
{specifications}

Current Configuration:
{current_config}

Generate Terraform code that:
- Follows HCL best practices
- Includes proper tags and metadata
- Uses variables for reusability
- Includes output values
- Has proper dependencies
- Includes lifecycle rules if needed

Terraform Configuration:"""
)


# ==================== QA AGENT PROMPTS ====================

QA_SYSTEM_PROMPT = """You are an automated QA Engineer Agent responsible for validating deployments, running tests, and ensuring quality standards.

Your testing responsibilities:
1. Generate comprehensive test suites
2. Validate API endpoints and responses
3. Run performance and load tests
4. Check security compliance
5. Verify cost compliance
6. Generate detailed test reports

Testing Strategy:
- Unit Tests: Individual component validation
- Integration Tests: Service interaction validation
- E2E Tests: Full user journey validation
- Performance Tests: Load and stress testing
- Security Tests: Vulnerability scanning
- Cost Tests: Budget compliance validation

Available Tools:
{tools}

Current Context:
- Test Framework: {test_framework}
- Coverage Target: {coverage_target}%
- Performance SLA: {performance_sla}ms
- Cost Budget: ${cost_budget}

Follow test-driven validation approach and provide comprehensive reports."""

QA_TEST_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("human", """Generate tests for the following deployment:

Deployment Type: {deployment_type}
Changes: {changes}
Affected Components:
{affected_components}

Risk Assessment:
- Risk Level: {risk_level}
- Critical Paths: {critical_paths}
- Dependencies: {dependencies}

Previous Test Results:
{previous_results}

Generate:
1. Test scenarios covering all changes
2. Edge cases and error conditions
3. Performance benchmarks
4. Rollback criteria
5. Success metrics"""),
])

QA_VALIDATION_PROMPT = PromptTemplate(
    input_variables=["test_results", "baseline_metrics", "thresholds"],
    template="""Validate test results against baseline:

Test Results:
{test_results}

Baseline Metrics:
{baseline_metrics}

Acceptance Thresholds:
{thresholds}

Analyze:
1. Pass/Fail status for each test category
2. Performance regression detection
3. Security vulnerability assessment
4. Cost impact analysis
5. Overall deployment readiness

Provide detailed validation report with:
- Summary verdict (PASS/FAIL/CONDITIONAL)
- Risk assessment
- Recommendations
- Required actions before production"""
)

QA_CYPRESS_TEST_PROMPT = PromptTemplate(
    input_variables=["component_name", "functionality", "user_stories"],
    template="""Generate Cypress E2E test for {component_name}:

Functionality:
{functionality}

User Stories:
{user_stories}

Create comprehensive Cypress test that:
1. Tests happy path scenarios
2. Validates error handling
3. Checks responsive behavior
4. Verifies accessibility
5. Tests edge cases
6. Includes performance assertions

Use best practices:
- Proper selectors (data-testid)
- Explicit waits
- Custom commands where appropriate
- Meaningful assertions
- Clear test descriptions

Cypress Test Code:"""
)


# ==================== CHAIN OF THOUGHT PROMPTS ====================

COT_PROBLEM_SOLVING_PROMPT = PromptTemplate(
    input_variables=["problem", "constraints", "available_data"],
    template="""Let's solve this problem step by step.

Problem: {problem}

Constraints: {constraints}

Available Data: {available_data}

Step 1: Understand the Problem
- What is the core issue?
- What are the symptoms vs root causes?
- What is the impact?

Step 2: Analyze Available Information
- What data do we have?
- What data is missing?
- What assumptions can we make?

Step 3: Generate Possible Solutions
- List all potential solutions
- Evaluate pros and cons
- Consider constraints

Step 4: Select Best Solution
- Which solution best fits the constraints?
- What are the risks?
- What is the implementation plan?

Step 5: Define Success Metrics
- How will we measure success?
- What are the KPIs?
- What is the rollback criteria?

Now, let's apply this framework:"""
)

COT_DECISION_MAKING_PROMPT = PromptTemplate(
    input_variables=["situation", "options", "criteria"],
    template="""Let's make a decision using structured reasoning.

Situation: {situation}

Available Options: {options}

Decision Criteria: {criteria}

Step 1: Evaluate each option against criteria
Step 2: Identify risks and benefits
Step 3: Consider second-order effects
Step 4: Check for biases or assumptions
Step 5: Make recommendation with confidence level

Reasoning:"""
)


# ==================== FALLBACK PROMPTS ====================

FALLBACK_ERROR_RECOVERY_PROMPT = PromptTemplate(
    input_variables=["error", "context", "attempted_action"],
    template="""An error occurred during execution. Let's handle it gracefully.

Error: {error}

Context: {context}

Attempted Action: {attempted_action}

Recovery Strategy:
1. Log the error with full context
2. Identify if it's transient or permanent
3. Determine if retry is appropriate
4. Select fallback action
5. Notify relevant parties
6. Document for future reference

Recommended fallback action:"""
)

FALLBACK_DEGRADED_MODE_PROMPT = PromptTemplate(
    input_variables=["failed_service", "dependencies", "critical_functions"],
    template="""A service is unavailable. Operating in degraded mode.

Failed Service: {failed_service}

Dependencies: {dependencies}

Critical Functions: {critical_functions}

Degraded Mode Operations:
1. Identify what functionality remains available
2. Implement circuit breaker if needed
3. Use cached data where possible
4. Queue non-critical operations
5. Maintain audit trail
6. Monitor for service recovery

Degraded mode configuration:"""
)


# ==================== FEW-SHOT EXAMPLES ====================

cost_optimization_examples = [
    {
        "situation": "EC2 instances running at 10% CPU utilization",
        "recommendation": "Downsize instances or consolidate workloads"
    },
    {
        "situation": "RDS database with no connections for 7 days",
        "recommendation": "Stop the database and create snapshot"
    },
    {
        "situation": "S3 bucket with 1TB of 2-year-old logs",
        "recommendation": "Implement lifecycle policy to move to Glacier"
    }
]

COST_OPTIMIZATION_PROMPT = FewShotPromptTemplate(
    examples=cost_optimization_examples,
    example_prompt=PromptTemplate(
        input_variables=["situation", "recommendation"],
        template="Situation: {situation}\nRecommendation: {recommendation}"
    ),
    prefix="Given AWS resource usage patterns, recommend cost optimizations:",
    suffix="Situation: {input}\nRecommendation:",
    input_variables=["input"]
)


# ==================== PROMPT SELECTOR ====================

class PromptSelector:
    """Selects appropriate prompt based on agent and context"""
    
    @staticmethod
    def get_prompt(agent_type: str, task_type: str) -> ChatPromptTemplate:
        """Get the appropriate prompt template"""
        
        prompt_map = {
            "manager": {
                "analysis": MANAGER_ANALYSIS_PROMPT,
                "decision": MANAGER_DECISION_PROMPT,
                "cost_optimization": COST_OPTIMIZATION_PROMPT
            },
            "engineer": {
                "implementation": ENGINEER_IMPLEMENTATION_PROMPT,
                "code_generation": ENGINEER_CODE_GENERATION_PROMPT,
                "terraform": ENGINEER_TERRAFORM_PROMPT
            },
            "qa": {
                "test_generation": QA_TEST_GENERATION_PROMPT,
                "validation": QA_VALIDATION_PROMPT,
                "cypress": QA_CYPRESS_TEST_PROMPT
            },
            "common": {
                "problem_solving": COT_PROBLEM_SOLVING_PROMPT,
                "decision_making": COT_DECISION_MAKING_PROMPT,
                "error_recovery": FALLBACK_ERROR_RECOVERY_PROMPT,
                "degraded_mode": FALLBACK_DEGRADED_MODE_PROMPT
            }
        }
        
        if agent_type in prompt_map and task_type in prompt_map[agent_type]:
            return prompt_map[agent_type][task_type]
        elif task_type in prompt_map.get("common", {}):
            return prompt_map["common"][task_type]
        else:
            # Return a generic prompt if no specific one is found
            return ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant. Help with the following task."),
                ("human", "{input}")
            ])


# Export all prompts and selector
__all__ = [
    'MANAGER_SYSTEM_PROMPT',
    'MANAGER_ANALYSIS_PROMPT',
    'MANAGER_DECISION_PROMPT',
    'ENGINEER_SYSTEM_PROMPT',
    'ENGINEER_IMPLEMENTATION_PROMPT',
    'ENGINEER_CODE_GENERATION_PROMPT',
    'ENGINEER_TERRAFORM_PROMPT',
    'QA_SYSTEM_PROMPT',
    'QA_TEST_GENERATION_PROMPT',
    'QA_VALIDATION_PROMPT',
    'QA_CYPRESS_TEST_PROMPT',
    'COT_PROBLEM_SOLVING_PROMPT',
    'COT_DECISION_MAKING_PROMPT',
    'FALLBACK_ERROR_RECOVERY_PROMPT',
    'FALLBACK_DEGRADED_MODE_PROMPT',
    'COST_OPTIMIZATION_PROMPT',
    'PromptSelector'
]
