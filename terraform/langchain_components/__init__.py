"""
LangChain Components for Agent Intelligence
Main initialization module that brings together all components
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .llm.llm_config import (
    FallbackLLMChain,
    LLMProvider,
    ModelSelector,
    ModelType,
    get_llm_provider,
)
from .memory.dynamodb_memory import (
    ConversationAnalyzer,
    DynamoDBChatMemory,
    MemoryTableCreator,
)
from .prompts.agent_prompts import (
    ENGINEER_IMPLEMENTATION_PROMPT,
    MANAGER_ANALYSIS_PROMPT,
    QA_TEST_GENERATION_PROMPT,
    PromptSelector,
)
from .storage.s3_prompt_store import (
    PromptVersionManager,
    S3PromptStore,
    initialize_prompt_store,
)

# Import all components
from .tools.aws_tools import (
    CloudWatchLogsTool,
    CloudWatchMetricsTool,
    CodeBuildTool,
    CostExplorerTool,
    ECSTaskTool,
    LambdaInvokeTool,
)

logger = logging.getLogger(__name__)


class LangChainAgentFactory:
    """Factory class to create configured LangChain agents"""

    def __init__(
        self, s3_bucket: str, memory_table: str, environment: str = "production"
    ):
        """
        Initialize the agent factory

        Args:
            s3_bucket: S3 bucket for prompt storage
            memory_table: DynamoDB table for memory storage
            environment: Environment name (production, staging, dev)
        """
        self.s3_bucket = s3_bucket
        self.memory_table = memory_table
        self.environment = environment

        # Initialize components
        self.prompt_store = S3PromptStore(s3_bucket)
        self.prompt_selector = PromptSelector()

        # Ensure memory table exists
        MemoryTableCreator.create_memory_table(memory_table)

    def create_manager_agent(self, session_id: str) -> Dict[str, Any]:
        """
        Create a configured Manager Agent

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with agent components
        """
        # Get LLM provider
        llm_provider = get_llm_provider("manager")

        # Create memory
        memory = DynamoDBChatMemory(
            table_name=self.memory_table, session_id=session_id, agent_name="manager"
        )

        # Get tools
        tools = [CloudWatchMetricsTool(), CostExplorerTool(), CloudWatchLogsTool()]

        # Get prompt
        prompt = self.prompt_selector.get_prompt("manager", "analysis")

        # Create fallback chain
        fallback_chain = FallbackLLMChain("manager")

        return {
            "llm": llm_provider.get_llm_with_fallback(),
            "memory": memory,
            "tools": tools,
            "prompt": prompt,
            "fallback_chain": fallback_chain,
            "session_id": session_id,
        }

    def create_engineer_agent(self, session_id: str) -> Dict[str, Any]:
        """
        Create a configured Engineer Agent

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with agent components
        """
        # Get LLM provider
        llm_provider = get_llm_provider("engineer")

        # Create memory
        memory = DynamoDBChatMemory(
            table_name=self.memory_table, session_id=session_id, agent_name="engineer"
        )

        # Get tools
        tools = [CodeBuildTool(), LambdaInvokeTool(), ECSTaskTool()]

        # Get prompt
        prompt = self.prompt_selector.get_prompt("engineer", "implementation")

        # Create fallback chain
        fallback_chain = FallbackLLMChain("engineer")

        return {
            "llm": llm_provider.get_llm_with_fallback(),
            "memory": memory,
            "tools": tools,
            "prompt": prompt,
            "fallback_chain": fallback_chain,
            "session_id": session_id,
        }

    def create_qa_agent(self, session_id: str) -> Dict[str, Any]:
        """
        Create a configured QA Agent

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with agent components
        """
        # Get LLM provider
        llm_provider = get_llm_provider("qa")

        # Create memory
        memory = DynamoDBChatMemory(
            table_name=self.memory_table, session_id=session_id, agent_name="qa"
        )

        # Get tools
        tools = [CodeBuildTool(), CloudWatchMetricsTool()]

        # Get prompt
        prompt = self.prompt_selector.get_prompt("qa", "test_generation")

        # Create fallback chain
        fallback_chain = FallbackLLMChain("qa")

        return {
            "llm": llm_provider.get_llm_with_fallback(),
            "memory": memory,
            "tools": tools,
            "prompt": prompt,
            "fallback_chain": fallback_chain,
            "session_id": session_id,
        }


def initialize_langchain_environment(
    s3_bucket: str,
    memory_table: str,
    anthropic_api_key_secret: str,
    environment: str = "production",
) -> LangChainAgentFactory:
    """
    Initialize the complete LangChain environment

    Args:
        s3_bucket: S3 bucket for prompt storage
        memory_table: DynamoDB table for memory storage
        anthropic_api_key_secret: Secret name for Anthropic API key
        environment: Environment name

    Returns:
        Configured LangChainAgentFactory
    """
    # Set environment variables
    os.environ["ANTHROPIC_API_KEY_SECRET"] = anthropic_api_key_secret
    os.environ["ENVIRONMENT"] = environment

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize prompt store with defaults
    logger.info(f"Initializing prompt store in S3 bucket: {s3_bucket}")
    initialize_prompt_store(s3_bucket)

    # Create and return factory
    factory = LangChainAgentFactory(
        s3_bucket=s3_bucket, memory_table=memory_table, environment=environment
    )

    logger.info("LangChain environment initialized successfully")

    return factory


# Cost estimation utilities
def estimate_monthly_cost(
    daily_invocations: int,
    avg_tokens_per_invocation: int = 2000,
    model_type: ModelType = ModelType.HAIKU,
) -> float:
    """
    Estimate monthly cost for LLM usage

    Args:
        daily_invocations: Number of daily agent invocations
        avg_tokens_per_invocation: Average tokens per invocation
        model_type: Type of Claude model to use

    Returns:
        Estimated monthly cost in USD
    """
    # Calculate monthly tokens
    monthly_tokens = daily_invocations * 30 * avg_tokens_per_invocation

    # Split between input and output (assume 40% input, 60% output)
    input_tokens = monthly_tokens * 0.4
    output_tokens = monthly_tokens * 0.6

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * model_type.input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * model_type.output_cost_per_million

    return input_cost + output_cost


def get_cost_optimization_recommendations(current_usage: Dict[str, Any]) -> List[str]:
    """
    Get recommendations for optimizing LLM costs

    Args:
        current_usage: Dictionary with current usage metrics

    Returns:
        List of recommendations
    """
    recommendations = []

    avg_tokens = current_usage.get("avg_tokens_per_call", 0)
    daily_calls = current_usage.get("daily_calls", 0)
    model_type = current_usage.get("model_type", "HAIKU")

    # Token optimization
    if avg_tokens > 3000:
        recommendations.append(
            "Consider optimizing prompts to reduce token usage. "
            "Current average is high at {avg_tokens} tokens per call."
        )

    # Model selection
    if model_type != "HAIKU" and daily_calls > 1000:
        recommendations.append(
            "For high-volume usage, consider using Claude Haiku "
            "which costs ~$0.25/1M tokens vs $3-15 for other models."
        )

    # Caching
    if daily_calls > 100:
        recommendations.append(
            "Implement response caching for common queries "
            "to reduce redundant LLM calls."
        )

    # Batching
    recommendations.append(
        "Batch similar requests together to reduce API call overhead."
    )

    return recommendations


# Export main components
__all__ = [
    # Factory
    "LangChainAgentFactory",
    "initialize_langchain_environment",
    # Tools
    "CloudWatchMetricsTool",
    "CostExplorerTool",
    "CodeBuildTool",
    "LambdaInvokeTool",
    "CloudWatchLogsTool",
    "ECSTaskTool",
    # Memory
    "DynamoDBChatMemory",
    "ConversationAnalyzer",
    # LLM
    "ModelType",
    "LLMProvider",
    "FallbackLLMChain",
    "ModelSelector",
    # Prompts
    "PromptSelector",
    "S3PromptStore",
    # Utilities
    "estimate_monthly_cost",
    "get_cost_optimization_recommendations",
]
