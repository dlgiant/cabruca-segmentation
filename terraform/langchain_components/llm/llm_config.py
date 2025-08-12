"""
LLM Provider Configuration
Configures Claude models with fallback strategies and cost optimization
"""

import os
import boto3
import json
from typing import Dict, Any, Optional, List
from langchain_anthropic import ChatAnthropic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from datetime import datetime
import logging
from functools import wraps
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# AWS clients
secretsmanager = boto3.client('secretsmanager')
cloudwatch = boto3.client('cloudwatch')


class ModelType(Enum):
    """Available Claude models with cost per million tokens"""
    HAIKU = ("claude-3-haiku-20240307", 0.25, 1.25)  # Input: $0.25/1M, Output: $1.25/1M
    SONNET = ("claude-3-sonnet-20240229", 3.00, 15.00)  # Input: $3/1M, Output: $15/1M
    OPUS = ("claude-3-opus-20240229", 15.00, 75.00)  # Input: $15/1M, Output: $75/1M
    
    def __init__(self, model_id: str, input_cost: float, output_cost: float):
        self.model_id = model_id
        self.input_cost_per_million = input_cost
        self.output_cost_per_million = output_cost


@dataclass
class LLMConfig:
    """Configuration for LLM instances"""
    model_type: ModelType
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 2
    
    @property
    def estimated_cost_per_call(self) -> float:
        """Estimate cost per API call (assuming average token usage)"""
        avg_input_tokens = 500
        avg_output_tokens = self.max_tokens / 2
        
        input_cost = (avg_input_tokens / 1_000_000) * self.model_type.input_cost_per_million
        output_cost = (avg_output_tokens / 1_000_000) * self.model_type.output_cost_per_million
        
        return input_cost + output_cost


class CostTrackingCallback(BaseCallbackHandler):
    """Callback handler to track LLM usage and costs"""
    
    def __init__(self, agent_name: str, model_type: ModelType):
        self.agent_name = agent_name
        self.model_type = model_type
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Calculate and track costs after each LLM call"""
        try:
            # Extract token usage from response
            if response.llm_output:
                usage = response.llm_output.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                
                # Calculate cost
                input_cost = (input_tokens / 1_000_000) * self.model_type.input_cost_per_million
                output_cost = (output_tokens / 1_000_000) * self.model_type.output_cost_per_million
                call_cost = input_cost + output_cost
                
                # Update tracking
                self.total_tokens += total_tokens
                self.total_cost += call_cost
                self.call_count += 1
                
                # Log metrics
                logger.info(f"LLM Call - Agent: {self.agent_name}, Model: {self.model_type.name}, "
                          f"Tokens: {total_tokens}, Cost: ${call_cost:.6f}")
                
                # Send metrics to CloudWatch
                self._send_metrics_to_cloudwatch(total_tokens, call_cost)
                
        except Exception as e:
            logger.error(f"Error tracking LLM costs: {str(e)}")
    
    def _send_metrics_to_cloudwatch(self, tokens: int, cost: float):
        """Send usage metrics to CloudWatch"""
        try:
            cloudwatch.put_metric_data(
                Namespace='LangChain/Agents',
                MetricData=[
                    {
                        'MetricName': 'TokensUsed',
                        'Dimensions': [
                            {'Name': 'AgentName', 'Value': self.agent_name},
                            {'Name': 'Model', 'Value': self.model_type.name}
                        ],
                        'Value': tokens,
                        'Unit': 'Count',
                        'Timestamp': datetime.utcnow()
                    },
                    {
                        'MetricName': 'EstimatedCost',
                        'Dimensions': [
                            {'Name': 'AgentName', 'Value': self.agent_name},
                            {'Name': 'Model', 'Value': self.model_type.name}
                        ],
                        'Value': cost,
                        'Unit': 'None',
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error sending metrics to CloudWatch: {str(e)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of usage and costs"""
        return {
            'agent_name': self.agent_name,
            'model': self.model_type.name,
            'total_calls': self.call_count,
            'total_tokens': self.total_tokens,
            'total_cost': round(self.total_cost, 6),
            'avg_tokens_per_call': self.total_tokens / self.call_count if self.call_count > 0 else 0,
            'avg_cost_per_call': self.total_cost / self.call_count if self.call_count > 0 else 0
        }


class LLMProvider:
    """Manages LLM instances with fallback and retry logic"""
    
    def __init__(self, agent_name: str, primary_model: ModelType = ModelType.HAIKU):
        self.agent_name = agent_name
        self.primary_model = primary_model
        self.api_key = self._get_api_key()
        self.llm_instances = {}
        self.cost_tracker = CostTrackingCallback(agent_name, primary_model)
        
    def _get_api_key(self) -> str:
        """Retrieve Anthropic API key from AWS Secrets Manager"""
        try:
            secret_name = os.environ.get('ANTHROPIC_API_KEY_SECRET', 'anthropic-api-key')
            response = secretsmanager.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except Exception as e:
            logger.error(f"Error retrieving API key: {str(e)}")
            # Try environment variable as fallback
            return os.environ.get('ANTHROPIC_API_KEY', '')
    
    def get_llm(self, 
                model_type: Optional[ModelType] = None,
                config: Optional[LLMConfig] = None) -> ChatAnthropic:
        """Get configured LLM instance"""
        
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        model_type = model_type or self.primary_model
        config = config or LLMConfig(model_type=model_type)
        
        # Cache LLM instances
        cache_key = f"{model_type.name}_{config.temperature}_{config.max_tokens}"
        
        if cache_key not in self.llm_instances:
            self.llm_instances[cache_key] = ChatAnthropic(
                model=model_type.model_id,
                anthropic_api_key=self.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                callbacks=[self.cost_tracker]
            )
        
        return self.llm_instances[cache_key]
    
    def get_llm_with_fallback(self) -> ChatAnthropic:
        """Get LLM with automatic fallback to cheaper models"""
        fallback_chain = [ModelType.HAIKU, ModelType.SONNET, ModelType.OPUS]
        
        for model in fallback_chain:
            try:
                return self.get_llm(model_type=model)
            except Exception as e:
                logger.warning(f"Failed to initialize {model.name}: {str(e)}")
                continue
        
        raise RuntimeError("All LLM models failed to initialize")


def retry_with_exponential_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator for retrying LLM calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


class FallbackLLMChain:
    """Chain that automatically falls back to simpler/cheaper models on failure"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.provider = LLMProvider(agent_name)
        self.model_hierarchy = [
            (ModelType.HAIKU, LLMConfig(ModelType.HAIKU, temperature=0.3, max_tokens=1000)),
            (ModelType.SONNET, LLMConfig(ModelType.SONNET, temperature=0.3, max_tokens=2000)),
            (ModelType.OPUS, LLMConfig(ModelType.OPUS, temperature=0.2, max_tokens=4000))
        ]
    
    @retry_with_exponential_backoff(max_retries=3)
    def invoke(self, prompt: str, require_complex_reasoning: bool = False) -> str:
        """
        Invoke LLM with automatic fallback
        
        Args:
            prompt: The prompt to send to the LLM
            require_complex_reasoning: If True, start with more capable model
        """
        
        # Determine starting model based on complexity
        start_index = 0 if not require_complex_reasoning else 1
        
        for i in range(start_index, len(self.model_hierarchy)):
            model_type, config = self.model_hierarchy[i]
            
            try:
                llm = self.provider.get_llm(model_type=model_type, config=config)
                response = llm.invoke(prompt)
                
                # Log successful call
                logger.info(f"Successfully used {model_type.name} for {self.agent_name}")
                
                return response.content
                
            except Exception as e:
                logger.warning(f"Failed with {model_type.name}: {str(e)}")
                
                # If this was the last model, raise the exception
                if i == len(self.model_hierarchy) - 1:
                    raise
                
                # Otherwise, continue to next model
                logger.info(f"Falling back to next model...")
                continue
        
        raise RuntimeError("All models in hierarchy failed")


class ModelSelector:
    """Intelligent model selection based on task requirements"""
    
    @staticmethod
    def select_model(task_type: str, estimated_complexity: str = "low") -> ModelType:
        """
        Select appropriate model based on task type and complexity
        
        Args:
            task_type: Type of task (analysis, generation, validation, etc.)
            estimated_complexity: Estimated complexity (low, medium, high)
        """
        
        # Task to model mapping for cost optimization
        task_model_map = {
            # Simple tasks - use Haiku
            "metric_analysis": ModelType.HAIKU,
            "cost_tracking": ModelType.HAIKU,
            "log_analysis": ModelType.HAIKU,
            "status_check": ModelType.HAIKU,
            "simple_validation": ModelType.HAIKU,
            
            # Medium complexity - consider Sonnet
            "code_generation": ModelType.SONNET if estimated_complexity != "low" else ModelType.HAIKU,
            "test_generation": ModelType.SONNET if estimated_complexity != "low" else ModelType.HAIKU,
            "error_diagnosis": ModelType.SONNET,
            "optimization": ModelType.SONNET,
            
            # High complexity - may need Opus
            "architecture_design": ModelType.OPUS if estimated_complexity == "high" else ModelType.SONNET,
            "complex_debugging": ModelType.OPUS if estimated_complexity == "high" else ModelType.SONNET,
            "security_analysis": ModelType.OPUS if estimated_complexity == "high" else ModelType.SONNET,
        }
        
        # Default to Haiku for unknown tasks (cost-efficient)
        return task_model_map.get(task_type, ModelType.HAIKU)


class PromptOptimizer:
    """Optimize prompts to reduce token usage and costs"""
    
    @staticmethod
    def optimize_prompt(prompt: str, max_context_length: int = 2000) -> str:
        """
        Optimize prompt to reduce tokens while maintaining effectiveness
        
        Args:
            prompt: Original prompt
            max_context_length: Maximum allowed context length
        """
        
        # Remove excessive whitespace
        prompt = ' '.join(prompt.split())
        
        # Truncate if too long
        if len(prompt) > max_context_length:
            # Smart truncation - try to keep the most important parts
            parts = prompt.split('\n\n')
            
            # Keep first and last parts (usually context and question)
            if len(parts) > 2:
                truncated = f"{parts[0]}\n\n...[truncated]...\n\n{parts[-1]}"
                if len(truncated) <= max_context_length:
                    prompt = truncated
                else:
                    prompt = prompt[:max_context_length-10] + "..."
        
        return prompt
    
    @staticmethod
    def create_concise_prompt(template: str, variables: Dict[str, Any]) -> str:
        """
        Create concise prompt from template and variables
        
        Args:
            template: Prompt template
            variables: Variables to fill in template
        """
        
        # Format template with variables
        prompt = template.format(**variables)
        
        # Remove any None or empty values
        lines = prompt.split('\n')
        filtered_lines = [line for line in lines if not any(
            marker in line for marker in ['None', 'N/A', '[]', '{}']
        )]
        
        return '\n'.join(filtered_lines)


# Singleton instances for each agent
_manager_provider = None
_engineer_provider = None
_qa_provider = None


def get_llm_provider(agent_type: str) -> LLMProvider:
    """Get or create LLM provider for specific agent type"""
    global _manager_provider, _engineer_provider, _qa_provider
    
    if agent_type == "manager":
        if _manager_provider is None:
            _manager_provider = LLMProvider("manager", ModelType.HAIKU)
        return _manager_provider
    elif agent_type == "engineer":
        if _engineer_provider is None:
            _engineer_provider = LLMProvider("engineer", ModelType.HAIKU)
        return _engineer_provider
    elif agent_type == "qa":
        if _qa_provider is None:
            _qa_provider = LLMProvider("qa", ModelType.HAIKU)
        return _qa_provider
    else:
        return LLMProvider(agent_type, ModelType.HAIKU)


# Export classes and functions
__all__ = [
    'ModelType',
    'LLMConfig',
    'CostTrackingCallback',
    'LLMProvider',
    'FallbackLLMChain',
    'ModelSelector',
    'PromptOptimizer',
    'get_llm_provider',
    'retry_with_exponential_backoff'
]
