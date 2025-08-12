"""
S3 Prompt Storage System
Manages prompt templates in S3 for easy updates without code changes
"""

import json
import boto3
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from functools import lru_cache
import yaml

logger = logging.getLogger(__name__)

# AWS clients
s3_client = boto3.client('s3')


class S3PromptStore:
    """Manages prompt templates stored in S3"""
    
    def __init__(self, bucket_name: str, prefix: str = "prompts/"):
        """
        Initialize S3 prompt store
        
        Args:
            bucket_name: S3 bucket name
            prefix: Prefix for prompt objects in S3
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cache_refresh = datetime.utcnow()
    
    def upload_prompt(self, 
                     agent_type: str,
                     prompt_name: str,
                     prompt_content: str,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a prompt template to S3
        
        Args:
            agent_type: Type of agent (manager, engineer, qa)
            prompt_name: Name of the prompt
            prompt_content: The prompt template content
            metadata: Optional metadata about the prompt
        """
        try:
            key = f"{self.prefix}{agent_type}/{prompt_name}.txt"
            
            # Prepare metadata
            s3_metadata = {
                'agent_type': agent_type,
                'prompt_name': prompt_name,
                'upload_time': datetime.utcnow().isoformat(),
                'content_hash': hashlib.md5(prompt_content.encode()).hexdigest()
            }
            
            if metadata:
                s3_metadata.update(metadata)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=prompt_content.encode('utf-8'),
                ContentType='text/plain',
                Metadata={k: str(v) for k, v in s3_metadata.items()},
                Tags=f"agent={agent_type}&prompt={prompt_name}"
            )
            
            # Invalidate cache for this prompt
            cache_key = f"{agent_type}_{prompt_name}"
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            logger.info(f"Uploaded prompt {prompt_name} for {agent_type} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading prompt to S3: {str(e)}")
            return False
    
    @lru_cache(maxsize=100)
    def get_prompt(self, agent_type: str, prompt_name: str) -> Optional[str]:
        """
        Retrieve a prompt template from S3
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
        
        Returns:
            Prompt content or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{agent_type}_{prompt_name}"
            if self._is_cache_valid() and cache_key in self.cache:
                logger.debug(f"Returning cached prompt {prompt_name} for {agent_type}")
                return self.cache[cache_key]
            
            # Fetch from S3
            key = f"{self.prefix}{agent_type}/{prompt_name}.txt"
            
            response = s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            prompt_content = response['Body'].read().decode('utf-8')
            
            # Update cache
            self.cache[cache_key] = prompt_content
            
            logger.info(f"Retrieved prompt {prompt_name} for {agent_type} from S3")
            return prompt_content
            
        except s3_client.exceptions.NoSuchKey:
            logger.warning(f"Prompt {prompt_name} for {agent_type} not found in S3")
            return None
        except Exception as e:
            logger.error(f"Error retrieving prompt from S3: {str(e)}")
            return None
    
    def list_prompts(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available prompts
        
        Args:
            agent_type: Optional filter by agent type
        
        Returns:
            List of prompt metadata
        """
        try:
            prefix = f"{self.prefix}{agent_type}/" if agent_type else self.prefix
            
            response = s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            prompts = []
            for obj in response.get('Contents', []):
                # Extract prompt info from key
                key_parts = obj['Key'].replace(self.prefix, '').split('/')
                if len(key_parts) >= 2:
                    agent = key_parts[0]
                    prompt_name = key_parts[1].replace('.txt', '')
                    
                    prompts.append({
                        'agent_type': agent,
                        'prompt_name': prompt_name,
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error listing prompts from S3: {str(e)}")
            return []
    
    def update_prompt(self,
                     agent_type: str,
                     prompt_name: str,
                     prompt_content: str,
                     create_version: bool = True) -> bool:
        """
        Update an existing prompt with optional versioning
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
            prompt_content: New prompt content
            create_version: Whether to create a versioned backup
        """
        try:
            if create_version:
                # Create a versioned backup
                current_prompt = self.get_prompt(agent_type, prompt_name)
                if current_prompt:
                    version_key = f"{self.prefix}{agent_type}/versions/{prompt_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
                    s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=version_key,
                        Body=current_prompt.encode('utf-8'),
                        ContentType='text/plain'
                    )
                    logger.info(f"Created version backup at {version_key}")
            
            # Update the prompt
            return self.upload_prompt(agent_type, prompt_name, prompt_content)
            
        except Exception as e:
            logger.error(f"Error updating prompt: {str(e)}")
            return False
    
    def delete_prompt(self, agent_type: str, prompt_name: str) -> bool:
        """
        Delete a prompt from S3
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
        """
        try:
            key = f"{self.prefix}{agent_type}/{prompt_name}.txt"
            
            s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            # Clear from cache
            cache_key = f"{agent_type}_{prompt_name}"
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            logger.info(f"Deleted prompt {prompt_name} for {agent_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting prompt: {str(e)}")
            return False
    
    def bulk_upload_prompts(self, prompts_config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Bulk upload prompts from a configuration dictionary
        
        Args:
            prompts_config: Dictionary with agent types and their prompts
        
        Returns:
            Dictionary with upload status for each prompt
        """
        results = {}
        
        for agent_type, prompts in prompts_config.items():
            for prompt_name, prompt_content in prompts.items():
                key = f"{agent_type}/{prompt_name}"
                results[key] = self.upload_prompt(
                    agent_type=agent_type,
                    prompt_name=prompt_name,
                    prompt_content=prompt_content
                )
        
        return results
    
    def export_prompts(self, output_format: str = "json") -> str:
        """
        Export all prompts to a specific format
        
        Args:
            output_format: Format to export to (json or yaml)
        
        Returns:
            Exported prompts as string
        """
        try:
            all_prompts = {}
            
            # Get all agent types
            agent_types = set()
            prompts_list = self.list_prompts()
            for prompt_info in prompts_list:
                agent_types.add(prompt_info['agent_type'])
            
            # Fetch all prompts
            for agent_type in agent_types:
                all_prompts[agent_type] = {}
                agent_prompts = [p for p in prompts_list if p['agent_type'] == agent_type]
                
                for prompt_info in agent_prompts:
                    prompt_content = self.get_prompt(agent_type, prompt_info['prompt_name'])
                    if prompt_content:
                        all_prompts[agent_type][prompt_info['prompt_name']] = prompt_content
            
            # Export in requested format
            if output_format == "yaml":
                return yaml.dump(all_prompts, default_flow_style=False)
            else:
                return json.dumps(all_prompts, indent=2)
                
        except Exception as e:
            logger.error(f"Error exporting prompts: {str(e)}")
            return "{}"
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        time_since_refresh = (datetime.utcnow() - self.last_cache_refresh).total_seconds()
        return time_since_refresh < self.cache_ttl
    
    def refresh_cache(self):
        """Force refresh of the cache"""
        self.cache.clear()
        self.last_cache_refresh = datetime.utcnow()
        logger.info("Prompt cache refreshed")


class PromptVersionManager:
    """Manages versioning of prompts in S3"""
    
    def __init__(self, bucket_name: str, prefix: str = "prompts/"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = s3_client
    
    def get_prompt_versions(self, agent_type: str, prompt_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a specific prompt
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
        
        Returns:
            List of version metadata
        """
        try:
            prefix = f"{self.prefix}{agent_type}/versions/{prompt_name}_"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            versions = []
            for obj in response.get('Contents', []):
                # Extract timestamp from key
                key_parts = obj['Key'].split('_')
                if len(key_parts) >= 2:
                    timestamp = '_'.join(key_parts[-2:]).replace('.txt', '')
                    
                    versions.append({
                        'version': timestamp,
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
            
            # Sort by timestamp descending
            versions.sort(key=lambda x: x['last_modified'], reverse=True)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting prompt versions: {str(e)}")
            return []
    
    def restore_version(self, agent_type: str, prompt_name: str, version: str) -> bool:
        """
        Restore a specific version of a prompt
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
            version: Version timestamp to restore
        """
        try:
            # Get the versioned content
            version_key = f"{self.prefix}{agent_type}/versions/{prompt_name}_{version}.txt"
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=version_key
            )
            
            content = response['Body'].read().decode('utf-8')
            
            # Restore as current version
            current_key = f"{self.prefix}{agent_type}/{prompt_name}.txt"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=current_key,
                Body=content.encode('utf-8'),
                ContentType='text/plain',
                Metadata={
                    'restored_from': version,
                    'restored_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Restored prompt {prompt_name} for {agent_type} from version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring prompt version: {str(e)}")
            return False
    
    def compare_versions(self, agent_type: str, prompt_name: str, 
                        version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a prompt
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
            version1: First version timestamp
            version2: Second version timestamp
        
        Returns:
            Comparison results
        """
        try:
            # Get both versions
            key1 = f"{self.prefix}{agent_type}/versions/{prompt_name}_{version1}.txt"
            key2 = f"{self.prefix}{agent_type}/versions/{prompt_name}_{version2}.txt"
            
            response1 = self.s3_client.get_object(Bucket=self.bucket_name, Key=key1)
            response2 = self.s3_client.get_object(Bucket=self.bucket_name, Key=key2)
            
            content1 = response1['Body'].read().decode('utf-8')
            content2 = response2['Body'].read().decode('utf-8')
            
            # Simple comparison
            import difflib
            diff = list(difflib.unified_diff(
                content1.splitlines(),
                content2.splitlines(),
                fromfile=f"{prompt_name}_{version1}",
                tofile=f"{prompt_name}_{version2}",
                lineterm=''
            ))
            
            return {
                'version1': version1,
                'version2': version2,
                'identical': content1 == content2,
                'diff': '\n'.join(diff) if diff else None,
                'size_change': len(content2) - len(content1)
            }
            
        except Exception as e:
            logger.error(f"Error comparing prompt versions: {str(e)}")
            return {'error': str(e)}


# Default prompts to initialize
DEFAULT_PROMPTS = {
    "manager": {
        "system": "You are a Manager Agent responsible for system monitoring and decision making.",
        "analysis": "Analyze the following metrics and provide recommendations:\n{metrics}",
        "decision": "Make a decision based on:\n{context}\nOptions: {options}"
    },
    "engineer": {
        "system": "You are an Engineer Agent capable of implementing solutions.",
        "implementation": "Implement the following solution:\n{requirements}",
        "code_gen": "Generate {language} code for:\n{specification}"
    },
    "qa": {
        "system": "You are a QA Agent responsible for testing and validation.",
        "test_gen": "Generate tests for:\n{component}",
        "validation": "Validate the following results:\n{test_results}"
    }
}


def initialize_prompt_store(bucket_name: str) -> S3PromptStore:
    """
    Initialize prompt store with default prompts
    
    Args:
        bucket_name: S3 bucket name
    
    Returns:
        Initialized S3PromptStore
    """
    store = S3PromptStore(bucket_name)
    
    # Upload default prompts
    results = store.bulk_upload_prompts(DEFAULT_PROMPTS)
    
    successful = sum(1 for v in results.values() if v)
    logger.info(f"Initialized prompt store with {successful}/{len(results)} default prompts")
    
    return store


# Export classes and functions
__all__ = [
    'S3PromptStore',
    'PromptVersionManager',
    'initialize_prompt_store',
    'DEFAULT_PROMPTS'
]
