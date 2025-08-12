"""
DynamoDB Memory Store for LangChain Agents
Implements persistent conversation history using DynamoDB
"""

import json
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.messages import get_buffer_string
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

# DynamoDB client
dynamodb = boto3.resource('dynamodb')


class DynamoDBChatMemory(BaseChatMemory):
    """
    Custom chat memory that persists conversation history to DynamoDB
    """
    
    def __init__(
        self,
        table_name: str,
        session_id: str,
        agent_name: str,
        ttl_days: int = 30,
        max_messages: int = 100
    ):
        """
        Initialize DynamoDB chat memory
        
        Args:
            table_name: Name of the DynamoDB table
            session_id: Unique session identifier
            agent_name: Name of the agent (manager, engineer, qa)
            ttl_days: Number of days to retain conversation history
            max_messages: Maximum number of messages to keep in memory
        """
        super().__init__()
        self.table_name = table_name
        self.session_id = session_id
        self.agent_name = agent_name
        self.ttl_days = ttl_days
        self.max_messages = max_messages
        self.table = dynamodb.Table(table_name)
        
        # Load existing conversation history
        self._load_history()
    
    def _load_history(self):
        """Load conversation history from DynamoDB"""
        try:
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={
                    ':sid': self.session_id
                },
                ScanIndexForward=True,  # Sort by timestamp ascending
                Limit=self.max_messages
            )
            
            messages = []
            for item in response.get('Items', []):
                message_type = item.get('message_type', 'human')
                content = item.get('content', '')
                
                if message_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif message_type == 'ai':
                    messages.append(AIMessage(content=content))
                elif message_type == 'system':
                    messages.append(SystemMessage(content=content))
            
            self.chat_memory.messages = messages
            logger.info(f"Loaded {len(messages)} messages from DynamoDB for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error loading history from DynamoDB: {str(e)}")
            self.chat_memory.messages = []
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to both memory and DynamoDB"""
        # Save to in-memory storage
        super().save_context(inputs, outputs)
        
        # Save to DynamoDB
        timestamp = datetime.utcnow()
        ttl = int((timestamp + timedelta(days=self.ttl_days)).timestamp())
        
        try:
            # Save human message
            if 'input' in inputs:
                self._save_message(
                    message_type='human',
                    content=inputs['input'],
                    timestamp=timestamp,
                    ttl=ttl
                )
            
            # Save AI response
            if 'output' in outputs:
                self._save_message(
                    message_type='ai',
                    content=outputs['output'],
                    timestamp=timestamp,
                    ttl=ttl
                )
            
            # Trim messages if exceeding max_messages
            self._trim_messages()
            
        except Exception as e:
            logger.error(f"Error saving context to DynamoDB: {str(e)}")
    
    def _save_message(
        self,
        message_type: str,
        content: str,
        timestamp: datetime,
        ttl: int
    ):
        """Save a single message to DynamoDB"""
        try:
            item = {
                'session_id': self.session_id,
                'timestamp': timestamp.isoformat(),
                'agent_name': self.agent_name,
                'message_type': message_type,
                'content': content,
                'ttl': ttl
            }
            
            self.table.put_item(Item=item)
            
        except Exception as e:
            logger.error(f"Error saving message to DynamoDB: {str(e)}")
    
    def _trim_messages(self):
        """Trim old messages if exceeding max_messages limit"""
        try:
            # Count current messages
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={
                    ':sid': self.session_id
                },
                Select='COUNT'
            )
            
            count = response.get('Count', 0)
            
            if count > self.max_messages:
                # Get oldest messages to delete
                messages_to_delete = count - self.max_messages
                
                response = self.table.query(
                    KeyConditionExpression='session_id = :sid',
                    ExpressionAttributeValues={
                        ':sid': self.session_id
                    },
                    ScanIndexForward=True,  # Sort by timestamp ascending
                    Limit=messages_to_delete,
                    ProjectionExpression='session_id, #ts',
                    ExpressionAttributeNames={
                        '#ts': 'timestamp'
                    }
                )
                
                # Delete old messages
                with self.table.batch_writer() as batch:
                    for item in response.get('Items', []):
                        batch.delete_item(
                            Key={
                                'session_id': item['session_id'],
                                'timestamp': item['timestamp']
                            }
                        )
                
                logger.info(f"Trimmed {messages_to_delete} old messages for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error trimming messages: {str(e)}")
    
    def clear(self) -> None:
        """Clear all messages for this session"""
        super().clear()
        
        try:
            # Delete all messages for this session from DynamoDB
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={
                    ':sid': self.session_id
                },
                ProjectionExpression='session_id, #ts',
                ExpressionAttributeNames={
                    '#ts': 'timestamp'
                }
            )
            
            with self.table.batch_writer() as batch:
                for item in response.get('Items', []):
                    batch.delete_item(
                        Key={
                            'session_id': item['session_id'],
                            'timestamp': item['timestamp']
                        }
                    )
            
            logger.info(f"Cleared all messages for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing messages: {str(e)}")


class ConversationAnalyzer:
    """
    Analyzes conversation history to extract insights and patterns
    """
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.table = dynamodb.Table(table_name)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a conversation session"""
        try:
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={
                    ':sid': session_id
                }
            )
            
            messages = response.get('Items', [])
            
            if not messages:
                return {'error': 'No messages found for session'}
            
            # Analyze messages
            human_messages = [m for m in messages if m.get('message_type') == 'human']
            ai_messages = [m for m in messages if m.get('message_type') == 'ai']
            
            # Calculate conversation metrics
            first_message = min(messages, key=lambda x: x.get('timestamp', ''))
            last_message = max(messages, key=lambda x: x.get('timestamp', ''))
            
            duration = self._calculate_duration(
                first_message.get('timestamp'),
                last_message.get('timestamp')
            )
            
            return {
                'session_id': session_id,
                'total_messages': len(messages),
                'human_messages': len(human_messages),
                'ai_messages': len(ai_messages),
                'duration_minutes': duration,
                'agents_involved': list(set(m.get('agent_name', 'unknown') for m in messages)),
                'first_message': first_message.get('timestamp'),
                'last_message': last_message.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            return {'error': str(e)}
    
    def get_agent_activity(self, agent_name: str, days_back: int = 7) -> Dict[str, Any]:
        """Get activity summary for a specific agent"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Query using GSI on agent_name (you'd need to create this index)
            response = self.table.scan(
                FilterExpression='agent_name = :agent AND #ts > :cutoff',
                ExpressionAttributeValues={
                    ':agent': agent_name,
                    ':cutoff': cutoff_date.isoformat()
                },
                ExpressionAttributeNames={
                    '#ts': 'timestamp'
                }
            )
            
            messages = response.get('Items', [])
            
            # Group by session
            sessions = {}
            for msg in messages:
                session_id = msg.get('session_id')
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(msg)
            
            return {
                'agent_name': agent_name,
                'period_days': days_back,
                'total_messages': len(messages),
                'unique_sessions': len(sessions),
                'average_messages_per_session': len(messages) / len(sessions) if sessions else 0,
                'message_types': self._count_message_types(messages)
            }
            
        except Exception as e:
            logger.error(f"Error getting agent activity: {str(e)}")
            return {'error': str(e)}
    
    def extract_key_decisions(self, session_id: str) -> List[Dict[str, Any]]:
        """Extract key decisions made during a conversation"""
        try:
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={
                    ':sid': session_id
                },
                FilterExpression='message_type = :ai',
                ExpressionAttributeValues={
                    ':ai': 'ai'
                }
            )
            
            ai_messages = response.get('Items', [])
            decisions = []
            
            # Look for decision patterns in AI responses
            decision_keywords = [
                'decided', 'recommendation', 'suggest', 'should', 
                'will', 'action', 'implement', 'deploy', 'rollback'
            ]
            
            for msg in ai_messages:
                content = msg.get('content', '').lower()
                
                if any(keyword in content for keyword in decision_keywords):
                    decisions.append({
                        'timestamp': msg.get('timestamp'),
                        'agent': msg.get('agent_name'),
                        'decision': self._extract_decision_text(content),
                        'full_content': msg.get('content')
                    })
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error extracting key decisions: {str(e)}")
            return []
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration in minutes between two timestamps"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = (end - start).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0.0
    
    def _count_message_types(self, messages: List[Dict]) -> Dict[str, int]:
        """Count messages by type"""
        counts = {'human': 0, 'ai': 0, 'system': 0}
        for msg in messages:
            msg_type = msg.get('message_type', 'unknown')
            if msg_type in counts:
                counts[msg_type] += 1
        return counts
    
    def _extract_decision_text(self, content: str) -> str:
        """Extract the key decision text from a message"""
        # Simple extraction - in production, use NLP
        sentences = content.split('.')
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in ['recommend', 'suggest', 'should', 'will']):
                return sentence.strip()
        return content[:200] + '...' if len(content) > 200 else content


class MemoryTableCreator:
    """
    Creates and manages DynamoDB tables for memory storage
    """
    
    @staticmethod
    def create_memory_table(table_name: str) -> bool:
        """Create DynamoDB table for memory storage"""
        try:
            dynamodb_client = boto3.client('dynamodb')
            
            response = dynamodb_client.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'session_id',
                        'KeyType': 'HASH'  # Partition key
                    },
                    {
                        'AttributeName': 'timestamp',
                        'KeyType': 'RANGE'  # Sort key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'session_id',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'agent_name',
                        'AttributeType': 'S'
                    }
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'AgentNameIndex',
                        'Keys': [
                            {
                                'AttributeName': 'agent_name',
                                'KeyType': 'HASH'
                            },
                            {
                                'AttributeName': 'timestamp',
                                'KeyType': 'RANGE'
                            }
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        },
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    }
                ],
                BillingMode='PAY_PER_REQUEST',  # On-demand billing
                StreamSpecification={
                    'StreamEnabled': True,
                    'StreamViewType': 'NEW_AND_OLD_IMAGES'
                },
                TimeToLiveSpecification={
                    'Enabled': True,
                    'AttributeName': 'ttl'
                },
                Tags=[
                    {
                        'Key': 'Environment',
                        'Value': 'production'
                    },
                    {
                        'Key': 'Purpose',
                        'Value': 'agent-memory'
                    }
                ]
            )
            
            logger.info(f"Created DynamoDB table: {table_name}")
            return True
            
        except dynamodb_client.exceptions.ResourceInUseException:
            logger.info(f"Table {table_name} already exists")
            return True
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            return False


# Export classes
__all__ = [
    'DynamoDBChatMemory',
    'ConversationAnalyzer',
    'MemoryTableCreator'
]
