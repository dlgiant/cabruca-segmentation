import json
import os
from datetime import datetime


def lambda_handler(event, context):
    """Engineer Agent Lambda Handler"""

    agent_type = os.environ.get("AGENT_TYPE", "ENGINEER")
    environment = os.environ.get("ENVIRONMENT", "mvp")

    action = event.get("action", "default")
    message = event.get("message", "")

    # Process test action
    if action == "test":
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "success",
                    "agent": agent_type,
                    "message": f"Received: {message}",
                    "response": "Engineer agent is operational",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
        }

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": f"{agent_type} agent processed request",
                "action": action,
                "timestamp": datetime.now().isoformat(),
            }
        ),
    }
