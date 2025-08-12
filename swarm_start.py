import os
import agentops
from dotenv import load_dotenv

load_dotenv()

# Ensure Anthropic API key is set from environment
if "ANTHROPIC_API_KEY" not in os.environ:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it in your .env file.")

# Get AgentOps API key from environment
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY environment variable is not set. Please set it in your .env file.")
agentops.init(
    api_key=AGENTOPS_API_KEY,
    default_tags=['anthropic']
)