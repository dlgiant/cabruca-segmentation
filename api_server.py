#!/usr/bin/env python
"""
Launch the Cabruca Segmentation API server.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description='Launch Cabruca Segmentation API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--model', type=str, default='outputs/checkpoint_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--plantation-data', type=str, default='plantation-data.json',
                       help='Path to plantation data JSON')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['MODEL_PATH'] = args.model
    os.environ['PLANTATION_DATA_PATH'] = args.plantation_data
    
    # Check if uvicorn is installed
    try:
        import uvicorn
    except ImportError:
        print("❌ Uvicorn not installed. Install with:")
        print("   pip install fastapi uvicorn[standard] python-multipart")
        sys.exit(1)
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"⚠️  Model not found at {args.model}")
        print("   Train a model first or specify correct path with --model")
    
    print(f"🌳 Launching Cabruca Segmentation API")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Model: {args.model}")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print(f"   Press Ctrl+C to stop")
    
    # Launch server
    uvicorn.run(
        "api.inference_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )

if __name__ == '__main__':
    main()