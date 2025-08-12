#!/usr/bin/env python
"""
Simplified API server for Cabruca Segmentation testing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import agentops
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image
import io

# Initialize AgentOps
agentops.init(tags=["cabruca", "api-server", "test"])

# Create FastAPI app
app = FastAPI(
    title="Cabruca Segmentation API",
    description="API for ML-based segmentation of Cabruca agroforestry systems",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str

class SegmentationResult(BaseModel):
    task_id: str
    status: str
    trees_detected: int
    crown_coverage: float
    species_distribution: Dict[str, int]
    confidence: float
    processing_time: float

# Global variables
MODEL_LOADED = False

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global MODEL_LOADED
    model_path = os.environ.get('MODEL_PATH', 'outputs/checkpoint_best.pth')
    
    if os.path.exists(model_path):
        print(f"✅ Model found at {model_path}")
        MODEL_LOADED = True
    else:
        print(f"⚠️ Model not found at {model_path}, using mock mode")
        MODEL_LOADED = False

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=MODEL_LOADED,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=MODEL_LOADED,
        version="1.0.0"
    )

@app.post("/segment", response_model=SegmentationResult)
async def segment_image(file: UploadFile = File(...)):
    """
    Segment an uploaded image.
    
    This is a mock implementation for testing.
    Replace with actual ML inference in production.
    """
    # Track with AgentOps
    agentops.record(agentops.ActionEvent(
        action_type="segmentation_request",
        params={
            "filename": file.filename,
            "content_type": file.content_type
        }
    ))
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Mock segmentation results
    result = SegmentationResult(
        task_id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        status="completed",
        trees_detected=np.random.randint(10, 50),
        crown_coverage=np.random.uniform(0.3, 0.7),
        species_distribution={
            "cacao": np.random.randint(5, 25),
            "shade_tree": np.random.randint(3, 15),
            "other": np.random.randint(2, 10)
        },
        confidence=np.random.uniform(0.85, 0.95),
        processing_time=np.random.uniform(1.0, 3.0)
    )
    
    # Track completion
    agentops.record(agentops.ActionEvent(
        action_type="segmentation_completed",
        params=result.dict()
    ))
    
    return result

@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "total_requests": np.random.randint(100, 1000),
        "average_processing_time": np.random.uniform(1.5, 2.5),
        "success_rate": 0.98,
        "model_version": "1.0.0",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/plantation-data")
async def get_plantation_data():
    """Get plantation data if available."""
    plantation_path = os.environ.get('PLANTATION_DATA_PATH', 'plantation-data.json')
    
    if os.path.exists(plantation_path):
        with open(plantation_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        return {
            "message": "Plantation data not found",
            "sample_data": {
                "farms": 3,
                "total_area_hectares": 150,
                "average_tree_density": 450
            }
        }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment or defaults
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    print(f"""
    🌳 Cabruca Segmentation API
    ============================
    Starting server at http://{host}:{port}
    Documentation at http://{host}:{port}/docs
    
    Model: {'Loaded' if MODEL_LOADED else 'Mock mode'}
    AgentOps: Enabled
    """)
    
    uvicorn.run(app, host=host, port=port)