"""
FastAPI service for Cabruca segmentation inference.
Provides REST endpoints for ML model predictions and analysis.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentops
import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
# Import ML components
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.agroforestry_metrics import AgroforestryMetrics
from inference.batch_inference import BatchInferenceEngine, InferenceResult
from integration.theobroma_integration import TheobromaIntegration


# Pydantic models for API
class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""

    image_url: Optional[str] = Field(None, description="URL of image to process")
    confidence_threshold: float = Field(
        0.5, description="Confidence threshold for detections"
    )
    tile_size: int = Field(512, description="Tile size for large images")
    overlap: int = Field(64, description="Overlap between tiles")


class TreeDetection(BaseModel):
    """Tree detection result."""

    id: int
    species: str
    confidence: float
    centroid: List[float]
    crown_diameter: float
    crown_area: float
    bbox: List[float]


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""

    job_id: str
    status: str
    timestamp: str
    image_path: str
    trees: List[TreeDetection]
    metrics: Dict[str, float]
    canopy_density: float
    processing_time: float


class HealthReport(BaseModel):
    """Plantation health report model."""

    overall_score: float
    status: str
    tree_counts: Dict[str, int]
    canopy_metrics: Dict[str, float]
    recommendations: List[str]


class ComparisonRequest(BaseModel):
    """Request for comparing ML with plantation data."""

    plantation_data_path: str
    distance_threshold: float = Field(
        2.0, description="Maximum distance for tree matching (meters)"
    )


class BatchInferenceRequest(BaseModel):
    """Request for batch processing."""

    image_paths: List[str]
    output_format: str = Field(
        "json", description="Output format: json, geojson, excel"
    )
    generate_report: bool = Field(False, description="Generate analysis report")


# Initialize FastAPI app
app = FastAPI(
    title="Cabruca Segmentation API",
    description="ML inference API for Cabruca agroforestry segmentation",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and job tracking
inference_engine = None
integration_module = None
job_store = {}
UPLOAD_DIR = Path("api_uploads")
RESULTS_DIR = Path("api_results")
MODEL_PATH = os.getenv("MODEL_PATH", "outputs/checkpoint_best.pth")
PLANTATION_DATA_PATH = os.getenv("PLANTATION_DATA_PATH", "plantation-data.json")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global inference_engine, integration_module

    # Initialize AgentOps for API service monitoring
    agentops.init(tags=["cabruca", "api-service", "inference"])

    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        inference_engine = BatchInferenceEngine(model_path=MODEL_PATH, device="auto")

        if os.path.exists(PLANTATION_DATA_PATH):
            integration_module = TheobromaIntegration(MODEL_PATH, PLANTATION_DATA_PATH)

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Cabruca Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/inference": "Single image inference",
            "/batch": "Batch processing",
            "/compare": "Compare with plantation data",
            "/jobs/{job_id}": "Check job status",
            "/docs": "API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "integration_available": integration_module is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/inference", response_model=InferenceResponse)
@agentops.trace(name="Single Image Inference")
async def inference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    tile_size: int = 512,
    overlap: int = 64,
):
    """
    Perform inference on a single image.

    Args:
        file: Image file to process
        confidence_threshold: Minimum confidence for detections
        tile_size: Size of tiles for large images
        overlap: Overlap between tiles

    Returns:
        Inference results with detected trees and metrics
    """
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Configure inference
        inference_engine.config["confidence_threshold"] = confidence_threshold
        inference_engine.config["tile_size"] = tile_size
        inference_engine.config["overlap"] = overlap

        # Process image
        import time

        start_time = time.time()
        result = inference_engine.process_single(str(upload_path))
        processing_time = time.time() - start_time

        # Convert trees to response format
        trees = []
        for tree in result.trees:
            trees.append(
                TreeDetection(
                    id=tree.id,
                    species=tree.species,
                    confidence=float(tree.confidence),
                    centroid=list(tree.centroid),
                    crown_diameter=float(tree.crown_diameter),
                    crown_area=float(tree.crown_area),
                    bbox=list(tree.bbox) if hasattr(tree, "bbox") else [0, 0, 0, 0],
                )
            )

        # Prepare response
        response = InferenceResponse(
            job_id=job_id,
            status="completed",
            timestamp=datetime.now().isoformat(),
            image_path=str(upload_path),
            trees=trees,
            metrics=result.metrics,
            canopy_density=float(result.canopy_density),
            processing_time=processing_time,
        )

        # Store result
        job_store[job_id] = {
            "status": "completed",
            "result": result,
            "response": response.dict(),
        }

        # Schedule cleanup
        background_tasks.add_task(cleanup_job, job_id, delay=3600)

        return response

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
@agentops.trace(name="Batch Inference Processing")
async def batch_inference(
    background_tasks: BackgroundTasks, request: BatchInferenceRequest
):
    """
    Process multiple images in batch.

    Args:
        request: Batch processing request

    Returns:
        Job ID for tracking batch processing
    """
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Start background processing
    background_tasks.add_task(
        process_batch_job,
        job_id,
        request.image_paths,
        request.output_format,
        request.generate_report,
    )

    # Store initial job status
    job_store[job_id] = {
        "status": "processing",
        "total_images": len(request.image_paths),
        "processed": 0,
        "start_time": datetime.now().isoformat(),
    }

    return {
        "job_id": job_id,
        "status": "processing",
        "message": f"Processing {len(request.image_paths)} images",
        "check_status": f"/jobs/{job_id}",
    }


@app.post("/compare")
@agentops.trace(name="ML Plantation Comparison")
async def compare_with_plantation(
    file: UploadFile = File(...),
    plantation_data: UploadFile = File(None),
    distance_threshold: float = 2.0,
):
    """
    Compare ML detection with plantation coordinates.

    Args:
        file: Image to process
        plantation_data: Optional plantation data JSON
        distance_threshold: Maximum distance for matching

    Returns:
        Comparison results and statistics
    """
    if not integration_module and not plantation_data:
        raise HTTPException(
            status_code=400,
            detail="No plantation data available. Upload plantation data or configure PLANTATION_DATA_PATH",
        )

    # Save uploaded image
    job_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Use uploaded plantation data if provided
        if plantation_data:
            plantation_path = UPLOAD_DIR / f"{job_id}_plantation.json"
            with open(plantation_path, "wb") as buffer:
                shutil.copyfileobj(plantation_data.file, buffer)

            integration = TheobromaIntegration(MODEL_PATH, str(plantation_path))
        else:
            integration = integration_module

        # Perform comparison
        comparison = integration.compare_with_ml_detection(
            str(image_path), distance_threshold
        )

        # Generate health metrics
        health_metrics = integration.generate_health_metrics(comparison["ml_result"])

        # Prepare response
        response = {
            "job_id": job_id,
            "timestamp": comparison["timestamp"],
            "ml_trees_detected": comparison["ml_trees_detected"],
            "plantation_trees_expected": comparison["plantation_trees_expected"],
            "statistics": comparison["statistics"],
            "health_report": {
                "overall_score": health_metrics["plantation_health"].get(
                    "overall_score", 0
                ),
                "status": health_metrics["plantation_health"].get("status", "N/A"),
                "tree_counts": health_metrics["tree_health"],
                "canopy_metrics": health_metrics["canopy_health"],
                "recommendations": health_metrics["recommendations"],
            },
        }

        return response

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Check status of a processing job.

    Args:
        job_id: Job identifier

    Returns:
        Job status and results if completed
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]

    if job["status"] == "completed":
        return job.get("response", job)
    else:
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": f"{job.get('processed', 0)}/{job.get('total_images', 0)}",
        }


@app.get("/results/{job_id}/visualization")
async def get_visualization(job_id: str):
    """
    Get visualization for completed job.

    Args:
        job_id: Job identifier

    Returns:
        Visualization image file
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    # Generate visualization
    viz_path = RESULTS_DIR / f"{job_id}_visualization.png"

    if not viz_path.exists():
        result = job.get("result")
        if not result:
            raise HTTPException(status_code=404, detail="No result available")

        # Load image and create visualization
        image_path = job.get("response", {}).get("image_path")
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            from inference.batch_inference import VisualizationTools

            VisualizationTools.create_comparison_figure(image, result, str(viz_path))

    if viz_path.exists():
        return FileResponse(viz_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Visualization not available")


@app.get("/results/{job_id}/geojson")
async def get_geojson(job_id: str):
    """
    Export results as GeoJSON.

    Args:
        job_id: Job identifier

    Returns:
        GeoJSON file
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    result = job.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="No result available")

    # Generate GeoJSON
    geojson_path = RESULTS_DIR / f"{job_id}.geojson"

    if not geojson_path.exists():
        from inference.batch_inference import ReportGenerator

        ReportGenerator.export_to_geojson(result, str(geojson_path))

    if geojson_path.exists():
        return FileResponse(
            geojson_path,
            media_type="application/json",
            filename=f"cabruca_{job_id}.geojson",
        )
    else:
        raise HTTPException(status_code=404, detail="GeoJSON not available")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.

    Args:
        job_id: Job identifier

    Returns:
        Deletion confirmation
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete job from store
    del job_store[job_id]

    # Delete associated files
    for directory in [UPLOAD_DIR, RESULTS_DIR]:
        for file in directory.glob(f"{job_id}*"):
            file.unlink()

    return {"message": f"Job {job_id} deleted successfully"}


# Background task functions
async def process_batch_job(
    job_id: str, image_paths: List[str], output_format: str, generate_report: bool
):
    """Process batch inference job."""
    try:
        # Process images
        results = []
        for i, image_path in enumerate(image_paths):
            result = inference_engine.process_single(image_path)
            results.append(result)

            # Update progress
            job_store[job_id]["processed"] = i + 1

        # Generate outputs based on format
        output_path = RESULTS_DIR / job_id
        output_path.mkdir(exist_ok=True)

        if output_format == "geojson":
            from inference.batch_inference import ReportGenerator

            for i, result in enumerate(results):
                geojson_path = output_path / f"result_{i}.geojson"
                ReportGenerator.export_to_geojson(result, str(geojson_path))

        elif output_format == "excel":
            from inference.batch_inference import ReportGenerator

            excel_path = output_path / "tree_inventory.xlsx"
            ReportGenerator.generate_tree_inventory(results, str(excel_path))

        if generate_report:
            from inference.batch_inference import ReportGenerator

            ReportGenerator.generate_analysis_report(results, str(output_path))

        # Update job status
        job_store[job_id].update(
            {
                "status": "completed",
                "results": [r.__dict__ for r in results],
                "output_path": str(output_path),
                "end_time": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        job_store[job_id].update(
            {
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat(),
            }
        )


async def cleanup_job(job_id: str, delay: int = 3600):
    """Clean up job after delay."""
    await asyncio.sleep(delay)

    if job_id in job_store:
        # Delete files
        for directory in [UPLOAD_DIR, RESULTS_DIR]:
            for file in directory.glob(f"{job_id}*"):
                file.unlink()

        # Remove from store
        del job_store[job_id]
        logger.info(f"Cleaned up job {job_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
