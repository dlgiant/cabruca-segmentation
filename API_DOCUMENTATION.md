# Cabruca Segmentation API Documentation

## Overview

The Cabruca Segmentation API provides REST endpoints for ML-based segmentation of agroforestry images, specifically designed for Cabruca systems. It offers tree detection, species classification, canopy analysis, and integration with existing plantation data.

## Quick Start

### 1. Start the API Server

```bash
# Install dependencies
pip install fastapi uvicorn[standard] python-multipart

# Start server
python api_server.py --model outputs/checkpoint_best.pth --port 8000

# Or with Docker
docker-compose -f docker/docker-compose.yml up
```

### 2. Access API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

### 3. Python Client Example

```python
from api.client import CabrucaAPIClient

# Initialize client
client = CabrucaAPIClient("http://localhost:8000")

# Process image
result = client.process_image("plantation.jpg", confidence_threshold=0.6)
print(f"Detected {len(result['trees'])} trees")

# Get health metrics
comparison = client.compare_with_plantation("plantation.jpg", "plantation-data.json")
print(f"Health Score: {comparison['health_report']['overall_score']:.2%}")
```

## API Endpoints

### Core Endpoints

#### `GET /health`
Health check endpoint to verify API status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "integration_available": true,
  "timestamp": "2024-01-20T10:30:00"
}
```

#### `POST /inference`
Process a single image for tree detection and segmentation.

**Parameters:**
- `file`: Image file (multipart/form-data)
- `confidence_threshold`: Min confidence (default: 0.5)
- `tile_size`: Tile size for processing (default: 512)
- `overlap`: Tile overlap (default: 64)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "timestamp": "2024-01-20T10:30:00",
  "trees": [
    {
      "id": 1,
      "species": "cacao",
      "confidence": 0.95,
      "centroid": [100.5, 200.3],
      "crown_diameter": 3.5,
      "crown_area": 9.62
    }
  ],
  "metrics": {
    "cacao_tree_coverage": 0.35,
    "shade_tree_coverage": 0.15,
    "understory_coverage": 0.30
  },
  "canopy_density": 0.45,
  "processing_time": 2.5
}
```

#### `POST /batch`
Process multiple images in batch mode.

**Request Body:**
```json
{
  "image_paths": ["path1.jpg", "path2.jpg"],
  "output_format": "json",
  "generate_report": true
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "message": "Processing 2 images",
  "check_status": "/jobs/uuid"
}
```

#### `POST /compare`
Compare ML detection with existing plantation coordinates.

**Parameters:**
- `file`: Image file
- `plantation_data`: Plantation data JSON (optional)
- `distance_threshold`: Max matching distance in meters (default: 2.0)

**Response:**
```json
{
  "job_id": "uuid",
  "ml_trees_detected": 150,
  "plantation_trees_expected": 145,
  "statistics": {
    "matched_trees": 140,
    "detection_f1": 0.93,
    "position_rmse": 1.2,
    "species_accuracy": 0.95
  },
  "health_report": {
    "overall_score": 0.85,
    "status": "healthy",
    "tree_counts": {
      "cacao_trees": 120,
      "shade_trees": 30
    },
    "canopy_metrics": {
      "canopy_density": 0.45,
      "canopy_coverage": 0.50
    },
    "recommendations": [
      "Consider planting additional shade trees"
    ]
  }
}
```

### Job Management

#### `GET /jobs/{job_id}`
Check status of a processing job.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "progress": "10/10",
  "results": {...}
}
```

#### `DELETE /jobs/{job_id}`
Delete a job and its associated files.

### Export Endpoints

#### `GET /results/{job_id}/visualization`
Download visualization image for a completed job.

**Response:** PNG image file

#### `GET /results/{job_id}/geojson`
Export results as GeoJSON for GIS integration.

**Response:** GeoJSON file

## Python Client

### Installation

```python
# Install required packages
pip install requests pandas
```

### Basic Usage

```python
from api.client import CabrucaAPIClient

# Initialize client
client = CabrucaAPIClient("http://localhost:8000")

# Process single image
result = client.process_image(
    "plantation.jpg",
    confidence_threshold=0.6,
    tile_size=512
)

# Parse results
trees = client.parse_trees(result)
df = client.trees_to_dataframe(trees)
print(df.head())

# Get visualization
client.get_visualization(result['job_id'], "output.png")

# Export to GeoJSON
client.get_geojson(result['job_id'], "trees.geojson")
```

### Advanced Usage

```python
# Batch processing
job = client.process_batch(
    ["image1.jpg", "image2.jpg"],
    output_format="excel",
    generate_report=True
)

# Wait for completion
result = client.wait_for_job(job['job_id'])

# Compare with plantation data
comparison = client.compare_with_plantation(
    "plantation.jpg",
    "plantation_data.json",
    distance_threshold=2.0
)

# Health analysis
health = comparison['health_report']
print(f"Overall Score: {health['overall_score']:.2%}")
print(f"Status: {health['status']}")

for recommendation in health['recommendations']:
    print(f"- {recommendation}")
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -f docker/Dockerfile -t cabruca-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs:ro \
  -v $(pwd)/plantation-data.json:/app/plantation-data.json:ro \
  cabruca-api
```

### Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose logs -f cabruca-api

# Stop services
docker-compose down
```

## Integration Examples

### JavaScript/TypeScript

```javascript
// Using fetch API
async function processImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8000/inference', {
    method: 'POST',
    body: formData,
    params: {
      confidence_threshold: 0.6
    }
  });
  
  const result = await response.json();
  console.log(`Detected ${result.trees.length} trees`);
  return result;
}
```

### cURL

```bash
# Single image inference
curl -X POST "http://localhost:8000/inference" \
  -F "file=@plantation.jpg" \
  -F "confidence_threshold=0.6"

# Health check
curl http://localhost:8000/health

# Get job status
curl http://localhost:8000/jobs/{job_id}
```

### Integration with Theobroma Digital

```python
# Update existing plantation system
from api.client import CabrucaAPIClient

client = CabrucaAPIClient("http://localhost:8000")

# Process plantation image
result = client.compare_with_plantation(
    "current_plantation.jpg",
    "plantation_coordinates.json"
)

# Update plantation database with ML insights
plantation_update = {
    'ml_tree_count': result['ml_trees_detected'],
    'detection_confidence': result['statistics']['detection_f1'],
    'health_score': result['health_report']['overall_score'],
    'last_analysis': result['timestamp']
}

# Use results for decision making
if result['health_report']['overall_score'] < 0.7:
    for recommendation in result['health_report']['recommendations']:
        print(f"Action needed: {recommendation}")
```

## Configuration

### Environment Variables

```bash
# Model configuration
export MODEL_PATH=/path/to/model.pth
export PLANTATION_DATA_PATH=/path/to/plantation-data.json

# API configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
```

### Model Configuration

Create `config.yaml`:

```yaml
model:
  num_instance_classes: 3
  num_semantic_classes: 5
  pretrained: false

inference:
  confidence_threshold: 0.5
  tile_size: 512
  overlap: 64
  batch_size: 4
```

## Performance Optimization

### Caching

The API includes automatic caching for repeated requests:
- Results cached for 15 minutes
- Automatic cleanup of old jobs
- Redis support for distributed caching

### Batch Processing

For multiple images, use batch endpoint for better performance:

```python
# Process 100 images efficiently
images = ["image_{}.jpg".format(i) for i in range(100)]
job = client.process_batch(images, output_format="excel")
```

### Resource Management

Configure resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```

## Error Handling

### Common Error Codes

- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Job or resource not found
- `413`: Payload Too Large - Image file too large
- `500`: Internal Server Error - Processing failed
- `503`: Service Unavailable - Model not loaded

### Error Response Format

```json
{
  "detail": "Error message",
  "status_code": 400,
  "timestamp": "2024-01-20T10:30:00"
}
```

## Security

### Authentication (Optional)

Add API key authentication:

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/inference")
async def inference(api_key: str = Depends(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=401)
    # Process request
```

### Rate Limiting

Configure Nginx rate limiting in `docker/nginx.conf`:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req zone=api_limit burst=20 nodelay;
```

## Monitoring

### Health Checks

```python
# Automated health monitoring
import requests
import time

def monitor_api(url, interval=60):
    while True:
        try:
            response = requests.get(f"{url}/health")
            health = response.json()
            print(f"Status: {health['status']}")
        except:
            print("API unavailable")
        time.sleep(interval)
```

### Logging

View API logs:

```bash
# Docker logs
docker logs cabruca-segmentation-api

# System logs
tail -f api_server.log
```

## Support

- GitHub Issues: [Report bugs or request features]
- Documentation: [Full API documentation]
- Examples: See `test_api.py` for complete examples