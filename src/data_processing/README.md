# Geospatial Data Integration Pipeline

This module provides comprehensive data loaders for working with geospatial data, including GeoJSON files, satellite imagery, and coordinate transformations.

## Features

### Core Functionality
- ✅ Read and parse GeoJSON files
- ✅ Extract Area of Interest (AOI) boundaries
- ✅ Convert between geographic and pixel coordinates
- ✅ Handle multi-band satellite imagery (RGB, NIR, etc.)
- ✅ Calculate vegetation indices (NDVI, EVI)
- ✅ Interface with Planet Labs API for satellite imagery
- ✅ Support for local satellite imagery files (GeoTIFF, etc.)

## Installation

### Basic Installation (Simple Loader Only)
The simple loader works with Python standard library only:
```bash
# No additional dependencies needed for simple_loader.py
python test_data_loader.py
```

### Full Installation (All Features)
```bash
# Install all dependencies
pip install -r requirements.txt

# For Planet Labs integration, set your API key
export PLANET_API_KEY='your-api-key-here'
```

## Usage

### Simple Loader (No Dependencies)

```python
from src.data_processing import SimpleGeoJSONLoader, CoordinateTransformer

# Load GeoJSON
loader = SimpleGeoJSONLoader("farm1.geojson")
aois = loader.extract_aois()

for aoi in aois:
    print(f"AOI: {aoi.name}")
    print(f"Center: {aoi.center}")
    print(f"Area: {aoi.area_approx_hectares} hectares")
    
# Convert coordinates
transformer = CoordinateTransformer()
pixel_x, pixel_y = transformer.geo_to_pixel_simple(
    lon=-39.5, lat=-15.4,
    bounds=(-39.6, -15.5, -39.4, -15.3),
    image_width=1000, image_height=1000
)
```

### Full Loader (With Dependencies)

```python
from src.data_processing import GeospatialDataPipeline

# Initialize pipeline
pipeline = GeospatialDataPipeline(planet_api_key="your-key")

# Load AOIs from GeoJSON
aois = pipeline.load_aoi_from_geojson("farm1.geojson")

# Process local imagery
bands = pipeline.process_local_imagery_for_aoi(
    image_path="satellite_image.tif",
    aoi=aois[0],
    calculate_indices=True
)

# Access processed data
ndvi = bands.get('ndvi')
coord_converter = bands.get('coord_converter')

# Prepare for model training
training_data = pipeline.prepare_training_data(
    bands=bands,
    normalize=True,
    stack=True,
    band_order=['red', 'green', 'blue', 'nir']
)
```

### Planet Labs Integration

```python
from src.data_processing import SatelliteImageryLoader
from datetime import datetime, timedelta

loader = SatelliteImageryLoader(api_key="your-key")

# Search for imagery
results = loader.search_planet_imagery(
    aoi=aoi,
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    item_types=["PSScene"],
    cloud_cover_max=0.1
)

# Download imagery
if results:
    image_path = loader.download_planet_image(
        item_id=results[0]['id'],
        asset_type="ortho_analytic_4b"
    )
```

## Module Structure

```
data_processing/
├── __init__.py           # Module initialization and imports
├── simple_loader.py      # Lightweight loader (no dependencies)
├── geospatial_loader.py  # Full-featured loader
├── README.md            # This file
└── test_data_loader.py  # Test suite
```

## Classes Overview

### Simple Loader (simple_loader.py)
- `SimpleAOI`: Basic AOI representation
- `SimpleGeoJSONLoader`: GeoJSON file reader
- `CoordinateTransformer`: Simple coordinate conversions
- `ImageryMetadata`: Metadata storage
- `BandCalculations`: Basic band math (requires NumPy)

### Full Loader (geospatial_loader.py)
- `AOIBounds`: Advanced AOI with accurate area calculations
- `GeoJSONLoader`: Full GeoJSON support with GeoPandas
- `CoordinateConverter`: Affine transform-based conversions
- `SatelliteImageryLoader`: Local and remote imagery loading
- `MultiBandProcessor`: Advanced band processing
- `GeospatialDataPipeline`: Main integration pipeline

## Supported Formats

### Input Formats
- **GeoJSON**: FeatureCollection, Feature, Polygon geometries
- **Raster**: GeoTIFF, any format supported by rasterio
- **Planet Labs**: PSScene, SkySat, RapidEye items

### Band Configurations
- **3-band**: RGB
- **4-band**: RGB + NIR
- **5-band**: Blue, Green, Red, RedEdge, NIR
- **8-band**: Planet SuperDove constellation

## Vegetation Indices

### NDVI (Normalized Difference Vegetation Index)
```python
ndvi = (nir - red) / (nir + red)
```
Range: -1 to 1, healthy vegetation typically > 0.3

### EVI (Enhanced Vegetation Index)
```python
evi = 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1))
```
More sensitive to high biomass regions

## Coordinate Systems

- Default: **EPSG:4326** (WGS84 Geographic)
- Automatic UTM zone detection for area calculations
- Support for any CRS via pyproj transformations

## Testing

Run the test suite:
```bash
python test_data_loader.py
```

This will test:
1. Simple loader functionality
2. Full loader (if dependencies installed)
3. Planet Labs integration (if API key set)

## Examples

### Example 1: Extract Multiple AOIs
```python
loader = SimpleGeoJSONLoader("regions.geojson")
aois = loader.extract_aois()

for i, aoi in enumerate(aois):
    planet_format = loader.to_planet_format(i)
    print(f"AOI {i}: {aoi.name}")
    print(f"  Planet format: {planet_format}")
```

### Example 2: Process Imagery with AOI Cropping
```python
pipeline = GeospatialDataPipeline()
aoi = pipeline.load_aoi_from_geojson("farm1.geojson")[0]

# Load and crop imagery to AOI
bands = pipeline.process_local_imagery_for_aoi(
    "landsat_scene.tif", 
    aoi,
    calculate_indices=True
)

# Stack bands for CNN input
stacked = pipeline.prepare_training_data(
    bands, 
    normalize=True,
    band_order=['red', 'green', 'blue', 'ndvi']
)
print(f"Training data shape: {stacked.shape}")
```

### Example 3: Coordinate Conversion Pipeline
```python
# With full loader
converter = CoordinateConverter(raster_transform, "EPSG:4326")
col, row = converter.geo_to_pixel(-39.5, -15.4)
lon, lat = converter.pixel_to_geo(col, row)

# With simple loader
pixel_x, pixel_y = CoordinateTransformer.geo_to_pixel_simple(
    -39.5, -15.4, bounds, width, height
)
```

## Troubleshooting

### Missing Dependencies
```
Error: No module named 'rasterio'
Solution: pip install -r requirements.txt
```

### Planet API Issues
```
Error: Planet API key not provided
Solution: export PLANET_API_KEY='your-key'
```

### Memory Issues with Large Images
```python
# Process in chunks
for chunk in process_image_chunks(image_path, chunk_size=1024):
    processed = pipeline.process_chunk(chunk)
```

## Performance Tips

1. **Use simple loader for basic operations** - No dependencies, faster startup
2. **Cache AOI calculations** - Avoid recalculating geometry operations
3. **Process imagery in tiles** - For images > 10,000x10,000 pixels
4. **Use appropriate resampling** - Bilinear for continuous data, nearest for categorical

## Future Enhancements

- [ ] Support for Sentinel-2 imagery
- [ ] Integration with Google Earth Engine
- [ ] Parallel processing for large datasets
- [ ] Time series analysis utilities
- [ ] Cloud mask generation
- [ ] Automatic image co-registration

## License

Part of the Cabruca Segmentation project. See main project LICENSE.

## Contact

For issues or questions about the data loaders, please open an issue in the main project repository.
