"""Data processing module for cabruca segmentation project"""

# Import main classes for easy access
try:
    from .geospatial_loader import (AOIBounds, CoordinateConverter,
                                    GeoJSONLoader, GeospatialDataPipeline,
                                    MultiBandProcessor, SatelliteImageryLoader)

    FULL_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full geospatial loader not available. Error: {e}")
    print(
        "Using simple loader instead. Install requirements with: pip install -r requirements.txt"
    )
    FULL_LOADER_AVAILABLE = False

# Always available simple loader
from .simple_loader import (BandCalculations, CoordinateTransformer,
                            ImageryMetadata, SimpleAOI, SimpleGeoJSONLoader)

__all__ = [
    # Simple loader (always available)
    "SimpleAOI",
    "SimpleGeoJSONLoader",
    "CoordinateTransformer",
    "ImageryMetadata",
    "BandCalculations",
    "FULL_LOADER_AVAILABLE",
]

# Add full loader exports if available
if FULL_LOADER_AVAILABLE:
    __all__.extend(
        [
            "AOIBounds",
            "GeoJSONLoader",
            "CoordinateConverter",
            "SatelliteImageryLoader",
            "MultiBandProcessor",
            "GeospatialDataPipeline",
        ]
    )
