"""
Simplified Geospatial Data Loader
A lightweight version that works with basic dependencies for initial testing.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import math


@dataclass
class SimpleAOI:
    """Simple Area of Interest representation"""
    name: str
    bounds: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    coordinates: List[List[float]]
    properties: Dict[str, Any]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point"""
        return (
            (self.bounds[0] + self.bounds[2]) / 2,
            (self.bounds[1] + self.bounds[3]) / 2
        )
    
    @property
    def area_approx_hectares(self) -> float:
        """Approximate area calculation in hectares"""
        # Simple approximation using equirectangular projection
        lat_center = self.center[1]
        lat_rad = math.radians(lat_center)
        
        # Approximate meters per degree
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(lat_rad)
        
        width_m = abs(self.bounds[2] - self.bounds[0]) * m_per_deg_lon
        height_m = abs(self.bounds[3] - self.bounds[1]) * m_per_deg_lat
        
        area_m2 = width_m * height_m
        return area_m2 / 10000  # Convert to hectares


class SimpleGeoJSONLoader:
    """Simple GeoJSON file loader"""
    
    def __init__(self, geojson_path: Union[str, Path]):
        """
        Initialize the loader
        
        Args:
            geojson_path: Path to GeoJSON file
        """
        self.path = Path(geojson_path)
        if not self.path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
        with open(self.path, 'r') as f:
            self.data = json.load(f)
    
    def extract_aois(self) -> List[SimpleAOI]:
        """
        Extract all AOIs from the GeoJSON
        
        Returns:
            List of SimpleAOI objects
        """
        aois = []
        
        if self.data.get('type') == 'FeatureCollection':
            features = self.data.get('features', [])
        elif self.data.get('type') == 'Feature':
            features = [self.data]
        else:
            # Assume it's a single geometry
            features = [{'geometry': self.data, 'properties': {}}]
        
        for feature in features:
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            if geometry.get('type') == 'Polygon':
                coords = geometry.get('coordinates', [[]])[0]
                if coords:
                    # Calculate bounds
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    bounds = (min(lons), min(lats), max(lons), max(lats))
                    
                    aoi = SimpleAOI(
                        name=properties.get('name', 'Unnamed AOI'),
                        bounds=bounds,
                        coordinates=coords,
                        properties=properties
                    )
                    aois.append(aoi)
        
        return aois
    
    def to_planet_format(self, aoi_index: int = 0) -> Dict:
        """
        Convert AOI to Planet Labs compatible format
        
        Args:
            aoi_index: Index of AOI to convert
        
        Returns:
            Dictionary in Planet format
        """
        aois = self.extract_aois()
        if aoi_index >= len(aois):
            raise IndexError(f"AOI index {aoi_index} out of range")
        
        aoi = aois[aoi_index]
        return {
            "type": "Polygon",
            "coordinates": [aoi.coordinates]
        }


class CoordinateTransformer:
    """Simple coordinate transformation utilities"""
    
    @staticmethod
    def geo_to_pixel_simple(lon: float, lat: float, 
                          bounds: Tuple[float, float, float, float],
                          image_width: int, image_height: int) -> Tuple[int, int]:
        """
        Simple geographic to pixel coordinate conversion
        
        Args:
            lon: Longitude
            lat: Latitude
            bounds: Image geographic bounds (min_lon, min_lat, max_lon, max_lat)
            image_width: Width of image in pixels
            image_height: Height of image in pixels
        
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Normalize to 0-1 range
        x_norm = (lon - min_lon) / (max_lon - min_lon)
        y_norm = (max_lat - lat) / (max_lat - min_lat)  # Inverted for image coordinates
        
        # Convert to pixel coordinates
        x = int(x_norm * image_width)
        y = int(y_norm * image_height)
        
        # Clamp to image bounds
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        
        return x, y
    
    @staticmethod
    def pixel_to_geo_simple(x: int, y: int,
                          bounds: Tuple[float, float, float, float],
                          image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Simple pixel to geographic coordinate conversion
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            bounds: Image geographic bounds
            image_width: Width of image in pixels
            image_height: Height of image in pixels
        
        Returns:
            Tuple of (longitude, latitude)
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Normalize pixel coordinates
        x_norm = x / image_width
        y_norm = y / image_height
        
        # Convert to geographic coordinates
        lon = min_lon + x_norm * (max_lon - min_lon)
        lat = max_lat - y_norm * (max_lat - min_lat)  # Inverted from image coordinates
        
        return lon, lat


class ImageryMetadata:
    """Store and manage imagery metadata"""
    
    def __init__(self, width: int, height: int, 
                 bounds: Tuple[float, float, float, float],
                 bands: List[str], 
                 crs: str = "EPSG:4326"):
        """
        Initialize imagery metadata
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            bounds: Geographic bounds (min_lon, min_lat, max_lon, max_lat)
            bands: List of band names
            crs: Coordinate reference system
        """
        self.width = width
        self.height = height
        self.bounds = bounds
        self.bands = bands
        self.crs = crs
        self.pixel_resolution = self._calculate_resolution()
    
    def _calculate_resolution(self) -> Tuple[float, float]:
        """Calculate approximate pixel resolution in meters"""
        lat_center = (self.bounds[1] + self.bounds[3]) / 2
        lat_rad = math.radians(lat_center)
        
        # Meters per degree
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(lat_rad)
        
        # Resolution
        res_lon = abs(self.bounds[2] - self.bounds[0]) / self.width * m_per_deg_lon
        res_lat = abs(self.bounds[3] - self.bounds[1]) / self.height * m_per_deg_lat
        
        return res_lon, res_lat
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            'width': self.width,
            'height': self.height,
            'bounds': self.bounds,
            'bands': self.bands,
            'crs': self.crs,
            'pixel_resolution_m': self.pixel_resolution
        }


class BandCalculations:
    """Simple band math calculations"""
    
    @staticmethod
    def calculate_ndvi(red: Any, nir: Any) -> Any:
        """
        Calculate NDVI (requires numpy)
        
        Args:
            red: Red band array
            nir: NIR band array
        
        Returns:
            NDVI array
        """
        try:
            import numpy as np
            red = np.asarray(red, dtype=float)
            nir = np.asarray(nir, dtype=float)
            
            # Avoid division by zero
            denominator = nir + red
            denominator[denominator == 0] = 1e-10
            
            ndvi = (nir - red) / denominator
            return ndvi
        except ImportError:
            raise ImportError("NumPy is required for NDVI calculation")
    
    @staticmethod
    def normalize_band(band: Any, method: str = 'minmax') -> Any:
        """
        Normalize band values (requires numpy)
        
        Args:
            band: Band array
            method: Normalization method
        
        Returns:
            Normalized array
        """
        try:
            import numpy as np
            band = np.asarray(band, dtype=float)
            
            if method == 'minmax':
                min_val = np.min(band)
                max_val = np.max(band)
                if max_val > min_val:
                    return (band - min_val) / (max_val - min_val)
                else:
                    return np.zeros_like(band)
            
            elif method == 'standardize':
                mean = np.mean(band)
                std = np.std(band)
                if std > 0:
                    return (band - mean) / std
                else:
                    return band - mean
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
        except ImportError:
            raise ImportError("NumPy is required for band normalization")


def demo_usage():
    """Demonstrate usage of the simple loader"""
    
    print("Simple Geospatial Data Loader Demo")
    print("=" * 50)
    
    # Load GeoJSON files
    geojson_files = ["farm1.geojson", "camacan-geo.geojson"]
    
    for geojson_file in geojson_files:
        try:
            print(f"\nLoading {geojson_file}...")
            loader = SimpleGeoJSONLoader(geojson_file)
            aois = loader.extract_aois()
            
            print(f"Found {len(aois)} AOI(s)")
            
            for i, aoi in enumerate(aois):
                print(f"\n  AOI {i + 1}:")
                print(f"    Name: {aoi.name}")
                print(f"    Bounds: {aoi.bounds}")
                print(f"    Center: {aoi.center}")
                print(f"    Approx. Area: {aoi.area_approx_hectares:.2f} hectares")
                print(f"    Properties: {list(aoi.properties.keys())}")
                
                # Show Planet format
                if i == 0:  # Just show for first AOI
                    planet_aoi = loader.to_planet_format(i)
                    print(f"    Planet Format: {planet_aoi['type']} with {len(planet_aoi['coordinates'][0])} vertices")
        
        except FileNotFoundError:
            print(f"  File not found: {geojson_file}")
        except Exception as e:
            print(f"  Error loading {geojson_file}: {e}")
    
    # Demonstrate coordinate conversion
    print("\n" + "=" * 50)
    print("Coordinate Conversion Example:")
    
    # Example image metadata
    example_bounds = (-39.6, -15.5, -39.35, -15.3)
    image_width, image_height = 1000, 800
    
    # Convert center point to pixel
    center_lon = (example_bounds[0] + example_bounds[2]) / 2
    center_lat = (example_bounds[1] + example_bounds[3]) / 2
    
    pixel_x, pixel_y = CoordinateTransformer.geo_to_pixel_simple(
        center_lon, center_lat, example_bounds, image_width, image_height
    )
    
    print(f"  Geographic: ({center_lon:.4f}, {center_lat:.4f})")
    print(f"  Pixel: ({pixel_x}, {pixel_y})")
    
    # Convert back
    lon_back, lat_back = CoordinateTransformer.pixel_to_geo_simple(
        pixel_x, pixel_y, example_bounds, image_width, image_height
    )
    print(f"  Back to Geographic: ({lon_back:.4f}, {lat_back:.4f})")
    
    # Show metadata example
    print("\n" + "=" * 50)
    print("Imagery Metadata Example:")
    
    metadata = ImageryMetadata(
        width=image_width,
        height=image_height,
        bounds=example_bounds,
        bands=['red', 'green', 'blue', 'nir']
    )
    
    meta_dict = metadata.to_dict()
    print(f"  Image Size: {meta_dict['width']}x{meta_dict['height']} pixels")
    print(f"  Bands: {', '.join(meta_dict['bands'])}")
    print(f"  Pixel Resolution: {meta_dict['pixel_resolution_m'][0]:.2f}m x {meta_dict['pixel_resolution_m'][1]:.2f}m")
    
    # Try NDVI calculation if numpy is available
    try:
        import numpy as np
        print("\n" + "=" * 50)
        print("Band Calculations Example (NumPy available):")
        
        # Create synthetic data
        red = np.random.rand(100, 100) * 0.3
        nir = np.random.rand(100, 100) * 0.7
        
        ndvi = BandCalculations.calculate_ndvi(red, nir)
        print(f"  NDVI calculated: shape={ndvi.shape}, range=[{ndvi.min():.3f}, {ndvi.max():.3f}]")
        
        normalized = BandCalculations.normalize_band(ndvi)
        print(f"  Normalized NDVI: range=[{normalized.min():.3f}, {normalized.max():.3f}]")
        
    except ImportError:
        print("\n  NumPy not available - skipping band calculations")


if __name__ == "__main__":
    demo_usage()
