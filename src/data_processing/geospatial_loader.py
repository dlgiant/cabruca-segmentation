"""
Geospatial Data Integration Pipeline
Handles loading GeoJSON files, AOI extraction, satellite imagery interfaces,
coordinate conversions, and multi-band imagery processing.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import transform
import geopandas as gpd
from pyproj import Transformer
import requests
from datetime import datetime, timedelta


@dataclass
class AOIBounds:
    """Area of Interest boundary representation"""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    geometry: Polygon
    properties: Dict[str, Any]
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return bounds as (min_lon, min_lat, max_lon, max_lat)"""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center point of AOI"""
        return ((self.min_lon + self.max_lon) / 2, 
                (self.min_lat + self.max_lat) / 2)
    
    @property
    def area_m2(self) -> float:
        """Calculate area in square meters"""
        # Transform to UTM for accurate area calculation
        utm_crs = self._get_utm_crs()
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        utm_geometry = transform(transformer.transform, self.geometry)
        return utm_geometry.area
    
    def _get_utm_crs(self) -> str:
        """Get appropriate UTM CRS for the AOI location"""
        lon, lat = self.center
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        epsg_code = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        return f"EPSG:{epsg_code}"


class GeoJSONLoader:
    """Load and parse GeoJSON files for AOI extraction"""
    
    def __init__(self, geojson_path: Union[str, Path]):
        """
        Initialize GeoJSON loader
        
        Args:
            geojson_path: Path to GeoJSON file
        """
        self.path = Path(geojson_path)
        if not self.path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
        with open(self.path, 'r') as f:
            self.data = json.load(f)
        
        self.gdf = gpd.read_file(str(self.path))
    
    def extract_aoi_bounds(self, feature_index: int = 0) -> AOIBounds:
        """
        Extract AOI boundaries from GeoJSON feature
        
        Args:
            feature_index: Index of feature to extract (default: 0)
        
        Returns:
            AOIBounds object containing boundary information
        """
        if feature_index >= len(self.gdf):
            raise IndexError(f"Feature index {feature_index} out of range")
        
        feature = self.gdf.iloc[feature_index]
        geometry = feature.geometry
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        
        return AOIBounds(
            min_lon=bounds[0],
            min_lat=bounds[1],
            max_lon=bounds[2],
            max_lat=bounds[3],
            geometry=geometry,
            properties=feature.to_dict()
        )
    
    def get_all_aois(self) -> List[AOIBounds]:
        """Extract all AOIs from the GeoJSON file"""
        aois = []
        for idx in range(len(self.gdf)):
            aois.append(self.extract_aoi_bounds(idx))
        return aois
    
    def to_planet_aoi(self, feature_index: int = 0) -> Dict:
        """
        Convert GeoJSON feature to Planet Labs AOI format
        
        Args:
            feature_index: Index of feature to convert
        
        Returns:
            Dictionary in Planet Labs AOI format
        """
        aoi = self.extract_aoi_bounds(feature_index)
        
        return {
            "type": "Polygon",
            "coordinates": list(aoi.geometry.exterior.coords)
        }


class CoordinateConverter:
    """Handle conversions between geographic and pixel coordinates"""
    
    def __init__(self, raster_transform: rasterio.transform.Affine, 
                 crs: Union[str, CRS] = "EPSG:4326"):
        """
        Initialize coordinate converter
        
        Args:
            raster_transform: Affine transform from rasterio
            crs: Coordinate reference system
        """
        self.transform = raster_transform
        self.crs = CRS.from_string(crs) if isinstance(crs, str) else crs
    
    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates
        
        Args:
            lon: Longitude
            lat: Latitude
        
        Returns:
            Tuple of (column, row) pixel coordinates
        """
        col, row = ~self.transform * (lon, lat)
        return int(col), int(row)
    
    def pixel_to_geo(self, col: int, row: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates
        
        Args:
            col: Column (x) coordinate
            row: Row (y) coordinate
        
        Returns:
            Tuple of (longitude, latitude)
        """
        lon, lat = self.transform * (col, row)
        return lon, lat
    
    def bounds_to_pixel_bbox(self, bounds: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """
        Convert geographic bounds to pixel bounding box
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
        
        Returns:
            Tuple of (min_col, min_row, max_col, max_row)
        """
        min_col, max_row = self.geo_to_pixel(bounds[0], bounds[1])
        max_col, min_row = self.geo_to_pixel(bounds[2], bounds[3])
        return (min_col, min_row, max_col, max_row)


class SatelliteImageryLoader:
    """Load and process satellite imagery from various sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize satellite imagery loader
        
        Args:
            api_key: API key for Planet Labs (optional, can be set via environment)
        """
        self.api_key = api_key or os.environ.get('PLANET_API_KEY')
        self.planet_base_url = "https://api.planet.com/data/v1"
    
    def load_local_imagery(self, image_path: Union[str, Path], 
                          aoi: Optional[AOIBounds] = None) -> Dict[str, np.ndarray]:
        """
        Load multi-band satellite imagery from local file
        
        Args:
            image_path: Path to raster file
            aoi: Optional AOI to crop the imagery
        
        Returns:
            Dictionary with band names as keys and arrays as values
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        bands = {}
        
        with rasterio.open(image_path) as src:
            # Store metadata
            bands['metadata'] = {
                'crs': src.crs.to_string(),
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'bounds': src.bounds
            }
            
            # Handle cropping if AOI is provided
            if aoi:
                # Ensure AOI is in same CRS as raster
                if src.crs != CRS.from_string("EPSG:4326"):
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    aoi_geometry = transform(transformer.transform, aoi.geometry)
                else:
                    aoi_geometry = aoi.geometry
                
                # Crop raster to AOI
                out_image, out_transform = mask(src, [aoi_geometry], crop=True)
                
                bands['metadata']['transform'] = out_transform
                bands['metadata']['width'] = out_image.shape[2]
                bands['metadata']['height'] = out_image.shape[1]
                
                # Extract individual bands
                band_names = self._get_band_names(src)
                for i, band_name in enumerate(band_names):
                    bands[band_name] = out_image[i]
            else:
                # Load full image
                band_names = self._get_band_names(src)
                for i, band_name in enumerate(band_names):
                    bands[band_name] = src.read(i + 1)
        
        return bands
    
    def _get_band_names(self, src: rasterio.DatasetReader) -> List[str]:
        """
        Get band names based on number of bands and descriptions
        
        Args:
            src: Rasterio dataset reader
        
        Returns:
            List of band names
        """
        # Try to get band descriptions
        descriptions = src.descriptions
        if descriptions and all(d for d in descriptions):
            return list(descriptions)
        
        # Default band naming based on common satellite imagery
        count = src.count
        if count == 1:
            return ['gray']
        elif count == 3:
            return ['red', 'green', 'blue']
        elif count == 4:
            return ['red', 'green', 'blue', 'nir']
        elif count == 5:
            return ['blue', 'green', 'red', 'rededge', 'nir']
        elif count == 8:
            # Common for Planet 8-band imagery
            return ['coastal_blue', 'blue', 'green_i', 'green', 
                   'yellow', 'red', 'rededge', 'nir']
        else:
            return [f'band_{i+1}' for i in range(count)]
    
    def search_planet_imagery(self, aoi: AOIBounds, 
                            start_date: datetime,
                            end_date: datetime,
                            item_types: List[str] = ["PSScene"],
                            cloud_cover_max: float = 0.1) -> List[Dict]:
        """
        Search Planet Labs API for available imagery
        
        Args:
            aoi: Area of Interest bounds
            start_date: Start date for search
            end_date: End date for search
            item_types: List of Planet item types to search
            cloud_cover_max: Maximum cloud cover (0-1)
        
        Returns:
            List of available imagery items
        """
        if not self.api_key:
            raise ValueError("Planet API key not provided")
        
        # Create search filter
        geometry_filter = {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": mapping(aoi.geometry)
        }
        
        date_filter = {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {
                "gte": start_date.isoformat() + "Z",
                "lte": end_date.isoformat() + "Z"
            }
        }
        
        cloud_filter = {
            "type": "RangeFilter",
            "field_name": "cloud_cover",
            "config": {
                "lte": cloud_cover_max
            }
        }
        
        combined_filter = {
            "type": "AndFilter",
            "config": [geometry_filter, date_filter, cloud_filter]
        }
        
        search_request = {
            "item_types": item_types,
            "filter": combined_filter
        }
        
        # Make API request
        search_url = f"{self.planet_base_url}/quick-search"
        headers = {"Authorization": f"api-key {self.api_key}"}
        
        response = requests.post(search_url, json=search_request, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Planet API error: {response.status_code} - {response.text}")
        
        return response.json().get("features", [])
    
    def download_planet_image(self, item_id: str, asset_type: str = "ortho_analytic_4b",
                            output_path: Union[str, Path] = None) -> Path:
        """
        Download Planet Labs imagery
        
        Args:
            item_id: Planet item ID
            asset_type: Type of asset to download
            output_path: Where to save the downloaded image
        
        Returns:
            Path to downloaded image
        """
        if not self.api_key:
            raise ValueError("Planet API key not provided")
        
        headers = {"Authorization": f"api-key {self.api_key}"}
        
        # Get item assets
        item_url = f"{self.planet_base_url}/item-types/PSScene/items/{item_id}/assets"
        response = requests.get(item_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get item assets: {response.status_code}")
        
        assets = response.json()
        
        if asset_type not in assets:
            available = list(assets.keys())
            raise ValueError(f"Asset type {asset_type} not available. Available: {available}")
        
        # Activate asset
        activation_url = assets[asset_type]["_links"]["activate"]
        requests.post(activation_url, headers=headers)
        
        # Wait for activation and get download URL
        download_url = None
        for _ in range(30):  # Wait up to 5 minutes
            response = requests.get(item_url, headers=headers)
            assets = response.json()
            
            if assets[asset_type]["status"] == "active":
                download_url = assets[asset_type]["location"]
                break
            
            import time
            time.sleep(10)
        
        if not download_url:
            raise Exception("Asset activation timeout")
        
        # Download the image
        if output_path is None:
            output_path = Path(f"planet_{item_id}_{asset_type}.tif")
        else:
            output_path = Path(output_path)
        
        response = requests.get(download_url, stream=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path


class MultiBandProcessor:
    """Process and manipulate multi-band satellite imagery"""
    
    @staticmethod
    def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index
        
        Args:
            red: Red band array
            nir: Near-infrared band array
        
        Returns:
            NDVI array
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-10)
        return ndvi
    
    @staticmethod
    def calculate_evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index
        
        Args:
            red: Red band array
            nir: Near-infrared band array
            blue: Blue band array
        
        Returns:
            EVI array
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            evi = 2.5 * ((nir.astype(float) - red.astype(float)) / 
                        (nir + 6 * red - 7.5 * blue + 1))
        return evi
    
    @staticmethod
    def stack_bands(bands: Dict[str, np.ndarray], 
                   band_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Stack multiple bands into a single array
        
        Args:
            bands: Dictionary of band arrays
            band_order: Optional specific order for bands
        
        Returns:
            Stacked array of shape (height, width, n_bands)
        """
        # Filter out metadata
        band_arrays = {k: v for k, v in bands.items() 
                      if isinstance(v, np.ndarray) and k != 'metadata'}
        
        if band_order:
            arrays = [band_arrays[name] for name in band_order if name in band_arrays]
        else:
            arrays = list(band_arrays.values())
        
        return np.stack(arrays, axis=-1)
    
    @staticmethod
    def normalize_bands(bands: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize band values
        
        Args:
            bands: Band array
            method: Normalization method ('minmax', 'standardize', 'percentile')
        
        Returns:
            Normalized array
        """
        if method == 'minmax':
            min_val = np.min(bands, axis=(0, 1), keepdims=True)
            max_val = np.max(bands, axis=(0, 1), keepdims=True)
            return (bands - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'standardize':
            mean = np.mean(bands, axis=(0, 1), keepdims=True)
            std = np.std(bands, axis=(0, 1), keepdims=True)
            return (bands - mean) / (std + 1e-10)
        
        elif method == 'percentile':
            # Normalize to 2-98 percentile range
            p2 = np.percentile(bands, 2, axis=(0, 1), keepdims=True)
            p98 = np.percentile(bands, 98, axis=(0, 1), keepdims=True)
            bands_clipped = np.clip(bands, p2, p98)
            return (bands_clipped - p2) / (p98 - p2 + 1e-10)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def resample_bands(bands: Dict[str, np.ndarray], 
                      target_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Resample bands to target shape
        
        Args:
            bands: Dictionary of band arrays
            target_shape: Target (height, width)
        
        Returns:
            Resampled bands dictionary
        """
        from scipy import ndimage
        
        resampled = {}
        for name, band in bands.items():
            if isinstance(band, np.ndarray) and name != 'metadata':
                zoom_factors = (target_shape[0] / band.shape[0],
                              target_shape[1] / band.shape[1])
                resampled[name] = ndimage.zoom(band, zoom_factors, order=1)
            elif name == 'metadata':
                resampled[name] = band.copy()
                resampled[name]['height'] = target_shape[0]
                resampled[name]['width'] = target_shape[1]
        
        return resampled


class GeospatialDataPipeline:
    """Main pipeline for geospatial data integration"""
    
    def __init__(self, planet_api_key: Optional[str] = None):
        """
        Initialize the geospatial data pipeline
        
        Args:
            planet_api_key: Optional Planet Labs API key
        """
        self.planet_api_key = planet_api_key
        self.imagery_loader = SatelliteImageryLoader(planet_api_key)
        self.band_processor = MultiBandProcessor()
    
    def load_aoi_from_geojson(self, geojson_path: Union[str, Path]) -> List[AOIBounds]:
        """
        Load AOIs from GeoJSON file
        
        Args:
            geojson_path: Path to GeoJSON file
        
        Returns:
            List of AOI bounds
        """
        loader = GeoJSONLoader(geojson_path)
        return loader.get_all_aois()
    
    def process_local_imagery_for_aoi(self, 
                                     image_path: Union[str, Path],
                                     aoi: AOIBounds,
                                     calculate_indices: bool = True) -> Dict:
        """
        Process local satellite imagery for a specific AOI
        
        Args:
            image_path: Path to satellite imagery
            aoi: Area of Interest
            calculate_indices: Whether to calculate vegetation indices
        
        Returns:
            Dictionary containing processed bands and indices
        """
        # Load imagery cropped to AOI
        bands = self.imagery_loader.load_local_imagery(image_path, aoi)
        
        # Calculate vegetation indices if requested
        if calculate_indices:
            if 'red' in bands and 'nir' in bands:
                bands['ndvi'] = self.band_processor.calculate_ndvi(
                    bands['red'], bands['nir']
                )
            
            if 'red' in bands and 'nir' in bands and 'blue' in bands:
                bands['evi'] = self.band_processor.calculate_evi(
                    bands['red'], bands['nir'], bands['blue']
                )
        
        # Create coordinate converter
        if 'metadata' in bands:
            bands['coord_converter'] = CoordinateConverter(
                bands['metadata']['transform'],
                bands['metadata'].get('crs', 'EPSG:4326')
            )
        
        return bands
    
    def prepare_training_data(self, 
                             bands: Dict[str, np.ndarray],
                             normalize: bool = True,
                             stack: bool = True,
                             band_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare multi-band imagery for model training
        
        Args:
            bands: Dictionary of band arrays
            normalize: Whether to normalize the bands
            stack: Whether to stack bands into single array
            band_order: Specific order for band stacking
        
        Returns:
            Processed array ready for training
        """
        if stack:
            data = self.band_processor.stack_bands(bands, band_order)
        else:
            # Return as separate bands
            data = {k: v for k, v in bands.items() 
                   if isinstance(v, np.ndarray) and k != 'metadata'}
        
        if normalize and stack:
            data = self.band_processor.normalize_bands(data, method='percentile')
        elif normalize and not stack:
            data = {k: self.band_processor.normalize_bands(v[..., np.newaxis], method='percentile')[..., 0]
                   for k, v in data.items()}
        
        return data


def main():
    """Example usage of the geospatial data pipeline"""
    
    # Initialize pipeline
    pipeline = GeospatialDataPipeline()
    
    # Load AOIs from GeoJSON files
    print("Loading AOIs from GeoJSON files...")
    farm1_aois = pipeline.load_aoi_from_geojson("farm1.geojson")
    camacan_aois = pipeline.load_aoi_from_geojson("camacan-geo.geojson")
    
    print(f"Found {len(farm1_aois)} AOI(s) in farm1.geojson")
    print(f"Found {len(camacan_aois)} AOI(s) in camacan-geo.geojson")
    
    # Display AOI information
    for i, aoi in enumerate(farm1_aois):
        print(f"\nFarm1 AOI {i}:")
        print(f"  Name: {aoi.properties.get('name', 'N/A')}")
        print(f"  Bounds: {aoi.bounds}")
        print(f"  Center: {aoi.center}")
        print(f"  Area: {aoi.area_m2:.2f} m²")
    
    for i, aoi in enumerate(camacan_aois):
        print(f"\nCamacan AOI {i}:")
        print(f"  Name: {aoi.properties.get('name', 'N/A')}")
        print(f"  Bounds: {aoi.bounds}")
        print(f"  Center: {aoi.center}")
        print(f"  Area: {aoi.area_m2:.2f} m²")
    
    # Example: Process local imagery if available
    # bands = pipeline.process_local_imagery_for_aoi(
    #     "path/to/satellite_image.tif",
    #     farm1_aois[0],
    #     calculate_indices=True
    # )
    
    # Example: Search Planet imagery (requires API key)
    # if pipeline.planet_api_key:
    #     from datetime import datetime, timedelta
    #     results = pipeline.imagery_loader.search_planet_imagery(
    #         aoi=farm1_aois[0],
    #         start_date=datetime.now() - timedelta(days=30),
    #         end_date=datetime.now(),
    #         cloud_cover_max=0.1
    #     )
    #     print(f"Found {len(results)} Planet imagery items")


if __name__ == "__main__":
    main()
