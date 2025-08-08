#!/usr/bin/env python3
"""
Test script for geospatial data loaders
Tests both simple and full loader functionality
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_simple_loader():
    """Test the simple loader functionality"""
    print("\n" + "="*60)
    print("TESTING SIMPLE LOADER")
    print("="*60)
    
    from src.data_processing import SimpleGeoJSONLoader, CoordinateTransformer, ImageryMetadata
    
    # Test loading GeoJSON files
    geojson_files = [
        Path(__file__).parent / "farm1.geojson",
        Path(__file__).parent / "camacan-geo.geojson"
    ]
    
    for geojson_path in geojson_files:
        if geojson_path.exists():
            print(f"\n✓ Testing {geojson_path.name}")
            
            loader = SimpleGeoJSONLoader(geojson_path)
            aois = loader.extract_aois()
            
            print(f"  Found {len(aois)} AOI(s)")
            
            for i, aoi in enumerate(aois):
                print(f"\n  AOI {i+1}: {aoi.name}")
                print(f"    - Center: {aoi.center[0]:.4f}, {aoi.center[1]:.4f}")
                print(f"    - Area: ~{aoi.area_approx_hectares:.1f} hectares")
                print(f"    - Bounds: ({aoi.bounds[0]:.4f}, {aoi.bounds[1]:.4f}) to ({aoi.bounds[2]:.4f}, {aoi.bounds[3]:.4f})")
                
                # Test Planet format conversion
                if i == 0:
                    planet_format = loader.to_planet_format(i)
                    print(f"    - Planet format: {planet_format['type']} with {len(planet_format['coordinates'][0])} points")
        else:
            print(f"\n✗ File not found: {geojson_path}")
    
    # Test coordinate conversion
    print("\n\nTesting Coordinate Conversion:")
    test_bounds = (-39.5, -15.4, -39.498, -15.398)
    test_width, test_height = 1000, 1000
    
    # Test geo to pixel
    test_lon, test_lat = -39.499, -15.399
    px, py = CoordinateTransformer.geo_to_pixel_simple(
        test_lon, test_lat, test_bounds, test_width, test_height
    )
    print(f"  Geo ({test_lon}, {test_lat}) -> Pixel ({px}, {py})")
    
    # Test pixel to geo
    back_lon, back_lat = CoordinateTransformer.pixel_to_geo_simple(
        px, py, test_bounds, test_width, test_height
    )
    print(f"  Pixel ({px}, {py}) -> Geo ({back_lon:.6f}, {back_lat:.6f})")
    print(f"  Conversion error: {abs(back_lon - test_lon):.8f}, {abs(back_lat - test_lat):.8f}")
    
    # Test metadata
    print("\n\nTesting Imagery Metadata:")
    metadata = ImageryMetadata(
        width=test_width,
        height=test_height,
        bounds=test_bounds,
        bands=['red', 'green', 'blue', 'nir', 'rededge']
    )
    
    meta_dict = metadata.to_dict()
    print(f"  Image dimensions: {meta_dict['width']}x{meta_dict['height']}")
    print(f"  Bands: {', '.join(meta_dict['bands'])}")
    print(f"  Pixel resolution: {meta_dict['pixel_resolution_m'][0]:.2f}m x {meta_dict['pixel_resolution_m'][1]:.2f}m")
    
    print("\n✓ Simple loader tests completed successfully!")


def test_full_loader():
    """Test the full loader functionality if dependencies are available"""
    print("\n" + "="*60)
    print("TESTING FULL LOADER")
    print("="*60)
    
    try:
        from src.data_processing import (
            GeospatialDataPipeline, 
            GeoJSONLoader,
            CoordinateConverter,
            MultiBandProcessor,
            FULL_LOADER_AVAILABLE
        )
        
        if not FULL_LOADER_AVAILABLE:
            print("✗ Full loader not available - missing dependencies")
            print("  Install with: pip install -r requirements.txt")
            return
        
        print("✓ Full loader dependencies available")
        
        # Initialize pipeline
        pipeline = GeospatialDataPipeline()
        
        # Test loading AOIs
        geojson_files = [
            Path(__file__).parent / "farm1.geojson",
            Path(__file__).parent / "camacan-geo.geojson"
        ]
        
        for geojson_path in geojson_files:
            if geojson_path.exists():
                print(f"\n✓ Testing {geojson_path.name}")
                
                aois = pipeline.load_aoi_from_geojson(geojson_path)
                print(f"  Found {len(aois)} AOI(s)")
                
                for i, aoi in enumerate(aois):
                    print(f"\n  AOI {i+1}:")
                    print(f"    - Name: {aoi.properties.get('name', 'N/A')}")
                    print(f"    - Center: {aoi.center[0]:.4f}, {aoi.center[1]:.4f}")
                    print(f"    - Area: {aoi.area_m2:.2f} m²")
                    print(f"    - Bounds: {aoi.bounds}")
        
        # Test band calculations
        print("\n\nTesting Band Calculations:")
        import numpy as np
        
        # Create synthetic bands
        height, width = 100, 100
        red = np.random.rand(height, width) * 0.3
        nir = np.random.rand(height, width) * 0.7
        blue = np.random.rand(height, width) * 0.2
        
        processor = MultiBandProcessor()
        
        # Calculate NDVI
        ndvi = processor.calculate_ndvi(red, nir)
        print(f"  NDVI calculated: shape={ndvi.shape}, range=[{ndvi.min():.3f}, {ndvi.max():.3f}]")
        
        # Calculate EVI
        evi = processor.calculate_evi(red, nir, blue)
        print(f"  EVI calculated: shape={evi.shape}, range=[{evi.min():.3f}, {evi.max():.3f}]")
        
        # Test normalization
        normalized = processor.normalize_bands(ndvi, method='percentile')
        print(f"  Normalized NDVI: range=[{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # Test band stacking
        bands = {'red': red, 'green': red*0.8, 'blue': blue, 'nir': nir}
        stacked = processor.stack_bands(bands)
        print(f"  Stacked bands: shape={stacked.shape}")
        
        # Test coordinate converter with affine transform
        from rasterio.transform import from_bounds
        
        bounds = (-39.5, -15.4, -39.498, -15.398)
        transform = from_bounds(*bounds, width, height)
        
        converter = CoordinateConverter(transform)
        
        # Test conversions
        test_lon, test_lat = -39.499, -15.399
        col, row = converter.geo_to_pixel(test_lon, test_lat)
        print(f"\n  Coordinate conversion with affine transform:")
        print(f"    Geo ({test_lon}, {test_lat}) -> Pixel ({col}, {row})")
        
        back_lon, back_lat = converter.pixel_to_geo(col, row)
        print(f"    Pixel ({col}, {row}) -> Geo ({back_lon:.6f}, {back_lat:.6f})")
        
        print("\n✓ Full loader tests completed successfully!")
        
    except ImportError as e:
        print(f"✗ Could not import full loader: {e}")
        print("  Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Error testing full loader: {e}")
        import traceback
        traceback.print_exc()


def test_planet_integration():
    """Test Planet Labs API integration (requires API key)"""
    print("\n" + "="*60)
    print("TESTING PLANET LABS INTEGRATION")
    print("="*60)
    
    import os
    
    if not os.environ.get('PLANET_API_KEY'):
        print("✗ PLANET_API_KEY environment variable not set")
        print("  To test Planet integration, set your API key:")
        print("  export PLANET_API_KEY='your-api-key-here'")
        return
    
    try:
        from src.data_processing import SatelliteImageryLoader, GeospatialDataPipeline
        from datetime import datetime, timedelta
        
        print("✓ Planet API key found")
        
        # Initialize with API key
        pipeline = GeospatialDataPipeline(os.environ['PLANET_API_KEY'])
        
        # Load an AOI
        aoi_path = Path(__file__).parent / "farm1.geojson"
        if aoi_path.exists():
            aois = pipeline.load_aoi_from_geojson(aoi_path)
            if aois:
                aoi = aois[0]
                print(f"  Using AOI: {aoi.properties.get('name', 'Unknown')}")
                
                # Search for recent imagery
                print("\n  Searching for imagery...")
                results = pipeline.imagery_loader.search_planet_imagery(
                    aoi=aoi,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    cloud_cover_max=0.2
                )
                
                print(f"  Found {len(results)} imagery items")
                
                if results:
                    # Show first few results
                    for i, item in enumerate(results[:3]):
                        item_id = item['id']
                        acquired = item['properties'].get('acquired', 'N/A')
                        cloud_cover = item['properties'].get('cloud_cover', 'N/A')
                        print(f"    {i+1}. ID: {item_id[:20]}... | Date: {acquired} | Cloud: {cloud_cover:.1%}")
        
        print("\n✓ Planet integration test completed!")
        
    except Exception as e:
        print(f"✗ Error testing Planet integration: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GEOSPATIAL DATA LOADER TEST SUITE")
    print("="*60)
    
    # Always test simple loader
    test_simple_loader()
    
    # Test full loader if available
    test_full_loader()
    
    # Test Planet integration if configured
    test_planet_integration()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)
    
    # Summary
    from src.data_processing import FULL_LOADER_AVAILABLE
    
    print("\nSummary:")
    print(f"  ✓ Simple loader: Available")
    print(f"  {'✓' if FULL_LOADER_AVAILABLE else '✗'} Full loader: {'Available' if FULL_LOADER_AVAILABLE else 'Not available (install dependencies)'}")
    
    import os
    planet_available = bool(os.environ.get('PLANET_API_KEY'))
    print(f"  {'✓' if planet_available else '✗'} Planet Labs: {'API key configured' if planet_available else 'No API key'}")
    
    print("\nNext steps:")
    if not FULL_LOADER_AVAILABLE:
        print("  1. Install dependencies: pip install -r requirements.txt")
    if not planet_available:
        print("  2. Set Planet API key: export PLANET_API_KEY='your-key'")
    print("  3. Add satellite imagery files to test with real data")
    print("  4. Integrate with model training pipeline")


if __name__ == "__main__":
    main()
