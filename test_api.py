#!/usr/bin/env python
"""
Test script for Cabruca Segmentation API.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from api.client import CabrucaAPIClient


def test_api():
    """Test API functionality."""
    print("🌳 Testing Cabruca Segmentation API")
    print("=" * 50)
    
    # Initialize client
    client = CabrucaAPIClient("http://localhost:8000")
    
    # 1. Health check
    print("\n1. Health Check")
    try:
        health = client.health_check()
        print(f"   ✅ API Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Integration available: {health['integration_available']}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return
    
    # 2. Test single image inference
    print("\n2. Single Image Inference")
    
    # Create a test image if it doesn't exist
    test_image = "test_image.jpg"
    if not Path(test_image).exists():
        print("   Creating test image...")
        import numpy as np
        import cv2
        # Create a simple test image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(test_image, img)
    
    try:
        print(f"   Processing {test_image}...")
        start_time = time.time()
        
        result = client.process_image(
            test_image,
            confidence_threshold=0.5,
            tile_size=256
        )
        
        elapsed = time.time() - start_time
        
        print(f"   ✅ Processing completed in {elapsed:.2f}s")
        print(f"   Job ID: {result['job_id']}")
        print(f"   Trees detected: {len(result['trees'])}")
        print(f"   Canopy density: {result['canopy_density']:.2%}")
        
        # Parse trees
        trees = client.parse_trees(result)
        if trees:
            print(f"\n   Tree summary:")
            species_count = {}
            for tree in trees:
                species_count[tree.species] = species_count.get(tree.species, 0) + 1
            for species, count in species_count.items():
                print(f"     - {species}: {count} trees")
        
        job_id = result['job_id']
        
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        job_id = None
    
    # 3. Test visualization download
    if job_id:
        print("\n3. Visualization Download")
        try:
            viz_path = f"test_viz_{job_id}.png"
            client.get_visualization(job_id, viz_path)
            print(f"   ✅ Visualization saved to {viz_path}")
        except Exception as e:
            print(f"   ❌ Visualization download failed: {e}")
    
    # 4. Test GeoJSON export
    if job_id:
        print("\n4. GeoJSON Export")
        try:
            geojson_path = f"test_result_{job_id}.geojson"
            client.get_geojson(job_id, geojson_path)
            print(f"   ✅ GeoJSON saved to {geojson_path}")
            
            # Validate GeoJSON
            with open(geojson_path, 'r') as f:
                geojson = json.load(f)
            print(f"   Features: {len(geojson.get('features', []))}")
            
        except Exception as e:
            print(f"   ❌ GeoJSON export failed: {e}")
    
    # 5. Test comparison with plantation data
    plantation_data_path = "plantation-data.json"
    if Path(plantation_data_path).exists():
        print("\n5. Plantation Comparison")
        try:
            comparison = client.compare_with_plantation(
                test_image,
                plantation_data_path,
                distance_threshold=2.0
            )
            
            print(f"   ✅ Comparison completed")
            print(f"   ML trees: {comparison['ml_trees_detected']}")
            print(f"   Expected trees: {comparison['plantation_trees_expected']}")
            
            stats = comparison['statistics']
            print(f"\n   Statistics:")
            print(f"     - Matched trees: {stats.get('matched_trees', 0)}")
            print(f"     - Detection F1: {stats.get('detection_f1', 0):.2%}")
            
            health = comparison['health_report']
            print(f"\n   Health Report:")
            print(f"     - Overall score: {health['overall_score']:.2%}")
            print(f"     - Status: {health['status']}")
            
            if health['recommendations']:
                print(f"\n   Recommendations:")
                for rec in health['recommendations'][:3]:
                    print(f"     • {rec}")
            
        except Exception as e:
            print(f"   ❌ Comparison failed: {e}")
    else:
        print(f"\n5. Plantation Comparison")
        print(f"   ⚠️  Skipped - {plantation_data_path} not found")
    
    # 6. Test batch processing
    print("\n6. Batch Processing")
    try:
        # Create multiple test images
        test_images = []
        for i in range(3):
            img_path = f"test_batch_{i}.jpg"
            if not Path(img_path).exists():
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(img_path, img)
            test_images.append(img_path)
        
        print(f"   Processing {len(test_images)} images...")
        
        batch_result = client.process_batch(
            test_images,
            output_format="json",
            generate_report=False
        )
        
        batch_job_id = batch_result['job_id']
        print(f"   ✅ Batch job created: {batch_job_id}")
        
        # Wait for completion
        print("   Waiting for completion...")
        final_result = client.wait_for_job(batch_job_id, timeout=60)
        print(f"   ✅ Batch processing completed")
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
    
    # 7. Test job deletion
    if job_id:
        print("\n7. Job Cleanup")
        try:
            result = client.delete_job(job_id)
            print(f"   ✅ {result['message']}")
        except Exception as e:
            print(f"   ❌ Job deletion failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")
    print("\nNext steps:")
    print("  - Check generated files (visualizations, GeoJSON)")
    print("  - Review API logs for any warnings")
    print("  - Test with real plantation images")
    print("  - Integrate with theobroma-digital project")


if __name__ == "__main__":
    test_api()