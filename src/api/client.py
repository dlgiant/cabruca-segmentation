"""
Python client for Cabruca Segmentation API.
Provides easy-to-use interface for API interactions.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


@dataclass
class TreeResult:
    """Container for tree detection result."""

    id: int
    species: str
    confidence: float
    centroid: List[float]
    crown_diameter: float
    crown_area: float


class CabrucaAPIClient:
    """
    Client for interacting with Cabruca Segmentation API.

    Example:
        client = CabrucaAPIClient("http://localhost:8000")
        result = client.process_image("plantation.jpg")
        print(f"Detected {len(result['trees'])} trees")
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """
        Check API health status.

        Returns:
            Health status dictionary
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def process_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        tile_size: int = 512,
        overlap: int = 64,
        wait_for_result: bool = True,
    ) -> Dict:
        """
        Process a single image.

        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for detections
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles
            wait_for_result: Wait for processing to complete

        Returns:
            Processing result or job information
        """
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            params = {
                "confidence_threshold": confidence_threshold,
                "tile_size": tile_size,
                "overlap": overlap,
            }

            response = self.session.post(
                f"{self.base_url}/inference", files=files, params=params
            )
            response.raise_for_status()

            result = response.json()

            if wait_for_result and result.get("status") == "processing":
                # Poll for completion
                job_id = result["job_id"]
                result = self.wait_for_job(job_id)

            return result

    def process_batch(
        self,
        image_paths: List[str],
        output_format: str = "json",
        generate_report: bool = False,
    ) -> Dict:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image file paths
            output_format: Output format (json, geojson, excel)
            generate_report: Generate analysis report

        Returns:
            Job information
        """
        payload = {
            "image_paths": image_paths,
            "output_format": output_format,
            "generate_report": generate_report,
        }

        response = self.session.post(f"{self.base_url}/batch", json=payload)
        response.raise_for_status()
        return response.json()

    def compare_with_plantation(
        self,
        image_path: str,
        plantation_data_path: Optional[str] = None,
        distance_threshold: float = 2.0,
    ) -> Dict:
        """
        Compare ML detection with plantation coordinates.

        Args:
            image_path: Path to image file
            plantation_data_path: Optional path to plantation data JSON
            distance_threshold: Maximum distance for matching

        Returns:
            Comparison results and health metrics
        """
        files = {"file": open(image_path, "rb")}

        if plantation_data_path:
            files["plantation_data"] = open(plantation_data_path, "rb")

        params = {"distance_threshold": distance_threshold}

        response = self.session.post(
            f"{self.base_url}/compare", files=files, params=params
        )

        # Close files
        for f in files.values():
            f.close()

        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get status of a processing job.

        Args:
            job_id: Job identifier

        Returns:
            Job status and results
        """
        response = self.session.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_job(self, job_id: str, timeout: int = 300) -> Dict:
        """
        Wait for job completion.

        Args:
            job_id: Job identifier
            timeout: Maximum wait time in seconds

        Returns:
            Job results
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")

            time.sleep(2)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def get_visualization(self, job_id: str, output_path: str) -> str:
        """
        Download visualization for a job.

        Args:
            job_id: Job identifier
            output_path: Path to save visualization

        Returns:
            Path to saved file
        """
        response = self.session.get(f"{self.base_url}/results/{job_id}/visualization")
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    def get_geojson(self, job_id: str, output_path: str) -> str:
        """
        Download GeoJSON for a job.

        Args:
            job_id: Job identifier
            output_path: Path to save GeoJSON

        Returns:
            Path to saved file
        """
        response = self.session.get(f"{self.base_url}/results/{job_id}/geojson")
        response.raise_for_status()

        with open(output_path, "w") as f:
            f.write(response.text)

        return output_path

    def delete_job(self, job_id: str) -> Dict:
        """
        Delete a job and its files.

        Args:
            job_id: Job identifier

        Returns:
            Deletion confirmation
        """
        response = self.session.delete(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def parse_trees(self, result: Dict) -> List[TreeResult]:
        """
        Parse tree detections from result.

        Args:
            result: API response

        Returns:
            List of TreeResult objects
        """
        trees = []
        for tree_data in result.get("trees", []):
            trees.append(
                TreeResult(
                    id=tree_data["id"],
                    species=tree_data["species"],
                    confidence=tree_data["confidence"],
                    centroid=tree_data["centroid"],
                    crown_diameter=tree_data["crown_diameter"],
                    crown_area=tree_data["crown_area"],
                )
            )
        return trees

    def trees_to_dataframe(self, trees: List[TreeResult]) -> pd.DataFrame:
        """
        Convert trees to pandas DataFrame.

        Args:
            trees: List of TreeResult objects

        Returns:
            DataFrame with tree data
        """
        data = []
        for tree in trees:
            tree_dict = asdict(tree)
            tree_dict["x"] = tree.centroid[0]
            tree_dict["y"] = tree.centroid[1]
            del tree_dict["centroid"]
            data.append(tree_dict)

        return pd.DataFrame(data)


def example_usage():
    """Example usage of the API client."""
    # Initialize client
    client = CabrucaAPIClient("http://localhost:8000")

    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")

    # Process single image
    result = client.process_image("test_image.jpg", confidence_threshold=0.6)

    print(f"Detected {len(result['trees'])} trees")
    print(f"Canopy density: {result['canopy_density']:.2%}")

    # Parse trees
    trees = client.parse_trees(result)

    # Convert to DataFrame
    df = client.trees_to_dataframe(trees)
    print("\nTree inventory:")
    print(df.head())

    # Compare with plantation data
    comparison = client.compare_with_plantation(
        "test_image.jpg", "plantation_data.json"
    )

    print(f"\nHealth Score: {comparison['health_report']['overall_score']:.2%}")
    print(f"Status: {comparison['health_report']['status']}")

    # Get recommendations
    for rec in comparison["health_report"]["recommendations"]:
        print(f"- {rec}")

    # Download visualization
    job_id = result["job_id"]
    client.get_visualization(job_id, f"viz_{job_id}.png")
    print(f"\nVisualization saved to viz_{job_id}.png")

    # Export to GeoJSON
    client.get_geojson(job_id, f"result_{job_id}.geojson")
    print(f"GeoJSON saved to result_{job_id}.geojson")


if __name__ == "__main__":
    example_usage()
