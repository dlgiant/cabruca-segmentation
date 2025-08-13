#!/usr/bin/env python
"""
Main inference script for Cabruca Segmentation.
Run batch inference with comprehensive analysis and visualization.
"""

import argparse
import sys
from pathlib import Path

import agentops

# Add src to path
sys.path.append("src")

from inference.batch_inference import (
    BatchInferenceEngine,
    ReportGenerator,
    VisualizationTools,
)


def main():
    # Initialize AgentOps for inference session tracking
    agentops.init(
        auto_start_session=False, tags=["cabruca", "inference", "batch-processing"]
    )

    parser = argparse.ArgumentParser(
        description="Cabruca Segmentation Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python infer.py --model checkpoint.pth --images image.tif
  
  # Process directory of images
  python infer.py --model checkpoint.pth --images data/test/ --output results/
  
  # Generate visualizations and reports
  python infer.py --model checkpoint.pth --images data/test/ --visualize --report
  
  # Export to GeoJSON for GIS
  python infer.py --model checkpoint.pth --images *.tif --export-geojson
  
  # Use specific device
  python infer.py --model checkpoint.pth --images data/ --device mps
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Image files or directory to process",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Model configuration file (YAML)"
    )

    # Processing options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, help="Tile size for large images"
    )
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )

    # Output options
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization figures"
    )
    parser.add_argument(
        "--heatmaps", action="store_true", help="Generate density and crown heatmaps"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive analysis report"
    )
    parser.add_argument(
        "--export-geojson", action="store_true", help="Export results to GeoJSON format"
    )
    parser.add_argument(
        "--inventory", action="store_true", help="Generate tree inventory spreadsheet"
    )

    # Display options
    parser.add_argument(
        "--show", action="store_true", help="Display visualizations (requires GUI)"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Update config with command line arguments
    config["batch_size"] = args.batch_size
    config["tile_size"] = args.tile_size
    config["overlap"] = args.overlap
    config["confidence_threshold"] = args.confidence

    if not args.quiet:
        print(f"🌳 Cabruca Segmentation Inference Pipeline")
        print(f"=" * 50)
        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        print(f"Tile size: {args.tile_size}")
        print(f"Confidence threshold: {args.confidence}")

    # Initialize inference engine
    print(f"\n📦 Loading model...")
    engine = BatchInferenceEngine(
        model_path=args.model, config=config, device=args.device
    )

    # Collect image paths
    image_paths = []
    for path_str in args.images:
        path = Path(path_str)
        if path.is_dir():
            # Add all images in directory
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
                found = list(path.glob(ext))
                image_paths.extend(found)
                if not args.quiet and found:
                    print(f"  Found {len(found)} {ext} files")
        elif path.is_file():
            image_paths.append(path)
        else:
            # Try as glob pattern
            import glob

            found = glob.glob(path_str)
            if found:
                image_paths.extend([Path(p) for p in found])

    if not image_paths:
        print("❌ No images found to process")
        sys.exit(1)

    image_paths = [str(p) for p in image_paths]

    print(f"\n🔍 Processing {len(image_paths)} images...")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start AgentOps session for this inference batch
    session_name = f"cabruca_inference_{len(image_paths)}_images"
    tracer = agentops.start_trace(
        trace_name=session_name, tags=["inference", "batch-processing"]
    )

    try:
        # Process images
        results = engine.process_batch(image_paths, str(output_path))
        agentops.end_trace(tracer, end_state="Success")
    except Exception as e:
        agentops.end_trace(tracer, end_state="Fail")
        raise

    if not results:
        print("❌ No results generated")
        sys.exit(1)

    # Print summary
    if not args.quiet:
        print(f"\n📊 Processing Summary:")
        print(f"  Images processed: {len(results)}")
        total_trees = sum(len(r.trees) for r in results)
        print(f"  Total trees detected: {total_trees}")

        cacao_total = sum(
            sum(1 for t in r.trees if t.species == "cacao") for r in results
        )
        shade_total = sum(
            sum(1 for t in r.trees if t.species == "shade") for r in results
        )
        print(f"    - Cacao trees: {cacao_total}")
        print(f"    - Shade trees: {shade_total}")

        avg_density = sum(r.canopy_density for r in results) / len(results)
        print(f"  Average canopy density: {avg_density:.2%}")

    # Generate visualizations
    if args.visualize or args.heatmaps:
        print(f"\n🎨 Generating visualizations...")
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for result in results:
            image_name = Path(result.image_path).stem

            # Load original image
            image, _ = engine._load_image(result.image_path)

            if args.visualize:
                # Create comparison figure
                fig_path = vis_dir / f"{image_name}_analysis.png"
                fig = VisualizationTools.create_comparison_figure(
                    image, result, str(fig_path)
                )

                if args.show:
                    import matplotlib.pyplot as plt

                    plt.show()
                else:
                    plt.close(fig)

                if not args.quiet:
                    print(f"  Saved: {fig_path.name}")

            if args.heatmaps:
                # Generate heatmaps
                for heatmap_type in ["density", "crown", "species"]:
                    heatmap = VisualizationTools.create_heatmap(result, heatmap_type)
                    heatmap_path = vis_dir / f"{image_name}_heatmap_{heatmap_type}.png"

                    import cv2

                    cv2.imwrite(
                        str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    )

                    if not args.quiet:
                        print(f"  Saved: {heatmap_path.name}")

    # Export to GeoJSON
    if args.export_geojson:
        print(f"\n🗺️  Exporting to GeoJSON...")
        geojson_dir = output_path / "geojson"
        geojson_dir.mkdir(exist_ok=True)

        for result in results:
            image_name = Path(result.image_path).stem
            geojson_path = geojson_dir / f"{image_name}.geojson"
            ReportGenerator.export_to_geojson(result, str(geojson_path))

            if not args.quiet:
                print(f"  Exported: {geojson_path.name}")

    # Generate tree inventory
    if args.inventory:
        print(f"\n📑 Generating tree inventory...")
        inventory_path = output_path / "tree_inventory.xlsx"
        ReportGenerator.generate_tree_inventory(results, str(inventory_path))
        print(f"  Saved: {inventory_path.name}")

    # Generate comprehensive report
    if args.report:
        print(f"\n📈 Generating analysis report...")
        ReportGenerator.generate_analysis_report(results, str(output_path))
        print(f"  Report saved to: {output_path}")

    print(f"\n✅ Inference complete!")
    print(f"📂 Results saved to: {output_path}")

    # Suggest next steps
    print(f"\n💡 Next steps:")
    print(f"  - View results: open {output_path}")
    print(f"  - Interactive viewer: streamlit run viewer.py")
    print(f"  - Import to QGIS: Load GeoJSON files from {output_path / 'geojson'}")


if __name__ == "__main__":
    main()
