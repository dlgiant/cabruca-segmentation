"""
Interactive viewer for Cabruca segmentation results.
Streamlit-based dashboard for analysis and visualization.
"""

import base64
import json
import os
# Import inference components
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.batch_inference import (BatchInferenceEngine, InferenceResult,
                                       ReportGenerator, VisualizationTools)


class InteractiveViewer:
    """
    Interactive viewer for segmentation results.
    """

    def __init__(self):
        """Initialize the viewer."""
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Cabruca Segmentation Viewer",
            page_icon="🌳",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS
        st.markdown(
            """
        <style>
        .main {
            padding-top: 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def initialize_session_state(self):
        """Initialize session state variables."""
        if "inference_engine" not in st.session_state:
            st.session_state.inference_engine = None
        if "current_result" not in st.session_state:
            st.session_state.current_result = None
        if "all_results" not in st.session_state:
            st.session_state.all_results = []
        if "current_image" not in st.session_state:
            st.session_state.current_image = None

    def run(self):
        """Main application loop."""
        st.title("🌳 Cabruca Segmentation Analysis Dashboard")

        # Sidebar
        self.render_sidebar()

        # Main content
        if st.session_state.inference_engine and st.session_state.current_result:
            self.render_main_content()
        else:
            self.render_welcome()

    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("🔧 Configuration")

        # Model loading
        st.sidebar.subheader("Model Setup")
        model_path = st.sidebar.text_input(
            "Model Checkpoint Path",
            value="outputs/checkpoint_best.pth",
            help="Path to trained model checkpoint",
        )

        device = st.sidebar.selectbox(
            "Device", ["auto", "cpu", "cuda", "mps"], help="Computation device"
        )

        if st.sidebar.button("Load Model"):
            self.load_model(model_path, device)

        # Image upload
        st.sidebar.subheader("Input Images")
        upload_method = st.sidebar.radio(
            "Input Method", ["Upload Files", "Directory Path"]
        )

        if upload_method == "Upload Files":
            uploaded_files = st.sidebar.file_uploader(
                "Choose images",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                accept_multiple_files=True,
            )

            if uploaded_files and st.sidebar.button("Process Images"):
                self.process_uploaded_files(uploaded_files)

        else:
            dir_path = st.sidebar.text_input("Directory Path", value="data/test")

            if st.sidebar.button("Process Directory"):
                self.process_directory(dir_path)

        # Results navigation
        if st.session_state.all_results:
            st.sidebar.subheader("Results Navigation")

            result_names = [
                Path(r.image_path).name for r in st.session_state.all_results
            ]
            selected = st.sidebar.selectbox(
                "Select Image",
                range(len(result_names)),
                format_func=lambda x: result_names[x],
            )

            if st.sidebar.button("Load Result"):
                st.session_state.current_result = st.session_state.all_results[selected]
                # Load corresponding image
                self.load_current_image()

        # Export options
        st.sidebar.subheader("Export Options")

        if st.sidebar.button("📊 Generate Report"):
            self.generate_report()

        if st.sidebar.button("🗺️ Export to GeoJSON"):
            self.export_geojson()

        if st.sidebar.button("📑 Download Inventory"):
            self.download_inventory()

    def render_welcome(self):
        """Render welcome screen."""
        st.markdown(
            """
        ## Welcome to Cabruca Segmentation Analysis
        
        This interactive dashboard allows you to:
        - 🔍 Analyze segmentation results from trained models
        - 📊 Visualize tree detection and classification
        - 🗺️ Generate heatmaps and overlay visualizations
        - 📑 Export results in various formats
        - 🌳 Calculate agroforestry metrics
        
        ### Getting Started
        1. Load a trained model using the sidebar
        2. Upload images or specify a directory
        3. Process images to generate results
        4. Explore visualizations and metrics
        
        ### Features
        - **Interactive Visualizations**: Zoom, pan, and explore results
        - **Real-time Analysis**: Instant metrics calculation
        - **Multiple Export Formats**: GeoJSON, Excel, PNG
        - **Batch Processing**: Handle multiple images efficiently
        """
        )

    def render_main_content(self):
        """Render main analysis content."""
        result = st.session_state.current_result
        image = st.session_state.current_image

        if not result or image is None:
            st.warning("No results to display")
            return

        # Create tabs
        tabs = st.tabs(
            [
                "📸 Overview",
                "🌳 Tree Analysis",
                "🗺️ Heatmaps",
                "📊 Metrics",
                "🔍 Detailed View",
            ]
        )

        with tabs[0]:
            self.render_overview(image, result)

        with tabs[1]:
            self.render_tree_analysis(result)

        with tabs[2]:
            self.render_heatmaps(result)

        with tabs[3]:
            self.render_metrics(result)

        with tabs[4]:
            self.render_detailed_view(image, result)

    def render_overview(self, image: np.ndarray, result: InferenceResult):
        """Render overview tab."""
        st.header("Segmentation Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Segmentation Overlay")
            overlay = VisualizationTools.create_overlay(image, result)
            st.image(overlay, use_column_width=True)

        # Summary metrics
        st.subheader("Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trees", len(result.trees))

        with col2:
            cacao_count = sum(1 for t in result.trees if t.species == "cacao")
            st.metric("Cacao Trees", cacao_count)

        with col3:
            shade_count = sum(1 for t in result.trees if t.species == "shade")
            st.metric("Shade Trees", shade_count)

        with col4:
            st.metric("Canopy Density", f"{result.canopy_density:.1%}")

    def render_tree_analysis(self, result: InferenceResult):
        """Render tree analysis tab."""
        st.header("Individual Tree Analysis")

        if not result.trees:
            st.info("No trees detected")
            return

        # Create tree dataframe
        tree_data = []
        for tree in result.trees:
            tree_data.append(
                {
                    "ID": tree.id,
                    "Species": tree.species,
                    "Confidence": f"{tree.confidence:.2f}",
                    "Crown Diameter (m)": f"{tree.crown_diameter:.2f}",
                    "Crown Area (m²)": f"{tree.crown_area:.2f}",
                    "Center X": f"{tree.centroid[0]:.1f}",
                    "Center Y": f"{tree.centroid[1]:.1f}",
                }
            )

        df = pd.DataFrame(tree_data)

        # Display table with selection
        st.subheader("Tree Inventory")
        selected_tree = st.selectbox(
            "Select tree for details",
            df["ID"].values,
            format_func=lambda x: f"Tree {x} ({df[df['ID']==x]['Species'].values[0]})",
        )

        # Display selected tree details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df, use_container_width=True)

        with col2:
            selected_data = df[df["ID"] == selected_tree].iloc[0]
            st.subheader(f"Tree {selected_tree} Details")

            for key, value in selected_data.items():
                if key != "ID":
                    st.write(f"**{key}:** {value}")

        # Visualization
        st.subheader("Tree Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Species distribution pie chart
            species_counts = df["Species"].value_counts()
            fig_pie = px.pie(
                values=species_counts.values,
                names=species_counts.index,
                title="Species Distribution",
                color_discrete_map={"cacao": "green", "shade": "brown"},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Crown diameter histogram
            crown_diameters = [float(d.split()[0]) for d in df["Crown Diameter (m)"]]
            fig_hist = px.histogram(
                x=crown_diameters,
                nbins=20,
                title="Crown Diameter Distribution",
                labels={"x": "Crown Diameter (m)", "y": "Count"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    def render_heatmaps(self, result: InferenceResult):
        """Render heatmaps tab."""
        st.header("Heatmap Visualizations")

        heatmap_type = st.selectbox(
            "Select Heatmap Type",
            ["Tree Density", "Crown Diameter", "Species Distribution"],
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            if heatmap_type == "Tree Density":
                heatmap = VisualizationTools.create_heatmap(result, "density")
                st.image(heatmap, use_column_width=True, caption="Tree Density Heatmap")

            elif heatmap_type == "Crown Diameter":
                heatmap = VisualizationTools.create_heatmap(result, "crown")
                st.image(
                    heatmap, use_column_width=True, caption="Crown Diameter Heatmap"
                )

            elif heatmap_type == "Species Distribution":
                heatmap = VisualizationTools.create_heatmap(result, "species")
                st.image(
                    heatmap,
                    use_column_width=True,
                    caption="Species Distribution (Green: Cacao, Red: Shade)",
                )

        with col2:
            st.subheader("Heatmap Info")

            if heatmap_type == "Tree Density":
                st.write(
                    """
                **Tree Density Heatmap**
                
                Shows the concentration of trees across the image.
                - Brighter areas indicate higher tree density
                - Useful for identifying clustering patterns
                """
                )

            elif heatmap_type == "Crown Diameter":
                st.write(
                    """
                **Crown Diameter Heatmap**
                
                Visualizes the distribution of tree crown sizes.
                - Brighter areas indicate larger crowns
                - Helps identify mature vs young trees
                """
                )

            elif heatmap_type == "Species Distribution":
                st.write(
                    """
                **Species Distribution**
                
                Shows spatial distribution of different species.
                - Green: Cacao trees
                - Red: Shade trees
                - Overlap appears yellow
                """
                )

    def render_metrics(self, result: InferenceResult):
        """Render metrics tab."""
        st.header("Agroforestry Metrics")

        # Land cover distribution
        st.subheader("Land Cover Distribution")

        if result.metrics:
            land_cover = {
                k.replace("_coverage", "").replace("_", " ").title(): v
                for k, v in result.metrics.items()
                if "coverage" in k
            }

            if land_cover:
                fig_bar = px.bar(
                    x=list(land_cover.keys()),
                    y=list(land_cover.values()),
                    title="Land Cover Percentages",
                    labels={"x": "Class", "y": "Coverage (%)"},
                )
                fig_bar.update_yaxis(tickformat=".1%")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Tree Metrics")
            metrics_dict = {
                "Total Trees": len(result.trees),
                "Cacao Trees": sum(1 for t in result.trees if t.species == "cacao"),
                "Shade Trees": sum(1 for t in result.trees if t.species == "shade"),
                "Avg Confidence": (
                    f"{np.mean([t.confidence for t in result.trees]):.2f}"
                    if result.trees
                    else "N/A"
                ),
            }
            for k, v in metrics_dict.items():
                st.write(f"**{k}:** {v}")

        with col2:
            st.subheader("Crown Metrics")
            if result.trees:
                crown_diameters = [t.crown_diameter for t in result.trees]
                crown_areas = [t.crown_area for t in result.trees]

                metrics_dict = {
                    "Avg Crown Diameter": f"{np.mean(crown_diameters):.2f} m",
                    "Max Crown Diameter": f"{np.max(crown_diameters):.2f} m",
                    "Min Crown Diameter": f"{np.min(crown_diameters):.2f} m",
                    "Total Crown Area": f"{np.sum(crown_areas):.2f} m²",
                }
            else:
                metrics_dict = {"No trees detected": "N/A"}

            for k, v in metrics_dict.items():
                st.write(f"**{k}:** {v}")

        with col3:
            st.subheader("Canopy Metrics")
            metrics_dict = {
                "Canopy Density": f"{result.canopy_density:.2%}",
                "Tree Coverage": f"{result.metrics.get('cacao_tree_coverage', 0) + result.metrics.get('shade_tree_coverage', 0):.2%}",
                "Understory": f"{result.metrics.get('understory_coverage', 0):.2%}",
                "Bare Soil": f"{result.metrics.get('bare_soil_coverage', 0):.2%}",
            }
            for k, v in metrics_dict.items():
                st.write(f"**{k}:** {v}")

    def render_detailed_view(self, image: np.ndarray, result: InferenceResult):
        """Render detailed view tab."""
        st.header("Detailed Analysis View")

        # Semantic segmentation
        st.subheader("Semantic Segmentation")

        from inference.batch_inference import BatchInferenceEngine

        semantic_colored = BatchInferenceEngine.SEMANTIC_COLORS[result.semantic_map]

        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(semantic_colored.astype(np.uint8), use_column_width=True)

        with col2:
            st.write("**Legend:**")
            classes = BatchInferenceEngine.SEMANTIC_CLASSES
            colors = BatchInferenceEngine.SEMANTIC_COLORS

            for class_id, class_name in classes.items():
                color = colors[class_id]
                color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(color[0]), int(color[1]), int(color[2])
                )
                st.markdown(
                    f'<div style="display: flex; align-items: center;">'
                    f'<div style="width: 20px; height: 20px; background-color: {color_hex}; '
                    f'margin-right: 10px; border: 1px solid black;"></div>'
                    f'{class_name.replace("_", " ").title()}</div>',
                    unsafe_allow_html=True,
                )

        # Interactive tree selection
        st.subheader("Interactive Tree Selection")

        if result.trees:
            # Create scatter plot of tree positions
            tree_data = {
                "x": [t.centroid[0] for t in result.trees],
                "y": [t.centroid[1] for t in result.trees],
                "species": [t.species for t in result.trees],
                "id": [t.id for t in result.trees],
                "diameter": [t.crown_diameter for t in result.trees],
            }

            fig = px.scatter(
                tree_data,
                x="x",
                y="y",
                color="species",
                size="diameter",
                hover_data=["id", "diameter"],
                title="Tree Positions (Click to select)",
                color_discrete_map={"cacao": "green", "shade": "brown"},
            )

            # Invert y-axis to match image coordinates
            fig.update_yaxis(autorange="reversed")
            fig.update_layout(
                xaxis_title="X Position", yaxis_title="Y Position", height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    def load_model(self, model_path: str, device: str):
        """Load the segmentation model."""
        try:
            with st.spinner("Loading model..."):
                st.session_state.inference_engine = BatchInferenceEngine(
                    model_path=model_path, device=device
                )
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded image files."""
        if not st.session_state.inference_engine:
            st.error("Please load a model first")
            return

        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            results = []

            for uploaded_file in uploaded_files:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Process image
                result = st.session_state.inference_engine.process_single(tmp_path)
                result.image_path = uploaded_file.name  # Update with original name
                results.append(result)

                # Clean up
                os.unlink(tmp_path)

            st.session_state.all_results = results
            if results:
                st.session_state.current_result = results[0]
                self.load_current_image()

        st.success(f"Processed {len(results)} images successfully!")

    def process_directory(self, dir_path: str):
        """Process images from directory."""
        if not st.session_state.inference_engine:
            st.error("Please load a model first")
            return

        if not os.path.exists(dir_path):
            st.error(f"Directory not found: {dir_path}")
            return

        # Find images
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            image_paths.extend(Path(dir_path).glob(ext))

        if not image_paths:
            st.error("No images found in directory")
            return

        with st.spinner(f"Processing {len(image_paths)} images..."):
            results = st.session_state.inference_engine.process_batch(
                [str(p) for p in image_paths]
            )
            st.session_state.all_results = results
            if results:
                st.session_state.current_result = results[0]
                self.load_current_image()

        st.success(f"Processed {len(results)} images successfully!")

    def load_current_image(self):
        """Load the current image for display."""
        if st.session_state.current_result:
            image_path = st.session_state.current_result.image_path

            if os.path.exists(image_path):
                if image_path.endswith((".tif", ".tiff")):
                    import rasterio

                    with rasterio.open(image_path) as src:
                        image = src.read()
                        if image.shape[0] > 3:
                            image = image[:3]
                        image = np.transpose(image, (1, 2, 0))
                        if image.dtype != np.uint8:
                            image = (image / image.max() * 255).astype(np.uint8)
                else:
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                st.session_state.current_image = image

    def generate_report(self):
        """Generate analysis report."""
        if not st.session_state.all_results:
            st.error("No results to export")
            return

        with st.spinner("Generating report..."):
            output_dir = tempfile.mkdtemp()
            ReportGenerator.generate_analysis_report(
                st.session_state.all_results, output_dir
            )

            # Create download link for report
            report_path = Path(output_dir) / "analysis_report.png"
            if report_path.exists():
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="Download Report",
                        data=f.read(),
                        file_name="cabruca_analysis_report.png",
                        mime="image/png",
                    )

        st.success("Report generated!")

    def export_geojson(self):
        """Export current result to GeoJSON."""
        if not st.session_state.current_result:
            st.error("No result to export")
            return

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False
        ) as tmp:
            ReportGenerator.export_to_geojson(st.session_state.current_result, tmp.name)

            with open(tmp.name, "r") as f:
                geojson_data = f.read()

            st.download_button(
                label="Download GeoJSON",
                data=geojson_data,
                file_name=f"{Path(st.session_state.current_result.image_path).stem}.geojson",
                mime="application/json",
            )

            os.unlink(tmp.name)

        st.success("GeoJSON exported!")

    def download_inventory(self):
        """Download tree inventory."""
        if not st.session_state.all_results:
            st.error("No results to export")
            return

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            ReportGenerator.generate_tree_inventory(
                st.session_state.all_results, tmp.name
            )

            with open(tmp.name, "rb") as f:
                st.download_button(
                    label="Download Tree Inventory",
                    data=f.read(),
                    file_name="tree_inventory.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            os.unlink(tmp.name)

        st.success("Inventory exported!")


def main():
    """Main application entry point."""
    viewer = InteractiveViewer()
    viewer.run()


if __name__ == "__main__":
    main()
