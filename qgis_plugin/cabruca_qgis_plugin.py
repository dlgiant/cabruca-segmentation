"""
QGIS Plugin for Cabruca Segmentation Model Integration.
This plugin allows users to run the Cabruca segmentation model directly from QGIS.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from qgis.core import (QgsCoordinateReferenceSystem, QgsFeature, QgsField,
                       QgsFields, QgsGeometry, QgsPointXY, QgsProcessing,
                       QgsProcessingAlgorithm, QgsProcessingContext,
                       QgsProcessingException, QgsProcessingFeedback,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination, QgsProject,
                       QgsRasterBandStats, QgsRasterLayer, QgsVectorFileWriter,
                       QgsVectorLayer, QgsWkbTypes)
# QGIS imports
from qgis.PyQt.QtCore import QCoreApplication, Qt, QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (QAction, QCheckBox, QComboBox, QDialog,
                                 QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                                 QLineEdit, QMessageBox, QProgressBar,
                                 QPushButton, QSpinBox, QTextEdit, QVBoxLayout)
from qgis.utils import iface

# Add model path to system path
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(PLUGIN_DIR), "src")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)


class CabrucaSegmentationDialog(QDialog):
    """
    Dialog for Cabruca segmentation parameters.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cabruca Segmentation Analysis")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Input section
        input_group = QGroupBox("Input Settings")
        input_layout = QVBoxLayout()

        # Raster input
        raster_layout = QHBoxLayout()
        raster_layout.addWidget(QLabel("Input Raster:"))
        self.raster_combo = QComboBox()
        self.populate_raster_layers()
        raster_layout.addWidget(self.raster_combo)
        input_layout.addLayout(raster_layout)

        # Model path
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_path = QLineEdit()
        self.model_browse_btn = QPushButton("Browse...")
        self.model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.model_browse_btn)
        input_layout.addLayout(model_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()

        # Tile size
        tile_layout = QHBoxLayout()
        tile_layout.addWidget(QLabel("Tile Size:"))
        self.tile_size = QSpinBox()
        self.tile_size.setRange(256, 2048)
        self.tile_size.setValue(512)
        self.tile_size.setSingleStep(128)
        tile_layout.addWidget(self.tile_size)
        options_layout.addLayout(tile_layout)

        # Overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Tile Overlap:"))
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 256)
        self.overlap.setValue(64)
        self.overlap.setSingleStep(16)
        overlap_layout.addWidget(self.overlap)
        options_layout.addLayout(overlap_layout)

        # Checkboxes
        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(True)
        options_layout.addWidget(self.use_gpu)

        self.export_instances = QCheckBox("Export Individual Trees")
        self.export_instances.setChecked(True)
        options_layout.addWidget(self.export_instances)

        self.export_semantic = QCheckBox("Export Land Cover")
        self.export_semantic.setChecked(True)
        options_layout.addWidget(self.export_semantic)

        self.calculate_metrics = QCheckBox("Calculate Metrics")
        self.calculate_metrics.setChecked(True)
        options_layout.addWidget(self.calculate_metrics)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Output section
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir = QLineEdit()
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(self.output_browse_btn)
        output_layout.addLayout(output_dir_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Progress section
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        layout.addWidget(self.log_output)

        # Buttons
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def populate_raster_layers(self):
        """Populate combo box with available raster layers."""
        self.raster_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.raster_combo.addItem(layer.name(), layer)

    def browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pth *.pt)"
        )
        if file_path:
            self.model_path.setText(file_path)

    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir.setText(dir_path)

    def log_message(self, message):
        """Add message to log output."""
        self.log_output.append(message)
        QCoreApplication.processEvents()

    def run_analysis(self):
        """Run the segmentation analysis."""
        # Validate inputs
        if not self.model_path.text():
            QMessageBox.warning(self, "Warning", "Please select a model file")
            return

        if not self.output_dir.text():
            QMessageBox.warning(self, "Warning", "Please select an output directory")
            return

        # Get selected layer
        layer = self.raster_combo.currentData()
        if not layer:
            QMessageBox.warning(self, "Warning", "Please select an input raster")
            return

        try:
            self.log_message("Starting Cabruca segmentation analysis...")
            self.progress_bar.setValue(10)

            # Run segmentation
            results = self.run_segmentation(layer)

            self.progress_bar.setValue(100)
            self.log_message("Analysis completed successfully!")

            # Load results into QGIS
            self.load_results_to_qgis(results)

            QMessageBox.information(
                self, "Success", "Cabruca segmentation completed successfully!"
            )
            self.accept()

        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def run_segmentation(self, layer):
        """Run the actual segmentation process."""
        import tempfile

        # Create temporary file for raster
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_raster = tmp.name

        # Export raster layer
        self.log_message("Exporting raster layer...")
        self.progress_bar.setValue(20)

        provider = layer.dataProvider()
        pipe = QgsRasterPipe()
        pipe.set(provider.clone())

        file_writer = QgsRasterFileWriter(temp_raster)
        file_writer.writeRaster(
            pipe, provider.xSize(), provider.ySize(), provider.extent(), layer.crs()
        )

        # Prepare inference command
        self.log_message("Running inference...")
        self.progress_bar.setValue(40)

        # Import inference module
        from inference.cabruca_inference import CabrucaInference

        # Initialize inference
        device = "cuda" if self.use_gpu.isChecked() else "cpu"
        config = {
            "tile_size": self.tile_size.value(),
            "overlap": self.overlap.value(),
            "batch_size": 1,
        }

        inference = CabrucaInference(
            model_path=self.model_path.text(), config=config, device=device
        )

        # Run prediction
        self.log_message("Processing tiles...")
        self.progress_bar.setValue(60)

        results = inference.predict_image(temp_raster, use_tiles=True)

        # Export results
        self.log_message("Exporting results...")
        self.progress_bar.setValue(80)

        output_dir = self.output_dir.text()
        os.makedirs(output_dir, exist_ok=True)

        # Export GeoJSON
        if self.export_instances.isChecked():
            geojson_path = os.path.join(output_dir, "trees.geojson")
            inference.export_to_geojson(results, temp_raster, geojson_path)
            results["geojson_path"] = geojson_path

        # Save metrics
        if self.calculate_metrics.isChecked():
            metrics_path = os.path.join(output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(results.get("metrics", {}), f, indent=2)
            results["metrics_path"] = metrics_path

        # Clean up temporary file
        os.remove(temp_raster)

        return results

    def load_results_to_qgis(self, results):
        """Load analysis results into QGIS."""
        self.log_message("Loading results into QGIS...")

        # Load vector layer if exists
        if "geojson_path" in results:
            vector_layer = QgsVectorLayer(
                results["geojson_path"], "Cabruca Trees", "ogr"
            )
            if vector_layer.isValid():
                QgsProject.instance().addMapLayer(vector_layer)

                # Style the layer
                self.style_tree_layer(vector_layer)

        # Display metrics
        if "metrics" in results:
            self.display_metrics(results["metrics"])

    def style_tree_layer(self, layer):
        """Apply styling to tree layer."""
        # Create categorized renderer based on tree class
        from qgis.core import (QgsCategorizedSymbolRenderer, QgsFillSymbol,
                               QgsRendererCategory, QgsSymbol)

        # Define categories
        categories = []

        # Cacao trees
        cacao_symbol = QgsFillSymbol.createSimple(
            {
                "color": "34,139,34,180",
                "outline_color": "0,100,0",
                "outline_width": "0.5",
            }
        )
        cacao_category = QgsRendererCategory("cacao_tree", cacao_symbol, "Cacao Tree")
        categories.append(cacao_category)

        # Shade trees
        shade_symbol = QgsFillSymbol.createSimple(
            {"color": "0,100,0,180", "outline_color": "0,50,0", "outline_width": "0.5"}
        )
        shade_category = QgsRendererCategory("shade_tree", shade_symbol, "Shade Tree")
        categories.append(shade_category)

        # Create renderer
        renderer = QgsCategorizedSymbolRenderer("class", categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    def display_metrics(self, metrics):
        """Display metrics in a message box."""
        metrics_text = "Cabruca System Analysis Results\n"
        metrics_text += "=" * 40 + "\n\n"

        metrics_text += "Tree Counts:\n"
        metrics_text += f"  Cacao Trees: {metrics.get('cacao_count', 0)}\n"
        metrics_text += f"  Shade Trees: {metrics.get('shade_tree_count', 0)}\n"
        metrics_text += f"  Total Trees: {metrics.get('total_trees', 0)}\n\n"

        metrics_text += "Crown Statistics:\n"
        metrics_text += (
            f"  Average Diameter: {metrics.get('avg_crown_diameter', 0):.2f} m\n"
        )
        metrics_text += (
            f"  Maximum Diameter: {metrics.get('max_crown_diameter', 0):.2f} m\n"
        )
        metrics_text += (
            f"  Minimum Diameter: {metrics.get('min_crown_diameter', 0):.2f} m\n\n"
        )

        metrics_text += f"Canopy Density: {metrics.get('canopy_density', 0):.2%}\n\n"

        metrics_text += "Land Cover Distribution:\n"
        for class_name in [
            "cacao_tree",
            "shade_tree",
            "understory",
            "bare_soil",
            "shadows",
        ]:
            coverage = metrics.get(f"{class_name}_coverage", 0)
            metrics_text += (
                f"  {class_name.replace('_', ' ').title()}: {coverage:.1%}\n"
            )

        QMessageBox.information(self, "Analysis Metrics", metrics_text)


class CabrucaSegmentationAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm for Cabruca segmentation.
    """

    INPUT_RASTER = "INPUT_RASTER"
    MODEL_PATH = "MODEL_PATH"
    OUTPUT_VECTOR = "OUTPUT_VECTOR"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    USE_GPU = "USE_GPU"

    def tr(self, string):
        """Translate string."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create instance of algorithm."""
        return CabrucaSegmentationAlgorithm()

    def name(self):
        """Algorithm name."""
        return "cabrucasegmentation"

    def displayName(self):
        """Algorithm display name."""
        return self.tr("Cabruca Segmentation Analysis")

    def group(self):
        """Algorithm group."""
        return self.tr("Cabruca Analysis")

    def groupId(self):
        """Algorithm group ID."""
        return "cabruca"

    def shortHelpString(self):
        """Short help string."""
        return self.tr(
            "Perform multi-class segmentation analysis on Cabruca "
            "agroforestry systems, detecting individual trees and "
            "calculating canopy metrics."
        )

    def initAlgorithm(self, config=None):
        """Initialize algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, self.tr("Input raster layer")
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.MODEL_PATH, self.tr("Model file path"), extension="pth"
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.TILE_SIZE,
                self.tr("Tile size"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=512,
                minValue=256,
                maxValue=2048,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.OVERLAP,
                self.tr("Tile overlap"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=64,
                minValue=0,
                maxValue=256,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_GPU, self.tr("Use GPU if available"), defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR, self.tr("Output vector layer")
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process the algorithm."""
        # Get parameters
        raster_layer = self.parameterAsRasterLayer(
            parameters, self.INPUT_RASTER, context
        )
        model_path = self.parameterAsFile(parameters, self.MODEL_PATH, context)
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        use_gpu = self.parameterAsBool(parameters, self.USE_GPU, context)
        output_path = self.parameterAsOutputLayer(
            parameters, self.OUTPUT_VECTOR, context
        )

        # Import inference module
        from inference.cabruca_inference import CabrucaInference

        # Initialize inference
        device = "cuda" if use_gpu else "cpu"
        config = {"tile_size": tile_size, "overlap": overlap, "batch_size": 1}

        feedback.pushInfo(f"Initializing model on {device}...")
        inference = CabrucaInference(model_path, config, device)

        # Create temporary raster file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_raster = tmp.name

        # Export raster
        feedback.pushInfo("Exporting raster layer...")
        provider = raster_layer.dataProvider()
        pipe = QgsRasterPipe()
        pipe.set(provider.clone())

        file_writer = QgsRasterFileWriter(temp_raster)
        file_writer.writeRaster(
            pipe,
            provider.xSize(),
            provider.ySize(),
            provider.extent(),
            raster_layer.crs(),
        )

        # Run inference
        feedback.pushInfo("Running segmentation...")
        results = inference.predict_image(temp_raster, use_tiles=True)

        # Export to GeoJSON
        feedback.pushInfo("Exporting results...")
        inference.export_to_geojson(results, temp_raster, output_path)

        # Clean up
        os.remove(temp_raster)

        return {self.OUTPUT_VECTOR: output_path}


class CabrucaQGISPlugin:
    """
    Main QGIS Plugin class for Cabruca segmentation.
    """

    def __init__(self, iface):
        """Constructor.

        Args:
            iface: Interface to QGIS application
        """
        self.iface = iface
        self.dialog = None
        self.action = None
        self.toolbar = self.iface.addToolBar("Cabruca Analysis")
        self.toolbar.setObjectName("CabrucaAnalysisToolbar")

    def initGui(self):
        """Initialize GUI."""
        # Create action
        icon_path = os.path.join(PLUGIN_DIR, "icon.png")
        if not os.path.exists(icon_path):
            # Create a simple icon if it doesn't exist
            self.create_default_icon(icon_path)

        self.action = QAction(
            QIcon(icon_path), "Cabruca Segmentation Analysis", self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.action.setStatusTip("Run Cabruca segmentation analysis")
        self.action.setWhatsThis("Multi-class segmentation for Cabruca systems")

        # Add to toolbar and menu
        self.toolbar.addAction(self.action)
        self.iface.addPluginToRasterMenu("Cabruca Analysis", self.action)

    def unload(self):
        """Remove plugin from GUI."""
        self.iface.removePluginRasterMenu("Cabruca Analysis", self.action)
        self.iface.removeToolBarIcon(self.action)
        del self.toolbar

    def run(self):
        """Run the plugin."""
        if not self.dialog:
            self.dialog = CabrucaSegmentationDialog(self.iface.mainWindow())

        self.dialog.show()
        result = self.dialog.exec_()

        if result:
            self.iface.messageBar().pushMessage(
                "Success",
                "Cabruca segmentation analysis completed",
                level=0,
                duration=3,
            )

    def create_default_icon(self, icon_path):
        """Create a default icon if none exists."""
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QBrush, QPainter, QPixmap

        # Create a simple green tree icon
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setBrush(QBrush(Qt.green))
        painter.drawEllipse(4, 4, 16, 16)
        painter.end()

        pixmap.save(icon_path)


def classFactory(iface):
    """Factory function for QGIS plugin.

    Args:
        iface: QgsInterface instance

    Returns:
        CabrucaQGISPlugin instance
    """
    return CabrucaQGISPlugin(iface)
