"""
Real-time training monitoring dashboard for Cabruca segmentation.
Provides live metrics visualization and system monitoring.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import GPUtil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
import yaml
from plotly.subplots import make_subplots


class TrainingMonitor:
    """
    Interactive dashboard for monitoring training progress.
    """

    def __init__(self, experiment_dir: str):
        """
        Initialize training monitor.

        Args:
            experiment_dir: Directory containing training outputs
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.history_file = self.experiment_dir / "training_history.json"

        # Load configuration
        self.config = self._load_config()

        # Initialize state
        if "last_update" not in st.session_state:
            st.session_state.last_update = time.time()

    def _load_config(self) -> Dict:
        """Load training configuration."""
        config_files = (
            list(self.experiment_dir.glob("*.yaml"))
            + list(self.experiment_dir.glob("*.yml"))
            + list(self.experiment_dir.glob("*.json"))
        )

        if config_files:
            with open(config_files[0], "r") as f:
                if config_files[0].suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        return {}

    def _load_history(self) -> Dict:
        """Load training history."""
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                return json.load(f)
        return {}

    def _get_latest_checkpoint(self) -> Optional[Dict]:
        """Get information about the latest checkpoint."""
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

        # Load checkpoint info
        import torch

        checkpoint = torch.load(latest, map_location="cpu")

        return {
            "path": str(latest),
            "epoch": checkpoint.get("epoch", 0),
            "best_metric": checkpoint.get("best_metric", 0),
            "best_epoch": checkpoint.get("best_epoch", 0),
            "size_mb": latest.stat().st_size / (1024 * 1024),
        }

    def _get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }

        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update(
                    {
                        "gpu_name": gpu.name,
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal,
                        "gpu_utilization": gpu.load * 100,
                        "gpu_temperature": gpu.temperature,
                    }
                )
        except:
            pass

        # MPS metrics for macOS
        if os.uname().sysname == "Darwin":
            try:
                # Check if MPS is being used
                result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                if "metal" in result.stdout.lower():
                    metrics["mps_active"] = True
            except:
                pass

        return metrics

    def create_loss_plot(self, history: Dict) -> go.Figure:
        """Create loss curves plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Total Loss",
                "Instance Loss",
                "Semantic Loss",
                "Crown & Density Loss",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        epochs = list(range(len(history.get("train/total_loss", []))))

        # Total loss
        if "train/total_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train/total_loss"],
                    name="Train",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )
        if "val/total_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/total_loss"],
                    name="Val",
                    line=dict(color="red"),
                ),
                row=1,
                col=1,
            )

        # Instance loss
        if "train/instance_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train/instance_loss"],
                    name="Train Instance",
                    line=dict(color="blue", dash="dash"),
                ),
                row=1,
                col=2,
            )
        if "val/instance_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/instance_loss"],
                    name="Val Instance",
                    line=dict(color="red", dash="dash"),
                ),
                row=1,
                col=2,
            )

        # Semantic loss
        if "train/semantic_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train/semantic_loss"],
                    name="Train Semantic",
                    line=dict(color="green"),
                ),
                row=2,
                col=1,
            )
        if "val/semantic_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/semantic_loss"],
                    name="Val Semantic",
                    line=dict(color="orange"),
                ),
                row=2,
                col=1,
            )

        # Crown and density losses
        if "train/crown_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train/crown_loss"],
                    name="Crown",
                    line=dict(color="purple"),
                ),
                row=2,
                col=2,
            )
        if "train/density_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train/density_loss"],
                    name="Density",
                    line=dict(color="brown"),
                ),
                row=2,
                col=2,
            )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss")
        fig.update_layout(
            height=600, showlegend=True, title_text="Training Loss Curves"
        )

        return fig

    def create_metrics_plot(self, history: Dict) -> go.Figure:
        """Create metrics plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "mAP@0.5",
                "Tree Count Accuracy",
                "Canopy IoU",
                "Shade Distribution Score",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        epochs = list(range(len(history.get("val/mAP_50", []))))

        # mAP@0.5
        if "val/mAP_50" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/mAP_50"],
                    name="mAP@0.5",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

        # Tree count accuracy
        if "val/total_count_accuracy" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/total_count_accuracy"],
                    name="Count Accuracy",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )

        # Canopy IoU
        if "val/canopy_iou" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/canopy_iou"],
                    name="Canopy IoU",
                    line=dict(color="orange"),
                ),
                row=2,
                col=1,
            )

        # Shade distribution
        if "val/shade_distribution_score" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val/shade_distribution_score"],
                    name="Shade Score",
                    line=dict(color="purple"),
                ),
                row=2,
                col=2,
            )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Score", range=[0, 1])
        fig.update_layout(
            height=600, showlegend=True, title_text="Agroforestry Metrics"
        )

        return fig

    def create_learning_rate_plot(self, history: Dict) -> go.Figure:
        """Create learning rate schedule plot."""
        if "train/learning_rate" not in history:
            return go.Figure()

        epochs = list(range(len(history["train/learning_rate"])))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train/learning_rate"],
                mode="lines",
                name="Learning Rate",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            yaxis_type="log",
            height=300,
        )

        return fig

    def create_system_gauges(self, metrics: Dict) -> go.Figure:
        """Create system monitoring gauges."""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("CPU Usage", "Memory Usage", "GPU/MPS Usage"),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ],
        )

        # CPU gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get("cpu_percent", 0),
                title={"text": "CPU %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # Memory gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get("memory_percent", 0),
                title={
                    "text": f"Memory % ({metrics.get('memory_used_gb', 0):.1f}/{metrics.get('memory_total_gb', 0):.1f} GB)"
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 85], "color": "yellow"},
                        {"range": [85, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 95,
                    },
                },
            ),
            row=1,
            col=2,
        )

        # GPU/MPS gauge
        if "gpu_utilization" in metrics:
            value = metrics["gpu_utilization"]
            title = f"GPU ({metrics.get('gpu_name', 'Unknown')})"
        elif metrics.get("mps_active", False):
            value = 50  # Placeholder as MPS doesn't provide direct utilization
            title = "MPS Active"
        else:
            value = 0
            title = "No GPU"

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkorange"},
                    "steps": [
                        {"range": [0, 70], "color": "lightgray"},
                        {"range": [70, 90], "color": "yellow"},
                        {"range": [90, 100], "color": "red"},
                    ],
                },
            ),
            row=1,
            col=3,
        )

        fig.update_layout(height=250, margin=dict(t=50, b=20))
        return fig

    def create_comparison_table(self, history: Dict) -> pd.DataFrame:
        """Create comparison table of best metrics."""
        if not history:
            return pd.DataFrame()

        # Get last epoch metrics
        metrics_dict = {}
        for key in history:
            if history[key]:
                metrics_dict[key] = history[key][-1]

        # Find best values
        best_dict = {}
        for key in history:
            if history[key]:
                if "loss" in key:
                    best_dict[f"best_{key}"] = min(history[key])
                else:
                    best_dict[f"best_{key}"] = max(history[key])

        # Create dataframe
        data = {"Metric": [], "Current": [], "Best": [], "Improvement": []}

        key_metrics = [
            ("val/total_loss", "Validation Loss", True),
            ("val/mAP_50", "mAP@0.5", False),
            ("val/total_count_accuracy", "Tree Count Accuracy", False),
            ("val/canopy_iou", "Canopy IoU", False),
            ("val/shade_distribution_score", "Shade Score", False),
        ]

        for key, name, minimize in key_metrics:
            if key in metrics_dict:
                current = metrics_dict[key]
                best = best_dict.get(f"best_{key}", current)

                if minimize:
                    improvement = (1 - current / best) * 100 if best != 0 else 0
                else:
                    improvement = (current / best - 1) * 100 if best != 0 else 0

                data["Metric"].append(name)
                data["Current"].append(f"{current:.4f}")
                data["Best"].append(f"{best:.4f}")
                data["Improvement"].append(f"{improvement:+.1f}%")

        return pd.DataFrame(data)


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Cabruca Training Monitor",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌳 Cabruca Segmentation Training Monitor")

    # Sidebar
    st.sidebar.header("Configuration")

    # Select experiment
    experiments_dir = Path("outputs")
    if experiments_dir.exists():
        experiments = [d.name for d in experiments_dir.iterdir() if d.is_dir()]
        if experiments:
            selected_experiment = st.sidebar.selectbox(
                "Select Experiment", experiments, index=len(experiments) - 1
            )
            experiment_path = experiments_dir / selected_experiment
        else:
            st.error("No experiments found in outputs directory")
            return
    else:
        st.error("Outputs directory not found")
        return

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

    # Initialize monitor
    monitor = TrainingMonitor(experiment_path)

    # Load data
    history = monitor._load_history()
    checkpoint_info = monitor._get_latest_checkpoint()
    system_metrics = monitor._get_system_metrics()

    # Display tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Training Progress",
            "🎯 Metrics",
            "💻 System Monitor",
            "🔧 Configuration",
            "📝 Logs",
        ]
    )

    with tab1:
        st.header("Training Progress")

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if checkpoint_info:
                st.metric("Current Epoch", checkpoint_info["epoch"])
            else:
                st.metric("Current Epoch", "N/A")

        with col2:
            if checkpoint_info:
                st.metric("Best Epoch", checkpoint_info["best_epoch"])
            else:
                st.metric("Best Epoch", "N/A")

        with col3:
            if checkpoint_info:
                st.metric("Best Metric", f"{checkpoint_info['best_metric']:.4f}")
            else:
                st.metric("Best Metric", "N/A")

        with col4:
            if checkpoint_info:
                st.metric("Checkpoint Size", f"{checkpoint_info['size_mb']:.1f} MB")
            else:
                st.metric("Checkpoint Size", "N/A")

        # Loss curves
        if history:
            st.plotly_chart(monitor.create_loss_plot(history), use_container_width=True)

            # Learning rate
            st.plotly_chart(
                monitor.create_learning_rate_plot(history), use_container_width=True
            )
        else:
            st.info("No training history available yet")

    with tab2:
        st.header("Agroforestry Metrics")

        if history:
            # Metrics plot
            st.plotly_chart(
                monitor.create_metrics_plot(history), use_container_width=True
            )

            # Comparison table
            st.subheader("Metrics Comparison")
            comparison_df = monitor.create_comparison_table(history)
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("No metrics available yet")

    with tab3:
        st.header("System Monitor")

        # System gauges
        st.plotly_chart(
            monitor.create_system_gauges(system_metrics), use_container_width=True
        )

        # Detailed metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("System Info")
            st.json(
                {
                    "CPU Cores": psutil.cpu_count(),
                    "Total Memory (GB)": f"{system_metrics['memory_total_gb']:.1f}",
                    "Platform": os.uname().sysname,
                    "Python Version": os.sys.version.split()[0],
                }
            )

        with col2:
            st.subheader("GPU Info")
            if "gpu_name" in system_metrics:
                st.json(
                    {
                        "GPU": system_metrics["gpu_name"],
                        "Memory Used (MB)": system_metrics["gpu_memory_used"],
                        "Memory Total (MB)": system_metrics["gpu_memory_total"],
                        "Temperature (°C)": system_metrics.get(
                            "gpu_temperature", "N/A"
                        ),
                    }
                )
            elif system_metrics.get("mps_active", False):
                st.info("Apple Metal Performance Shaders (MPS) Active")
            else:
                st.info("No GPU detected")

    with tab4:
        st.header("Training Configuration")

        if monitor.config:
            st.json(monitor.config)
        else:
            st.info("No configuration file found")

        # Checkpoint management
        st.subheader("Checkpoint Management")

        if checkpoint_info:
            st.write(f"Latest checkpoint: `{checkpoint_info['path']}`")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Checkpoint"):
                    with open(checkpoint_info["path"], "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=Path(checkpoint_info["path"]).name,
                            mime="application/octet-stream",
                        )

            with col2:
                if st.button("🔄 Resume Training"):
                    st.info("Run training script with --resume flag")
                    st.code(
                        f"python train.py --config config.yaml --resume {checkpoint_info['path']}"
                    )

    with tab5:
        st.header("Training Logs")

        # TensorBoard launcher
        st.subheader("TensorBoard")
        if monitor.tensorboard_dir.exists():
            st.code(f"tensorboard --logdir {monitor.tensorboard_dir}")
            if st.button("🚀 Launch TensorBoard"):
                subprocess.Popen(
                    ["tensorboard", "--logdir", str(monitor.tensorboard_dir)]
                )
                st.success("TensorBoard launched on http://localhost:6006")

        # Recent logs
        st.subheader("Recent Training Events")

        if history:
            # Create a simple log view
            log_data = []
            epochs = len(history.get("train/total_loss", []))

            for i in range(max(0, epochs - 10), epochs):
                log_data.append(
                    {
                        "Epoch": i + 1,
                        "Train Loss": history.get("train/total_loss", [None])[i],
                        "Val Loss": history.get("val/total_loss", [None])[i],
                        "mAP@0.5": history.get("val/mAP_50", [None])[i],
                        "LR": history.get("train/learning_rate", [None])[i],
                    }
                )

            if log_data:
                df = pd.DataFrame(log_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No training logs available")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("🌳 Cabruca Segmentation v1.0")
    st.sidebar.markdown(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
