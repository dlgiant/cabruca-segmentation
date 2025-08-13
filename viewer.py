#!/usr/bin/env python
"""
Launch interactive viewer for Cabruca segmentation results.
"""

import os
import sys

# Add src to path
sys.path.append("src")

if __name__ == "__main__":
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed. Install with:")
        print("   pip install streamlit plotly")
        sys.exit(1)

    print("🌳 Launching Cabruca Segmentation Viewer...")
    print("   Opening browser at http://localhost:8501")
    print("   Press Ctrl+C to stop")

    # Launch viewer
    os.system("streamlit run src/inference/interactive_viewer.py")
