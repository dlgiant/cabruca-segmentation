#!/usr/bin/env python
"""
Launch training monitor dashboard.
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
        print("   pip install streamlit")
        sys.exit(1)

    # Launch monitor
    os.system("streamlit run src/training/training_monitor.py")
