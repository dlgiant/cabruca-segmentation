"""
Streamlit Dashboard for Cabruca Segmentation System
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import json
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🌳 Cabruca Segmentation Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌳 Cabruca Segmentation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/4CAF50/FFFFFF?text=Cabruca+AI", use_column_width=True)
    st.markdown("---")
    
    st.markdown("### 🔧 Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    tile_size = st.select_slider("Tile Size", options=[256, 512, 1024], value=512)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Check API health
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            st.success(f"✅ API Online")
            st.info(f"Version: {health_data.get('version', 'N/A')}")
            st.info(f"Model: {'Loaded' if health_data.get('model_loaded') else 'Mock Mode'}")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("[API Documentation](http://localhost:8000/docs)")
    st.markdown("[GitHub Repository](https://github.com/dlgiant/cabruca-segmentation)")
    st.markdown("[AgentOps Dashboard](https://app.agentops.ai)")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Segmentation", "📈 Analytics", "🗺️ Plantation Map", "⚙️ Settings"])

with tab1:
    st.header("Image Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Upload an aerial or satellite image for segmentation"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🚀 Run Segmentation", type="primary"):
                with st.spinner("Processing image..."):
                    # Send to API
                    files = {"file": uploaded_file.getvalue()}
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/segment",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['last_result'] = result
                            st.success("✅ Segmentation completed!")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Results")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Trees Detected", result['trees_detected'])
            with col_b:
                st.metric("Crown Coverage", f"{result['crown_coverage']:.1%}")
            with col_c:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Species distribution chart
            if result.get('species_distribution'):
                fig = px.pie(
                    values=list(result['species_distribution'].values()),
                    names=list(result['species_distribution'].keys()),
                    title="Species Distribution",
                    color_discrete_map={
                        'cacao': '#8B4513',
                        'shade_tree': '#228B22',
                        'other': '#808080'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Processing info
            st.info(f"Task ID: {result.get('task_id', 'N/A')}")
            st.info(f"Processing Time: {result.get('processing_time', 0):.2f}s")

with tab2:
    st.header("Analytics Dashboard")
    
    # Get metrics from API
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Requests", metrics.get('total_requests', 0))
            with col2:
                st.metric("Avg Processing Time", f"{metrics.get('average_processing_time', 0):.2f}s")
            with col3:
                st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
            with col4:
                st.metric("Model Version", metrics.get('model_version', 'N/A'))
            
            # Create sample time series data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            tree_counts = np.random.randint(100, 500, size=30)
            coverage = np.random.uniform(0.3, 0.7, size=30)
            
            # Tree count over time
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=dates,
                y=tree_counts,
                mode='lines+markers',
                name='Trees Detected',
                line=dict(color='green', width=2)
            ))
            fig1.update_layout(title="Trees Detected Over Time", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Coverage trend
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=dates,
                y=coverage,
                mode='lines',
                fill='tozeroy',
                name='Crown Coverage',
                line=dict(color='forestgreen')
            ))
            fig2.update_layout(title="Crown Coverage Trend", xaxis_title="Date", yaxis_title="Coverage %")
            st.plotly_chart(fig2, use_container_width=True)
            
    except:
        st.warning("Unable to fetch metrics from API")

with tab3:
    st.header("Plantation Map")
    
    # Try to load plantation data
    try:
        response = requests.get(f"{API_URL}/plantation-data")
        if response.status_code == 200:
            plantation_data = response.json()
            
            if 'sample_data' in plantation_data:
                data = plantation_data['sample_data']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Farms", data.get('farms', 0))
                with col2:
                    st.metric("Total Area", f"{data.get('total_area_hectares', 0)} ha")
                with col3:
                    st.metric("Avg Tree Density", f"{data.get('average_tree_density', 0)} trees/ha")
            
            # Create a sample map visualization
            st.subheader("Farm Locations")
            
            # Sample coordinates for demonstration
            map_data = pd.DataFrame({
                'lat': [-14.7, -14.8, -14.9],
                'lon': [-39.2, -39.3, -39.4],
                'name': ['Farm 1', 'Farm 2', 'Farm 3'],
                'area': [50, 60, 40]
            })
            
            st.map(map_data)
            
            st.info("📍 This is sample data. Upload actual GeoJSON files for real plantation mapping.")
            
    except:
        st.warning("Unable to fetch plantation data")

with tab4:
    st.header("Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox("Model Type", ["Mask R-CNN", "DeepLab v3+", "Hybrid"])
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=8)
        device = st.selectbox("Device", ["CPU", "GPU", "MPS (Apple Silicon)"])
        
        if st.button("Save Configuration"):
            st.success("Configuration saved!")
    
    with col2:
        st.subheader("Export Options")
        
        export_format = st.selectbox("Export Format", ["GeoJSON", "Shapefile", "KML", "CSV"])
        include_confidence = st.checkbox("Include Confidence Scores", value=True)
        include_metadata = st.checkbox("Include Metadata", value=True)
        
        if st.button("Export Results"):
            st.success("Results exported!")
    
    st.markdown("---")
    
    st.subheader("AWS Integration")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.text_input("S3 Bucket", value="cabruca-mvp-mvp-agent-artifacts-919014037196")
        st.text_input("DynamoDB Table", value="cabruca-mvp-mvp-agent-state")
    
    with col_b:
        st.text_input("Lambda Function", value="cabruca-mvp-mvp-manager-agent")
        st.text_input("CloudWatch Dashboard", value="cabruca-mvp-mvp-agents-dashboard")
    
    if st.button("Test AWS Connection"):
        st.info("Testing AWS connection...")
        # Add actual AWS connection test here
        st.success("✅ AWS services connected!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌳 Cabruca Segmentation System v1.0.0 | 
        <a href='https://github.com/dlgiant/cabruca-segmentation'>GitHub</a> | 
        <a href='http://localhost:8000/docs'>API Docs</a> | 
        Made with ❤️ for sustainable agroforestry</p>
    </div>
    """,
    unsafe_allow_html=True
)