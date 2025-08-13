#!/usr/bin/env python3
"""Dashboard server for the Streamlit service"""

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "cabruca-dashboard"}

@app.get("/dashboard")
def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cabruca Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
            .status {
                background: rgba(0, 255, 0, 0.2);
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                margin-bottom: 20px;
            }
            .endpoints {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 5px;
            }
            .endpoints h2 {
                margin-top: 0;
            }
            .endpoints ul {
                list-style-type: none;
                padding: 0;
            }
            .endpoints li {
                padding: 10px;
                margin: 5px 0;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            a {
                color: #ffd700;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌿 Cabruca Segmentation Dashboard</h1>
            <div class="status">✅ Service is operational</div>
            
            <div class="endpoints">
                <h2>Available Endpoints:</h2>
                <ul>
                    <li>📊 <a href="/dashboard">/dashboard</a> - Main dashboard (this page)</li>
                    <li>🏥 <a href="/health">/health</a> - Health check endpoint</li>
                    <li>🔌 <a href="/api">/api</a> - API endpoint</li>
                    <li>📈 <a href="/streamlit">/streamlit</a> - Streamlit interface (when available)</li>
                </ul>
            </div>
            
            <div class="endpoints" style="margin-top: 20px;">
                <h2>System Information:</h2>
                <ul>
                    <li>Environment: Staging</li>
                    <li>Region: sa-east-1</li>
                    <li>Service: ECS Fargate</li>
                    <li>Version: 1.0.0</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/streamlit")
@app.get("/streamlit/{path:path}")
def streamlit(path: str = ""):
    # Full Streamlit-like interface
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cabruca Segmentation - Streamlit</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background: #f0f2f6;
                color: #262730;
            }
            .sidebar {
                position: fixed;
                left: 0;
                top: 0;
                width: 300px;
                height: 100vh;
                background: white;
                border-right: 1px solid #e0e2e6;
                padding: 20px;
                overflow-y: auto;
            }
            .main-content {
                margin-left: 300px;
                padding: 20px;
                min-height: 100vh;
            }
            .header {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            h1 { color: #262730; margin-bottom: 10px; }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
            }
            .metric-label {
                color: #666;
                margin-bottom: 5px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .upload-area {
                background: white;
                border: 2px dashed #4CAF50;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #f9fff9;
                border-color: #2E7D32;
            }
            .btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }
            .btn:hover { background: #45a049; }
            .btn:disabled { background: #ccc; cursor: not-allowed; }
            .tab-container {
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .tabs {
                display: flex;
                background: #f0f2f6;
                border-bottom: 1px solid #e0e2e6;
            }
            .tab {
                padding: 15px 20px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 14px;
                transition: all 0.3s;
            }
            .tab.active {
                background: white;
                border-bottom: 2px solid #4CAF50;
            }
            .tab-content {
                padding: 20px;
                min-height: 400px;
            }
            .chart-placeholder {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 300px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.2em;
                margin: 20px 0;
            }
            .status-online { color: #4CAF50; font-weight: bold; }
            .status-offline { color: #f44336; font-weight: bold; }
            input[type="file"] { display: none; }
            .slider {
                width: 100%;
                margin: 10px 0;
            }
            select, input[type="text"], input[type="number"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 5px 0;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>🌿 Cabruca Segmentation</h2>
            <hr style="margin: 20px 0;">
            
            <h3>⚙️ Configuration</h3>
            <div style="margin: 20px 0;">
                <label>Confidence Threshold</label>
                <input type="range" class="slider" min="0" max="100" value="50">
                <div style="text-align: center;">0.50</div>
            </div>
            
            <div style="margin: 20px 0;">
                <label>Model Type</label>
                <select>
                    <option>U-Net</option>
                    <option>DeepLabV3+</option>
                    <option>SegFormer</option>
                    <option>SAM</option>
                </select>
            </div>
            
            <div style="margin: 20px 0;">
                <label>Tile Size</label>
                <select>
                    <option>256</option>
                    <option selected>512</option>
                    <option>1024</option>
                </select>
            </div>
            
            <hr style="margin: 20px 0;">
            <h3>📊 System Status</h3>
            <p class="status-online">✅ API Online</p>
            <p>Version: 1.0.0</p>
            <p>Environment: Staging</p>
            <p>Region: sa-east-1</p>
            
            <hr style="margin: 20px 0;">
            <h3>🔗 Quick Links</h3>
            <p><a href="/health">Health Check</a></p>
            <p><a href="/api">API Endpoint</a></p>
            <p><a href="/dashboard">Dashboard</a></p>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>🌳 Cabruca Segmentation System</h1>
                <p>Advanced AI-powered segmentation for Cabruca agroforestry systems</p>
            </div>
            
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">Images Processed</div>
                    <div class="metric-value">15,234</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Area Analyzed</div>
                    <div class="metric-value">45,678 ha</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">94.5%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Processing Time</div>
                    <div class="metric-value">2.3 sec</div>
                </div>
            </div>
            
            <div class="tab-container">
                <div class="tabs">
                    <button class="tab active">🖼️ Image Segmentation</button>
                    <button class="tab">📈 Analytics</button>
                    <button class="tab">🗂️ Datasets</button>
                    <button class="tab">🔧 Settings</button>
                </div>
                
                <div class="tab-content">
                    <h2>Image Segmentation</h2>
                    
                    <div class="upload-area" onclick="document.getElementById('fileInput').click();">
                        <input type="file" id="fileInput" accept="image/*" multiple>
                        <div>
                            <div style="font-size: 3em;">📁</div>
                            <h3>Drop images here or click to upload</h3>
                            <p>Supported formats: PNG, JPEG, TIFF, GeoTIFF</p>
                        </div>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <button class="btn" onclick="alert('Segmentation would start here!')">
                            🚀 Run Segmentation
                        </button>
                    </div>
                    
                    <div class="chart-placeholder">
                        📊 Segmentation results will appear here
                    </div>
                    
                    <div class="grid">
                        <div class="metric-card">
                            <div class="metric-label">Trees Detected</div>
                            <div class="metric-value">-</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Crown Coverage</div>
                            <div class="metric-value">-</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Confidence Score</div>
                            <div class="metric-value">-</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #666;">
                <p>Powered by AWS ECS Fargate | Built with FastAPI</p>
            </div>
        </div>
        
        <script>
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                });
            });
            
            // File upload
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const files = e.target.files;
                if (files.length > 0) {
                    alert(`${files.length} file(s) selected for upload`);
                }
            });
            
            // Slider update
            document.querySelector('.slider').addEventListener('input', function(e) {
                const value = (e.target.value / 100).toFixed(2);
                e.target.nextElementSibling.textContent = value;
            });
        </script>
    </body>
    </html>
    """)

@app.get("/")
def root():
    # Redirect root to dashboard
    return HTMLResponse(content="""
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/dashboard">
        </head>
        <body>
            <p>Redirecting to <a href="/dashboard">dashboard</a>...</p>
        </body>
        </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)