#!/bin/bash

# Cabruca Segmentation - Local Startup Script

echo "🌳 Starting Cabruca Segmentation System..."
echo "========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install fastapi "uvicorn[standard]" python-multipart streamlit plotly agentops
    pip install opencv-python Pillow numpy scipy pandas
else
    source venv/bin/activate
fi

# Create required directories
mkdir -p outputs api_uploads api_results

# Create dummy model file if not exists
if [ ! -f "outputs/checkpoint_best.pth" ]; then
    echo "# Dummy model file" > outputs/checkpoint_best.pth
fi

# Kill any existing processes on our ports
echo -e "${YELLOW}Checking for existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

# Start API server
echo -e "${GREEN}Starting API server on http://localhost:8000${NC}"
python simple_api.py &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit dashboard
echo -e "${GREEN}Starting Dashboard on http://localhost:8501${NC}"
streamlit run streamlit_app.py --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

echo ""
echo "========================================="
echo -e "${GREEN}✅ Cabruca Segmentation System Started!${NC}"
echo "========================================="
echo ""
echo "📍 Access Points:"
echo "   • API Server: http://localhost:8000"
echo "   • API Docs:   http://localhost:8000/docs"
echo "   • Dashboard:  http://localhost:8501"
echo ""
echo "📊 Monitoring:"
echo "   • AgentOps: https://app.agentops.ai"
echo ""
echo "To stop all services, press Ctrl+C or run:"
echo "   kill $API_PID $STREAMLIT_PID"
echo ""
echo "========================================="

# Keep script running and handle cleanup
trap "echo 'Shutting down...'; kill $API_PID $STREAMLIT_PID 2>/dev/null; exit" INT TERM

# Wait for processes
wait $API_PID $STREAMLIT_PID