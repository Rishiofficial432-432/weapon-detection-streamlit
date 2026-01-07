#!/bin/bash

# Weapon Detection App Startup Script
# Runs both backend (if available) and frontend

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔫 Weapon Detection System - Starting...${NC}"
echo "---"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

echo ""
echo -e "${GREEN}✅ Environment ready!${NC}"
echo "---"

# Check if backend script exists
if [ -f "backend.py" ]; then
    echo -e "${BLUE}🚀 Starting backend...${NC}"
    python backend.py &
    BACKEND_PID=$!
    sleep 2
    echo -e "${GREEN}✅ Backend running (PID: $BACKEND_PID)${NC}"
fi

# Start frontend (Streamlit)
echo -e "${BLUE}🚀 Starting frontend (Streamlit)...${NC}"
echo "---"
streamlit run app_thermal.py

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
