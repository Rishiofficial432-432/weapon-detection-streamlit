#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}🛑 Stopping Weapon Detection System...${NC}"
echo "---"

# Kill Streamlit
if pgrep -f "streamlit run app_thermal.py" > /dev/null; then
    echo -e "${YELLOW}Killing Streamlit...${NC}"
    pkill -f "streamlit run app_thermal.py"
    echo -e "${GREEN}✅ Streamlit stopped.${NC}"
else
    echo "⚠️ Streamlit not running."
fi

# Kill Backend
if pgrep -f "python backend.py" > /dev/null; then
    echo -e "${YELLOW}Killing Backend...${NC}"
    pkill -f "python2 backend.py"
    echo -e "${GREEN}✅ Backend stopped.${NC}"
else
    echo "⚠️ Backend not running."
fi

echo "---"
echo -e "${GREEN}✅ All services stopped.${NC}"
