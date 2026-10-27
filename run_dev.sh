#!/bin/bash

# Simple development script to run the Argus API locally with sensible defaults

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Change to script directory
cd "$(dirname "$0")"

# Check .env file
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed!${NC}"
    echo -e "${YELLOW}Install: curl -sSL https://install.python-poetry.org | python3 -${NC}"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Install dependencies if needed
if [ ! -f "poetry.lock" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    poetry install
fi

# Quick start message
echo -e "${GREEN}Starting Argus API in development mode...${NC}"
echo -e "${YELLOW}URL: http://localhost:8000${NC}"
echo -e "${YELLOW}Docs: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Press CTRL+C to stop${NC}"
echo ""

# Run with auto-reload enabled for development
poetry run uvicorn argus.api.app:app --host 0.0.0.0 --port 8000 --reload
