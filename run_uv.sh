#!/bin/bash

# Script to run the Argus API locally using uv

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    print_message $RED "Error: .env file not found!"
    print_message $YELLOW "Please create a .env file with the necessary environment variables."
    exit 1
fi

# Load environment variables from .env
print_message $GREEN "Loading environment variables from .env..."
export $(grep -v '^#' .env | xargs)

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_message $RED "Error: uv is not installed!"
    print_message $YELLOW "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    print_message $YELLOW "Or: pip install uv"
    exit 1
fi

# Check Python version
print_message $GREEN "Checking Python version..."
PYTHON_VERSION=$(uv python list | grep -E "3\.1[1-9]" | head -1 | awk '{print $1}')
if [ -z "$PYTHON_VERSION" ]; then
    print_message $YELLOW "Python 3.11+ not found. Installing Python 3.11..."
    uv python install 3.11
    PYTHON_VERSION="3.11"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_message $GREEN "Creating virtual environment with Python $PYTHON_VERSION..."
    uv venv --python $PYTHON_VERSION
fi

# Sync dependencies
print_message $GREEN "Syncing dependencies..."
if [ -f "pyproject.toml" ]; then
    # If there's a pyproject.toml, sync from it
    uv sync
elif [ -f "requirements.txt" ]; then
    # Fall back to requirements.txt
    uv pip sync requirements.txt
else
    print_message $YELLOW "No pyproject.toml or requirements.txt found, installing core dependencies..."
    uv pip install fastapi uvicorn python-dotenv sqlalchemy pandas pydantic
fi

# Set PYTHONPATH to include the src directory
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Default host and port
HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-1}"
RELOAD="${API_RELOAD:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --no-reload)
            RELOAD="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST        Host to bind to (default: 127.0.0.1)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --workers N        Number of worker processes (default: 1)"
            echo "  --no-reload        Disable auto-reload on code changes"
            echo "  --help             Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  API_HOST           Default host (overridden by --host)"
            echo "  API_PORT           Default port (overridden by --port)"
            echo "  API_WORKERS        Default workers (overridden by --workers)"
            echo "  API_RELOAD         Default reload setting (overridden by --no-reload)"
            exit 0
            ;;
        *)
            print_message $RED "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build reload flag
if [ "$RELOAD" = "true" ]; then
    RELOAD_FLAG="--reload"
else
    RELOAD_FLAG=""
fi

# Print startup information
print_message $GREEN "Starting Argus API with uv..."
print_message $YELLOW "Host: $HOST"
print_message $YELLOW "Port: $PORT"
print_message $YELLOW "Workers: $WORKERS"
print_message $YELLOW "Reload: $RELOAD"
print_message $YELLOW "Environment: $ENVIRONMENT"
echo ""

# Check if the app module exists
APP_PATH="src/argus/api/app.py"
if [ ! -f "$APP_PATH" ]; then
    print_message $RED "Error: API app not found at $APP_PATH"
    exit 1
fi

# Run the FastAPI application
print_message $GREEN "API will be available at http://$HOST:$PORT"
print_message $GREEN "API docs will be available at http://$HOST:$PORT/docs"
print_message $GREEN "Press CTRL+C to stop the server"
echo ""

# Run uvicorn with uv
exec uv run uvicorn argus.api.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    $RELOAD_FLAG \
    --log-level info
