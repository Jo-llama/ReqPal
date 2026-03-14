#!/bin/bash
# ReqPal startup script for Lightning AI
set -e

echo "=== ReqPal startup ==="

# Kill any existing processes on ports 8000/8001
echo "=== Cleaning up existing processes ==="
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "python server.py" 2>/dev/null || true
sleep 2

# Upgrade pip + setuptools first — required for Python 3.12 compatibility
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Create storage dirs if needed
mkdir -p storage uploads

# Start Qwen LitServe server in background (port 8000)
echo "=== Starting Qwen LitServe server (port 8000) ==="
python server.py &
LITSERVE_PID=$!
echo "LitServe PID: $LITSERVE_PID"

# Give LitServe a moment to bind the port before FastAPI starts
sleep 3

# Start ReqPal FastAPI app (port 8001)
echo "=== Starting ReqPal FastAPI app (port 8001) ==="
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
