#!/bin/bash
# ReqPal startup script for Lightning AI
set -e

echo "=== ReqPal startup ==="

# Install dependencies
pip install -r requirements.txt --quiet

# Create storage dirs if needed
mkdir -p storage uploads

# Pre-cache Qwen model (optional, speeds up first request)
# The model also loads automatically on first API call
echo "=== Starting ReqPal server ==="
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
