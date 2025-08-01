#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script located at: $SCRIPT_DIR"

# Change to the script's directory
cd "$SCRIPT_DIR" || exit 1
echo "Changed to directory: $(pwd)"

# Check if graphrag has been initialized (by checking for settings.yaml and .env files)
if [ ! -f "./data/settings.yaml" ] || [ ! -f "./data/.env" ]; then
    echo "GraphRAG not initialized. Running graphrag init..."
    graphrag init --root ./data
else
    echo "GraphRAG already initialized. Skipping initialization step."
fi

# Clean cache and output directories
# If you do not do this, sometimes the indexing can fail with weird column errors
rm -rf data/cache/* data/output/*

# Run the indexing
graphrag index --root ./data
