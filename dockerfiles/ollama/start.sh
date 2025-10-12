#!/bin/sh
set -e

ollama serve &
SERVER_PID=$!

sleep 5

# Pull embedding model
echo "Pulling embedding model: mxbai-embed-large..."
ollama pull mxbai-embed-large || true

# Pull LLM model
echo "Pulling LLM model: llama3..."
ollama pull llama3 || true

echo "All models pulled successfully"

wait ${SERVER_PID}

