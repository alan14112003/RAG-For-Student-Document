#!/bin/sh
set -e

ollama serve &
SERVER_PID=$!

sleep 5
ollama pull mxbai-embed-large || true

wait ${SERVER_PID}

