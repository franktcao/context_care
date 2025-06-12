#!/bin/bash

ollama serve &

# Wait for Ollama to start
sleep 5

ollama pull deepseek-r1:latest
