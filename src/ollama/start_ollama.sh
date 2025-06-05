#!/bin/bash

ollama serve &

# Wait for Ollama to start
sleep 5

ollama run tinyllama
