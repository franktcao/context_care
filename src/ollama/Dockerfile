FROM ollama/ollama:latest

# Customize API base url and expose default Ollama default port
ENV OLLAMA_API_BASE_URL "http://ollama:11434/api"
EXPOSE 11434

# Copy and run script to run and serve Ollama models locally
COPY start_ollama.sh .
RUN chmod +x start_ollama.sh \
    && ./start_ollama.sh
