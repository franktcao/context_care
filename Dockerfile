FROM python:3.12.2

# Setup environments
ARG ENV=dev
ARG OLLAMA_HOST=0.0.0.0
ARG OLLAMA_PORT=11434
ENV ENV=${ENV} \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  POETRY_VERSION=2.1.3
ENV OLLAMA_HOST=OLLAMA_HOST
ENV OLLAMA_SERVER_URL=OLLAMA_HOST
ENV OLLAMA_PORT=OLLAMA_PORT

# Install system apps
RUN apt-get -y update \
  && apt-get -y install vim

# Define and move into working directory
WORKDIR /opt/app

# Copy over files from project root -- will be overridden in `dev` environment with volume mount
COPY . .

# Install dependencies
RUN make

EXPOSE 8501

# Run bash to interact
CMD ["streamlit", "run", "app.py"]
