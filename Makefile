.ONESHELL: # Runs all commands in single shell
.PHONY: setup # Make setup (or other commands run on default)
SHELL := /bin/bash # Use bash as shell instead of sh
VENV = .venv # Virtual environment (relative) directory

setup: pyproject.toml
	@echo "=== Installing virtual environment (does not activate it!) and package manager ==="
	python -m venv $(VENV)
	( \
       . .venv/bin/activate; \
			 pip install --upgrade pip; \
			 pip install poetry; \
	)
	@echo "=== Install dependencies ==="
	( \
       . .venv/bin/activate; \
       poetry install; \
	)

ollama:
	docker build -t local_ollama src/ollama
	docker run -d --name ollama-container --net carenet -p 11434:11434 local_ollama

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	docker system prune
