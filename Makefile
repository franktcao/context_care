.ONESHELL: # Runs all commands in single shell
.PHONY: setup # Make setup (or other commands run on default)
SHELL := /bin/bash # Use bash as shell instead of sh
VENV = .venv # Virtual environment (relative) directory

setup: pyproject.toml
	@echo "=== Initializing git"
	git init
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
	@echo "=== Installing pre-commit hooks ==="
	pre-commit install
	@echo "=== Setup complete ==="
	@echo "To use the environment, run `source .venv/bin/activate`"
	#@echo "=== Adding kernel for jupyter/lab"
	## $(PYTHON) -m ipykernel install --user --name=contraction

setup_dev_env: Dockerfile docker-compose.yml
	@echo "=== Building development environment image and run container in detached mode"
	docker compose up --build -d dev

setup_deploy_env: Dockerfile docker-compose.yml
	@echo "=== Building deploy environment image"
	docker compose build deploy --build-arg ENV=prod
	@echo "=== Running deploy environment container"
	docker compose run --rm -dit deploy

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	docker system prune
