FROM python:3.13.3

# Setup environments
ARG ENV=dev
ENV ENV=${ENV} \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  POETRY_VERSION=2.1.3

# Install system apps
RUN apt-get -y update \
  && apt-get -y install vim

# Install `poetry` to manage package management
RUN pip install --upgrade pip \
  && pip install "poetry==${POETRY_VERSION}"

# Define and move into working directory
WORKDIR /opt/app

# Copy `poetry.lock` only if it exists
COPY poetry.loc[k] pyproject.toml .

# Install poetry dependencies
RUN poetry config virtualenvs.create false \
  && poetry install $(test "$ENV" == prod && echo "--no-dev") --no-interaction --no-ansi

# Copy over files from project root -- will be overridden in `dev` environment with volume mount
COPY . .

# Run bash to interact
CMD ["bash"]
