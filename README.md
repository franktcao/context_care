# context_care

## Getting Started (DEV)
To setup and install dependencies, run the following command in the project root directory:

### Docker (recommended)
Set up the develop environment running

```bash
make setup_dev_env
```

This will build the development image, subsequently create and run the container in detached mode.

To attach to the container, and run
```bash
docker attach context_care_dev_inst
```
This will attach to the interactive terminal in the container


```bash
make setup
```
This will install a virtual environment to develop and/or run the app.

### Makefile

### Docker
Use the Docker image as your environment. Check to see if the the `Dockerfile` has all that is needed for development.
To develop, build the `dev` image by running:

```bash
docker compose build dev
```

Now that the image has been built, run the container with

```bash
docker compose run dev
```

to run the `dev` container
