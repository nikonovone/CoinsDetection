# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set environment variables
ENV POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# Add Poetry to PATH
ENV PATH="/root/.poetry/bin:$PATH"


COPY ./src /app


# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml and poetry.lock (if available)
COPY Makefile pyproject.toml poetry.lock* /app/

# Install the project itself
RUN make setup_ws
RUN make generate_dataset
RUN make train

# Set the entry point for the container
ENTRYPOINT ["poetry", "run"]

