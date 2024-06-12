# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set environment variables
ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry"

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

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -


ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
# Add Poetry to PATH
RUN export PATH="$HOME/.local/bin:$PATH"
# Set the working directory in the container
WORKDIR /app

COPY ./src /app/src
COPY Makefile pyproject.toml poetry.lock* /app/
