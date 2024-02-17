# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set the working directory in the container to /app
WORKDIR /app

# Copy the pyproject.toml (and poetry.lock if available) into the container
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry
# Disable virtual env creation by Poetry as the container itself is isolated
RUN poetry config virtualenvs.create false

# Install dependencies using Poetry, respecting the lock file
RUN poetry install --no-interaction --no-ansi --no-dev
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install curl and other necessary tools (if not already installed)
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Stockfish
RUN curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64.tar -o stockfish.tar \
    && tar -xf stockfish.tar -C /usr/local/bin \
    && rm stockfish.tar

# Make sure the Stockfish binary is executable
RUN chmod +x /usr/local/bin/stockfish/stockfish-ubuntu-x86-64

# Copy your source code into the container
COPY src/ /app/

# Expose port 8000 to access the FastAPI application
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "elocator.api.app:app", "--host", "0.0.0.0", "--port", "8000"]