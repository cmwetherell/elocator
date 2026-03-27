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
RUN poetry install --no-interaction --no-ansi --only main --no-root
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install curl and other necessary tools (if not already installed)
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Download Stockfish 18 and install to /usr/local/bin/stockfish
RUN curl -sL https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-avx2.tar -o /tmp/stockfish.tar \
    && cd /tmp && tar xf stockfish.tar \
    && cp /tmp/stockfish/stockfish-ubuntu-x86-64-avx2 /usr/local/bin/stockfish \
    && chmod +x /usr/local/bin/stockfish \
    && rm -rf /tmp/stockfish /tmp/stockfish.tar

# Copy your source code into the container
COPY src/ /app/

# Expose port 8000 to access the FastAPI application
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "elocator.api.app:app", "--host", "0.0.0.0", "--port", "8000"]