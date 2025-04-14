# Stage 1: Just copy your project and build dependencies
FROM ghcr.io/astral-sh/uv:python3.11-alpine

WORKDIR /app
COPY . /app

# Optional: install uv via pipx or cargo here
RUN pip install uv

RUN uv sync

WORKDIR /app
COPY . /app

