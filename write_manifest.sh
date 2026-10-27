#!/bin/bash

# Create manifest
rsconnect write-manifest fastapi ./ \
  --entrypoint src.argus.api.app:app \
  --overwrite \
  --exclude logs \
  --exclude notebooks \
  --exclude data \
  --exclude dist \
  --exclude assets \
  --exclude output \
  --exclude tests \
  --exclude results \
  --exclude sandbox \
  --exclude build \
  --exclude .dvc \
  --exclude .github \
  --exclude .ruff_cache \
  --exclude .coverage \
  --exclude .pytest_cache \
  --exclude .venv \
  --exclude .vscode \
  --exclude __pycache__ \
  --exclude docs \
  --exclude "**/*.db" \
  --exclude "**/*.pyc" \
  --exclude "**/*.egg-info/*" \
  --exclude "**/*.egg-info" \
  --exclude ".mypy_cache/**" \
  --exclude "*cache*" 

