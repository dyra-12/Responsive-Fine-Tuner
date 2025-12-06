#!/usr/bin/env bash
set -euo pipefail

# Build and push Docker image to Docker Hub.
# Expects DOCKERHUB_USER and DOCKERHUB_REPO and DOCKERHUB_TAG env vars.

USER=${DOCKERHUB_USER:-}
REPO=${DOCKERHUB_REPO:-responsive-fine-tuner}
TAG=${DOCKERHUB_TAG:-latest}

if [ -z "$USER" ]; then
  echo "Please set DOCKERHUB_USER environment variable"
  exit 1
fi

IMAGE="$USER/$REPO:$TAG"

docker build -t "$IMAGE" -f deployment/Dockerfile .
echo "Built $IMAGE"

docker push "$IMAGE"
echo "Pushed $IMAGE"
