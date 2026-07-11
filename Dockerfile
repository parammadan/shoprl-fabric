# Reproducible test/dev environment for ShopRL Fabric.
# CPU torch keeps the image lean; the unit tests are model-free (they don't need
# transformers/peft/a GPU), so this image builds and runs the whole suite.
#   docker build -t shoprl-fabric .
#   docker run --rm shoprl-fabric            # runs pytest
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

COPY . .

# Torch CPU wheels first (avoids pulling CUDA), then the package + dev tools.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e ".[dev]"

CMD ["python", "-m", "pytest", "-q"]
