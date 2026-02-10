#!/bin/bash

echo "Installing pytorch and related libraries for using pytorch models"
uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install flow_matching
uv pip install schedulefree
uv pip install geomloss
uv pip install numpy
uv pip install einops
uv pip install timm
uv pip install transformers==4.53.2
uv pip install pytest
uv pip install flax
uv pip install augmax
uv pip install beartype
uv pip install jaxtyping==0.2.34
uv pip install cloudpickle
uv pip install "ray[default]"
uv pip install sentencepiece
uv pip install chex
uv pip install tqdm-loggable
uv pip install numpydantic
uv pip install --no-deps lerobot
uv pip install datasets
uv pip install accelerate
uv pip install av
uv pip install tyro
uv pip install ml_collections
uv pip install gcsfs