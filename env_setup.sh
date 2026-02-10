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