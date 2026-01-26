# Updated to PyTorch 2.4.1 with CUDA 12.1 
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libspatialindex-dev git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Updated fastai to 2.7.17 and mmcv to >=2.2.0 for PyTorch 2.4 compatibility 
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastai==2.7.17 transformers safetensors openmim && \
    mim install --no-cache-dir mmengine "mmcv>=2.2.0" && \
    pip install --no-cache-dir mmsegmentation

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
