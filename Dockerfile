# Updated to PyTorch 2.4.1 with CUDA 12.1 
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libspatialindex-dev git build-essential \
    && rm -rf /var/lib/apt/lists/*


#dynamically queries the version of the system library (libgdal) and forces pip to install the exact matching Python bindings version.
RUN export GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir "gdal==$GDAL_VERSION.*"
# Updated fastai to 2.7.17 and mmcv 2.1.0 for mmsegmentation 1.2.2 compatibility 
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastai==2.7.17 transformers safetensors openmim ftfy && \
    pip install --no-cache-dir mmengine && \
    pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html && \
    pip install --no-cache-dir mmsegmentation==1.2.2

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install the project package
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["/bin/bash"]
