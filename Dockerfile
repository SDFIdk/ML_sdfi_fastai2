# PyTorch 2.6+ required by transformers (CVE-2025-32434)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Install system dependencies (libgl1 for opencv/cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libspatialindex-dev git build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


#dynamically queries the version of the system library (libgdal) and forces pip to install the exact matching Python bindings version.
RUN export GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir "gdal==$GDAL_VERSION.*"
# Updated fastai to 2.7.17 and mmcv 2.1.0 for mmsegmentation 1.2.2 compatibility
# Install setuptools first so mmcv build (if from source) has pkg_resources
# fastai pins torch<2.5; reinstall torch 2.6 afterward (transformers CVE-2025-32434 requires torch>=2.6)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastai==2.7.17 transformers safetensors openmim ftfy && \
    pip install --no-cache-dir "torch>=2.6" "torchvision" --upgrade --force-reinstall && \
    pip install --no-cache-dir mmengine && \
    pip install --no-cache-dir --no-build-isolation mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu126/torch2.6/index.html && \
    pip install --no-cache-dir mmsegmentation==1.2.2

WORKDIR /workspace

# Copy requirements and install (--no-build-isolation for git deps that need pkg_resources)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy and install the project package
COPY . .
RUN pip install --no-cache-dir -e .

# Ensure torch 2.6.x after all installs (transformers CVE-2025-32434; avoid 2.10 for fastai compatibility)
RUN pip install --no-cache-dir "torch>=2.6,<2.7" "torchvision" --upgrade --force-reinstall

# Patch fastai _FakeLoader for PyTorch 2.6+ (DataLoader expects loader.in_order)
COPY patch_fastai_fakeloader.py /tmp/
RUN python /tmp/patch_fastai_fakeloader.py

CMD ["/bin/bash"]
