# SDFI_ml

Machine learning framework developed and maintained by **KDS** for performing **semantic segmentation** on **multichannel images**.

---

## Installation

### Docker is the recomended install method


```sh
docker build -t  ml_sdfi_fastai2-dev-env:latest .

docker run --gpus all --shm-size=80g -it   -v "$(realpath ..):/projects"   -v /mnt/T/mnt:/mnt/T/mnt   -w /projects/ML_sdfi_fastai2   ml_sdfi_fastai2-dev-env:latest /bin/bash


```
Or pull a prebuilt image from Docker Hub:
```bash
sudo docker pull rasmuspjohansson/kds_cuda_pytorch:20260129

sudo docker run --gpus all -it \
  -w /home/projects/ML_sdfi_fastai2 \
  rasmuspjohansson/kds_cuda_pytorch:20260129 \
  /bin/bash
```

### mamba installation should work but is not as actively maintained


```sh
mamba env create --file environment.yml
mamba activate ML_sdfi
pip install -e .
```

### Verify CUDA Support

```sh
python
>>> import torch
>>> torch.cuda.is_available()
True
```

If `torch.cuda.is_available()` returns `False`, you may need to reinstall a CUDA-compatible version of PyTorch following the instructions at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

For example:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Training

Configuration file:  
`configs/example_configs/train_example_dataset.ini`

Run:
```sh
python src/ML_sdfi_fastai2/train.py --config configs/example_configs/train_example_dataset.ini
```

---

### Inference

Configuration file:  
`configs/example_configs/infer_example_dataset.ini`

Run:
```sh
python src/ML_sdfi_fastai2/infer.py --config configs/example_configs/infer_example_dataset.ini
```

---

### Reporting

Configuration file:  
`configs/example_configs/report_example_dataset.ini`

Run:
```sh
In order to analyse how well the model does on a dataset we can create a .pdf report with loss plots , confusion matrixes and example iamges and predictions

cd ../multi_channel_dataset_creation
python src/multi_channel_dataset_creation/geopackage_to_label_v2.py --geopackage example_dataset/labels/example_dataset_buildings.gpkg --input_folder example_dataset/data/splitted/rgb/ --output_folder example_dataset/buildings/splitted_buildings/ --background_value 0 --value_used_for_all_polygons 1
cd ../ML_sdfi_fastai2
python src/ML_sdfi_fastai2/report.py --config configs/example_configs/report_example_dataset.ini

If you get confusing errors when creating report it is often casued by a faulty .csv file 
check ../logs_and_models/example_dataset_iter_1/iter_1/logs/iter_1.csv 
This file should look like this

epoch,train_loss,valid_loss,valid_accuracy,time,lr_0,lr_1,lr_2
0,2.671426296234131,2.261967182159424,0.15685869753360748,00:07,8e-05,8e-05,8e-05
1,2.4981231689453125,1.2677741050720215,0.6439558863639832,00:06
...
...
9,3.6438961029052734,1.2790676355361938,0.6442530751228333,00:06,8.64737438191444e-05,8.64737438191444e-05,8.64737438191444e-05

```

---

### Downloading and uploading models (Hugging Face)

Trained `.pth` models can be downloaded from or uploaded to [Hugging Face Hub](https://huggingface.co/). The default repo used by the scripts is [rasmuspjohansson/KDS_buildings](https://huggingface.co/rasmuspjohansson/KDS_buildings). The dependency `huggingface_hub` is included in the environment; ensure it is installed (`pip install huggingface_hub` if needed).

#### Download models from Hugging Face

Download one or all `.pth` model files from the repo into a local directory:

```sh
# Download all .pth files from the repo into a folder
python src/ML_sdfi_fastai2/download_models_from_huggingface.py \
  --output_dir /mnt/T/mnt/logs_and_models/bygningsudpegning

# Download a single model file
python src/ML_sdfi_fastai2/download_models_from_huggingface.py \
  --output_dir ./models \
  --model_file andringsudpegning_1km2benchmark_iter_73.pth
```

Options:
- `--repo_id`: Hugging Face repo (default: `rasmuspjohansson/KDS_buildings`)
- `--output_dir`: Local directory where files are saved (required)
- `--model_file`: If set, only this filename is downloaded; otherwise all `.pth` files in the repo are downloaded
- `--token_file`: Path to a file containing your Hugging Face token (for private repos). Default: `../laz-superpoint_transformer/hftoken_write.txt` relative to the project root

To use a downloaded model for inference, point your inference config’s `model_to_load` (or equivalent) to the downloaded file, e.g. `.../bygningsudpegning/andringsudpegning_1km2benchmark_iter_73.pth`.

#### Upload models to Hugging Face

Upload models that were trained using `train.py` and whose configs live in a directory of `train_*.ini` files. The script reads each config’s `experiment_root` and `job_name`, finds the corresponding `{experiment_root}/{job_name}/models/{job_name}.pth`, and uploads it to the Hub:

```sh
# Upload all models from configs in the default config directory
python src/ML_sdfi_fastai2/upload_models_to_huggingface.py \
  --config_dir /mnt/T/mnt/config_files/bygnings_udpegning/2026_production/

# Dry run: only print what would be uploaded
python src/ML_sdfi_fastai2/upload_models_to_huggingface.py \
  --config_dir /mnt/T/mnt/config_files/bygnings_udpegning/2026_production/ \
  --dry_run
```

Options:
- `--config_dir`: Directory containing `train_*.ini` config files (default: `/mnt/T/mnt/config_files/bygnings_udpegning/2026_production`)
- `--token_file`: Path to a file containing your Hugging Face **write** token. Default: `../laz-superpoint_transformer/hftoken_write.txt` relative to the project root
- `--repo_id`: Target Hugging Face repo (default: `rasmuspjohansson/KDS_buildings`)
- `--dry_run`: List configs and model paths that would be uploaded without uploading

Only configs for which the corresponding `.pth` file exists are uploaded; others are skipped with a message.

---

## Example Dataset

All example configuration files are compatible with the example dataset available at:  
👉 [https://github.com/SDFIdk/multi_channel_dataset_creation](https://github.com/SDFIdk/multi_channel_dataset_creation)




