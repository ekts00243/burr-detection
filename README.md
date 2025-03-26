# Burr Detection System

This project implements a burr detection system using computer vision and deep learning techniques.

## Project Structure

- `data/`: Contains raw videos, processed frames, and trained models
- `src/`: Source code for data processing, models, and visualization
- `notebooks/`: Jupyter notebooks for exploration and development
- `docker/`: Docker configuration files
- `config.py`: Configuration parameters
- `train.py`: Model training script
- `predict.py`: Inference script

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Or use Docker:
```bash
docker-compose up
```

## Usage

1. Place raw videos in `data/raw/`
2. Run preprocessing:
```bash
python src/data/video_processor.py
```
3. Train model:
```bash
python train.py
```
4. Run inference:
```bash
python predict.py
```
