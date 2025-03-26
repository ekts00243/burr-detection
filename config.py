"""
Configuration parameters for the burr detection system.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Video processing settings
FRAME_EXTRACTION_INTERVAL = 5  # Extract every Nth frame
TARGET_RESOLUTION = (256, 256)  # (height, width)

# Data preprocessing settings
NORMALIZATION_MEAN = 0.5
NORMALIZATION_STD = 0.5
TRAIN_TEST_SPLIT = 0.2

# Model settings
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Anomaly detection settings
ANOMALY_THRESHOLD_PERCENTILE = 95  # Set threshold at this percentile of training reconstruction errors