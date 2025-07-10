"""
Configuration settings for object detection models and backdoor detection.
"""

import os
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture selection: 'ssd' or 'frcnn'
DEFAULT_ARCHITECTURE = "ssd"

# Paths configuration
DEFAULT_ROOT_DIR = os.getenv("TROJDET_ROOT_DIR", "/home/public/data/masoud/Projects/Backdoor/object-detection-jul2022-train/models/id-0000000x")
DEFAULT_OUTPUT_DIR = os.getenv("TROJDET_OUTPUT_DIR", "./Object_detection_triggers")
DEFAULT_OUTPUT_FILE = os.getenv("TROJDET_OUTPUT_FILE", "./my_results.txt")

# Model parameters
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_NUM_CLASSES = {
    "ssd": 21,  # Default for SSD (COCO)
    "frcnn": 91  # Default for Faster R-CNN (COCO)
}

# Trigger generation parameters
TRIGGER_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_GUIDANCE_SCALE = 25
DEFAULT_TIMESTEPS = 50
DEFAULT_BATCH_SIZE = 1

# Set these based on your environment
USE_WARNINGS = False  # Set to False to suppress warnings 