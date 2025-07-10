"""
Model definitions and loaders for object detection models.
"""

from .loader import load_detection_model, get_classifier_layer
from .feature_extractor import extract_features_from_ssd, extract_features_from_detection
from .similarity import greedy_class_farthest
from .frcnn import load_frcnn_checkpoint, fix_fasterrcnn_classifier
