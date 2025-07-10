"""
Utility functions for the DISTIL package.
"""

# GPU utilities
from .gpu_check import list_gpus, set_gpu

# Model utilities 
from .model_utils import (
    get_classifier_layer,
    get_images_and_labels_by_label,
    greedy_class_farthest,
    prepare_classifier,
    trigger_evaluation
)

# Transform utilities
from .transforms import (
    get_dataset_normalization,
    get_transform,
    transform_image,
    transform_image_tensor
)

# Visualization utilities
from .visualization import (
    plot_image_tensor,
    show_diffusion_images,
    plot_images_grid
)

# Metrics utilities
from .metrics import (
    calculate_accuracy,
    # calculate_asr,
    # calculate_confusion_matrix
)

__all__ = [
    # GPU utilities
    "list_gpus", 
    "set_gpu",
    
    # Model utilities
    "get_classifier_layer",
    "get_images_and_labels_by_label",
    "greedy_class_farthest",
    "prepare_classifier",
    "trigger_evaluation",
    
    # Transform utilities
    "get_dataset_normalization",
    "get_transform",
    "transform_image",
    "transform_image_tensor",
    
    # Visualization utilities
    "plot_image_tensor",
    "show_diffusion_images",
    "plot_images_grid",
    
    # Metrics utilities
    "calculate_accuracy",
    "calculate_asr",
    "calculate_confusion_matrix"
]
