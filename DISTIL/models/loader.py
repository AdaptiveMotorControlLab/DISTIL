"""
Unified model loading interface for detection models.
"""

import torch
import torch.nn as nn
from .ssd import load_ssd_checkpoint
from .frcnn import load_frcnn_checkpoint, fix_fasterrcnn_classifier

def load_detection_model(ckpt_path, architecture="ssd", device=None):
    """
    Unified interface for loading detection models.
    
    Args:
        ckpt_path (str): Path to the checkpoint file.
        architecture (str): 'ssd' or 'frcnn'.
        device (torch.device, optional): Device to load the model onto.
        
    Returns:
        nn.Module: The loaded model.
        
    Raises:
        ValueError: If architecture is not supported.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if architecture.lower() == "ssd":
        return load_ssd_checkpoint(ckpt_path, device)
    elif architecture.lower() == "frcnn":
        model = load_frcnn_checkpoint(ckpt_path, device)
        if model is not None:
            model = fix_fasterrcnn_classifier(model)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Use 'ssd' or 'frcnn'.")

def get_classifier_layer(model):
    """
    Retrieve the classifier layer from a detection model.
    Returns the SSD classification head (if available), the FasterRCNN box predictor,
    or falls back to the last nn.Linear in the model.
    
    Args:
        model (nn.Module): The detection model.
        
    Returns:
        nn.Module: The classifier layer.
        
    Raises:
        ValueError: If no suitable classifier is found.
    """
    if hasattr(model, "model"):
        # Check for SSD head
        if hasattr(model.model, "head") and hasattr(model.model.head, "classification_head"):
            cls_head = model.model.head.classification_head
            if cls_head is not None:
                return cls_head
                
        # Check for Faster R-CNN box predictor
        if hasattr(model.model, "roi_heads") and hasattr(model.model.roi_heads, "box_predictor"):
            return model.model.roi_heads.box_predictor

    # Fall back to finding any linear layer
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    
    if last_linear is None:
        raise ValueError("No linear (or detection classification head) found in the model.")
        
    return last_linear