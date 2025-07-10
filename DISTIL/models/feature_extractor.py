"""
Feature extraction utilities for object detection models.

This module provides functionality to extract features, classifier layers, and class
vector representations from different object detection architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any

from ..configs.config import DEVICE, DEFAULT_ARCHITECTURE

# Set architecture from environment or use default
ARCHITECTURE = DEFAULT_ARCHITECTURE


def get_classifier_layer(model: nn.Module) -> nn.Module:
    """
    Retrieve the classifier layer from a detection model.
    
    Supports both SSD classification heads and FasterRCNN box predictors.
    Falls back to the last nn.Linear layer if no specific classifier is found.
    
    Args:
        model: The detection model
        
    Returns:
        The classifier module
        
    Raises:
        ValueError: If no suitable classifier layer is found
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


def extract_class_vectors_from_ssd_head(classifier_layer: nn.Module, num_classes: int) -> torch.Tensor:
    """
    Extract class vectors from a convolutional classification head by spatially averaging weights.
    
    Works for both SSD models and FasterRCNN models that use convolutional layers for classification.
    
    Args:
        classifier_layer: The classification head module
        num_classes: Number of classes to extract vectors for
        
    Returns:
        Tensor of class vectors with shape [num_classes, feature_dim]
        
    Raises:
        ValueError: If no convolutional layers are found in the classifier
    """
    all_scale_protos = []
    for module in classifier_layer.modules():
        if isinstance(module, nn.Conv2d):
            # Average each convolution weight spatially
            conv_w = module.weight.data  # [out_channels, in_channels, k, k]
            w = conv_w.mean(dim=(2, 3))   # [out_channels, in_channels]
            # Split into groups for each class
            chunks = torch.chunk(w, num_classes, dim=0)
            proto = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
            all_scale_protos.append(proto)

    if not all_scale_protos:
        raise ValueError("No convolutional layers found in classification head; cannot extract embeddings.")

    # Trim prototypes to a common feature dimension
    common_dim = min(proto.size(1) for proto in all_scale_protos)
    trimmed_protos = [proto[:, :common_dim] for proto in all_scale_protos]
    stacked = torch.stack(trimmed_protos, dim=0)
    final_prototypes = stacked.mean(dim=0)
    return final_prototypes


def greedy_class_farthest(model: nn.Module) -> List[Tuple[int, int, float]]:
    """
    For each class, find the farthest (least similar) class based on cosine similarity.
    
    Works with both SSD and FasterRCNN models by dynamically detecting the appropriate
    classifier structure and extracting class vectors accordingly.
    
    Args:
        model: The detection model
        
    Returns:
        A list of tuples (source_class, target_class, similarity_score) where
        target_class is the farthest class from source_class
    """
    classifier_layer = get_classifier_layer(model)
    
    # Extract weight matrix based on classifier type
    if hasattr(classifier_layer, "cls_score"):
        # FasterRCNN box predictor
        weight = classifier_layer.cls_score.weight.data
        num_classes = weight.shape[0]
    elif hasattr(classifier_layer, "cls_logits") or "SSDClassificationHead" in str(type(classifier_layer)):
        # SSD classification head
        num_classes = 21  # Default for COCO/Pascal VOC, adjust as needed
        weight = extract_class_vectors_from_ssd_head(classifier_layer, num_classes)
    else:
        # Check if it uses convolutional layers
        has_conv = any(isinstance(module, nn.Conv2d) for module in classifier_layer.modules())
        if has_conv:
            num_classes = 21  # Default for COCO/Pascal VOC, adjust as needed
            weight = extract_class_vectors_from_ssd_head(classifier_layer, num_classes)
        else:
            # Standard linear layer
            weight = classifier_layer.weight.data
            num_classes = weight.shape[0]

    # Normalize and compute cosine similarity
    norm_w = weight / weight.norm(dim=1, keepdim=True)
    sim_matrix = norm_w @ norm_w.t()
    
    # Mask out self-similarity
    diag_mask = torch.eye(num_classes, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix.masked_fill_(diag_mask, float('inf'))

    # Find farthest class for each class
    results = []
    for i in range(num_classes):
        j = torch.argmin(sim_matrix[i]).item()
        sim_val = sim_matrix[i, j].item()
        results.append((i, j, sim_val))
    return results


def extract_features_from_ssd(model: nn.Module, image: torch.Tensor, layer_name: str = None) -> torch.Tensor:
    """
    Extract intermediate features from an SSD model using a forward hook.
    
    Args:
        model: The SSD model
        image: Input image tensor [B, C, H, W]
        layer_name: Optional specific layer name to extract features from
        
    Returns:
        Extracted feature tensor
        
    Raises:
        RuntimeError: If feature extraction fails
        AttributeError: If specified layer doesn't exist
    """
    candidate_obj = model.model if hasattr(model, "model") else model
    if hasattr(candidate_obj, "backbone"):
        candidate_obj = candidate_obj.backbone

    if layer_name is not None:
        if hasattr(candidate_obj, layer_name):
            selected_layer = getattr(candidate_obj, layer_name)
        else:
            raise AttributeError(f"SSD model backbone has no attribute '{layer_name}'.")
    else:
        selected_layer = getattr(candidate_obj, "features", candidate_obj)

    features = None

    def hook_fn(module, input, output):
        nonlocal features
        features = output

    handle = selected_layer.register_forward_hook(hook_fn)
    _ = model(image)
    handle.remove()

    if features is None:
        raise RuntimeError("Feature extraction hook did not capture any output.")
    if features.ndim == 3:
        features = features.unsqueeze(0)
    return features


def extract_features_from_detection(model: nn.Module, image: torch.Tensor, layer_name: str = None) -> torch.Tensor:
    """
    Extract intermediate features from a detection model (especially FasterRCNN).
    
    Args:
        model: The detection model
        image: Input image tensor [B, C, H, W]
        layer_name: Optional specific layer name to extract features from
        
    Returns:
        Extracted feature tensor
        
    Raises:
        RuntimeError: If feature extraction fails
        AttributeError: If specified layer doesn't exist
    """
    candidate_obj = model.model if hasattr(model, "model") else model
    if hasattr(candidate_obj, "backbone"):
        candidate_obj = candidate_obj.backbone

    if layer_name is not None:
        if hasattr(candidate_obj, layer_name):
            selected_layer = getattr(candidate_obj, layer_name)
        else:
            raise AttributeError(f"Detection model backbone has no attribute '{layer_name}'.")
    else:
        selected_layer = getattr(candidate_obj, "features", candidate_obj)

    features = None

    def hook_fn(module, input, output):
        nonlocal features
        features = output

    handle = selected_layer.register_forward_hook(hook_fn)
    
    # For detection models, the forward pass might expect a list of images
    if hasattr(model, "model") and hasattr(model.model, "roi_heads"):
        _ = model([image.squeeze(0) if image.dim() == 4 and image.size(0) == 1 else image])
    else:
        _ = model(image)
    handle.remove()

    if features is None:
        raise RuntimeError("Feature extraction hook did not capture any output.")
    if isinstance(features, dict):
        features = list(features.values())[0]
    if features.ndim == 3:
        features = features.unsqueeze(0)
    return features


def remap_fastrcnn_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap state dictionary keys for FasterRCNN checkpoints with different key structures.
    
    Handles various key naming patterns for FPN layers and RPN head components.
    
    Args:
        state_dict: The original state dictionary
        
    Returns:
        Remapped state dictionary with compatible keys
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.fpn.inner_blocks."):
            parts = k.split('.')
            if len(parts) == 5:
                new_k = ".".join(parts[:3] + [parts[3], "0"] + parts[4:])
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        elif k.startswith("backbone.fpn.layer_blocks."):
            parts = k.split('.')
            if len(parts) == 5:
                new_k = ".".join(parts[:3] + [parts[3], "0"] + parts[4:])
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        elif k.startswith("rpn.head.conv") and ("." not in k[len("rpn.head.conv"):]):
            new_k = "rpn.head.conv.0.0" + k[len("rpn.head.conv"):]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def fix_fasterrcnn_classifier(model: nn.Module, num_classes: int = 91) -> nn.Module:
    """
    Ensure the FasterRCNN classifier head has the expected number of classes.
    
    Replaces the box predictor with a new FastRCNNPredictor if needed.
    
    Args:
        model: The FasterRCNN model
        num_classes: Number of classes for the classifier
        
    Returns:
        Model with fixed classifier head
    """
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    if hasattr(model, "model") and hasattr(model.model, "roi_heads"):
        predictor = model.model.roi_heads.box_predictor
        if predictor.cls_score.weight.shape[0] < num_classes:
            in_features = predictor.cls_score.in_features
            model.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            print(f"Replaced FasterRCNN classifier head with {num_classes} classes.")
    return model


# Unified feature extraction interface that works with both architectures
def extract_features(
    model: nn.Module, 
    image: torch.Tensor, 
    layer_name: str = None
) -> torch.Tensor:
    """
    Unified interface to extract features from any detection model.
    
    Automatically detects the model type and uses the appropriate extraction method.
    
    Args:
        model: The detection model (SSD or FasterRCNN)
        image: Input image tensor
        layer_name: Optional specific layer name to extract features from
        
    Returns:
        Extracted feature tensor
    """
    # Detect if this is a FasterRCNN model
    is_frcnn = hasattr(model, "model") and hasattr(model.model, "roi_heads")
    
    if is_frcnn or ARCHITECTURE.lower() == "frcnn":
        return extract_features_from_detection(model, image, layer_name)
    else:
        return extract_features_from_ssd(model, image, layer_name)
