"""
Similarity metrics and class relationship analysis for detection models.
"""

import torch
import torch.nn as nn
from .loader import get_classifier_layer
from .feature_extractor import extract_class_vectors_from_ssd_head

def greedy_class_farthest(model):
    """
    For each class, find the farthest (least similar) class based on cosine similarity.
    
    Args:
        model (nn.Module): The detection model to analyze.
    
    Returns:
        A list of tuples (i, j, similarity) where 'i' is the source class 
        and 'j' the farthest class.
    """
    classifier_layer = get_classifier_layer(model)
    
    # If the SSD-specific head is detected based on a property or type string:
    if hasattr(classifier_layer, "cls_logits") or "SSDClassificationHead" in str(type(classifier_layer)):
        num_classes = 21  # Standard COCO classes
        weight = extract_class_vectors_from_ssd_head(classifier_layer, num_classes)
    # If the FRCNN-specific head is detected:
    elif hasattr(classifier_layer, "cls_score"):
        weight = classifier_layer.cls_score.weight.data
        num_classes = weight.shape[0]
    # Generic case - use any available weight matrix:
    else:
        has_conv = any(isinstance(module, nn.Conv2d) for module in classifier_layer.modules())
        if has_conv:
            num_classes = 21  # Assume standard COCO classes
            weight = extract_class_vectors_from_ssd_head(classifier_layer, num_classes)
        else:
            # Try to find weights in any available attribute
            for attr_name in dir(classifier_layer):
                attr = getattr(classifier_layer, attr_name)
                if isinstance(attr, nn.Parameter) and len(attr.shape) == 2:
                    weight = attr.data
                    num_classes = weight.shape[0]
                    break
            else:  # No break occurred
                raise ValueError("Could not locate classifier weights in the model.")

    # Normalize and compute cosine similarity.
    norm_w = weight / weight.norm(dim=1, keepdim=True)
    sim_matrix = norm_w @ norm_w.t()
    diag_mask = torch.eye(num_classes, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix.masked_fill_(diag_mask, float('inf'))

    results = []
    for i in range(num_classes):
        j = torch.argmin(sim_matrix[i]).item()
        sim_val = sim_matrix[i, j].item()
        results.append((i, j, sim_val))
    return results

def class_similarity_matrix(model):
    """
    Compute the full similarity matrix between all classes.
    
    Args:
        model (nn.Module): The detection model to analyze.
    
    Returns:
        Tuple[torch.Tensor, int]: The similarity matrix and the number of classes.
    """
    classifier_layer = get_classifier_layer(model)
    
    # Determine the source of class weights based on model type
    if hasattr(classifier_layer, "cls_logits") or "SSDClassificationHead" in str(type(classifier_layer)):
        num_classes = 21  # Standard COCO classes
        weight = extract_class_vectors_from_ssd_head(classifier_layer, num_classes)
    elif hasattr(classifier_layer, "cls_score"):
        weight = classifier_layer.cls_score.weight.data
        num_classes = weight.shape[0]
    else:
        has_conv = any(isinstance(module, nn.Conv2d) for module in classifier_layer.modules())
        if has_conv:
            num_classes = 21  # Assume standard COCO classes
            weight = extract_class_vectors_from_ssd_head(classifier_layer, num_classes)
        else:
            # Try to find weights in any available attribute
            for attr_name in dir(classifier_layer):
                attr = getattr(classifier_layer, attr_name)
                if isinstance(attr, nn.Parameter) and len(attr.shape) == 2:
                    weight = attr.data
                    num_classes = weight.shape[0]
                    break
            else:  # No break occurred
                raise ValueError("Could not locate classifier weights in the model.")

    # Normalize and compute cosine similarity
    norm_w = weight / weight.norm(dim=1, keepdim=True)
    sim_matrix = norm_w @ norm_w.t()
    
    return sim_matrix, num_classes 