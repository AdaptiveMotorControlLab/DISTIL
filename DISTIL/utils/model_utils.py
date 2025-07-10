import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import inception_v3
from copy import deepcopy
from torchvision import transforms
from torchvision.transforms import Resize

from .transforms import transform_image_tensor
from .visualization import plot_image_tensor, show_diffusion_images

def get_classifier_layer(model):
    """
    Get the classifier layer of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        layer: Classifier layer of the model
    """
    # Try common attribute names
    for attr in ['classifier', 'fc']:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if hasattr(layer, 'weight'):
                return layer
                
    # Fallback: search for the last nn.Linear layer in the model's modules
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    
    if last_linear is not None:
        return last_linear
        
    raise ValueError("No linear classifier layer with weights found in the model.")


def get_images_and_labels_by_label(images, labels, target_label):
    """
    Filter images and labels by a specific label.
    
    Args:
        images: Tensor of images
        labels: Tensor of labels
        target_label: The label to filter by
        
    Returns:
        filtered_images: Tensor containing only images with the target label
        filtered_labels: Tensor containing only target labels
    """
    indices = torch.where(labels == target_label)[0]
    filtered_images = images[indices]
    filtered_labels = labels[indices]
    return filtered_images, filtered_labels


def greedy_class_farthest(model):
    """
    For each class, find the farthest (least similar) class based on 
    the weights of the classifier.
    
    Args:
        model: PyTorch model
        
    Returns:
        matching: List of tuples (class_i, class_j, similarity) where
                 class_j is the farthest class from class_i
    """
    # Extract the classifier layer and weight tensor
    classifier_layer = get_classifier_layer(model)
    weight = classifier_layer.weight.data  # shape: [num_classes, feature_dim]
    num_classes = weight.shape[0]
    
    # Normalize weight vectors to compute cosine similarities
    normalized_weights = weight / weight.norm(dim=1, keepdim=True)
    similarity_matrix = normalized_weights @ normalized_weights.t()
    
    # Exclude self-similarity by masking the diagonal with +infinity
    mask = torch.eye(num_classes, dtype=torch.bool, device=similarity_matrix.device)
    similarity_matrix.masked_fill_(mask, float('inf'))
    
    matching = []
    # For each class, find the farthest (least similar) class
    for i in range(num_classes):
        # Find the index j with the minimum similarity for class i
        j = similarity_matrix[i].argmin().item()
        sim = similarity_matrix[i, j].item()
        matching.append((i, j, sim))
    
    return matching


def show_images(batch, scale_factor=3):
    """
    Display a batch of images with an enlarged size.
    
    Args:
        batch: PyTorch tensor of images
        scale_factor: Factor to scale the images by for display
    """
    show_diffusion_images(batch, scale_factor)


def prepare_classifier(model_data, interested_class=None):
    """
    Prepare a classifier model for evaluation.
    
    Args:
        model_data: Dictionary containing model information
        interested_class: Optional specific class to focus on
        
    Returns:
        classifier: Prepared classifier model
        images: Tensor of relevant images
        labels: Tensor of relevant labels
        trigger_target: Target class for backdoor trigger
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images = model_data["images"]
    labels = model_data["labels"]
    classifier = model_data['model'].to(device)
    
    # Rebuild model if it is an Inception3 model
    if classifier.__class__.__name__ == 'Inception3':
        temp_model = inception_v3(num_classes=classifier.fc.out_features)
        temp_model.load_state_dict(deepcopy(classifier.state_dict()), strict=False)
        classifier = temp_model.to(device).eval()
    
    # Filter images and labels if an interested class is provided
    if interested_class is not None:
        images, labels = get_images_and_labels_by_label(images, labels, interested_class)
    
    return classifier, images, labels


def trigger_evaluation(classifier, trigger, images, pred_labels):
    """
    Evaluate the effectiveness of a trigger on a classifier.
    
    Args:
        classifier: The classifier model
        trigger: The trigger tensor
        images: Tensor of clean images
        pred_labels: Target label for the triggered samples
        
    Returns:
        prob_a: Probability of the target class when trigger is added
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = images 
    x2 = transform_image_tensor(trigger) 
    modified_image = (x1 + x2).clip(0, 1)
    
    with torch.no_grad():
        logits_blended = classifier(modified_image.to(device).clip(0, 1))
        probs_blended = torch.nn.functional.softmax(logits_blended, dim=1)
        prob_a = probs_blended[:, pred_labels].mean().item()
    
    # print(f"Score is (Adding) {prob_a}")
    return prob_a 