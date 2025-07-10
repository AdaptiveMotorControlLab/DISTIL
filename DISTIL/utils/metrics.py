"""
Metrics and evaluation utilities for backdoor detection.
"""

import math
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def calculate_accuracy(outputs, targets):
    """
    Calculate classification accuracy from model outputs and targets.
    
    Args:
        outputs (torch.Tensor): Model output predictions (logits)
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        float: Accuracy as a percentage (0-100)
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100.0 * correct / total

def load_results_and_compute_auc(file_path, architecture="ssd"):
    """
    Reads lines from a results file and computes the AUC-ROC for detecting 'backdoored'
    vs 'clean' models based on Avg_Trigger_Score.
    
    Args:
        file_path (str): Path to the results file containing model scores.
        architecture (str): The architecture type, 'ssd' or 'frcnn'.
            If the model's architecture is 'frcnn', the label for 'backdoored' models is set to 0.
            
    Returns:
        float: The computed AUC-ROC score.
    """
    models = []
    states = []
    avg_scores = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Skip header:
    header = lines[0]
    data_lines = lines[1:]

    for line in data_lines:
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        model_id = cols[0].strip()
        state_str = cols[1].strip()
        avg_str = cols[2].strip()
        
        if architecture.lower() == 'ssd':
            is_backdoored = 1 if state_str.lower() == 'backdoored' else 0
        elif architecture.lower() == 'frcnn':
            is_backdoored = 0 if state_str.lower() == 'backdoored' else 1
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Use 'ssd' or 'frcnn'.")
        
        try:
            avg_val = float(avg_str)
        except ValueError:
            continue
        if math.isnan(avg_val):
            continue
        models.append(model_id)
        states.append(is_backdoored)
        avg_scores.append(avg_val)
    
    if not states:
        print("No valid data to compute AUC.")
        return 0.5  # Random classifier
    
    y_true = np.array(states, dtype=int)
    y_score = np.array(avg_scores, dtype=float)
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC-ROC: {auc:.4f}")
    return auc

def get_model_state(model_path):
    """
    Determine if a model is backdoored based on directory structure.
    Return 'backdoored' if 'poisoned-example-data' folder exists, else 'clean'.
    
    Args:
        model_path (str): Path to the model directory.
        
    Returns:
        str: 'backdoored' or 'clean'
    """
    import os
    poisoned_dir = os.path.join(model_path, "poisoned-example-data")
    return "backdoored" if os.path.isdir(poisoned_dir) else "clean" 