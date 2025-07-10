import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ..utils.model_utils import plot_image_tensor

def plot_confidence_histogram(clean_scores, backdoor_scores, threshold=0.5, save_path=None):
    """
    Plot a histogram of confidence scores for clean and backdoored models.
    
    Args:
        clean_scores: List of scores for clean models
        backdoor_scores: List of scores for backdoored models
        threshold: Decision threshold for backdoor detection
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(clean_scores, bins=20, alpha=0.5, label='Clean Models', color='green')
    plt.hist(backdoor_scores, bins=20, alpha=0.5, label='Backdoored Models', color='red')
    
    # Add threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
    
    # Add labels and legend
    plt.xlabel('Backdoor Score')
    plt.ylabel('Number of Models')
    plt.title('Distribution of Backdoor Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_roc_curve(clean_scores, backdoor_scores, save_path=None):
    """
    Plot ROC curve for backdoor detection.
    
    Args:
        clean_scores: List of scores for clean models
        backdoor_scores: List of scores for backdoored models
        save_path: Path to save the figure (optional)
    """
    from sklearn.metrics import roc_curve, auc
    
    # Prepare data for ROC curve
    y_true = [0] * len(clean_scores) + [1] * len(backdoor_scores)
    y_scores = clean_scores + backdoor_scores
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return roc_auc


def visualize_triggers(triggers, target_labels, scores, max_triggers=5, save_path=None):
    """
    Visualize top triggers and their effectiveness.
    
    Args:
        triggers: List of trigger tensors
        target_labels: List of target labels for each trigger
        scores: List of backdoor scores for each trigger
        max_triggers: Maximum number of triggers to visualize
        save_path: Path to save the figure (optional)
    """
    # Sort triggers by score
    sorted_indices = np.argsort(scores)[::-1]
    
    # Get top triggers
    top_indices = sorted_indices[:max_triggers]
    top_triggers = [triggers[i] for i in top_indices]
    top_labels = [target_labels[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    
    # Create figure
    fig, axes = plt.subplots(1, len(top_triggers), figsize=(4*len(top_triggers), 4))
    
    if len(top_triggers) == 1:
        axes = [axes]
    
    for i, (trigger, label, score) in enumerate(zip(top_triggers, top_labels, top_scores)):
        if torch.is_tensor(trigger):
            # Convert tensor to numpy for plotting
            trigger = trigger.squeeze(0).cpu()
            trigger = trigger.permute(1, 2, 0).numpy()
        
        axes[i].imshow(trigger)
        axes[i].set_title(f"Label: {label}\nScore: {score:.3f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def compare_clean_backdoor_models(clean_results, backdoor_results, save_path=None):
    """
    Compare detection results between clean and backdoored models.
    
    Args:
        clean_results: Dictionary of results for clean models
        backdoor_results: Dictionary of results for backdoored models
        save_path: Path to save the figure (optional)
    """
    # Extract scores
    clean_scores = list(clean_results.values())
    backdoor_scores = list(backdoor_results.values())
    
    # Calculate statistics
    clean_mean = np.mean(clean_scores)
    clean_std = np.std(clean_scores)
    backdoor_mean = np.mean(backdoor_scores)
    backdoor_std = np.std(backdoor_scores)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create box plots
    plt.boxplot([clean_scores, backdoor_scores], 
                labels=['Clean Models', 'Backdoored Models'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red'))
    
    # Add scatter points for individual models
    x1 = np.random.normal(1, 0.1, size=len(clean_scores))
    x2 = np.random.normal(2, 0.1, size=len(backdoor_scores))
    
    plt.scatter(x1, clean_scores, alpha=0.5, color='blue')
    plt.scatter(x2, backdoor_scores, alpha=0.5, color='red')
    
    # Add statistics as text
    plt.text(0.8, max(clean_scores + backdoor_scores) * 0.9, 
             f"Clean: μ={clean_mean:.3f}, σ={clean_std:.3f}\nBackdoor: μ={backdoor_mean:.3f}, σ={backdoor_std:.3f}",
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Add labels and title
    plt.ylabel('Backdoor Score')
    plt.title('Comparison of Backdoor Scores')
    plt.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return clean_mean, clean_std, backdoor_mean, backdoor_std 