#!/usr/bin/env python
"""
Script to evaluate all models in a dataset and get backdoor detection scores.
This version uses a configuration file instead of command-line arguments.
"""
import sys
import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm

# Add the repository root to the Python path
sys.path.append('../../../')

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
import torchvision
from sklearn.metrics import roc_auc_score

from DISTIL.DISTIL.data_loader import model_loader
from DISTIL.DISTIL.detection import backdoor_detector
# from DISTIL.DISTIL.utils import visualization
# from DISTIL.DISTIL.utils import metrics
# from DISTIL.DISTIL.utils import logger

def load_config(config_path):
    """
    Load configuration from a YAML file and convert it to a namespace object.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        config: Namespace containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to namespace for attribute-style access (like args)
    config = SimpleNamespace(**config_dict)
    
    return config

def parse_args():
    """Parse command line arguments for the config file path."""
    parser = argparse.ArgumentParser(description="Evaluate backdoor detection using a config file")
    parser.add_argument("--config", type=str, default='../scripts/config.yaml',
                        help="Path to the configuration YAML file")
    return parser.parse_args()

def evaluate_models(dataset, config, is_backdoored):
    """
    Evaluate a set of models and return their backdoor detection scores.
    
    Args:
        dataset: Dataset of models to evaluate
        config: Configuration parameters
        is_backdoored: Whether the models are known to be backdoored
        
    Returns:
        results: List of dictionaries with model information and scores
    """
    results = []
    
    # Create a directory for trigger images if saving
    triggers_dir = os.path.join(config.output_dir, "triggers")
    if config.save_triggers:
        os.makedirs(triggers_dir, exist_ok=True)
    
    # Evaluate each model
    for i in tqdm(range(len(dataset))):
        model_data = dataset[i]
        model_id = model_data["model_id"]
        parent_id = model_data["parent_id"]
        
        print(f"\nEvaluating {'backdoored' if is_backdoored else 'clean'} model {i+1}/{len(dataset)}: {parent_id}/{model_id}")
        
        # Get start time
        start_time = time.time()
        
        # Detect backdoor and get score
        try:
            max_score, trigger = backdoor_detector.detect_backdoor(
                model_data,
                metadata_path=config.metadata_path if is_backdoored else None,
                guidance_scale=config.guidance_scale,
                num_iterations=config.num_iterations,
                timestep=config.timestep,
                search_strategy=config.search_strategy,
                add_noise=config.add_noise,
                grad_scale_factor=config.grad_scale_factor
            )
            
            # Save trigger image if requested
            if config.save_triggers and trigger is not None:
                trigger_path = os.path.join(triggers_dir, f"{parent_id}_{model_id}.png")
                torchvision.utils.save_image(trigger.cpu(),trigger_path)
                
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            # Store results
            results.append({
                "model_id": model_id,
                "parent_id": parent_id,
                "full_path": model_data["model_path"],
                "is_backdoored": 1 if is_backdoored else 0,
                "score": max_score,
                "time_taken": time_taken,
                "search_strategy": config.search_strategy
            })
            
            print(f"Score: {max_score:.4f}, Time: {time_taken:.2f}s")
            
        except Exception as e:
            print(f"Error evaluating model {parent_id}/{model_id}: {e}")
            results.append({
                "model_id": model_id,
                "parent_id": parent_id,
                "full_path": model_data["model_path"],
                "is_backdoored": 1 if is_backdoored else 0,
                "score": -1,  # Error indicator
                "time_taken": -1,
                "error": str(e),
                "search_strategy": config.search_strategy
            })
    
    return results

def main():
    # Load configuration file
    args = parse_args()
    config = load_config(args.config)
    
    print(f"Using configuration from: {args.config}")
    
    # Set device
    if config.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for evaluation")
    else:
        device = torch.device("cuda")
        print(f"Using GPU for evaluation: {torch.cuda.get_device_name(0)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load clean and backdoored models
    print(f"Loading models from {config.models_dir}")
    
    clean_dataset,backdoor_dataset=model_loader.load_models(config.models_dir)
    
    print(f"Found {len(backdoor_dataset)} backdoored models")
    
    # Start evaluation
    all_results = []
    
    # Evaluate clean models
    if len(clean_dataset) > 0:
        print("\nEvaluating clean models...")
        clean_results = evaluate_models(clean_dataset, config, is_backdoored=False)
        all_results.extend(clean_results)
    
    # Evaluate backdoored models
    if len(backdoor_dataset) > 0:
        print("\nEvaluating backdoored models...")
        backdoor_results = evaluate_models(backdoor_dataset, config, is_backdoored=True)
        all_results.extend(backdoor_results)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    csv_path = os.path.join(config.output_dir, "detection_scores.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Calculate and display statistics
    if len(results_df) > 0:
        print("\nEvaluation Statistics:")
        
        # Filter out error cases
        valid_results = results_df[results_df["score"] >= 0]
        
        if len(valid_results) > 0:
            clean_scores = valid_results[valid_results["is_backdoored"] == 0]["score"].values
            backdoor_scores = valid_results[valid_results["is_backdoored"] == 1]["score"].values
            
            # Print statistics for clean models
            if len(clean_scores) > 0:
                print(f"Clean models (n={len(clean_scores)}):")
                print(f"  Mean score: {np.mean(clean_scores):.4f}")
                print(f"  Std dev: {np.std(clean_scores):.4f}")
                print(f"  Min: {np.min(clean_scores):.4f}, Max: {np.max(clean_scores):.4f}")
            
            # Print statistics for backdoored models
            if len(backdoor_scores) > 0:
                print(f"Backdoored models (n={len(backdoor_scores)}):")
                print(f"  Mean score: {np.mean(backdoor_scores):.4f}")
                print(f"  Std dev: {np.std(backdoor_scores):.4f}")
                print(f"  Min: {np.min(backdoor_scores):.4f}, Max: {np.max(backdoor_scores):.4f}")
            
            # Calculate separation statistics
            if len(clean_scores) > 0 and len(backdoor_scores) > 0:
                # Find threshold that maximizes accuracy
                all_scores = np.concatenate([clean_scores, backdoor_scores])
                all_labels = np.concatenate([np.zeros_like(clean_scores), np.ones_like(backdoor_scores)])
                
                # Calculate ROC AUC
                auc_score = roc_auc_score(all_labels, all_scores)
                print(f"\nROC AUC Score: {auc_score:.4f}")
                
                best_threshold = 0.5
                best_accuracy = 0.0
                
                for threshold in np.linspace(0, 1, 100):
                    predictions = (all_scores > threshold).astype(int)
                    accuracy = np.mean(predictions == all_labels)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
                
                print(f"\nBest separation threshold: {best_threshold:.4f} (Accuracy: {best_accuracy:.4f})")
                
                # Calculate true positive and true negative rates
                true_pos_rate = np.mean(backdoor_scores > best_threshold)
                true_neg_rate = np.mean(clean_scores <= best_threshold)
                
                print(f"True positive rate: {true_pos_rate:.4f}")
                print(f"True negative rate: {true_neg_rate:.4f}")
        
        # Print error statistics
        error_cases = results_df[results_df["score"] < 0]
        if len(error_cases) > 0:
            print(f"\nEncountered errors in {len(error_cases)}/{len(results_df)} models")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()