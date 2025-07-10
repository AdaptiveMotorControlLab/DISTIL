#!/usr/bin/env python
"""
Script to perform grid search over different configurations for backdoor detection.
Tests various parameter combinations and reports the best configuration based on AUC scores.
"""

import os
import yaml
import argparse
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import torch
from sklearn.metrics import roc_auc_score

from DISTIL.DISTIL.data_loader import model_loader
from DISTIL.DISTIL.detection import backdoor_detector

def load_config(config_path: str) -> Dict[str, Any]:
    """Load base configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_config_combinations(base_config: Dict[str, Any], param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all possible combinations of parameters from the parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    configs = []
    for combo in combinations:
        config = base_config.copy()
        for key, value in zip(keys, combo):
            config[key] = value
        configs.append(config)
    
    return configs

def get_model_architecture(model_data: Dict[str, Any]) -> str:
    """Extract model architecture from model data."""
    model = model_data['model_weight']
    return model.__class__.__name__

def evaluate_config(config: Dict[str, Any], output_dir: str) -> Dict[str, float]:
    """
    Evaluate a single configuration and return metrics.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create output directory for this configuration
    config_dir = os.path.join(output_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    # Load models
    clean_dataset = model_loader.TrojAIModelBatchDataset(
        root_dir=config['models_dir'],
        transform=model_loader.transform_image_tensor(),
        poison=False
    )
    
    backdoor_dataset = model_loader.TrojAIModelBatchDataset(
        root_dir=config['models_dir'],
        transform=model_loader.transform_image_tensor(),
        poison=True
    )
    
    # Filter models by architecture if specified
    if 'model_architecture' in config:
        clean_dataset = [m for m in clean_dataset if get_model_architecture(m) == config['model_architecture']]
        backdoor_dataset = [m for m in backdoor_dataset if get_model_architecture(m) == config['model_architecture']]
        print(f"Filtered to {len(clean_dataset)} clean and {len(backdoor_dataset)} backdoored {config['model_architecture']} models")
    
    # Evaluate models
    all_results = []
    
    # Evaluate clean models
    for i in range(min(len(clean_dataset), config.get('num_clean', 5))):
        model_data = clean_dataset[i]
        try:
            score, _ = backdoor_detector.detect_backdoor(
                model_data,
                metadata_path=None,
                guidance_scale=config['guidance_scale'],
                num_iterations=config['num_iterations'],
                timestep=config['timestep'],
                search_strategy=config['search_strategy'],
                model_index=i
            )
            all_results.append({
                'model_id': model_data['model_id'],
                'architecture': get_model_architecture(model_data),
                'is_backdoored': 0,
                'score': score
            })
        except Exception as e:
            print(f"Error evaluating clean model {model_data['model_id']}: {e}")
            all_results.append({
                'model_id': model_data['model_id'],
                'architecture': get_model_architecture(model_data),
                'is_backdoored': 0,
                'score': -1
            })
    
    # Evaluate backdoored models
    for i in range(min(len(backdoor_dataset), config.get('num_backdoor', 5))):
        model_data = backdoor_dataset[i]
        try:
            score, _ = backdoor_detector.detect_backdoor(
                model_data,
                metadata_path=config['metadata_path'],
                guidance_scale=config['guidance_scale'],
                num_iterations=config['num_iterations'],
                timestep=config['timestep'],
                search_strategy=config['search_strategy'],
                model_index=i
            )
            all_results.append({
                'model_id': model_data['model_id'],
                'architecture': get_model_architecture(model_data),
                'is_backdoored': 1,
                'score': score
            })
        except Exception as e:
            print(f"Error evaluating backdoored model {model_data['model_id']}: {e}")
            all_results.append({
                'model_id': model_data['model_id'],
                'architecture': get_model_architecture(model_data),
                'is_backdoored': 1,
                'score': -1
            })
    
    # Calculate metrics
    results_df = pd.DataFrame(all_results)
    valid_results = results_df[results_df['score'] >= 0]
    
    if len(valid_results) == 0:
        return {
            'auc_score': 0.0,
            'mean_clean_score': 0.0,
            'mean_backdoor_score': 0.0,
            'std_clean_score': 0.0,
            'std_backdoor_score': 0.0,
            'num_errors': len(results_df) - len(valid_results)
        }
    
    clean_scores = valid_results[valid_results['is_backdoored'] == 0]['score'].values
    backdoor_scores = valid_results[valid_results['is_backdoored'] == 1]['score'].values
    
    # Calculate AUC
    all_scores = np.concatenate([clean_scores, backdoor_scores])
    all_labels = np.concatenate([np.zeros_like(clean_scores), np.ones_like(backdoor_scores)])
    auc_score = roc_auc_score(all_labels, all_scores)
    
    # Save results
    results_df.to_csv(os.path.join(config_dir, "results.csv"), index=False)
    
    return {
        'auc_score': auc_score,
        'mean_clean_score': np.mean(clean_scores) if len(clean_scores) > 0 else 0.0,
        'mean_backdoor_score': np.mean(backdoor_scores) if len(backdoor_scores) > 0 else 0.0,
        'std_clean_score': np.std(clean_scores) if len(clean_scores) > 0 else 0.0,
        'std_backdoor_score': np.std(backdoor_scores) if len(backdoor_scores) > 0 else 0.0,
        'num_errors': len(results_df) - len(valid_results)
    }

def main():
    parser = argparse.ArgumentParser(description="Perform grid search over backdoor detection configurations")
    parser.add_argument("--base_config", type=str, required=True,
                        help="Path to the base configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="./grid_search_results",
                        help="Directory to save grid search results")
    parser.add_argument("--model_architecture", type=str, default=None,
                        help="Filter models by architecture (e.g., 'ResNet')")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base configuration
    base_config = load_config(args.base_config)
    
    # Add model architecture filter if specified
    if args.model_architecture:
        base_config['model_architecture'] = args.model_architecture
    
    # Define parameter grid
    param_grid = {
        'guidance_scale': [50, 100, 200],
        'num_iterations': [1, 2, 3],
        'timestep': [25, 50, 75],
        'search_strategy': ['greedy', 'random']
    }
    
    # Generate all configurations
    configs = generate_config_combinations(base_config, param_grid)
    print(f"Testing {len(configs)} different configurations...")
    
    # Evaluate each configuration
    results = []
    for i, config in enumerate(configs):
        print(f"\nEvaluating configuration {i+1}/{len(configs)}")
        print("Parameters:", {k: v for k, v in config.items() if k in param_grid})
        
        metrics = evaluate_config(config, args.output_dir)
        results.append({
            **{k: v for k, v in config.items() if k in param_grid},
            **metrics
        })
        
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        print(f"Mean Clean Score: {metrics['mean_clean_score']:.4f}")
        print(f"Mean Backdoor Score: {metrics['mean_backdoor_score']:.4f}")
        print(f"Number of Errors: {metrics['num_errors']}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by AUC score
    results_df = results_df.sort_values('auc_score', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, "grid_search_results.csv"), index=False)
    
    # Print best configuration
    best_config = results_df.iloc[0]
    print("\nBest Configuration:")
    print("Parameters:")
    for param in param_grid.keys():
        print(f"  {param}: {best_config[param]}")
    print(f"AUC Score: {best_config['auc_score']:.4f}")
    print(f"Mean Clean Score: {best_config['mean_clean_score']:.4f}")
    print(f"Mean Backdoor Score: {best_config['mean_backdoor_score']:.4f}")
    
    # Save best configuration
    best_config_dict = {k: v for k, v in best_config.items() if k in param_grid}
    with open(os.path.join(args.output_dir, "best_config.yaml"), 'w') as f:
        yaml.dump(best_config_dict, f)

if __name__ == "__main__":
    main() 