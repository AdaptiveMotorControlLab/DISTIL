#!/usr/bin/env python3
"""
Script to detect target classes in backdoored models.
"""

import os
import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append('../../../')

from DISTIL.DISTIL.detection.backdoor_target_detector import BackdoorTargetDetector
from DISTIL.DISTIL.data_loader import backbench

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Detect target classes in backdoored models")
    
    parser.add_argument("--model_dir", type=str, default="", # Backdoorbench dataset path
                        help="Directory containing backdoored models")
    parser.add_argument("--output_file", type=str, default="backdoor_target_class_prediction_results.txt",
                        help="Path to output file for detection results")
    parser.add_argument("--dataset", type=str, default='cifar10',
                        help="Filter models by dataset name (e.g., cifar10, gtsrb)")
    parser.add_argument("--model_arch", type=str, default='vgg19',
                        help="Filter models by architecture (e.g., vgg19, preactresnet18)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of models to analyze")
    parser.add_argument("--search_all_classes", action="store_true",
                        help="Search all possible classes as potential targets (computationally expensive)")
    
    return parser.parse_args()

def main():
    """Main function to detect target classes in backdoored models."""
    args = parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create detector
    detector = BackdoorTargetDetector(output_file=args.output_file)
    
    # Load backdoored models
    print(f"Loading backdoored models from {args.model_dir}...")
    backdoor_models = backbench.BackbenchDataset(args.model_dir,model_name=args.model_arch,dataset_filter=args.dataset)
    
    if not backdoor_models:
        print("No models loaded. Please check your filters and model directory.")
        return
    
    print(f"Successfully loaded {len(backdoor_models)} models. Starting detection...")
    
    # Process each model
    results = []
    for model_data in tqdm(backdoor_models, desc="Detecting backdoor targets"):
        model_name = model_data["model_name"]
        print(f"\nAnalyzing model: {model_name}")
        
        try:
            # Detect target class
            predicted_target, scores, trigger = detector.detect_target_class(
                model_data,
            )
            
            # Print results
            print(f"Model: {model_name}")
            print(f"Predicted target class: {predicted_target}")
            print(f"Confidence scores: {scores}")
            
            results.append({
                "model_name": model_name,
                "predicted_target": predicted_target,
                "scores": scores
            })
            
        except Exception as e:
            print(f"Error analyzing model {model_name}: {e}")
    
    print(f"\nDetection completed. Results saved to {args.output_file}")
    print(f"Total models analyzed: {len(results)}")

if __name__ == "__main__":
    main() 