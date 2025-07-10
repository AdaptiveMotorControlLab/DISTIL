#!/usr/bin/env python
"""
Script to evaluate backdoor detection performance on multiple models.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from DISTIL.DISTIL.data_loader.model_loader import load_models
from DISTIL.DISTIL.detection.backdoor_detector import evaluate_backdoor_detection
from DISTIL.DISTIL.visualization.result_visualizer import (
    plot_confidence_histogram,
    plot_roc_curve,
    compare_clean_backdoor_models
)
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate backdoor detection on multiple models")
    parser.add_argument("--models_dir", type=str, required=True,
                        help="Path to the directory containing model folders")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to the metadata CSV file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_output",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_clean", type=int, default=5,
                        help="Number of clean models to evaluate")
    parser.add_argument("--num_backdoor", type=int, default=5,
                        help="Number of backdoored models to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for backdoor detection")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print(f"Loading models from {args.models_dir}")
    clean_models, backdoor_models = load_models(args.models_dir)
    
    # Limit the number of models to evaluate
    if args.num_clean < len(clean_models):
        clean_models = clean_models[:args.num_clean]
    
    if args.num_backdoor < len(backdoor_models):
        backdoor_models = backdoor_models[:args.num_backdoor]
    
    print(f"Selected {len(clean_models)} clean models and {len(backdoor_models)} backdoored models for evaluation")
    
    # Output file
    output_file = os.path.join(args.output_dir, "detection_results.txt")
    
    # Evaluate detection
    print("Starting evaluation...")
    results = evaluate_backdoor_detection(
        clean_models,
        backdoor_models,
        args.metadata_path,
        output_file
    )
    
    # Process results
    clean_results = {k: v for k, v in results.items() if k.startswith("clean_")}
    backdoor_results = {k: v for k, v in results.items() if k.startswith("backdoored_")}
    
    clean_scores = list(clean_results.values())
    backdoor_scores = list(backdoor_results.values())
    
    # Print summary statistics
    print("\nEvaluation complete!")
    print(f"Clean models - Mean score: {np.mean(clean_scores):.4f}, Std: {np.std(clean_scores):.4f}")
    print(f"Backdoored models - Mean score: {np.mean(backdoor_scores):.4f}, Std: {np.std(backdoor_scores):.4f}")
    
    # Calculate detection accuracy
    true_positives = sum(score > args.threshold for score in backdoor_scores)
    true_negatives = sum(score <= args.threshold for score in clean_scores)
    
    total_models = len(clean_scores) + len(backdoor_scores)
    accuracy = (true_positives + true_negatives) / total_models if total_models > 0 else 0
    
    # Calculate ROC AUC
    all_scores = clean_scores + backdoor_scores
    all_labels = [0] * len(clean_scores) + [1] * len(backdoor_scores)
    auc_score = roc_auc_score(all_labels, all_scores)
    
    print(f"\nDetection Accuracy: {accuracy * 100:.2f}%")
    print(f"True Positive Rate: {true_positives / len(backdoor_scores) * 100:.2f}% ({true_positives}/{len(backdoor_scores)})")
    print(f"True Negative Rate: {true_negatives / len(clean_scores) * 100:.2f}% ({true_negatives}/{len(clean_scores)})")
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Save statistics to file
    stats_file = os.path.join(args.output_dir, "detection_stats.txt")
    with open(stats_file, "w") as f:
        f.write(f"Clean models - Mean score: {np.mean(clean_scores):.4f}, Std: {np.std(clean_scores):.4f}\n")
        f.write(f"Backdoored models - Mean score: {np.mean(backdoor_scores):.4f}, Std: {np.std(backdoor_scores):.4f}\n")
        f.write(f"\nDetection Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"True Positive Rate: {true_positives / len(backdoor_scores) * 100:.2f}% ({true_positives}/{len(backdoor_scores)})\n")
        f.write(f"True Negative Rate: {true_negatives / len(clean_scores) * 100:.2f}% ({true_negatives}/{len(clean_scores)})\n")
        f.write(f"ROC AUC Score: {auc_score:.4f}\n")
    
    # Generate visualizations
    # Histogram
    hist_path = os.path.join(args.output_dir, "score_histogram.png")
    plot_confidence_histogram(clean_scores, backdoor_scores, args.threshold, hist_path)
    
    # ROC curve
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    roc_auc = plot_roc_curve(clean_scores, backdoor_scores, roc_path)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Comparison plot
    comparison_path = os.path.join(args.output_dir, "score_comparison.png")
    compare_clean_backdoor_models(clean_results, backdoor_results, comparison_path)
    
    print(f"\nEvaluation results and visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
   