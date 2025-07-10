#!/usr/bin/env python
"""
Script to detect backdoors in a single model.
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from DISTIL.DISTIL.data_loader.model_loader import load_models
from DISTIL.DISTIL.detection.backdoor_detector import detect_backdoor
from DISTIL.DISTIL.utils.model_utils import plot_image_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Detect backdoors in a model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to the metadata CSV file")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output results")
    parser.add_argument("--guidance_scale", type=float, default=100.0,
                        help="Initial guidance scale for the diffusion model")
    parser.add_argument("--num_iterations", type=int, default=2,
                        help="Number of iterations to try different guidance scales")
    parser.add_argument("--timestep", type=int, default=50,
                        help="Number of timesteps for the diffusion model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    clean_models, backdoor_models = load_models(args.model_path)
    
    if args.metadata_path and len(backdoor_models) > 0:
        # If metadata is available and we found backdoored models, use them
        print("Detected backdoored model, using it for analysis...")
        model_data = backdoor_models[0]
        metadata_path = args.metadata_path
    else:
        # Otherwise use the first clean model (or whatever we found)
        models = clean_models + backdoor_models
        if len(models) == 0:
            raise ValueError("No models found in the specified path.")
        print("Using the first available model for analysis...")
        model_data = models[0]
        metadata_path = None
    
    # Detect backdoors
    print("Starting backdoor detection...")
    score, trigger = detect_backdoor(
        model_data,
        metadata_path=metadata_path,
        guidance_scale=args.guidance_scale,
        num_iterations=args.num_iterations,
        timestep=args.timestep
    )
    
    # Output results
    print(f"\nBackdoor detection score: {score:.4f}")
    if score > 0.5:
        print("RESULT: Model is likely backdoored.")
    else:
        print("RESULT: Model appears to be clean.")
    
    # Save the trigger image if one was found
    if trigger is not None:
        trigger_path = os.path.join(args.output_dir, f"trigger_{model_data['id']}.png")
        plt.figure()
        plot_image_tensor(trigger.cpu())
        plt.savefig(trigger_path)
        print(f"Trigger image saved to {trigger_path}")
    
    # Save score to a file
    score_path = os.path.join(args.output_dir, f"score_{model_data['id']}.txt")
    with open(score_path, "w") as f:
        f.write(f"Model ID: {model_data['id']}\n")
        f.write(f"Backdoor Score: {score:.4f}\n")
        f.write(f"Classification: {'Backdoored' if score > 0.5 else 'Clean'}\n")
    print(f"Score saved to {score_path}")

if __name__ == "__main__":
    main()