#!/usr/bin/env python3
"""
Main entry point script for running the backdoor detection pipeline.
"""

import os
import sys
import argparse
import torch

# Add the parent directory to sys.path for importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from the project
from DISTIL.DISTIL.detection.backdoor_detector_objDetection import main_detection_trigger_pipeline
from DISTIL.DISTIL.utils.metrics import load_results_and_compute_auc
from DISTIL.DISTIL.configs.config import DEFAULT_ROOT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backdoor detection for object detection models.")
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR,
                        help="Root directory containing model folders")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to store output triggers and reports")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="File path for detailed results")
    parser.add_argument("--architecture", type=str, default="ssd", choices=["ssd", "frcnn"],
                        help="Model architecture (ssd or frcnn)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--compute_auc", action="store_true", help="Compute AUC-ROC after detection")
    
    return parser.parse_args()

def main():
    """Main function to run backdoor detection."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run detection pipeline
    print(f"Starting backdoor detection on {args.architecture.upper()} models...")
    print(f"Root directory: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Detailed results will be saved to: {args.output_file}")
    
    results = main_detection_trigger_pipeline(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        output_file=args.output_file,
        architecture=args.architecture,
        device=device
    )
    
    # Compute AUC-ROC if requested
    if args.compute_auc and results:
        summary_file = os.path.join(args.output_dir, "overall_trigger_summary.txt")
        if os.path.exists(summary_file):
            print("\nComputing AUC-ROC score...")
            auc = load_results_and_compute_auc(summary_file, args.architecture)
            print(f"Final AUC-ROC: {auc:.4f}")
        else:
            print(f"Summary file not found: {summary_file}")
    
    print("\nDetection process completed.")

if __name__ == "__main__":
    main() 