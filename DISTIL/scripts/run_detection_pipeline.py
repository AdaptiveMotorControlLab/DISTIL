# scripts/run_detection_pipeline.py

import sys
import os
import argparse

# Add project root to Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs import detection_config as config
from detection.pipeline import run_pipeline
from evaluation.detection_analysis import load_results_and_compute_auc

def main():
    parser = argparse.ArgumentParser(description="Run the DISTIL detection pipeline.")
    parser.add_argument(
        '--arch',
        type=str,
        default=config.ARCHITECTURE,
        choices=['ssd', 'frcnn'],
        help=f"Model architecture to process (default: {config.ARCHITECTURE} from config)."
    )
    # Add other potential arguments here, e.g., overriding paths
    # parser.add_argument('--model-dir', type=str, help="Override MODEL_DATA_ROOT from config.")
    # parser.add_argument('--output-dir', type=str, help="Override OUTPUT_DIR_BASE from config.")

    args = parser.parse_args()

    selected_architecture = args.arch

    print(f"Using Device: {config.DEVICE}")
    print(f"Selected Architecture: {selected_architecture.upper()}")
    print(f"Model Data Root: {config.MODEL_DATA_ROOT}")
    print(f"Output Base Directory: {config.OUTPUT_DIR_BASE}")

    # --- Run the main pipeline ---
    run_pipeline(architecture=selected_architecture)

    # --- Compute and print AUC --- 
    summary_file = config.SUMMARY_FILE.replace(".txt", f"_{selected_architecture}.txt")
    if os.path.exists(summary_file):
        print(f"\n--- Calculating AUC from Summary: {summary_file} ---")
        load_results_and_compute_auc(summary_file, architecture=selected_architecture)
    else:
        print(f"\nSummary file not found, skipping AUC calculation: {summary_file}")

    print("\nScript finished.")

if __name__ == "__main__":
    main() 