"""
Script to load the trigger generation summary results and compute AUC-ROC score.
"""
import os
import sys
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import argparse

# --- Setup Project Root --- 
# Ensure the 'src' directory is in the Python path
if '__file__' in globals():
    project_root = Path(__file__).resolve().parent.parent
else:
    project_root = Path(os.getcwd()).resolve()
    
sys.path.insert(0, str(project_root))
print(f"Project root set to: {project_root}")

# --- Import Project Modules --- 
# Need ARCHITECTURE to determine label mapping
from src.config import ARCHITECTURE, SUMMARY_FILE 

def load_results_and_compute_auc(file_path: str, architecture: str):
    """
    Reads the summary file, parses results, computes AUC-ROC score for
    detecting backdoored models based on the average trigger evaluation score.

    Args:
        file_path (str): Path to the overall trigger summary file.
        architecture (str): The architecture ('ssd' or 'frcnn') used, which 
                           determines how 'backdoored' state maps to the binary label.
    """
    if not os.path.isfile(file_path):
        print(f"[ERROR] Summary file not found: {file_path}")
        return

    models = []
    states = []      # Ground truth labels (0 or 1)
    avg_scores = []  # Scores used for prediction

    print(f"Loading results from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"[ERROR] Could not read summary file: {e}")
        return

    if not lines or len(lines) < 2: # Check for header and at least one data line
        print("[ERROR] Summary file is empty or contains only a header.")
        return

    # Assuming header is the first line
    header = lines[0].strip()
    print(f"Header: {header}")
    data_lines = lines[1:]

    for i, line in enumerate(data_lines): 
        line = line.strip()
        if not line:
            continue # Skip empty lines
            
        cols = line.split('\t')
        if len(cols) < 3: # Expecting at least Model_ID, State, Avg_Score
            print(f"[Warning] Skipping malformed line {i+2}: {line}")
            continue
            
        model_id = cols[0].strip()
        state_str = cols[1].strip().lower()
        avg_str = cols[2].strip()
        
        # --- Determine Ground Truth Label (y_true) --- 
        # Based on the logic in the notebook:
        # - For SSD: backdoored=1, clean=0
        # - For FRCNN: backdoored=0, clean=1 (AUC seemed inverted)
        if state_str == 'backdoored':
            is_backdoored_label = 1 if architecture == 'ssd' else 0
        elif state_str == 'clean':
             is_backdoored_label = 0 if architecture == 'ssd' else 1
        else:
            print(f"[Warning] Unknown state '{cols[1]}' for model {model_id} on line {i+2}. Skipping.")
            continue
            
        # --- Parse Score (y_score) --- 
        try:
            avg_val = float(avg_str)
            # Handle NaN or infinite values potentially produced
            if math.isnan(avg_val) or math.isinf(avg_val):
                 print(f"[Warning] Invalid score (NaN/Inf: {avg_val}) for model {model_id} on line {i+2}. Skipping.")
                 continue
        except ValueError:
            print(f"[Warning] Could not parse score '{avg_str}' for model {model_id} on line {i+2}. Skipping.")
            continue
            
        # Append valid data
        models.append(model_id)
        states.append(is_backdoored_label)
        avg_scores.append(avg_val)

    # --- Compute AUC --- 
    if not states or not avg_scores:
        print("No valid model results found in the summary file to compute AUC.")
        return
        
    y_true = np.array(states, dtype=int)
    y_score = np.array(avg_scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        print(f"[Warning] Only one class found in y_true ({np.unique(y_true)}). AUC is not defined.")
        # Optionally print counts: print(f"Counts: {np.bincount(y_true)}")
        return
        
    try:
        auc = roc_auc_score(y_true, y_score)
        print("\n--- Results ---")
        print(f"Architecture: {architecture.upper()}")
        print(f"Models processed: {len(models)}")
        print(f"Label mapping: clean={1 if architecture=='frcnn' else 0}, backdoored={0 if architecture=='frcnn' else 1}")
        print(f"AUC-ROC Score: {auc:.6f}")
        print("---------------")
    except ValueError as e:
        print(f"[ERROR] Could not compute AUC-ROC: {e}")
        print("Ensure both classes (0 and 1) are present in the 'states' column.")
        print(f"Labels found: {np.unique(y_true)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load results from trigger generation and compute AUC-ROC.")
    parser.add_argument(
        "--summary_file", 
        type=str, 
        default=SUMMARY_FILE, 
        help=f"Path to the overall trigger summary file (default: {SUMMARY_FILE})"
    )
    parser.add_argument(
         "--arch",
         type=str,
         default=ARCHITECTURE,
         choices=['ssd', 'frcnn'],
         help=f"Model architecture used for generation (determines label mapping for AUC) (default: {ARCHITECTURE})"
    )

    args = parser.parse_args()

    load_results_and_compute_auc(args.summary_file, args.arch) 