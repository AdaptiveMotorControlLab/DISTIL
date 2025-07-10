import math
import os
import numpy as np
from sklearn.metrics import roc_auc_score

def load_results_and_compute_auc(file_paths):
    """
    Accepts a single path or a list of paths of the form
      /…/overall_trigger_summary_{ARCHITECTURE}.txt
    Infers ARCHITECTURE from each path using simple splits,
    computes its AUC-ROC, and returns the mean AUC across all valid files.
    """

    aucs = []
    for file_path in file_paths:

        # 1) Infer ARCHITECTURE via split on '/' then '_' then '.'
        basename = os.path.basename(file_path)                  # e.g. "overall_trigger_summary_frcnn.txt"
        arch = basename.split("_")[-1].split(".")[0].lower()    # -> "frcnn"

        # 2) Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except Exception as e:
            print(f"[ERROR] Could not open {file_path}: {e}")
            continue

        # 3) Parse & label
        data_lines = lines[1:]  # skip header
        states, scores = [], []
        for line in data_lines:
            cols = line.split('\t')
            if len(cols) < 3:
                continue

            state_str = cols[1].strip().lower()
            try:
                val = float(cols[2].strip())
            except ValueError:
                continue
            if math.isnan(val):
                continue

            if arch == 'ssd':
                label = 1 if state_str == 'backdoored' else 0
            elif arch == 'frcnn':
                label = 0 if state_str == 'backdoored' else 1
            else:
                # treat unknown arch same as 'ssd'
                label = 1 if state_str == 'backdoored' else 0

            states.append(label)
            scores.append(val)

        if not states:
            print(f"[WARN] No valid data in {file_path}, skipping.")
            continue

        # 4) Compute this file's AUC
        try:
            auc = roc_auc_score(states, scores)
            # print(f"{file_path} ({arch}) → AUC-ROC: {auc:.4f}")
            aucs.append(auc)
        except Exception as e:
            print(f"[ERROR] AUC computation failed for {file_path}: {e}")

    # 5) Compute & return mean AUC
    if not aucs:
        print("No valid AUCs computed.")
        return None

    final_auc = sum(aucs) / len(aucs)
    print(f"\nFinal AUC‑ROC: {final_auc:.4f}")
    return final_auc