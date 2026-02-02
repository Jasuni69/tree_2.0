"""
Weekend training pipeline - runs full ensemble training on cleaned data.

1. Train 5 models with different seeds on CLEANED data
2. Compute embeddings from best model
3. Evaluate and compare

Run: python scripts/weekend_train.py
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import torch

PROJECT_DIR = Path(r'E:\tree_id_2.0')
VENV_PYTHON = str(PROJECT_DIR / 'venv' / 'Scripts' / 'python.exe')
LOG_FILE = PROJECT_DIR / 'weekend_training_log.txt'

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def kill_zombie_workers():
    """Kill any leftover Python worker processes to free RAM.
    Uses taskkill with PID exclusion to avoid killing ourselves."""
    import time
    my_pid = os.getpid()
    log(f"Cleaning up zombie workers (my PID: {my_pid})")

    # Get list of python PIDs, skip our own
    result = subprocess.run(
        'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH',
        shell=True, capture_output=True, text=True
    )
    killed = 0
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        try:
            parts = line.strip('"').split('","')
            pid = int(parts[1])
            if pid != my_pid:
                subprocess.run(f'taskkill /F /PID {pid}', shell=True,
                             capture_output=True)
                killed += 1
        except (ValueError, IndexError):
            pass

    if killed:
        log(f"Killed {killed} zombie Python processes")
    time.sleep(5)  # Let OS reclaim memory

def run_cmd(cmd, desc):
    log(f">>> {desc}")
    log(f"    {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR,
                           capture_output=True, text=True)
    if result.stdout:
        # Log last 50 lines of stdout
        lines = result.stdout.strip().split('\n')
        for line in lines[-50:]:
            log(f"  {line}")
    if result.returncode != 0:
        log(f"!!! FAILED: {desc}")
        if result.stderr:
            # Log last 20 lines of stderr
            err_lines = result.stderr.strip().split('\n')
            for line in err_lines[-20:]:
                log(f"  [ERR] {line}")
        return False
    log(f"<<< Done: {desc}")
    return True

def main():
    log("=" * 70)
    log("WEEKEND TRAINING - CLEANED DATA ENSEMBLE")
    log("=" * 70)

    seeds = [42, 123, 456, 789, 1337]

    # Step 1: Train 5 models on cleaned data
    log("\n" + "=" * 70)
    log("STEP 1: TRAINING 5 MODELS ON CLEANED DATA")
    log("=" * 70)

    for seed in seeds:
        log(f"\n--- Training seed {seed} ---")
        kill_zombie_workers()
        cmd = (
            f"\"{VENV_PYTHON}\" training/train.py "
            f"--seed {seed} "
            f"--epochs 30 "
            f"--backbone efficientnet_b2 "
            f"--batch_size 128 "
            f"--input training_data_cleaned.xlsx"
        )
        run_cmd(cmd, f"Train model seed {seed}")

    # Find best model
    log("\n--- Finding best model ---")
    best_acc = 0
    best_seed = 42
    results = {}

    for seed in seeds:
        model_path = PROJECT_DIR / 'models' / f'model_seed{seed}.pt'
        if model_path.exists():
            ckpt = torch.load(model_path, map_location='cpu')
            acc = ckpt.get('val_acc', 0)
            results[seed] = acc
            log(f"Seed {seed}: {acc*100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_seed = seed

    log(f"\nBest model: seed {best_seed} with {best_acc*100:.2f}%")

    # Step 2: Compute embeddings with best model
    log("\n" + "=" * 70)
    log("STEP 2: COMPUTING EMBEDDINGS WITH BEST MODEL")
    log("=" * 70)

    cmd = (
        f"\"{VENV_PYTHON}\" inference/compute_embeddings_v2.py "
        f"--model models/model_seed{best_seed}.pt "
        f"--output models/tree_embeddings_cleaned.pt "
        f"--prototypes 3 "
        f"--outlier_threshold 0.5 "
        f"--input training_data_cleaned.xlsx"
    )
    run_cmd(cmd, "Compute embeddings v2")

    # Step 3: Evaluate
    log("\n" + "=" * 70)
    log("STEP 3: EVALUATION")
    log("=" * 70)

    log("\n--- Evaluating on CLEANED data ---")
    cmd = (
        f"\"{VENV_PYTHON}\" inference/evaluate.py "
        f"--model models/model_seed{best_seed}.pt "
        f"--embeddings models/tree_embeddings_cleaned.pt "
        f"--input training_data_cleaned.xlsx"
    )
    run_cmd(cmd, "Evaluate on cleaned data")

    log("\n--- Evaluating on ORIGINAL data (comparison) ---")
    cmd = (
        f"\"{VENV_PYTHON}\" inference/evaluate.py "
        f"--model models/model_seed{best_seed}.pt "
        f"--embeddings models/tree_embeddings_cleaned.pt "
        f"--input training_data_with_ground_truth.xlsx"
    )
    run_cmd(cmd, "Evaluate on original data")

    # Summary
    log("\n" + "=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)
    log(f"Models trained: {seeds}")
    log(f"Best model: seed {best_seed} ({best_acc*100:.2f}%)")
    log(f"New embeddings: models/tree_embeddings_cleaned.pt")
    log(f"Full log: {LOG_FILE}")

    # Save summary JSON
    summary = {
        'completed': datetime.now().isoformat(),
        'seeds': seeds,
        'results': {str(k): v for k, v in results.items()},
        'best_seed': best_seed,
        'best_val_acc': best_acc,
        'input_file': 'training_data_cleaned.xlsx'
    }
    with open(PROJECT_DIR / 'weekend_training_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log("Results saved to weekend_training_results.json")

if __name__ == '__main__':
    main()
