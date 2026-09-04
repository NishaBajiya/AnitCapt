#!/usr/bin/env python3
"""
AntiCAPt: A method for predicting, designing and scanning anticancer
(AC) aptamers.

Developed in the style of Raghava-lab standalone tools (see e.g. AntiCP2,
IL6Pred at https://github.com/raghavagps). Web server companion:
https://webs.iiitd.edu.in/raghava/anticapt/index.php

Models: Two models are incorporated for predicting anticancer aptamers,
both using the SAME feature pipeline (fine-tuned HyenaDNA embeddings)
with different classifiers trained on different datasets:
    Model 1 (Dataset1, main / default): fine-tuned HyenaDNA embeddings + LightGBM
    Model 2 (Dataset2, alternate):      fine-tuned HyenaDNA embeddings + LightGBM
                                         (separate training data / scaler / classifier)

Modules/Jobs: This program implements three modules (job types):
    1. Predict: predicting anticancer-aptamer potential of input sequences.
    2. Design:  generating all single-point mutants of input sequences and
                computing their anticancer-aptamer potential (score).
    3. Scan:    creating all overlapping windows of a given length from
                input sequences and computing anticancer-aptamer potential
                of each window.

Minimum USAGE:
    python anticapt.py -i aptamer.fasta
This predicts the anticancer-aptamer potential of sequences in FASTA
format, using Model 1 by default, threshold 0.5, saving to outfile.csv.

Full USAGE:
    python anticapt.py [-h] -i INPUT [-o OUTPUT] [-j {1,2,3}] [-m {1,2}]
                        [-t THRESHOLD] [-w WINLENG] [-d {1,2}]

    -h, --help            show this help message and exit
    -i, --input           Input: DNA aptamer sequence(s) in FASTA format,
                           or one sequence per line (simple format)
    -o, --output          Output: file for saving results, default outfile.csv
    -j, --job             Job Type: 1:predict, 2:design, 3:scan; default 1
    -m, --model           Model: 1:Dataset1 (main), 2:Dataset2 (alternate);
                           both use fine-tuned HyenaDNA embeddings + LGBM;
                           default 1
    -t, --threshold       Threshold: value between 0 and 1, default 0.5
    -w, --winleng         Window Length: for scan mode only, default 30
    -d, --display         Display: 1:AC-aptamers only, 2:all sequences;
                           default 2

Output File: results saved in CSV format.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.seq_io import read_sequences, write_output
from utils.design_scan import generate_point_mutants, generate_scan_windows
from utils.hyenadna_features import extract_hyenadna_embeddings

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Per-dataset subfolders: models/dataset1/, models/dataset2/
# Each contains: finetuned_hyenadna/ (checkpoint dir), LGBM_model.joblib, LGBM_scaler.joblib
MODEL1_PATH = os.path.join(MODEL_DIR, "dataset1", "LGBM_model.joblib")
SCALER1_PATH = os.path.join(MODEL_DIR, "dataset1", "LGBM_scaler.joblib")
MODEL2_PATH = os.path.join(MODEL_DIR, "dataset2", "LGBM_model.joblib")
SCALER2_PATH = os.path.join(MODEL_DIR, "dataset2", "LGBM_scaler.joblib")


# ---------------------------------------------------------------------
# Argument parsing (mirrors Raghava-lab tool conventions, e.g. il6pred.py)
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="AntiCAPt: predicting, designing and scanning anticancer aptamers."
    )
    parser.add_argument("-i", "--input", required=True,
                         help="Input: DNA aptamer sequence(s) in FASTA format, or one "
                              "sequence per line (simple format)")
    parser.add_argument("-o", "--output", default="outfile.csv",
                         help="Output: file for saving results, by default outfile.csv")
    parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3], default=1,
                         help="Job Type: 1:predict, 2:design, 3:scan, by default 1")
    parser.add_argument("-m", "--model", type=int, choices=[1, 2], default=1,
                         help="Model: 1:Dataset1 (main), 2:Dataset2 (alternate); "
                              "both use fine-tuned HyenaDNA embeddings + LGBM, by default 1")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                         help="Threshold: value between 0 and 1, by default 0.5")
    parser.add_argument("-w", "--winleng", type=int, default=30,
                         help="Window Length: scan mode only, by default 30")
    parser.add_argument("-d", "--display", type=int, choices=[1, 2], default=2,
                         help="Display: 1:AC-aptamers only, 2:all sequences, by default 2")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Model + scaler loading (joblib)
# ---------------------------------------------------------------------
def load_model_and_scaler(model_choice):
    model_path = MODEL1_PATH if model_choice == 1 else MODEL2_PATH
    scaler_path = SCALER1_PATH if model_choice == 1 else SCALER2_PATH

    for path, label in [(model_path, "model"), (scaler_path, "scaler")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label.capitalize()} file not found: {path}\n"
                f"Place your saved joblib {label} at this path (see models/README.txt), "
                f"or edit the *_PATH constants at the top of anticapt.py."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_probabilities(sequences_with_ids, model_choice):
    """
    Computes fine-tuned HyenaDNA embeddings, then loads the classifier
    and scaler and performs prediction.
    """
    sequences = [seq for _, seq in sequences_with_ids]

    # IMPORTANT:
    # Initialize and run PyTorch/HyenaDNA BEFORE loading the joblib
    # LightGBM model. This avoids a native-library conflict on macOS.
    print("  Computing fine-tuned HyenaDNA embeddings...", flush=True)
    X = extract_hyenadna_embeddings(
        sequences,
        model_choice=model_choice
    )

    print("  Loading LightGBM model and scaler...", flush=True)
    model, scaler = load_model_and_scaler(model_choice)

    print("  Running LightGBM prediction...", flush=True)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled, num_threads=1)[:, 1]

    return probs


# ---------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------
def run_predict(sequences_with_ids, args):
    probs = predict_probabilities(sequences_with_ids, args.model)
    rows = []
    for (seq_id, seq), prob in zip(sequences_with_ids, probs):
        prediction = "AC-Aptamer" if prob >= args.threshold else "Non-AC-Aptamer"
        rows.append({
            "Sequence_ID": seq_id,
            "Sequence": seq,
            "Length": len(seq),
            "Score": round(float(prob), 4),
            "Prediction": prediction,
        })
    df = pd.DataFrame(rows)
    if args.display == 1:
        df = df[df["Prediction"] == "AC-Aptamer"].reset_index(drop=True)
    return df


def run_design(sequences_with_ids, args):
    all_mutants = []
    for seq_id, seq in sequences_with_ids:
        all_mutants.extend(generate_point_mutants(seq_id, seq))
    if not all_mutants:
        raise ValueError("No mutants generated -- check input sequences are non-empty.")

    mutant_records = [(m["Mutant_ID"], m["Mutant_Sequence"]) for m in all_mutants]
    probs = predict_probabilities(mutant_records, args.model)

    for m, prob in zip(all_mutants, probs):
        m["Score"] = round(float(prob), 4)
        m["Prediction"] = "AC-Aptamer" if prob >= args.threshold else "Non-AC-Aptamer"

    df = pd.DataFrame(all_mutants)
    df = df[["Sequence_ID", "Position", "Original_Base", "Mutant_Base",
             "Mutant_Sequence", "Score", "Prediction"]]
    df = df.sort_values(["Sequence_ID", "Score"], ascending=[True, False]).reset_index(drop=True)
    if args.display == 1:
        df = df[df["Prediction"] == "AC-Aptamer"].reset_index(drop=True)
    return df


def run_scan(sequences_with_ids, args):
    all_windows = []
    for seq_id, seq in sequences_with_ids:
        if len(seq) < args.winleng:
            print(f"  WARNING: sequence '{seq_id}' (length {len(seq)}) is shorter than "
                  f"window length {args.winleng} -- skipped.")
            continue
        all_windows.extend(generate_scan_windows(seq_id, seq, args.winleng))

    if not all_windows:
        raise ValueError(f"No windows generated -- all input sequences are shorter than "
                          f"the window length ({args.winleng}). Use -w to reduce it.")

    window_records = [(w["Window_ID"], w["Window_Sequence"]) for w in all_windows]
    probs = predict_probabilities(window_records, args.model)

    for w, prob in zip(all_windows, probs):
        w["Score"] = round(float(prob), 4)
        w["Prediction"] = "AC-Aptamer" if prob >= args.threshold else "Non-AC-Aptamer"

    df = pd.DataFrame(all_windows)
    df = df[["Sequence_ID", "Start_Position", "End_Position",
             "Window_Sequence", "Score", "Prediction"]]
    if args.display == 1:
        df = df[df["Prediction"] == "AC-Aptamer"].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        sys.exit("Error: --threshold must be between 0 and 1.")
    if args.job == 3 and args.winleng < 1:
        sys.exit("Error: --winleng must be a positive integer for scan mode.")

    print(f"Reading input sequences from {args.input} ...")
    sequences_with_ids = read_sequences(args.input)
    print(f"  {len(sequences_with_ids)} sequence(s) loaded.")

    model_label = f"Dataset{args.model} (fine-tuned HyenaDNA + LightGBM)"
    job_label = {1: "Predict", 2: "Design", 3: "Scan"}[args.job]
    print(f"Job: {job_label} | Model: {model_label} | Threshold: {args.threshold}")

    if args.job == 1:
        result_df = run_predict(sequences_with_ids, args)
    elif args.job == 2:
        result_df = run_design(sequences_with_ids, args)
    else:
        result_df = run_scan(sequences_with_ids, args)

    write_output(result_df, args.output)
    print(f"\n{len(result_df)} result row(s) written.")


if __name__ == "__main__":
    main()
