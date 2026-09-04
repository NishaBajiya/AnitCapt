"""
utils/seq_io.py -- Sequence input/output utilities for AntiCAPt.

Supports two input formats (matching Raghava-lab tool conventions):
    1. FASTA format:
           >seq_id
           ACGTACGT...
    2. Simple format: one sequence per line, no headers. IDs are
       auto-generated as Seq_1, Seq_2, ...

Only A/C/G/T (case-insensitive) are treated as valid nucleotides; U is
accepted and converted to T (in case someone pastes an RNA sequence).
"""

import os
import re
import pandas as pd

VALID_BASES = set("ACGT")


def _clean_sequence(seq):
    seq = seq.strip().upper().replace("U", "T")
    return seq


def read_sequences(input_path):
    """
    Reads either FASTA or simple-format input. Returns a list of
    (seq_id, sequence) tuples, in file order.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path) as f:
        content = f.read()

    records = []
    if content.lstrip().startswith(">"):
        # FASTA format
        blocks = re.split(r"^>", content, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            seq_id = lines[0].strip().split()[0]  # first token of header line
            seq = _clean_sequence("".join(lines[1:]))
            if seq:
                records.append((seq_id, seq))
    else:
        # Simple format: one sequence per non-empty line
        idx = 1
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            seq = _clean_sequence(line)
            records.append((f"Seq_{idx}", seq))
            idx += 1

    if not records:
        raise ValueError(f"No sequences parsed from {input_path}. "
                          f"Check the file is valid FASTA or one-sequence-per-line format.")

    # Validate alphabet, warn (don't crash) on unexpected characters
    cleaned_records = []
    for seq_id, seq in records:
        invalid_chars = set(seq) - VALID_BASES
        if invalid_chars:
            print(f"  WARNING: sequence '{seq_id}' contains non-ACGT characters "
                  f"{sorted(invalid_chars)} -- they will be kept as-is, but this may "
                  f"affect feature computation and predictions.")
        cleaned_records.append((seq_id, seq))

    return cleaned_records


def write_output(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
