"""
utils/design_scan.py -- Design (point-mutant generation) and Scan
(sliding-window) utilities for AntiCAPt, mirroring the "design"/"scan"
job types used across Raghava-lab standalone tools (e.g. AntiCP2, IL6Pred),
adapted here from amino-acid mutation/windowing to nucleotide sequences.

Design module: for every input sequence, generates every possible single-
point mutant (each position substituted with each of the 3 alternative
bases), so downstream prediction can rank which single substitutions most
improve (or most reduce) predicted anticancer-aptamer potential -- useful
for guided aptamer engineering.

Scan module: for every input sequence, generates all overlapping windows
of a fixed length, so downstream prediction can localize which sub-region
of a longer sequence carries the anticancer-aptamer signal.
"""

BASES = ["A", "C", "G", "T"]


def generate_point_mutants(seq_id, sequence):
    """
    Returns a list of dicts, one per single-point mutant:
        {Sequence_ID, Position (1-indexed), Original_Base, Mutant_Base,
         Mutant_ID, Mutant_Sequence}
    """
    mutants = []
    sequence = sequence.upper()
    for pos in range(len(sequence)):
        original_base = sequence[pos]
        for alt_base in BASES:
            if alt_base == original_base:
                continue
            mutant_seq = sequence[:pos] + alt_base + sequence[pos + 1:]
            mutants.append({
                "Sequence_ID": seq_id,
                "Position": pos + 1,  # 1-indexed for readability
                "Original_Base": original_base,
                "Mutant_Base": alt_base,
                "Mutant_ID": f"{seq_id}_pos{pos + 1}_{original_base}to{alt_base}",
                "Mutant_Sequence": mutant_seq,
            })
    return mutants


def generate_scan_windows(seq_id, sequence, window_length):
    """
    Returns a list of dicts, one per overlapping window of the given
    length:
        {Sequence_ID, Start_Position, End_Position (1-indexed, inclusive),
         Window_ID, Window_Sequence}

    Sequences shorter than window_length are skipped (with a warning
    printed by the caller), matching the lab-tool convention of ignoring
    sequences too short for the requested window.
    """
    windows = []
    sequence = sequence.upper()
    n = len(sequence)
    if n < window_length:
        return windows

    for start in range(0, n - window_length + 1):
        end = start + window_length  # exclusive in slicing, inclusive 1-indexed below
        window_seq = sequence[start:end]
        windows.append({
            "Sequence_ID": seq_id,
            "Start_Position": start + 1,
            "End_Position": end,
            "Window_ID": f"{seq_id}_win{start + 1}-{end}",
            "Window_Sequence": window_seq,
        })
    return windows
