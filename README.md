# AntiCapt: A LLM method for predicting, designing and scanning anticancer aptamers

## Introduction

AntiCapt is developed for predicting, designing, and scanning anticancer
(AC) ssDNA aptamers. Two models are incorporated -- both use the SAME
feature pipeline (fine-tuned HyenaDNA embeddings), with different
classifiers trained on different datasets:

- **Model 1 (Dataset1, main / default):** fine-tuned HyenaDNA embeddings + LightGBM
- **Model 2 (Dataset2, alternate):** fine-tuned HyenaDNA embeddings + LightGBM
  (separate training data, scaler, and classifier)

**Modules/Jobs:** This program implements three modules (job types):

1. **Predict** -- predicting anticancer-aptamer potential of input ssDNA sequences.
2. **Design** -- generating all single-point mutants (every position x
   every alternative base) of input sequences and computing the
   anticancer-aptamer potential (score) of each mutant. Useful for
   identifying which single substitutions most improve predicted potency.
3. **Scan** -- creating all overlapping windows of a given length from
   longer input nucleotide sequences and computing the anticancer-aptamer potential
   of each window. Useful for identifying the active region of a longer
   nucleotide sequence.

## Installation

```bash
git clone <this-repo>
cd anticapt_standalone
pip install -r requirements.txt
```

Both models (Dataset1 and Dataset2 -- fine-tuned HyenaDNA checkpoints,
LightGBM classifiers, and scalers) are already included under `models/`
and ready to use immediately -- no additional setup needed. The layout is:

```
models/
├── dataset1/
│   ├── finetuned_hyenadna/    <- fine-tuned HyenaDNA checkpoint
│   │                             (config.json, weights, tokenizer files)
│   ├── LGBM_model.joblib
│   └── LGBM_scaler.joblib
└── dataset2/
    ├── finetuned_hyenadna/    <- fine-tuned HyenaDNA checkpoint
    ├── LGBM_model.joblib
    └── LGBM_scaler.joblib
```

If you later want to update either model, see `models/README.txt` for
the layout requirements. If both datasets should share one fine-tuned
embedding model, copy/symlink it into both `finetuned_hyenadna/` folders.

`MAX_LENGTH` in `utils/hyenadna_features.py` is set to **128**, matching
the fine-tuning pipeline, this was verified against -- change it there if
yours differs.

## Minimum Usage

```bash
python anticapt.py -i example/aptamer.fasta
```

This predicts the anticancer-aptamer potential of sequences in FASTA
format, using Model 1 by default, threshold 0.5. Output saved to
`outfile.csv`.

## Full Usage

```
anticapt.py [-h] -i INPUT [-o OUTPUT] [-j {1,2,3}] [-m {1,2}]
            [-t THRESHOLD] [-w WINLENG] [-d {1,2}]

optional arguments:
  -h, --help            show this help message and exit
  -i, --input           Input: DNA aptamer sequence(s) in FASTA format,
                         or one sequence per line (simple format)
  -o, --output          Output: file for saving results, default outfile.csv
  -j, --job {1,2,3}     Job Type: 1:predict, 2:design, 3:scan, default 1
  -m, --model {1,2}     Model: 1:Dataset1 (main), 2:Dataset2 (alternate);
                         both use fine-tuned HyenaDNA embeddings + LGBM,
                         default 1
  -t, --threshold       Threshold: value between 0 and 1, default 0.5
  -w, --winleng         Window Length: scan mode only, default 30
  -d, --display {1,2}   Display: 1:AC-aptamers only, 2:all sequences,
                         default 2
```

## Examples

Predict, using Dataset2's model, only showing positives:
```bash
python anticapt.py -i example/aptamer.fasta -m 2 -d 1 -o predict_results.csv
```

Design mutants of your sequences and rank by predicted score:
```bash
python anticapt.py -i example/aptamer.fasta -j 2 -o design_results.csv
```

Scan a long sequence with a 25 nt window:
```bash
python anticapt.py -i long_sequence.fasta -j 3 -w 25 -o scan_results.csv
```

## Input File

Two formats accepted:
1. **FASTA format** (standard) -- `>header` line followed by sequence.
2. **Simple format** -- one sequence per line, no headers; IDs are
   auto-generated as `Seq_1`, `Seq_2`, ...

Non-ACGT characters trigger a warning but are not stripped automatically
(Check your sequences if you see this warning -- it usually means stray
whitespace, ambiguity codes, or an accidental protein/RNA sequence).

## Output File

Results are saved in CSV format. Column sets differ slightly by job type:

- **Predict:** `Sequence_ID, Sequence, Length, Score, Prediction`
- **Design:** `Sequence_ID, Position, Original_Base, Mutant_Base, Mutant_Sequence, Score, Prediction`
- **Scan:** `Sequence_ID, Start_Position, End_Position, Window_Sequence, Score, Prediction`

`Score` is the predicted probability (0 to 1) of the anticancer-aptamer
class; `Prediction` is the thresholded call at `--threshold`.

## Package Files

```
anticapt.py                          : Main CLI program (predict/design/scan)
requirements.txt                     : Python dependencies
LICENSE                              : GNU GPL v3.0 license text
CITATION.cff                         : Citation metadata (update once published)
models/
    README.txt                        : Expected model directory layout (for future updates)
    dataset1/
        finetuned_hyenadna/            : Model 1's fine-tuned HyenaDNA checkpoint
        LGBM_model.joblib              : Model 1 classifier
        LGBM_scaler.joblib             : Model 1 scaler
    dataset2/
        finetuned_hyenadna/            : Model 2's fine-tuned HyenaDNA checkpoint
        LGBM_model.joblib              : Model 2 classifier
        LGBM_scaler.joblib             : Model 2 scaler
utils/
    seq_io.py                        : FASTA/simple-format reading, output writing
    design_scan.py                   : Point-mutant and sliding-window generation
    hyenadna_features.py             : Fine-tuned HyenaDNA embedding extraction (both models)
example/
    aptamer.fasta                    : Example input file
```

## Citation

Manuscript in preparation.
## Web Server

It is available as a user-friendly, open-access web server at https://webs.iiitd.edu.in/raghava/anticapt/

## Address for Contact

In case of any query, please contact:
```
Prof. G. P. S. Raghava, Professor, Department of Computational Biology,
Indraprastha Institute of Information Technology (IIIT),
Okhla Phase III, New Delhi 110020; Phone: +91-11-26907444;
Email: raghava@iiitd.ac.in  Web: http://webs.iiitd.edu.in/raghava/
```
