This directory is already fully populated with both models, ready to
use -- the layout below documents what's here now, and what to match if
you ever swap either model out for an updated version.

Directory layout (one subfolder per dataset):

models/
├── dataset1/
│   ├── finetuned_hyenadna/       <- fine-tuned HyenaDNA checkpoint files
│   │                                (config.json, model.safetensors or
│   │                                 pytorch_model.bin, tokenizer files --
│   │                                 i.e. everything save_pretrained()/
│   │                                 trainer.save_model() wrote out)
│   ├── LGBM_model.joblib          <- LightGBM classifier trained on Dataset1
│   └── LGBM_scaler.joblib         <- StandardScaler fit on Dataset1's
│                                      fine-tuned HyenaDNA embeddings
└── dataset2/
    ├── finetuned_hyenadna/       <- fine-tuned HyenaDNA checkpoint files
    ├── LGBM_model.joblib          <- LightGBM classifier trained on Dataset2
    └── LGBM_scaler.joblib         <- StandardScaler fit on Dataset2's
                                       fine-tuned HyenaDNA embeddings

If both datasets share ONE fine-tuned HyenaDNA embedding model, just copy
(or symlink) the same checkpoint folder into both
models/dataset1/finetuned_hyenadna/ and models/dataset2/finetuned_hyenadna/.

LGBM_model.joblib is expected to be a scikit-learn-API compatible object
exposing .predict_proba(X) -> array of shape (n_samples, 2), where
column 1 is the probability of the positive (anticancer-aptamer) class.

LGBM_scaler.joblib is expected to expose .transform(X), and must be the
SAME scaler fit during training -- never re-fit it at prediction time.

IMPORTANT: MAX_LENGTH in utils/hyenadna_features.py is set to 128 to
match the original fine-tuning pipeline. If your fine-tuning used a
different max_length, update it there -- a mismatch will silently
truncate/pad sequences differently than during training and degrade
prediction quality without any visible error.

VERSION PIN: LGBM_scaler.joblib was pickled with scikit-learn 1.7.2.
requirements.txt pins this exact version -- loading it with a different
scikit-learn version (tested: 1.8.0) still works but prints an
InconsistentVersionWarning; per scikit-learn's own documentation this can
occasionally cause silently different results, not just a cosmetic
warning, so don't loosen this pin without re-verifying predictions
against a known-good output first.
