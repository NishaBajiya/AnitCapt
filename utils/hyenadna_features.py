"""
utils/hyenadna_features.py -- Shared embedding pipeline for BOTH models
(Dataset1 and Dataset2): fine-tuned HyenaDNA embeddings, mean-pooled,
matching the exact preprocessing/pooling used during fine-tuning and
embedding generation.

Paths point at:
    models/dataset1/finetuned_hyenadna/   -- Model 1's fine-tuned HyenaDNA checkpoint
    models/dataset2/finetuned_hyenadna/   -- Model 2's fine-tuned HyenaDNA checkpoint
(each should contain config.json, a weights file such as
model.safetensors or pytorch_model.bin, and the tokenizer files -- i.e.
whatever save_pretrained()/trainer.save_model() wrote out after
fine-tuning). If both datasets share one fine-tuned embedding model, just
copy/symlink the same checkpoint into both folders.

MAX_LENGTH = 128 to match the original fine-tuning script exactly --
do not change this unless your fine-tuning used a different value.
"""

import numpy as np
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FT_MODEL_DIR_1 = os.path.join(BASE_DIR, "models", "dataset1", "finetuned_hyenadna")
FT_MODEL_DIR_2 = os.path.join(BASE_DIR, "models", "dataset2", "finetuned_hyenadna")

MAX_LENGTH = 128     # must match the fine-tuning pipeline exactly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_loaded = {}   # cache: {model_dir: (tokenizer, model)} so each checkpoint loads only once


def clean_seq(seq):
    """Same cleaning as the original fine-tuning pipeline: upper-case,
    U->T (accepts RNA-style input), strip whitespace."""
    return str(seq).upper().replace("U", "T").strip()


def extract_hidden(outputs):
    """Extract last hidden state from a HF model output, matching the
    original embedding-generation script exactly."""
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _load_model(model_dir):
    if model_dir not in _loaded:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Fine-tuned HyenaDNA checkpoint directory not found: {model_dir}\n"
                f"Place your fine-tuned model files there (config.json, weights, "
                f"tokenizer files) -- see README.md for the expected layout."
            )
        print(f"Loading fine-tuned HyenaDNA checkpoint '{model_dir}' on {DEVICE}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        model.to(DEVICE)
        model.eval()
        _loaded[model_dir] = (tokenizer, model)
    return _loaded[model_dir]


@torch.no_grad()
def _embed_one(sequence, tokenizer, model, max_length=MAX_LENGTH):
    """
    Embeds exactly ONE sequence at a time -- deliberately NOT batched.

    HyenaDNA's custom modeling code (modeling_hyena.py) implements its
    long convolutions via FFT, which does not reliably support padded/
    batched inputs in the way standard attention-based transformers do.
    Batching sequences together (which requires padding shorter ones)
    was found to cause a segmentation fault. Processing one sequence at
    a time avoids padding entirely.
    """
    sequence = clean_seq(sequence)
    enc = tokenizer(
        [sequence], return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    outputs = model(**enc)
    hidden = extract_hidden(outputs)

    if "attention_mask" in enc and enc["attention_mask"] is not None:
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
    else:
        pooled = hidden.mean(dim=1)

    return pooled.float().cpu().numpy()


def extract_hyenadna_embeddings(sequences, model_choice=1, max_length=MAX_LENGTH):
    """
    sequences: list of raw DNA/RNA strings.
    model_choice: 1 or 2 -- selects which fine-tuned checkpoint to use.
    Returns: np.ndarray of shape (n_sequences, hidden_dim) -- mean-pooled
    embeddings, computed ONE SEQUENCE AT A TIME (see _embed_one's
    docstring for why batching is deliberately avoided here).
    """
    model_dir = FT_MODEL_DIR_1 if model_choice == 1 else FT_MODEL_DIR_2
    tokenizer, model = _load_model(model_dir)

    all_embeds = []
    total = len(sequences)
    for i, seq in enumerate(sequences, 1):
        if total > 10 and i % 10 == 0:
            print(f"  Embedding sequence {i}/{total}...")
        embedding = _embed_one(seq, tokenizer, model, max_length=max_length)
        all_embeds.append(embedding)

    return np.concatenate(all_embeds, axis=0)