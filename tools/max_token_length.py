"""
For each segmentation dataset, find the sample with the longest combined
prompt+answer token sequence (as seen by the processor/collate_fn).

Run from project root:
    python tools/max_token_length.py
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

from transformers import AutoProcessor
from training.training_utils import KvasirVQADataset

# ── config ──────────────────────────────────────────────────────────────────
MODEL_ID    = "microsoft/Florence-2-base"
IMG_DIR     = "data/images"
DATASET_DIR = "data/combined"

SEG_DATASETS = [
    dict(name="instruments",        csv=f"{DATASET_DIR}/instruments_mask_phrases_v2.csv",
         mask_dir="data/instruments_masks",  tag="instrument_v2"),
    dict(name="polyp",              csv=f"{DATASET_DIR}/polyps_mask_phrases.csv",
         mask_dir="data/polyp_masks",        tag="polyp"),
    dict(name="z_line",             csv=f"{DATASET_DIR}/z-line_mask_phrases.csv",
         mask_dir="data/pseudo_masks",       tag="z-line"),
    dict(name="oesophagitis",       csv=f"{DATASET_DIR}/oesophatigis_mask_phrases.csv",
         mask_dir="data/gradcam_masks",      tag="oesophagitis"),
    dict(name="ulcerative_colitis", csv=f"{DATASET_DIR}/ulcerative_colitis_mask_phrases.csv",
         mask_dir="data/gradcam_masks",      tag="ulcerative_colitis"),
    dict(name="cecum",              csv=f"{DATASET_DIR}/cecum_mask_phrases.csv",
         mask_dir="data/pseudo_masks",       tag="cecum"),
]
# ────────────────────────────────────────────────────────────────────────────

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer


def token_len(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


print("=" * 70)
for cfg in SEG_DATASETS:
    print(f"\nLoading: {cfg['name']}  ({cfg['csv']})")
    ds = KvasirVQADataset(
        dataset=cfg["csv"],
        image_dir=IMG_DIR,
        mask_dir=cfg["mask_dir"],
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag=cfg["tag"],
        processor=processor,
    )
    print(f"  Samples after preprocessing: {len(ds)}")

    max_len      = -1
    max_idx      = -1
    max_prompt   = ""
    max_answer   = ""

    for idx in range(len(ds)):
        sample  = ds[idx]
        prompt  = sample["prompt"]
        answer  = sample["answer"]
        total   = token_len(prompt) + token_len(answer)

        if total > max_len:
            max_len    = total
            max_idx    = idx
            max_prompt = prompt
            max_answer = answer

    print(f"  Max combined token length : {max_len}")
    print(f"  Sample index              : {max_idx}")
    print(f"  Prompt  ({token_len(max_prompt):>5} tokens) : {max_prompt[:80]}")
    print(f"  Answer  ({token_len(max_answer):>5} tokens) : {max_answer[:120]}...")
    print("-" * 70)
