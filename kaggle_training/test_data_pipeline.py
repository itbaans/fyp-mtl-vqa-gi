"""
test_data_pipeline.py
=====================
Lightweight pre-training data sanity checks.
No GPU / model load required — runs fast in a Kaggle notebook cell.

Usage:
    python kaggle_training/test_data_pipeline.py --config kaggle_training/config.yaml

In a Kaggle notebook cell:
    !python /kaggle/working/repo/kaggle_training/test_data_pipeline.py \
            --config /kaggle/working/repo/kaggle_training/config.yaml
"""

import argparse
import os
import sys
import re
import traceback
from typing import List, Tuple

import yaml

# ── ANSI colours ──────────────────────────────────────────────────────────────
OK   = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"
INFO = "\033[94mℹ️ \033[0m"
WARN = "\033[93m⚠️ \033[0m"

_pass_count = 0
_fail_count = 0


def _check(label: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    status = OK if condition else FAIL
    suffix = f"  → {detail}" if detail else ""
    print(f"  {status} {label}{suffix}")
    if condition:
        _pass_count += 1
    else:
        _fail_count += 1
    return condition


def _info(msg: str):
    print(f"  {INFO} {msg}")


def _warn(msg: str):
    print(f"  {WARN} {msg}")


def _section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Datasets with > this many samples will be checked on a random subset only.
# Smaller datasets (all seg datasets) are always fully checked.
SAMPLE_LIMIT = 300


def _resolve(root: str, rel: str) -> str:
    """Join root + relative path; if rel is already absolute, return as-is."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(root, rel)


def _is_kaggle() -> bool:
    """True when running inside a Kaggle notebook/kernel."""
    return os.path.isdir("/kaggle/input")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(cfg: dict) -> dict:
    """
    Return the appropriate paths block depending on the environment.
      On Kaggle  -> cfg['paths']
      Locally    -> cfg['local_paths']  (falls back to cfg['paths'] if missing)
    """
    if _is_kaggle():
        env = "Kaggle"
        p = cfg["paths"]
    else:
        env = "Local"
        p = cfg.get("local_paths", cfg["paths"])

    _info(f"Environment detected: {env}")
    _info(f"data_root = {p['data_root']}")
    return p


def _has_loc_tokens(s: str) -> bool:
    return bool(re.search(r"<loc_\d+>", s))


def _count_polygon_groups(answer: str) -> int:
    """
    Count the number of space-separated polygon/bbox groups in a coord string.
    Each group is a run of <loc_X><loc_Y> tokens joined without spaces;
    multiple objects are separated by a single space.
    """
    parts = answer.strip().split(" ")
    return len([p for p in parts if p])


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — File & Path Sanity
# ─────────────────────────────────────────────────────────────────────────────

def group1_paths(cfg: dict):
    _section("GROUP 1 — File & Path Sanity")

    p            = _resolve_paths(cfg)
    data_root    = p["data_root"]
    combined_dir = _resolve(data_root, p["combined_dir"])
    img_dir      = _resolve(data_root, p["img_dir"])

    _check("data_root exists",    os.path.isdir(data_root),    data_root)
    _check("combined_dir exists", os.path.isdir(combined_dir), combined_dir)
    _check("img_dir exists",      os.path.isdir(img_dir),      img_dir)

    if os.path.isdir(img_dir):
        jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        _check("img_dir has .jpg files", len(jpgs) > 0, f"{len(jpgs)} .jpg files found")

    ds = cfg["datasets"]
    for key, filename in ds.items():
        full = os.path.join(combined_dir, filename)
        _check(f"datasets.{key} exists", os.path.isfile(full), filename)

    mask_keys = [
        ("instruments_mask_dir", p["instruments_mask_dir"]),
        ("polyp_mask_dir",       p["polyp_mask_dir"]),
        ("pseudo_mask_dir",      p["pseudo_mask_dir"]),
        ("gradcam_mask_dir",     p["gradcam_mask_dir"]),
    ]
    for label, rel in mask_keys:
        full = _resolve(data_root, rel)
        _check(f"paths.{label} exists", os.path.isdir(full), full)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Dataset Loading  (all datasets)
# ─────────────────────────────────────────────────────────────────────────────

def _try_load(name, KvasirVQADataset, **kwargs):
    """Load a single KvasirVQADataset, return (dataset | None)."""
    try:
        ds = KvasirVQADataset(**kwargs)
        _check(f"{name} loads without error", True)
        _check(f"{name} len > 0", len(ds) > 0, f"{len(ds)} samples")
        return ds
    except Exception as e:
        _check(f"{name} loads without error", False, str(e))
        return None


def group2_loading(cfg: dict, KvasirVQADataset, processor_stub):
    _section("GROUP 2 — Dataset Loading")

    p            = _resolve_paths(cfg)
    data_root    = p["data_root"]
    combined_dir = _resolve(data_root, p["combined_dir"])
    img_dir      = _resolve(data_root, p["img_dir"])
    ds_files     = cfg["datasets"]

    def fp(key):
        return os.path.join(combined_dir, ds_files[key])

    import pandas as pd

    # ── VQA (combined train + test) ───────────────────────────────────────────
    vqa_ds = None
    try:
        df1 = pd.read_parquet(fp("vqa_train_parquet"))
        df2 = pd.read_parquet(fp("vqa_test_parquet"))
        vqa_df = pd.concat([df1, df2], ignore_index=True)
        vqa_ds = KvasirVQADataset(
            dataset=vqa_df, image_dir=img_dir,
            task="<MedVQA>", processor=processor_stub
        )
        _check("VQA dataset loads without error", True)
        _check("VQA dataset len > 0", len(vqa_ds) > 0, f"{len(vqa_ds)} samples")
    except Exception as e:
        _check("VQA dataset loads without error", False, str(e))

    # ── Segmentation datasets ─────────────────────────────────────────────────
    instruments_ds = _try_load(
        "Instruments", KvasirVQADataset,
        dataset=fp("instruments_mask_phrases"),
        image_dir=img_dir,
        mask_dir=_resolve(data_root, p["instruments_mask_dir"]),
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag="instrument_v2",
        processor=processor_stub,
    )
    polyp_ds = _try_load(
        "Polyp", KvasirVQADataset,
        dataset=fp("polyps_mask_phrases"),
        image_dir=img_dir,
        mask_dir=_resolve(data_root, p["polyp_mask_dir"]),
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag="polyp",
        processor=processor_stub,
    )
    zline_ds = _try_load(
        "Z-line", KvasirVQADataset,
        dataset=fp("zline_mask_phrases"),
        image_dir=img_dir,
        mask_dir=_resolve(data_root, p["pseudo_mask_dir"]),
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag="z-line",
        processor=processor_stub,
    )
    oeso_ds = _try_load(
        "Oesophagitis", KvasirVQADataset,
        dataset=fp("oesophagitis_mask_phrases"),
        image_dir=img_dir,
        mask_dir=_resolve(data_root, p["gradcam_mask_dir"]),
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag="oesophagitis",
        processor=processor_stub,
    )
    uc_ds = _try_load(
        "Ulcerative Colitis", KvasirVQADataset,
        dataset=fp("ulcerative_colitis_phrases"),
        image_dir=img_dir,
        mask_dir=_resolve(data_root, p["gradcam_mask_dir"]),
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag="ulcerative_colitis",
        processor=processor_stub,
    )
    cecum_ds = _try_load(
        "Cecum", KvasirVQADataset,
        dataset=fp("cecum_mask_phrases"),
        image_dir=img_dir,
        mask_dir=_resolve(data_root, p["pseudo_mask_dir"]),
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        tag="cecum",
        processor=processor_stub,
    )
    vqa_exp_ds = _try_load(
        "VQA-Explain", KvasirVQADataset,
        dataset=fp("vqa_exp_csv"),
        image_dir=img_dir,
        task="<MedVQA_EXPLAIN>",
        tag="v2",
        processor=processor_stub,
    )

    # ── __getitem__ on first sample check ─────────────────────────────────────
    all_datasets = [
        ("VQA",               vqa_ds),
        ("Instruments",       instruments_ds),
        ("Polyp",             polyp_ds),
        ("Z-line",            zline_ds),
        ("Oesophagitis",      oeso_ds),
        ("Ulcerative Colitis",uc_ds),
        ("Cecum",             cecum_ds),
        ("VQA-Explain",       vqa_exp_ds),
    ]
    for name, ds_obj in all_datasets:
        if ds_obj is None or len(ds_obj) == 0:
            _warn(f"Skipping __getitem__ test for {name} (not loaded / empty)")
            continue
        try:
            sample = ds_obj[0]
            has_keys = all(k in sample for k in ("prompt", "answer", "image"))
            _check(f"{name} dataset[0] has required keys", has_keys,
                   str(list(sample.keys())))
        except Exception as e:
            _check(f"{name} dataset[0] returns valid sample", False, str(e))

    seg_datasets = [
        ("Instruments",        instruments_ds,  True),
        ("Polyp",              polyp_ds,         True),
        ("Z-line",             zline_ds,         True),
        ("Oesophagitis",       oeso_ds,          True),
        ("Ulcerative Colitis", uc_ds,            True),
        ("Cecum",              cecum_ds,         True),
        ("VQA-Explain",        vqa_exp_ds,       False),
    ]

    return vqa_ds, seg_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Sample Integrity
# ─────────────────────────────────────────────────────────────────────────────

def _inspect_dataset(name: str, ds_obj, is_segmentation: bool = False):
    """
    Check sample integrity.
    • Datasets larger than SAMPLE_LIMIT → random sample of SAMPLE_LIMIT indices.
    • All others → full scan.
    This means VQA (159k) is sampled; all seg datasets (~1k-10k) are fully checked.
    """
    if ds_obj is None or len(ds_obj) == 0:
        _warn(f"Skipping sample integrity for {name} (not loaded / empty)")
        return

    import random as _random
    total = len(ds_obj)

    if total > SAMPLE_LIMIT:
        indices = _random.sample(range(total), SAMPLE_LIMIT)
        _info(f"[{name}] Sampling {SAMPLE_LIMIT} random samples (total={total})")
    else:
        indices = list(range(total))
        _info(f"[{name}] Full check — {total} samples")

    checked = len(indices)

    bad_image      = 0
    bad_prompt     = 0
    none_answer    = 0
    empty_answer   = 0
    no_loc_tokens  = 0
    multi_object   = 0
    multi_object_examples: List[Tuple[int, int]] = []

    from PIL import Image as PILImage

    for idx in indices:
        try:
            sample = ds_obj[idx]
        except Exception:
            bad_image += 1
            continue

        img = sample.get("image")
        if not isinstance(img, PILImage.Image) or img.mode != "RGB":
            bad_image += 1

        prompt = sample.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip() or not prompt.startswith("<"):
            bad_prompt += 1

        answer = sample.get("answer")
        if answer is None:
            none_answer += 1
            continue

        if not isinstance(answer, str):
            answer = str(answer)

        if answer == "":
            empty_answer += 1

        if is_segmentation:
            if not _has_loc_tokens(answer):
                no_loc_tokens += 1
            else:
                groups = _count_polygon_groups(answer)
                if groups > 1:
                    multi_object += 1
                    if len(multi_object_examples) < 3:
                        multi_object_examples.append((idx, groups))

    # ── Print results ─────────────────────────────────────────────────────────
    _check(f"[{name}] All images are RGB PIL",
           bad_image == 0, f"{bad_image}/{checked} failed")

    _check(f"[{name}] All prompts start with '<'",
           bad_prompt == 0, f"{bad_prompt}/{checked} failed")

    _check(f"[{name}] No None answers",
           none_answer == 0, f"{none_answer}/{checked} are None")

    _info(f"[{name}] Empty-string answers: {empty_answer}/{checked}")

    if is_segmentation:
        _check(f"[{name}] All seg answers have <loc_*> tokens",
               no_loc_tokens == 0,
               f"{no_loc_tokens}/{checked} missing loc tokens")

        if multi_object > 0:
            _info(f"[{name}] Multi-object samples (space-separated groups): "
                  f"{multi_object}/{checked}")
            for idx, groups in multi_object_examples:
                ans_preview = ds_obj[idx]["answer"][:80]
                _info(f"    └─ idx={idx}  groups={groups}  preview={ans_preview!r}")
        else:
            _info(f"[{name}] No multi-object answers found.")


def group3_sample_integrity(vqa_ds, seg_datasets):
    _section("GROUP 3 — Sample Integrity")

    # VQA: large → sampled
    _inspect_dataset("VQA", vqa_ds, is_segmentation=False)

    # All seg datasets: small → full check
    for name, ds_obj, is_seg in seg_datasets:
        _inspect_dataset(name, ds_obj, is_segmentation=is_seg)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — Collator Output
# ─────────────────────────────────────────────────────────────────────────────

def group4_collator(vqa_ds, FlorenceCollator, processor_stub):
    _section("GROUP 4 — Collator Output")

    if vqa_ds is None or len(vqa_ds) < 2:
        _warn("Skipping collator tests (VQA dataset not available or too small)")
        return

    try:
        import torch
        from PIL import Image as PILImage

        # Use synthetic answers of intentionally DIFFERENT lengths so padding
        # is guaranteed — this makes the -100 masking check reliable regardless
        # of which real samples happen to be picked.
        real_sample = vqa_ds[0]
        batch = [
            {
                "prompt":  real_sample["prompt"],
                "answer":  "Yes",                              # short answer
                "image":   real_sample["image"],
            },
            {
                "prompt":  real_sample["prompt"],
                "answer":  "This is a detailed description of the colonoscopy finding.",  # long answer
                "image":   real_sample["image"],
            },
        ]

        collator = FlorenceCollator(processor_stub)
        out = collator(batch)

        required_keys = {"input_ids", "attention_mask", "labels"}
        _check("Collator output has required keys",
               required_keys.issubset(out.keys()),
               str(list(out.keys())))

        if "input_ids" in out:
            shape = tuple(out["input_ids"].shape)
            _check("input_ids shape is (batch=2, seq_len)",
                   len(shape) == 2 and shape[0] == 2, str(shape))

        if "labels" in out:
            has_ignore = (out["labels"] == -100).any().item()
            _check("labels contain -100 (padding masked out)",
                   has_ignore,
                   f"-100 present: {has_ignore}  "
                   f"(if False → padding masking is broken in collate_fn)")

    except Exception as e:
        _check("Collator runs without error", False, str(e))
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Add repo root to sys.path so training_utils imports work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from training.training_utils import KvasirVQADataset, FlorenceCollator

    # ── Processor (only config files downloaded, no model weights) ────────────
    print("\n[Setup] Loading Florence-2 processor (no model weights) ...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        print("  Processor loaded.\n")
    except Exception as e:
        print(f"  {FAIL} Could not load processor: {e}")
        print("  Collator tests will be skipped.")
        processor = None

    # ── Run groups ────────────────────────────────────────────────────────────
    group1_paths(cfg)

    vqa_ds, seg_datasets = group2_loading(cfg, KvasirVQADataset, processor)

    group3_sample_integrity(vqa_ds, seg_datasets)

    if processor:
        group4_collator(vqa_ds, FlorenceCollator, processor)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = _pass_count + _fail_count
    print(f"\n{'='*60}")
    print(f"  Results: {_pass_count}/{total} passed   "
          f"{'🎉 All good!' if _fail_count == 0 else f'⚠️  {_fail_count} failed — fix before training'}")
    print(f"{'='*60}\n")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
