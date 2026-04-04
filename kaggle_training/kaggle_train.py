"""
kaggle_train.py
===============
Kaggle-compatible Florence-2 fine-tuning script.

Usage (in a Kaggle notebook cell):
    %run kaggle_training/kaggle_train.py --config kaggle_training/config.yaml

Or equivalently:
    import subprocess
    subprocess.run(["python", "kaggle_training/kaggle_train.py",
                    "--config", "kaggle_training/config.yaml"], check=True)
"""

import argparse
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import glob

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(root: str, rel: str) -> str:
    """Join root + relative path; if rel is already absolute, return as-is."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(root, rel)


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the latest checkpoint in output_dir, or None."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.split("-")[-1])
    )
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Florence-2 Kaggle training launcher")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ------------------------------------------------------------------
    # 1. Resolve paths
    # ------------------------------------------------------------------
    data_root    = cfg["paths"]["data_root"]
    output_dir   = cfg["paths"]["output_dir"]
    img_dir      = _resolve(data_root, cfg["paths"]["img_dir"])
    combined_dir = _resolve(data_root, cfg["paths"]["combined_dir"])

    mask_dirs = {
        "instruments": _resolve(data_root, cfg["paths"]["instruments_mask_dir"]),
        "polyp":       _resolve(data_root, cfg["paths"]["polyp_mask_dir"]),
        "pseudo":      _resolve(data_root, cfg["paths"]["pseudo_mask_dir"]),
        "gradcam":     _resolve(data_root, cfg["paths"]["gradcam_mask_dir"]),
    }

    os.makedirs(output_dir, exist_ok=True)

    model_id_str  = cfg["model_id"]
    lora_rank     = cfg["training"]["lora_rank"]
    lora_alpha    = cfg["training"]["lora_alpha"]
    eval_training = cfg["training"]["eval_training"]
    multi_task    = cfg["training"]["multi_task"]

    # ------------------------------------------------------------------
    # 2. Weights & Biases
    # ------------------------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)

    if wandb_enabled:
        import wandb
        wandb_key = wandb_cfg.get("api_key") or os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        os.environ["WANDB_PROJECT"] = model_id_str
        os.environ["WANDB_DISABLED"] = "false"
        report_to = "wandb"
        print(f"[W&B] Logging to project: {model_id_str}")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"
        print("[W&B] Disabled.")

    # ------------------------------------------------------------------
    # 3. Add project root to sys.path so training_utils can be imported
    # ------------------------------------------------------------------
    # The repo root is detected by locating this file's parent's parent.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from training.training_utils import (
        FlorenceCollator,
        KvasirVQADataset,
        load_florence_model,
        get_florence2_lora_targets,
        apply_lora_config,
        show_random_samples,
    )
    from transformers import Trainer, TrainingArguments, TrainerCallback

    # ------------------------------------------------------------------
    # Linux malloc_trim callback — forces glibc to return free heap pages
    # to the OS after every N steps, preventing linear RSS growth.
    # ------------------------------------------------------------------
    class MallocTrimCallback(TrainerCallback):
        def __init__(self, every_n_steps: int = 5):
            self.every_n_steps = every_n_steps
            try:
                import ctypes
                self._libc = ctypes.CDLL("libc.so.6")
            except OSError:
                self._libc = None  # not on Linux, no-op

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.every_n_steps == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()        # release cached CUDA blocks → fixes RSS growth
                if self._libc is not None:
                    self._libc.malloc_trim(0)   # release CPU heap fragments

    # ------------------------------------------------------------------
    # tracemalloc callback — identifies WHERE the persistent 40MB/step
    # growth is coming from. Prints top 10 allocation sites every 50 steps.
    # Disable once the leak is identified.
    # ------------------------------------------------------------------
    import tracemalloc

    class TraceMallocCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0 and state.global_step > 0:
                snapshot = tracemalloc.take_snapshot()
                top = snapshot.statistics("lineno")
                print(f"\n[tracemalloc] Step {state.global_step} — top memory allocations:", flush=True)
                for stat in top[:10]:
                    print(f"  {stat}", flush=True)

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    print("[Model] Loading Florence-2-base ...")
    model, processor = load_florence_model("microsoft/Florence-2-base", model_adapters=None)
    target_modules, _ = get_florence2_lora_targets(model)
    model = apply_lora_config(model, target_modules, rank=lora_rank, alpha=lora_alpha)

    # ------------------------------------------------------------------
    # 5. Build datasets
    # ------------------------------------------------------------------
    ds_files = cfg["datasets"]

    def ds_path(key: str) -> str:
        return os.path.join(combined_dir, ds_files[key])

    print("[Data] Building VQA dataset ...")
    if eval_training:
        # Use train split only (for sanity / eval run)
        vqa_dataset = KvasirVQADataset(
            dataset=ds_path("vqa_train_parquet"),
            image_dir=img_dir,
            task="<MedVQA>",
            processor=processor,
        )
    else:
        # Combine train + test for final training
        df1 = pd.read_parquet(ds_path("vqa_train_parquet"))
        df2 = pd.read_parquet(ds_path("vqa_test_parquet"))
        vqa_combined = pd.concat([df1, df2], ignore_index=True)
        vqa_dataset = KvasirVQADataset(
            dataset=vqa_combined,
            image_dir=img_dir,
            task="<MedVQA>",
            processor=processor,
        )

    show_random_samples(vqa_dataset)

    if not multi_task:
        training_dataset = vqa_dataset
        print(f"[Data] Single-task VQA mode. Total samples: {len(training_dataset)}")
    else:
        print("[Data] Building multi-task segmentation datasets ...")

        instruments_dataset = KvasirVQADataset(
            dataset=ds_path("instruments_mask_phrases"),
            image_dir=img_dir,
            mask_dir=mask_dirs["instruments"],
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            tag="instrument_v2",
            processor=processor,
        )
        polyp_dataset = KvasirVQADataset(
            dataset=ds_path("polyps_mask_phrases"),
            image_dir=img_dir,
            mask_dir=mask_dirs["polyp"],
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            tag="polyp",
            processor=processor,
        )
        z_line_dataset = KvasirVQADataset(
            dataset=ds_path("zline_mask_phrases"),
            image_dir=img_dir,
            mask_dir=mask_dirs["pseudo"],
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            tag="z-line",
            processor=processor,
        )
        oesophagitis_dataset = KvasirVQADataset(
            dataset=ds_path("oesophagitis_mask_phrases"),
            image_dir=img_dir,
            mask_dir=mask_dirs["gradcam"],
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            tag="oesophagitis",
            processor=processor,
        )
        ulcerative_colitis_dataset = KvasirVQADataset(
            dataset=ds_path("ulcerative_colitis_phrases"),
            image_dir=img_dir,
            mask_dir=mask_dirs["gradcam"],
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            tag="ulcerative_colitis",
            processor=processor,
        )
        cecum_dataset = KvasirVQADataset(
            dataset=ds_path("cecum_mask_phrases"),
            image_dir=img_dir,
            mask_dir=mask_dirs["pseudo"],
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            tag="cecum",
            processor=processor,
        )
        vqa_exp_dataset = KvasirVQADataset(
            dataset=ds_path("vqa_exp_csv"),
            image_dir=img_dir,
            task="<MedVQA_EXPLAIN>",
            tag="v2",
            processor=processor,
        )

        for ds in [polyp_dataset, z_line_dataset, oesophagitis_dataset,
                   ulcerative_colitis_dataset, cecum_dataset,
                   vqa_exp_dataset, instruments_dataset]:
            show_random_samples(ds)

        training_dataset = torch.utils.data.ConcatDataset([
            instruments_dataset,
            polyp_dataset,
            z_line_dataset,
            oesophagitis_dataset,
            ulcerative_colitis_dataset,
            cecum_dataset,
            vqa_exp_dataset,
            vqa_dataset,
        ])
        print(f"[Data] Multi-task mode. Total samples: {len(training_dataset)}")

    # ------------------------------------------------------------------
    # 6. Resolve checkpoint (resume)
    # ------------------------------------------------------------------
    resume_cfg = cfg.get("resume", {})
    resume_enabled = resume_cfg.get("enabled", False)
    resume_from_checkpoint = None   # value passed to trainer.train()

    if resume_enabled:
        explicit_ckpt = resume_cfg.get("checkpoint_path", "").strip()
        if explicit_ckpt and os.path.isdir(explicit_ckpt):
            resume_from_checkpoint = explicit_ckpt
            print(f"[Resume] Resuming from explicit checkpoint: {resume_from_checkpoint}")
        else:
            latest = _find_latest_checkpoint(output_dir)
            if latest:
                resume_from_checkpoint = latest
                print(f"[Resume] Auto-detected latest checkpoint: {resume_from_checkpoint}")
            else:
                print("[Resume] resume.enabled=true but no checkpoint found. Starting fresh.")
    else:
        print("[Resume] Starting from scratch (resume.enabled=false).")

    # ------------------------------------------------------------------
    # 7. Training arguments
    # ------------------------------------------------------------------
    tr_cfg = cfg["training"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=tr_cfg.get("save_steps", 1000),
        save_total_limit=tr_cfg.get("save_total_limit", 2),
        logging_steps=tr_cfg.get("logging_steps", 5),
        per_device_train_batch_size=tr_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tr_cfg.get("gradient_accumulation_steps", 3),
        num_train_epochs=tr_cfg.get("num_train_epochs", 3),
        learning_rate=tr_cfg.get("learning_rate", 5e-5),
        warmup_ratio=tr_cfg.get("warmup_ratio", 0.1),
        fp16=tr_cfg.get("fp16", True),
        remove_unused_columns=False,
        report_to=report_to,
    )

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=FlorenceCollator(processor),
        tokenizer=processor.tokenizer,
        callbacks=[MallocTrimCallback(every_n_steps=5)],
    )

    print(f"\n{'='*60}")
    print(f"  Starting training: {model_id_str}")
    print(f"  Output dir:        {output_dir}")
    print(f"  Resume checkpoint: {resume_from_checkpoint}")
    print(f"{'='*60}\n")

    # Start tracemalloc HERE — after datasets are built, only tracks training
    #tracemalloc.start(3)  # nframes=3 keeps overhead low (10 was too slow)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("\n[Training] Done!")

    # ------------------------------------------------------------------
    # 9. Push to HuggingFace Hub (optional)
    # ------------------------------------------------------------------
    hf_cfg = cfg.get("huggingface", {})
    if hf_cfg.get("push_to_hub", False):
        from huggingface_hub import login, whoami

        hf_key = hf_cfg.get("api_key") or os.environ.get("HF_TOKEN", "")
        if not hf_key:
            print("[HF] push_to_hub=true but no HF API key found. Skipping.")
        else:
            login(token=hf_key)
            hf_user = whoami()["name"]
            print(f"[HF] Logged in as: {hf_user}")

            # Load the best/latest checkpoint to push
            latest_ckpt = _find_latest_checkpoint(output_dir)
            if latest_ckpt:
                final_model, final_processor = load_florence_model(
                    "microsoft/Florence-2-base",
                    model_adapters=latest_ckpt,
                )
            else:
                final_model = model

            hf_username = hf_cfg.get("username", hf_user)
            repo_id = f"{hf_username}/{model_id_str}"
            print(f"[HF] Pushing model to: {repo_id}")
            final_model.push_to_hub(repo_id)
            print(f"[HF] Upload complete: https://huggingface.co/{repo_id}")
    else:
        print("[HF] push_to_hub=false. Skipping upload.")

    print("\n[Done] kaggle_train.py finished successfully.")


if __name__ == "__main__":
    main()
