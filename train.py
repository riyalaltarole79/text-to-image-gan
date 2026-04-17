"""
train.py
========
Main entry point for training the Text-to-Image GAN.

Examples
--------
# Standard training (64×64, default hyperparameters)
    python train.py

# Custom settings via CLI
    python train.py \
        --image_size 128 \
        --epochs 200 \
        --batch_size 32 \
        --lr_g 1e-4 \
        --lr_d 1e-4

# Resume from checkpoint
    python train.py --resume checkpoints/latest.pt
"""

import argparse
import logging
import os
import sys

# ── Ensure package root is on path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config, DataConfig, TextConfig, ModelConfig, TrainConfig
from src.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text-to-Image GAN on 102 Flowers")

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_root",      default="flower_dataset/102 flower/flowers")
    parser.add_argument("--cat_json",       default="flower_dataset/102 flower/cat_to_name.json")
    parser.add_argument("--captions_json",  default="data/captions.json")
    parser.add_argument("--image_size",     type=int,   default=64)
    parser.add_argument("--num_workers",    type=int,   default=4)

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--noise_dim",      type=int,   default=100)
    parser.add_argument("--ca_dim",         type=int,   default=128)
    parser.add_argument("--gf_dim",         type=int,   default=64)
    parser.add_argument("--df_dim",         type=int,   default=64)
    parser.add_argument("--text_proj_dim",  type=int,   default=256)

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=120)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr_g",           type=float, default=2e-4)
    parser.add_argument("--lr_d",           type=float, default=2e-4)
    parser.add_argument("--lambda_kl",      type=float, default=2.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument("--use_amp",        action="store_true", default=False)

    # ── Logging ───────────────────────────────────────────────────────────
    parser.add_argument("--log_dir",        default="logs")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir",     default="outputs")
    parser.add_argument("--save_every",     type=int,   default=10)
    parser.add_argument("--sample_every",   type=int,   default=5)

    return parser.parse_args()


def build_config(args) -> Config:
    return Config(
        data=DataConfig(
            data_root      = args.data_root,
            cat_json       = args.cat_json,
            captions_json  = args.captions_json,
            image_size     = args.image_size,
            num_workers    = args.num_workers,
        ),
        text=TextConfig(
            projected_dim  = args.text_proj_dim,
        ),
        model=ModelConfig(
            noise_dim = args.noise_dim,
            ca_dim    = args.ca_dim,
            gf_dim    = args.gf_dim,
            df_dim    = args.df_dim,
        ),
        train=TrainConfig(
            epochs         = args.epochs,
            batch_size     = args.batch_size,
            lr_g           = args.lr_g,
            lr_d           = args.lr_d,
            lambda_kl      = args.lambda_kl,
            seed           = args.seed,
            resume         = args.resume,
            use_amp        = args.use_amp,
            log_dir        = args.log_dir,
            checkpoint_dir = args.checkpoint_dir,
            output_dir     = args.output_dir,
            save_every     = args.save_every,
            sample_every   = args.sample_every,
        ),
    )


def preflight_checks(cfg: Config):
    """Verify all required files exist before training."""
    errors = []

    if not os.path.exists(cfg.data.data_root):
        errors.append(f"data_root not found: {cfg.data.data_root}")

    if not os.path.exists(cfg.data.captions_json):
        errors.append(
            f"captions_json not found: {cfg.data.captions_json}\n"
            f"  → Run:  python data/generate_captions.py  first."
        )

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    logger.info("Pre-flight checks passed ✓")


def main():
    args = parse_args()
    cfg  = build_config(args)

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(cfg.train.log_dir, "train.log")),
        ],
    )

    logger.info("=" * 60)
    logger.info("Text-to-Image GAN — 102 Flower Dataset")
    logger.info("=" * 60)
    logger.info(f"Image size  : {cfg.data.image_size}×{cfg.data.image_size}")
    logger.info(f"Epochs      : {cfg.train.epochs}")
    logger.info(f"Batch size  : {cfg.train.batch_size}")
    logger.info(f"Device      : {cfg.device}")
    logger.info(f"AMP         : {cfg.train.use_amp}")
    logger.info("=" * 60)

    preflight_checks(cfg)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
