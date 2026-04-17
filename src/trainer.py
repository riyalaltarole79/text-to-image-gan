"""
trainer.py
==========
Full GAN training loop for Text-to-Image synthesis.

Features
--------
• Generator + Discriminator alternating updates
• Conditioning Augmentation (CA) with KL regularisation
• Wrong-caption mismatch loss for better text–image alignment
• Learning-rate scheduling (linear decay after threshold epoch)
• Mixed-precision training (AMP) support
• TensorBoard + CSV logging
• Periodic sample-grid generation
• Checkpoint save / resume
• Optional FID computation every N epochs
"""

import csv
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image, make_grid

from src.config import Config
from src.generator import Generator
from src.discriminator import Discriminator, d_loss, g_loss
from src.text_embedder import TextEmbedder
from src.dataset import build_dataloader
from src.utils import (
    AverageMeter, set_seed, denorm,
    save_sample_grid, log_to_csv, EarlyStopping,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


class Trainer:
    """
    Orchestrates the full training pipeline.

    Parameters
    ----------
    cfg : Config   Full configuration object.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = cfg.device
        set_seed(cfg.train.seed)

        logger.info(f"Using device: {self.device}")

        # ── Text embedder ──────────────────────────────────────────────────
        self.embedder = TextEmbedder(cfg.text, device=self.device)

        # ── Data loaders ───────────────────────────────────────────────────
        logger.info("Building data loaders …")
        self.train_loader = build_dataloader(
            data_root       = cfg.data.data_root,
            captions_json   = cfg.data.captions_json,
            split           = cfg.data.train_split,
            image_size      = cfg.data.image_size,
            batch_size      = cfg.train.batch_size,
            num_workers     = cfg.data.num_workers,
            captions_per_image=cfg.data.captions_per_image,
            seed            = cfg.train.seed,
        )
        self.valid_loader = build_dataloader(
            data_root       = cfg.data.data_root,
            captions_json   = cfg.data.captions_json,
            split           = cfg.data.valid_split,
            image_size      = cfg.data.image_size,
            batch_size      = cfg.train.batch_size,
            num_workers     = cfg.data.num_workers,
            seed            = cfg.train.seed,
        )

        # ── Fit TF-IDF embedder on training captions ───────────────────────
        self._fit_embedder()

        # ── Models ────────────────────────────────────────────────────────
        self.G = Generator(
            noise_dim    = cfg.model.noise_dim,
            ca_dim       = cfg.model.ca_dim,
            gf_dim       = cfg.model.gf_dim,
            text_dim     = cfg.text.projected_dim,
            image_size   = cfg.data.image_size,
            use_instance = cfg.train.use_instance_norm,
        ).to(self.device)

        self.D = Discriminator(
            df_dim       = cfg.model.df_dim,
            text_dim     = cfg.text.projected_dim,
            image_size   = cfg.data.image_size,
            use_sn       = cfg.train.use_spectral_norm,
            dropout_rate = cfg.train.dropout_d,
        ).to(self.device)

        logger.info(f"G params: {sum(p.numel() for p in self.G.parameters()):,}")
        logger.info(f"D params: {sum(p.numel() for p in self.D.parameters()):,}")

        # ── Optimisers ────────────────────────────────────────────────────
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.train.lr_g, betas=(cfg.train.beta1, cfg.train.beta2),
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=cfg.train.lr_d, betas=(cfg.train.beta1, cfg.train.beta2),
        )

        # ── LR Schedulers ─────────────────────────────────────────────────
        def _make_scheduler(opt, start_epoch=cfg.train.lr_decay_epoch,
                            total=cfg.train.epochs,
                            gamma=cfg.train.lr_decay_gamma):
            def lr_lambda(epoch):
                if epoch < start_epoch:
                    return 1.0
                # Linear decay
                return max(0.0, 1.0 - (epoch - start_epoch) / (total - start_epoch))
            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        self.sched_G = _make_scheduler(self.opt_G)
        self.sched_D = _make_scheduler(self.opt_D)

        # ── Mixed precision ────────────────────────────────────────────────
        self.scaler = GradScaler() if (cfg.train.use_amp and self.device.type == "cuda") else None
        self.use_amp = self.scaler is not None

        # ── Logging ───────────────────────────────────────────────────────
        self.csv_path = os.path.join(cfg.train.log_dir, "training.csv")
        self._csv_headers_written = False

        # ── Fixed noise & captions for periodic sample grids ──────────────
        self.fixed_noise = torch.randn(
            cfg.train.num_samples, cfg.model.noise_dim, device=self.device
        )
        self.fixed_captions: Optional[torch.Tensor] = None  # set at first batch

        # ── State ─────────────────────────────────────────────────────────
        self.start_epoch = 0
        self.best_d_loss = float("inf")
        self.early_stop  = EarlyStopping(patience=20, min_delta=0.001)

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

    # ── TF-IDF fitting ─────────────────────────────────────────────────────

    def _fit_embedder(self):
        """Collect all training captions and fit the TF-IDF pipeline."""
        import json
        cap_path = self.cfg.data.captions_json
        if not os.path.exists(cap_path):
            logger.warning(f"Captions JSON not found: {cap_path}. Skipping TF-IDF fit.")
            return
        with open(cap_path) as f:
            all_caps = json.load(f)
        corpus = [cap for caps in all_caps.values() for cap in caps]
        self.embedder.fit(corpus)
        emb_path = os.path.join(self.cfg.train.checkpoint_dir, "embedder.pkl")
        self.embedder.save(emb_path)
        logger.info(f"Embedder fitted & saved → {emb_path}")

    # ── Embedding helper ────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed_captions(self, captions, wrong_captions):
        real_emb  = self.embedder.embed(list(captions)).to(self.device)
        wrong_emb = self.embedder.embed(list(wrong_captions)).to(self.device)
        return real_emb, wrong_emb

    # ── Train one epoch ─────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.G.train()
        self.D.train()

        meters = {
            "d_loss": AverageMeter(), "g_loss": AverageMeter(),
            "adv_g":  AverageMeter(), "kl":     AverageMeter(),
        }
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            real_img  = batch["image"].to(self.device)
            captions  = batch["caption"]
            wrong_caps= batch["wrong_caption"]
            B = real_img.size(0)

            with torch.no_grad():
                real_emb, wrong_emb = self._embed_captions(captions, wrong_caps)

            # Store fixed captions for sample grid (first batch, first epoch)
            if self.fixed_captions is None:
                n = self.cfg.train.num_samples
                self.fixed_captions = real_emb[:n].detach()

            noise = self.G.sample_noise(B, self.device)

            # ── Discriminator step ─────────────────────────────────────────
            self.opt_D.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    fake_img, mu, lv = self.G(noise, real_emb)
                    ld = d_loss(self.D, real_img, fake_img, real_emb, wrong_emb)
                self.scaler.scale(ld).backward()
                self.scaler.step(self.opt_D)
            else:
                fake_img, mu, lv = self.G(noise, real_emb)
                ld = d_loss(self.D, real_img, fake_img, real_emb, wrong_emb)
                ld.backward()
                self.opt_D.step()

            # ── Generator step ─────────────────────────────────────────────
            self.opt_G.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    fake_img, mu, lv = self.G(noise, real_emb)
                    lg, adv, kl = g_loss(
                        self.D, fake_img, real_emb, mu, lv, self.cfg.train.lambda_kl
                    )
                self.scaler.scale(lg).backward()
                nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=5.0)
                self.scaler.step(self.opt_G)
                self.scaler.update()
            else:
                fake_img, mu, lv = self.G(noise, real_emb)
                lg, adv, kl = g_loss(
                    self.D, fake_img, real_emb, mu, lv, self.cfg.train.lambda_kl
                )
                lg.backward()
                nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=5.0)
                self.opt_G.step()

            # ── Metrics ───────────────────────────────────────────────────
            meters["d_loss"].update(ld.item(), B)
            meters["g_loss"].update(lg.item(), B)
            meters["adv_g"].update(adv.item(), B)
            meters["kl"].update(kl.item(), B)

            if step % 50 == 0:
                lr_g = self.opt_G.param_groups[0]["lr"]
                lr_d = self.opt_D.param_groups[0]["lr"]
                logger.info(
                    f"Ep {epoch:3d} [{step:4d}/{len(self.train_loader)}] "
                    f"D={meters['d_loss'].avg:.4f}  G={meters['g_loss'].avg:.4f}  "
                    f"KL={meters['kl'].avg:.4f}  "
                    f"lr_G={lr_g:.6f}  lr_D={lr_d:.6f}  "
                    f"({time.time()-t0:.1f}s)"
                )

        return {k: v.avg for k, v in meters.items()}

    # ── Validation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.G.eval()
        self.D.eval()
        meters = {"val_d_loss": AverageMeter(), "val_g_loss": AverageMeter()}

        for batch in self.valid_loader:
            real_img   = batch["image"].to(self.device)
            captions   = batch["caption"]
            wrong_caps = batch["wrong_caption"]
            B = real_img.size(0)

            real_emb, wrong_emb = self._embed_captions(captions, wrong_caps)
            noise    = self.G.sample_noise(B, self.device)
            fake_img, mu, lv = self.G(noise, real_emb)

            ld = d_loss(self.D, real_img, fake_img, real_emb, wrong_emb)
            lg, _, _ = g_loss(self.D, fake_img, real_emb, mu, lv, self.cfg.train.lambda_kl)

            meters["val_d_loss"].update(ld.item(), B)
            meters["val_g_loss"].update(lg.item(), B)

        return {k: v.avg for k, v in meters.items()}

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        ckpt = {
            "epoch":         epoch,
            "G_state":       self.G.state_dict(),
            "D_state":       self.D.state_dict(),
            "opt_G":         self.opt_G.state_dict(),
            "opt_D":         self.opt_D.state_dict(),
            "sched_G":       self.sched_G.state_dict(),
            "sched_D":       self.sched_D.state_dict(),
            "metrics":       metrics,
            "fixed_captions":self.fixed_captions,
        }
        path = os.path.join(self.cfg.train.checkpoint_dir, f"ckpt_epoch{epoch:04d}.pt")
        torch.save(ckpt, path)
        # Also save "latest"
        latest = os.path.join(self.cfg.train.checkpoint_dir, "latest.pt")
        torch.save(ckpt, latest)
        logger.info(f"Checkpoint saved → {path}")

    def _load_checkpoint(self, path: str):
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.sched_G.load_state_dict(ckpt["sched_G"])
        self.sched_D.load_state_dict(ckpt["sched_D"])
        self.start_epoch     = ckpt["epoch"] + 1
        self.fixed_captions  = ckpt.get("fixed_captions")
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # ── Main train loop ─────────────────────────────────────────────────────

    def train(self):
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            # ── Train ───────────────────────────────────────────────────────
            train_metrics = self._train_epoch(epoch)
            # ── Validate ────────────────────────────────────────────────────
            val_metrics   = self._validate(epoch)
            # ── LR Step ─────────────────────────────────────────────────────
            self.sched_G.step()
            self.sched_D.step()

            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            logger.info(
                f"Epoch {epoch:3d}  "
                f"D={train_metrics['d_loss']:.4f}  "
                f"G={train_metrics['g_loss']:.4f}  "
                f"val_D={val_metrics['val_d_loss']:.4f}  "
                f"val_G={val_metrics['val_g_loss']:.4f}"
            )

            # ── Log to CSV ──────────────────────────────────────────────────
            log_to_csv(self.csv_path, all_metrics, write_header=not self._csv_headers_written)
            self._csv_headers_written = True

            # ── Sample grid ─────────────────────────────────────────────────
            if epoch % self.cfg.train.sample_every == 0:
                self._save_sample_grid(epoch)

            # ── Checkpoint ──────────────────────────────────────────────────
            if epoch % self.cfg.train.save_every == 0:
                self._save_checkpoint(epoch, all_metrics)

            # ── Early stopping ──────────────────────────────────────────────
            if self.early_stop(val_metrics["val_g_loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Final save
        self._save_checkpoint(epoch, all_metrics)
        logger.info("Training complete.")

    # ── Sample grid ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _save_sample_grid(self, epoch: int):
        self.G.eval()
        if self.fixed_captions is None:
            return
        n = min(self.cfg.train.num_samples, self.fixed_captions.size(0))
        fake = self.G.generate(self.fixed_captions[:n])
        grid_path = os.path.join(self.cfg.train.output_dir, f"samples_ep{epoch:04d}.png")
        save_image(denorm(fake), grid_path, nrow=4, normalize=False)
        logger.info(f"Sample grid → {grid_path}")
        self.G.train()
