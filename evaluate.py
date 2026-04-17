"""
evaluate.py
===========
Evaluation metrics for the Text-to-Image GAN.

Metrics
-------
• FID  (Fréchet Inception Distance)   — lower is better
• IS   (Inception Score)              — higher is better
• CLIP score                          — semantic alignment (optional)
• Per-class diversity                 — average pairwise LPIPS distance

Usage
-----
    python evaluate.py \
        --checkpoint checkpoints/latest.pt \
        --embedder   checkpoints/embedder.pkl \
        --n_samples  1000 \
        --output     logs/eval_results.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

sys.path.insert(0, os.path.dirname(__file__))

from src.config import CFG
from src.inference import TextToImagePipeline
from src.dataset import FlowerCaptionDataset
from src.utils import denorm, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Inception helper ──────────────────────────────────────────────────────

class InceptionFeatureExtractor:
    """Extract pool3 features and softmax outputs from InceptionV3."""

    def __init__(self, device):
        self.device = device
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        # Replace fc head to get features
        self.features = []
        self.logits   = []

        # Hook for pool3 features
        def hook_fn(m, i, o):
            self.features.append(o.squeeze(-1).squeeze(-1).detach().cpu())

        model.avgpool.register_forward_hook(hook_fn)
        model.eval()
        self.model = model.to(device)
        self._resize = transforms.Resize((299, 299), antialias=True)

    @torch.no_grad()
    def get_features(self, images: torch.Tensor) -> np.ndarray:
        """
        images : [N, 3, H, W]  in [-1, 1]
        returns: [N, 2048]  pool3 features
        """
        imgs = denorm(images).clamp(0, 1)  # [0, 1]
        imgs = self._resize(imgs).to(self.device)
        self.features = []
        _ = self.model(imgs)
        return torch.cat(self.features, dim=0).numpy()

    @torch.no_grad()
    def get_logits(self, images: torch.Tensor) -> np.ndarray:
        imgs = denorm(images).clamp(0, 1)
        imgs = self._resize(imgs).to(self.device)
        out  = self.model(imgs)
        if hasattr(out, "logits"):
            out = out.logits
        return F.softmax(out, dim=1).cpu().numpy()


# ── FID ───────────────────────────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Compute FID between two sets of Inception pool3 features."""
    from scipy.linalg import sqrtm

    mu_r = real_feats.mean(0)
    mu_f = fake_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_f = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return fid


# ── IS ────────────────────────────────────────────────────────────────────

def compute_is(logits: np.ndarray, splits: int = 10) -> tuple:
    """Inception Score: E[KL(p(y|x) || p(y))]."""
    n    = len(logits)
    step = n // splits
    scores = []
    for i in range(splits):
        p_yx = logits[i * step: (i + 1) * step]
        p_y  = p_yx.mean(axis=0, keepdims=True)
        kl   = p_yx * (np.log(p_yx + 1e-8) - np.log(p_y + 1e-8))
        scores.append(np.exp(kl.sum(axis=1).mean()))
    mean, std = float(np.mean(scores)), float(np.std(scores))
    return mean, std


# ── Main evaluator ─────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, pipe: TextToImagePipeline, device: torch.device):
        self.pipe      = pipe
        self.device    = device
        self.inception = InceptionFeatureExtractor(device)

    def _collect_real_features(
        self,
        dataset: FlowerCaptionDataset,
        n: int = 1000,
        batch_size: int = 32,
    ) -> np.ndarray:
        from torch.utils.data import DataLoader
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        feats   = []
        total   = 0
        for batch in loader:
            imgs = batch["image"].to(self.device)
            f    = self.inception.get_features(imgs)
            feats.append(f)
            total += len(f)
            if total >= n:
                break
        return np.concatenate(feats)[:n]

    def _collect_fake_features(
        self,
        prompts: List[str],
        n_per: int = 10,
        batch_size: int = 16,
    ):
        all_feats  = []
        all_logits = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            imgs = self.pipe.generate(batch_prompts, n_per_text=n_per)
            # Batch Inception
            for j in range(0, len(imgs), 32):
                chunk = imgs[j: j + 32].to(self.device)
                all_feats.append(self.inception.get_features(chunk))
                all_logits.append(self.inception.get_logits(chunk))

        return np.concatenate(all_feats), np.concatenate(all_logits)

    def evaluate(
        self,
        dataset: FlowerCaptionDataset,
        prompts: List[str],
        n_real: int = 1000,
        n_per_prompt: int = 5,
    ) -> Dict:
        logger.info(f"Collecting real features ({n_real} images) …")
        real_feats = self._collect_real_features(dataset, n=n_real)

        logger.info(f"Generating fake images ({len(prompts) * n_per_prompt}) …")
        fake_feats, fake_logits = self._collect_fake_features(prompts, n_per=n_per_prompt)

        fid = compute_fid(real_feats, fake_feats)
        is_mean, is_std = compute_is(fake_logits)

        results = {
            "fid":            round(fid, 2),
            "is_mean":        round(is_mean, 4),
            "is_std":         round(is_std, 4),
            "n_real":         len(real_feats),
            "n_fake":         len(fake_feats),
            "n_prompts":      len(prompts),
        }
        return results


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-Image GAN")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--embedder",   default="checkpoints/embedder.pkl")
    parser.add_argument("--n_samples",  type=int, default=500)
    parser.add_argument("--n_per_prompt", type=int, default=4)
    parser.add_argument("--output",     default="logs/eval_results.json")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = CFG.device
    pipe   = TextToImagePipeline.from_checkpoint(args.checkpoint, args.embedder, CFG, device)

    dataset = FlowerCaptionDataset(
        data_root    = CFG.data.data_root,
        captions_json= CFG.data.captions_json,
        split        = "test",
        image_size   = CFG.data.image_size,
    )

    # Use a subset of unique captions as prompts
    import json
    with open(CFG.data.captions_json) as f:
        all_caps = json.load(f)
    prompts = []
    for caps in all_caps.values():
        prompts.extend(caps[:2])
    prompts = prompts[:args.n_samples // args.n_per_prompt]

    evaluator = Evaluator(pipe, device)
    results   = evaluator.evaluate(dataset, prompts, n_real=args.n_samples,
                                   n_per_prompt=args.n_per_prompt)

    logger.info("=" * 40)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 40)
    for k, v in results.items():
        logger.info(f"  {k:<20}: {v}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
