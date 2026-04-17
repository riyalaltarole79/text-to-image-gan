"""
inference.py
============
Text → Image inference pipeline.

Usage
-----
    from src.inference import TextToImagePipeline

    pipe = TextToImagePipeline.from_checkpoint(
        "checkpoints/latest.pt",
        "checkpoints/embedder.pkl",
    )
    images = pipe.generate(
        texts=["a vibrant red tulip in sunlight", "purple lavender in a meadow"],
        n_per_text=4,
    )
    pipe.show(images, texts=["red tulip", "lavender"])
    pipe.save(images, "outputs/result.png")
"""

import os
import logging
from typing import List, Optional, Union

import torch
from torchvision.utils import save_image

from src.config import Config, CFG
from src.generator import Generator
from src.text_embedder import TextEmbedder
from src.text_preprocessor import TextPreprocessor
from src.utils import denorm, save_sample_grid

logger = logging.getLogger(__name__)


class TextToImagePipeline:
    """
    End-to-end text-to-image generation.

    Parameters
    ----------
    G        : Generator
    embedder : TextEmbedder
    cfg      : Config
    device   : torch.device
    """

    def __init__(
        self,
        G:        Generator,
        embedder: TextEmbedder,
        cfg:      Config,
        device:   Optional[torch.device] = None,
    ):
        self.G        = G
        self.embedder = embedder
        self.cfg      = cfg
        self.device   = device or cfg.device
        self.preprocessor = TextPreprocessor(augment=False)

        self.G.eval()
        self.G.to(self.device)

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        embedder_path:   str,
        cfg:             Config = CFG,
        device:          Optional[torch.device] = None,
    ) -> "TextToImagePipeline":
        device = device or cfg.device
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device)

        G = Generator(
            noise_dim    = cfg.model.noise_dim,
            ca_dim       = cfg.model.ca_dim,
            gf_dim       = cfg.model.gf_dim,
            text_dim     = cfg.text.projected_dim,
            image_size   = cfg.data.image_size,
            use_instance = cfg.train.use_instance_norm,
        )
        G.load_state_dict(ckpt["G_state"])
        G.eval()

        embedder = TextEmbedder.load(embedder_path, device=device)

        logger.info(f"Pipeline ready  (epoch {ckpt.get('epoch', '?')})")
        return cls(G, embedder, cfg, device)

    # ── Core generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        texts:      List[str],
        n_per_text: int = 1,
        seed:       Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate images conditioned on text descriptions.

        Parameters
        ----------
        texts      : list of str   Input text descriptions.
        n_per_text : int           Number of images to generate per text.
        seed       : int or None   Fix RNG for reproducibility.

        Returns
        -------
        images : Tensor  [N * n_per_text, 3, H, W]  values in [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Preprocess text
        cleaned = [self.preprocessor.clean(t) for t in texts]

        # 2. Repeat each text n_per_text times
        repeated = [c for c in cleaned for _ in range(n_per_text)]

        # 3. Embed
        text_emb = self.embedder.embed(repeated).to(self.device)  # [N*n, 256]

        # 4. Generate
        noise = self.G.sample_noise(len(repeated), self.device)
        fake_imgs, _, _ = self.G(noise, text_emb)

        return fake_imgs.cpu()

    @torch.no_grad()
    def interpolate(
        self,
        text_a: str,
        text_b: str,
        steps:  int = 8,
    ) -> torch.Tensor:
        """
        Linearly interpolate between two text embeddings.

        Returns
        -------
        images : Tensor [steps, 3, H, W]
        """
        emb_a = self.embedder.embed([self.preprocessor.clean(text_a)]).to(self.device)
        emb_b = self.embedder.embed([self.preprocessor.clean(text_b)]).to(self.device)

        alphas = torch.linspace(0, 1, steps, device=self.device)
        interp_embs = torch.stack([
            (1 - a) * emb_a + a * emb_b for a in alphas
        ]).squeeze(1)  # [steps, 256]

        noise = self.G.sample_noise(steps, self.device)
        imgs, _, _ = self.G(noise, interp_embs)
        return imgs.cpu()

    # ── Save / show ────────────────────────────────────────────────────────

    def save(
        self,
        images:   torch.Tensor,
        path:     str,
        captions: Optional[List[str]] = None,
        nrow:     int = 4,
    ):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_sample_grid(images, path, nrow=nrow, captions=captions)
        logger.info(f"Saved → {path}")

    def show(self, images: torch.Tensor, texts: Optional[List[str]] = None):
        """Display generated images in a matplotlib window (if available)."""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        imgs = denorm(images).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        n    = len(imgs)
        nrow = min(4, n)
        ncol = (n + nrow - 1) // nrow
        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3.5))
        axes = axes.ravel() if n > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < n:
                ax.imshow(imgs[i])
                if texts and i < len(texts):
                    ax.set_title(texts[i], fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


# ── Demo runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    CKPT    = "checkpoints/latest.pt"
    EMB_PKL = "checkpoints/embedder.pkl"

    if not os.path.exists(CKPT):
        print(f"No checkpoint found at {CKPT}. Run train.py first.")
        sys.exit(0)

    pipe = TextToImagePipeline.from_checkpoint(CKPT, EMB_PKL)

    sample_texts = [
        "a vibrant red rose with soft petals in sunlight",
        "purple lavender in a meadow with green leaves",
        "a delicate white daisy with a golden center",
        "a beautiful pink primrose in full bloom",
    ]

    print("Generating images …")
    images = pipe.generate(sample_texts, n_per_text=2, seed=42)
    out_path = "outputs/inference_demo.png"
    pipe.save(images, out_path, captions=sample_texts * 2)
    print(f"Done → {out_path}")

    # Interpolation demo
    print("Generating interpolation …")
    interp = pipe.interpolate("red fire lily", "blue iris flower", steps=8)
    pipe.save(interp, "outputs/interpolation.png", nrow=8)
    print("Interpolation saved → outputs/interpolation.png")
