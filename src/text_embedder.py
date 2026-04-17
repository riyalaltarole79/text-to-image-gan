"""
text_embedder.py
================
Dual-channel text embedding:
  1. Sentence-BERT   → dense semantic embeddings (384-d or 768-d)
  2. TF-IDF          → sparse lexical signal projected to 128-d

Both channels are L2-normalised and optionally concatenated, then
projected through a linear layer to `projected_dim` (default 256-d).

Usage
-----
    embedder = TextEmbedder(cfg.text)
    embedder.fit(corpus)                     # fit TF-IDF
    emb = embedder.embed(["red rose"])       # Tensor [1, 256]
    embedder.save("checkpoints/embedder.pkl")
    embedder = TextEmbedder.load("checkpoints/embedder.pkl")
"""

import os
import pickle
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)


# ── Lightweight projection MLP ─────────────────────────────────────────────
class EmbeddingProjector(nn.Module):
    """
    Projects concatenated [SBERT | TF-IDF] → projected_dim.
    Architecture: Linear → LayerNorm → GELU → Linear → L2-norm
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, dim=-1)


# ── Main TextEmbedder class ─────────────────────────────────────────────────
class TextEmbedder:
    """
    Parameters
    ----------
    cfg : TextConfig
        Configuration dataclass (src/config.py).
    device : torch.device
        'cpu' or 'cuda'.
    """

    def __init__(self, cfg, device: Optional[torch.device] = None):
        self.cfg    = cfg
        self.device = device or torch.device("cpu")
        self._sbert  = None
        self._tfidf_pipeline = None
        self._projector: Optional[EmbeddingProjector] = None
        self._fitted = False

        # Derive dimensions
        self.sbert_dim   = cfg.raw_embed_dim          # e.g. 384
        self.tfidf_dim   = cfg.tfidf_dim if cfg.use_tfidf_augment else 0
        self.concat_dim  = self.sbert_dim + self.tfidf_dim
        self.proj_dim    = cfg.projected_dim           # e.g. 256

        self._load_sbert()
        self._build_projector()

    # ── SBERT ───────────────────────────────────────────────────────────────

    def _load_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading Sentence-BERT: {self.cfg.sbert_model}")
            self._sbert = SentenceTransformer(self.cfg.sbert_model)
            self._sbert.to(self.device)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to random embeddings (install for real use)."
            )
            self._sbert = None

    def _sbert_encode(self, texts: List[str]) -> np.ndarray:
        if self._sbert is not None:
            embs = self._sbert.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embs.astype(np.float32)                    # [N, sbert_dim]
        # Fallback: deterministic pseudo-random from text hash
        rng  = np.random.RandomState(42)
        embs = np.stack([
            rng.randn(self.sbert_dim).astype(np.float32) / np.sqrt(self.sbert_dim)
            for _ in texts
        ])
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return embs / norms

    # ── TF-IDF ──────────────────────────────────────────────────────────────

    def _build_tfidf_pipeline(self) -> Pipeline:
        """
        TF-IDF → TruncatedSVD (LSA) → L2 normalise.
        Output: [N, tfidf_dim]
        """
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=10_000,
            )),
            ("svd",   TruncatedSVD(n_components=self.tfidf_dim, random_state=42)),
            ("norm",  Normalizer(copy=False)),
        ])

    def fit(self, corpus: List[str]):
        """Fit TF-IDF pipeline on the training corpus."""
        if not self.cfg.use_tfidf_augment:
            self._fitted = True
            return self
        logger.info(f"Fitting TF-IDF on {len(corpus)} captions …")
        self._tfidf_pipeline = self._build_tfidf_pipeline()
        self._tfidf_pipeline.fit(corpus)
        self._fitted = True
        logger.info("TF-IDF fitting complete.")
        return self

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        if not self.cfg.use_tfidf_augment or self._tfidf_pipeline is None:
            return np.zeros((len(texts), 0), dtype=np.float32)
        return self._tfidf_pipeline.transform(texts).astype(np.float32)

    # ── Projector ───────────────────────────────────────────────────────────

    def _build_projector(self):
        self._projector = EmbeddingProjector(
            in_dim=self.concat_dim,
            out_dim=self.proj_dim,
        ).to(self.device)
        # Initialise weights
        for m in self._projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Embed a list of strings → Tensor [N, proj_dim].

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        torch.Tensor  shape [N, projected_dim]
        """
        sbert_emb = self._sbert_encode(texts)        # [N, sbert_dim]
        tfidf_emb = self._tfidf_encode(texts)        # [N, tfidf_dim] or [N,0]

        if self.cfg.use_tfidf_augment and tfidf_emb.shape[1] > 0:
            concat = np.concatenate([sbert_emb, tfidf_emb], axis=1)
        else:
            concat = sbert_emb                        # [N, sbert_dim]

        tensor = torch.tensor(concat, dtype=torch.float32, device=self.device)

        # If concat_dim != proj_dim, project; otherwise just normalise
        if self.concat_dim != self.proj_dim:
            projected = self._projector(tensor)
        else:
            projected = F.normalize(tensor, dim=-1)

        return projected

    def embed_single(self, text: str) -> torch.Tensor:
        """Convenience: embed one string → Tensor [1, proj_dim]."""
        return self.embed([text])

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "cfg":              self.cfg,
            "tfidf_pipeline":  self._tfidf_pipeline,
            "projector_state": self._projector.state_dict(),
            "fitted":          self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"TextEmbedder saved → {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "TextEmbedder":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        embedder = cls(payload["cfg"], device=device)
        embedder._tfidf_pipeline = payload["tfidf_pipeline"]
        embedder._projector.load_state_dict(payload["projector_state"])
        embedder._fitted = payload["fitted"]
        return embedder

    # ── Utility ─────────────────────────────────────────────────────────────

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text embeddings."""
        ea = self.embed_single(text_a)
        eb = self.embed_single(text_b)
        return float(F.cosine_similarity(ea, eb).item())

    def get_output_dim(self) -> int:
        return self.proj_dim


# ── Conditioning Augmentation module (used inside Generator) ────────────────
class ConditioningAugmentation(nn.Module):
    """
    Converts a fixed text embedding into a stochastic conditioning vector
    using the re-parametrisation trick (μ, log σ²).

    This stabilises training and enables a richer conditioning manifold.
    (From Zhang et al., "StackGAN", 2017)

    Parameters
    ----------
    text_dim : int   Dimensionality of the incoming text embedding.
    ca_dim   : int   Dimensionality of the output conditioning vector.
    """

    def __init__(self, text_dim: int, ca_dim: int):
        super().__init__()
        self.fc = nn.Linear(text_dim, ca_dim * 2, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        text_emb : Tensor [B, text_dim]

        Returns
        -------
        c_hat  : Tensor [B, ca_dim]   — sampled conditioning vector
        mu     : Tensor [B, ca_dim]
        log_var: Tensor [B, ca_dim]
        """
        out    = self.fc(text_emb)
        mu, log_var = out.chunk(2, dim=-1)
        # Clamp log_var for numerical stability
        log_var = log_var.clamp(-4, 4)
        if self.training:
            std  = (0.5 * log_var).exp()
            eps  = torch.randn_like(std)
            c_hat = mu + eps * std
        else:
            c_hat = mu
        return c_hat, mu, log_var


def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence: 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)."""
    return 0.5 * torch.mean(log_var.exp() + mu.pow(2) - 1.0 - log_var)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.config import CFG

    emb = TextEmbedder(CFG.text)
    sample_texts = [
        "a beautiful red rose with soft petals in sunlight",
        "purple lavender in a meadow",
        "white daisy with a golden center",
    ]
    emb.fit(sample_texts)
    result = emb.embed(sample_texts)
    print(f"Embedding shape: {result.shape}")   # [3, 256]
    print(f"Similarity (rose vs daisy): {emb.similarity(sample_texts[0], sample_texts[2]):.4f}")

    # CA module test
    ca  = ConditioningAugmentation(text_dim=256, ca_dim=128)
    c, mu, lv = ca(result)
    print(f"CA output shape: {c.shape}, KL: {kl_loss(mu, lv).item():.4f}")
