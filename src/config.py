"""
config.py
=========
Central configuration for the Text-to-Image GAN pipeline.
All hyperparameters, paths, and training settings live here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # ── Dataset paths ──────────────────────────────────────────────────────
    data_root: str        = "flower_dataset/102 flower/flowers"
    cat_json:  str        = "flower_dataset/102 flower/cat_to_name.json"
    captions_json: str    = "data/captions.json"
    train_split:  str     = "train"
    valid_split:  str     = "valid"
    test_split:   str     = "test"

    # ── Image settings ─────────────────────────────────────────────────────
    image_size:   int     = 64        # 64×64; set to 128 for higher quality
    channels:     int     = 3         # RGB
    num_workers:  int     = 4

    # ── Caption augmentation ───────────────────────────────────────────────
    captions_per_image: int = 5       # how many caption variants per image


@dataclass
class TextConfig:
    # ── Embedding model ────────────────────────────────────────────────────
    # Sentence-BERT model (downloaded automatically on first run)
    sbert_model: str      = "all-MiniLM-L6-v2"   # 384-d, fast & good
    # Set to "paraphrase-mpnet-base-v2" for 768-d, higher quality

    raw_embed_dim:  int   = 384       # matches all-MiniLM-L6-v2 output
    projected_dim:  int   = 256       # projected text embedding dimension

    # ── TF-IDF fallback ────────────────────────────────────────────────────
    use_tfidf_augment: bool = True    # concat TF-IDF signal to SBERT
    tfidf_dim:         int  = 128     # TF-IDF vocabulary projection size

    # ── Preprocessing ──────────────────────────────────────────────────────
    max_seq_len:  int     = 64
    do_augment:   bool    = True      # synonym/paraphrase augmentation


@dataclass
class ModelConfig:
    # ── Noise / latent ─────────────────────────────────────────────────────
    noise_dim:     int    = 100
    ca_dim:        int    = 128       # Conditioning Augmentation output dim
    # Generator input = ca_dim + noise_dim = 228 by default

    # ── Generator channel widths (coarse → fine) ───────────────────────────
    gf_dim:        int    = 64        # base generator feature-map size
    # actual widths: [gf_dim*8, gf_dim*4, gf_dim*2, gf_dim]

    # ── Discriminator ──────────────────────────────────────────────────────
    df_dim:        int    = 64        # base discriminator feature-map size

    # ── Image output ───────────────────────────────────────────────────────
    image_channels: int   = 3


@dataclass
class TrainConfig:
    # ── Basic training ─────────────────────────────────────────────────────
    epochs:         int   = 120
    batch_size:     int   = 64
    seed:           int   = 42

    # ── Optimisers ─────────────────────────────────────────────────────────
    lr_g:           float = 2e-4      # Generator learning rate
    lr_d:           float = 2e-4      # Discriminator learning rate
    beta1:          float = 0.5
    beta2:          float = 0.999

    # ── Loss weights ───────────────────────────────────────────────────────
    lambda_kl:      float = 2.0       # KL-divergence weight (CA module)
    lambda_feat:    float = 1.0       # feature-matching weight

    # ── Scheduling ─────────────────────────────────────────────────────────
    lr_decay_epoch: int   = 100       # epoch at which LR begins decaying
    lr_decay_gamma: float = 0.1

    # ── Regularisation ─────────────────────────────────────────────────────
    use_spectral_norm: bool = True    # spectral norm in discriminator
    use_instance_norm: bool = True    # instance norm in generator
    dropout_d:      float = 0.3

    # ── Logging & checkpointing ────────────────────────────────────────────
    log_dir:        str   = "logs"
    checkpoint_dir: str   = "checkpoints"
    output_dir:     str   = "outputs"
    save_every:     int   = 10        # save checkpoint every N epochs
    sample_every:   int   = 5         # generate sample grid every N epochs
    num_samples:    int   = 16        # images in sample grid
    fid_every:      int   = 20        # compute FID every N epochs

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp:        bool  = False     # set True if CUDA available & GPU >= A10

    # ── Resume ─────────────────────────────────────────────────────────────
    resume:         Optional[str] = None  # path to checkpoint .pt file


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    text:  TextConfig  = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        os.makedirs(self.train.log_dir,        exist_ok=True)
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.output_dir,     exist_ok=True)
        os.makedirs("data",                    exist_ok=True)

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Singleton accessor ──────────────────────────────────────────────────────
CFG = Config()
