"""
discriminator.py
================
Conditional Discriminator for Text-to-Image GAN.

Architecture
------------
  image [B, 3, 64, 64]
     │
  DownBlock × 4  (64→32→16→8→4)  with optional Spectral Norm
     │
  text_emb [B, text_dim]  →  broadcast → [B, text_dim, 4, 4]
     │           │
     └────cat────┘   → [B, img_ch + text_dim, 4, 4]
          │
      Conv1×1 → Conv3×3 → AvgPool  →  scalar logit
          │
        sigmoid (optionally)

Three forward passes per training step
--------------------------------------
  1. real image   + real text   → should output 1
  2. fake image   + real text   → should output 0
  3. real image   + wrong text  → should output 0 (mis-match penalty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ── Helpers ────────────────────────────────────────────────────────────────

def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    return spectral_norm(layer) if use_sn else layer


class DownBlock(nn.Module):
    """
    Strided convolution to halve spatial resolution.
    Conv(stride=2) → [InstanceNorm / BatchNorm] → LeakyReLU
    Optionally wrapped with Spectral Normalisation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_sn: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        layers = [_maybe_sn(conv, use_sn)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Discriminator ──────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Conditional PatchGAN-style discriminator.

    Parameters
    ----------
    df_dim       : int   Base feature-map width.
    text_dim     : int   Projected text embedding dim (from TextEmbedder).
    image_size   : int   Input image size (64 or 128).
    use_sn       : bool  Spectral normalisation in conv layers.
    dropout_rate : float Dropout before final logit.
    """

    def __init__(
        self,
        df_dim:       int   = 64,
        text_dim:     int   = 256,
        image_size:   int   = 64,
        use_sn:       bool  = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.df_dim    = df_dim
        self.text_dim  = text_dim

        # ── Image encoder: image_size → 4×4 ────────────────────────────────
        # 64→32→16→8→4 (4 down-blocks for 64px input)
        n_down = {64: 4, 128: 5}.get(image_size, 4)

        img_layers = []
        in_ch = 3
        for i in range(n_down):
            out_ch = min(df_dim * (2 ** i), df_dim * 8)
            img_layers.append(DownBlock(in_ch, out_ch, use_sn=use_sn, use_norm=(i > 0)))
            in_ch = out_ch
        self.img_encoder = nn.Sequential(*img_layers)
        self.img_ch = in_ch   # channels at 4×4

        # ── Text conditioning projection ────────────────────────────────────
        # Project text to same channel count as image feature map
        self.text_proj = nn.Sequential(
            _maybe_sn(nn.Linear(text_dim, self.img_ch), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Joint head ──────────────────────────────────────────────────────
        joint_ch = self.img_ch + self.img_ch  # img_feat + text_feat
        self.joint_head = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            _maybe_sn(nn.Conv2d(joint_ch, self.img_ch, kernel_size=1, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(self.img_ch, 1, kernel_size=4, stride=1, padding=0), use_sn),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, image: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image    : [B, 3, H, W]
        text_emb : [B, text_dim]

        Returns
        -------
        logits : [B, 1]   (raw, no sigmoid — use BCEWithLogitsLoss)
        """
        B = image.size(0)

        # 1. Encode image → [B, img_ch, 4, 4]
        img_feat = self.img_encoder(image)

        # 2. Project text → [B, img_ch, 4, 4]
        txt_feat = self.text_proj(text_emb)           # [B, img_ch]
        txt_feat = txt_feat.view(B, -1, 1, 1).expand(-1, -1, 4, 4)

        # 3. Concatenate & classify
        joint = torch.cat([img_feat, txt_feat], dim=1)  # [B, 2*img_ch, 4, 4]
        logit = self.joint_head(joint)                   # [B, 1, 1, 1]
        return logit.view(B, 1)

    def forward_uncond(self, image: torch.Tensor) -> torch.Tensor:
        """Unconditional forward pass (for optional gradient penalty)."""
        img_feat = self.img_encoder(image)
        # Use zero text embedding
        zero_txt = torch.zeros(image.size(0), self.text_dim, device=image.device)
        return self.forward(image, zero_txt)


# ── Loss helpers ───────────────────────────────────────────────────────────

_bce_logits = nn.BCEWithLogitsLoss()


def d_loss(
    D: Discriminator,
    real_img: torch.Tensor,
    fake_img: torch.Tensor,
    real_emb: torch.Tensor,
    wrong_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Three-term discriminator loss (StackGAN formulation).

    L_D = BCE(D(real, real_emb),   1)
        + BCE(D(fake, real_emb),   0)
        + BCE(D(real, wrong_emb),  0)
    """
    B      = real_img.size(0)
    ones   = torch.ones(B,  1, device=real_img.device)
    zeros  = torch.zeros(B, 1, device=real_img.device)

    loss_real    = _bce_logits(D(real_img, real_emb),  ones)
    loss_fake    = _bce_logits(D(fake_img.detach(), real_emb), zeros)
    loss_mismatch= _bce_logits(D(real_img, wrong_emb), zeros)

    return (loss_real + loss_fake + loss_mismatch) / 3.0


def g_loss(
    D: Discriminator,
    fake_img: torch.Tensor,
    real_emb: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    lambda_kl: float = 2.0,
) -> torch.Tensor:
    """
    Generator loss: fool discriminator + KL regulariser.
    L_G = BCE(D(fake, real_emb), 1) + λ_KL * KL(μ, σ²)
    """
    from src.text_embedder import kl_loss

    B    = fake_img.size(0)
    ones = torch.ones(B, 1, device=fake_img.device)

    adv  = _bce_logits(D(fake_img, real_emb), ones)
    kl   = kl_loss(mu, log_var)
    return adv + lambda_kl * kl, adv, kl


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.config import CFG

    D = Discriminator(
        df_dim       = CFG.model.df_dim,
        text_dim     = CFG.text.projected_dim,
        image_size   = CFG.data.image_size,
        use_sn       = CFG.train.use_spectral_norm,
        dropout_rate = CFG.train.dropout_d,
    )

    B = 4
    real = torch.randn(B, 3, 64, 64)
    emb  = torch.randn(B, CFG.text.projected_dim)
    out  = D(real, emb)
    print(f"Discriminator output: {out.shape}  values: {out.detach().numpy().ravel()}")
    print(f"Params: {sum(p.numel() for p in D.parameters()):,}")
