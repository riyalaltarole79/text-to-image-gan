"""
generator.py
============
Conditional GAN Generator for Text-to-Image synthesis.

Architecture
------------
  noise z (noise_dim)  +  text_emb (ca_dim)
          │
     ConditioningAugmentation → c_hat [B, ca_dim]
          │
     concat [z, c_hat] → [B, noise_dim + ca_dim]
          │
     FC  → [B, gf_dim*8 * 4 * 4]  (reshape to feature map)
          │
     Upsample Block × 4  (4×4 → 8×8 → 16×16 → 32×32 → 64×64)
     Each block: Upsample(×2) → Conv3×3 → InstanceNorm → ReLU
     With text-attention injection at each scale
          │
     Conv1×1 → 3 channels  →  Tanh  →  image [-1, 1]

Text Spatial Attention
----------------------
At each resolution the text embedding is broadcast to a spatial map
and concatenated to the feature map before convolution (channel concatenation).
This forces every spatial location to be conditioned on text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.text_embedder import ConditioningAugmentation


# ── Helpers ────────────────────────────────────────────────────────────────

def weights_init(m):
    """Orthogonal init for Conv; zero-mean normal for BatchNorm."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif "BatchNorm" in classname or "InstanceNorm" in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif "Linear" in classname:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def _norm(channels: int, use_instance: bool) -> nn.Module:
    return nn.InstanceNorm2d(channels, affine=True) if use_instance \
           else nn.BatchNorm2d(channels)


# ── Residual Text-Attention Block ──────────────────────────────────────────

class TextAttentionBlock(nn.Module):
    """
    Injects text conditioning into a spatial feature map.

    Mechanism
    ---------
    1. Project text embedding → spatial weights [B, channels, 1, 1]
    2. Broadcast & element-wise scale (FiLM-style)
    3. Residual add

    Parameters
    ----------
    channels   : int   Feature-map channel count.
    text_dim   : int   Text embedding dimensionality.
    """

    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(text_dim, channels)
        self.beta_fc  = nn.Linear(text_dim, channels)
        nn.init.ones_(self.gamma_fc.weight.data.fill_(0))
        nn.init.zeros_(self.beta_fc.weight)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        x        : [B, C, H, W]
        text_emb : [B, text_dim]
        """
        gamma = self.gamma_fc(text_emb).view(-1, x.size(1), 1, 1) + 1.0
        beta  = self.beta_fc(text_emb).view(-1, x.size(1), 1, 1)
        return gamma * x + beta


# ── Upsample Block ─────────────────────────────────────────────────────────

class UpsampleBlock(nn.Module):
    """
    Doubles spatial resolution.
    Upsample(2×) → Conv3×3 → Norm → ReLU → FiLM conditioning.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        text_dim: int,
        use_instance: bool = True,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = _norm(out_ch, use_instance)
        self.act  = nn.ReLU(inplace=True)
        self.film = TextAttentionBlock(out_ch, text_dim)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.film(x, text_emb)
        return x


# ── Generator ──────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional Generator.

    Parameters
    ----------
    noise_dim    : int   Latent noise dimension (default 100).
    ca_dim       : int   Conditioning Augmentation output dim (default 128).
    gf_dim       : int   Base generator feature-map width (default 64).
    text_dim     : int   Projected text embedding dim going into CA (default 256).
    image_size   : int   Target output size (64 or 128; must be power-of-2 ≥ 32).
    use_instance : bool  Use InstanceNorm (True) or BatchNorm (False).
    """

    def __init__(
        self,
        noise_dim:    int  = 100,
        ca_dim:       int  = 128,
        gf_dim:       int  = 64,
        text_dim:     int  = 256,
        image_size:   int  = 64,
        use_instance: bool = True,
    ):
        super().__init__()
        self.noise_dim  = noise_dim
        self.ca_dim     = ca_dim
        self.gf_dim     = gf_dim
        self.text_dim   = text_dim
        self.image_size = image_size

        # Conditioning Augmentation
        self.ca = ConditioningAugmentation(text_dim, ca_dim)

        # Stem: FC → reshape to 4×4 feature map
        stem_in  = noise_dim + ca_dim
        stem_out = gf_dim * 8 * 4 * 4  # channels × H × W at 4×4
        self.stem = nn.Sequential(
            nn.Linear(stem_in, stem_out, bias=False),
            nn.LayerNorm(stem_out),
            nn.ReLU(inplace=True),
        )

        # Upsample blocks (4→8→16→32→64)
        # Compute how many up-blocks needed
        n_up = {64: 4, 128: 5}.get(image_size, 4)

        self.up_blocks = nn.ModuleList()
        ch = gf_dim * 8
        for i in range(n_up):
            out_ch = max(ch // 2, gf_dim)
            self.up_blocks.append(UpsampleBlock(ch, out_ch, text_dim, use_instance))
            ch = out_ch

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(
        self,
        noise: torch.Tensor,
        text_emb: torch.Tensor,
    ):
        """
        Parameters
        ----------
        noise    : [B, noise_dim]
        text_emb : [B, text_dim]   (from TextEmbedder)

        Returns
        -------
        fake_image : [B, 3, image_size, image_size]  in [-1, 1]
        mu         : [B, ca_dim]
        log_var    : [B, ca_dim]
        """
        # 1. Conditioning Augmentation
        c_hat, mu, log_var = self.ca(text_emb)  # [B, ca_dim]

        # 2. Concatenate noise + conditioned text
        z = torch.cat([noise, c_hat], dim=1)    # [B, noise_dim + ca_dim]

        # 3. Stem → spatial feature map
        h = self.stem(z)                         # [B, gf_dim*8*16]
        h = h.view(h.size(0), self.gf_dim * 8, 4, 4)

        # 4. Upsample blocks with text conditioning
        for block in self.up_blocks:
            h = block(h, text_emb)

        # 5. Output head
        img = self.out_conv(h)
        return img, mu, log_var

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim, device=device)

    def generate(self, text_emb: torch.Tensor) -> torch.Tensor:
        """Convenience: generate without returning CA params."""
        noise = self.sample_noise(text_emb.size(0), text_emb.device)
        img, _, _ = self.forward(noise, text_emb)
        return img


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.config import CFG

    G = Generator(
        noise_dim  = CFG.model.noise_dim,
        ca_dim     = CFG.model.ca_dim,
        gf_dim     = CFG.model.gf_dim,
        text_dim   = CFG.text.projected_dim,
        image_size = CFG.data.image_size,
        use_instance=CFG.train.use_instance_norm,
    )

    B = 4
    noise    = torch.randn(B, CFG.model.noise_dim)
    text_emb = torch.randn(B, CFG.text.projected_dim)

    img, mu, lv = G(noise, text_emb)
    print(f"Generator output: {img.shape}  (min={img.min():.2f}, max={img.max():.2f})")
    print(f"Params: {sum(p.numel() for p in G.parameters()):,}")
