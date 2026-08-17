"""Vision Transformer (Dosovitskiy et al. 2020) — the v3.0 CIFAR-10 notebook model.

    input -> PatchEmbedding[Conv2d(k=ps, s=ps) -> flatten -> +CLS -> +pos_embed]
          -> depth x TransformerEncoderLayer (pre-LN)
          -> LayerNorm -> take CLS token -> Linear(embed_dim, num_classes)

Pre-LN (norm before both sub-blocks, residual added after) is what keeps deep
transformer stacks trainable — post-LN needs a much more careful warmup.

Resolution is baked in: `pos_embed` has one row per patch, so a ViT trained at
32x32 cannot be fed 224x224 inputs without interpolating the position table.
`image_size` and `patch_size` must therefore match DataConfig.image_size.

There is no convolutional locality/translation prior here, so on small datasets
a ViT leans much harder on augmentation and regularization than a ResNet does
(CIFAR-10 v3.0: 84.4% test vs the ResNet-18's 95.6%). ImageNet-1K is the scale
where that trade starts paying off.

One file per architecture family; register new families in models/__init__.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbedding(nn.Module):
    """Image -> patch token sequence, with a CLS token and learned positions.

    Given B=batch, C=in_channels, H=W=img_size, ps=patch_size, D=embed_dim:

        (B, C, H, W) -> (B, D, H//ps, W//ps) -> (B, D, num_patches)
                     -> (B, num_patches, D) -> (B, 1 + num_patches, D)

    The strided conv IS the patch split: stride == kernel_size means the patches
    never overlap, and the conv weights are the per-patch linear projection.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"image_size={img_size} must be divisible by patch_size={patch_size}"
            )

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Sequential(
            # (B, C, H, W) -> (B, D, H//ps, W//ps)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,  # stride == kernel -> no overlap
            ),
            # patch grid -> sequence: (B, D, H//ps, W//ps) -> (B, D, num_patches)
            nn.Flatten(start_dim=2),
        )

        # Learned, both updated by training: the CLS token is the slot the head
        # reads, the position table is the only source of spatial order (attention
        # itself is permutation-invariant).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))

        # std=0.02, not the default randn: unit-std tokens would swamp the patch
        # projections at init.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        x = self.projection(x)
        x = x.transpose(1, 2)  # (B, D, num_patches) -> (B, num_patches, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # -> (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # -> (B, 1 + num_patches, D)

        return x + self.pos_embed


def scaled_dot_product(q, k, v, mask=None, dropout: nn.Module | None = None):
    """softmax(QK^T / sqrt(d_k)) V. Returns (values, attention).

    The 1/sqrt(d_k) scale keeps the logits out of the region where softmax
    saturates and gradients vanish. `mask` is additive (-inf -> 0 after softmax).
    """
    d_k = q.size(-1)

    # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention = dropout(attention)

    # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadSelfAttention(nn.Module):
    """Explicit MSA: split the embedding across heads, attend in parallel, merge.

    Mathematically equivalent to `nn.MultiheadAttention(batch_first=True)` in
    self-attention mode, but readable and able to hand back the attention maps
    (`return_attention=True`) for visualisation. The torch version is fused and
    faster — see `attention="torch"` in TransformerEncoderLayer.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # One projection for Q, K and V together — a single GEMM instead of three.
        self.qkv_layer = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_projection_layer = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention: bool = False):
        B, N, E = x.shape

        qkv = self.qkv_layer(x)  # (B, N, E) -> (B, N, 3E)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # -> (3, B, H, N, D)
        q, k, v = qkv.unbind(dim=0)

        values, attention = scaled_dot_product(
            q, k, v, mask, dropout=self.attention_dropout
        )

        output = values.transpose(1, 2)  # (B, H, N, D) -> (B, N, H, D)
        output = output.reshape(B, N, self.num_heads * self.head_dim)  # -> (B, N, E)

        output = self.output_dropout(self.out_projection_layer(output))

        if return_attention:
            return output, attention
        return output


class MLP(nn.Module):
    """Per-token feed-forward: Linear -> GELU -> Drop -> Linear -> Drop.

    Attention mixes information *across* tokens; this mixes it *within* each one.
    """

    def __init__(self, in_features: int, hidden_features: int, drop_rate: float):
        super().__init__()

        self.feat_layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.feat_layers(x)


ATTENTIONS = ("torch", "custom")


class TransformerEncoderLayer(nn.Module):
    """Pre-LN encoder block:  x = x + MSA(LN(x));  x = x + MLP(LN(x))

    attention="torch":  nn.MultiheadAttention (fused, what the v3.0 run used)
    attention="custom": MultiHeadSelfAttention above (same math, inspectable)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        drop_rate: float,
        attention: str = "torch",
    ):
        super().__init__()

        if attention not in ATTENTIONS:
            raise ValueError(f"Unknown attention {attention!r}. Known: {ATTENTIONS}")

        self.attention_kind = attention
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        if attention == "torch":
            self.attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=drop_rate, batch_first=True
            )
        else:
            self.attention = MultiHeadSelfAttention(
                embed_dim, num_heads, dropout=drop_rate
            )

        self.MLP = MLP(embed_dim, mlp_dim, drop_rate=drop_rate)

    def forward(self, x):
        h = self.ln1(x)
        if self.attention_kind == "torch":
            attn_out, _ = self.attention(h, h, h, need_weights=False)
        else:
            attn_out = self.attention(h)
        x = x + attn_out
        return x + self.MLP(self.ln2(x))


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attention: str = "torch",
    ):
        super().__init__()

        # mlp_dim wins when given; otherwise the paper's ratio (4x embed_dim).
        if mlp_dim is None:
            mlp_dim = int(embed_dim * mlp_ratio)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.ViT_layers = nn.Sequential(
            self.patch_embed,
            *[
                TransformerEncoderLayer(
                    embed_dim, num_heads, mlp_dim, drop_rate, attention=attention
                )
                for _ in range(depth)
            ],
            # Final norm before the head: the last block's residual output is
            # otherwise unnormalized.
            nn.LayerNorm(embed_dim),
        )

        self.head = nn.Linear(embed_dim, num_classes)

    def no_weight_decay(self) -> set[str]:
        """Params weight decay must skip.

        `split_decay_params` filters by ndim == 1, which catches LayerNorm gains
        and biases but not these two: both are (1, ·, D) tensors that encode
        position/identity rather than a learned transform, and decaying them
        pulls the position table toward zero.
        """
        return {"patch_embed.cls_token", "patch_embed.pos_embed"}

    def forward(self, x):
        x = self.ViT_layers(x)
        cls_tokens = x[:, 0]  # the CLS row is the sequence summary
        return self.head(cls_tokens)


# Depth/width presets. "vit_simple" is the CIFAR-10 v3.0 model (3.20M params);
# the tiny/small/base trio is the paper's ImageNet family at patch 16.
_VARIANTS: dict[str, dict] = {
    # embed_dim, depth, num_heads, patch_size, mlp_dim (None -> 4x embed_dim)
    "vit_simple": dict(embed_dim=256, depth=6, num_heads=8, patch_size=4, mlp_dim=512),
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3, patch_size=16, mlp_dim=None),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6, patch_size=16, mlp_dim=None),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12, patch_size=16, mlp_dim=None),
}


def build_vit(
    arch: str = "vit_simple",
    num_classes: int = 10,
    input_channels: int = 3,
    image_size: int = 32,
    patch_size: int | None = None,
    embed_dim: int | None = None,
    depth: int | None = None,
    num_heads: int | None = None,
    mlp_dim: int | None = None,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.1,
    attention: str = "torch",
    **_ignored,
) -> ViT:
    """Build a ViT from a preset name; any explicitly-passed knob overrides it.

    arch="vit_custom" starts from vit_simple's shape, so pass the knobs you want.
    """
    preset = _VARIANTS.get(arch if arch != "vit_custom" else "vit_simple")
    if preset is None:
        raise ValueError(
            f"Unknown vit arch {arch!r}. Known: {sorted(_VARIANTS)}, or 'vit_custom'."
        )

    # None means "not overridden" — fall back to the preset.
    def pick(value, key):
        return preset[key] if value is None else value

    return ViT(
        img_size=image_size,
        patch_size=pick(patch_size, "patch_size"),
        in_channels=input_channels,
        num_classes=num_classes,
        embed_dim=pick(embed_dim, "embed_dim"),
        depth=pick(depth, "depth"),
        num_heads=pick(num_heads, "num_heads"),
        mlp_dim=pick(mlp_dim, "mlp_dim"),
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attention=attention,
    )
