import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Shared constants
# -----------------------------
IGNORE_INDEX = -100

DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_EMBED_DIM = 384
DEFAULT_NUM_HEADS = 6
DEFAULT_NUM_LAYERS = 4
DEFAULT_IMAGE_EMBED_DIM = 192

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<image>": 4,
    "<n>": 5,
}

# -----------------------------
# MNIST (logistic regression)
# -----------------------------
MNIST_NUM_CLASSES = 10
MNIST_INPUT_DIM = 28 * 28  # 784


@dataclass
class SimpleModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class TorchLogisticRegression(nn.Module):
    """Multiclass logistic regression for MNIST."""

    def __init__(
        self,
        input_dim: int = MNIST_INPUT_DIM,
        num_classes: int = MNIST_NUM_CLASSES,
        loss_reduction: str = "mean",   # "mean" | "sum" | "none"
        l2_reg: float = 0.0,            # If >0, add L2 on weights into the loss.
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.linear = nn.Linear(self.input_dim, self.num_classes, bias=True)

        if loss_reduction not in ("mean", "sum", "none"):
            raise ValueError(f"loss_reduction must be mean/sum/none, got {loss_reduction}")
        self.loss_reduction = loss_reduction

        self.criterion = nn.CrossEntropyLoss(reduction=loss_reduction)
        self.l2_reg = float(l2_reg)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> SimpleModelOutput:
        """
        Args:
            x: [B, 1, 28, 28] or [B, 784]
            labels: [B] (optional)
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}")

        logits = self.linear(x)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                labels = labels.view(-1)
            loss = self.criterion(logits, labels)

            if self.l2_reg > 0.0:
                # L2 on weights only (exclude bias).
                l2 = (self.linear.weight ** 2).sum()
                if self.loss_reduction == "mean":
                    l2 = l2 / x.size(0)
                loss = loss + self.l2_reg * l2

        return SimpleModelOutput(logits=logits, loss=loss)


# -----------------------------
# Multimodal for transformer
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} / {num_heads}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
            attn_mask: broadcastable to [B, H, T, T], with 0 marking masked positions
        """
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(mult) * int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniMultimodalModel(nn.Module):
    """
    Minimal multimodal LM:
      - Text tokens + positional embeddings
      - Image patches encoded into a fixed number of image tokens
      - Autoregressive transformer decoder
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        image_embed_dim: int = DEFAULT_IMAGE_EMBED_DIM,
        image_size: int = 224,
        num_image_tokens_per_patch: int = 12,
        max_seq_len: int = 768,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.max_seq_len = int(max_seq_len)

        self.image_size = int(image_size)
        self.num_image_tokens_per_patch = int(num_image_tokens_per_patch)

        self.text_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

        # CNN encoder producing K tokens per image patch group.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(128, image_embed_dim, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.num_image_tokens_per_patch, 1)),
        )
        self.image_proj = nn.Linear(image_embed_dim, self.embed_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(self.embed_dim, int(num_heads), dropout=0.1) for _ in range(int(num_layers))]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # Shape: [1, 1, T, T], lower-triangular 1s.
        m = torch.tril(torch.ones((T, T), device=device, dtype=torch.uint8))
        return m.view(1, 1, T, T)

    def _merge_padding(self, causal: torch.Tensor, padding: Optional[torch.Tensor]) -> torch.Tensor:
        if padding is None:
            return causal
        B, T = padding.shape
        pad = padding.view(B, 1, 1, T).to(dtype=causal.dtype)
        return causal * pad

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
        image_flags: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, P, 3, H, W] (P patches/tiles per sample)
            image_flags: [B, P] indicating which patches are present
        Returns:
            img_tokens: [B, P*K, C]
            img_token_mask: [B, P*K] in {0,1}
        """
        B, P, _, H, W = pixel_values.shape
        K = self.num_image_tokens_per_patch

        pv = pixel_values.view(B * P, 3, H, W)
        feat = self.image_encoder(pv)                   # [B*P, D, K, 1]
        feat = feat.squeeze(-1).transpose(1, 2)         # [B*P, K, D]
        tok = self.image_proj(feat)                     # [B*P, K, C]
        img_tokens = tok.view(B, P * K, self.embed_dim) # [B, P*K, C]

        if image_flags is None:
            flags = torch.ones((B, P), device=pixel_values.device, dtype=torch.long)
        else:
            flags = image_flags.to(device=pixel_values.device, dtype=torch.long)

        img_token_mask = flags.unsqueeze(-1).repeat(1, 1, K).view(B, P * K)
        return img_tokens, img_token_mask

    def _insert_image_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        img_tokens: torch.Tensor,
        img_token_mask: torch.Tensor,
        image_token_id: int = SPECIAL_TOKENS["<image>"],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Replace the first <image> token in each sequence with the image token block.
        """
        B, T = input_ids.shape
        device = input_ids.device
        text_emb = self.text_embedding(input_ids)

        new_embeds: List[torch.Tensor] = []
        new_masks: List[torch.Tensor] = []
        new_labels: Optional[List[torch.Tensor]] = [] if labels is not None else None

        for b in range(B):
            ids = input_ids[b]
            m = attention_mask[b]
            e = text_emb[b]
            lab = labels[b] if labels is not None else None

            pos = (ids == image_token_id).nonzero(as_tuple=False)
            insert_at = int(pos[0].item()) if pos.numel() > 0 else 0

            left_e, right_e = e[:insert_at], e[insert_at + 1 :]
            left_m, right_m = m[:insert_at], m[insert_at + 1 :]

            e2 = torch.cat([left_e, img_tokens[b], right_e], dim=0)
            m2 = torch.cat([left_m, img_token_mask[b], right_m], dim=0)

            new_embeds.append(e2)
            new_masks.append(m2)

            if labels is not None and new_labels is not None:
                left_l, right_l = lab[:insert_at], lab[insert_at + 1 :]
                img_l = torch.full((img_tokens.size(1),), IGNORE_INDEX, device=device, dtype=lab.dtype)
                new_labels.append(torch.cat([left_l, img_l, right_l], dim=0))

        max_len = min(max(x.size(0) for x in new_embeds), self.max_seq_len)
        C = self.embed_dim

        out_e = torch.zeros((B, max_len, C), device=device, dtype=new_embeds[0].dtype)
        out_m = torch.zeros((B, max_len), device=device, dtype=attention_mask.dtype)
        out_l = torch.full((B, max_len), IGNORE_INDEX, device=device, dtype=labels.dtype) if labels is not None else None

        for b in range(B):
            L = min(new_embeds[b].size(0), max_len)
            out_e[b, :L] = new_embeds[b][:L]
            out_m[b, :L] = new_masks[b][:L]
            if labels is not None and out_l is not None and new_labels is not None:
                out_l[b, :L] = new_labels[b][:L]

        return out_e, out_m, out_l

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SimpleModelOutput:
        if attention_mask is None:
            attention_mask = (input_ids != SPECIAL_TOKENS["<pad>"]).long()

        input_ids = input_ids[:, : self.max_seq_len]
        attention_mask = attention_mask[:, : self.max_seq_len]
        if labels is not None:
            labels = labels[:, : self.max_seq_len]

        x = self.text_embedding(input_ids)
        m = attention_mask
        l = labels

        if pixel_values is not None:
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            img_tokens, img_mask = self._encode_images(pixel_values, image_flags)
            x, m, l = self._insert_image_tokens(input_ids, m, l, img_tokens, img_mask)

        B, T = x.shape[:2]
        pos = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        x = x + self.pos_embedding(pos)

        causal = self._causal_mask(T, device=x.device)
        full_mask = self._merge_padding(causal, m)

        for layer in self.layers:
            x = layer(x, full_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if l is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = l[:, 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return SimpleModelOutput(logits=logits, loss=loss)


class SimpleTokenizer:
    """A tiny tokenizer with stable hashing for out-of-vocab tokens."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, max_seq_len: int = 1024):
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)

        self.pad_token_id = SPECIAL_TOKENS["<pad>"]
        self.eos_token_id = SPECIAL_TOKENS["<eos>"]
        self.bos_token_id = SPECIAL_TOKENS["<bos>"]
        self.unk_token_id = SPECIAL_TOKENS["<unk>"]
        self.image_token_id = SPECIAL_TOKENS["<image>"]

        self._base = 100

    @staticmethod
    def _stable_hash(s: str) -> int:
        import hashlib
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a whitespace-split sequence with a few special tokens.
        Newlines are mapped to the <n> token.
        """
        text = text.replace("\n", " <n> ")
        parts = text.split(" ")

        ids: List[int] = [self.bos_token_id]
        for p in parts:
            if not p:
                continue
            if p in SPECIAL_TOKENS:
                ids.append(SPECIAL_TOKENS[p])
            else:
                hid = self._stable_hash(p)
                ids.append(self._base + (hid % (self.vocab_size - self._base)))

            if len(ids) >= self.max_seq_len - 1:
                break

        ids.append(self.eos_token_id)
        return torch.tensor(ids, dtype=torch.long)
