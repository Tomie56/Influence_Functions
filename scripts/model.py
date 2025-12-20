import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

def log_clip(x):
    return torch.log(torch.clamp(x, 1e-10, None))

# MNIST multi-class logistic regression (aligned with reference)
class MNISTLogisticRegression(nn.Module):
    def __init__(self, weight_decay, is_multi=False):
        super(MNISTLogisticRegression, self).__init__()
        self.is_multi = is_multi
        self.weight_decay = weight_decay
        self.flatten = nn.Flatten() 
        
        # Define weight parameter manually
        if self.is_multi:
            self.w = torch.nn.Parameter(torch.zeros([10, 784], requires_grad=True))  # [num_classes, input_dim]
        else:
            self.w = torch.nn.Parameter(torch.zeros([784], requires_grad=True))     # [input_dim]

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # flat: [batch, 1, 28, 28] -> [batch, 784]
        x_flat = self.flatten(x)
        
        # logits
        if self.is_multi:
            logits = torch.matmul(x_flat, self.w.T)  # [batch, 784] @ [784, 10] -> [batch, 10]
        else:
            logits = torch.matmul(x_flat, torch.reshape(self.w, [-1, 1]))  # [batch, 784] @ [784, 1] -> [batch, 1]
        
        return logits

    def loss(self, logits, y, train=True):
        if self.is_multi:
            criterion = torch.nn.CrossEntropyLoss()
            y = y.type(torch.FloatTensor).to(logits.device)
            ce_loss = criterion(logits, y.long())
            l2_loss = 0.5 * self.weight_decay * torch.norm(self.w, p=2) ** 2
            total_loss = ce_loss + l2_loss
        else:
            preds = torch.sigmoid(logits)
            bce_loss = -torch.mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds))
            l2_loss = 0.5 * self.weight_decay * torch.norm(self.w, p=2) ** 2
            total_loss = bce_loss + l2_loss
        
        return total_loss

# Simple tokenizer for multimodal task
class SimpleTokenizer(nn.Module):
    def __init__(self, vocab_size: int = 10000, max_seq_len: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2

    def encode(self, text: str) -> List[int]:
        tokens = [self.cls_token_id] + [min(ord(c), self.vocab_size-1) for c in text[:self.max_seq_len-2]] + [self.sep_token_id]
        tokens += [self.pad_token_id] * (self.max_seq_len - len(tokens))
        return tokens[:self.max_seq_len]

    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)

# Lightweight multimodal model with image tokens
class MiniMultimodalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        image_embed_dim: int = 768,
        num_image_tokens_per_patch: int = 12,
        max_seq_len: int = 768
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_image_tokens = num_image_tokens_per_patch ** 2
        self.weight_decay = 0.01

        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Image projection (map to text embed dim)
        self.image_proj = nn.Linear(image_embed_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LM head
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len = input_ids.shape
        loss = None

        # Text embedding with position
        text_embeds = self.text_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.position_embedding(position_ids)
        embeds = text_embeds + pos_embeds

        # Image embedding fusion
        if pixel_values is not None and image_flags is not None:
            image_embeds = self.image_proj(pixel_values)
            pad_mask = (input_ids == 0)
            image_token_mask = pad_mask[:, :self.num_image_tokens]
            embeds[:, :self.num_image_tokens][image_token_mask[:, :self.num_image_tokens]] = image_embeds.flatten(1, 2)[:,:self.embed_dim]

        # Transformer encoding
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1, 1, seq_len, 1)
        attn_mask = (1.0 - attn_mask) * -10000.0
        outputs = self.transformer(embeds, mask=attn_mask)

        # LM logits and loss
        logits = self.lm_head(outputs)
        if labels is not None:
            loss_mask = (labels != 0)
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
            ce_loss = (ce_loss * loss_mask.reshape(-1)).sum() / loss_mask.sum()
            l2_loss = 0.5 * self.weight_decay * sum(torch.norm(p, p=2)**2 for p in self.parameters() if "weight" in p.name)
            loss = ce_loss + l2_loss

        return logits, loss

# Model builder for experiment switching
def build_model(experiment: str, **kwargs) -> nn.Module:
    if experiment == "mnist":
        weight_decay = kwargs.pop("weight_decay", 0.01)
        is_multi = kwargs.pop("is_multi", True)
        return MNISTLogisticRegression(weight_decay=weight_decay, is_multi=is_multi)
    elif experiment == "multimodal":
        return MiniMultimodalModel(**kwargs)
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")
    
def log_clip(x):
    return torch.log(torch.clamp(x, 1e-10, None))