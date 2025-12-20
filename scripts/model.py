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
class SimpleTokenizer:
    """Lightweight tokenizer with standard NLP pipeline support"""
    def __init__(self, vocab_size: int = 10000, max_seq_len: int = 768):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = 0  # Padding token ID
        self.unk_token_id = 1  # Unknown token ID

    def _tokenize(self, text: str) -> List[int]:
        """Convert text to character-level token IDs"""
        token_ids = [ord(c) % (self.vocab_size - 2) + 2 for c in text]
        return token_ids

    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with padding/truncation (HuggingFace-compatible API)"""
        final_max_len = max_length if max_length else self.max_seq_len
        input_ids = self._tokenize(text)
        
        if truncation and len(input_ids) > final_max_len:
            input_ids = input_ids[:final_max_len]
        if padding == "max_length" and len(input_ids) < final_max_len:
            input_ids += [self.pad_token_id] * (final_max_len - len(input_ids))
        
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        if return_tensors == "pt":
            input_ids_tensor = input_ids_tensor.unsqueeze(0)
        
        return {"input_ids": input_ids_tensor}

class MiniMultimodalModel(nn.Module):
    """Minimal multimodal Transformer for text-image tasks"""
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        image_size: int = 224,
        in_channels: int = 3,
        num_image_tokens_per_patch: int = 12,
        max_seq_len: int = 768,
        weight_decay: float = 0.01
    ):
        super().__init__()
        # Core parameters
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_image_tokens = num_image_tokens_per_patch ** 2
        self.vocab_size = vocab_size
        self.weight_decay = weight_decay
        self.pad_token_id = 0
        self.image_size = image_size

        # Text embedding layers
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.image_feature_size = (image_size // 16) ** 2
        self.image_proj = nn.Linear(embed_dim * self.image_feature_size, embed_dim * self.num_image_tokens)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """初始化模型参数，提升训练稳定性"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None 
    ):
        batch_size, seq_len = input_ids.shape

        # Text + position embedding
        text_embeds = self.text_embedding(input_ids)  # (batch, seq_len, embed_dim)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.position_embedding(position_ids)  # (batch, seq_len, embed_dim)
        embeds = text_embeds + pos_embeds

        # -------------------------- 重构后：图像嵌入融合 --------------------------
        if pixel_values is not None:
            image_features = self.image_feature_extractor(pixel_values)
            image_flat = image_features.flatten(1)
            image_embeds = self.image_proj(image_flat)
            image_embeds = image_embeds.reshape(batch_size, self.num_image_tokens, self.embed_dim)
            
            del image_features, image_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Replace text tokens with image tokens (front positions)
            if seq_len >= self.num_image_tokens:
                embeds[:, :self.num_image_tokens] = image_embeds
            else:
                embeds[:, :] = image_embeds[:, :seq_len, :]

        if attention_mask is None:
            src_key_padding_mask = (input_ids == self.pad_token_id)  # True = padding token (to be masked)
        else:
            src_key_padding_mask = (attention_mask == 0)  # attention_mask=0 means padding
        
        src_mask = None

        transformer_outputs = self.transformer(
            embeds,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, embed_dim)

        # Language modeling logits
        logits = self.lm_head(transformer_outputs)  # (batch, seq_len, vocab_size)

        # Return loss if labels provided
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss
        return logits

    def loss(self, logits: torch.Tensor, labels: torch.Tensor, train: bool = True):
        # Mask padding tokens for loss calculation
        loss_mask = (labels != self.pad_token_id)  # (batch, seq_len)
        
        # Cross entropy loss (ignoring padding)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none"
        )
        ce_loss = (ce_loss * loss_mask.reshape(-1)).sum() / loss_mask.sum()

        # Calculate L2 regularization (兼容参数name为None的情况)
        l2_loss = 0.0
        for name, p in self.named_parameters():
            if p.requires_grad and "weight" in name:
                l2_loss += torch.norm(p, p=2) ** 2
        l2_loss = 0.5 * self.weight_decay * l2_loss

        # Total loss
        total_loss = ce_loss + l2_loss
        return total_loss

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