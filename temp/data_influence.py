import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Any, Optional
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
import logging

# 全局配置
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
IGNORE_INDEX = -100
DEFAULT_VOCAB_SIZE = 10000
DEFAULT_EMBED_DIM = 256
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2
DEFAULT_IMAGE_EMBED_DIM = 128

# 设置CUDA同步执行（便于调试）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# -----------------------
# 辅助函数
# -----------------------
def _amp_ctx(device: torch.device):
    """只有CUDA才启用autocast"""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def display_progress(msg: str, current: int, total: int):
    if current % 100 == 0 or current == total - 1:
        progress = (current / max(total, 1)) * 100
        logging.info(f"[{msg}] {current}/{total} ({progress:.2f}%)")

def safe_index_select(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """安全的索引选择，防止越界"""
    max_idx = tensor.size(dim) - 1
    # 裁剪索引到有效范围
    safe_index = torch.clamp(index, 0, max_idx)
    return torch.index_select(tensor, dim, safe_index)

# -----------------------
# 指令微调数据集工具函数（新增图片路径解析）
# -----------------------
def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载jsonl格式的指令微调数据集"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_image(image_path: str, image_root: Optional[str] = None) -> Image.Image:
    """加载图像（支持相对路径+根路径拼接）"""
    # 拼接图片根路径
    if image_root and not os.path.isabs(image_path):
        image_path = os.path.join(image_root, image_path)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path} (原始路径: {image_path if not image_root else image_path.replace(image_root+'/', '')})")
    
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"加载图像失败 {image_path}: {str(e)}")

def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 4, 
                       image_size: int = 224, use_thumbnail: bool = True) -> List[Image.Image]:
    """动态图像预处理（适配多patch）"""
    images = []
    if use_thumbnail:
        thumbnail = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        images.append(thumbnail)
    
    # 生成多patch
    num_patches = random.randint(min_num, max_num)
    for _ in range(num_patches - len(images)):
        images.append(image.resize((image_size, image_size), Image.Resampling.LANCZOS))
    
    return images

# -----------------------
# 纯PyTorch实现的Tokenizer（极简版）
# -----------------------
class SimpleTokenizer:
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.image_token_id = 4  # 图像token标识
        self.max_seq_len = 1024  # 限制最大序列长度

    def encode(self, text: str, return_tensors: str = "pt", max_length: int = None) -> torch.Tensor:
        """极简编码：将文本转换为ID（适配多轮对话）"""
        max_length = max_length or self.max_seq_len
        tokens = [self.bos_token_id]
        # 简单分词（按空格/标点分割）
        for char in text.replace("\n", " <n> ").replace("\\(", " <lp> ").replace("\\)", " <rp> "):
            if char.strip() == "":
                continue
            token_id = hash(char) % (self.vocab_size - 10) + 10  # 避开特殊token
            tokens.append(token_id)
            if len(tokens) >= max_length - 1:  # 留位置给eos
                break
        tokens.append(self.eos_token_id)
        
        # 截断或填充到max_length
        if len(tokens) < max_length:
            tokens += [self.pad_token_id] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        tensor = torch.tensor(tokens, dtype=torch.long)
        if return_tensors == "pt":
            return tensor
        return tokens

    def decode(self, ids: torch.Tensor) -> str:
        """极简解码"""
        return " ".join([f"<tok_{id}>" for id in ids.tolist()])

# -----------------------
# 纯PyTorch实现的Transformer模块
# -----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, embed_dim = x.shape
        
        # 安全投影（防止维度不匹配）
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 安全处理mask
        if mask is not None:
            # 确保mask维度匹配
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            # 裁剪mask到当前序列长度
            mask = mask[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 自注意力 + 残差
        attn_out = self.attn(x, mask)
        # 确保残差连接维度匹配
        if attn_out.shape != x.shape:
            attn_out = attn_out[:, :x.shape[1], :]
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差
        ffn_out = self.ffn(x)
        if ffn_out.shape != x.shape:
            ffn_out = ffn_out[:, :x.shape[1], :]
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

# -----------------------
# 多模态指令微调最小模型（纯PyTorch）
# -----------------------
class MiniMultimodalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        image_embed_dim: int = DEFAULT_IMAGE_EMBED_DIM,
        image_size: int = 224,
        num_image_patches: int = 16,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)  # 位置编码
        
        # 图像嵌入（简单CNN）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, image_embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((num_image_patches, 1)),
        )
        self.image_proj = nn.Linear(image_embed_dim, embed_dim)
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 输出层
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        # 基础维度检查
        batch_size, seq_len = input_ids.shape[:2]
        
        # 限制序列长度不超过max_seq_len
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]
            if labels is not None:
                labels = labels[:, :self.max_seq_len]
        
        # 文本嵌入
        text_embeds = self.text_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # 位置编码（核心修复：确保维度匹配）
        if position_ids is None:
            # 生成安全的位置ID（0到seq_len-1）
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        else:
            # 裁剪位置ID到有效范围
            position_ids = position_ids[:, :seq_len]
            position_ids = torch.clamp(position_ids, 0, self.max_seq_len - 1)
        
        # 安全获取位置嵌入
        pos_embeds = self.pos_embedding(position_ids)  # [batch, seq_len, embed_dim]
        
        # 确保文本嵌入和位置嵌入维度完全匹配
        if text_embeds.shape != pos_embeds.shape:
            min_len = min(text_embeds.shape[1], pos_embeds.shape[1])
            text_embeds = text_embeds[:, :min_len, :]
            pos_embeds = pos_embeds[:, :min_len, :]
            input_ids = input_ids[:, :min_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :min_len]
        
        # 文本+位置嵌入
        x = text_embeds + pos_embeds
        
        # 处理图像输入（安全拼接）
        if pixel_values is not None and pixel_values.nelement() > 0:
            # 确保pixel_values维度正确 [batch, num_patches, 3, H, W]
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            
            batch_size_pix, num_patches = pixel_values.shape[:2]
            
            # 限制图像patch数量
            max_patches = self.max_seq_len // 8
            if num_patches > max_patches:
                pixel_values = pixel_values[:, :max_patches, :, :, :]
                num_patches = max_patches
            
            image_embeds = []
            for i in range(num_patches):
                patch = pixel_values[:, i, :, :, :]  # [batch, 3, H, W]
                img_feat = self.image_encoder(patch)  # [batch, img_embed_dim, num_patches, 1]
                img_feat = img_feat.flatten(2).transpose(1, 2)  # [batch, num_patches, img_embed_dim]
                img_feat = self.image_proj(img_feat)  # [batch, num_patches, embed_dim]
                image_embeds.append(img_feat)
            
            # 拼接图像嵌入
            if image_embeds:
                image_embeds = torch.cat(image_embeds, dim=1)
                # 确保总长度不超过max_seq_len
                total_len = image_embeds.shape[1] + x.shape[1]
                if total_len > self.max_seq_len:
                    # 裁剪文本部分
                    x = x[:, :self.max_seq_len - image_embeds.shape[1], :]
                
                # 拼接图像和文本嵌入
                x = torch.cat([image_embeds, x], dim=1)
                
                # 更新attention mask
                if attention_mask is not None:
                    img_mask = torch.ones((batch_size, image_embeds.shape[1]), device=attention_mask.device)
                    # 确保mask长度匹配
                    if attention_mask.shape[0] != batch_size:
                        img_mask = img_mask[:attention_mask.shape[0]]
                    attention_mask = torch.cat([img_mask, attention_mask[:, :x.shape[1]-image_embeds.shape[1]]], dim=1)
        
        # 确保attention mask维度正确
        if attention_mask is not None:
            attention_mask = attention_mask[:, :x.shape[1]]
            # 防止全零mask
            if attention_mask.sum() == 0:
                attention_mask = torch.ones_like(attention_mask)
        
        # Transformer前向传播
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 对齐logits和labels长度
            logits_len = logits.shape[1]
            labels_len = labels.shape[1]
            
            if logits_len > labels_len:
                logits = logits[:, :labels_len, :]
            elif labels_len > logits_len:
                labels = labels[:, :logits_len]
            
            # 安全计算损失
            loss = self.loss_fn(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1)
            )
        
        return SimpleModelOutput(logits=logits, loss=loss)

# 模型输出类
class SimpleModelOutput:
    def __init__(self, logits: torch.Tensor, loss: Optional[torch.Tensor] = None):
        self.logits = logits
        self.loss = loss

# -----------------------
# 影响力函数核心实现（适配多模态）
# -----------------------
def hvp(loss, params, v):
    """Hessian-vector product: H(loss, params) @ v"""
    if not params or not v:
        return []

    # 确保v与params在同一设备且维度匹配
    device = params[0].device
    v = [vi.to(device) for vi in v]
    v = [vi[:p.shape.numel()].reshape(p.shape) for vi, p in zip(v, params)]

    # 一阶梯度
    first_grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )

    # 二阶：计算Hessian-vector乘积
    hv = torch.autograd.grad(
        outputs=first_grads,
        inputs=params,
        grad_outputs=v,
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )
    return hv

def grad_z(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: Callable = None
) -> List[torch.Tensor]:
    """计算单个训练样本对损失的梯度（适配多模态）"""
    model.zero_grad(set_to_none=True)

    # 准备多模态输入（设备和维度检查）
    batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "labels", "attention_mask", 
                                                               "position_ids", "pixel_values", "image_flags"]}
    
    # 确保input_ids是二维的
    if batch["input_ids"].dim() == 1:
        batch["input_ids"] = batch["input_ids"].unsqueeze(0)
    if "labels" in batch and batch["labels"].dim() == 1:
        batch["labels"] = batch["labels"].unsqueeze(0)

    with _amp_ctx(device):
        # 多模态模型前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            pixel_values=batch.get("pixel_values"),
            image_flags=batch.get("image_flags"),
            labels=batch["labels"]
        )
        loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, batch["labels"])

    # 获取可训练参数
    params = getattr(model, "_if_trainable_params", None)
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        logging.error("grad_z(): 没有可训练参数")
        return []

    # 计算梯度
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=False
    )
    
    # 处理NaN/Inf梯度
    grads = [torch.nan_to_num(g, 0.0, 0.0, 0.0) for g in grads]
    return grads

def s_test(
    model: nn.Module,
    test_batch: Dict[str, torch.Tensor],
    train_loader,
    device: torch.device,
    loss_fn: Callable = None,
    damp: float = 0.01,
    scale: float = 25.0,
    recursion_depth: int = 5000
) -> List[torch.Tensor]:
    """计算s_test = H^{-1} ∇L(test)"""
    # 获取可训练参数
    params = getattr(model, "_if_trainable_params", None)
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        logging.error("s_test(): 没有可训练参数")
        return []

    # 计算测试样本的梯度
    v = grad_z(model, test_batch, device, loss_fn)
    if not v:
        logging.error("s_test(): 计算v失败")
        return []

    # 初始化Hessian逆的估计
    h_estimate = [vi.detach().clone() for vi in v]

    # 递归计算Hessian逆向量乘积
    train_iter = iter(train_loader)
    for i in range(recursion_depth):
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)

        # 计算训练样本的Hessian-vector乘积
        model.zero_grad(set_to_none=True)
        with _amp_ctx(device):
            # 多模态前向传播
            batch = {k: v.to(device) for k, v in train_batch.items() if k in ["input_ids", "labels", "attention_mask", 
                                                                             "position_ids", "pixel_values", "image_flags"]}
            
            # 确保input_ids维度正确
            if batch["input_ids"].dim() == 1:
                batch["input_ids"] = batch["input_ids"].unsqueeze(0)
            if "labels" in batch and batch["labels"].dim() == 1:
                batch["labels"] = batch["labels"].unsqueeze(0)
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                pixel_values=batch.get("pixel_values"),
                image_flags=batch.get("image_flags"),
                labels=batch["labels"]
            )
            loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, batch["labels"])

        hv = hvp(loss, params, h_estimate)
        if not hv:
            logging.warning(f"s_test(): 迭代{i}时Hessian-vector为空")
            continue

        # 更新估计（防止梯度爆炸）
        h_estimate = [
            torch.clamp(v_i + (1.0 - damp) * h_i - hv_i / scale, -1e3, 1e3)
            for v_i, h_i, hv_i in zip(v, h_estimate, hv)
        ]
        
        del loss, hv, outputs
        display_progress("计算s_test", i, recursion_depth)

    return h_estimate

def calculate_influences(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    loss_fn: Callable = None,
    num_test_samples: int = 5,
    recursion_depth: int = 1000,
    damp: float = 0.01,
    scale: float = 25.0,
    save_intermediate: bool = True,
    output_dir: str = "./influence_results"
) -> Tuple[Dict[int, List[float]], Dict[int, List[int]], Dict[int, List[int]]]:
    """计算训练样本对测试样本的影响值（修复类型错误）"""
    # 检查可训练参数
    params = getattr(model, "_if_trainable_params", None)
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("没有可训练参数，请确保模型参数已解冻")

    influences: Dict[int, List[float]] = {}
    harmful_samples: Dict[int, List[int]] = {}
    helpful_samples: Dict[int, List[int]] = {}

    if save_intermediate:
        intermediate_dir = os.path.join(output_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)

    processed_samples = 0

    for test_batch_idx, test_batch in enumerate(tqdm(test_loader, desc="处理测试样本")):
        if processed_samples >= num_test_samples:
            break

        # 确保test_sample_ids是列表
        try:
            test_sample_ids = test_batch["sample_index"].tolist()
            if not isinstance(test_sample_ids, list):
                test_sample_ids = [test_sample_ids]
        except Exception as e:
            batch_size = test_batch["input_ids"].shape[0]
            test_sample_ids = list(range(processed_samples, processed_samples + batch_size))
            logging.warning(f"测试批次{test_batch_idx}缺少sample_index或解析失败: {str(e)}，使用默认ID: {test_sample_ids}")

        # 遍历批次内样本
        for idx_in_batch, test_sample_id in enumerate(test_sample_ids):
            if processed_samples >= num_test_samples:
                break
            
            # 跳过无效的样本ID
            if not isinstance(test_sample_id, int):
                test_sample_id = processed_samples
                logging.warning(f"测试批次{test_batch_idx}样本{idx_in_batch}ID无效，使用默认ID: {test_sample_id}")

            logging.info(f"处理测试样本 {test_sample_id} ({processed_samples+1}/{num_test_samples})")

            # 提取单个测试样本（确保维度正确）
            single_test_batch = {}
            for k in test_batch.keys():
                if k in ["input_ids", "labels", "attention_mask", "position_ids", "pixel_values", "image_flags", "sample_index"]:
                    tensor = test_batch[k][idx_in_batch:idx_in_batch+1]
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    single_test_batch[k] = tensor

            # 计算s_test
            s_test_vec = s_test(
                model=model,
                test_batch=single_test_batch,
                train_loader=train_loader,
                device=device,
                loss_fn=loss_fn,
                damp=damp,
                scale=scale,
                recursion_depth=recursion_depth
            )
            if not s_test_vec:
                logging.error(f"测试样本{test_sample_id}的s_test计算失败")
                processed_samples += 1
                continue

            if save_intermediate:
                s_test_path = os.path.join(intermediate_dir, f"s_test_{test_sample_id}.pt")
                torch.save([s.cpu() for s in s_test_vec], s_test_path)
                logging.info(f"s_test已保存到 {s_test_path}")

            # 计算每个训练样本的影响力
            train_influences: List[float] = []
            grad_z_dir = os.path.join(intermediate_dir, f"grad_z_test_{test_sample_id}") if save_intermediate else None
            if save_intermediate and grad_z_dir:
                os.makedirs(grad_z_dir, exist_ok=True)

            for train_batch in tqdm(train_loader, desc=f"测试样本{test_sample_id} - 计算影响力"):
                # 正确获取训练批次大小
                try:
                    batch_size = train_batch["input_ids"].shape[0]
                except Exception:
                    logging.error("无法获取训练批次大小，跳过该批次")
                    continue

                # 处理每个训练样本
                for train_idx_in_batch in range(batch_size):
                    # 确保train_sample_id是整数
                    try:
                        train_sample_id = train_batch["sample_index"][train_idx_in_batch].item()
                    except Exception as e:
                        train_sample_id = len(train_influences)
                        logging.warning(f"训练样本批次{train_idx_in_batch}缺少ID或解析失败: {str(e)}，使用默认ID: {train_sample_id}")

                    # 提取单个训练样本
                    single_train_batch = {}
                    for k in train_batch.keys():
                        if k in ["input_ids", "labels", "attention_mask", "position_ids", "pixel_values", "image_flags", "sample_index"]:
                            tensor = train_batch[k][train_idx_in_batch:train_idx_in_batch+1]
                            if tensor.ndim == 1:
                                tensor = tensor.unsqueeze(0)
                            single_train_batch[k] = tensor

                    # 计算梯度
                    gz = grad_z(model, single_train_batch, device, loss_fn)
                    if not gz:
                        train_influences.append(0.0)
                        continue

                    if save_intermediate and grad_z_dir:
                        grad_z_path = os.path.join(grad_z_dir, f"grad_z_{train_sample_id}.pt")
                        torch.save([g.cpu() for g in gz], grad_z_path)

                    # 计算影响力：-grad_z · s_test（防止维度不匹配）
                    sdev = s_test_vec[0].device
                    gz = [g.to(sdev) for g in gz]
                    
                    # 确保梯度维度匹配
                    infl_sum = 0.0
                    for g, s in zip(gz, s_test_vec):
                        if g.shape == s.shape:
                            infl_sum += torch.sum(g * s).item()
                        else:
                            # 取最小维度
                            min_size = min(g.numel(), s.numel())
                            infl_sum += torch.sum(g.flatten()[:min_size] * s.flatten()[:min_size]).item()
                    
                    infl = -infl_sum
                    train_influences.append(float(infl))

            # 保存结果
            influences[test_sample_id] = train_influences

            # 确定最有害和最有益的样本
            sorted_idx = np.argsort(train_influences)
            helpful_samples[test_sample_id] = sorted_idx[:50].tolist()  # 最有益（值最小）
            harmful_samples[test_sample_id] = sorted_idx[-50:].tolist()  # 最有害（值最大）

            if save_intermediate:
                influence_path = os.path.join(output_dir, f"influences_test_{test_sample_id}.npy")
                np.save(influence_path, np.array(train_influences, dtype=np.float32))
                logging.info(f"影响力值已保存到 {influence_path}")

            processed_samples += 1

    return influences, harmful_samples, helpful_samples

# -----------------------
# 指令微调数据集类（适配你的数据格式）
# -----------------------
class InstructionTuningDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        image_root: Optional[str] = None,  # 图片根路径
        image_size: int = 224,
        dynamic_image_size: bool = True,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 4,
        use_thumbnail: bool = True,
        num_image_token: int = 8,
        max_seq_len: int = 1024,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = "",
        template_name: str = "default",
        preprocess_function: Callable = None,
        transform: Optional[Callable] = None
    ):
        self.data = load_jsonl_data(data_path)
        self.tokenizer = tokenizer
        self.image_root = image_root  # 新增：图片根路径
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.num_image_token = num_image_token
        self.max_seq_len = max_seq_len
        self.group_by_length = group_by_length
        self.use_packed_ds = use_packed_ds
        self.ds_name = ds_name
        self.template_name = template_name
        self.preprocess_function = preprocess_function or self.default_preprocess
        self.transform = transform or self.default_image_transform()

    def default_image_transform(self):
        """默认图像变换"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def default_preprocess(self, template_name, conversations, tokenizer, num_image_tokens, **kwargs):
        """适配多轮对话的预处理函数"""
        # 拼接多轮对话（human + gpt）
        full_text = ""
        for conv in conversations:
            role = conv["from"]
            content = conv["value"]
            full_text += f"{role}: {content}\n"
        
        # 编码完整对话（限制长度）
        input_ids = tokenizer.encode(full_text, return_tensors="pt", max_length=self.max_seq_len)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        
        # 处理padding
        if input_ids.shape[0] < self.max_seq_len:
            pad_len = self.max_seq_len - input_ids.shape[0]
            input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        
        return {"input_ids": [input_ids], "labels": [labels], "attention_mask": [attention_mask]}

    def get_image_path(self, image_info: str or Dict) -> str:
        """获取图像路径（兼容你的格式）"""
        if isinstance(image_info, str):
            return image_info
        elif isinstance(image_info, dict) and "path" in image_info:
            return image_info["path"]
        else:
            raise ValueError(f"不支持的图像信息格式: {type(image_info)}")

    def _align_inputs_and_labels(self, input_ids, labels):
        """对齐输入和标签"""
        min_len = min(input_ids.shape[0], labels.shape[0])
        return input_ids[:min_len], labels[:min_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, sample_idx: int) -> Dict[str, torch.Tensor]:
        data_item = self.data[sample_idx]
        
        # 1. 处理对话内容（确保包含<image>标签）
        conversations = data_item["conversations"]
        # 检查所有对话轮次的<image>标签
        has_image_token = any("<image>" in conv["value"] for conv in conversations)
        if not has_image_token:
            # 给第一个human对话添加<image>标签
            for conv in conversations:
                if conv["from"] == "human":
                    conv["value"] = f"<image>\n{conv['value']}"
                    break

        # 2. 加载图像（核心：拼接根路径）
        image_path = self.get_image_path(data_item["image"])
        image = load_image(image_path, self.image_root)

        # 3. 动态图像预处理
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail
            )
        else:
            images = [image]

        # 4. 图像变换
        pixel_values = [self.transform(img).to(dtype=torch.bfloat16) for img in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # 5. 文本预处理（多轮对话）
        ret = self.preprocess_function(
            self.template_name,
            deepcopy(conversations),
            self.tokenizer,
            [self.num_image_token * num_patches],
            max_seq_len=self.max_seq_len,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name
        )

        # 6. 对齐输入和标签
        input_ids, labels = self._align_inputs_and_labels(ret["input_ids"][0], ret["labels"][0])
        ret["input_ids"][0] = input_ids
        ret["labels"][0] = labels
        ret["attention_mask"][0] = ret["attention_mask"][0][:input_ids.shape[0]]

        # 7. 处理attention mask
        attention_mask = ret["attention_mask"][0]
        if attention_mask.ndim != 1:
            attention_mask = attention_mask.flatten()

        # 8. 生成position ids（安全范围）
        seq_len = attention_mask.size(0)
        position_ids = torch.arange(seq_len, device=attention_mask.device)
        position_ids = position_ids.masked_fill_(attention_mask == 0, 0)
        # 限制position ids范围
        position_ids = torch.clamp(position_ids, 0, self.max_seq_len - 1)

        return {
            "input_ids": ret["input_ids"][0],
            "labels": ret["labels"][0],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_flags": torch.tensor([1] * num_patches, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
        }

# -----------------------
# 数据加载器构建函数（新增图片根路径）
# -----------------------
def build_dataloader(
    data_path: str,
    tokenizer: SimpleTokenizer,
    image_root: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int = 1024,
    **dataset_kwargs
) -> DataLoader:
    """构建指令微调数据集加载器"""
    dataset = InstructionTuningDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_root=image_root,
        max_seq_len=max_seq_len,
        **dataset_kwargs
    )
    
    # 自定义collate函数，处理变长张量
    def custom_collate_fn(batch):
        """自定义collate函数，处理变长张量"""
        if len(batch) == 0:
            return {}
        
        collated = {}
        for key in batch[0].keys():
            # 收集所有样本的该字段
            tensors = [item[key] for item in batch]
            
            # 对不同类型的张量进行不同处理
            if key in ["input_ids", "labels", "attention_mask", "position_ids"]:
                # 找到批次中的最大长度
                max_len = max(t.size(0) for t in tensors)
                padded = []
                for t in tensors:
                    # 裁剪过长的张量
                    if t.size(0) > max_len:
                        t = t[:max_len]
                    # padding到最大长度
                    pad_len = max_len - t.size(0)
                    if pad_len > 0:
                        if key == "labels":
                            pad = torch.full((pad_len,), IGNORE_INDEX, dtype=t.dtype, device=t.device)
                        elif key == "input_ids":
                            pad = torch.full((pad_len,), tokenizer.pad_token_id, dtype=t.dtype, device=t.device)
                        else:
                            pad = torch.zeros(pad_len, dtype=t.dtype, device=t.device)
                        padded_t = torch.cat([t, pad])
                    else:
                        padded_t = t
                    padded.append(padded_t.unsqueeze(0))
                collated[key] = torch.cat(padded, dim=0)
            
            elif key == "pixel_values":
                # 处理图像patch
                max_patches = max(t.size(0) for t in tensors)
                padded = []
                for t in tensors:
                    # 裁剪过多的patch
                    if t.size(0) > max_patches:
                        t = t[:max_patches]
                    # padding
                    pad_patches = max_patches - t.size(0)
                    if pad_patches > 0:
                        pad = torch.zeros(pad_patches, *t.shape[1:], dtype=t.dtype, device=t.device)
                        padded_t = torch.cat([t, pad])
                    else:
                        padded_t = t
                    padded.append(padded_t.unsqueeze(0))
                collated[key] = torch.cat(padded, dim=0)
            
            else:
                # 其他字段直接拼接
                collated[key] = torch.cat([t.unsqueeze(0) for t in tensors], dim=0)
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=True  # 丢弃不完整的批次
    )

# -----------------------
# 可视化工具
# -----------------------
def plot_influence_distribution(
    influences: Dict[int, List[float]],
    save_dir: str,
    test_idx: int = 0,
    top_n: int = 50
) -> None:
    """绘制影响力分布直方图"""
    os.makedirs(save_dir, exist_ok=True)
    if test_idx not in influences:
        raise ValueError(f"测试样本 {test_idx} 不存在")

    influence_values = np.array(influences[test_idx], dtype=np.float32)
    sorted_values = np.sort(influence_values)

    helpful = sorted_values[:top_n]   # 最有益（值更负）
    harmful = sorted_values[-top_n:]  # 最有害（更正）

    plt.figure(figsize=(10, 6))
    plt.hist(helpful, bins=20, alpha=0.5, label="有益样本 (值更负)")
    plt.hist(harmful, bins=20, alpha=0.5, label="有害样本 (值更正)")
    plt.xlabel("影响力值")
    plt.ylabel("数量")
    plt.title(f"影响力分布 (测试样本 {test_idx})")
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = os.path.join(save_dir, f"influence_dist_test_{test_idx}.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"分布图已保存到 {save_path}")

# -----------------------
# 主函数（新增图片根路径参数）
# -----------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="纯PyTorch实现的指令微调数据集影响力函数计算")
    # 数据集路径（用户指定）
    parser.add_argument("--train_data_path", type=str, required=True, help="训练集路径（jsonl格式）")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试集路径（jsonl格式）")
    parser.add_argument("--image_root", type=str, default="", help="图片根路径（用于拼接相对路径）")
    # 模型配置
    parser.add_argument("--image_size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--embed_dim", type=int, default=256, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer层数")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    # 计算参数
    parser.add_argument("--output_dir", type=str, default="./influence_results")
    parser.add_argument("--num_test_samples", type=int, default=1)
    parser.add_argument("--recursion_depth", type=int, default=50)  # 进一步降低递归深度
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="必须设置为1以避免维度问题")
    parser.add_argument("--num_workers", type=int, default=0)  # 禁用多线程
    parser.add_argument("--log_file", type=str, default=None)

    args = parser.parse_args()

    # 初始化输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 初始化Tokenizer和模型（纯PyTorch实现）
    logging.info("初始化纯PyTorch多模态模型...")
    tokenizer = SimpleTokenizer(vocab_size=DEFAULT_VOCAB_SIZE)
    model = MiniMultimodalModel(
        vocab_size=DEFAULT_VOCAB_SIZE,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len
    ).to(device)

    # 构建数据加载器
    logging.info("构建数据加载器...")
    dataset_kwargs = {
        "image_size": args.image_size,
        "dynamic_image_size": True,
        "min_dynamic_patch": 1,
        "max_dynamic_patch": 2,  # 减少patch数量
        "use_thumbnail": True,
        "num_image_token": 8,
        "max_seq_len": args.max_seq_len,
    }
    
    train_loader = build_dataloader(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataset_kwargs
    )
    
    test_loader = build_dataloader(
        data_path=args.test_data_path,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataset_kwargs
    )

    # 缓存可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("没有可训练参数，请检查模型配置")
    model._if_trainable_params = trainable_params
    logging.info(f"已缓存可训练参数: {len(trainable_params)} 个张量")

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # 计算影响力
    influences, harmful_samples, helpful_samples = calculate_influences(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        num_test_samples=args.num_test_samples,
        recursion_depth=args.recursion_depth,
        damp=args.damp,
        scale=args.scale,
        save_intermediate=True,
        output_dir=args.output_dir,
    )

    # 保存结果
    np.save(os.path.join(args.output_dir, "influences.npy"), influences, allow_pickle=True)
    np.save(os.path.join(args.output_dir, "harmful_samples.npy"), harmful_samples, allow_pickle=True)
    np.save(os.path.join(args.output_dir, "helpful_samples.npy"), helpful_samples, allow_pickle=True)
    logging.info(f"结果已保存到 {args.output_dir}")

    # 可视化
    if influences:
        first_test_idx = next(iter(influences.keys()))
        plot_influence_distribution(influences, args.output_dir, test_idx=first_test_idx)

    logging.info("影响力计算完成!")

if __name__ == "__main__":
    main()