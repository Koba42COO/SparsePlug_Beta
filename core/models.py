import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .quantization import Q4_K, Q4_DELTA

class SparseFFN(nn.Module):
    """
    UPG-Compatible FFN. 
    Can operate in:
    1. 'Dense' mode (Standard Training)
    2. 'Sparse' mode (Inference/Fine-tuning with masks)
    """
    def __init__(self, hidden_size, intermediate_size, sparsity=0.0):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.sparsity = sparsity
    
    def forward(self, x):
        # Standard SwiGLU-ish
        h = F.gelu(self.up_proj(x))
        
        # Simulated Sparsity (Top-K) if needed
        if self.sparsity > 0:
            k = int(h.shape[-1] * (1 - self.sparsity))
            if k < 1: k = 1
            # Keep top-k magnitude
            vals, _ = torch.topk(h.abs(), k, dim=-1)
            # Threshold is k-th value
            thresh = vals[..., -1].unsqueeze(-1)
            mask = (h.abs() >= thresh).float()
            h = h * mask
            
        return self.down_proj(h)

class UPGModel(nn.Module):
    """
    Base class for UPG-native models.
    """
    def __init__(self, vocab_size, hidden_size, n_layers, max_seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_size, 4, batch_first=True),
                'norm1': nn.LayerNorm(hidden_size),
                'ffn': SparseFFN(hidden_size, hidden_size*4),
                'norm2': nn.LayerNorm(hidden_size)
            }) for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        
        x = self.embed(input_ids) + self.pos_emb(pos)
        
        for layer in self.layers:
            # Attn
            norm_x = layer['norm1'](x)
            attn_out, _ = layer['attn'](norm_x, norm_x, norm_x)
            x = x + attn_out
            
            # FFN
            norm_x = layer['norm2'](x)
            x = x + layer['ffn'](norm_x)
            
        logits = self.head(x)
        
        loss = None
        if labels is not None:
             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
             
        return logits, loss

    def generate(self, input_ids, max_new_tokens=100, do_sample=False, pad_token_id=None):
        """
        Autoregressive generation loop.
        """
        for _ in range(max_new_tokens):
            # 1. Forward
            # Only use last seq_len if exceeding max_seq
            idx_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # 2. Decode strategy
            if do_sample:
                 probs = F.softmax(logits, dim=-1)
                 next_token = torch.multinomial(probs, num_samples=1)
            else:
                 next_token = torch.argmax(logits, dim=-1, keepdim=True)
                 
            # 3. Update
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
        return input_ids
