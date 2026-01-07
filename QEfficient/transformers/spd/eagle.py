# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EagleConfig:
    def __init__(self, vocab_size, hidden_size, num_attention_heads, rms_norm_eps=1e-5):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps


class EagleHead(nn.Module):
    """
    Eagle Draft Head module.
    It simulates a lightweight Transformer layer that takes current token embedding 
    and previous feature state to predict the next token and next feature state.
    """
    def __init__(self, config, hidden_size=None):
        super().__init__()
        self.config = config
        
        # Use config hidden size if not provided
        if hidden_size is None:
            hidden_size = config.hidden_size
            
        self.hidden_size = hidden_size
        
        # Embedding for tokens (re-using model vocabulary)
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        
        # Input projection (combining Token Embed + Feature)
        # Eagle often just adds them, or concatenates. 
        # For generality, let's assume a linear projection layer to mix them if needed,
        # but standard Eagle 1 implementation just adds embedding to the feature.
        # We will implement a specialized FC layer to fuse them.
        self.feature_fusion = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # Lightweight Transformer Layer (Decoder style)
        # Simplified to a self-attention + MLP block
        rms_eps = config.rms_norm_eps if hasattr(config, "rms_norm_eps") else 1e-5
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=rms_eps)
        
        # Self-Attention
        num_heads = getattr(config, "num_attention_heads", 4)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,  # Reduced heads for speed, typically
            batch_first=True
        )
        
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Output Head (to logits)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        return proj.chunk(3, dim=-1)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _attention(
        self,
        x: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self._project_qkv(x)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if (past_key is None) ^ (past_value is None):
            raise ValueError("past_key and past_value must be provided together.")
        if past_key is not None:
            if past_key.dim() != 4 or past_value.dim() != 4:
                raise ValueError("past_key/past_value must have shape [batch, heads, seq_len, head_dim].")
            if past_key.size(1) != self.num_heads or past_key.size(-1) != self.head_dim:
                raise ValueError("past_key/past_value must match model head dimensions.")
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        present_key = k
        present_value = v

        bsz, _, tgt_len, head_dim = q.shape
        src_len = k.size(2)
        q = q.reshape(bsz * self.num_heads, tgt_len, head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, head_dim)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_dim)
        if tgt_len > 1:
            past_len = src_len - tgt_len
            causal = torch.ones(tgt_len, tgt_len, device=attn_scores.device, dtype=torch.bool).triu(1)
            if past_len > 0:
                pad = torch.zeros(tgt_len, past_len, device=attn_scores.device, dtype=torch.bool)
                causal = torch.cat([pad, causal], dim=1)
            attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.hidden_size)
        attn_output = self.self_attn.out_proj(attn_output)
        return attn_output, present_key, present_value

    def forward(
        self,
        input_ids: torch.Tensor,
        history_features: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [batch_size, seq_len] - Token IDs from the previous step(s).
            history_features: [batch_size, seq_len, hidden_size] - Feature vectors aligned to input_ids.
                              (either Target Model's last hidden state or Eagle's previous output).
        
        Returns:
            logits: [batch_size, 1, vocab_size] - Prediction for next token.
            new_features: [batch_size, 1, hidden_size] - Feature for next step.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")
        if history_features.dim() == 2:
            history_features = history_features.unsqueeze(1)
        if history_features.dim() != 3:
            raise ValueError(
                f"history_features must have shape [batch, seq_len, hidden], got {tuple(history_features.shape)}"
            )
        if input_ids.size(1) != history_features.size(1):
            raise ValueError(
                "input_ids and history_features must share the same seq_len, "
                f"got {input_ids.size(1)} and {history_features.size(1)}"
            )
        if history_features.size(-1) != self.hidden_size:
            raise ValueError(
                f"history_features last dim must be hidden_size={self.hidden_size}, got {history_features.size(-1)}"
            )

        # 1. Embed Token
        inputs_embeds = self.embed_tokens(input_ids)  # [batch, seq_len, hidden]
        
        # 2. Fuse with History Feature
        # Concatenate along feature dim then project back
        combined = torch.cat([inputs_embeds, history_features], dim=-1)
        x = self.feature_fusion(combined) 
        
        # 3. Transformer Block
        residual = x
        x = self.input_layernorm(x)
        
        # Self-Attention (With trivial mask for length 1, effectively just processing the state)
        # Note: In a real auto-regressive loop, Eagle might attend to PAST Eagle states.
        # For 'Eagle 1/2/3' this is often a stateless "next step predictor" relative to the full history,
        # OR it maintains its own KV cache. 
        # For simplicity in Phase 1, we assume a stateless step (like a refined MLP) or simple attention.
        attn_out, present_key, present_value = self._attention(x, past_key=past_key, past_value=past_value)
        x = residual + attn_out
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        new_features = residual + x  # [batch, seq_len, hidden]
        
        # 4. Logits
        logits = self.lm_head(new_features)
        
        if past_key is not None and past_value is not None:
            return logits, new_features, present_key, present_value
        return logits, new_features


class EagleDraftLoop(nn.Module):
    def __init__(self, eagle_head: EagleHead, num_steps: int, use_cache: bool = False):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.eagle_head = eagle_head
        self.num_steps = num_steps
        self.use_cache = use_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        history_features: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ):
        if self.use_cache and (past_key is None or past_value is None):
            raise ValueError("past_key and past_value are required when use_cache=True.")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        if history_features.dim() == 2:
            history_features = history_features.unsqueeze(1)
        cur_token = input_ids[:, -1:]
        cur_feature = history_features[:, -1:, :]

        tokens = []
        features = []
        for _ in range(self.num_steps):
            if self.use_cache:
                logits, new_features, past_key, past_value = self.eagle_head(
                    cur_token, cur_feature, past_key, past_value
                )
            else:
                logits, new_features = self.eagle_head(cur_token, cur_feature)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            tokens.append(next_token)
            features.append(new_features[:, -1:, :])
            cur_token = next_token
            cur_feature = new_features[:, -1:, :]

        tokens = torch.cat(tokens, dim=1)
        features = torch.cat(features, dim=1)
        if self.use_cache:
            return tokens, features, past_key, past_value
        return tokens, features
