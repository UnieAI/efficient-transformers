# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

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
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-5)
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4, # Reduced heads for speed, typically
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

    def forward(self, input_ids, history_features):
        """
        Args:
            input_ids: [batch_size, 1] - The token ID just generated/verified.
            history_features: [batch_size, 1, hidden_size] - The feature vector from the previous step 
                              (either Target Model's last hidden state or Eagle's previous output).
        
        Returns:
            logits: [batch_size, 1, vocab_size] - Prediction for next token.
            new_features: [batch_size, 1, hidden_size] - Feature for next step.
        """
        # 1. Embed Token
        inputs_embeds = self.embed_tokens(input_ids) # [batch, 1, hidden]
        
        # 2. Fuse with History Feature
        # Concatenate along feature dim then project back
        combined = torch.cat([inputs_embeds, history_features], dim=-1)
        x = self.feature_fusion(combined) 
        
        # 3. Transformer Block
        residual = x
        x = self.input_layernorm(x)
        
        # Self-Attention (With trivial mask for length 1, effectively just processing the state)
        # Note: In a real auto-regressive loop, Eagle might attend to PAST Eagle states.
        # For 'Eagle 1/2/3' strictly speaking, it's often a stateless "next step predictor" relative to the full history,
        # OR it maintains its own KV cache. 
        # For simplicity in Phase 1, we assume a stateless step (like a refined MLP) or simple attention.
        attn_out, _ = self.self_attn(x, x, x) 
        x = residual + attn_out
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        new_features = residual + x # [batch, 1, hidden]
        
        # 4. Logits
        logits = self.lm_head(new_features)
        
        return logits, new_features
