import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSelfAttention
from typing import List

class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention # no V and no O
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer norm and dropout
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        residual = False,
        **kwargs
    ) -> torch.Tensor:

        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)

        # Pre-norm architecture
        hidden_states = self.layer_norm1(hidden_states)
        
        def shape(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        q = shape(self.q_proj(hidden_states))
        k = shape(self.k_proj(encoder_hidden_states))
        v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        if residual:
            context_layer += hidden_states
        
        outputs = (context_layer, attention_scores) 
        return outputs
