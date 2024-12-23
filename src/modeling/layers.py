import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class CrossAttentionLayer(nn.Module):
    def __init__(self, config, zero_init=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention (adapted from BertSelfAttention)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Cross attention output layer (adapted from BertSelfOutput)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if zero_init:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.output_proj.bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        beta = 0.0,
        **kwargs
    ) -> torch.Tensor:

        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        
        def shape(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # B N_head L H_head
        q = shape(self.q_proj(hidden_states))
        k = shape(self.k_proj(encoder_hidden_states))
        v = shape(self.v_proj(encoder_hidden_states))
        # v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )

        attention_probs = F.softmax(attention_scores, dim=-1) # B N_head Lq Lk
        # attention_probs = F.gumbel_softmax(attention_scores, dim=-1, hard=True)
        
        context_layer = torch.matmul(attention_probs, v) # B N_head Lq Lk x B N_head Lk H_head = B N_head Lq H_head
        context_layer = context_layer.permute(0, 2, 1 ,3).contiguous() # B L_q N_head H_head
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size) # B L_q H

        # output layer
        attention_output = self.output_proj(context_layer)
        attention_output = self.output_norm(attention_output + hidden_states)
        
        return (attention_output, attention_scores)
