import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention # no V and no O
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer norm and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def forward(
        self,
        q_hidden_states: torch.Tensor,
        f_hidden_states: torch.Tensor,
        f_attention_mask: torch.Tensor
    ) -> torch.Tensor:

        batch_size = q_hidden_states.size(0)
        seq_length = q_hidden_states.size(1)

        # Pre-norm architecture
        q_hidden_states = self.layer_norm1(q_hidden_states)
        
        def shape(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        q = shape(self.q_proj(q_hidden_states))
        k = shape(self.k_proj(f_hidden_states))
        v = shape(f_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            attention_mask[:, None, None, :] == 0,
            float('-inf')
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        
        return context_layer
        
# class PretrainedSparseLLMRetriever(nn.Module):
#     def __init__(
#         self,
#         pretrained_model,
#         hidden_size: int = 768,
#         num_attention_heads: int = 8,
#         num_cross_layers: int = 3,
#         dropout: float = 0.1,
#         freeze_pretrained: bool = True
#     ):
#         super().__init__()
#         self.pretrained_retriever = pretrained_model
#         if freeze_pretrained:
#             for param in self.pretrained_retriever.parameters():
#                 param.requires_grad = False
#         
#         # Multi-layer cross attention
#         self.cross_attention = MultiLayerCrossAttention(
#             num_layers=num_cross_layers,
#             hidden_size=hidden_size,
#             num_attention_heads=num_attention_heads,
#             dropout=dropout
#         )
#         
#         self._init_weights()
#     
#     def _init_weights(self):
#         def _init_layer(module):
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight, gain=0.1)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#         
#         self.cross_attention.apply(_init_layer)
#     
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         llm_feedback_embeds: torch.Tensor
#     ) -> torch.Tensor:
#         # Get pretrained representations
#         with torch.set_grad_enabled(not self.pretrained_retriever.training):
#             pretrained_hidden = self.pretrained_retriever.get_hidden_states(
#                 input_ids, attention_mask
#             )
#         
#         # Multi-layer cross attention
#         enhanced_hidden = self.cross_attention(
#             pretrained_hidden,
#             llm_feedback_embeds,
#             attention_mask
#         )
#         
#         # Get sparse logits
#         with torch.set_grad_enabled(not self.pretrained_retriever.training):
#             sparse_logits = self.pretrained_retriever.get_sparse_logits(enhanced_hidden)
#         
#         return F.relu(sparse_logits)
