import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers.models.bert.modeling_bert import BertEmbeddings

class STEFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        return (input > 0).float()

    @staticmethod
    def backward(self, grad_output):
        return F.hardtanh(grad_output)

class AdaptiveBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
    
    def forward(
        self,
        input_ids= None,
        token_type_ids = None,
        position_ids= None,
        inputs_embeds = None,
        past_key_values_length = 0,
    ) -> torch.Tensor:

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            # adaptation
            adaptation_embeds = self.word_embeddings_proj(inputs_embeds)
            inputs_embeds = inputs_embeds + adaptation_embeds

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossAttentionLayer(nn.Module):
    def __init__(self, config, zero_init=False, mono_attend=False, output_layer=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention (adapted from BertSelfAttention)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Cross attention output layer (adapted from BertSelfOutput)
        self.output_layer = output_layer

        if zero_init:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.output_proj.bias)

        self.mono_attend = mono_attend
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
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
        # k = shape(encoder_hidden_states)
        # v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )

        if self.mono_attend:
            attention_probs = F.gumbel_softmax(attention_scores, dim=-1, hard=True)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1) # B N_head Lq Lk
        
        context_layer = torch.matmul(attention_probs, v) # B N_head Lq Lk x B N_head Lk H_head = B N_head Lq H_head
        context_layer = context_layer.permute(0, 2, 1 ,3).contiguous() # B L_q N_head H_head
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size) # B L_q H

        # output layer
        if self.output_layer:
            attention_output = context_layer + hidden_states
        else:
            attention_output = context_layer
        
        return (attention_output, attention_scores)

class CrossAttentionSelector(nn.Module):
    def __init__(self, config, zero_init=False, mono_attend=False, output_layer=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention (adapted from BertSelfAttention)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Cross attention output layer (adapted from BertSelfOutput)
        self.output_layer = output_layer

        if zero_init:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.output_proj.bias)

        self.mono_attend = mono_attend
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
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
        # k = shape(encoder_hidden_states)
        # v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )

        if self.mono_attend:
            attention_probs = F.gumbel_softmax(attention_scores, dim=-1, hard=True)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1) # B N_head Lq Lk
        
        context_layer = torch.matmul(attention_probs, v) # B N_head Lq Lk x B N_head Lk H_head = B N_head Lq H_head
        context_layer = context_layer.permute(0, 2, 1 ,3).contiguous() # B L_q N_head H_head
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size) # B L_q H

        # output layer
        if self.output_layer:
            attention_output = context_layer + hidden_states
        else:
            attention_output = context_layer
        
        return (attention_output, attention_scores)
