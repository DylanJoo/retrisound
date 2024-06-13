import os
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Contriever(BertModel):
    def __init__(self, config, add_pooling_layer=False, pooling='mean', **kwargs):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config.pooling = pooling
        self.outputs = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_multi_vectors=False,
        pooling=None
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True
        )

        last_hidden_states = model_output["last_hidden_state"]
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        pooling = (pooling or self.config.pooling)
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        if return_multi_vectors:
            return emb, last_hidden 
        else:
            return (emb, None)
