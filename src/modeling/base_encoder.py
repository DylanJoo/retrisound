# import torch.nn.functional as F
import torch
from typing import Optional, Tuple
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass

@dataclass
class EncoderOutput(BaseModelOutput):
    emb: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class Contriever(BertModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.pooling = kwargs.pop('pooling', 'mean')

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

        if attention_mask is not None:
            last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        if self.pooling == 'mean':
            emb = last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        if self.pooling == 'cls':
            emb = last_hidden_states[:, 0]
        return EncoderOutput(emb=emb, last_hidden_states=last_hidden_states)
