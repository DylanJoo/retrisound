import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from modeling.outputs import SparseEncoderOutput

def normalize(tensor, eps=1e-9):
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

class SparseEncoder(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

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
        context_mask=None,
    ):
        """ Add a context masking, to exclude the logits used for final sparse vectors."""

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        last_hidden_states = outputs[0]
        logits = self.cls(last_hidden_states)
        logits = logits * (context_mask or attention_mask).unsqueeze(-1)

        # pooling/aggregation
        values, _ = torch.max(
            torch.log(1 + torch.relu(logits)) 
            * attention_mask.unsqueeze(-1), dim=1
        )

        nonzero_indices = [row.nonzero(as_tuple=False).squeeze(1) for row in values]

        return SparseEncoderOutput(
            reps=values, 
            logits=logits, 
            indices=nonzero_indices,
            last_hidden_states=last_hidden_states, 
            mask=attention_mask
        )
