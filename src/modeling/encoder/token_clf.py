import torch
import torch.nn as nn
from transformers import BertForTokenClassification
from modeling.outputs import SparseEncoderOutput

class SparseEncoderForTokenClf(BertForTokenClassification):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        context_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """ [Adjustment]
        - position_ids: should be newly clone when you are going to freeze them. 
        """

        position_ids = self.bert.embeddings.position_ids[:, 0 : input_ids.size(1) + 0].clone()
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        last_hidden_states = outputs[0]
        tok_logits = self.classifier(last_hidden_states)
        nonzero_indices = None

        return SparseEncoderOutput(
            logits=tok_logits, 
            indices=nonzero_indices,
            last_hidden_states=last_hidden_states, 
            all_hidden_states=outputs["hidden_states"], 
            mask=attention_mask
        )

