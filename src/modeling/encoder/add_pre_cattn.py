import torch
import torch.nn as nn
from transformers import BertForMaskedLM, AutoConfig
from modeling.outputs import SparseEncoderOutput
from modeling.layers import CrossAttentionLayer, normalization


class SparseEncoderWithPreAttention(BertForMaskedLM):
    def __init__(self, config, num_cross_attention=0):
        super().__init__()
        self.num_cross_attention = num_cross_attention
        if self.num_cross_attention >= 1:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.crossattention_layers = nn.ModuleList([
                CrossAttentionLayer(config, zero_init=False, mono_attend=False) for \
                        _ in range(num_post_cross_attention)
            ])

        self.norm = kwargs.pop('norm', False)

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
        sub_input_ids=None,
        sub_attention_mask=None,
        sub_token_type_ids=None,
        context_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        # Pre-crossattention
        input_embeds = self.bert.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        input_ids = None

        if (self.num_cross_attention >= 1) and (sub_input_ids is not None):
            sub_input_embeds = self.bert.embeddings(
                input_ids=sub_input_ids,
                attention_mask=sub_attention_mask,
                token_type_ids=sub_token_type_ids,
            )

            for i, layer_module in enumerate(self.crossattentionlayer):
                input_embeds = layer_module(
                    hidden_states=input_embeds,
                    attention_mask=attention_mask,
                    encoder_hidden_states=sub_input_embeds,
                    encoder_attention_mask=sub_attention_mask
                )[0]

        outputs = self.bert(
            input_ids=None,
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

        values, _ = torch.max(
            torch.log(1 + torch.relu(logits)) 
            * attention_mask.unsqueeze(-1), dim=1
        )

        if self.norm:
            values = normalize(values)

        return SparseEncoderOutput(
            reps=values, 
            logits=None, 
            last_hidden_states=last_hidden_states, 
            all_hidden_states=outputs["hidden_states"], 
            mask=attention_mask
        )

