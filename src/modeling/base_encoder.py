import torch
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM, AutoConfig
from modeling.outputs import SparseEncoderOutput, DenseEncoderOutput
from modeling.layers import CrossAttentionLayer

def normalize(tensor, eps=1e-9):
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

class SparseEncoder(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()

        self.add_cross_attention = kwargs.pop('cross_attention', False)
        if self.add_cross_attention:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.crossattentionlayer = CrossAttentionLayer(config)
            # config = AutoConfig.from_pretrained(model_name_or_path, 
            #     is_decoder=True, 
            #     add_cross_attention=True,
            #     num_hidden_layers=2
            # )
        else:
            config = None
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path, config=config)
        self.output = kwargs.pop('output', 'MLM')
        self.agg = kwargs.pop('agg', 'max')
        self.activation = kwargs.pop('activation', 'relu') 
        self.norm = kwargs.pop('norm', False)

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

        outputs = self.model.bert(
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

        if self.add_cross_attention:
            last_hidden_states = self.crossattentionlayer(
                hidden_states=outputs[0],
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )[0]
        else: 
            last_hidden_states = outputs[0]

        logits = self.model.cls(last_hidden_states)

        # pooling/aggregation
        values = torch.sum(
            torch.log(1 + torch.relu(logits)) 
            * attention_mask.unsqueeze(-1), dim=1
        ) 

        # normalization (for cos)
        if self.norm:
            values = normalize(values)

        return SparseEncoderOutput(
            reps=values, 
            logits=logits, 
            last_hidden_states=last_hidden_states, 
            all_hidden_states=outputs["hidden_states"], 
            mask=attention_mask
        )

## Learned dense encoder
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
        return DenseEncoderOutput(
            emb=emb, 
            last_hidden_states=last_hidden_states
        )
