import inspect
import torch
import torch.nn.functional as F
from .base import RMTBaseModel
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass

@dataclass
class RMTEncoderOutput(BaseModelOutput):
    emb: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None

class RMTEncoder(RMTBaseModel):

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        head_mask=None,
        inputs_embeds=None, 
        labels=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
    ):
        """ typical inputs of segments: there has length of `n_max_semgnets`. 

        Each with a standalone input_ids/attention_mask
        params: input_ids: (a list of) tensors (shape: B x |q_0|)
        params: attention_mask: (a list of) attention_mask for (segment) input_ids 

        """
        if isinstance(input_ids, list) is False:
            input_ids = [input_ids]
            attention_mask = [attention_mask]

        # use the first input_ids to setup memory
        memory = self.set_memory(input_ids[0].shape)

        ## split long input into n_max_semgents shorter inputs
        ## Also, the memory tokens are ready for each segments
        ## [NOTE] in adaretrieval, we revise this by turns
        # segmented = self.pad_and_segment(input_ids) 
        # move this stage into the for loop next

        if self.num_mem_tokens == 0: # means the original one
            input_ids = input_ids[-1:] 

        ada_embeds = []
        # base_model_outputs = []
        for seg_num in range(len(input_ids)): # length of semgnets

            ## first obtaib the non-empty segment (each would be differ)
            segment_input_ids = input_ids[seg_num]
            segment_attention_mask = attention_mask[seg_num]
            non_empty_mask = [len(s) > 2 for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue # skip this loop if all are empty

            ## expand input_ids with memory and special tokens
            segment_input_ids, segment_attention_mask = self.add_special_tokens_and_masks(
                tensors=segment_input_ids[non_empty_mask],
                masks=segment_attention_mask[non_empty_mask]
            )
            ## transform ids into mebeddings
            segment_inputs_embeds = self.model.embeddings(segment_input_ids)
            segment_inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            # put to the embedding space
            seg_kwargs = {
                'input_ids': None,
                'inputs_embeds': segment_inputs_embeds,
                'attention_mask': segment_attention_mask,
                'token_type_ids': None, # put everything zeros
            }
            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
            if seg_num == 0:
                ada_hidden_state = self.adaptive_pooling(out) # available for each batch
            else:
                ada_hidden_state[non_empty_mask] = self.adaptive_pooling(out)

            # print(ada_hidden_state)
            ## log outputs and ada_hidden_state
            # base_model_outputs.append(out)
            ada_embeds.append(ada_hidden_state.clone())

        return RMTEncoderOutput(
            emb=None,
            last_hidden_state=torch.stack(ada_embeds, dim=1) # B N_segs H
        )
        # out = self.process_outputs(base_model_outputs, 
        #                            output_attentions=False,
        #                            output_hidden_states=True)
        # return {'outputs': out, 'embeds': torch.stack(ada_embeds, dim=1)} # B N_segs H

    def adaptive_pooling(self, out):
        """
        option1: average of [CLS] [MEM] [SEP]
        """
        hidden_state = out.last_hidden_state[:, :(self.num_mem_tokens+2)].mean(1)
        return hidden_state
        # option2: average of everything (but this is not like contriever only use segment1)

    def add_special_tokens_and_masks(self, tensors, masks):
        """
        tensor: the raw token ids of text input (any special tokens is excluded)
        mask: the raw mask of text input 
        """
        batch_size = tensors.shape[0]

        pre = torch.cat([self.cls_token, self.mem_token_ids, self.sep_token]).repeat((batch_size, 1))
        tensors_ = torch.cat([pre, tensors[:, 1:]], dim=1)

        pre = torch.ones( (batch_size, pre.size(1)), dtype=torch.long )
        masks_ = torch.cat([pre, masks[:, 1:]], dim=1)
        return tensors_, masks_

    def process_outputs(
        self, 
        model_outputs, 
        output_attentions, 
        output_hidden_states
    ):
        """
        # [TODO] 
        # we may need to adjust the output here (or directly change the return of forward)
        Save the multiple encoded representation of RMT-Enc
        """
        rmt_out = model_outputs[-1] # the last segment

        segment_keys = ['loss']
        if output_attentions:
            segment_keys.append('attentions')
        if output_hidden_states:
            segment_keys.append('last_hidden_state')
            segment_keys.append('hidden_states')

        extracted = {}
        for seg_num, out in enumerate(model_outputs):
            for key, value in out.items():
                if any([sk in key for sk in segment_keys]):
                    extracted[f'{key}_{seg_num}'] = value

        if self.rmt_config['sum_loss']:
            losses = [out['loss'] for out in model_outputs]
            extracted['loss'] = torch.stack(losses).mean(dim=0)

        for key, value in extracted.items():
            rmt_out[key] = value

        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        return rmt_out 

