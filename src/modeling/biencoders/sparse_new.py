import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from modeling.biencoders.layers import CrossAttentionLayer
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        opt, 
        encoder,
        q_encoder, 
        n_candidates=None
    ):
        super().__init__()
        self.opt = opt
        
        # modeling
        self.encoder = encoder
        self.q_encoder = q_encoder
        self.n_candidates = n_candidates

        for n, p in self.named_parameters():
            p.requires_grad = True if 'crossattention' in n else False

    def forward(
        self, 
        q_tokens, q_masks, 
        f_tokens=None, f_masks=None, 
        d_tokens=None, d_masks=None, 
        step=0
        **kwargs
    ):
        batch_size = q_tokens.size(0)
        max_num_steps = kwargs.pop('include_n_feedbacks', 0)

        reps = []

        # encode query feedback
        output = self.encoder(q_tokens, q_masks)
        if step != 0:
            f_output = self.encoder(f_tokens, f_masks)
            output = self.q_encoder(
                inputs_embeds=output.last_hidden_states,
                attention_mask=q_masks, 
                encoder_hidden_states=f_output.last_hidden_states, 
                encoder_attention_mask=f_masks
            )
        reps.append(output.reps)
        q_reps = torch.stack(reps, dim=1) # B N V

        # loss calculation
        loss_ct = 0.0
        CELoss = nn.CrossEntropyLoss()

        # encode document if using contrastive signals
        d_reps = []
        if d_tokens is not None:
            n_candidates = min(self.n_candidates, len(d_tokens))
            for i in range(n_candidates):
                d_rep = self.d_encoder(d_tokens[i], d_masks[i]).reps
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=0) # N_cand B H

            ## merge from different sources 
            scores_0 = output.reps @ d_reps.view(-1, q_reps.size(-1)).permute(1, 0)
            scores_t = q_reps[:, -1, :] @ d_reps.view(-1, q_reps.size(-1)).permute(1, 0)
            labels = torch.arange(0, batch_size, dtype=torch.long, device=q_reps.device)
            loss_ct = CELoss(scores_t, labels) 

            d_reps_T = d_reps[0]
            d_reps_F = d_reps[1]

        return SparseAdaptiveEncoderOutput(
            reps=q_reps,
            d_reps=d_reps,
            loss_ct=loss_ct,
            logs={'InfoNCE': loss_ct}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)
