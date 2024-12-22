import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from modeling.layers import STEFunction
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        q_encoder,
        encoder=None, 
        n_candidates=None,
        **kwargs # opt is unused
    ):
        super().__init__()
        self.q_encoder = q_encoder
        self.encoder = encoder     # use directly the frozen q_encoder
        self.n_candidates = n_candidates
        self.binarize = STEFunction()

        for n, p in self.named_parameters():
            if 'crossattention' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(
        self, 
        q_tokens=None, q_masks=None,
        f_tokens=None, f_masks=None, 
        d_tokens=None, d_masks=None, 
        prev_output=None,
        step=0,
        **kwargs
    ):
        q_reps, d_reps = None, []
        loss_ct, loss_mr = None, None

        if (step == 0) and (prev_output is None):
            output = self.q_encoder(q_tokens, q_masks)
        else:
            f_output = self.q_encoder(f_tokens, f_masks)
            output = self.q_encoder(
                q_tokens, 
                q_masks, 
                encoder_hidden_states=f_output.last_hidden_states, 
                encoder_attention_mask=f_masks
            )

            ## residual
            if prev_output is not None:
                output.reps += prev_output.reps

            # encode positive and negative 
            batch_size, vocab_size = output.reps.shape
            CELoss = nn.CrossEntropyLoss(reduction='mean')
            MRLoss = nn.MarginRankingLoss() 

            if d_tokens is not None:
                n_candidates = min(self.n_candidates, len(d_tokens))
                for i in range(n_candidates):
                    d_rep = self.q_encoder(d_tokens[i], d_masks[i]).reps
                    d_reps.append(d_rep)
                d_reps = torch.stack(d_reps, dim=0) # N_cand B H

                ## L0: contrastive learning
                scores_t = output.reps @ d_reps.view(-1, vocab_size).transpose(1, 0)    # B NB
                labels_ct = torch.arange(0, batch_size, device=output.reps.device, dtype=torch.long)
                loss_ct = CELoss(scores_t, labels_ct)
                # scores_0 = prev_output.reps @ d_reps.view(-1, vocab_size).transpose(1, 0) # B B
                # loss_ct_baseline = CELoss(scores_0, labels)

                ## L1: margin-rank (hinge)
                # d_reps_T = d_reps[0]
                # d_reps_F = d_reps[1:] # negatives
                # margin_0 = (bin_0 @ d_reps_T).diag() - \
                #            (bin_0 @ d_reps_F.transpose(1,2)).max(-1).values
                # margin_t = (bin_t @ d_reps_T).diag() - \
                #            (bin_t @ d_reps_F.transpose(1,2)).max(-1).values
                # labels_mr = torch.ones(margin_0.shape, dtype=torch.long, device=output.reps.device) 
                # loss_mr = MRLoss(margin_t, margin_0, labels_mr)

        return SparseAdaptiveEncoderOutput(
            reps=output.reps,
            prev_out=output,
            d_reps=d_reps,
            loss_ct=loss_ct,
            loss_mr=loss_mr,
            logs={'InfoNCE': loss_ct}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
