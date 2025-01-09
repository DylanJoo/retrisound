import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput

def make_labels(d_tokens, candidate_tokens, candidate_masks):
    binary_matrix = torch.zeros_like(candidate_tokens)
    for i in range(len(d_tokens)):
        binary_matrix[i] = (candidate_tokens[i].unsqueeze(1) == d_tokens[i]).any(dim=1)
    
    binary_matrix = torch.where(
        candidate_masks==0, torch.full_like(binary_matrix, -100), binary_matrix
    )
    return binary_matrix.to(candidate_tokens.device)

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
        self.encoder = (encoder or q_encoder)
        self.n_candidates = n_candidates
        self.config = q_encoder.model.config

        for n, p in self.named_parameters():
            if 'crossattention' in n:
                p.requires_grad = True
                print(n)
            else:
                p.requires_grad = False

    def _flop(self, q_value):
        lambda_t_q = 0.005
        q_value = torch.sum(torch.mean(torch.abs(q_value), dim=0) ** 2)
        return q_value * lambda_t_q

    def get_contrastive_loss(self):
        batch_size, vocab_size = output.reps.shape
        CELoss = nn.CrossEntropyLoss()

        n_candidates = min(self.n_candidates, len(d_tokens))
        for i in range(n_candidates):
            d_rep = self.encoder(d_tokens[i], d_masks[i]).reps
            d_reps.append(d_rep)
        d_reps = torch.stack(d_reps, dim=0) # N_cand B H

        ## L0: flop
        loss_flop = self._flop(output.reps)

        ## L1: contrastive learning
        scores_t = output.reps @ d_reps.view(-1, vocab_size).transpose(1, 0)    # B NB
        labels_ct = torch.arange(0, batch_size, device=output.reps.device, dtype=torch.long)
        loss_ct = CELoss(scores_t, labels_ct)
        # scores_0 = prev_output.reps @ d_reps.view(-1, vocab_size).transpose(1, 0) # B B
        # loss_ct_baseline = CELoss(scores_0, labels)
        return loss_ct

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
        loss_tc, loss_flop, loss_ct, loss_mr = None, None, None, None
        pos_ratio = 0

        if (step == 0) and (prev_output is None):
            prev_output = output = self.encoder(q_tokens, q_masks)
            reps = q_tokens
        else:
            f_output = self.encoder(f_tokens, f_masks)
            output = self.q_encoder(
                q_tokens, 
                q_masks, 
                encoder_hidden_states=f_output.last_hidden_states,
                encoder_attention_mask=f_masks
            )

            candidate_tokens = torch.cat([q_tokens, f_tokens], 1)
            candidate_masks = torch.cat([q_masks, f_masks], 1)

            reps = output.logits.softmax(-1)[:, :, 1]
            reps = torch.where(
                reps>0.5, candidate_tokens, torch.full_like(candidate_tokens, 0)
            )

            # encode positive and negative 
            batch_size, seq_size, _ = output.logits.shape
            CELoss = nn.CrossEntropyLoss()

            if d_tokens is not None:
                d_reps = self.encoder(d_tokens[0], d_masks[0]).indices
                labels = make_labels(d_reps, candidate_tokens, candidate_masks)
                loss_tc = CELoss(output.logits.view(-1, 2), labels.view(-1))
                pos_ratio = (labels==1).sum()  / (labels!=-100).sum()

        return SparseAdaptiveEncoderOutput(
            reps=reps,
            prev_out=output,
            d_reps=d_reps,
            loss_ct=torch.tensor([0.0]),
            loss_mr=torch.tensor([0.0]),
            loss_flop=torch.tensor([0.0]),
            loss_tc=loss_tc,
            logs={'InfoNCE': loss_ct, 'PosRatio': pos_ratio},
            logits=output.logits,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
