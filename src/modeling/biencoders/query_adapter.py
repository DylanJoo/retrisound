import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput
from modeling.biencoders.utils import make_labels, transform_weights_to_vector, sample_actions

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        q_encoder,
        encoder=None, 
        **kwargs # opt is unused
    ):
        super().__init__()
        self.q_encoder = q_encoder
        self.encoder = (encoder or q_encoder)
        self.config = q_encoder.config

        for n, p in self.named_parameters():
            if 'q_encoder' in n:
                p.requires_grad = True
                print(n)
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
        loss_tc, loss_flop, loss_ct, loss_mr = None, None, None, None
        pos_ratio = 0 
        pos_ratio_pred = 0
        logprob = None

        if (step == 0) and (prev_output is None):
            prev_output = output = self.encoder(q_tokens, q_masks)
            reps = q_tokens
        else:
            if self.q_encoder.config.is_decoder:
                output = self.q_encoder(
                    input_ids=f_tokens,
                    attention_mask=f_masks,
                    sub_input_ids=q_tokens,
                    sub_attention_mask=q_masks,
                    sub_token_type_ids=kwargs.pop('sub_token_type_ids', None),
                )
            else:
                output = self.q_encoder(
                    input_ids=f_tokens,
                    attention_mask=f_masks,
                    token_type_ids=kwargs.pop('sub_token_type_ids', None),
                )

            candidate_tokens = f_tokens
            candidate_masks = f_masks

            # add sampling here
            action, logprob = sample_actions(output.logits, samples=2)
            # print('action', action[0][0, :, 1])
            logprob = logprob[0]
            select_tokens = torch.where(
                action[0][:, :, 1]==1, f_tokens, torch.full_like(candidate_tokens, 0)
            )

            # expand tokens
            reps = select_tokens
            batch_size, seq_size, _ = output.logits.shape
            CELoss = nn.CrossEntropyLoss()

            if d_tokens is not None:
                labels_tc = []

                n_candidates = len(d_tokens)
                for i in range(n_candidates):
                    d_output = self.encoder(d_tokens[i], d_masks[i])
                    d_indices = d_output.indices
                    d_reps.append(d_output.reps)
                    label = make_labels(d_indices, candidate_tokens, candidate_masks)
                    labels_tc.append(label)

                ## L1: token classification
                loss_tc = CELoss(output.logits.view(-1, 2), labels_tc[0].view(-1))
                pos_ratio = (labels_tc[0]>=1).sum()  / (labels_tc[0]!=-100).sum()
                pos_ratio_pred = (select_tokens>=1).sum()  / (labels_tc[0]!=-100).sum()

                ## L1: token classification
                d_reps = torch.stack(d_reps, dim=0)
                q_rep = transform_weights_to_vector(
                    select_tokens, output.logits[:, :, 1], self.config.vocab_size
                )

                scores_t = q_rep @ d_reps.view(-1, self.config.vocab_size).transpose(1, 0)   # B V x BN V
                labels_ct = torch.arange(0, batch_size, device=q_rep.device, dtype=torch.long)
                loss_ct = CELoss(scores_t, labels_ct)

        return SparseAdaptiveEncoderOutput(
            reps=reps,
            prev_out=output,
            d_reps=d_reps,
            loss_ct=loss_ct,
            loss_mr=torch.tensor([0.0]),
            loss_flop=torch.tensor([0.0]),
            loss_tc=loss_tc,
            logs={'InfoNCE': loss_ct, 'PosRatio': pos_ratio, 'PosRatioPred': pos_ratio_pred},
            logprobs=logprob,
            logits=output.logits,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.gradient_checkpointing_enable(**kwargs)
