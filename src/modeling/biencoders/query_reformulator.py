import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput
from modeling.biencoders.utils import (
    make_labels, transform_weights_to_vector, sample_actions_dist, transform_ids_to_vector
)

class SparseAdaptiveRetriever(nn.Module):
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
        self.num_samples = kwargs.get('num_samples')

        if kwargs.get('sample_type') == 'deterministic':
            self.selected_sample = 0
        if kwargs.get('sample_type') == 'random':
            self.selected_sample = 1

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
        d_reps = []
        loss_tc, loss_ct, loss_mr = None, None, None
        pos_ratio_truth = 0 
        pos_ratio = 0
        # logprob = None
        sampled_reps = []
        logprobs = []
        tokenizer = kwargs.get('tokenizer')

        if (step == 0) and (prev_output is None):
            prev_output = output = self.encoder(q_tokens, q_masks)
            rep = transform_ids_to_vector(q_tokens, tokenizer, count=True)
        else:
            output = self.q_encoder(
                input_ids=f_tokens,
                attention_mask=f_masks,
                token_type_ids=kwargs.pop('sub_token_type_ids', None),
            )

            candidate_tokens = f_tokens
            candidate_masks = f_masks

            # add sampling here
            actions, logprobs, selections = sample_actions_dist(
                candidate_tokens, output.logits, samples=self.num_samples, topk=30
            )

            logprobs = logprobs.transpose(1, 0) # N B
            for selection in selections:
                rep = transform_ids_to_vector(selection, tokenizer, count=True)
                sampled_reps.append(rep)

            # expand tokens
            batch_size, seq_size, _ = output.logits.shape
            CELoss = nn.CrossEntropyLoss()
            MSELoss = nn.MSELoss(reduction='none')

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
                loss_tc = MSELoss(output.logits.squeeze(-1), labels_tc[0].float())
                loss_tc = loss_tc * (labels_tc[0]!=-100).float()
                loss_tc = loss_tc.sum() / (labels_tc[0]!=-100).sum()

                pos_ratio_truth = (labels_tc[0]>=1).sum()  / (labels_tc[0]!=-100).sum()
                pos_ratio = selections[0].numel() / (labels_tc[0]!=-100).sum()

                ## L2: contrastive learning
                d_reps = torch.stack(d_reps, dim=0)
                q_rep = transform_weights_to_vector(
                    actions[:, 0, :], output.logits[:, :, 0], self.config.vocab_size
                )

                scores_t = q_rep @ d_reps.view(-1, self.config.vocab_size).transpose(1, 0)   # B V x BN V
                labels_ct = torch.arange(0, batch_size, device=q_rep.device, dtype=torch.long)
                loss_ct = CELoss(scores_t, labels_ct)

        return SparseAdaptiveEncoderOutput(
            reps=sampled_reps[self.selected_sample] if step > 0 else rep,
            logprobs=logprobs[:, self.selected_sample] if step > 0 else None,
            prev_out=output,
            d_reps=d_reps,
            loss_ct=loss_ct,
            loss_mr=torch.tensor([0.0]),
            loss_flop=torch.tensor([0.0]),
            loss_tc=loss_tc,
            logs={'InfoNCE': loss_ct, 'PosRatioTruth': pos_ratio_truth, 'PosRatio': pos_ratio},
            samples={"logprobs": logprobs, "reps": sampled_reps},
            logits=output.logits,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.gradient_checkpointing_enable(**kwargs)
