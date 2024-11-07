import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoConfig
from modeling.biencoders.layers import AttentionLayer
from modeling.biencoders.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput

class AttentionHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        config = AutoConfig.from_pretrained(opt.retriever_name_or_path)
        config.num_attention_heads = 1
        self.attn_layer = AttentionLayer(config)
        self.encoder = encoder
        self.samples = opt.samples
        self.args = opt

    def forward(self, input_ids, attention_mask, q_out):
        f_out = self.encoder(input_ids, attention_mask)

        ## query and feedback vectors
        q_embeds = q_out.last_hidden_states
        f_embeds = f_out.last_hidden_states
        q_logits = q_out.logits
        f_logits = f_out.logits

        ## Policy model: cross-attention
        attn_out = self.attn_layer(
            hidden_states=q_embeds, 
            encoder_hidden_states=f_embeds,
            output_attention_scores=True
        )
        # B Lf Lq H  
        qf_embeds = attn_out[0] 
        qf_logits = self.encoder.model.cls(qf_embeds)
        # B Lq Lf
        attention_scores = attn_out[1].squeeze(1)

        ## transform into probability of actions 
        values = []
        actions, logprobs = self.sample_actions(states=attention_scores, attention_mask=attention_mask)

        ## convert actions into values
        for action in actions:
            value = action.max(1).values.unsqueeze(-1) * f_logits
            value, _ = torch.max(torch.log(1 + torch.relu(value)), dim=1)
            # value = (value + q_out.reps) / 2
            values.append(value)

        # Self token-level to feedback squence distillation
        loss_sft = torch.tensor([0.0])
        if self.args.sft:
            # mse
            m = torch.nn.MSELoss()
            # B Lf V / B 1 V
            pseudo_scores = torch.log(
                (1+torch.relu(f_out.logits)) / (1+torch.relu(q_out.logits).max(1).values.unsqueeze(1))
            )
            pseudo_scores = pseudo_scores.mean(-1)
            # loss_sft = m(selections, pseudo_scores)

            # ce
            # m = torch.nn.CrossEntropyLoss()
            loss_sft = m(actions[-1].view(-1), (pseudo_scores > 0).long().view(-1).to(selections.device) )

        return AdaptiveHeadOutput(
            actions=actions,
            logprobs=logprobs,
            values=values,
            output=f_out,
            loss_sft=loss_sft
        )

    def sample_actions(self, states, attention_mask=None):
        actions, logprobs = [], []
        probs = states.softmax(-1)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)

        for i in range(self.samples):
            if i == (self.samples - 1): 
                print(states.argmax(-1)[0])
                action = torch.zeros_like(states).scatter_(2, states.argmax(-1).unsqueeze(-1), 1.)
                action = action.type(states.dtype)
            else:
                action = m.sample()

            # if attention_mask is not None:
            #     action = action * attention_mask.unsqueeze(1)

            actions.append(action)
            logprob = m.log_prob(action).sum(-1)
            logprobs.append(logprob)

            # [opt1] combine them and do the pooling together
            # aggregated_logits = torch.cat(
            #     [q_out.logits, out.logits * action.unsqueeze(-1)], 
            #     dim=1
            # )
            # value, _ = torch.max(
            #     torch.log(1 + torch.relu(aggregated_logits)) * 
            #     torch.cat([q_out.mask, attention_mask], dim=1).unsqueeze(-1), 
            #     dim=1
            # )
            # [opt2] combine them before pooling feedback-aware logit
            # old_logits = torch.max(q_out.logits, dim=1).values
            # new_logits = torch.max(out.logits * action.unsqueeze(-1), dim=1).values
            # value = torch.log(1 + torch.relu(old_logits + new_logits))
            # [opt3] replace with feedback-aware logit
            # new_logits = out.logits * action.unsqueeze(-1)
            # value, _ = torch.max(
            #     torch.log(1 + torch.relu(new_logits)) * attention_mask.unsqueeze(-1),
            #     dim=1
            # )
        return actions, logprobs

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        opt, 
        encoder, 
        modifier=None,
        d_encoder=None,
        n_candidates=None
    ):
        super().__init__()
        self.opt = opt
        
        # modeling
        self.q_encoder = encoder
        self.d_encoder = encoder
        self.modifier = modifier
        self.tau = opt.tau
        self.n_candidates = n_candidates

        for n, p in self.named_parameters():
            if 'encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, q_tokens, q_masks, prev_out, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('include_n_feedbacks', n_segments)
        batch_size = q_tokens[0].size(0)
        losses_sft = []
        q_reps = []
        q_logprobs = []
        q_actions = []

        # encode query feedback
        for i in range(1, max_num_steps+1): # [1, ..., max_num_steps]
            output = self.modifier(q_tokens[i], q_masks[i], prev_out)
            q_reps += output.values
            q_logprobs += output.logprobs
            q_actions += output.actions
            losses_sft.append(output.loss_sft)

        q_reps = torch.stack(q_reps, dim=1) # B N V
        q_logprobs = torch.stack(q_logprobs, dim=1) # B N
        losses_sft = torch.stack(losses_sft, dim=0).mean()

        # loss calculation
        scores, loss_r = None, 0.0
        CELoss = nn.CrossEntropyLoss()

        # encode document if using contrastive signals
        d_reps = []
        if (d_tokens is not None):
            n_candidates = min(self.n_candidates, len(d_tokens))
            for i in range(n_candidates):
                d_rep = self.d_encoder(d_tokens[i], d_masks[i]).reps # B H
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=1) # B N_cand H

            # B Nq V x B d+ V = B Nq V x B d- V
            scores = (q_reps[:, -1, :]/self.tau) @ (d_reps[:, 0, :]).T # last query x positive context 
            labels = torch.arange(0, batch_size, dtype=torch.long, device=q_reps.device)
            loss_r = CELoss(scores, labels) # first query and document

        return SparseAdaptiveEncoderOutput(
            reps=q_reps,
            logprobs=q_logprobs,
            actions=q_actions,
            out=output.output,
            d_reps=d_reps,
            loss=loss_r,
            loss_sft=losses_sft,
            scores=scores, 
            logs={'InfoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)

# class BERTHead(nn.Module):
#     def __init__(self, opt, encoder):
#         super().__init__()
#         config = AutoConfig.from_pretrained(opt.retriever_name_or_path)
#         self.attn_layer = AttentionLayer(config)
#         self.q_encoder = encoder
#         self.samples = opt.samples
#         if opt.zero_init:
#             for p in self.attn_layer.parameters():
#                 torch.nn.init.zeros_(p)
#
#     def forward(self, input_ids, attention_mask, q_out=None):
#         q_embeds = q_out.last_hidden_state
#         out = self.q_encoder(input_ids, attention_mask)
#         f_embeds = out.last_hidden_state
#         attention_output = self.attn_layer(
#             hidden_states=q_embeds, encoder_hidden_states=f_embeds,
#             output_attention_scores=True
#         )
#         attention_scores = attention_output[1].mean(1)
#
#         aggregated_scores = torch.max(attention_scores, dim=1).values
#         logprobs, actions, values = [], [], []
#
#         # Self token-level to feedback squence distillation
#         mse = torch.nn.MSELoss()
#         pseudo_scores = torch.log( 1+torch.relu(out.logits) ) / (1+q_out.reps.unsqueeze(1))
#         pseudo_scores = pseudo_scores.max(-1).values
#         loss_mse = mse(aggregated_scores, pseudo_scores)
#
#         ## Opt1: sampling
#         probs = torch.sigmoid(aggregated_scores)
#         probs = probs * attention_mask
#         print('\nprob (pos)', (probs>=0.5).sum(-1) / probs.shape[-1])
#         print('\nprob (neg)', (probs<0.5).sum(-1) / probs.shape[-1])
#         dist = torch.distributions.Bernoulli(probs) # B Lf
#         ## actions:  [ (B Lf), (B Lf), ...]
#         ## values:   [ (B V), (B V), ...]
#         ## logprob:  [ (B), (B), ...]
#         for i in range(self.samples):
#             if i == (self.samples - 1): 
#                 # put the deterministic sample at the end
#                 action = (probs >= 0.5).clone()
#                 action = action.type(q_out.logits.dtype)
#             else:
#                 action = dist.sample()
#             actions.append(action)
#             logprob = dist.log_prob(action).sum(-1)
#             logprobs.append(logprob)
#
#             logits = self.q_encoder.model.cls(attention_output[0])
#             value, _ = torch.max(
#                 torch.log(1 + torch.relu(logits)) 
#                 * attention_mask.unsqueeze(-1), dim=1
#             )
#
#             # old_logits = torch.max(q_out.logits, dim=1).values
#             # new_logits = torch.max(out.logits * action.unsqueeze(-1), dim=1).values
#             values.append(value)
#
#         return values, logprobs, actions, out, loss_mse
