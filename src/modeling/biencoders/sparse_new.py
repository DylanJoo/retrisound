import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoConfig
from modeling.biencoders.layers import AttentionLayer
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput

class AttentionHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        config = AutoConfig.from_pretrained(opt.retriever_name_or_path)
        config.num_attention_heads = 1
        self.attn_layer = AttentionLayer(config)
        self.encoder = encoder.eval()
        self.samples = opt.samples
        self.args = opt
        # if opt.zero_init:
        #     self.attn_layer.query.weight.data.zero_()
        #     self.attn_layer.query.bias.data.zero_()
        #     self.attn_layer.key.weight.data.zero_()
        #     self.attn_layer.key.bias.data.zero_()
        #     self.attn_layer.value.weight.data.zero_()
        #     self.attn_layer.value.bias.data.zero_()

    def forward(self, input_ids, attention_mask, q_out, ignore_value_projection=True):
        device = input_ids.device
        f_out = self.encoder(input_ids, attention_mask)

        ## query and feedback vectors
        q_logits = q_out.logits
        f_logits = f_out.logits

        ## Policy model: cross-attention
        attn_out = self.attn_layer(
            hidden_states=q_out.last_hidden_states, 
            attention_mask=q_out.mask,
            encoder_hidden_states=f_out.last_hidden_states,
            encoder_attention_mask=f_out.mask,
            output_attention_scores=True,
            ignore_value_projection=ignore_value_projection
        )
        # B Lf Lq H  
        if self.attn_layer.num_attention_heads == 1:
            qf_attentions = attn_out[1].squeeze(1)
        else:
            qf_attentions = attn_out[1].max(1).values
        qf_embeds = attn_out[0] 
        qf_logits = self.encoder.model.cls(qf_embeds)

        ## transform into probability of actions 
        values = []
        if self.samples > 1:
            actions, logprobs = self.sample_actions(states=qf_attentions, attention_mask=attention_mask)

            for action in actions:
                # deterministic
                value, _ = torch.max(torch.log(1 + torch.relu(q_out.logits + qf_logits)), dim=1)
                # sampled
                values.append(value)
        else:
            actions = []
            logprobs = [torch.tensor([0.0] * f_logits.size(0)).to(device)] 
            # early fusion (logits)
            # value, _ = torch.max(torch.log(1 + torch.relu(f_logits)), dim=1)
            # value = q_out.reps + torch.max(torch.log(1 + torch.relu(qf_logits)), dim=1).values
            value = torch.max(torch.log(1 + torch.relu(qf_logits)), dim=1).values
            print('nonzero', (value > 0).sum(-1))

            # late fusion (logits)
            values.append(value)

        # Self token-level to feedback squence distillation
        loss_sft = torch.tensor([0.0]).to(device)
        if self.args.sft:
            # mse
            m = torch.nn.MSELoss()
            resid_scores = torch.log(1+torch.relu(qf_logits))
            pseudo_scores = torch.log(1+torch.relu(f_out.logits))
            pseudo_scores = pseudo_scores / q_out.reps.unsqueeze(1)
            loss_sft = m(resid_scores, pseudo_scores)

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
                action = torch.zeros_like(states).scatter_(2, states.argmax(-1).unsqueeze(-1), 1.)
                action = action.type(states.dtype)
            else:
                action = m.sample()

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
        self.d_encoder = encoder
        self.modifier = modifier
        self.tau = opt.tau
        self.n_candidates = n_candidates

        for n, p in self.named_parameters():
            if 'encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        print(self.modifier)

    def forward(self, q_tokens, q_masks, prev_out, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('include_n_feedbacks', n_segments)
        batch_size = q_tokens[0].size(0)
        losses_sft = []
        q_reps = []
        q_logprobs = []
        q_actions = []

        # encode query feedback
        for i in range(0, max_num_steps+1): # [1, ..., max_num_steps]
            if i == 0:
                pass
            else:
                output = self.modifier(q_tokens[i], q_masks[i], prev_out, ignore_value_projection=True) 
                q_reps += output.values
                q_logprobs += output.logprobs
                q_actions += output.actions
                losses_sft.append(output.loss_sft)

        q_reps = torch.stack(q_reps, dim=1) # B N V
        q_logprobs = torch.stack(q_logprobs, dim=1) # B N
        losses_sft = torch.stack(losses_sft, dim=0).mean()

        # loss calculation
        scores, loss_ct = None, 0.0
        CELoss = nn.CrossEntropyLoss()
        MRLoss = nn.MarginRankingLoss()

        # encode document if using contrastive signals
        d_reps = []
        if d_tokens is not None:
            n_candidates = min(self.n_candidates, len(d_tokens))
            for i in range(n_candidates):
                d_rep = self.d_encoder(d_tokens[i], d_masks[i]).reps # B H
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=0) # N_cand B H

            # B Nq V x B d+ V = B V x B V N --last query x positive context 
            scores_0 = prev_out.reps @ d_reps.view(-1, q_reps.size(-1)).permute(1, 0)
            scores_t = q_reps[:, -1, :] @ d_reps.view(-1, q_reps.size(-1)).permute(1, 0)
            labels = torch.arange(0, batch_size, dtype=torch.long, device=q_reps.device)
            print('q0', scores_0[0].softmax(-1).tolist())
            print('qt', scores_t[0].softmax(-1).tolist())
            loss_ct_0 = CELoss(scores_0, labels) 
            loss_ct_t = CELoss(scores_t, labels) 

            labels_mr = torch.ones(batch_size, dtype=torch.long, device=q_reps.device)
            loss_mr = MRLoss(scores_t.diag(), scores_0.diag(), labels_mr)

        return SparseAdaptiveEncoderOutput(
            reps=q_reps,
            logprobs=q_logprobs,
            actions=q_actions,
            out=output.output,
            d_reps=d_reps,
            loss_ct=loss_ct_t + loss_mr, 
            loss_sft=losses_sft,
            scores=scores, 
            logs={'InfoNCE': loss_ct_t}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)
