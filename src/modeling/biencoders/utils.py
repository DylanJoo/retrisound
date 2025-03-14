import numpy as np
import torch.nn as nn
import torch

def make_labels(d_tokens, candidate_tokens, candidate_masks, q_tokens=None):
    binary_matrix = torch.zeros_like(candidate_tokens)
    for i in range(len(d_tokens)):
        binary_matrix[i] = (candidate_tokens[i].unsqueeze(1) == d_tokens[i]).any(dim=1)
        if q_tokens is not None:
            binary_matrix[i] += (candidate_tokens[i].unsqueeze(1) == q_tokens[i]).any(dim=1)

    # mask unused token
    mask_matrix = torch.full_like(binary_matrix, -100)
    binary_matrix = torch.where(candidate_masks==0, mask_matrix, binary_matrix)
    return binary_matrix.to(candidate_tokens.device)

def transform_weights_to_vector(inputs, weights, vocab_size):
    vector = torch.zeros(inputs.size(0), vocab_size, dtype=weights.dtype).to(inputs.device)
    vector = vector.scatter(1, inputs, weights)
    return vector

def sample_actions(logits, samples=1, attention_mask=None):
    actions, logprobs = [], []
    if logits.size(-1) == 1:
        probs = torch.zeros( (logits.size(0), logits.size(1), 2), device=logits.device) # (B L 2)
        probs[:, :, 1] += logits.squeeze(-1).softmax(-1)
        probs[:, :, 0] += 1 - probs[:, :, 1]
    else:
        probs = logits.softmax(-1) # (B L 2)

    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)

    for i in range(samples):
        if i == 0: # deterministic
            action = torch.zeros_like(probs).scatter_(2, probs.argmax(-1).unsqueeze(-1), 1.)
            action = action.type(probs.dtype)
        else: # sampled
            action = m.sample()

        if attention_mask is not None:
            action = action * attention_mask
            seq_logprob = m.log_prob(action)
            seq_logprob = seq_logprob * attention_mask
            logprob = seq_logprob.sum(-1) / attention_mask.sum(-1)
        else:
            logprob = m.log_prob(action).mean(-1)

        logprobs.append(logprob)
        actions.append(action)
    return actions, logprobs

def sample_actions_dist(token_indices, scores, samples=1, attention_mask=None, topk=5):
    # the sorted
    action_d, logprob_d = multiple_sample_and_log_probability(
        scores=scores.squeeze(-1),
        sample_size=1, 
        batch=True,
        topk_estimate=topk
    ) # (B N L) (B N)

    actions, logprobs = multiple_sample_and_log_probability(
        scores=scores.squeeze(-1),
        sample_size=samples, 
        batch=True,
        topk_estimate=topk
    ) # (B N L) (B N)

    actions[:, 0, :] = action_d[:, 0, :]
    logprobs[:, 0] = logprob_d[:, 0]

    selections = []
    for i in range(samples):
        # map the sampled positions to the original indices (actions)
        selection = torch.gather(token_indices, 1, actions[:, i, :])[:, :topk]
        selections.append(selection)

    return actions, logprobs, selections

def transform_ids_to_vector(inputs, tokenizer=None, count=False):
    vector = torch.zeros(inputs.size(0), tokenizer.vocab_size).to(inputs.device)
    if count:
        vector = vector.scatter_add(1, inputs, torch.ones_like(inputs, dtype=vector.dtype))
    else:
        vector = vector.scatter(1, inputs, 1)

    # clean the added tokens
    for tok, idx in tokenizer.get_added_vocab().items():
        vector[:, idx] = 0
    return vector

def multiple_sample_and_log_probability(
    scores, 
    sample_size, 
    return_prob=True, 
    batch=False,
    sort=False,
    baseline=False,
    topk_estimate=None,
    tau=1
):
    assert scores.dim() == 2
    batch_size, candidiate_size = scores.size(0), scores.size(1)
    subtracts = scores.new_zeros((batch_size, sample_size, candidiate_size))
    batch_index = torch.arange(
        batch_size, device=scores.device).unsqueeze(1).expand(
        batch_size, sample_size)
    sample_index = torch.arange(
        sample_size, device=scores.device).expand(
        batch_size, sample_size)
    if return_prob:
        log_probs = torch.zeros_like(subtracts, dtype=torch.float)
    rankings = []
    topk_estimate = (topk_estimate or scores.size(1))

    # real sampling
    for j in range(scores.size(1)):
        probs = nn.functional.softmax(
            (scores.unsqueeze(1) - subtracts)/tau, dim=-1) + 1e-10
        if (sort or j > topk_estimate):
            posj = torch.argmax(
                probs.reshape(batch_size * sample_size, -1),
                1
            ).squeeze(-1).reshape(batch_size, sample_size)
        elif baseline:
            posj = torch.tensor(
                [j] * (batch_size * sample_size)
            ).reshape(batch_size, sample_size)
        else:
            posj = torch.multinomial(
                probs.reshape(batch_size * sample_size, -1),
                1
            ).squeeze(-1).reshape(batch_size, sample_size)
        rankings.append(posj)
        if return_prob:
            log_probs[:, :, j] = probs[batch_index,
                                       sample_index, posj].log()
        subtracts[batch_index, sample_index,
                  posj] = scores[batch_index, posj] + 1e6
    rankings = torch.stack(rankings, dim=-1)
    # rankings = rankings[:, :, :topk_estimate]
    if return_prob:
        log_probs = log_probs[:, :, :topk_estimate].mean(dim=-1)
        # log_probs = log_probs[:, :, :topk_estimate].sum(dim=-1)
        return rankings, log_probs
    else:
        return rankings
