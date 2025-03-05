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
    probs = logits.softmax(-1)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)

    for i in range(samples):
        if i == 0:
            action = torch.zeros_like(logits).scatter_(2, logits.argmax(-1).unsqueeze(-1), 1.)
            action = action.type(logits.dtype)
        else:
            action = m.sample()
        actions.append(action)
        logprob = m.log_prob(action).mean(-1)
        logprobs.append(logprob)

    return actions, logprobs

# def sample_actions(logits, samples=1, attention_mask=None):
#     actions, logprobs = [], []
#     probs = logits.softmax(-1)
#     m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
#
#     for i in range(samples):
#         if i == 0:
#             action = torch.zeros_like(logits).scatter_(2, logits.argmax(-1).unsqueeze(-1), 1.)
#             action = action.type(logits.dtype)
#         else:
#             action = m.sample()
#
#         if attention_mask is not None: # action [B L 2]; logp [B L]
#             seq_logprob = m.log_prob(action) * attention_mask
#             logprob = seq_logprob.sum(-1) / attention_mask.sum(-1)
#             logprobs.append(logprob)
#             action = action * attention_mask.unsqueeze(-1)
#             actions.append(action)
#         else:
#             logprob = m.log_prob(action).mean(-1)
#             actions.append(action)
#             logprobs.append(logprob)
#
#     return actions, logprobs
#
