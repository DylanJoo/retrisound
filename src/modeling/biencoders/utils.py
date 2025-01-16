import torch

def make_labels(d_tokens, candidate_tokens, candidate_masks, q_tokens=None):
    binary_matrix = torch.zeros_like(candidate_tokens)
    for i in range(len(d_tokens)):
        binary_matrix[i] = (candidate_tokens[i].unsqueeze(1) == d_tokens[i]).any(dim=1)
        if q_tokens is not None:
            binary_matrix[i] = (candidate_tokens[i].unsqueeze(1) == q_tokens[i]).any(dim=1)

    # mask unused token
    mask_matrix = torch.full_like(binary_matrix, -100)
    binary_matrix = torch.where(candidate_masks==0, mask_matrix, binary_matrix)
    return binary_matrix.to(candidate_tokens.device)

    # random sample negatives
    # rand_matrix = torch.randint(0, 2, binary_matrix.shape).to(candidate_tokens.device)
    # rand_matrix = rand_matrix * (binary_matrix == 0) 
    # binary_matrix = torch.where(rand_matrix==1, 0, -100)

def transform_weights_to_vector(inputs, weights, vocab_size):
    vector = torch.zeros(inputs.size(0), vocab_size, dtype=weights.dtype).to(inputs.device)
    vector = vector.scatter(1, inputs, weights)
    return vector

def sample_actions(logits, samples=1, attention_mask=None):
    actions, logprobs = [], []
    probs = logits.softmax(-1)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)

    for i in range(samples):
        if i == (samples - 1): 
            action = torch.zeros_like(logits).scatter_(2, logits.argmax(-1).unsqueeze(-1), 1.)
            action = action.type(logits.dtype)
        else:
            action = m.sample()
        actions.append(action)
        logprob = m.log_prob(action).mean(-1)
        logprobs.append(logprob)

    return actions, logprobs

