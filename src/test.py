import torch
import torch.nn as nn

# Example input
token_indices = torch.tensor([1000, 2000])  # Token indices from tokenizer
input_embeddings = torch.randn(len(token_indices), 768, requires_grad=True)  # Token embeddings

# Define MLP for token weight prediction
mlp = nn.Sequential(
    nn.Linear(768, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # Predict scalar weight
)

# Predict token weights
token_weights = mlp(input_embeddings).squeeze()  # Shape: [n_tokens]

# Create vocabulary vector (30522-dimensional)
vocab_size = 30522
vocab_vector = torch.zeros(vocab_size)
vocab_vector.index_add_(0, token_indices, token_weights)

# Example similarity computation
document_vector = torch.randn(30522, requires_grad=True)  # Document representation
similarity = torch.dot(vocab_vector, document_vector)

print(similarity)

# Backward pass
similarity.backward()

