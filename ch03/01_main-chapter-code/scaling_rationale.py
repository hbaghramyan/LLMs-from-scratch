import torch

# Example attention scores for one query against six keys
attn_scores_2 = torch.tensor([2.0, 3.0, 1.5, 4.0, 3.5, 2.5])
d_k = 6  # Dimensionality

# Apply scaling
scaled_attn_scores = attn_scores_2 / d_k**0.5

# Apply softmax to scaled attention scores
softmax_attn_scores = torch.softmax(scaled_attn_scores, dim=-1)

# Apply softmax directly without scaling
softmax_attn_scores_noscale = torch.softmax(attn_scores_2, dim=-1)

print("Unscaled Attention Scores:", attn_scores_2)
print(
    "Softmax of Attention Scores without Scaling:", softmax_attn_scores_noscale
)
print("Softmax of Scaled Attention Scores:", softmax_attn_scores)
print("Scaled Attention Scores:", scaled_attn_scores)
