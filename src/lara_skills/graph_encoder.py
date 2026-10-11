"""GraphSAGE feature/neighbor layer with a defined empty-neighborhood path."""

import torch
from torch import nn


class SelfAndNeighborLayer(nn.Module):
    """Weighted mean aggregation; isolated nodes use their self features only.

    neighbor_mean_weights is a torch sparse COO/CSR matrix with zero rows for
    isolated nodes. Nonempty rows must sum to one. The neighbor transform has
    no bias so an empty neighborhood contributes exactly zero.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features, neighbor_mean_weights):
        if neighbor_mean_weights.shape != (features.shape[0], features.shape[0]):
            raise ValueError("Adjacency and features use different node dimensions")
        if neighbor_mean_weights.layout not in {torch.sparse_coo, torch.sparse_csr}:
            raise ValueError("Expected a sparse neighbor aggregation matrix")
        neighborhood = torch.sparse.mm(neighbor_mean_weights, features)
        return torch.relu(self.self_projection(features) + self.neighbor_projection(neighborhood))
