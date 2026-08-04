import torch
import torch.nn as nn
import torch.nn.functional as F

class eMRFLayer(nn.Module):
    def __init__(self, in_channels, beta=0.44):
        super(eMRFLayer, self).__init__()
        self.beta = beta
        # Learnable weight matrix H^(2)
        self.H2 = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, X2, edge_index, y, num_edges=None):
        """
        Forward pass for the eMRF layer.
        In a full graph scenario, this calculates pairwise potentials dense-ly.
        For large graphs, this should be applied on subgraphs or mini-batches.
        
        Args:
            X2: Node embeddings from GAT layers (N x F)
            edge_index: Adjacency list (2 x E)
            y: True labels of nodes (N)
            num_edges: Total number of edges in the full graph (for modularity calc)
        Returns:
            X3: Updated node embeddings (N x F)
        """
        N = X2.size(0)
        E = num_edges if num_edges is not None else edge_index.size(1) // 2
        if E == 0:
            E = 1 # prevent division by zero
            
        # 1. Topological similarity epsilon(v_i, v_j)
        # Calculate degrees
        deg = torch.zeros(N, dtype=torch.float, device=X2.device)
        deg.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        
        # epsilon = (d_i * d_j) / 2E - a_ij
        deg_matrix = deg.unsqueeze(1) @ deg.unsqueeze(0)  # N x N
        epsilon = deg_matrix / (2 * E + 1e-8)
        
        # Subtract a_ij
        adj = torch.zeros((N, N), dtype=torch.float, device=X2.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        epsilon = epsilon - adj
        
        # 2. Attribute similarity zeta(v_i, v_j)
        # Cosine similarity between all pairs
        X2_norm = F.normalize(X2, p=2, dim=1)
        zeta = torch.mm(X2_norm, X2_norm.t())  # N x N
        
        # Regularization R_i: assuming MinMax normalization across rows to match scale
        zeta_min = zeta.min(dim=1, keepdim=True)[0]
        zeta_max = zeta.max(dim=1, keepdim=True)[0]
        R_zeta = (zeta - zeta_min) / (zeta_max - zeta_min + 1e-8)
        
        # 3. gamma(v_i, v_j)
        gamma = self.beta * epsilon + (1 - self.beta) * R_zeta
        
        # 4. Pairwise potential Psi(v_i, v_j) = -1^sigma * gamma
        # sigma(v_i, v_j) = 1 if y_i == y_j else 0
        y_i = y.unsqueeze(1)
        y_j = y.unsqueeze(0)
        sigma = (y_i == y_j).float()
        
        # -1^sigma: if sigma=1 -> -1, if sigma=0 -> 1
        sign = torch.where(sigma == 1.0, torch.tensor(-1.0, device=X2.device), torch.tensor(1.0, device=X2.device))
        
        Gamma = sign * gamma  # N x N matrix
        
        # 5. Update X3 = X2 - Gamma * X2 * H2
        # Mean-field approximation step
        transformed_X2 = self.H2(X2)  # N x F
        X3 = X2 - torch.mm(Gamma, transformed_X2)
        
        return X3
