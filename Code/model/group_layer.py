import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import scatter

class GroupRepresentationLayer(nn.Module):
    def __init__(self, node_emb_dim, edge_feat_dim):
        super(GroupRepresentationLayer, self).__init__()
        # MLP prediction network for edge classification
        # Input: [Z_i, Z_j, l] (concatenation of node embeddings and edge features)
        in_dim = 2 * node_emb_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X3, edge_index, edge_attr):
        """
        Args:
            X3: Node embeddings from community-centric encoder (N x F)
            edge_index: Graph topology (2 x E)
            edge_attr: Original basic edge features (E x D)
        Returns:
            p_trans: Predicted probabilities for transactions (E x 1)
            X_hat: Grouped node features
            edge_index_hat: Grouped edge index
            group_mapping: Mapping from original nodes to groups
        """
        # 1. Edge Representation and Prediction
        row, col = edge_index
        Z_i = X3[row]
        Z_j = X3[col]
        
        if edge_attr is not None:
            edge_rep = torch.cat([Z_i, Z_j, edge_attr], dim=1)
        else:
            edge_rep = torch.cat([Z_i, Z_j], dim=1)
            
        p_trans = torch.sigmoid(self.mlp(edge_rep))
        
        # 2. Node Aggregation Policy
        # If p_trans > 0.5, we believe nodes are conducting money laundering together
        # We find connected components based on these predicted positive edges
        N = X3.size(0)
        is_ml_edge = (p_trans.squeeze() > 0.5).cpu().numpy()
        
        # Build sparse adjacency matrix of ML edges
        row_np = row.cpu().numpy()[is_ml_edge]
        col_np = col.cpu().numpy()[is_ml_edge]
        data = np.ones_like(row_np)
        
        adj = sp.coo_matrix((data, (row_np, col_np)), shape=(N, N))
        
        # Make undirected for connected components
        adj = adj + adj.T
        
        # Find groups (connected components)
        num_groups, group_mapping = sp.csgraph.connected_components(
            adj, directed=False, return_labels=True
        )
        group_mapping = torch.tensor(group_mapping, dtype=torch.long, device=X3.device)
        
        # 3. Generate New Graph \hat{G} and features \hat{X}
        # The paper uses element-wise mean for \hat{v}_i
        X_hat = torch.zeros((num_groups, X3.size(1)), device=X3.device)
        
        # Using scatter_mean to average node features belonging to the same group
        X_hat.scatter_reduce_(0, group_mapping.unsqueeze(1).expand(-1, X3.size(1)), X3, reduce='mean', include_self=False)
        
        # Construct \hat{E}
        # Edges between groups
        row_hat = group_mapping[row]
        col_hat = group_mapping[col]
        
        # Remove self-loops (edges within the same group)
        mask = row_hat != col_hat
        edge_index_hat = torch.stack([row_hat[mask], col_hat[mask]], dim=0)
        
        # Remove duplicate edges in the new graph to keep it clean (optional but good practice)
        # We can use torch_geometric.utils.coalesce if available, but manual unique is fine
        edge_index_hat = torch.unique(edge_index_hat, dim=1)
        
        return p_trans, X_hat, edge_index_hat, group_mapping
