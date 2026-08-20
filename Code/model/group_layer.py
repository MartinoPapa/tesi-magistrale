import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import scatter

class GroupRepresentationLayer(nn.Module):
    def __init__(self, node_emb_dim, edge_feat_dim, mlp_hidden_dim=64, num_classes=1):
        super(GroupRepresentationLayer, self).__init__()
        # MLP prediction network for edge classification
        # Input: [Z_i, Z_j, l] (concatenation of node embeddings and edge features)
        in_dim = 2 * node_emb_dim + edge_feat_dim
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, X3, x, edge_index, edge_attr, y_trans=None, trans_mask=None):
        """
        Args:
            X3: Node embeddings from community-centric encoder (N x F)
            x: Original node features (N x F_orig)
            edge_index: Graph topology (2 x E)
            edge_attr: Original basic edge features (E x D)
            y_trans: Ground truth edge labels (E x 1 or E)
            trans_mask: Boolean mask indicating training edges (E,)
        Returns:
            p_trans: Predicted probabilities or logits for transactions (E x num_classes)
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
            
        p_trans = self.mlp(edge_rep)
        
        # 2. Node Aggregation Policy
        # Use predicted probabilities or ground truth for Bernoulli sampling
        if self.num_classes == 1:
            p = torch.sigmoid(p_trans.squeeze())
            # If we are training, we might use the ground truth labels for training edges
            if y_trans is not None and trans_mask is not None:
                p[trans_mask] = y_trans.squeeze()[trans_mask]
        else:
            # For multiclass, group nodes if they belong to any illicit pattern (class > 0)
            probs = torch.softmax(p_trans, dim=1)
            # Probability of being illicit is sum of probabilities of all illicit classes (1 to num_classes-1)
            # Alternatively, 1 - P(LEGIT)
            p = 1.0 - probs[:, 0].squeeze()
            if y_trans is not None and trans_mask is not None:
                p[trans_mask] = (y_trans.squeeze()[trans_mask] > 0).float()
            
        # Check for NaNs which occur if model weights are corrupted (exploding gradients)
        if torch.isnan(p).any():
            raise RuntimeError("Model predicted NaN values! The model weights have been corrupted. Please re-run the cell that initializes `model = GAGNN(...)` to reset the weights.")
            
        # Sample edge existence using Bernoulli distribution
        # Clamp just to be absolutely safe against precision errors
        is_ml_edge = torch.bernoulli(torch.clamp(p, 0.0, 1.0)).bool().cpu().numpy()
        
        # Build sparse adjacency matrix of ML edges
        N = x.size(0)
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
        group_mapping = torch.tensor(group_mapping, dtype=torch.long, device=x.device)
        
        # 3. Generate New Graph \hat{G} and features \hat{X}
        # The value of the grouped node features is the mean of the ORIGINAL feature values of the nodes
        X_hat = torch.zeros((num_groups, x.size(1)), device=x.device)
        
        # Using scatter_mean to average original node features belonging to the same group
        X_hat.scatter_reduce_(0, group_mapping.unsqueeze(1).expand(-1, x.size(1)), x, reduce='mean', include_self=False)
        
        # Construct \hat{E}
        # Edges between groups
        row_hat = group_mapping[row]
        col_hat = group_mapping[col]
        
        # Remove self-loops (edges within the same group)
        mask = row_hat != col_hat
        edge_index_hat = torch.stack([row_hat[mask], col_hat[mask]], dim=0)
        
        # Remove duplicate edges in the new graph to keep it clean
        edge_index_hat = torch.unique(edge_index_hat, dim=1)
        
        return p_trans, X_hat, edge_index_hat, group_mapping
