import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from model.emrf import eMRFLayer

class CommunityCentricEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=5, beta=0.44, nn_t_hidden_dim=128):
        super(CommunityCentricEncoder, self).__init__()
        # 1. Two layers of GAT
        # The paper mentions setting attention head k to 5
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False)
        
        # 2. eMRF Layer
        self.emrf = eMRFLayer(out_channels, beta=beta)
        
        # 3. Node classification MLP
        # Generates X(4) = sigmoid(NN_t(X(3), W_t))
        # The paper says NN_t is set at 128 to extend the original node feature 
        self.nn_t = nn.Sequential(
            nn.Linear(out_channels, nn_t_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(nn_t_hidden_dim, 1) # Binary classification output (logits)
        )
        
        # Coarse classifier to generate label probabilities for eMRF
        self.coarse_classifier = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, num_edges=None):
        """
        Args:
            x: Node attributes matrix X^(1) (N x F)
            edge_index: Adjacency list (2 x E)
            num_edges: Total edges for eMRF modularity calculation
        Returns:
            X3: Node embeddings (N x out_channels)
            X4: Node classification logits (N x 1)
        """
        # GAT layers
        x2 = self.gat1(x, edge_index)
        x2 = torch.relu(x2)
        x2 = self.gat2(x2, edge_index) # This is X^(2)
        
        # eMRF Layer requires a coarse prediction of the label
        p_coarse = torch.sigmoid(self.coarse_classifier(x2))
        x3 = self.emrf(x2, edge_index, p_coarse, num_edges=num_edges) # This is X^(3)
        
        # Classification (return logits directly for numerical stability in BCEWithLogitsLoss)
        x4 = self.nn_t(x3) # This is X^(4)
        
        return x3, x4
