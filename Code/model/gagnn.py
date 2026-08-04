import torch
import torch.nn as nn
from model.encoder import CommunityCentricEncoder
from model.group_layer import GroupRepresentationLayer

class GAGNN(nn.Module):
    def __init__(self, node_in_dim, edge_feat_dim, hidden_dim, out_dim, heads=5, beta=0.44):
        """
        GAGNN Model
        Args:
            node_in_dim: Dimension of input node features X^(1)
            edge_feat_dim: Dimension of edge features l
            hidden_dim: Hidden dimension for GAT layers
            out_dim: Output dimension of node embeddings X^(3)
            heads: Number of attention heads for GAT
            beta: Trade-off parameter for eMRF similarity
        """
        super(GAGNN, self).__init__()
        
        self.training_losses = []
        
        # 1. Base community-centric encoder
        self.encoder = CommunityCentricEncoder(
            in_channels=node_in_dim, 
            hidden_channels=hidden_dim, 
            out_channels=out_dim, 
            heads=heads, 
            beta=beta
        )
        
        # 2. Group Representation Layer
        self.group_layer = GroupRepresentationLayer(
            node_emb_dim=out_dim, 
            edge_feat_dim=edge_feat_dim
        )
        
        # 3. Group-level community-centric encoder
        # Receives \hat{X} which has dimension out_dim
        self.group_encoder = CommunityCentricEncoder(
            in_channels=out_dim, 
            hidden_channels=hidden_dim, 
            out_channels=out_dim, 
            heads=heads, 
            beta=beta
        )

    def forward(self, x, edge_index, edge_attr, y, num_edges=None):
        """
        Args:
            x: Node features (N x F)
            edge_index: Adjacency list (2 x E)
            edge_attr: Edge features (E x D)
            y: Node labels (N x 1)
            num_edges: Total edges for eMRF calculation
        """
        # Step 1: Base node-level encoding
        X3, p_node = self.encoder(x, edge_index, y, num_edges)
        
        # Step 2: Group representation and graph reconstruction
        p_trans, X_hat, edge_index_hat, group_mapping = self.group_layer(X3, edge_index, edge_attr)
        
        # Calculate group ground-truth labels y_group
        # y_group_i = 1 if group_i contains more than 1 node, else 0
        num_groups = X_hat.size(0)
        group_sizes = torch.zeros(num_groups, dtype=torch.float, device=x.device)
        group_sizes.scatter_add_(0, group_mapping, torch.ones_like(group_mapping, dtype=torch.float))
        
        y_group = (group_sizes > 1).float().unsqueeze(1) # (Num_Groups x 1)
        
        # Step 3: Group-level encoding
        # Note: the paper feeds \hat{X} and \hat{G} back into the community-centric encoder.
        # eMRF needs labels for the groups. We use y_group for the eMRF label similarity step.
        _, p_group = self.group_encoder(X_hat, edge_index_hat, y_group.squeeze(), num_edges)
        
        return p_node, p_trans, p_group, y_group

    def save(self, path):
        """Saves the model weights and training losses to the specified path."""
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'training_losses': self.training_losses
        }, path)

    def load(self, path):
        """Loads the model weights and training losses from the specified path."""
        import torch
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.eval()
        
    def plot_training_losses(self):
        """Plots the training losses recorded during training."""
        import matplotlib.pyplot as plt
        if not self.training_losses:
            print("No training losses to plot.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Training Iterations / Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
