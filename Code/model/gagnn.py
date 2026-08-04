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

        # Store constructor arguments so they can be saved/restored from checkpoint
        self._init_kwargs = dict(
            node_in_dim=node_in_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads=heads,
            beta=beta
        )

        self.training_losses = []
        self.val_losses = []
        
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

    def forward(self, x, edge_index, edge_attr, num_edges=None):
        """
        Args:
            x: Node features (N x F)
            edge_index: Adjacency list (2 x E)
            edge_attr: Edge features (E x D)
            num_edges: Total edges for eMRF calculation
        """
        # Step 1: Base node-level encoding
        X3, p_node = self.encoder(x, edge_index, num_edges)
        
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
        _, p_group = self.group_encoder(X_hat, edge_index_hat, num_edges)
        
        return p_node, p_trans, p_group, y_group

    def save(self, path):
        """Saves the model weights, training losses, and architecture config to the specified path."""
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'training_losses': self.training_losses,
            'val_losses': self.val_losses,
            'init_kwargs': self._init_kwargs
        }, path)

    @classmethod
    def load_saved(cls, path, **fallback_kwargs):
        """
        Alternative constructor: instantiates and loads a GAGNN model directly from
        a checkpoint file without needing to know the architecture hyperparameters.

        For checkpoints saved with the new save() (which includes 'init_kwargs'),
        no extra arguments are needed. For older checkpoints that lack 'init_kwargs',
        pass the architecture hyperparameters as keyword arguments, e.g.:
            GAGNN.load_saved(path, node_in_dim=8, edge_feat_dim=4, ...)

        Args:
            path:            Path to the .pt checkpoint saved by GAGNN.save().
            **fallback_kwargs: Architecture kwargs used when the checkpoint does
                             not contain 'init_kwargs' (backwards compatibility).

        Returns:
            A GAGNN instance in eval() mode with weights and losses restored.
        """
        import torch
        checkpoint = torch.load(path, map_location='cpu')
        init_kwargs = checkpoint.get('init_kwargs', fallback_kwargs)
        if not init_kwargs:
            raise KeyError(
                f"Checkpoint '{path}' does not contain architecture info ('init_kwargs'). "
                "Pass the model hyperparameters as keyword arguments to load_saved()."
            )
        model = cls(**init_kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.training_losses = checkpoint.get('training_losses', [])
        model.val_losses = checkpoint.get('val_losses', [])
        model.eval()
        return model

    def load(self, path):
        """Loads the model weights and training losses from the specified path."""
        import torch
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.eval()
        
    def plot_training_history(self):
        """Plots the training and validation losses recorded during training."""
        import matplotlib.pyplot as plt
        if not self.training_losses and not getattr(self, 'val_losses', []):
            print("No training history to plot.")
            return
            
        plt.figure(figsize=(10, 6))
        if self.training_losses:
            plt.plot(self.training_losses, label='Training Loss', color='blue', linewidth=2)
        if hasattr(self, 'val_losses') and self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='orange', linewidth=2)
            
        plt.xlabel('Training Iterations / Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
