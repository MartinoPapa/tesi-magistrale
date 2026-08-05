import torch
import torch.nn as nn
from model.encoder import CommunityCentricEncoder
from model.group_layer import GroupRepresentationLayer

class GAGNN(nn.Module):
    def __init__(self, node_in_dim, edge_feat_dim, hidden_dim, out_dim, heads=5, beta=0.44, mlp_hidden_dim=64, nn_t_hidden_dim=128, minibatches=True):
        """
        GAGNN Model
        Args:
            node_in_dim: Dimension of input node features X^(1)
            edge_feat_dim: Dimension of edge features l
            hidden_dim: Hidden dimension for GAT layers
            out_dim: Output dimension of node embeddings X^(3)
            heads: Number of attention heads for GAT
            beta: Trade-off parameter for eMRF similarity
            mlp_hidden_dim: Hidden dimension for the edge classification MLP
            nn_t_hidden_dim: Hidden dimension for node classification MLP
            minibatches: If True, indicates the model is trained with minibatches
        """
        super(GAGNN, self).__init__()

        # Store constructor arguments so they can be saved/restored from checkpoint
        self._init_kwargs = dict(
            node_in_dim=node_in_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads=heads,
            beta=beta,
            mlp_hidden_dim=mlp_hidden_dim,
            nn_t_hidden_dim=nn_t_hidden_dim,
            minibatches=minibatches
        )

        self.training_losses = []
        self.val_losses = []
        
        # 1. Base community-centric encoder
        self.encoder = CommunityCentricEncoder(
            in_channels=node_in_dim, 
            hidden_channels=hidden_dim, 
            out_channels=out_dim, 
            heads=heads, 
            beta=beta,
            nn_t_hidden_dim=nn_t_hidden_dim
        )
        
        # 2. Group Representation Layer
        self.group_layer = GroupRepresentationLayer(
            node_emb_dim=out_dim, 
            edge_feat_dim=edge_feat_dim,
            mlp_hidden_dim=mlp_hidden_dim
        )

        # 3. Linear projection to bridge group features (out_dim) back to node_in_dim
        # so that the base encoder weights can be re-used for group-level encoding (paper Eq. 12)
        self.group_proj = nn.Linear(out_dim, node_in_dim)

    def forward(self, x, edge_index, edge_attr, num_edges=None, y_node=None):
        """
        Args:
            x:          Node features (N x F)
            edge_index: Adjacency list (2 x E)
            edge_attr:  Edge features (E x D)
            num_edges:  Total edges for eMRF calculation
            y_node:     Optional ground-truth node labels (N,) — float in [0,1].
                        When provided (training), y_group is derived from ground truth
                        by propagating labels through group_mapping (paper Eq. definition).
                        When None (inference), falls back to group-size > 1 heuristic.
        """
        # Step 1: Base node-level encoding
        X3, p_node = self.encoder(x, edge_index, num_edges)
        
        # Step 2: Group representation and graph reconstruction
        p_trans, X_hat, edge_index_hat, group_mapping = self.group_layer(X3, edge_index, edge_attr)
        
        # Step 3: Compute y_group
        num_groups = X_hat.size(0)
        if y_node is not None:
            # Paper definition: group is ML if ANY constituent node has a ground-truth ML label.
            # y_node may be soft [0,1] — threshold at 0.5 to obtain binary assignment.
            node_is_ml = (y_node.view(-1) > 0.5).float()
            y_group_raw = torch.zeros(num_groups, dtype=torch.float, device=x.device)
            y_group_raw.scatter_reduce_(0, group_mapping, node_is_ml, reduce='amax', include_self=True)
            y_group = y_group_raw.unsqueeze(1)  # (Num_Groups x 1)
        else:
            # Inference fallback: a group is suspicious if it contains more than 1 node
            group_sizes = torch.zeros(num_groups, dtype=torch.float, device=x.device)
            group_sizes.scatter_add_(0, group_mapping, torch.ones_like(group_mapping, dtype=torch.float))
            y_group = (group_sizes > 1).float().unsqueeze(1)
        
        # Step 4: Group-level encoding
        # Project X_hat from out_dim → node_in_dim, then re-use the same encoder weights
        # as in Step 1 (paper Eq. 12 — weight sharing via community-centric encoder).
        X_hat_proj = self.group_proj(X_hat)
        _, p_group = self.encoder(X_hat_proj, edge_index_hat, num_edges)
        
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
