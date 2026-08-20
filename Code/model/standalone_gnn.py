import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, GPSConv

# Valid GNN backbone identifiers (mirrors encoder.py)
VALID_GNN_TYPES = ('gat', 'gin', 'gps_conv')


# =============================================================================
# StandaloneGNNLoss
# =============================================================================

class StandaloneGNNLoss(nn.Module):
    """
    Simple edge-level binary cross-entropy loss with optional class weighting.

    Args:
        laundry_weight: Positive-class weight for BCEWithLogitsLoss.
                        Values > 1 penalise false negatives more heavily.
        task:           'binary' or 'multiclass'
    """

    def __init__(self, laundry_weight: float | torch.Tensor = 1.0, task: str = 'binary'):
        super().__init__()
        self.laundry_weight = laundry_weight
        self.task = task

    def forward(self, p_trans: torch.Tensor, y_trans: torch.Tensor,
                trans_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            p_trans:    Raw logits for each edge  (E x 1) or (E, num_classes)
            y_trans:    Ground-truth edge labels   (E x 1) or (E,)
            trans_mask: Optional boolean mask to restrict loss to a subset
                        of edges (e.g. only train edges within a mini-batch).
        Returns:
            Scalar loss tensor.
        """
        if self.task == 'binary':
            pos_weight = torch.tensor([self.laundry_weight], device=p_trans.device)
            if trans_mask is not None and trans_mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    p_trans[trans_mask], y_trans[trans_mask], pos_weight=pos_weight
                )
            elif trans_mask is None:
                loss = F.binary_cross_entropy_with_logits(
                    p_trans, y_trans, pos_weight=pos_weight
                )
            else:
                loss = torch.tensor(0.0, device=p_trans.device, requires_grad=True)
        elif self.task == 'multiclass':
            if isinstance(self.laundry_weight, torch.Tensor):
                weight = self.laundry_weight.to(p_trans.device)
            else:
                weight = None
            if trans_mask is not None and trans_mask.any():
                loss = F.cross_entropy(
                    p_trans[trans_mask], y_trans[trans_mask].squeeze().long(), weight=weight
                )
            elif trans_mask is None:
                loss = F.cross_entropy(
                    p_trans, y_trans.squeeze().long(), weight=weight
                )
            else:
                loss = torch.tensor(0.0, device=p_trans.device, requires_grad=True)
        else:
            raise ValueError(f"Unknown task {self.task}")

        return loss


# =============================================================================
# StandaloneGNN
# =============================================================================

class StandaloneGNN(nn.Module):
    """
    Baseline ablation model: bare GNN backbone + EdgeMLP classifier.

    Architecture
    ------------
    1. GNN layer stack (N layers, no eMRF, no node-level MLP):
       - GAT: edge features are fed during message passing
         via the ``edge_dim`` constructor parameter.
       - GIN / GPSConv: no native edge-feature support in message passing;
         edge features are only used at the EdgeMLP stage.
    2. EdgeMLP: for each edge (i, j) concatenates
       [Z_i ‖ Z_j ‖ edge_attr_ij] and predicts a binary logit.

    The loss is computed externally by :class:`StandaloneGNNLoss`.

    Args:
        node_in_dim:        Input node-feature dimension.
        edge_feat_dim:      Edge-feature dimension.
        hidden_dim:         Hidden dimension inside GNN layers.
        out_dim:            Output node-embedding dimension (last GNN layer).
        heads:              Attention heads for GAT / GPSConv.
                            Silently ignored for GIN.
        num_layers:         Number of stacked GNN layers.
        gnn_type:           GNN backbone ('gat', 'gin', 'gps_conv').
        gin_mlp_hidden_dim: Hidden dim of the 2-layer MLP inside GINConv.
                            Also used by GPSConv for its local GIN component.
        mlp_hidden_dim:     Hidden dim of the EdgeMLP.
        minibatches:        Stored in checkpoint for reference; not used in forward.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 5,
        num_layers: int = 2,
        gnn_type: str = 'gat',
        gin_mlp_hidden_dim: int = 128,
        mlp_hidden_dim: int = 128,
        minibatches: bool = True,
        num_classes: int = 1,
    ):
        super().__init__()

        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(
                f"Unknown gnn_type '{gnn_type}'. Must be one of {VALID_GNN_TYPES}."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        # Save for checkpoint reconstruction
        self._init_kwargs = dict(
            node_in_dim=node_in_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads=heads,
            num_layers=num_layers,
            gnn_type=gnn_type,
            gin_mlp_hidden_dim=gin_mlp_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            minibatches=minibatches,
            num_classes=num_classes,
        )

        self.gnn_type = gnn_type
        self.num_layers = num_layers

        # Training-history buffers (same interface as GAGNN)
        self.training_losses: list[float] = []
        self.val_F1: list[float] = []

        # -----------------------------------------------------------------
        # 1. GNN layer stack (no eMRF)
        # -----------------------------------------------------------------
        self.gnn_layers = self._build_gnn_layers(
            node_in_dim, edge_feat_dim, hidden_dim, out_dim,
            heads, gnn_type, num_layers, gin_mlp_hidden_dim
        )

        # -----------------------------------------------------------------
        # 2. EdgeMLP: concat(Z_i, Z_j, edge_attr) → logit
        #    Input dim = 2 * out_dim + edge_feat_dim
        # -----------------------------------------------------------------
        edge_mlp_in = 2 * out_dim + edge_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Layer factory helpers
    # ------------------------------------------------------------------

    def _build_gnn_layers(
        self,
        node_in_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int,
        gnn_type: str,
        num_layers: int,
        gin_mlp_hidden_dim: int,
    ) -> nn.ModuleList:
        if gnn_type == 'gat':
            return self._build_gat_layers(
                node_in_dim, edge_feat_dim, hidden_dim, out_dim, heads, num_layers
            )
        elif gnn_type == 'gin':
            return self._build_gin_layers(
                node_in_dim, hidden_dim, out_dim, num_layers, gin_mlp_hidden_dim
            )
        elif gnn_type == 'gps_conv':
            return self._build_gps_layers(
                node_in_dim, hidden_dim, out_dim, heads, num_layers, gin_mlp_hidden_dim
            )

    # --- GAT (edge features via edge_dim) --------------------------------
    @staticmethod
    def _build_gat_layers(
        node_in_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int,
        num_layers: int,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = node_in_dim if i == 0 else hidden_dim * heads
            if i < num_layers - 1:
                layers.append(GATConv(layer_in, hidden_dim,
                                      heads=heads, concat=True,
                                      edge_dim=edge_feat_dim))
            else:
                layers.append(GATConv(layer_in, out_dim,
                                      heads=heads, concat=False,
                                      edge_dim=edge_feat_dim))
        return layers

    # --- GIN (no native edge-feature support) ----------------------------
    @staticmethod
    def _build_gin_layers(
        node_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        gin_mlp_hidden_dim: int,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = node_in_dim if i == 0 else hidden_dim
            layer_out = out_dim if i == num_layers - 1 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(layer_in, gin_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden_dim, layer_out),
            )
            layers.append(GINConv(mlp))
        return layers

    # --- GPSConv (no native edge-feature support in local GINConv) -------
    def _build_gps_layers(
        self,
        node_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int,
        num_layers: int,
        gin_mlp_hidden_dim: int,
    ) -> nn.ModuleList:
        if hidden_dim % heads != 0:
            raise ValueError(
                f"GPSConv requires hidden_dim ({hidden_dim}) to be "
                f"divisible by heads ({heads})."
            )
        # Input/output projections stored on self (participate in state_dict)
        self.gps_input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.gps_output_proj = nn.Linear(hidden_dim, out_dim)

        layers = nn.ModuleList()
        for _ in range(num_layers):
            local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, gin_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden_dim, hidden_dim),
            )
            local_conv = GINConv(local_mlp)
            layers.append(
                GPSConv(hidden_dim, conv=local_conv,
                        heads=heads, attn_type='multihead')
            )
        return layers

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features  (N x node_in_dim)
            edge_index: Adjacency list (2 x E)
            edge_attr:  Edge features  (E x edge_feat_dim)
        Returns:
            p_trans: Raw logits for each edge (E x 1)
        """
        # ---- GNN forward ----
        if self.gnn_type == 'gps_conv':
            z = self.gps_input_proj(x)
            for layer in self.gnn_layers:
                z = layer(z, edge_index)
            z = self.gps_output_proj(z)

        elif self.gnn_type == 'gat':
            # These layers consume edge_attr during message passing
            z = x
            for i, layer in enumerate(self.gnn_layers):
                z = layer(z, edge_index, edge_attr=edge_attr)
                if i < len(self.gnn_layers) - 1:
                    z = torch.relu(z)

        else:  # gin
            z = x
            for i, layer in enumerate(self.gnn_layers):
                z = layer(z, edge_index)
                if i < len(self.gnn_layers) - 1:
                    z = torch.relu(z)

        # ---- EdgeMLP ----
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([z[src], z[dst], edge_attr], dim=-1)  # (E x edge_mlp_in)
        p_trans = self.edge_mlp(edge_input)  # (E x 1)

        return p_trans

    # ------------------------------------------------------------------
    # Persistence helpers (same interface as GAGNN)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves model weights, training history, and architecture config."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'training_losses': self.training_losses,
            'val_F1': self.val_F1,
            'init_kwargs': self._init_kwargs,
        }, path)

    @classmethod
    def load_saved(cls, path: str, **fallback_kwargs) -> 'StandaloneGNN':
        """
        Instantiates and loads a StandaloneGNN directly from a checkpoint file.

        Args:
            path:             Path to a .pt file saved by :meth:`save`.
            **fallback_kwargs: Architecture kwargs used when the checkpoint
                              does not contain 'init_kwargs'.
        Returns:
            A StandaloneGNN in eval() mode with weights and history restored.
        """
        checkpoint = torch.load(path, map_location='cpu')
        init_kwargs = checkpoint.get('init_kwargs', fallback_kwargs)
        if not init_kwargs:
            raise KeyError(
                f"Checkpoint '{path}' does not contain architecture info "
                "('init_kwargs'). Pass hyperparameters as keyword arguments."
            )
        model = cls(**init_kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.training_losses = checkpoint.get('training_losses', [])
        model.val_F1 = checkpoint.get('val_F1', [])
        model.eval()
        return model

    def plot_training_history(self) -> None:
        """Plots training loss and validation F1 over time."""
        import matplotlib.pyplot as plt

        if not self.training_losses and not self.val_F1:
            print("No training history to plot.")
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        if self.training_losses:
            ax1.plot(self.training_losses, label='Training Loss',
                     color='blue', linewidth=2)
            ax1.set_ylabel('Loss (log scale)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_yscale('log')

        ax1.set_xlabel('Epochs')

        if self.val_F1:
            ax2 = ax1.twinx()
            ax2.plot(self.val_F1, label='Validation F1-Score',
                     color='orange', linewidth=2)
            ax2.set_ylabel('F1-Score', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

        plt.title('Baseline — Training Loss and Validation F1-Score Over Time')
        fig.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
