import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GINConv, TransformerConv, GPSConv
from model.emrf import eMRFLayer

# Valid GNN backbone identifiers
VALID_GNN_TYPES = ('gat', 'gin', 'transformer_conv', 'gps_conv')


class CommunityCentricEncoder(nn.Module):
    """
    Community-Centric Encoder that stacks N GNN layers followed by an eMRF
    layer and a node classification MLP.

    Supports four GNN backbones:
        - 'gat'              : Graph Attention Network (GATConv)
        - 'gin'              : Graph Isomorphism Network (GINConv)
        - 'transformer_conv' : Lightweight Graph Transformer (TransformerConv)
        - 'gps_conv'         : General, Powerful, Scalable Graph Transformer (GPSConv)

    Args:
        in_channels:        Dimension of input node features X^(1).
        hidden_channels:    Hidden dimension inside GNN layers.
        out_channels:       Output dimension of node embeddings X^(3).
        heads:              Number of attention heads (GAT / TransformerConv / GPSConv).
                            Silently ignored for GIN.
        beta:               eMRF trade-off between topological and attribute similarity.
        nn_t_hidden_dim:    Hidden dimension of the node classification MLP (NN_t).
        gnn_type:           GNN backbone identifier (see VALID_GNN_TYPES).
        num_layers:         Number of stacked GNN layers.
        gin_mlp_hidden_dim: Hidden dimension of the 2-layer MLP inside each GIN layer.
                            Also used by GPSConv for its local GIN component.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=5, beta=0.44, nn_t_hidden_dim=128,
                 gnn_type='gat', num_layers=2, gin_mlp_hidden_dim=128):
        super(CommunityCentricEncoder, self).__init__()

        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(
                f"Unknown gnn_type '{gnn_type}'. Must be one of {VALID_GNN_TYPES}."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.gnn_type = gnn_type
        self.num_layers = num_layers

        # 1. GNN layer stack
        self.gnn_layers = self._build_gnn_layers(
            in_channels, hidden_channels, out_channels,
            heads, gnn_type, num_layers, gin_mlp_hidden_dim
        )

        # 2. eMRF Layer
        self.emrf = eMRFLayer(out_channels, beta=beta)

        # 3. Node classification MLP
        # Generates X(4) = sigmoid(NN_t(X(3), W_t))
        # The paper says NN_t is set at 128 to extend the original node feature
        self.nn_t = nn.Sequential(
            nn.Linear(out_channels, nn_t_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(nn_t_hidden_dim, 1)  # Binary classification output (logits)
        )

        # Coarse classifier to generate label probabilities for eMRF
        self.coarse_classifier = nn.Linear(out_channels, 1)

    # ------------------------------------------------------------------
    # Factory: build the GNN layer stack
    # ------------------------------------------------------------------
    def _build_gnn_layers(self, in_channels, hidden_channels, out_channels,
                          heads, gnn_type, num_layers, gin_mlp_hidden_dim):
        """
        Build an nn.ModuleList of GNN convolution layers.

        For attention-based models (GAT, TransformerConv) intermediate layers
        use ``concat=True`` so their output dimension is ``hidden_channels * heads``,
        while the final layer uses ``concat=False`` to produce ``out_channels``.

        For GIN the MLP hidden dimension is controlled by ``gin_mlp_hidden_dim``.

        GPSConv operates at a fixed channel width, so it uses input/output
        linear projections stored as ``self.gps_input_proj`` and
        ``self.gps_output_proj``.
        """
        layers = nn.ModuleList()

        if gnn_type == 'gat':
            layers = self._build_attention_layers(
                GATConv, in_channels, hidden_channels, out_channels,
                heads, num_layers
            )

        elif gnn_type == 'transformer_conv':
            layers = self._build_attention_layers(
                TransformerConv, in_channels, hidden_channels, out_channels,
                heads, num_layers
            )

        elif gnn_type == 'gin':
            layers = self._build_gin_layers(
                in_channels, hidden_channels, out_channels,
                num_layers, gin_mlp_hidden_dim
            )

        elif gnn_type == 'gps_conv':
            layers = self._build_gps_layers(
                in_channels, hidden_channels, out_channels,
                heads, num_layers, gin_mlp_hidden_dim
            )

        return layers

    # --- Attention-based layers (GAT / TransformerConv) ----------------
    @staticmethod
    def _build_attention_layers(conv_cls, in_channels, hidden_channels,
                                out_channels, heads, num_layers):
        """Build layers for attention-based GNNs that support ``heads`` and ``concat``."""
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer_in = in_channels
            else:
                layer_in = hidden_channels * heads  # previous concat=True output

            if i < num_layers - 1:
                # Intermediate layer: concat heads
                layers.append(conv_cls(layer_in, hidden_channels,
                                       heads=heads, concat=True))
            else:
                # Final layer: average heads
                layers.append(conv_cls(layer_in, out_channels,
                                       heads=heads, concat=False))
        return layers

    # --- GIN layers ----------------------------------------------------
    @staticmethod
    def _build_gin_layers(in_channels, hidden_channels, out_channels,
                          num_layers, gin_mlp_hidden_dim):
        """Build GINConv layers, each wrapping a 2-layer MLP."""
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else hidden_channels
            layer_out = out_channels if i == num_layers - 1 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(layer_in, gin_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden_dim, layer_out),
            )
            layers.append(GINConv(mlp))
        return layers

    # --- GPSConv layers ------------------------------------------------
    def _build_gps_layers(self, in_channels, hidden_channels, out_channels,
                          heads, num_layers, gin_mlp_hidden_dim):
        """
        Build GPSConv layers.

        GPSConv requires a fixed channel width across all layers, so we add
        linear projections before and after the GPS stack.

        Note:
            ``hidden_channels`` must be divisible by ``heads`` because the
            internal multi-head attention splits the embedding across heads.
        """
        if hidden_channels % heads != 0:
            raise ValueError(
                f"GPSConv requires hidden_channels ({hidden_channels}) to be "
                f"divisible by heads ({heads})."
            )
        # Projection layers (stored on self so they participate in state_dict)
        self.gps_input_proj = nn.Linear(in_channels, hidden_channels)
        self.gps_output_proj = nn.Linear(hidden_channels, out_channels)

        layers = nn.ModuleList()
        for _ in range(num_layers):
            # Local message-passing module: a GINConv with a 2-layer MLP
            local_mlp = nn.Sequential(
                nn.Linear(hidden_channels, gin_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden_dim, hidden_channels),
            )
            local_conv = GINConv(local_mlp)
            layers.append(
                GPSConv(hidden_channels, conv=local_conv,
                        heads=heads, attn_type='multihead')
            )
        return layers

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        # --- GNN layers ---
        if self.gnn_type == 'gps_conv':
            x2 = self.gps_input_proj(x)
            for layer in self.gnn_layers:
                x2 = layer(x2, edge_index)
            x2 = self.gps_output_proj(x2)
        else:
            x2 = x
            for i, layer in enumerate(self.gnn_layers):
                x2 = layer(x2, edge_index)
                if i < len(self.gnn_layers) - 1:
                    x2 = torch.relu(x2)
        # x2 is now X^(2)

        # eMRF Layer requires a coarse prediction of the label
        p_coarse = torch.sigmoid(self.coarse_classifier(x2))
        x3 = self.emrf(x2, edge_index, p_coarse, num_edges=num_edges)  # X^(3)

        # Classification (return logits directly for numerical stability in BCEWithLogitsLoss)
        x4 = self.nn_t(x3)  # X^(4)

        return x3, x4
