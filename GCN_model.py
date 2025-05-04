import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for graph-level binary classification.
    Returns raw logits for use with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_channels: Dimensionality of each node feature vector.
            hidden_channels: Hidden dimension for all GCN layers.
            num_layers: Number of GCNConv layers (must be ≥1).
            dropout: Dropout probability after each GCN layer.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")

        # Build a stack of GCNConv layers
        self.convs = torch.nn.ModuleList()
        # first layer: in_channels → hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # subsequent layers: hidden_channels → hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final linear classifier from graph embedding → 1 logit
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node feature matrix of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges*2] (undirected).
            batch: Batch vector mapping each node to its graph ID [num_nodes].

        Returns:
            logits: Tensor of shape [num_graphs], raw scores (no sigmoid applied).
        """
        # 1) Apply each GCNConv + ReLU + Dropout
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2) Global mean pooling: [num_nodes, hidden] → [num_graphs, hidden]
        x = global_mean_pool(x, batch)

        # 3) Final linear layer → [num_graphs, 1] → reshape to [num_graphs]
        logits = self.lin(x).view(-1)
        return logits
