import torch
import torch.nn as nn

class GAGNNLoss(nn.Module):
    def __init__(self, c1=1.0, c2=1.0, c3=1.0):
        """
        Args:
            c1: Weight for group loss
            c2: Weight for node loss
            c3: Weight for transaction loss
        """
        super(GAGNNLoss, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        # Using BCEWithLogitsLoss for all three tasks (numerically stable)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, p_node, y_node, p_trans, y_trans, p_group, y_group,
                node_mask=None, trans_mask=None, group_mask=None):
        """
        Args:
            p_node: Predicted node probabilities
            y_node: Ground truth node labels
            p_trans: Predicted transaction probabilities
            y_trans: Ground truth transaction labels
            p_group: Predicted group probabilities
            y_group: Ground truth group labels (1 if aggregated, 0 if single node)
            node_mask: Boolean mask for nodes (e.g. train_mask)
            trans_mask: Boolean mask for transactions
            group_mask: Boolean mask for groups
        """
        if node_mask is not None:
            l_node = self.bce(p_node[node_mask], y_node[node_mask])
        else:
            l_node = self.bce(p_node, y_node)
            
        if trans_mask is not None:
            l_trans = self.bce(p_trans[trans_mask], y_trans[trans_mask])
        else:
            l_trans = self.bce(p_trans, y_trans)
            
        if group_mask is not None:
            l_group = self.bce(p_group[group_mask], y_group[group_mask])
        else:
            l_group = self.bce(p_group, y_group)
        
        total_loss = self.c1 * l_group + self.c2 * l_node + self.c3 * l_trans
        return total_loss, l_node, l_trans, l_group
