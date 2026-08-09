import torch
import torch.nn as nn
import torch.nn.functional as F

class GAGNNLoss(nn.Module):
    def __init__(self, c1=1.0, c2=1.0, c3=1.0, laundry_weight=1.0):
        """
        Args:
            c1: Weight for group loss
            c2: Weight for node loss
            c3: Weight for transaction loss
            laundry_weight: Weight for the positive class (Is Laundering = 1)
        """
        super(GAGNNLoss, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.laundry_weight = laundry_weight

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
        pos_weight = torch.tensor([self.laundry_weight], device=p_node.device)
        
        l_node = torch.tensor(0.0, device=p_node.device)
        if node_mask is not None:
            if node_mask.any():
                l_node = F.binary_cross_entropy_with_logits(p_node[node_mask], y_node[node_mask], pos_weight=pos_weight)
        else:
            l_node = F.binary_cross_entropy_with_logits(p_node, y_node, pos_weight=pos_weight)
            
        l_trans = torch.tensor(0.0, device=p_trans.device)
        if trans_mask is not None:
            if trans_mask.any():
                l_trans = F.binary_cross_entropy_with_logits(p_trans[trans_mask], y_trans[trans_mask], pos_weight=pos_weight)
        else:
            l_trans = F.binary_cross_entropy_with_logits(p_trans, y_trans, pos_weight=pos_weight)
            
        l_group = torch.tensor(0.0, device=p_group.device)
        if group_mask is not None:
            if group_mask.any():
                l_group = F.binary_cross_entropy_with_logits(p_group[group_mask], y_group[group_mask], pos_weight=pos_weight)
        else:
            l_group = F.binary_cross_entropy_with_logits(p_group, y_group, pos_weight=pos_weight)
        
        total_loss = self.c1 * l_group + self.c2 * l_node + self.c3 * l_trans
        return total_loss, l_node, l_trans, l_group
