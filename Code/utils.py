import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)


class Evaluator:
    """Utility class for model evaluation and reporting."""

    @staticmethod
    def evaluation_report(model, test_loader, criterion, device, threshold=0.5, edge_mask_name=None, task='binary'):
        """
        Loads the best saved model, evaluates it on the test set, and reports:
          - Accuracy, Precision, Recall, F1-score, ROC-AUC
          - Confusion matrix plot for TRANSACTIONS (edges).

        Uses `batch.e_id` to prevent double-counting edges sampled in multiple subgraphs.

        Args:
            model:       The GAGNN model instance.
            test_loader: PyG NeighborLoader for the test split.
            criterion:   GAGNNLoss instance.
            device:      torch.device to run inference on.
            threshold:   Decision threshold applied to p_trans predictions (default 0.5, binary only).
            edge_mask_name: String name of the boolean edge mask attribute to filter edges (e.g. 'edge_test_mask').
            task:        'binary' or 'multiclass'
        """
        model.eval()

        all_probs  = []
        all_preds  = []
        all_labels = []
        
        # Track which edges have been evaluated to avoid double counting from overlapping subgraphs
        num_total_edges = test_loader.data.num_edges if hasattr(test_loader, 'data') else test_loader[0].num_edges
        evaluated_edges = torch.zeros(num_total_edges, dtype=torch.bool)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Check which edges in this batch have already been evaluated
                if hasattr(batch, 'e_id'):
                    e_id = batch.e_id.cpu()
                    # Filter for edges that have NOT been evaluated yet
                    mask = ~evaluated_edges[e_id]
                    evaluated_edges[e_id] = True
                else:
                    # Fallback for full-batch testing where e_id might be missing
                    mask = torch.ones(batch.num_edges, dtype=torch.bool)
                
                if edge_mask_name is not None and hasattr(batch, edge_mask_name):
                    split_mask = getattr(batch, edge_mask_name).cpu()
                    mask = mask & split_mask
                
                if not mask.any():
                    continue

                p_node, p_trans, p_group, y_group = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.num_edges
                )

                # Filter p_trans and y_trans using the mask
                p_trans_new = p_trans[mask]
                y_trans_new = batch.y_trans[mask]

                if task == 'binary':
                    probs = torch.sigmoid(p_trans_new)
                    probs_flat = probs.view(-1).cpu().numpy()
                    preds = (probs_flat >= threshold).astype(int)
                else:
                    probs = torch.softmax(p_trans_new, dim=1)
                    probs_flat = probs.cpu().numpy()
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    
                labels = y_trans_new.view(-1).long().cpu().numpy()

                all_probs.append(probs_flat)
                all_preds.append(preds)
                all_labels.append(labels)

        if len(all_preds) == 0:
            print("No edges evaluated!")
            return

        all_probs  = np.concatenate(all_probs)
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        if task == 'binary':
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec  = recall_score(all_labels, all_preds, zero_division=0)
            f1   = f1_score(all_labels, all_preds, zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.0  # Handle case with only one class present
        else:
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except ValueError:
                auc = 0.0

        print("=" * 50)
        print("    EVALUATION REPORT (Transactions / Edges)")
        print("=" * 50)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print("-" * 50)
        
        if task == 'binary':
            target_names = ["Legit", "Laundering"]
        else:
            pattern_mapping = {
                0: 'LEGIT', 1: 'FAN-OUT', 2: 'FAN-IN', 3: 'CYCLE',
                4: 'GATHER-SCATTER', 5: 'SCATTER-GATHER', 6: 'BIPARTITE',
                7: 'STACK', 8: 'RANDOM'
            }
            # Only include classes that are present in the dataset (or all if we want fixed report)
            unique_labels = sorted(list(set(all_labels) | set(all_preds)))
            target_names = [pattern_mapping.get(i, f"Class {i}") for i in unique_labels]

        print(classification_report(
            all_labels, all_preds,
            target_names=target_names,
            zero_division=0,
            labels=unique_labels if task == 'multiclass' else None
        ))

        # Confusion matrix plot
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8 if task == 'multiclass' else 6, 6 if task == 'multiclass' else 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation='vertical' if task == 'multiclass' else 'horizontal')
        ax.set_title("Confusion Matrix — Transactions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()
    @staticmethod
    def baseline_evaluation_report(model, test_loader, device, threshold=0.5, edge_mask_name=None, task='binary'):
        """
        Evaluates a :class:`StandaloneGNN` on the test set and reports:
          - Accuracy, Precision, Recall, F1-score, ROC-AUC
          - Confusion matrix plot for TRANSACTIONS (edges).

        Uses ``batch.e_id`` to prevent double-counting edges sampled in
        multiple overlapping subgraphs (same logic as ``evaluation_report``).

        Args:
            model:          A StandaloneGNN instance in eval() mode.
            test_loader:    PyG NeighborLoader (or list containing the full Data).
            device:         torch.device to run inference on.
            threshold:      Decision threshold for p_trans (default 0.5, binary only).
            edge_mask_name: Attribute name of the boolean edge mask to restrict
                            evaluation to the test split (e.g. 'edge_test_mask').
            task:           'binary' or 'multiclass'
        """
        model.eval()

        all_probs  = []
        all_preds  = []
        all_labels = []

        # Track which edges have been evaluated to avoid double counting
        num_total_edges = (
            test_loader.data.num_edges
            if hasattr(test_loader, 'data')
            else test_loader[0].num_edges
        )
        evaluated_edges = torch.zeros(num_total_edges, dtype=torch.bool)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                # Build dedup mask
                if hasattr(batch, 'e_id'):
                    e_id = batch.e_id.cpu()
                    mask = ~evaluated_edges[e_id]
                    evaluated_edges[e_id] = True
                else:
                    mask = torch.ones(batch.num_edges, dtype=torch.bool)

                if edge_mask_name is not None and hasattr(batch, edge_mask_name):
                    split_mask = getattr(batch, edge_mask_name).cpu()
                    mask = mask & split_mask

                if not mask.any():
                    continue

                # StandaloneGNN returns only p_trans
                p_trans = model(batch.x, batch.edge_index, batch.edge_attr)

                p_trans_masked = p_trans[mask]
                y_trans_masked = batch.y_trans[mask]

                if task == 'binary':
                    probs = torch.sigmoid(p_trans_masked)
                    probs_flat = probs.view(-1).cpu().numpy()
                    preds  = (probs_flat >= threshold).astype(int)
                else:
                    probs = torch.softmax(p_trans_masked, dim=1)
                    probs_flat = probs.cpu().numpy()
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    
                labels = y_trans_masked.view(-1).long().cpu().numpy()

                all_probs.append(probs_flat)
                all_preds.append(preds)
                all_labels.append(labels)

        if len(all_preds) == 0:
            print("No edges evaluated!")
            return

        all_probs  = np.concatenate(all_probs)
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        if task == 'binary':
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec  = recall_score(all_labels, all_preds, zero_division=0)
            f1   = f1_score(all_labels, all_preds, zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.0
        else:
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except ValueError:
                auc = 0.0

        print("=" * 50)
        print("  BASELINE EVALUATION REPORT (Transactions / Edges)")
        print("=" * 50)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print("-" * 50)
        
        if task == 'binary':
            target_names = ["Legit", "Laundering"]
        else:
            pattern_mapping = {
                0: 'LEGIT', 1: 'FAN-OUT', 2: 'FAN-IN', 3: 'CYCLE',
                4: 'GATHER-SCATTER', 5: 'SCATTER-GATHER', 6: 'BIPARTITE',
                7: 'STACK', 8: 'RANDOM'
            }
            # Only include classes that are present in the dataset (or all if we want fixed report)
            unique_labels = sorted(list(set(all_labels) | set(all_preds)))
            target_names = [pattern_mapping.get(i, f"Class {i}") for i in unique_labels]
            
        print(classification_report(
            all_labels, all_preds,
            target_names=target_names,
            zero_division=0,
            labels=unique_labels if task == 'multiclass' else None
        ))

        # Confusion matrix plot
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8 if task == 'multiclass' else 6, 6 if task == 'multiclass' else 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, colorbar=True, cmap="Oranges", xticks_rotation='vertical' if task == 'multiclass' else 'horizontal')
        ax.set_title("Baseline — Confusion Matrix (Transactions)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiclass_distribution(y_trans, train_edge_mask, val_edge_mask, test_edge_mask):
        """
        Plots the distribution of money laundering types in the dataset,
        training set, validation set, and test set. Excludes the LEGIT class (0).

        Args:
            y_trans (torch.Tensor or np.ndarray): Ground truth labels for all transactions.
            train_edge_mask (torch.Tensor or np.ndarray): Boolean mask for the training edges.
            val_edge_mask (torch.Tensor or np.ndarray): Boolean mask for the validation edges.
            test_edge_mask (torch.Tensor or np.ndarray): Boolean mask for the test edges.
        """
        import pandas as pd
        import seaborn as sns
        
        if isinstance(y_trans, torch.Tensor):
            y_trans = y_trans.cpu().numpy()
        
        y_trans = y_trans.flatten()
        
        pattern_mapping = {
            1: 'FAN-OUT', 2: 'FAN-IN', 3: 'CYCLE',
            4: 'GATHER-SCATTER', 5: 'SCATTER-GATHER', 6: 'BIPARTITE',
            7: 'STACK', 8: 'RANDOM'
        }

        # Filter out legit transactions (class 0)
        def get_counts(labels, mask=None):
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                labels = labels[mask]
            
            # Keep only classes > 0
            labels = labels[labels > 0]
            counts = pd.Series(labels).value_counts().sort_index()
            return {pattern_mapping.get(k, f"Class {k}"): v for k, v in counts.items()}

        overall_counts = get_counts(y_trans)
        train_counts = get_counts(y_trans, train_edge_mask)
        val_counts = get_counts(y_trans, val_edge_mask)
        test_counts = get_counts(y_trans, test_edge_mask)

        # Create a combined DataFrame for plotting
        df_list = []
        for class_name in pattern_mapping.values():
            df_list.append({'Class': class_name, 'Count': overall_counts.get(class_name, 0), 'Split': 'Overall'})
            df_list.append({'Class': class_name, 'Count': train_counts.get(class_name, 0), 'Split': 'Train'})
            df_list.append({'Class': class_name, 'Count': val_counts.get(class_name, 0), 'Split': 'Validation'})
            df_list.append({'Class': class_name, 'Count': test_counts.get(class_name, 0), 'Split': 'Test'})
            
        df = pd.DataFrame(df_list)

        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x='Class', y='Count', hue='Split')
        plt.title('Distribution of Money Laundering Types (Excluding LEGIT)', fontsize=14, fontweight='bold')
        plt.xlabel('Money Laundering Pattern', fontsize=12)
        plt.ylabel('Number of Transactions', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Dataset Split')
        plt.tight_layout()
        plt.show()


def get_random_splits(y, num_nodes, val_size=0.1, test_size=0.2, seed=42):
    """
    Creates stratified train/val/test splits using a fast random shuffle.

    Strategy (aligned with the GAGNN paper):
      - All nodes are split into positives and negatives.
      - Each group is independently shuffled and proportionally allocated to
        val / test / train.
      - Val and test keep the natural class distribution for fair evaluation.
      - The training set down-samples negatives to 1:1 ratio with positives
        to address class imbalance (as described in the paper).

    Args:
        y        : 1-D torch.Tensor of binary node labels (0 = legit, 1 = fraud).
        num_nodes: Total number of nodes in the graph.
        val_size : Fraction of nodes to use for validation (default 0.1).
        test_size: Fraction of nodes to use for test (default 0.2).
        seed     : Random seed for reproducibility (default 42).

    Returns:
        train_idx, val_idx, test_idx : numpy int64 arrays of node indices.
    """
    rng = np.random.default_rng(seed)

    y_np = y.cpu().numpy().flatten()

    pos_idx = np.where(y_np == 1)[0]
    neg_idx = np.where(y_np == 0)[0]

    # Shuffle both groups independently (stratified)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    total_pos = len(pos_idx)
    total_neg = len(neg_idx)

    # --- Validation quotas (natural distribution) ---
    p_val = int(val_size * total_pos)
    n_val = int(val_size * total_neg)

    # --- Test quotas (natural distribution) ---
    p_test = int(test_size * total_pos)
    n_test = int(test_size * total_neg)

    # --- Training quotas (remaining positives; down-sample negatives 1:1) ---
    p_train = total_pos - p_val - p_test
    n_train = p_train  # 1:1 down-sampling to address class imbalance

    # --- Slice out the splits ---
    val_idx  = np.concatenate([
        pos_idx[:p_val],
        neg_idx[:n_val]
    ])
    test_idx = np.concatenate([
        pos_idx[p_val : p_val + p_test],
        neg_idx[n_val : n_val + n_test]
    ])
    train_idx = np.concatenate([
        pos_idx[p_val + p_test : p_val + p_test + p_train],
        neg_idx[n_val + n_test : n_val + n_test + n_train]
    ])

    print(f"Extracting Validation Set: {p_val} pos, {n_val} neg")
    print(f"Extracting Test Set:       {p_test} pos, {n_test} neg")
    print(f"Extracting Train Set (downsampled): {p_train} pos, {n_train} neg")

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)
