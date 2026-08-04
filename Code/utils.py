import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)


class Evaluator:
    """Utility class for model evaluation and reporting."""

    @staticmethod
    def evaluation_report(model, test_loader, criterion, device, threshold=0.5):
        """
        Loads the best saved model, evaluates it on the test set, and reports:
          - Accuracy, Precision, Recall, F1-score
          - Confusion matrix plot

        Only seed nodes (batch.batch_size) contribute to the metrics.

        Args:
            model:       The GAGNN model instance (will be reloaded from best checkpoint).
            test_loader: NeighborLoader for the test split.
            criterion:   GAGNNLoss instance (unused for metric computation, kept for API consistency).
            device:      torch.device to run inference on.
            threshold:   Decision threshold applied to p_node predictions (default 0.5).
        """
        model.eval()

        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                p_node, p_trans, p_group, y_group = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.y_node, batch.num_edges
                )

                # Restrict to seed nodes only
                n_seed = batch.batch_size
                p_node_seed = p_node[:n_seed]        # (n_seed, 1)
                y_node_seed = batch.y_node[:n_seed]  # (n_seed,)

                preds  = (p_node_seed.squeeze() >= threshold).long().cpu().numpy()
                labels = y_node_seed.long().cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 2. Compute metrics
        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)

        print("=" * 50)
        print("         EVALUATION REPORT (Test Set)")
        print("=" * 50)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print("-" * 50)
        print(classification_report(
            all_labels, all_preds,
            target_names=["Legit", "Laundering"],
            zero_division=0
        ))

        # 3. Confusion matrix plot
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Laundering"])
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()
