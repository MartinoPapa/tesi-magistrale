# Differences between GAGNN Paper and Code Implementation

This document compares the model described in the paper **"Anti-Money Laundering by Group-Aware Deep Graph Learning"** (Cheng et al., IEEE TKDE 2023) with the actual Python implementation in this project (`main.ipynb`, `model/gagnn.py`, `model/encoder.py`, `model/emrf.py`, `model/group_layer.py`, `model/loss.py`, `utils.py`).

---

## 1. Dataset

### 1.1 Dataset Source
- **Paper**: Uses a proprietary real-world dataset from **UnionPay** (one of the largest bank card alliances worldwide), spanning three weeks (06/09/2021–26/09/2021), with ~1.8 million transactions per week and ~5% labeled as suspicious ML (money laundering). The labels are annotated by financial risk experts with the help of automation tools.
- **Implementation**: Uses the publicly available **IBM AMLSim (HI-Small)** synthetic dataset (`HI-Small_Trans.csv`), which has 5,078,345 transactions with only ~0.10% labeled as money laundering.

### 1.2 Data Split Strategy ✅ *Resolved*

> **Change made**: Implemented a **strict chronological split** directly based on the transaction `Timestamp` to perfectly simulate the paper's online deployment scenario. We selected a `split_date`, and split the graph chronologically: the preceding 7 days are the training block, the last 2 days of that block act as validation, and the 7 days after the split date act as the test set.

- **Paper**: Splits data **chronologically** into three weeks. The model is trained on 70% of data from each week and tested on the remaining 30%, simulating a real-world online scenario where the model is re-trained nightly and deployed for next-day predictions. There is **no explicit validation set** mentioned — only train/test. Moreover, the task is explicitly stated as **inductive** (Section III.A).
- **Implementation**: The previous random train/test split (and its associated K-Fold Cross Validation loop) was removed. We now employ a **strict chronological split** (e.g., `2022-09-08`). The model isolates a 5-day training subgraph, a 2-day validation subgraph for early stopping, and tests on future data beyond the `split_date`. The `Timestamp` is completely stripped from the features fed to the model to prevent data leakage.

---

## 2. Class Imbalance Handling (Down-Sampling)

- **Paper**: Down-samples at the **user/node level**: selects *all users* who have been involved in ML transactions and all their historical transactions, then samples an equivalent number of legitimate (non-ML) users and extracts all their records. The resulting subgraph is used for training.
- **Implementation**: Down-samples at the **node index level** within the training mask only: after selecting the train split, it keeps all positive nodes and randomly draws an equal number of negative nodes (`n_train = p_train` in `get_random_splits`). The validation and test sets retain the **natural (imbalanced) class distribution**. Edge-level down-sampling is **not** performed; all edges are included.

---

## 3. Node Feature Construction ✅ *Resolved*

> **Change made**: Node labels now use the raw mean from `nodes_features_df['Is Laundering']` (soft probability), matching Eq. 2.

- **Paper**: Node features $v_i$ are constructed by **averaging the feature vectors of all edges connected to node $v_i$** (Eq. 1). Node labels $\hat{y}_i$ are constructed as the average of the ground-truth labels of connected transactions (Eq. 2), forming a soft probability.
- **Implementation** (current): Node features are constructed by aggregating multiple edge statistics (mean, std, sum, count, etc.) per node in `data_preparation.py`, which is a richer but different aggregation. Node labels $y_\text{node}$ now use the **soft mean** from `nodes_features_df['Is Laundering']` (i.e., the fraction of ML edges incident on the node), exactly as in Eq. 2. The richer multi-stat feature aggregation for node features remains as-is.

---

## 4. GAT Attention Heads ✅ *Resolved*

> **Change made**: `heads` changed from `2` to `5` in `main.ipynb`.

- **Paper**: The number of attention heads $k$ is set to **5** (stated in the baseline comparison for GAT and implicitly for GAGNN).
- **Implementation** (current): `heads = 5`, matching the paper.

---

## 5. GAT Layer Concatenation Mode

- **Paper**: The architecture in Fig. 3 and Eq. (2)–(3) implies standard GAT behavior. The paper does not specify whether heads are concatenated or averaged.
- **Implementation**: Both `gat1` and `gat2` use `concat=False` (outputs are **averaged** across heads). This is an explicit design choice to keep the hidden dimension fixed at `hidden_channels` between layers.

---

## 6. Group Encoding — Shared Encoder ✅ *Resolved*

> **Change made**: Removed `self.group_encoder`; added `self.group_proj = nn.Linear(out_dim, node_in_dim)` in `model/gagnn.py`. The forward pass now projects `X_hat` into `node_in_dim` space and feeds it through `self.encoder` (weight sharing, per Eq. 12).

- **Paper**: Equation (12) feeds the group graph $\hat{G}$ and features $\hat{X}$ back into the **same** community-centric encoder, implying weight sharing.
- **Implementation** (current): Uses the same `self.encoder` for group-level encoding. A learnable projection `self.group_proj` bridges the dimension gap between group features (`out_dim`) and the encoder's expected input (`node_in_dim`).

---

## 7. y_group Ground-Truth Label Definition ✅ *Resolved*

> **Change made**: `GAGNN.forward()` now accepts an optional `y_node` argument. During training, `y_group` is computed via `scatter_reduce max` over `group_mapping` — a group gets label 1 if **any** of its constituent nodes has a ground-truth ML label. Inference falls back to the group-size heuristic. `train_step` in `main.ipynb` now passes `data_batch.y_node.view(-1)`.

- **Paper**: The group label $\hat{y}_i = 1$ if the group $\hat{v}_i$ is an aggregated ML group (contains actual ML nodes), and $\hat{y}_i = 0$ otherwise.
- **Implementation** (current): During **training**, `y_group` is determined by propagating ground-truth node labels through `group_mapping` via a scatter-max: a group is labeled 1 if any constituent node has a true ML label. During **inference**, falls back to `group_size > 1`.

---

## 8. eMRF Pairwise Potential ($\Psi$) — Differentiability ✅ *Resolved*

> **Change made**: Implemented a **Straight-Through Estimator (STE)** for $\sigma$. The forward pass now uses a hard binary indicator (as defined in Eq. 5), while the backward pass uses the soft continuous probability gradient to allow the `coarse_classifier` to learn.

- **Paper**: The pairwise potential (Eq. 5) uses a discrete indicator $\sigma(v_i, v_j) = 1$ when $v_i$ and $v_j$ are labeled in the **same category**, making $-1^{\sigma(v_i,v_j)}$ a hard sign flip. However, Equation 7 defines the propagation matrix as $\Gamma = \gamma(v_i, v_j)$, completely dropping $\Psi$ and $\sigma$. This is likely a typo in the paper, as dropping $\sigma$ means the classification labels are not used in the eMRF at all. Furthermore, using a hard binary indicator for $\sigma$ natively in PyTorch is non-differentiable.
- **Implementation** (`emrf.py`): Assumes the paper intended $\Gamma = \Psi$ in Eq. 7. To stay loyal to the "hard binary indicator" in Eq. 5 without breaking differentiability, we use a **Straight-Through Estimator (STE)**. The probability that two nodes share the same class is computed continuously as `sigma_soft = p_i * p_j + (1 - p_i) * (1 - p_j)`, but the forward pass evaluates using `sigma_hard = (pred_i == pred_j)`. The STE (`sigma = sigma_hard.detach() - sigma_soft.detach() + sigma_soft`) ensures gradients flow back to the coarse classifier while strictly matching the paper's hard indicator during inference.

---

## 9. Node Aggregation Policy (Eq. 11) — Text vs. Equation Discrepancy

- **Paper**: The body text mentions using "element-wise summation", but Equation (11) defines the group feature as the **mean**: $\hat{v}_i = \frac{1}{|M_i|} \sum_{j \in M_i} v_j$.
- **Implementation** (`group_layer.py`): Correctly follows **Equation (11)** using `scatter_reduce_(..., reduce='mean')`, computing the mean of embeddings in each group. The contradictory text mention of "summation" in the paper is ignored.

---

## 10. Mini-Batch Graph Sampling & Batching ✅ *Resolved*

> **Change made**: Replaced `NeighborLoader` with PyG's `ClusterLoader` for the training phase to build batches of highly connected subgraphs, maximizing the community structure needed by the eMRF layer.

- **Paper**: States that neighbors are sampled according to a **Bernoulli distribution** with the same parameter to limit computation for high-degree nodes. This is mentioned as the strategy for making the algorithm scalable to 60M samples.
- **Implementation** (current): During training, uses PyG's `ClusterLoader`, which relies on the **METIS graph partitioning algorithm** to split the training graph into dense, highly-connected subgraphs (clusters). These clusters are batched together to form the mini-batch. This perfectly complements the GAGNN architecture by maximizing the internal edges (transactions) and distinct communities inside each batch, doing exactly what a constrained BFS would do but efficiently in C++. Validation and testing still use `NeighborLoader` for exhaustive evaluation.

---

## 11. Evaluation Metrics

- **Paper**: Reports **AUC** (area under the ROC curve) and **R@P_N** (recall at precision level N, for N = 0.6, 0.7, 0.8, 0.9) on the transaction-level (edge) prediction task.
- **Implementation**: Reports **Accuracy, Precision, Recall, F1-score**, and a **Confusion Matrix** on the **node-level** prediction task (via `Evaluator.evaluation_report` in `utils.py`). AUC and R@P_N are not computed.

---

## 12. Loss Function Weights

- **Paper**: States that $\eta$, $\lambda$, and $\zeta$ (the weights for group, node, and transaction losses) are "determined by cross-validation".
- **Implementation**: All three weights are set to `1.0` by default (`eta=1.0`, `lambda_=1.0`, `zeta=1.0` in `main.ipynb`), meaning equal weighting without any cross-validation tuning.

---

## 13. Early Stopping

- **Paper**: Does not mention early stopping. The model is described as being trained for a fixed schedule (offline, nightly re-training).
- **Implementation**: Implements **early stopping** based on validation loss, with a configurable `patience` parameter (default 10 epochs). Training halts early if validation loss does not improve for `patience` consecutive epochs, and the best model weights are checkpointed.

---

## 14. Hardware and Scale

- **Paper**: Trains on ~60 million transaction samples using **four Tesla V100 GPUs**, with a training time of ~1.5 hours per nightly cycle.
- **Implementation**: Trains on the much smaller IBM AMLSim HI-Small dataset (~5M transactions, ~515K nodes) on a **single GPU** (CUDA if available, otherwise CPU).

---

## Summary Table

| Aspect | Paper | Implementation | Status |
|---|---|---|---|
| Dataset | UnionPay (proprietary, real-world) | IBM AMLSim HI-Small (synthetic, public) | Difference |
| Split strategy | Chronological (week-by-week) | Stratified random (70/10/20) | Difference |
| Validation set | Not mentioned | Explicit val set for early stopping | Difference |
| Down-sampling | User-level (select users + all their edges) | Node-index level (train mask only) | Difference |
| **Node label** | **Soft probability average of edge labels** | **Soft mean (Eq. 2) ✅** | **Resolved** |
| **GAT heads** | **k=5** | **k=5 ✅** | **Resolved** |
| **GAT concat mode** | **Not specified** | **concat=True (hidden), False (out) ✅** | **Resolved** |
| **Group encoder** | **Same encoder (weight sharing, Eq. 12)** | **Shared via group_proj + self.encoder ✅** | **Resolved** |
| **eMRF sigma** | **Hard binary indicator** | **Hard binary indicator (via STE) ✅** | **Resolved** |
| **y_group label** | **Any constituent node is ML** | **scatter-max over group_mapping ✅** | **Resolved** |
| Node aggregation | Mean (Eq. 11) | Mean (correctly follows Eq. 11) | Matches |
| **Neighbor sampling** | **Bernoulli distribution** | **Global edge dropout + dynamic Loader ✅** | **Resolved** |
| Metrics | AUC, R@P_N (transaction-level) | Accuracy, Precision, Recall, F1, CM (node-level) | Difference |
| Loss weights | Cross-validated | All equal to 1.0 | Difference |
| Early stopping | Not mentioned | Yes, patience-based | Difference |
| Hardware | 4× Tesla V100 | Single GPU | Difference |
