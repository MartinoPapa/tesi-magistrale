import sys
sys.path.append('..')

from data_preparation import DataPreparation
from loader.dataset_factory import DatasetFactory
from model.gagnn import GAGNN
from model.loss import GAGNNLoss
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from utils import Evaluator
import itertools
import json
import numpy as np
import os
import pandas as pd
import torch

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


# =============================================================================
# GNN Configuration
# =============================================================================

# GNN backbone used in the CommunityCentricEncoder.
# Options: 'gat', 'gin', 'transformer_conv', 'gps_conv'
gnn_type = 'gat'

# Folder to save models and parameters
save_to_folder = 'saved_models_GAT'

# =============================================================================
# Model Architecture — Grid Search Lists
# =============================================================================

# Hidden dimension inside GNN layers (intermediate representation size)
hidden_dims = [64, 128]

# Output dimension of the community-centric encoder (node embedding size fed to eMRF and classifiers)
out_dims = [64, 128, 256]

# Number of attention heads in GAT layers. Controls how many independent
# attention mechanisms run in parallel. The paper uses k=5.
heads_list = [3, 5, 7, 9]

# Number of stacked GNN layers (depth of the message-passing network).
# More layers = larger receptive field but risk of over-smoothing.
num_layers_list = [2, 3]

# eMRF trade-off parameter (beta) between topological similarity (modularity)
# and attribute similarity (cosine). beta=0.44 as per paper.
betas = [0.44]

# Hidden dimension of the MLP used for edge classification in the GroupRepresentationLayer.
# This MLP takes [Z_i, Z_j, edge_features] as input and predicts edge labels.
mlp_hidden_dims = [64, 128, 256]

# Hidden dimension of the node classification MLP (NN_t). This MLP generates
# the final node-level predictions X^(4). The paper uses 128.
nn_t_hidden_dims = [128, 256]

# =============================================================================
# Loss Weights
# =============================================================================

# Weight for the group-level loss (L_group)
c1_list = [0.75]

# Weight for the node-level loss (L_node)
c2_list = [0.75]

# Weight for the transaction-level loss (L_trans)
c3_list = [1]

# Class weight multiplier for the laundering (positive) class.
# Higher values penalize false negatives more.
laundry_weights = [1.0]

# =============================================================================
# Training & Cross-Validation
# =============================================================================

# Adam optimizer learning rate
learning_rates = [0.0005]

# Maximum gradient norm for gradient clipping (prevents exploding gradients)
max_norm_clipping = [5.0]

# Maximum number of epochs during model selection (grid search)
epochs_model_selection = 700

# Maximum number of epochs during final training
epochs_train = 1000

# Whether to use mini-batch training with NeighborLoader
minibatches = True

# Number of nodes per mini-batch
batch_size = 512

# Max number of neighbours sampled per node per GNN layer during mini-batching.
# Set to [-1, -1] to sample all neighbours (full neighbourhood).
num_neighbors = [10, 10]

# Early stopping patience during model selection (grid search)
patience_selection = 100

# Early stopping patience during final training
patience_training = 200

# Print training metrics every N epochs
print_every = 2

# Save model checkpoint every N epochs
save_every = 10

# =============================================================================
# Class Imbalance
# =============================================================================

# Down-sample majority class (legitimate nodes) in training set to balance classes
downsample = True

# =============================================================================
# Temporal Train / Validation / Test Split
# =============================================================================

# trainset ends at this timestamp
date_time_1 = '2022-09-07 14:55:00'
# validation set ends at this timestamp
date_time_2 = '2022-09-08 16:12:00'
# test set starts from date_time_2

from loader.dataset_factory import DatasetFactory

# Inizializza il loader specificando il nome canonico del dataset
loader = DatasetFactory.get_loader(
    "ibm_amlsim", 
    engine="pyarrow", 
    dtype_backend="pyarrow", 
    nrows=None
)
# pyarrow is faster and uses less memory, it takes NumPy place

# Carica i dati in memoria (file delle transazioni e, se presente, degli account)
loader.load()

# Stampa un riepilogo conciso del dataset caricato per verificare che sia tutto ok
loader.summary()

loader.print_features()

from data_preparation import DataPreparation
import pandas as pd

# Otteniamo il dataframe raw delle transazioni dal nostro loader
transactions_df = loader.get_transactions()

# Parse multiclass patterns
data_prep = DataPreparation(scaler_type='robust')
transactions_df = data_prep.parse_patterns_file(transactions_df, 'data/ibm_amlsim/HI-Small_Patterns.txt')

# Calcoliamo la maschera di training per evitare data leakage nello scaler
ts_raw = pd.to_datetime(transactions_df['Timestamp'])
thresh_val = pd.to_datetime(date_time_1)
train_mask_raw = ts_raw < thresh_val

# Inizializziamo la classe scegliendo il robust scaler
data_prep = DataPreparation(scaler_type='robust')

# 1. Calcoliamo le feature standardizzate per le transazioni (gli archi E)
edges_features_df = data_prep.fit_transform_edges(transactions_df, train_mask=train_mask_raw)

# 2. Aggreghiamo le transazioni per ricavare l'embedding iniziale dei nodi (V)
nodes_features_df = data_prep.get_node_features(edges_features_df, train_mask=train_mask_raw)

print(f"Dimensione feature archi: {edges_features_df.shape}")
print(f"Dimensione feature nodi: {nodes_features_df.shape}")

# Stampa bilanciamento classi (Nodi e Archi)
print("\n--- Bilanciamento Dataset ---")
num_fraud_nodes = (nodes_features_df["Is Laundering"] > 0).sum()
num_legit_nodes = len(nodes_features_df) - num_fraud_nodes
print(f"Nodi fraudolenti: {num_fraud_nodes} ({num_fraud_nodes/len(nodes_features_df)*100:.2f}%)")
print(f"Nodi leciti:      {num_legit_nodes} ({num_legit_nodes/len(nodes_features_df)*100:.2f}%)")

num_fraud_edges = (edges_features_df["Is Laundering"] > 0).sum()
num_legit_edges = len(edges_features_df) - num_fraud_edges
print(f"Archi fraudolenti: {num_fraud_edges} ({num_fraud_edges/len(edges_features_df)*100:.2f}%)")
print(f"Archi leciti:      {num_legit_edges} ({num_legit_edges/len(edges_features_df)*100:.2f}%)")

print("\n--- Features degli archi ---")
print(list(edges_features_df.columns))

print("\n--- Features dei nodi ---")
print(list(nodes_features_df.columns))


import torch
import numpy as np
import pandas as pd

print("--- Downsampling Strategy (1:1 Ratio) ---")
# 1. Identify Fraudulent and Legitimate Nodes
# A node is fraudulent if it was involved in at least one ML transaction
fraud_nodes = nodes_features_df[nodes_features_df["Is Laundering"] > 0].index
legit_nodes = nodes_features_df[nodes_features_df["Is Laundering"] == 0].index

print(f"Total Fraudulent Nodes: {len(fraud_nodes)}")
print(f"Total Legitimate Nodes (before sampling): {len(legit_nodes)}")

# 2. Sample Legitimate Nodes (1:1 ratio)
sampled_legit_nodes = pd.Series(legit_nodes).sample(n=len(fraud_nodes), random_state=42).values

# 3. Combine to form the final set of sampled users
sampled_users = set(fraud_nodes).union(set(sampled_legit_nodes))
print(f"Total Nodes after downsampling: {len(sampled_users)}")

# 4. Filter nodes_features_df and edges_features_df efficiently
nodes_features_df = nodes_features_df.loc[list(sampled_users)]

# Filter transactions: keep only those where BOTH sender and receiver are in sampled_users
edges_features_df = edges_features_df[
    edges_features_df['Account'].isin(sampled_users) & 
    edges_features_df['Account.1'].isin(sampled_users)
]

print(f"Total Edges after downsampling: {len(edges_features_df)}")

# 5. Map string Account IDs to integers (0 to N-1) for PyTorch Geometric
unique_nodes = nodes_features_df.index.unique()
node_mapping = pd.Series(index=unique_nodes, data=np.arange(len(unique_nodes)))

# 6. Extract edge_index
src = edges_features_df['Account'].map(node_mapping).values
dst = edges_features_df['Account.1'].map(node_mapping).values
edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

# 7. Extract edge features and transaction labels
edge_features_cols = [c for c in edges_features_df.columns if c not in ['Account', 'Account.1', 'Is Laundering', 'Timestamp']]
edge_attr = torch.tensor(edges_features_df[edge_features_cols].values, dtype=torch.float)
y_trans = torch.tensor(edges_features_df['Is Laundering'].values, dtype=torch.float).unsqueeze(1)

# 8. Extract node features and node labels
node_features_cols = [c for c in nodes_features_df.columns if c != 'Is Laundering']
x = torch.tensor(nodes_features_df[node_features_cols].values, dtype=torch.float)

y_node = torch.tensor(nodes_features_df['Is Laundering'].values, dtype=torch.float)

print(f"Node features shape: {x.shape}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge features shape: {edge_attr.shape}")


import pandas as pd
from torch_geometric.data import Data

# Convert Timestamp column to datetime (use downsampled edges_features_df)
ts = pd.to_datetime(edges_features_df['Timestamp'])

# Thresholds for splits
thresh_val = pd.to_datetime(date_time_1)
thresh_test = pd.to_datetime(date_time_2)

# Create edge masks based on chronological splits
train_edge_mask = torch.tensor((ts < thresh_val).values, dtype=torch.bool)
val_edge_mask = torch.tensor(((ts >= thresh_val) & (ts < thresh_test)).values, dtype=torch.bool)
test_edge_mask = torch.tensor((ts >= thresh_test).values, dtype=torch.bool)

print(f"Train edges: {train_edge_mask.sum().item()} ({train_edge_mask.sum().item()/len(ts)*100:.1f}%)")
print(f"Val edges: {val_edge_mask.sum().item()} ({val_edge_mask.sum().item()/len(ts)*100:.1f}%)")
print(f"Test edges: {test_edge_mask.sum().item()} ({test_edge_mask.sum().item()/len(ts)*100:.1f}%)")

# Create node masks for NeighborLoader input_nodes
# A node is in the training set if it is connected to at least one training edge
train_nodes_idx = edge_index[:, train_edge_mask].flatten().unique()
train_node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
train_node_mask[train_nodes_idx] = True

val_nodes_idx = edge_index[:, val_edge_mask].flatten().unique()
val_node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
val_node_mask[val_nodes_idx] = True

test_nodes_idx = edge_index[:, test_edge_mask].flatten().unique()
test_node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
test_node_mask[test_nodes_idx] = True

# Node mask: Using all True, as GAGNN expects a probability label for all nodes
node_mask = torch.ones(x.shape[0], dtype=torch.bool)

# Build PyG Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_trans=y_trans, y_node=y_node)
data.node_mask = node_mask
data.edge_train_mask = train_edge_mask
data.edge_val_mask = val_edge_mask
data.edge_test_mask = test_edge_mask
data.train_node_mask = train_node_mask
data.val_node_mask = val_node_mask
data.test_node_mask = test_node_mask


import os
import json
import itertools
from model.gagnn import GAGNN
from model.loss import GAGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if minibatches:
    train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    
    val_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    train_loader = [d_dev]
    val_loader = [d_dev]

print(f"Batches per epoch (Train): {len(train_loader)} | (Val): {len(val_loader)}")

keys = ['max_norm', 'hidden_dim', 'out_dim', 'heads', 'num_layers',
        'beta', 'mlp_hidden_dim', 'nn_t_hidden_dim', 'lr',
        'c1', 'c2', 'c3', 'laundry_weight']
combinations = list(itertools.product(
    max_norm_clipping, hidden_dims, out_dims, heads_list, num_layers_list,
    betas, mlp_hidden_dims, nn_t_hidden_dims, learning_rates,
    c1_list, c2_list, c3_list, laundry_weights
))

print(f"Total hyperparameter combinations: {len(combinations)}\n")

from sklearn.metrics import f1_score
import numpy as np
best_val_f1 = -1.0
best_params = None

os.makedirs(save_to_folder, exist_ok=True)

for idx, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    print(f"--- Experiment {idx+1}/{len(combinations)} ---")
    print(params)
    
    model = GAGNN(
        node_in_dim=x.shape[1],
        edge_feat_dim=edge_attr.shape[1],
        hidden_dim=params['hidden_dim'],
        out_dim=params['out_dim'],
        heads=params['heads'],
        beta=params['beta'],
        mlp_hidden_dim=params['mlp_hidden_dim'],
        nn_t_hidden_dim=params['nn_t_hidden_dim'],
        minibatches=minibatches,
        gnn_type=gnn_type,
        num_layers=params['num_layers']
    ).to(device)
    
    criterion = GAGNNLoss(c1=params['c1'], c2=params['c2'], c3=params['c3'], laundry_weight=params['laundry_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    epochs_no_improve = 0
    best_model_val_f1 = -1.0
    
    for epoch in range(epochs_model_selection):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            p_node, p_trans, p_group, y_group = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_train_mask
            )
            
            if p_group.numel() > 0:
                loss, _, _, _ = criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_train_mask
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                
        train_loss = epoch_loss / max(n_batches, 1)
        model.training_losses.append(train_loss)
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                p_node, p_trans, p_group, y_group = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                )
                if p_group.numel() > 0:
                    p_trans_val = p_trans[batch.edge_val_mask]
                    y_trans_val = batch.y_trans[batch.edge_val_mask]
                    probs = torch.softmax(p_trans_val, dim=1)
                    probs_flat = probs.cpu().numpy()
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    labels = y_trans_val.view(-1).long().cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels)
                    
        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            val_f1 = 0.0

        model.val_F1.append(val_f1)
        if val_f1 > best_model_val_f1:
            best_model_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_selection:
            print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val F1-score: {val_f1:.4f}")
            break
            
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val F1-score: {val_f1:.4f}")   
        
            
    print(f"Best Val F1-score for this config: {best_model_val_f1:.4f}")
    if best_model_val_f1 > best_val_f1:
        best_val_f1 = best_model_val_f1
        best_params = params
        model.save(f"{save_to_folder}/gagnn_best_model_selection.pt")
        print(">>> New Best Parameters! Model saved.")
    else:
        print(f"No improvement over best parameters. Current: {best_model_val_f1:.4f} | Best: {best_val_f1:.4f}")

print("\n=========================================")
print(f"Overall Best Val F1-score: {best_val_f1:.4f}")
print(f"Best Parameters: {best_params}")
print("=========================================")
with open(f"{save_to_folder}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("\nPlotting training history of the best model from model selection...")
best_selection_model = GAGNN.load_saved(f"{save_to_folder}/gagnn_best_model_selection.pt")
best_selection_model.plot_training_history()

import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best hyperparameters for final training...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train dataset with validation...")

if minibatches:
    final_train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    final_train_loader = [d_dev]
    final_val_loader = [d_dev]

print(f"Batches per epoch (Final Train): {len(final_train_loader)}")

# Re-initialize the model with best parameters
final_model = GAGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=best_params['hidden_dim'],
    out_dim=best_params['out_dim'],
    heads=best_params['heads'],
    beta=best_params['beta'],
    mlp_hidden_dim=best_params['mlp_hidden_dim'],
    nn_t_hidden_dim=best_params['nn_t_hidden_dim'],
    minibatches=minibatches,
    gnn_type=gnn_type,
    num_layers=best_params['num_layers']
).to(device)

final_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'], laundry_weight=best_params.get('laundry_weight', 2.0))
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

os.makedirs(save_to_folder, exist_ok=True)

best_final_val_f1 = -1.0
epochs_no_improve = 0

for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch in final_train_loader:
        batch = batch.to(device)
        final_optimizer.zero_grad()
        
        p_node, p_trans, p_group, y_group = final_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
            y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_train_mask
        )
        
        if p_group.numel() > 0:
            loss, _, _, _ = final_criterion(
                p_node, batch.y_node.view(-1, 1),
                p_trans, batch.y_trans,
                p_group, y_group,
                node_mask=batch.node_mask,
                trans_mask=batch.edge_train_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            final_optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    final_model.training_losses.append(train_loss)
    
    # Validation step
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in final_val_loader:
            batch = batch.to(device)
            p_node, p_trans, p_group, y_group = final_model(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
            )
            if p_group.numel() > 0:
                p_trans_val = p_trans[batch.edge_val_mask]
                y_trans_val = batch.y_trans[batch.edge_val_mask]
                probs = torch.sigmoid(p_trans_val)
                probs_flat = probs.view(-1).cpu().numpy()
                preds = (probs_flat >= 0.5).astype(int)
                labels = y_trans_val.view(-1).long().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)
                
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        val_f1 = 0.0
        
    final_model.val_F1.append(val_f1)
    
    if (epoch + 1) % print_every == 0:
        print(f"Train Epoch {epoch+1:03d}/{epochs_train} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
    if val_f1 > best_final_val_f1:
        best_final_val_f1 = val_f1
        epochs_no_improve = 0
        final_model.save(f"{save_to_folder}/model_trained_final.pt")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience_training:
        print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val F1-score: {val_f1:.4f}")
        break
        
    # Save model every 10 epochs
    if (epoch + 1) % save_every == 0:
        final_model.save(f"{save_to_folder}/model_trained_{epoch+1}.pt")

print("\nPlotting training history of the final trained model...")
final_model.plot_training_history()

from utils import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best model for final evaluation on Test set...")
eval_model = GAGNN.load_saved(f"{save_to_folder}/model_trained_final.pt").to(device)

with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

eval_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'])

if minibatches:
    test_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.test_node_mask, shuffle=False
    )
    Evaluator.evaluation_report(eval_model, test_loader, eval_criterion, device, edge_mask_name='edge_test_mask', task='multiclass')
else:
    Evaluator.evaluation_report(eval_model, [d_dev], eval_criterion, device, edge_mask_name='edge_test_mask', task='multiclass')


eval_model.plot_training_history()

import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best hyperparameters for final training...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train dataset with validation...")

if minibatches:
    final_train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    final_train_loader = [d_dev]
    final_val_loader = [d_dev]

print(f"Batches per epoch (Final Train): {len(final_train_loader)}")

# Re-initialize the model with best parameters
final_model = GAGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=best_params['hidden_dim_GAT'],
    out_dim=best_params['out_dim'],
    heads=best_params['heads'],
    beta=best_params['beta'],
    mlp_hidden_dim=best_params['mlp_hidden_dim'],
    nn_t_hidden_dim=best_params['nn_t_hidden_dim'],
    minibatches=minibatches
).to(device)

final_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'], laundry_weight=2.0)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

os.makedirs(save_to_folder, exist_ok=True)

best_final_val_f1 = -1.0
epochs_no_improve = 0

for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch in final_train_loader:
        batch = batch.to(device)
        final_optimizer.zero_grad()
        
        p_node, p_trans, p_group, y_group = final_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
            y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_train_mask
        )
        
        if p_group.numel() > 0:
            loss, _, _, _ = final_criterion(
                p_node, batch.y_node.view(-1, 1),
                p_trans, batch.y_trans,
                p_group, y_group,
                node_mask=batch.node_mask,
                trans_mask=batch.edge_train_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            final_optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    final_model.training_losses.append(train_loss)
    
    # Validation step
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in final_val_loader:
            batch = batch.to(device)
            p_node, p_trans, p_group, y_group = final_model(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
            )
            if p_group.numel() > 0:
                p_trans_val = p_trans[batch.edge_val_mask]
                y_trans_val = batch.y_trans[batch.edge_val_mask]
                probs = torch.sigmoid(p_trans_val)
                probs_flat = probs.view(-1).cpu().numpy()
                preds = (probs_flat >= 0.5).astype(int)
                labels = y_trans_val.view(-1).long().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)
                
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        val_f1 = 0.0
        
    final_model.val_F1.append(val_f1)
    
    if (epoch + 1) % print_every == 0:
        print(f"Train Epoch {epoch+1:03d}/{epochs_train} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
    if val_f1 > best_final_val_f1:
        best_final_val_f1 = val_f1
        epochs_no_improve = 0
        final_model.save(f"{save_to_folder}/model_trained_weighted2_final.pt")
        # print(">>> Saved new best model checkpoint.")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience_training:
        print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val F1-score: {val_f1:.4f}")
        break
        
    # Save model every 10 epochs
    if (epoch + 1) % save_every == 0:
        final_model.save(f"{save_to_folder}/model_trained_weighted2_{epoch+1}.pt")
        # print(f"  -> Saved model checkpoint to saved_models/model_trained_weighted2_{epoch+1}.pt")

print("\nPlotting training history of the final trained model...")
final_model.plot_training_history()


from utils import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best model for final evaluation on Test set...")
eval_model = GAGNN.load_saved(f"{save_to_folder}/model_trained_weighted2_final.pt").to(device)

with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

eval_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'], laundry_weight=2.0)

if minibatches:
    test_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.test_node_mask, shuffle=False
    )
    Evaluator.evaluation_report(eval_model, test_loader, eval_criterion, device, edge_mask_name='edge_test_mask', task='multiclass')
else:
    Evaluator.evaluation_report(eval_model, [d_dev], eval_criterion, device, edge_mask_name='edge_test_mask', task='multiclass')


# =============================================================================
# Baseline GAT — Hyperparameter Grid
# =============================================================================

# Folder where baseline models and params are saved
baseline_save_to_folder = 'saved_models_GAT/baseline'

# Hidden dimension inside GNN layers
baseline_hidden_dims = [64, 128]

# Output node-embedding dimension (last GNN layer)
baseline_out_dims = [64, 128, 256]

# Number of attention heads
baseline_heads_list = [3, 5, 7]

# Number of stacked GNN layers
baseline_num_layers_list = [2, 3]

# Hidden dim of the EdgeMLP: concat(Z_i, Z_j, edge_attr) -> logit
baseline_mlp_hidden_dims = [64, 128, 256]

# Adam optimizer learning rate
baseline_learning_rates = [0.0005]

# Positive-class weight (penalises false negatives)
baseline_laundry_weights = [1.0]

# Max gradient norm for clipping
baseline_max_norm_clipping = [5.0]

# Maximum epochs during model selection
baseline_epochs_model_selection = 700

# Maximum epochs during final training
baseline_epochs_train = 1000

# Early stopping patience (model selection / final training)
baseline_patience_selection = 100
baseline_patience_training  = 200

import os
import json
import itertools
import numpy as np
from sklearn.metrics import f1_score
from model.standalone_gnn import StandaloneGNN, StandaloneGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if minibatches:
    baseline_train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    baseline_val_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    baseline_train_loader = [d_dev]
    baseline_val_loader   = [d_dev]

print(f"Batches per epoch (Train): {len(baseline_train_loader)} | (Val): {len(baseline_val_loader)}")

keys         = ['max_norm', 'hidden_dim', 'out_dim', 'heads', 'num_layers', 'mlp_hidden_dim', 'lr', 'laundry_weight']
combinations = list(itertools.product(
    baseline_max_norm_clipping,
    baseline_hidden_dims,
    baseline_out_dims,
    baseline_heads_list,
    baseline_num_layers_list,
    baseline_mlp_hidden_dims,
    baseline_learning_rates,
    baseline_laundry_weights
))
print(f"Total baseline combinations: {len(combinations)}\n")

best_val_f1 = -1.0
best_params = None
os.makedirs(baseline_save_to_folder, exist_ok=True)

for idx, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    print(f"--- Experiment {idx + 1}/{len(combinations)} ---")
    print(params)

    model = StandaloneGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=params['hidden_dim'],
    out_dim=params['out_dim'],
    heads=params['heads'],
    num_layers=params['num_layers'],
    gnn_type='gat',
    mlp_hidden_dim=params['mlp_hidden_dim'],
    minibatches=minibatches,
).to(device)

    criterion = StandaloneGNNLoss(laundry_weight=params['laundry_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    best_model_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(baseline_epochs_model_selection):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in baseline_train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            p_trans = model(batch.x, batch.edge_index, batch.edge_attr)
            loss    = criterion(p_trans, batch.y_trans,
                                trans_mask=batch.edge_train_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_norm'])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        model.training_losses.append(epoch_loss / max(n_batches, 1))

        # --- Validation ---
        model.eval()
        val_preds, val_labels = [], []
        evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool)
        with torch.no_grad():
            for batch in baseline_val_loader:
                batch = batch.to(device)
                if hasattr(batch, 'e_id'):
                    e_id     = batch.e_id.cpu()
                    new_mask = ~evaluated_val[e_id]
                    evaluated_val[e_id] = True
                else:
                    new_mask = torch.ones(batch.num_edges, dtype=torch.bool)
                val_mask = new_mask & batch.edge_val_mask.cpu()
                if not val_mask.any():
                    continue
                p_trans = model(batch.x, batch.edge_index, batch.edge_attr)
                probs   = torch.sigmoid(p_trans[val_mask]).view(-1).cpu().numpy()
                preds   = (probs >= 0.5).astype(int)
                labels  = batch.y_trans[val_mask].view(-1).long().cpu().numpy()
                val_preds.append(preds)
                val_labels.append(labels)

        val_f1 = (
            f1_score(np.concatenate(val_labels), np.concatenate(val_preds), zero_division=0)
            if val_preds else 0.0
        )
        model.val_F1.append(val_f1)

        if val_f1 > best_model_val_f1:
            best_model_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= baseline_patience_selection:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    print(f"  Best val F1 = {best_model_val_f1:.4f}")
    if best_model_val_f1 > best_val_f1:
        best_val_f1 = best_model_val_f1
        best_params = params
        print(f"  ** New best! Val F1 = {best_val_f1:.4f}")

print(f"\nBest baseline params : {best_params}")
print(f"Best baseline val F1 : {best_val_f1:.4f}")
with open(f"{baseline_save_to_folder}/best_baseline_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("Saved best_baseline_params.json")


import os
import json
import numpy as np
from sklearn.metrics import f1_score
from model.standalone_gnn import StandaloneGNN, StandaloneGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best baseline hyperparameters...")
with open(f"{baseline_save_to_folder}/best_baseline_params.json", "r") as f:
    best_baseline_params = json.load(f)
print(best_baseline_params)

print("\nStarting baseline final training...")

if minibatches:
    final_train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_val_loader   = [d_dev]

final_model = StandaloneGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=best_baseline_params['hidden_dim'],
    out_dim=best_baseline_params['out_dim'],
    heads=best_baseline_params['heads'],
    num_layers=best_baseline_params['num_layers'],
    gnn_type='gat',
    mlp_hidden_dim=best_baseline_params['mlp_hidden_dim'],
    minibatches=minibatches,
).to(device)

final_criterion = StandaloneGNNLoss(
    laundry_weight=best_baseline_params['laundry_weight']
)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=best_baseline_params['lr']
)

os.makedirs(baseline_save_to_folder, exist_ok=True)

best_final_val_f1 = -1.0
epochs_no_improve = 0

for epoch in range(baseline_epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch in final_train_loader:
        batch = batch.to(device)
        final_optimizer.zero_grad()

        p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
        loss    = final_criterion(p_trans, batch.y_trans,
                                  trans_mask=batch.edge_train_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            final_model.parameters(), best_baseline_params['max_norm']
        )
        final_optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    final_model.training_losses.append(epoch_loss / max(n_batches, 1))

    # --- Validation ---
    final_model.eval()
    val_preds, val_labels = [], []
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool)
    with torch.no_grad():
        for batch in final_val_loader:
            batch = batch.to(device)
            if hasattr(batch, 'e_id'):
                e_id     = batch.e_id.cpu()
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool)
            val_mask = new_mask & batch.edge_val_mask.cpu()
            if not val_mask.any():
                continue
            p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
            probs   = torch.sigmoid(p_trans[val_mask]).view(-1).cpu().numpy()
            preds   = (probs >= 0.5).astype(int)
            labels  = batch.y_trans[val_mask].view(-1).long().cpu().numpy()
            val_preds.append(preds)
            val_labels.append(labels)

    val_f1 = (
        f1_score(np.concatenate(val_labels), np.concatenate(val_preds), zero_division=0)
        if val_preds else 0.0
    )
    final_model.val_F1.append(val_f1)

    if epoch % print_every == 0:
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1:4d} | Loss {avg_loss:.4f} | Val F1 {val_f1:.4f}")

    if val_f1 > best_final_val_f1:
        best_final_val_f1 = val_f1
        epochs_no_improve = 0
        final_model.save(f"{baseline_save_to_folder}/baseline_model_final.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= baseline_patience_training:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"\nBest baseline val F1 (final training): {best_final_val_f1:.4f}")
print("Model saved to baseline_model_final.pt")


from model.standalone_gnn import StandaloneGNN
from utils import Evaluator
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best baseline model for test evaluation...")
baseline_eval_model = StandaloneGNN.load_saved(
    f"{baseline_save_to_folder}/baseline_model_final.pt"
).to(device)

if minibatches:
    baseline_test_loader = NeighborLoader(
        data, num_neighbors=[-1, -1], batch_size=batch_size,
        input_nodes=data.test_node_mask, shuffle=False
    )
    Evaluator.baseline_evaluation_report(
        baseline_eval_model, baseline_test_loader, device,
        edge_mask_name='edge_test_mask'
    )
else:
    Evaluator.baseline_evaluation_report(
        baseline_eval_model, [d_dev], device,
        edge_mask_name='edge_test_mask'
    )


baseline_eval_model.plot_training_history()

