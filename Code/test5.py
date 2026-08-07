from loader.dataset_factory import DatasetFactory

# Inizializza il loader specificando il nome canonico del dataset
loader = DatasetFactory.get_loader("ibm_amlsim")

# Carica i dati in memoria (file delle transazioni e, se presente, degli account)
loader.load()

# Stampa un riepilogo conciso del dataset caricato per verificare che sia tutto ok
loader.summary()
loader.print_features()
from data_preparation import DataPreparation

# Otteniamo il dataframe raw delle transazioni dal nostro loader
transactions_df = loader.get_transactions()

# Inizializziamo la classe scegliendo il robust scaler
data_prep = DataPreparation(scaler_type='robust')

# 1. Calcoliamo le feature standardizzate per le transazioni (gli archi E)
edges_features_df = data_prep.fit_transform_edges(transactions_df)

# 2. Aggreghiamo le transazioni per ricavare l'embedding iniziale dei nodi (V)
nodes_features_df = data_prep.get_node_features(edges_features_df)

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

import torch
import numpy as np
import pandas as pd

# 1. Map string Account IDs to integers (0 to N-1) for PyTorch Geometric
unique_nodes = nodes_features_df.index.unique()
# Using a pandas Series for highly optimized, vectorized mapping
node_mapping = pd.Series(index=unique_nodes, data=np.arange(len(unique_nodes)))

# 2. Extract edge_index
src = edges_features_df['Account'].map(node_mapping).values
dst = edges_features_df['Account.1'].map(node_mapping).values
edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

# 3. Extract edge features and transaction labels
# Explicitly exclude 'Timestamp' so the Unix time is not used as a feature
edge_features_cols = [c for c in edges_features_df.columns if c not in ['Account', 'Account.1', 'Is Laundering', 'Timestamp']]
edge_attr = torch.tensor(edges_features_df[edge_features_cols].values, dtype=torch.float)
y_trans = torch.tensor(edges_features_df['Is Laundering'].values, dtype=torch.float).unsqueeze(1)

# 4. Extract node features and node labels
node_features_cols = [c for c in nodes_features_df.columns if c != 'Is Laundering']
x = torch.tensor(nodes_features_df[node_features_cols].values, dtype=torch.float)

# If a node was involved in at least one ML transaction, label it as 1 (suspicious)
y_node = torch.tensor(nodes_features_df['Is Laundering'].values, dtype=torch.float)  # Soft probability (paper Eq. 2)

print(f"Node features shape: {x.shape}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge features shape: {edge_attr.shape}")

# =============================================================================
# Hyperparameters — edit these values to configure the model and training
# =============================================================================

# Model Architecture Lists for Grid Search
hidden_dims_GAT  = [64]       # Hidden dimension for GAT layers (as per paper)
out_dims         = [32, 64, 128]       # Output dimension of community-centric encoder
heads_list       = [5]        # Number of GAT attention heads (k=5 as per paper)
betas            = [0.44]     # eMRF trade-off parameter (as per paper)
mlp_hidden_dims  = [32, 64, 128]       # Hidden dimension for the edge classification MLP
nn_t_hidden_dims = [128]      # Hidden dimension for node classification MLP (as per paper)

# Loss weights
c1_list         = [1] # Group loss weight
c2_list         = [0.25, 0.5, 1] # Node loss weight
c3_list         = [0.25, 0.5, 1] # Transaction loss weight

# Training & CV
learning_rates  = [0.001]    # Adam optimizer learning rate
max_norm_clipping = [5.0]
epochs_model_selection = 100        # Epochs for Cross Validation
epochs_train    = 100         # Maximum number of training epochs
minibatches     = True       # Train with minibatches
batch_size      = 128        # Seed nodes per mini-batch
num_neighbors   = [10, 10]    # max number of neighbours sampled for each node in the batch, set to -1 -1 to have no limits
patience = 10 # patience during model selection

# Class Imbalance
downsample      = True     # Down-sample majority class (legitimate nodes) in training set
print_every     = 1

# Split dataset
# trainset
date_time_1 = '2022-09-07 14:55:00'
# validation set
date_time_2 = '2022-09-08 16:12:00'
# test set
import pandas as pd
from torch_geometric.data import Data

# Convert Timestamp column to datetime
ts = pd.to_datetime(transactions_df['Timestamp'])

# Thresholds for 70% and 80% quantiles
thresh_val = pd.to_datetime('2022-09-07 14:55:00')
thresh_test = pd.to_datetime('2022-09-08 16:12:00')

# Create edge masks based on chronological splits
train_edge_mask = torch.tensor((ts < thresh_val).values, dtype=torch.bool)
val_edge_mask = torch.tensor(((ts >= thresh_val) & (ts < thresh_test)).values, dtype=torch.bool)
test_edge_mask = torch.tensor((ts >= thresh_test).values, dtype=torch.bool)

print(f"Train edges: {train_edge_mask.sum().item()} ({train_edge_mask.sum().item()/len(ts)*100:.1f}%)")
print(f"Val edges: {val_edge_mask.sum().item()} ({val_edge_mask.sum().item()/len(ts)*100:.1f}%)")
print(f"Test edges: {test_edge_mask.sum().item()} ({test_edge_mask.sum().item()/len(ts)*100:.1f}%)")

# Node mask: Using all True, as GAGNN expects a probability label for all nodes
node_mask = torch.ones(x.shape[0], dtype=torch.bool)

# Build PyG Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_trans=y_trans, y_node=y_node)
data.node_mask = node_mask
data.edge_train_mask = train_edge_mask
data.edge_val_mask = val_edge_mask
data.edge_test_mask = test_edge_mask

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
        input_nodes=None, shuffle=True
    )
    
    val_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=None, shuffle=False
    )
else:
    d_dev = data.to(device)
    train_loader = [d_dev]
    val_loader = [d_dev]

keys = ['max_norm', 'hidden_dim_GAT', 'out_dim', 'heads', 'beta', 'mlp_hidden_dim', 'nn_t_hidden_dim', 'lr', 'c1', 'c2', 'c3']
combinations = list(itertools.product(
    max_norm_clipping, hidden_dims_GAT, out_dims, heads_list, betas, mlp_hidden_dims, nn_t_hidden_dims, learning_rates, c1_list, c2_list, c3_list
))

print(f"Total hyperparameter combinations: {len(combinations)}\n")

best_val_loss = float('inf')
best_params = None

os.makedirs("saved_models", exist_ok=True)

for idx, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    print(f"--- Experiment {idx+1}/{len(combinations)} ---")
    print(params)
    
    model = GAGNN(
        node_in_dim=x.shape[1],
        edge_feat_dim=edge_attr.shape[1],
        hidden_dim=params['hidden_dim_GAT'],
        out_dim=params['out_dim'],
        heads=params['heads'],
        beta=params['beta'],
        mlp_hidden_dim=params['mlp_hidden_dim'],
        nn_t_hidden_dim=params['nn_t_hidden_dim'],
        minibatches=minibatches
    ).to(device)
    
    criterion = GAGNNLoss(c1=params['c1'], c2=params['c2'], c3=params['c3'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    epochs_no_improve = 0
    best_model_val_loss = float('inf')
    
    for epoch in range(epochs_model_selection):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            p_node, p_trans, p_group, y_group = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                y_trans=batch.y_trans, trans_mask=batch.edge_train_mask
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
        
        model.eval()
        val_loss_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                p_node, p_trans, p_group, y_group = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                )
                if p_group.numel() > 0:
                    val_loss, _, _, _ = criterion(
                        p_node, batch.y_node.view(-1, 1),
                        p_trans, batch.y_trans,
                        p_group, y_group,
                        node_mask=batch.node_mask,
                        trans_mask=batch.edge_val_mask
                    )
                    val_loss_total += val_loss.item()
                    n_val_batches += 1
                    
        avg_val_loss = val_loss_total / max(n_val_batches, 1)
        
        if avg_val_loss < best_model_val_loss:
            best_model_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    print(f"Best Val Loss for this config: {best_model_val_loss:.4f}")
    if best_model_val_loss < best_val_loss:
        best_val_loss = best_model_val_loss
        best_params = params
        model.save("saved_models/gagnn_best_model_selection.pt")
        print(">>> New Best Parameters! Model saved.")

print("\n=========================================")
print(f"Overall Best Val Loss: {best_val_loss:.4f}")
print(f"Best Parameters: {best_params}")
print("=========================================")
with open("saved_models/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("\nPlotting training history of the best model from model selection...")
best_selection_model = GAGNN.load_saved("saved_models/gagnn_best_model_selection.pt")
best_selection_model.plot_training_history()

import os
import json

print("\nLoading best hyperparameters for final training...")
with open("saved_models/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train + Val datasets...")
# Combine masks
final_train_mask = data.edge_train_mask | data.edge_val_mask

if minibatches:
    final_train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=None, shuffle=True
    )
else:
    final_train_loader = [d_dev]

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

final_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'])
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

os.makedirs("saved_models", exist_ok=True)

# No patience, just train for epochs_train
for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch in final_train_loader:
        batch = batch.to(device)
        final_optimizer.zero_grad()
        
        p_node, p_trans, p_group, y_group = final_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
            y_trans=batch.y_trans, trans_mask=final_train_mask
        )
        
        if p_group.numel() > 0:
            loss, _, _, _ = final_criterion(
                p_node, batch.y_node.view(-1, 1),
                p_trans, batch.y_trans,
                p_group, y_group,
                node_mask=batch.node_mask,
                trans_mask=final_train_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            final_optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    print(f"Final Train Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}")
    
    # Save model every 5 epochs
    if (epoch + 1) % 5 == 0:
        final_model.save(f"saved_models/model_trained_{epoch+1}.pt")
        print(f"  -> Saved model checkpoint to saved_models/model_trained_{epoch+1}.pt")
        
final_model.save(f"saved_models/model_trained_final.pt")

print("\nPlotting training history of the final trained model...")
final_model.plot_training_history()

from utils import Evaluator

print("\nLoading best model for final evaluation on Test set...")
eval_model = GAGNN.load_saved("saved_models/model_trained_final.pt").to(device)
eval_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'])

if minibatches:
    test_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=None, shuffle=False
    )
    Evaluator.evaluation_report(eval_model, test_loader, eval_criterion, device)
else:
    Evaluator.evaluation_report(eval_model, [d_dev], eval_criterion, device)

