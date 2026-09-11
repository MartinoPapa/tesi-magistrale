# Code/Binary/main_GAT.ipynb

## Cell 18
`python
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best hyperparameters for final training...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train dataset with validation...")



# Merge patience into train
data.train_node_mask = data.train_node_mask | data.patience_node_mask
data.edge_train_mask = data.edge_train_mask | data.edge_patience_mask

# Use val as new patience
data.patience_node_mask = data.val_node_mask
data.edge_patience_mask = data.edge_val_mask

if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
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
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch in final_train_loader:
        final_optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            scaler.scale(loss).backward()
            scaler.unscale_(final_optimizer)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            scaler.step(final_optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    final_model.training_losses.append(train_loss)
    
    # Validation step
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=val_mask
                )
                val_loss_epoch += loss.item()
                n_val_batches += 1
                
        
    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)
    
    if (epoch + 1) % print_every == 0:
        print(f"Train Epoch {epoch+1:03d}/{epochs_train} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{save_to_folder}/model_trained_final.pt")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience_training:
        print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        break
        
    # Save model every 10 epochs
    if (epoch + 1) % save_every == 0:
        final_model.save(f"{save_to_folder}/model_trained_{epoch+1}.pt")

print("\nPlotting training history of the final trained model...")
`

## Cell 25
`python
import os
import json
import numpy as np
from sklearn.metrics import f1_score
from model.standalone_gnn import StandaloneGNN, StandaloneGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best GAGNN hyperparameters to use for baseline...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_baseline_params = json.load(f)
print(best_baseline_params)

print("\nStarting baseline final training...")


if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
    final_val_loader = [d_dev]

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
    laundry_weight=1.0
)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=best_baseline_params['lr']
)
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(baseline_save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(baseline_epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch in final_train_loader:
        final_optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                loss = final_criterion(p_trans, batch.y_trans, trans_mask=batch.edge_train_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(final_optimizer)
        torch.nn.utils.clip_grad_norm_(
            final_model.parameters(), best_baseline_params['max_norm']
        )
        scaler.step(final_optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1

    final_model.training_losses.append(epoch_loss / max(n_batches, 1))

    # --- Validation ---
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            if hasattr(batch, 'e_id'):
                e_id     = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = final_criterion(p_trans, batch.y_trans, trans_mask=val_mask)
                    val_loss_epoch += loss.item()
                    n_val_batches += 1

    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)

    if epoch % print_every == 0:
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1:4d} | Loss {avg_loss:.4f} | Val Loss {val_loss:.4f}")

    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{baseline_save_to_folder}/baseline_model_final.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= baseline_patience_training:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"\nBest baseline val Loss (final training): {best_final_val_loss:.4f}")
print("Model saved to baseline_model_final.pt")

`
# Code/Binary/main_GIN.ipynb

## Cell 14
`python
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best hyperparameters for final training...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train dataset with validation...")



# Merge patience into train
data.train_node_mask = data.train_node_mask | data.patience_node_mask
data.edge_train_mask = data.edge_train_mask | data.edge_patience_mask

# Use val as new patience
data.patience_node_mask = data.val_node_mask
data.edge_patience_mask = data.edge_val_mask

if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
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
    num_layers=best_params['num_layers'],
    gin_mlp_hidden_dim=best_params['gin_mlp_hidden_dim']
).to(device)

final_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'], laundry_weight=best_params.get('laundry_weight', 2.0))
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch in final_train_loader:
        final_optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            scaler.scale(loss).backward()
            scaler.unscale_(final_optimizer)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            scaler.step(final_optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    final_model.training_losses.append(train_loss)
    
    # Validation step
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=val_mask
                )
                val_loss_epoch += loss.item()
                n_val_batches += 1
                
        
    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)
    
    if (epoch + 1) % print_every == 0:
        print(f"Train Epoch {epoch+1:03d}/{epochs_train} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{save_to_folder}/model_trained_final.pt")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience_training:
        print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        break
        
    # Save model every 10 epochs
    if (epoch + 1) % save_every == 0:
        final_model.save(f"{save_to_folder}/model_trained_{epoch+1}.pt")

print("\nPlotting training history of the final trained model...")
final_model.plot_training_history()


`

## Cell 21
`python
import os
import json
import numpy as np
from sklearn.metrics import f1_score
from model.standalone_gnn import StandaloneGNN, StandaloneGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best GAGNN hyperparameters to use for baseline...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_baseline_params = json.load(f)
print(best_baseline_params)

print("\nStarting baseline final training...")


if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
    final_val_loader = [d_dev]

final_model = StandaloneGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=best_baseline_params['hidden_dim'],
    out_dim=best_baseline_params['out_dim'],
    num_layers=best_baseline_params['num_layers'],
    gnn_type='gin',
    gin_mlp_hidden_dim=best_baseline_params['gin_mlp_hidden_dim'],
    mlp_hidden_dim=best_baseline_params['mlp_hidden_dim'],
    minibatches=minibatches,
).to(device)

final_criterion = StandaloneGNNLoss(
    laundry_weight=1.0
)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=best_baseline_params['lr']
)
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(baseline_save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(baseline_epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch in final_train_loader:
        final_optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                loss = final_criterion(p_trans, batch.y_trans, trans_mask=batch.edge_train_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(final_optimizer)
        torch.nn.utils.clip_grad_norm_(
            final_model.parameters(), best_baseline_params['max_norm']
        )
        scaler.step(final_optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1

    final_model.training_losses.append(epoch_loss / max(n_batches, 1))

    # --- Validation ---
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            if hasattr(batch, 'e_id'):
                e_id     = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = final_criterion(p_trans, batch.y_trans, trans_mask=val_mask)
                    val_loss_epoch += loss.item()
                    n_val_batches += 1

    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)

    if epoch % print_every == 0:
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1:4d} | Loss {avg_loss:.4f} | Val Loss {val_loss:.4f}")

    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{baseline_save_to_folder}/baseline_model_final.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= baseline_patience_training:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"\nBest baseline val Loss (final training): {best_final_val_loss:.4f}")
print("Model saved to baseline_model_final.pt")








`
# Code/Multiclass/main_GAT_multiclass.ipynb

## Cell 19
`python
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best hyperparameters for final training...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train dataset with validation...")



# Merge patience into train
data.train_node_mask = data.train_node_mask | data.patience_node_mask
data.edge_train_mask = data.edge_train_mask | data.edge_patience_mask

# Use val as new patience
data.patience_node_mask = data.val_node_mask
data.edge_patience_mask = data.edge_val_mask

if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
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
    minibatches=minibatches, num_classes=9,
    gnn_type=gnn_type,
    num_layers=best_params['num_layers']
).to(device)

final_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'], laundry_weight=class_weights, task='multiclass')
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    evaluated_train = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    for batch in final_train_loader:
        if hasattr(batch, 'e_id'):
            e_id = batch.e_id
            new_mask = ~evaluated_train[e_id]
            evaluated_train[e_id] = True
        else:
            new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        train_mask = new_mask & batch.edge_train_mask
        if not train_mask.any():
            continue
        final_optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            p_node, p_trans, p_group, y_group = final_model(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=train_mask
            )
        
        if p_group.numel() > 0:
            loss, _, _, _ = final_criterion(
                p_node, batch.y_node.view(-1, 1),
                p_trans, batch.y_trans,
                p_group, y_group,
                node_mask=batch.node_mask,
                trans_mask=train_mask
            )
            scaler.scale(loss).backward()
            scaler.unscale_(final_optimizer)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            scaler.step(final_optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    final_model.training_losses.append(train_loss)
    
    # Validation step
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=val_mask
                )
                val_loss_epoch += loss.item()
                n_val_batches += 1
                
        
    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)
    
    if (epoch + 1) % print_every == 0:
        print(f"Train Epoch {epoch+1:03d}/{epochs_train} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{save_to_folder}/model_trained_final.pt")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience_training:
        print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        break
        
    # Save model every 10 epochs
    if (epoch + 1) % save_every == 0:
        final_model.save(f"{save_to_folder}/model_trained_{epoch+1}.pt")

print("\nPlotting training history of the final trained model...")
final_model.plot_training_history()

`

## Cell 25
`python
import os
import json
import numpy as np
from sklearn.metrics import f1_score
from model.standalone_gnn import StandaloneGNN, StandaloneGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best GAGNN hyperparameters to use for baseline...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_baseline_params = json.load(f)
print(best_baseline_params)

print("\nStarting baseline final training...")


if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
    final_val_loader = [d_dev]

final_model = StandaloneGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=best_baseline_params['hidden_dim'],
    out_dim=best_baseline_params['out_dim'],
    heads=best_baseline_params['heads'],
    num_layers=best_baseline_params['num_layers'],
    gnn_type='gat',
    mlp_hidden_dim=best_baseline_params['mlp_hidden_dim'],
    minibatches=minibatches, num_classes=9,
).to(device)

final_criterion = StandaloneGNNLoss(
    laundry_weight=class_weights, task='multiclass'
)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=best_baseline_params['lr']
)
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(baseline_save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(baseline_epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches  = 0

    evaluated_train = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    for batch in final_train_loader:
        if hasattr(batch, 'e_id'):
            e_id = batch.e_id
            new_mask = ~evaluated_train[e_id]
            evaluated_train[e_id] = True
        else:
            new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        train_mask = new_mask & batch.edge_train_mask
        if not train_mask.any():
            continue
        final_optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                loss = final_criterion(p_trans, batch.y_trans, trans_mask=train_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(final_optimizer)
        torch.nn.utils.clip_grad_norm_(
            final_model.parameters(), best_baseline_params['max_norm']
        )
        scaler.step(final_optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1

    final_model.training_losses.append(epoch_loss / max(n_batches, 1))

    # --- Validation ---
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = final_criterion(p_trans, batch.y_trans, trans_mask=val_mask)
                    val_loss_epoch += loss.item()
                    n_val_batches += 1

    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)

    if epoch % print_every == 0:
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1:4d} | Loss {avg_loss:.4f} | Val Loss {val_loss:.4f}")

    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{baseline_save_to_folder}/baseline_model_final.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= baseline_patience_training:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"\nBest baseline val Loss (final training): {best_final_val_loss:.4f}")
print("Model saved to baseline_model_final.pt")
`
# Code/Multiclass/main_GIN_multiclass.ipynb

## Cell 15
`python
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best hyperparameters for final training...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nStarting final training on Train dataset with validation...")



# Merge patience into train
data.train_node_mask = data.train_node_mask | data.patience_node_mask
data.edge_train_mask = data.edge_train_mask | data.edge_patience_mask

# Use val as new patience
data.patience_node_mask = data.val_node_mask
data.edge_patience_mask = data.edge_val_mask

if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
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
    minibatches=minibatches, num_classes=9,
    gnn_type=gnn_type,
    num_layers=best_params['num_layers'],
    gin_mlp_hidden_dim=best_params['gin_mlp_hidden_dim']
).to(device)

final_criterion = GAGNNLoss(c1=best_params['c1'], c2=best_params['c2'], c3=best_params['c3'], laundry_weight=class_weights, task='multiclass')
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    evaluated_train = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    for batch in final_train_loader:
        if hasattr(batch, 'e_id'):
            e_id = batch.e_id
            new_mask = ~evaluated_train[e_id]
            evaluated_train[e_id] = True
        else:
            new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        train_mask = new_mask & batch.edge_train_mask
        if not train_mask.any():
            continue
        final_optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            p_node, p_trans, p_group, y_group = final_model(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=train_mask
            )
        
        if p_group.numel() > 0:
            loss, _, _, _ = final_criterion(
                p_node, batch.y_node.view(-1, 1),
                p_trans, batch.y_trans,
                p_group, y_group,
                node_mask=batch.node_mask,
                trans_mask=train_mask
            )
            scaler.scale(loss).backward()
            scaler.unscale_(final_optimizer)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=best_params['max_norm'])
            scaler.step(final_optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
            
    train_loss = epoch_loss / max(n_batches, 1)
    final_model.training_losses.append(train_loss)
    
    # Validation step
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=val_mask
                )
                val_loss_epoch += loss.item()
                n_val_batches += 1
                
        
    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)
    
    if (epoch + 1) % print_every == 0:
        print(f"Train Epoch {epoch+1:03d}/{epochs_train} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{save_to_folder}/model_trained_final.pt")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience_training:
        print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        break
        
    # Save model every 10 epochs
    if (epoch + 1) % save_every == 0:
        final_model.save(f"{save_to_folder}/model_trained_{epoch+1}.pt")

print("\nPlotting training history of the final trained model...")
final_model.plot_training_history()







`

## Cell 22
`python
import os
import json
import numpy as np
from sklearn.metrics import f1_score
from model.standalone_gnn import StandaloneGNN, StandaloneGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nLoading best GAGNN hyperparameters to use for baseline...")
with open(f"{save_to_folder}/best_params.json", "r") as f:
    best_baseline_params = json.load(f)
print(best_baseline_params)

print("\nStarting baseline final training...")


if minibatches:
    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < best_params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:best_params['num_layers']]

    final_train_loader = NeighborLoader(
        data, num_neighbors=current_num_neighbors, batch_size=batch_size,
        input_nodes=data.train_node_mask, shuffle=True
    )
    final_patience_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.patience_node_mask, shuffle=False
    )
    final_val_loader = NeighborLoader(
        data, num_neighbors=[-1] * best_params['num_layers'], batch_size=batch_size,
        input_nodes=data.val_node_mask, shuffle=False
    )
else:
    d_dev = data.to(device)
    final_train_loader = [d_dev]
    final_patience_loader = [d_dev]
    final_val_loader = [d_dev]

final_model = StandaloneGNN(
    node_in_dim=x.shape[1],
    edge_feat_dim=edge_attr.shape[1],
    hidden_dim=best_baseline_params['hidden_dim'],
    out_dim=best_baseline_params['out_dim'],
    num_layers=best_baseline_params['num_layers'],
    gnn_type='gin',
    gin_mlp_hidden_dim=best_baseline_params['gin_mlp_hidden_dim'],
    mlp_hidden_dim=best_baseline_params['mlp_hidden_dim'],
    minibatches=minibatches, num_classes=9,
).to(device)

final_criterion = StandaloneGNNLoss(
    laundry_weight=class_weights, task='multiclass'
)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=best_baseline_params['lr']
)
scaler = torch.amp.GradScaler('cuda', enabled=False)

os.makedirs(baseline_save_to_folder, exist_ok=True)

best_final_val_loss = float('inf')
best_final_patience_loss = float('inf')
epochs_no_improve = 0

for epoch in range(baseline_epochs_train):
    final_model.train()
    epoch_loss = 0.0
    n_batches  = 0

    evaluated_train = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    for batch in final_train_loader:
        if hasattr(batch, 'e_id'):
            e_id = batch.e_id
            new_mask = ~evaluated_train[e_id]
            evaluated_train[e_id] = True
        else:
            new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        train_mask = new_mask & batch.edge_train_mask
        if not train_mask.any():
            continue
        final_optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                loss = final_criterion(p_trans, batch.y_trans, trans_mask=train_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(final_optimizer)
        torch.nn.utils.clip_grad_norm_(
            final_model.parameters(), best_baseline_params['max_norm']
        )
        scaler.step(final_optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1

    final_model.training_losses.append(epoch_loss / max(n_batches, 1))

    # --- Validation ---
    final_model.eval()
    patience_loss_epoch = 0.0
    n_patience_batches = 0
    with torch.no_grad():
        for batch in final_patience_loader:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = final_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                )
            if p_group.numel() > 0:
                loss, _, _, _ = final_criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=batch.edge_patience_mask
                )
                patience_loss_epoch += loss.item()
                n_patience_batches += 1
    patience_loss = patience_loss_epoch / max(n_patience_batches, 1)

    val_loss_epoch = 0.0
    n_val_batches = 0
    evaluated_val = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in final_val_loader:
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_val[e_id]
                evaluated_val[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            val_mask = new_mask & batch.edge_val_mask
            if not val_mask.any():
                continue
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    p_trans = final_model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = final_criterion(p_trans, batch.y_trans, trans_mask=val_mask)
                    val_loss_epoch += loss.item()
                    n_val_batches += 1

    val_loss = val_loss_epoch / max(n_val_batches, 1)
    final_model.val_losses.append(val_loss)

    if epoch % print_every == 0:
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1:4d} | Loss {avg_loss:.4f} | Val Loss {val_loss:.4f}")

    if patience_loss < best_final_patience_loss:
        best_final_patience_loss = patience_loss
        best_final_val_loss = val_loss
        epochs_no_improve = 0
        final_model.save(f"{baseline_save_to_folder}/baseline_model_final.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= baseline_patience_training:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"\nBest baseline val Loss (final training): {best_final_val_loss:.4f}")
print("Model saved to baseline_model_final.pt")








`
