import os
import json
import itertools
from model.gagnn import GAGNN
from model.loss import GAGNNLoss
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("Computing class weights for the training set...")
y_train_flat = y_trans[train_edge_mask].view(-1).cpu().numpy()
classes = np.unique(y_train_flat)
class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_flat)

class_weights = torch.ones(9, dtype=torch.float)
for i, c in enumerate(classes):
    class_weights[int(c)] = class_weights_np[i]

class_weights = class_weights.to(device)
print(f"Class weights: {class_weights}")

keys = ['max_norm', 'hidden_dim', 'out_dim', 'heads', 'num_layers',
        'beta', 'mlp_hidden_dim', 'nn_t_hidden_dim', 'lr',
        'c1', 'c2', 'c3']
combinations = list(itertools.product(
    max_norm_clipping, hidden_dims, out_dims, heads_list, num_layers_list,
    betas, mlp_hidden_dims, nn_t_hidden_dims, learning_rates,
    c1_list, c2_list, c3_list
))

print(f"Total hyperparameter combinations: {len(combinations)}\n")
os.makedirs(save_to_folder, exist_ok=True)
with open(f"{save_to_folder}/configurations_architecture.txt", "w") as _f:
    _f.write(f"Total configurations tested: {len(combinations)}\n\n")
    _f.write("Hyperparameter grids:\n")
    _f.write(f"max_norm_clipping: {max_norm_clipping}\n")
    _f.write(f"hidden_dims: {hidden_dims}\n")
    _f.write(f"out_dims: {out_dims}\n")
    _f.write(f"heads_list: {heads_list}\n")
    _f.write(f"num_layers_list: {num_layers_list}\n")
    _f.write(f"betas: {betas}\n")
    _f.write(f"mlp_hidden_dims: {mlp_hidden_dims}\n")
    _f.write(f"nn_t_hidden_dims: {nn_t_hidden_dims}\n")
    _f.write(f"learning_rates: {learning_rates}\n")
    _f.write(f"c1_list: {c1_list}\n")
    _f.write(f"c2_list: {c2_list}\n")
    _f.write(f"c3_list: {c3_list}\n")

from sklearn.metrics import f1_score
import numpy as np
best_val_loss = float('inf')
best_params = None

os.makedirs(save_to_folder, exist_ok=True)

for idx, combo in enumerate(combinations):
    params = dict(zip(keys, combo))

    current_num_neighbors = num_neighbors.copy()
    while len(current_num_neighbors) < params['num_layers']:
        current_num_neighbors.append(current_num_neighbors[-1] if current_num_neighbors else 10)
    current_num_neighbors = current_num_neighbors[:params['num_layers']]
    
    if minibatches:
        train_loader = NeighborLoader(
            data, num_neighbors=current_num_neighbors, batch_size=batch_size,
            input_nodes=data.train_node_mask, shuffle=True
        )
        patience_loader = NeighborLoader(
            data, num_neighbors=[-1] * params['num_layers'], batch_size=batch_size,
            input_nodes=data.patience_node_mask, shuffle=False
        )
        val_loader = NeighborLoader(
            data, num_neighbors=[-1] * params['num_layers'], batch_size=batch_size,
            input_nodes=data.val_node_mask, shuffle=False
        )
    else:
        d_dev = data.to(device)
        train_loader = [d_dev]
        patience_loader = [d_dev]
        val_loader = [d_dev]
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
        minibatches=minibatches, num_classes=9,
        gnn_type=gnn_type,
        num_layers=params['num_layers']
    ).to(device)
    
    criterion = GAGNNLoss(c1=params['c1'], c2=params['c2'], c3=params['c3'], laundry_weight=class_weights, task='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    
    epochs_no_improve = 0
    best_model_val_loss = float('inf')
    best_epoch_patience_loss = float('inf')
    
    for epoch in range(epochs_model_selection):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        evaluated_train = torch.zeros(data.num_edges, dtype=torch.bool, device=device)
        for batch in train_loader:
            batch = batch.to(device)
            if hasattr(batch, 'e_id'):
                e_id = batch.e_id
                new_mask = ~evaluated_train[e_id]
                evaluated_train[e_id] = True
            else:
                new_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
            train_mask = new_mask & batch.edge_train_mask
            if not train_mask.any():
                continue
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_node, p_trans, p_group, y_group = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                    y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=train_mask
                )
            
            if p_group.numel() > 0:
                loss, _, _, _ = criterion(
                    p_node, batch.y_node.view(-1, 1),
                    p_trans, batch.y_trans,
                    p_group, y_group,
                    node_mask=batch.node_mask,
                    trans_mask=train_mask
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1
                
        train_loss = epoch_loss / max(n_batches, 1)
        model.training_losses.append(train_loss)
        
        model.eval()
        patience_loss_epoch = 0.0
        n_patience_batches = 0
        with torch.no_grad():
            for batch in patience_loader:
                batch = batch.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    p_node, p_trans, p_group, y_group = model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                        y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_patience_mask
                    )
                if p_group.numel() > 0:
                    loss, _, _, _ = criterion(
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
            for batch in val_loader:
                    batch = batch.to(device)
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
                        p_node, p_trans, p_group, y_group = model(
                            batch.x, batch.edge_index, batch.edge_attr, batch.edge_index.size(1),
                            y_node=batch.y_node, y_trans=batch.y_trans, trans_mask=batch.edge_val_mask
                        )
                    if p_group.numel() > 0:
                        loss, _, _, _ = criterion(
                            p_node, batch.y_node.view(-1, 1),
                            p_trans, batch.y_trans,
                            p_group, y_group,
                            node_mask=batch.node_mask,
                            trans_mask=val_mask
                        )
                        val_loss_epoch += loss.item()
                        n_val_batches += 1
                    

            val_loss = val_loss_epoch / max(n_val_batches, 1)
            model.val_losses.append(val_loss)
            if patience_loss < best_epoch_patience_loss:
                best_epoch_patience_loss = patience_loss
                best_model_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience_selection:
                print(f"Patience triggered at epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                break
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")   
        
            
        print(f"Best Val Loss for this config: {best_model_val_loss:.4f}")
        if best_model_val_loss < best_val_loss:
            best_val_loss = best_model_val_loss
            best_params = params
            model.save(f"{save_to_folder}/gagnn_best_model_selection.pt")
            print(">>> New Best Parameters! Model saved.")
        else:
            print(f"No improvement over best parameters. Current: {best_model_val_loss:.4f} | Best: {best_val_loss:.4f}")

    print("\n=========================================")
    print(f"Overall Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Parameters: {best_params}")
    print("=========================================")
    with open(f"{save_to_folder}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\nPlotting training history of the best model from model selection...")
    best_selection_model = GAGNN.load_saved(f"{save_to_folder}/gagnn_best_model_selection.pt")
    best_selection_model.plot_training_history()