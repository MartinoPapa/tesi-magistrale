import json
import re
import os

files = [
    'Code/Binary/main_GAT.ipynb', 
    'Code/Binary/main_GIN.ipynb', 
    'Code/Multiclass/main_GAT_multiclass.ipynb', 
    'Code/Multiclass/main_GIN_multiclass.ipynb'
]

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update patience_training values
            source = re.sub(r'patience_training\s*=\s*\d+', 'patience_training = 150', source)
            source = re.sub(r'baseline_patience_training\s*=\s*\d+', 'baseline_patience_training = 150', source)

            # 2. Refactor final training cells
            if 'Starting final training' in source or 'Starting baseline final training' in source:
                # Replace data in-place modification with data_final
                source = source.replace('data.train_node_mask = data.train_node_mask | data.patience_node_mask', 
                                        'data_final = data.clone()\ndata_final.train_node_mask = data_final.train_node_mask | data_final.patience_node_mask')
                source = source.replace('data.edge_train_mask = data.edge_train_mask | data.edge_patience_mask',
                                        'data_final.edge_train_mask = data_final.edge_train_mask | data_final.edge_patience_mask')
                source = source.replace('data.patience_node_mask = data.val_node_mask',
                                        'data_final.patience_node_mask = data_final.val_node_mask')
                source = source.replace('data.edge_patience_mask = data.edge_val_mask',
                                        'data_final.edge_patience_mask = data_final.edge_val_mask')

                # Replace 'data' with 'data_final' in loader creation
                source = source.replace('NeighborLoader(\n        data,', 'NeighborLoader(\n        data_final,')
                source = source.replace('input_nodes=data.', 'input_nodes=data_final.')
                source = source.replace('d_dev = data.to(device)', 'd_dev = data_final.to(device)')

                # Now, for the final_val_loader creation, remove it
                val_loader_pattern = re.compile(r'final_val_loader\s*=\s*NeighborLoader\([^)]+\)', re.DOTALL)
                source = val_loader_pattern.sub('', source)
                source = source.replace('final_val_loader = [d_dev]', '')

                # For baseline, the merging code is MISSING! We need to ADD IT before creating the loaders.
                # If "Starting baseline final training" is in source and "data_final = data.clone()" is not, we inject it.
                if 'Starting baseline final training' in source and 'data_final = data.clone()' not in source:
                    injection = """data_final = data.clone()
data_final.train_node_mask = data_final.train_node_mask | data_final.patience_node_mask
data_final.edge_train_mask = data_final.edge_train_mask | data_final.edge_patience_mask
data_final.patience_node_mask = data_final.val_node_mask
data_final.edge_patience_mask = data_final.edge_val_mask

if minibatches:"""
                    source = source.replace('if minibatches:', injection, 1)

                # Remove the validation loop. It starts with 'val_loss_epoch = 0.0' and ends before 'if (epoch + 1) % print_every == 0:'
                # We can use a regex to match from 'val_loss_epoch = 0.0' to 'final_model.val_losses.append(val_loss)'
                val_loop_pattern = re.compile(r'val_loss_epoch\s*=\s*0\.0.*?final_model\.val_losses\.append\(val_loss\)\n?', re.DOTALL)
                source = val_loop_pattern.sub('', source)
                
                # Replace 'Val Loss' prints with 'Patience Loss' prints
                source = source.replace('Val Loss: {val_loss:.4f}', 'Patience Loss: {patience_loss:.4f}')
                source = source.replace('Val Loss {val_loss:.4f}', 'Patience Loss {patience_loss:.4f}')
                
                # Remove best_final_val_loss logic
                source = source.replace("best_final_val_loss = float('inf')\n", '')
                source = source.replace("best_final_val_loss = val_loss\n", '')
                source = source.replace('print(f"\\nBest baseline val Loss (final training): {best_final_val_loss:.4f}")', 'print(f"\\nBest baseline patience Loss (final training): {best_final_patience_loss:.4f}")')

            # Fix up lines to be list of strings
            lines = []
            for line in source.split('\n'):
                lines.append(line + '\n')
            if len(lines) > 0:
                lines[-1] = lines[-1][:-1] # remove last newline to match format
            
            cell['source'] = lines

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')

print('Done.')
