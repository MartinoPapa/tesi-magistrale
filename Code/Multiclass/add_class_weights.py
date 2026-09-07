import json
import os
import glob

notebooks = glob.glob(r'c:\Users\marti\OneDrive\Desktop\Tesi v2\Code\Multiclass\*.ipynb')

class_weights_code = """
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
"""

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # 1. Insert class weights computation
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            source_str = "".join(source)
            if 'combinations = list(itertools.product(' in source_str:
                # Insert the class weights code before this block if not already there
                if 'compute_class_weight' not in source_str:
                    lines = class_weights_code.strip().split('\n')
                    insert_idx = 0
                    for i, line in enumerate(source):
                        if 'keys =' in line:
                            insert_idx = i
                            break
                    for line in reversed(lines):
                        source.insert(insert_idx, line + "\n")
                    source.insert(insert_idx + len(lines), "\n")
                    modified = True
            
            # 2. Modify GAGNNLoss instantiations
            for i, line in enumerate(source):
                if 'criterion = GAGNNLoss(' in line and 'laundry_weight=params[' in line:
                    source[i] = line.replace("laundry_weight=params['laundry_weight']", "laundry_weight=class_weights")
                    modified = True
                if 'final_criterion = GAGNNLoss(' in line and "laundry_weight=best_params.get('laundry_weight', 2.0)" in line:
                    source[i] = line.replace("laundry_weight=best_params.get('laundry_weight', 2.0)", "laundry_weight=class_weights")
                    modified = True
                if 'final_criterion = GAGNNLoss(' in line and "laundry_weight=best_params['laundry_weight']" in line:
                    source[i] = line.replace("laundry_weight=best_params['laundry_weight']", "laundry_weight=class_weights")
                    modified = True

    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            f.write('\n')
        print(f"Fixed {nb_path}")
