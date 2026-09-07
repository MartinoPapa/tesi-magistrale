import json

for nb_path in ['Code/Binary/main_GIN.ipynb', 'Code/Binary/main_GPSConv.ipynb']:
    print(f'\n\n=== {nb_path} ===')
    nb = json.load(open(nb_path))
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])
        ctype = c['cell_type']
        if ctype == 'markdown' or 'gnn_type' in src or 'GAGNN' in src or 'GPSConv' in src or 'GIN' in src:
            print(f'--- Cell {i} ({ctype}) ---')
            print(src[:2000])
            print()
