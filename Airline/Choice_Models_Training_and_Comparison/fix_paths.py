import json

def fix_notebook(file_path):
    with open(file_path, 'r') as f:
        nb = json.load(f)
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if '\\\\' in line:
                    source[i] = line.replace('\\\\', '/')
    
    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)

fix_notebook('choice model comparison with real data.ipynb')
