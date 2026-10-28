import os

def fix_torch_load(file_path):
    if not os.path.exists(file_path): return
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple replacement if they don't have weights_only
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'torch.load(' in line and 'weights_only' not in line:
            lines[i] = line.replace(')', ', weights_only=False)')
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))

fix_torch_load('training_function_real_data.py')
fix_torch_load('training_function.py')
