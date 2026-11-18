"""Run existing notebook code in order without changing its source or outputs."""
import json
import os
from pathlib import Path
import sys

import matplotlib
from IPython.display import display

matplotlib.use('Agg')

notebook = Path(sys.argv[1]).resolve()
os.chdir(notebook.parent)
namespace = {'__name__': '__main__', 'display': display}
selected = {int(index) for index in sys.argv[2:]} if len(sys.argv) > 2 else None
for index, cell in enumerate(json.loads(notebook.read_text())['cells']):
    if selected is not None and index not in selected:
        continue
    source = ''.join(cell['source'])
    if cell['cell_type'] != 'code' or not source.strip():
        continue
    print(f'RUN {notebook.name} cell {index}', flush=True)
    exec(compile(source, f'{notebook}:cell{index}', 'exec'), namespace)
    print(f'DONE cell {index}', flush=True)
print(f'COMPLETE {notebook}', flush=True)
