import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
root = Path(__file__).resolve().parents[2]
os.chdir(root / 'scripts/plotting')
notebook = json.loads(Path('basic_plots.ipynb').read_text())
ns = {}
for cell in notebook['cells'][:12]:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), 'basic_plots.ipynb', 'exec'), ns)
all_meta = ns['meta']
original_save = Figure.savefig
count = 0
def checked_save(fig, *args, **kwargs):
    global count
    axes = fig.axes[:-2]
    assert not axes[len(axes)//2].axison
    assert [t.get_text() for t in fig.texts][-2:] == ['AC', 'DC']
    assert len(fig.legends) == 1
    for ax in axes:
        if ax.axison:
            assert ax.yaxis_inverted()
    original_save(fig, *args, **kwargs)
    count += 1
Figure.savefig = checked_save
for core, section in [('ALHIC2502', '15'), ('ALHIC2501', '1'), ('ALHIC2201', '10_1'), ('ALHIC2301', '1'), ('ALHIC2401', '1'), ('ALHIC2302', '10')]:
    ns['meta'] = all_meta[(all_meta.core == core) & (all_meta.section == section)]
    assert len(ns['meta'])
    ns['plot_basic'](core)
print(f'Validated {count} figures across all four layouts and six notebook cores.')
