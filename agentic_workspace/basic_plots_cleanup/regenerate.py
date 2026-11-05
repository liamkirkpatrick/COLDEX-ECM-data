import concurrent.futures
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
CORES = ['ALHIC2501', 'ALHIC2502', 'ALHIC2201', 'ALHIC2301', 'ALHIC2401', 'ALHIC2302']

def regenerate(core, sections=None):
    os.chdir(ROOT / 'scripts/plotting')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    notebook = json.loads(Path('basic_plots.ipynb').read_text())
    ns = {}
    for cell in notebook['cells'][:12]:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'basic_plots.ipynb', 'exec'), ns)
    if sections is not None:
        ns['meta'] = ns['meta'][ns['meta'].section.isin(sections)]
    expected = ns['meta'].loc[ns['meta'].core == core, 'section'].nunique()
    saved = []
    original_save = Figure.savefig
    def checked_save(fig, fname, **kwargs):
        axes = fig.axes[:-2]
        assert not axes[len(axes)//2].axison
        assert [t.get_text() for t in fig.texts][-2:] == ['AC', 'DC']
        assert len(fig.legends) == 1
        assert all(ax.yaxis_inverted() for ax in axes if ax.axison)
        original_save(fig, fname, **kwargs)
        saved.append(str(Path(fname).resolve()))
        print(f'SAVED {core} {len(saved)}/{expected}: {fname}', flush=True)
    Figure.savefig = checked_save
    try:
        ns['plot_basic'](core)
    finally:
        Figure.savefig = original_save
    assert len(saved) == expected
    return {'core': core, 'count': len(saved), 'files': saved}

if __name__ == '__main__':
    started = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(regenerate, core): core for core in CORES}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            print(f'COMPLETED {results[-1]["core"]}', flush=True)
    report = {'elapsed_seconds': time.time()-started, 'results': results}
    (ROOT/'agentic_workspace/basic_plots_cleanup/regeneration.json').write_text(json.dumps(report, indent=2))
    print(f'ALL COMPLETE: {sum(r["count"] for r in results)} figures', flush=True)
