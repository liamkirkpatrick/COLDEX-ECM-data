import json
from pathlib import Path
import pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'agentic_workspace/basic_plots_cleanup'
cores = ['ALHIC2501','ALHIC2502','ALHIC2201','ALHIC2301','ALHIC2401','ALHIC2302']
meta = pd.read_csv(ROOT/'data/metadata.csv')
saved = set()
for name in ('regeneration.log', 'top_face.log'):
    for line in (WORK/name).read_text().splitlines():
        if line.startswith('SAVED '):
            saved.add((ROOT/'scripts/plotting'/line.split(': ',1)[1]).resolve())
report = {}
for core in cores:
    sections = meta.loc[meta.core == core, 'section'].unique()
    expected = {ROOT/f'figures/basic_plots/{core}/{core}-{section}.png' for section in sections}
    missing = expected - saved
    assert not missing, f'Not regenerated: {missing}'
    for path in expected:
        with Image.open(path) as image:
            image.verify()
    report[core] = len(expected)
(WORK/'output_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(report)
print(f'Verified {sum(report.values())} newly saved PNG files.')
