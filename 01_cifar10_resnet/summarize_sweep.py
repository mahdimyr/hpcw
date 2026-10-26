"""Compare the runs of a sweep:  python summarize_sweep.py <sweep directory>

Reads every <sweep directory>/*/results.json written by train.py (standard
library only, so it runs anywhere, even without the workshop environment).
"""
import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
runs = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*/results.json")]
if not runs:
    sys.exit(f"No results.json files under {sys.argv[1]} yet -- are the jobs finished? (squeue --me)")

print(f"{'run':24s} {'model':9s} {'lr':>6s} {'batch':>6s} {'epochs':>6s} {'test acc':>9s} {'time':>7s}")
for r in sorted(runs, key=lambda r: -r["test_acc"]):
    print(f"{Path(r['out_dir']).name:24s} {r['model']:9s} {r['lr']:6g} {r['batch_size']:6d} {r['epochs']:6d} "
          f"{r['test_acc']:9.2%} {r['train_seconds']:6.0f}s")
