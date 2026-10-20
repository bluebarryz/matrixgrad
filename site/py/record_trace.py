"""Writes what the page shows before training starts: a first trace, and a sample from a longer run.

usage: python record_trace.py shakespeare.txt out_dir
"""
import json
import sys
from pathlib import Path

from train import B, T, Trainer

LONG_STEPS = -(-2_000_000 // (B * T))
DEFAULT_TRACE = ('b0.attn', 0, 0)


def main(text_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(Path(text_path).read_text())
    run = {'info': json.loads(trainer.info()), 'trace': json.loads(trainer.train_step(DEFAULT_TRACE))['trace']}
    trainer.buf.tofile(out / 'trace0.bin')
    while trainer.step < LONG_STEPS:
        trainer.train_step()
    run['longRun'] = trainer.sample()
    (out / 'run.json').write_text(json.dumps(run, separators=(',', ':')))


if __name__ == '__main__':
    main(*sys.argv[1:])
