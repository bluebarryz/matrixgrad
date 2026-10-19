"""Writes the Recorded Trace: a Train run the page replays until the live engine takes over.

usage: python record_trace.py shakespeare.txt out_dir
"""
import json
import sys
from pathlib import Path

import numpy as np

from train import B, T, Trainer

STEPS = -(-500_000 // (B * T))        # the live run auto-pauses at 0.5M tokens
LONG_STEPS = -(-2_000_000 // (B * T))  # the "from a longer run" sample
HANDOFFS = (50, *range(100, STEPS, 100), STEPS)
SNAPSHOT_EVERY = SAMPLE_EVERY = 100
DEFAULT_TRACE = ('b0.attn', 0, 0)


def main(text_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(Path(text_path).read_text())
    run = {'info': json.loads(trainer.info()), 'steps': [], 'samples': {}, 'snapshots': [], 'handoffs': HANDOFFS}
    snapshots = []

    for s in range(LONG_STEPS):
        weights = trainer.save(np.float16, optimizer=False)
        step = json.loads(trainer.train_step(DEFAULT_TRACE if s == 0 else None))
        if s == 0:
            run['trace'] = step['trace']
            trainer.buf.tofile(out / 'trace0.bin')
        if s % SNAPSHOT_EVERY == 0 and s < STEPS:
            run['snapshots'].append({'step': s, 'ix': step['ix']})
            snapshots.append(weights)
        if trainer.step <= STEPS:
            run['components'] = list(step['norms'])
            run['steps'].append([round(step['loss'], 4), *(round(v, 5) for v in step['norms'].values())])
        if trainer.step in HANDOFFS:
            trainer.save().tofile(out / f'state{trainer.step}.bin')
        if (trainer.step % SAMPLE_EVERY == 0 and trainer.step <= STEPS) or trainer.step == LONG_STEPS:
            run['samples'][trainer.step] = trainer.sample()

    np.concatenate(snapshots).tofile(out / 'snapshots.bin')
    (out / 'run.json').write_text(json.dumps(run, separators=(',', ':')))


if __name__ == '__main__':
    main(*sys.argv[1:])
