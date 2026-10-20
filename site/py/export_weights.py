"""Strips a pretrain.ipynb checkpoint down to fp16 weights for the Generate model.

usage: python export_weights.py checkpoints/best.pkl out_dir
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def main(ckpt_path, out_dir):
    ckpt = pickle.loads(Path(ckpt_path).read_bytes())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = np.concatenate([p.ravel() for p in ckpt['params']]).astype(np.float16)
    weights.tofile(out / 'gpt.bin')
    meta = {'config': ckpt['config'], 'bytes': weights.nbytes, 'step': ckpt['step'], 'val_loss': float(ckpt['val_loss'])}
    (out / 'gpt.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    main(*sys.argv[1:])
