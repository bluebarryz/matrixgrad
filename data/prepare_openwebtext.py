import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

TARGET_TRAIN_TOKENS = 200_000_000
TARGET_VAL_TOKENS   = 1_000_000
OUT_DIR = os.path.join(os.path.dirname(__file__), 'openwebtext')
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token

ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
ds_iter = iter(ds)


def collect(target, label):
    chunks = []
    total = 0
    pbar = tqdm(total=target, desc=f'tokenizing {label}', unit='tok')
    for item in ds_iter:
        ids = enc.encode_ordinary(item['text'])
        ids.append(EOT)
        chunks.append(ids)
        total += len(ids)
        pbar.update(len(ids))
        if total >= target:
            break
    pbar.close()
    out = np.empty(total, dtype=np.uint16)
    i = 0
    for c in chunks:
        out[i:i+len(c)] = c
        i += len(c)
    return out


# val is read first so the two splits never overlap
val_arr = collect(TARGET_VAL_TOKENS, 'val')
val_path = os.path.join(OUT_DIR, 'val.bin')
val_arr.tofile(val_path)
print(f'wrote {val_path}: {len(val_arr):,} tokens ({val_arr.nbytes / 1e6:.1f} MB)')

train_arr = collect(TARGET_TRAIN_TOKENS, 'train')
train_path = os.path.join(OUT_DIR, 'train.bin')
train_arr.tofile(train_path)
print(f'wrote {train_path}: {len(train_arr):,} tokens ({train_arr.nbytes / 1e6:.1f} MB)')
