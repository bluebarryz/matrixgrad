import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = os.path.join(os.path.dirname(__file__), 'wikitext')
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token

ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')


def encode_split(split):
    chunks = []
    total = 0
    for item in tqdm(ds[split], desc=f'tokenizing {split}'):
        text = item['text']
        if not text.strip():
            continue
        ids = enc.encode_ordinary(text)
        ids.append(EOT)
        chunks.append(ids)
        total += len(ids)
    out = np.empty(total, dtype=np.uint16)
    i = 0
    for c in chunks:
        out[i:i+len(c)] = c
        i += len(c)
    return out


for hf_split, file_name in (('train', 'train.bin'), ('validation', 'val.bin')):
    arr = encode_split(hf_split)
    path = os.path.join(OUT_DIR, file_name)
    arr.tofile(path)
    print(f'wrote {path}: {len(arr):,} tokens ({arr.nbytes / 1e6:.1f} MB)')
