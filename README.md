# matrixgrad

An autograd engine written from scratch in NumPy/CuPy, used to pretrain a
small GPT on real web text. No PyTorch or JAX.

The repo holds two packages:

- **matrixgrad**: [micrograd](https://github.com/karpathy/micrograd) extended
  from scalars to matrices, plus everything needed to train a GPT the way
  [nanoGPT](https://github.com/karpathy/nanoGPT) does. Runs on CuPy if it's
  installed and falls back to NumPy on CPU.
- **cudagrad**: the same engine with the hot ops replaced by hand-written CUDA
  kernels: SGEMM, LayerNorm, fused cross-entropy, embedding gather/scatter and
  fused AdamW.

## Layout

```
matrixgrad/
  engine.py      Matrix (the autograd tensor), no_grad
  nn.py          Linear, LayerNorm, CausalSelfAttention, GPT, ...
  optimizer.py   SGD, Adam, AdamW, clip_grad_norm_
  backend.py     picks CuPy or NumPy
cudagrad/
  engine.py, nn.py, optimizer.py   same API as matrixgrad, ops go through K
  backend.py     K: sends each op to a custom kernel or to CuPy
  kernels/       .cu sources and the CuPy RawModule launchers
data/            dataset tokenization scripts
tests/           kernel tests against CuPy, plus finite-difference grad checks
bench/           SGEMM and softmax benchmarks
pretrain.ipynb   GPT pretraining with matrixgrad
```

## Setup

```bash
pip install -e .
pip install cupy-cuda12x   # optional, for GPU
```

## Pretraining a GPT

Tokenize a dataset first:

```bash
python data/prepare_openwebtext.py   # ~200M tokens, ~400 MB
python data/prepare_wikitext.py      # ~117M tokens, ~235 MB, much faster to download
```

Then run [pretrain.ipynb](pretrain.ipynb) with `DATASET` set to the one you
prepared.

The model has 4 layers, 4 heads, 256-dim embeddings, a context of 256 and the
GPT-2 vocabulary. With tied embeddings that comes to about 16M parameters.
Training uses AdamW with weight decay on matrices only, warmup followed by
cosine decay, gradient clipping and GPT-2 style init. The initial loss lands
within 0.01 of `log(50257) ≈ 10.82`, which is what a correct forward pass
should give.

A sample after training on OpenWebText, prompted with *"Computer science would
not be what it is today without"*:

> what we do?
>
> A recent article: "That is not a good deal, if you're going to take on the
> rest of the business that you're going to be going," he said.

That's about what to expect from 16M parameters: grammatical English that
doesn't mean much.

The bottleneck is the Python overhead of every op in the autograd engine, not
the GPU. Making the model bigger barely changes tokens/sec.

## cudagrad

Each kernel has a flag on `K`, and every flag is off by default. With a flag
off, the op runs through CuPy. That gives you a reference to check the kernel
against.

```python
from cudagrad.backend import K

K.USE_CUSTOM_MATMUL = True         # tiled SGEMM
K.USE_CUSTOM_ELEMENTWISE = True    # exp, log, relu, pow
K.USE_CUSTOM_LAYERNORM = True      # LayerNorm forward + backward
K.USE_CUSTOM_CROSS_ENTROPY = True  # loss and grad in one kernel
K.USE_CUSTOM_EMBEDDING = True      # gather + atomicAdd scatter
K.USE_CUSTOM_ADAMW = True          # one kernel per parameter tensor
```

| Kernel | File |
|---|---|
| Naive and shared-memory tiled SGEMM (128x128 block tile, 8x8 per thread) | [sgemm.cu](cudagrad/kernels/sgemm.cu) |
| LayerNorm forward and dx backward | [layernorm.cu](cudagrad/kernels/layernorm.cu) |
| Fused softmax + cross-entropy forward/backward | [cross_entropy.cu](cudagrad/kernels/cross_entropy.cu) |
| Embedding gather and atomic scatter-add | [embedding.cu](cudagrad/kernels/embedding.cu) |
| Fused AdamW | [adamw.cu](cudagrad/kernels/adamw.cu) |
| Elementwise unary and binary ops | [elementwise.cu](cudagrad/kernels/elementwise.cu) |
| Row softmax forward/backward | [softmax.cu](cudagrad/kernels/softmax.cu) |
| Row sum / mean / max | [reduce.cu](cudagrad/kernels/reduce.cu) |

The engine doesn't use the softmax and reduce kernels yet. They are tested and
benchmarked on their own.

Everything is fp32. There are no Tensor Cores, and the batched matmul in
attention loops over the batch in Python.

### Tests and benchmarks

These need an NVIDIA GPU and CuPy. Without a GPU, the tests are skipped.

```bash
pytest tests/
python bench/bench_sgemm.py
python bench/bench_softmax.py
```

## References

- Karpathy, [micrograd](https://github.com/karpathy/micrograd) and [nanoGPT](https://github.com/karpathy/nanoGPT)
- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2017)
