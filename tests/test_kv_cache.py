from matrixgrad.backend import np
from matrixgrad.engine import no_grad
from matrixgrad.nn import GPT, gpt2_init_


def test_cached_logits_match_full_forward():
    model = GPT(97, 32, 4, 2, 16, 0.0, tie_weights=True)
    gpt2_init_(model)
    ids = np.random.randint(0, 97, (2, 12))
    with no_grad():
        full = model(ids)[0].data
        caches = [{} for _ in model.blocks]
        prefill = model(ids[:, :5], caches=caches)[0].data
        steps = [model(ids[:, i:i + 1], caches=caches)[0].data for i in range(5, 12)]
    assert np.allclose(np.concatenate([prefill, *steps], axis=1), full, atol=1e-5)


def test_stream_refills_cache_past_block_size():
    model = GPT(97, 32, 4, 2, 16, 0.0, tie_weights=True)
    out = model.generate(np.zeros((1, 3), dtype=np.int64), 40, top_k=5)
    assert out.shape == (1, 43)
