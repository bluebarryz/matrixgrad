import pytest

try:
    import cupy as cp  # noqa: F401
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False

if not _HAS_GPU:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _seed():
    import cupy as cp
    cp.random.seed(0)
