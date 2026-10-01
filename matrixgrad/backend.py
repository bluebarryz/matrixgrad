try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False
