import functools
import sys
import time

import numpy as np
from sklearn.metrics import pairwise_distances  # type: ignore

from dowker_complex import DowkerComplex  # type: ignore
from timer import Timer  # type: ignore


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


n_vertices, n_witnesses, dim = list(map(int, sys.argv[1:]))
n = n_vertices + n_witnesses
X = np.random.randn(n, dim)
V, W = X[:n_vertices], X[n_vertices:]

dm = pairwise_distances(V)
dc = DowkerComplex(
    swap=True,
    verbose=False,
)

print(f"{dm.shape = }")

with Timer():
    print("Started fitting.")
    dc.fit(
        [V, W],
    )

with Timer():
    print("Started transforming.")
    dc.transform(
        [V, W],
    )
