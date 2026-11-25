"""locfitByCol (R: edgeR locfitByCol.R)."""

import numpy as np

from . import _C


def locfitByCol(y, x=None, weights=1, span=0.5, degree=0, nthreads=1):
    """edgeR locfitByCol(): fitted values of a locfit of each column of y on x.

    Each column gets fitted(locfit(y[, j] ~ x, weights=weights, alpha=span,
    deg=degree)) with locfit's other defaults, computed in C by edgeR's
    c/locfit_lite.c. As R: weights are recycled to the number of rows, x
    defaults to 1..n, degree must be 0 to 3, y is returned unchanged when
    span*n < 2, n <= 1 or y has no columns, and x, y and weights must be
    finite. The columns are fitted on up to nthreads threads; results do
    not depend on it.
    """
    y = np.array(y, dtype=np.float64, order="F", copy=True)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    ntags = y.shape[0]
    weights = np.resize(np.asarray(weights, dtype=np.float64).reshape(-1), ntags)
    x = np.arange(1, ntags + 1, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
    if np.size(degree) != 1 or degree not in (0, 1, 2, 3):
        raise ValueError("degree must be 0, 1, 2 or 3")
    if span * ntags < 2 or ntags <= 1 or y.shape[1] == 0:
        return y
    if not (np.isfinite(y).all() and np.isfinite(x).all() and np.isfinite(weights).all()):
        raise ValueError("x, y and weights must be finite")
    return _C.locfit_by_col(y, x, weights, float(span), int(degree), int(nthreads))
