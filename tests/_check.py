"""Helpers of the release tests: load a scenario's data and compare edgerpy's results with R's.

data/<scenario>.npz holds the inputs (keys in/...) and R's results (keys out/<object>/<path>), from
edgeR 4.99.16, limma 3.99.1 and R 4.6.1 with the reference BLAS/LAPACK. Values are compared with a
relative tolerance of 1e-10, so the tests pass on every platform; set EDGERPY_TEST_BITWISE=1 to require
identical bits (expected on Linux x86_64 with a reference build).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
BITWISE = os.environ.get("EDGERPY_TEST_BITWISE") == "1"
RTOL = 1e-10


def load(name):
    with np.load(DATA / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def factor(d, name):
    return pd.Categorical.from_codes(d[f"in/{name}_codes"], categories=list(d[f"in/{name}_levels"]))


def design(d):
    return pd.DataFrame(d["in/design"], columns=list(d["in/design_colnames"]))


def counts(d, key="in/counts"):
    return d[key].astype(np.float64)


def get(obj, path):
    """A component of an edgerpy result by an R path ('table/PValue', 'table/@rownames')."""
    for p in path.split("/"):
        if p == "@rownames":
            return np.asarray([str(x) for x in obj.index])
        obj = obj[p]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_numpy()
    return np.asarray(obj)


def same(got, exp, what):
    got, exp = np.asarray(got), np.asarray(exp)
    if exp.ndim <= 1 and got.size == exp.size:
        got, exp = got.ravel(), exp.ravel()
    assert got.shape == exp.shape, f"{what}: shape {got.shape} vs R {exp.shape}"
    if exp.dtype.kind in "US":
        assert np.array_equal(got.astype(str), exp), f"{what}: differs from R"
        return
    got = got.astype(np.float64)
    if BITWISE:
        assert np.array_equal(got, exp, equal_nan=True), f"{what}: not bitwise equal to R"
        return
    finite = exp[np.isfinite(exp)]
    atol = RTOL * float(np.max(np.abs(finite))) if finite.size else 0.0
    np.testing.assert_allclose(got, exp, rtol=RTOL, atol=atol, equal_nan=True, err_msg=what)


def check_all(d, objects):
    """Compare every R result in d with the edgerpy object of the same name."""
    keys = [k for k in d if k.startswith("out/") and "@nrow" not in k]
    assert keys
    for k in keys:
        _, name, path = k.split("/", 2)
        same(get(objects[name], path), d[k], k)
