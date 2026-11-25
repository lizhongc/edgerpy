"""The public API, input errors, and that every plot runs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import edgerpy as ep
from _check import counts, design, load


def test_public_api():
    assert ep.__version__
    assert len(ep.__all__) == len(set(ep.__all__))
    for name in ep.__all__:
        assert callable(getattr(ep, name)), name


@pytest.fixture(scope="module")
def nb():
    d = load("NB1")
    y = ep.normLibSizes(ep.DGEList(counts(d)))
    return y, ep.glmQLFit(y, design(d))


@pytest.mark.parametrize("coef", [-1, 2, 5, [1, -1]])
def test_coef_out_of_range(nb, coef):
    _, fit = nb
    with pytest.raises(IndexError, match="out of range"):
        ep.glmQLFTest(fit, coef=coef)


def test_coef_by_name(nb):
    _, fit = nb
    name = fit["design"].columns[1]
    a = ep.glmQLFTest(fit, coef=1)["table"]
    b = ep.glmQLFTest(fit, coef=name)["table"]
    assert a.equals(b)


def test_subset(nb):
    y, _ = nb
    sub = y[:10, [0, 2]]
    assert sub["counts"].shape == (10, 2)
    assert np.array_equal(sub["counts"].to_numpy(), y["counts"].to_numpy()[:10][:, [0, 2]])


def test_plots_run(nb):
    y, fit = nb
    d = load("DS1")
    ys = ep.DGEList(counts(d), lib_size=np.full(d["in/counts"].shape[1], 1e6),
                    genes=pd.DataFrame({"GeneID": d["in/GeneID"]}))
    ds = ep.diffSplice(ep.glmQLFit(ys, design(d)), geneid="GeneID", verbose=False)
    calls = [lambda: ep.plotMDS(y), lambda: ep.plotMD(ep.glmQLFTest(fit)), lambda: ep.plotQLDisp(fit),
             lambda: ep.plotBCV(ep.estimateDisp(y, fit["design"])), lambda: ep.plotSplice(ds)]
    for call in calls:
        call()
        plt.close("all")
