"""The three pipelines and estimateDisp on small data, against R's results (see _check.py)."""

import math

import numpy as np
import pandas as pd
import pytest

import edgerpy as ep
from _check import check_all, counts, design, factor, load, same


@pytest.fixture(scope="module")
def nb1():
    d = load("NB1")
    y = ep.normLibSizes(ep.DGEList(counts(d)))
    X = design(d)
    fit = ep.glmQLFit(y, X)
    res = ep.glmQLFTest(fit)
    fitR = ep.glmQLFit(y, X, robust=True)
    return d, {"d": y, "fit": fit, "res": res, "top": ep.topTags(res, n=np.inf), "fitR": fitR,
               "resR": ep.glmQLFTest(fitR)}


def test_nb_ql(nb1):
    d, objects = nb1
    check_all(d, objects)


def test_nb_ql_group_filter_default_design():
    d = load("NB3")
    y0 = ep.DGEList(counts(d), group=factor(d, "group"))
    keep = ep.filterByExpr(y0)
    y = ep.normLibSizes(y0.subset(keep, None, keep_lib_sizes=False))
    assert int(np.sum(keep)) == int(d["out/d/@nrow"])
    fit = ep.glmQLFit(y)
    check_all(d, {"d": y, "fit": fit, "res": ep.glmQLFTest(fit), "res2": ep.glmQLFTest(fit, coef=1)})


def test_binomial_ql():
    d = load("BIN1")
    y = ep.PCList(counts(d), counts(d, "in/counts2"), group=factor(d, "group"))
    fit = ep.binQLFit(y)
    res = ep.binQLFTest(fit, coef=1)
    check_all(d, {"fit": fit, "res": res, "top": ep.topTags(res, n=np.inf)})


def test_binomial_ql_design():
    d = load("BIN3")
    y = ep.PCList(counts(d), counts(d, "in/counts2"), group=factor(d, "group"))
    X = design(d)
    fitQ = ep.binQLFit(y, X)
    check_all(d, {"fit0": ep.binFit(y, X, prior_count=0), "fitQ": fitQ,
                  "resF": ep.binQLFTest(fitQ, coef="groupC"),
                  "resU": ep.binQLFTest(fitQ, coef="groupC", lor=math.log2(1.5))})


@pytest.fixture(scope="module")
def ds1():
    d = load("DS1")
    y = ep.DGEList(counts(d), lib_size=np.full(d["in/counts"].shape[1], 1e6),
                   genes=pd.DataFrame({"GeneID": d["in/GeneID"]}))
    fit = ep.glmQLFit(y, design(d))
    ds = ep.diffSplice(fit, geneid="GeneID", verbose=False)
    return d, ds


def test_diffsplice(ds1):
    d, ds = ds1
    check_all(d, {"ds": ds, "topF": ep.topSplice(ds, test="F", number=np.inf),
                  "topS": ep.topSplice(ds, test="simes", number=np.inf),
                  "topT": ep.topSplice(ds, test="t", number=np.inf)})


def test_estimate_disp():
    d = load("E1")
    y = ep.DGEList(counts(d), group=factor(d, "group"), lib_size=d["in/lib_size"], norm_factors=d["in/norm_factors"])
    check_all(d, {"edisp": ep.estimateDisp(y, design(d))})


def test_cpm_and_ave_log_cpm(nb1):
    d, objects = nb1
    y = objects["d"]
    eff = y["samples"]["lib.size"].to_numpy() * y["samples"]["norm.factors"].to_numpy()
    same(ep.cpm(y).to_numpy().sum(axis=0), 1e6 * y["counts"].to_numpy().sum(axis=0) / eff, "cpm column sums")
    same(ep.aveLogCPM(y), d["out/fit/AveLogCPM"], "aveLogCPM")
    assert ep.cpm(y, log=True).shape == y["counts"].shape


def test_decide_tests(nb1):
    _, objects = nb1
    res = objects["res"]
    calls = ep.decideTests(res).to_numpy().ravel().astype(int)
    tab = ep.topTags(res, n=np.inf)["table"].reindex(res["table"].index)
    expected = np.where(tab["FDR"].to_numpy() < 0.05, np.sign(tab["logFC"].to_numpy()), 0).astype(int)
    assert np.array_equal(calls, expected)
