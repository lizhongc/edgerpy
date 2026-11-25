"""limma squeezeVar / fitFDistUnequalDF1 and statmod logmdigamma.

R: limma/R/squeezeVar.R, limma/R/fitFDistUnequalDF1.R (limma 3.99.1),
statmod logmdigamma (1.5.2). edgerpy always uses the fitFDistUnequalDF1
hyperparameter method (plan decision 7); there is no legacy branch.
"""

import numpy as np

from . import _r
from .helpers import chooseLowessSpan, loessFit


def logmdigamma(x):
    """statmod logmdigamma(x) = log(x) - digamma(x), as statmod computes it."""
    z = np.atleast_1d(np.asarray(x, dtype=np.float64)).copy()
    scalar = np.ndim(x) == 0
    ps = np.full(z.shape, np.nan)
    omit = np.isnan(z) | (z <= 0)
    ok = ~omit
    if ok.any():
        ps[ok] = _logmdigamma_pos(z[ok])
    return float(ps[0]) if scalar else ps


def _logmdigamma_pos(z):
    small = np.abs(z) < 5
    ps = np.empty_like(z)
    if small.any():
        x = z[small]
        ps[small] = (((((_r.log(x / (x + 5)) + _logmdigamma_pos(x + 5)) + 1 / x) + 1 / (x + 1)) + 1 / (x + 2))
                     + 1 / (x + 3)) + 1 / (x + 4)
    big = ~small
    if big.any():
        zz = z[big]
        x = 1 / (zz * zz)
        t = -1 / 12 + (3617 * x) / 8160
        t = 691 / 32760 + x * t
        t = -1 / 132 + x * t
        t = 1 / 240 + x * t
        t = -1 / 252 + x * t
        t = 1 / 120 + x * t
        t = -1 / 12 + x * t
        tail = x * t
        ps[big] = 1 / (2 * zz) - tail
    return ps


def _unit(v, i):
    """limma .subsetUnitVector(x, i)."""
    v = np.asarray(v)
    return v if v.size == 1 else v[i]


def fitFDistUnequalDF1(x, df1, covariate=None, span=None, robust=False, prior_weights=None):
    """limma fitFDistUnequalDF1(): dict(scale, df2[, df2.outlier, df2.shrunk])."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    df1 = np.atleast_1d(np.asarray(df1, dtype=np.float64)).copy()
    if df1.size not in (1, n):
        raise ValueError("x and df1 are different lengths")
    if np.isnan(df1).any():
        raise ValueError("NA df1 values")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if covariate.size != n:
            raise ValueError("x and covariate are different lengths")
        if np.isnan(covariate).any():
            raise ValueError("prior.weights contain NA values")
    if prior_weights is not None:
        prior_weights = np.asarray(prior_weights, dtype=np.float64).copy()
        if prior_weights.size != n:
            raise ValueError("x and covariate are different lengths")
        m = np.min(prior_weights)
        if np.isnan(m):
            raise ValueError("prior.weights contain NA values")
        if m < 0:
            raise ValueError("prior.weights are negative")

    if np.isnan(x).any():
        i = np.isnan(x)
        if prior_weights is None:
            prior_weights = (~i).astype(np.float64)
        else:
            prior_weights[i] = 0
        x[i] = 0
    if np.min(df1) < 0.01:
        i = df1 < 0.01
        if prior_weights is None:
            prior_weights = (~i).astype(np.float64) * np.ones(n)
        else:
            prior_weights[np.broadcast_to(i, (n,))] = 0
        df1[i] = 1
    PriorWeights = prior_weights is not None

    i = x > 0
    if PriorWeights:
        i[prior_weights == 0] = False
    n_informative = int(i.sum())
    if n_informative < 2:
        return {"scale": np.nan, "df2": np.nan}
    if n_informative == 2:
        covariate = None
        robust = False
        prior_weights = None

    m = _r.median(x[i])
    xpos = np.maximum(x, 1e-12 * m)
    z = _r.log(xpos)
    d1 = df1 / 2
    e = z + logmdigamma(d1)
    w = 1 / _r.trigamma(d1)
    if w.size < n:
        w = np.resize(w, n)
    if PriorWeights:
        w = w * prior_weights

    if covariate is None:
        emean = _r.rsum(w * e) / _r.rsum(w)
    else:
        if span is None:
            span = chooseLowessSpan(n, small_n=500)
        emean = loessFit(e, covariate, weights=w / _r.quantile7(w, 0.75), min_weight=1e-8, max_weight=100,
                         span=span, iterations=1)["fitted"]

    d1x = d1 * xpos
    pw = prior_weights if PriorWeights else None

    def minusTwiceLogLik(par):
        d2 = par / (1 - par)
        d2s20 = d2 * _r.exp(emean - logmdigamma(d2))
        terms = (-(d1 + d2) * _r.log1p(d1x / d2s20) - d1 * _r.log(d2s20)) + _r.lgamma(d1 + d2) - _r.lgamma(d2)
        if pw is not None:
            terms = pw * terms
        return -2 * _r.rsum(terms)

    par = _r.optimize(minusTwiceLogLik, 1 / 2, 0.9998)
    d2 = par / (1 - par)
    s20 = _r.exp(emean - logmdigamma(d2))
    if not robust:
        return {"scale": s20, "df2": 2 * d2}

    df2 = 2 * d2
    FStat = x / s20
    RightP = _r.pf(FStat, df1, df2, lower_tail=False)
    LeftP = 1 - RightP
    if np.min(LeftP) < 0.001:
        i = LeftP < 0.001
        LeftP[i] = _r.pf(FStat[i], _unit(df1, i), df2, lower_tail=True)
    TwoSidedP = 2 * np.minimum(LeftP, RightP)
    FDR = _r.p_adjust(TwoSidedP, "BH")
    FDR[FDR > 0.3] = 1
    if np.min(FDR) == 1:
        return {"scale": s20, "df2": df2}

    outpw = fitFDistUnequalDF1(x, df1, covariate=covariate, robust=False, prior_weights=FDR)
    s20 = outpw["scale"]
    df2 = outpw["df2"]

    r = _r.rank_average(FStat)
    UniformP = (n - r + 0.5) / n
    ProbNotOutlier = np.minimum(RightP / UniformP, 1)
    if np.min(ProbNotOutlier) == 1:
        return outpw

    i = _r.which_min(RightP)
    minRightP = RightP[i]
    if minRightP == 0:
        df2_outlier = 0.0
        df2_shrunk = ProbNotOutlier * df2
    else:
        df2_outlier = _r.log(0.5) / _r.log(minRightP) * df2
        NewLogRightP = _r.pf(FStat[i], _unit(df1, i), df2_outlier, lower_tail=False, log_p=True)
        df2_outlier = _r.log(0.5) / NewLogRightP * df2_outlier
        df2_shrunk = ProbNotOutlier * df2 + (1 - ProbNotOutlier) * df2_outlier

    o = _r.order(RightP)
    df2_ordered = df2_shrunk[o]
    mm = _r.cumsum(df2_ordered)
    mm = mm / np.arange(1, n + 1, dtype=np.float64)
    imin = _r.which_min(mm)
    df2_ordered[:imin + 1] = mm[imin]
    df2_shrunk = df2_shrunk.copy()
    df2_shrunk[o] = _r.cummax(df2_ordered)
    return {"scale": s20, "df2": df2, "df2.outlier": df2_outlier, "df2.shrunk": df2_shrunk}


def _squeezeVar(var, df, var_prior, df_prior):
    """limma .squeezeVar(var, df, var.prior, df.prior)."""
    var = np.asarray(var, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    var_prior = np.asarray(var_prior, dtype=np.float64)
    df_prior = np.asarray(df_prior, dtype=np.float64)
    if np.isfinite(np.max(df_prior)):
        return (df * var + df_prior * var_prior) / (df + df_prior)
    n = var.size
    var_post = var_prior.copy() if var_prior.size == n else np.resize(var_prior, n)
    if np.min(df_prior) > 1e100:
        return var_post
    i = np.nonzero(np.isfinite(df_prior))[0]
    if df.size > 1:
        df = df[i]
    dfp = df_prior[i]
    var_post[i] = (df * var[i] + dfp * var_post[i]) / (df + dfp)
    return var_post


def squeezeVar(var, df, covariate=None, span=None, robust=False, winsor_tail_p=(0.05, 0.1)):
    """limma squeezeVar() with the fitFDistUnequalDF1 method.

    Returns dict(df.prior, var.prior, var.post). winsor_tail_p is accepted
    for call compatibility; like R's non-legacy path, it is not used.
    """
    var = np.asarray(var, dtype=np.float64).copy()
    n = var.size
    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return {"df.prior": np.zeros(1), "var.prior": var, "var.post": var}
    df = np.atleast_1d(np.asarray(df, dtype=np.float64))
    if df.size > 1:
        var[df == 0] = 0
    fit = fitFDistUnequalDF1(var, df, covariate=covariate, span=span, robust=robust)
    df_prior = fit.get("df2.shrunk")
    if df_prior is None:
        df_prior = fit["df2"]
    df_prior = np.atleast_1d(np.asarray(df_prior, dtype=np.float64))
    if np.isnan(df_prior).any():
        raise ValueError("Could not estimate prior df")
    var_prior = np.atleast_1d(np.asarray(fit["scale"], dtype=np.float64))
    var_post = _squeezeVar(var, df, var_prior, df_prior)
    return {"df.prior": df_prior, "var.prior": var_prior, "var.post": var_post}
