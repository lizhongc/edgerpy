"""estimateDisp for DGEList, simplified, and its building blocks
(R: edgeR estimateDisp.R, adjustedProfileLik.R, maximizeInterpolant.R, residDF.R).

The trend is always locfit (degree 0); there is no classic mode, so a design
is required. prior.df, when not given, is estimated as in R, but always
squeezed with fitFDistUnequalDF1: the result is R's whenever the residual df
are unequal (some zero fits), which is when limma makes the same choice.
"""

import warnings

import numpy as np
import pandas as pd

from . import _C, _r
from .locfitByCol import locfitByCol
from .aveLogCPM import _aveLogCPM_default, aveLogCPM
from .classes import DGEList
from .DGEList import getOffset
from .glmfit import _glmFit_default
from .helpers import (CompressedMatrix, as_matrix, chooseLowessSpan, compressDispersions, compressOffsets,
                      compressWeights, mat, nonEstimable)
from ._squeezeVar import squeezeVar


def adjustedProfileLik(dispersion, y, design, offset, weights=None, adjust=True, start=None, get_coef=False,
                       nthreads=1):
    """R adjustedProfileLik(): genewise Cox-Reid adjusted profile log-likelihoods.

    With get_coef=True returns {"apl": ..., "beta": coefficients}, the
    coefficients serving as `start` for a hot-started next fit.
    """
    ydf = as_matrix(y)
    ym = mat(ydf)
    X = np.asfortranarray(np.asarray(mat(design), dtype=np.float64))
    colnames = list(design.columns) if isinstance(design, pd.DataFrame) else None
    ne = nonEstimable(X, colnames)
    if ne is not None:
        raise ValueError("Design matrix not of full rank. The following coefficients not estimable:\n" + " ".join(ne))
    offset = compressOffsets(ym, offset=offset)
    dispersion = compressDispersions(ym, dispersion)
    weights = compressWeights(ym, weights)
    if start is not None:
        start = np.asfortranarray(np.asarray(mat(start), dtype=np.float64))
    fit = _C.fit_apl(ym, offset.to_c(), dispersion.to_c(), weights.to_c(), X, 250, 1e-6, start, bool(adjust),
                     int(nthreads))
    if get_coef:
        beta = pd.DataFrame(fit["coefficients"], index=ydf.index, columns=colnames, copy=False)
        return {"apl": fit["apl"], "beta": beta}
    return fit["apl"]


def maximizeInterpolant(x, y):
    """R maximizeInterpolant(): for each row of the likelihood matrix y on the
    sorted grid x, the position of the maximum of its interpolating spline."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asfortranarray(np.asarray(mat(y), dtype=np.float64))
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if x.size != y.shape[1]:
        raise ValueError("number of columns must equal number of spline points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("spline points must be unique and sorted")
    return _C.maximize_interpolant(x, y)


def _comboGroups(truths):
    """R .comboGroups(): row indices (0-based) grouped by their TRUE/FALSE pattern,
    groups in R's order (integer packing of 31 columns at a time, then order())."""
    truths = np.asarray(truths, dtype=bool)
    nr, nc = truths.shape
    assembly = []
    collected = 0
    step = 31
    bits = (2 ** np.arange(step)).astype(np.int64)
    while collected < nc:
        upper = min(nc - collected, step)
        assembly.append((truths[:, collected:collected + upper] * bits[:upper]).sum(axis=1).astype(np.float64))
        collected += step
    o = _r.order(*assembly)
    is_different = np.zeros(nr, dtype=bool)
    for key in assembly:
        ko = key[o]
        is_different |= np.concatenate(([True], np.diff(ko) != 0))
    first = np.flatnonzero(is_different)
    last = np.concatenate((first[1:] - 1, [nr - 1]))
    return [o[a:b + 1] for a, b in zip(first, last)]


def estimateDisp(y, design=None, prior_df=None, tagwise=True, robust=False, span=None, min_row_sum=5,
                 grid_length=21, grid_range=(-10, 10), nthreads=1):
    """Estimate Common, Trended and Tagwise NB Dispersions by Weighted Likelihood Empirical Bayes

    Maximizes the negative binomial likelihood to give the estimate of the common, trended and
    tagwise dispersions across all tags.

    Parameters
    ----------
    y : DGEList
        The counts. The effective library sizes (``lib.size * norm.factors``), offsets and observation
        weights are taken from ``y``.
    design : DataFrame or ndarray
        The design matrix. Required: raises ``ValueError`` if not given.
    prior_df : float or array-like, optional
        Prior degrees of freedom, used in calculating ``prior.n``. If ``None``, they are estimated
        (see Notes).
    tagwise : bool, default True
        Whether the tagwise dispersions should be estimated.
    robust : bool, default False
        Whether the estimation of ``prior.df`` should be robustified against hypervariable genes.
    span : float, optional
        Width of the smoothing window, as a proportion of the data set. Defaults to the value chosen
        by ``chooseLowessSpan(n, small.n=50, min.span=0.3, power=1/3)``, ``n`` being the number of
        tags used.
    min_row_sum : float, default 5
        Filter for low abundance tags: only tags with total sum of counts at least this value are
        used. Low abundance tags can adversely affect the dispersion estimation, so this argument
        allows the user to select an appropriate filter threshold for the tag abundance.
    grid_length : int, default 21
        The number of points on which the interpolation is applied for each tag.
    grid_range : tuple of two floats, default (-10, 10)
        The range of the grid points around the trend on a log2 scale.
    nthreads : int, default 1
        Number of OpenMP threads, used by edgeR's genewise C code (the likelihood grid, the GLM fits,
        ``aveLogCPM``) and by edgeR's locfit, which fits the grid columns in parallel. The results do
        not depend on it.

    Returns
    -------
    DGEList
        A copy of ``y`` with the following components added:

        ``common.dispersion``
            Estimate of the common dispersion.
        ``trended.dispersion``
            Estimates of the trended dispersions.
        ``tagwise.dispersion``
            Tagwise estimates of the dispersion parameter, if ``tagwise=True``.
        ``AveLogCPM``
            log2(AveCPM) for each row of ``y``.
        ``trend.method``
            ``"locfit"``.
        ``prior.df``
            Prior degrees of freedom. If ``robust=True`` then ``prior.df`` is a vector with smaller
            values assigned to hypervariable outlier genes.
        ``prior.n``
            Estimate of the prior weight, i.e. the smoothing parameter that indicates the weight to
            put on the common likelihood compared to the individual tag's likelihood.
        ``span``
            Width of the smoothing window used in estimating dispersions.

        ``prior.df``, ``prior.n`` and ``span`` are present only when estimated or given.

    Notes
    -----
    This function calculates a matrix of likelihoods for each tag at a set of dispersion grid points,
    and then applies the weighted likelihood empirical Bayes method to obtain posterior dispersion
    estimates. It calculates the adjusted profile log-likelihood for each tag and then maximizes it,
    similar to the functions ``estimateGLMCommonDisp``, ``estimateGLMTrendedDisp`` and
    ``estimateGLMTagwiseDisp``. The trend is always fitted by locfit (R's ``trend.method="locfit"``).

    Note that the terms 'tag' and 'gene' are synonymous here.

    With ``prior_df=None``, the prior degrees of freedom are estimated as in R: a GLM fit with the
    trended dispersions, residual degrees of freedom reduced for exact zero fits, then limma's
    ``squeezeVar``. The squeeze always uses limma's ``fitFDistUnequalDF1`` method. limma makes the
    same choice whenever the residual degrees of freedom are unequal, which happens as soon as a gene
    has zero fitted values in a whole group; then every component equals R's
    ``estimateDisp(y, design)``. When all residual degrees of freedom are equal, R uses limma's legacy
    ``fitFDist`` (``fitFDistRobustly`` with ``robust``): then only the common and trended
    dispersions, ``AveLogCPM`` and ``span`` equal R's, and ``prior.df``, ``prior.n`` and the tagwise
    dispersions differ. With a given ``prior_df``, every component equals R's.

    R documentation by Yunshun Chen, Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(y, DGEList):
        raise TypeError("estimateDisp() needs a DGEList")
    if design is None:
        raise ValueError("estimateDisp needs a design matrix")
    y = y.copy()
    s = y["samples"]
    lib_size = s["lib.size"].to_numpy(dtype=np.float64) * s["norm.factors"].to_numpy(dtype=np.float64)
    # edgeR 4.99.16 stores y$design only when the design was not given, so a given design leaves y$design as is
    d = _estimateDisp_default(y["counts"], design, lib_size=lib_size, offset=getOffset(y), prior_df=prior_df,
                              tagwise=tagwise, robust=robust, span=span, min_row_sum=min_row_sum,
                              grid_length=grid_length, grid_range=grid_range, weights=y.get("weights"),
                              nthreads=nthreads)
    y["common.dispersion"] = d["common.dispersion"]
    y["trended.dispersion"] = d["trended.dispersion"]
    if tagwise:
        y["tagwise.dispersion"] = d["tagwise.dispersion"]
    y["AveLogCPM"] = aveLogCPM(y, nthreads=nthreads)
    y["trend.method"] = "locfit"
    for key in ("prior.df", "prior.n", "span"):      # R's y$x <- NULL removes the component
        if d.get(key) is None:
            y.pop(key, None)
        else:
            y[key] = d[key]
    return y


def _seq_length_out(start, stop, n):
    """R seq(from, to, length.out=n) for double from and to (seq.default)."""
    start, stop = float(start), float(stop)
    if n > 2:
        if start == stop:
            return np.full(n, start)
        n1 = float(n - 1)
        return np.concatenate(([start], start + np.arange(1, n - 1, dtype=np.float64) * ((stop - start) / n1), [stop]))
    return np.array([start, stop])[:n]


def _estimateDisp_default(y, design, lib_size=None, offset=None, prior_df=None, tagwise=True, robust=False, span=None,
                          min_row_sum=5, grid_length=21, grid_range=(-10, 10), weights=None, nthreads=1):
    """R estimateDisp.default(), GLM branch with trend.method = "locfit" -> dict."""
    ym = mat(as_matrix(y))
    ntags, nlibs = ym.shape
    if ntags == 0:
        raise ValueError("y has no rows")
    if lib_size is None:
        lib_size = _r.colSums(ym)
    offset = compressOffsets(ym, offset=offset, lib_size=lib_size)
    weights = compressWeights(ym, weights)

    # genes with small counts are left out of the fits
    sel = _r.rowSums(ym) >= min_row_sum
    rows = np.flatnonzero(sel)
    sely = ym[rows] if rows.size < ntags else ym
    seloffset = offset.subset_rows(rows) if rows.size < ntags else offset
    selweights = weights.subset_rows(rows) if rows.size < ntags else weights

    # spline points
    spline_pts = _seq_length_out(grid_range[0], grid_range[1], grid_length)
    spline_disp = 0.1 * _r.pow(2.0, spline_pts)
    l0 = np.zeros((rows.size, grid_length), order="F")

    X = np.asfortranarray(np.asarray(mat(design), dtype=np.float64))
    if X.shape[1] >= nlibs:
        warnings.warn("No residual df: setting dispersion to NA")
        return {"common.dispersion": np.nan, "trended.dispersion": np.nan, "tagwise.dispersion": np.nan}

    # observations with means of zero, grouped by pattern
    glmfit = _glmFit_default(sely, X, offset=seloffset, dispersion=0.05, prior_count=0, nthreads=nthreads)
    zerofit = (mat(glmfit["counts"]) < 1e-4) & (mat(glmfit["fitted.values"]) < 1e-4)
    for subg in _comboGroups(zerofit):
        cur_nzero = ~zerofit[subg[0]]
        if not cur_nzero.any():
            continue
        # samples with zero means are removed from the design, avoiding fits converging to -Inf
        if cur_nzero.all():
            redesign = X
        else:
            redesign = X[cur_nzero]
            qr = _r.qr(redesign)
            redesign = np.asfortranarray(redesign[:, np.asarray(qr["pivot"][:qr["rank"]]) - 1])
            if redesign.shape[0] == redesign.shape[1]:
                continue
        cols = np.flatnonzero(cur_nzero)
        whole = cur_nzero.all()
        cury = np.asfortranarray(sely[np.ix_(subg, cols)])
        curo = seloffset.subset_rows(subg) if whole else seloffset.subset_rows(subg).subset_cols(cols)
        curw = selweights.subset_rows(subg) if whole else selweights.subset_rows(subg).subset_cols(cols)
        # the last fit hot-starts the next
        last_beta = None
        for i in range(grid_length):
            out = adjustedProfileLik(spline_disp[i], cury, redesign, curo, weights=curw, start=last_beta,
                                     get_coef=True, nthreads=nthreads)
            l0[subg, i] = out["apl"]
            last_beta = out["beta"]

    # common dispersion
    overall = maximizeInterpolant(spline_pts, _r.colSums(l0).reshape(1, -1))
    common = float(0.1 * _r.pow(2.0, overall)[0])

    # trended dispersion (WLEB with trend.method = "locfit")
    AveLogCPM = _aveLogCPM_default(ym, lib_size=lib_size, dispersion=common, weights=weights, nthreads=nthreads)
    cov = AveLogCPM[rows]
    if span is None:
        span = chooseLowessSpan(rows.size)
    m0 = locfitByCol(l0, cov, span=span, degree=0, nthreads=nthreads)
    disp_trend = 0.1 * _r.pow(2.0, maximizeInterpolant(spline_pts, m0))
    trended = np.full(ntags, disp_trend[_r.which_min(cov)])
    trended[rows] = disp_trend

    if not tagwise:
        return {"common.dispersion": common, "trended.dispersion": trended}

    # prior df, unless given: squeeze the residual deviances of a fit with the trended dispersions
    if prior_df is None:
        glmfit = _glmFit_default(sely, X, offset=seloffset, weights=selweights, dispersion=disp_trend,
                                 prior_count=0, nthreads=nthreads)
        # adjust df.residual for fitted values at zero
        zerofit = (mat(glmfit["counts"]) < 1e-4) & (mat(glmfit["fitted.values"]) < 1e-4)
        df_residual = _residDF(zerofit, X)
        # empirical Bayes squeezing of the quasi-likelihood variance factors
        with np.errstate(divide="ignore", invalid="ignore"):
            s2 = np.asarray(glmfit["deviance"], dtype=np.float64) / df_residual
        s2[df_residual == 0] = 0
        s2 = np.maximum(s2, 0)
        prior_df = squeezeVar(s2, df=df_residual, covariate=cov, robust=robust)["df.prior"]
    prior_df = np.asarray(prior_df, dtype=np.float64)
    prior_n = prior_df / (nlibs - X.shape[1])

    tagwise_disp = trended.copy()
    too_large = prior_n > 1e6
    if not too_large.all():
        temp_n = np.where(too_large, 1e6, prior_n)
        # weighted empirical Bayes posterior (WLEB, individual)
        if not np.isfinite(temp_n).all():
            raise ValueError("prior.n must be finite")
        l0a = l0 + (temp_n.reshape(-1, 1) if temp_n.ndim else temp_n) * m0
        individual = 0.1 * _r.pow(2.0, maximizeInterpolant(spline_pts, l0a))
        if not robust:
            tagwise_disp[rows] = individual
        else:
            keep = ~np.broadcast_to(too_large, (rows.size,))
            tagwise_disp[rows[keep]] = individual[keep]

    if robust:
        df_all, n_all = np.full(ntags, np.inf), np.full(ntags, np.inf)
        df_all[rows] = prior_df
        n_all[rows] = prior_n
        prior_df, prior_n = df_all, n_all

    return {"common.dispersion": common, "trended.dispersion": trended, "tagwise.dispersion": tagwise_disp,
            "span": span, "prior.df": prior_df, "prior.n": prior_n}



def _residDF(zero, design):
    """R edgeR .residDF(): effective residual df after adjusting for exact zeros."""
    nlib = zero.shape[1]
    ncoef = design.shape[1]
    nzero = zero.sum(axis=1).astype(np.int32)
    # default is no zero
    DF = np.full(nzero.size, nlib - ncoef, dtype=np.int32)
    # all zero case
    DF[nzero == nlib] = 0
    # anything in between?
    somezero = (nzero > 0) & (nzero < nlib)
    if somezero.any():
        zero2 = zero[somezero]
        DF2 = nlib - nzero[somezero]
        for i in _comboGroups(zero2):
            zeroi = zero2[i[0]]
            DF2[i] -= _r.qr(design[~zeroi])["rank"]
        DF[somezero] = np.maximum(DF2, 0)
    return DF
