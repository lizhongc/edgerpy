"""glmFit (R: glmfit.R), the NB GLM fitter used inside glmQLFit."""

import warnings

import numpy as np
import pandas as pd

from . import _C, _r
from .aveLogCPM import aveLogCPM
from .classes import DGEGLM, DGEList
from .DGEList import getOffset
from .helpers import (as_matrix, compressDispersions, CompressedMatrix, compressOffsets, compressPrior,
                   compressWeights, design_matrix, frame, mat, nonEstimable, treatment_design)


def _default_design(nlib, rownames):
    return pd.DataFrame(np.ones((nlib, 1)), index=rownames, columns=["Intercept"], copy=False)


def checkSampleWeights(x, nlib):
    """R .checkSampleWeights(): technical or biological sample weights, None (all 1) or a finite positive numeric
    vector with one value per sample, normalized to geometric mean 1 (with a warning) if it is not."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.dtype.kind not in "fiu" or a.ndim != 1 or a.size != nlib or not np.all(np.isfinite(a) & (a > 0)):
        raise ValueError("Sample weights must be a numeric vector of finite positive values, one per sample")
    a = a.astype(np.float64)
    # weights are relative: rescale to geometric mean 1, beyond rounding
    lm = _r.mean(_r.log(a))
    if abs(lm) > 1e-8:
        warnings.warn("Sample weights normalized to geometric mean 1")
        a = a / _r.exp(lm)
    return a


def getDispersion(y):
    """R getDispersion(): the most complex dispersion in a DGEList (tagwise, then trended, then common), or None."""
    for key in ("tagwise.dispersion", "trended.dispersion", "common.dispersion"):
        if key in y:
            return y[key]
    return None


def glmFit(y, design=None, dispersion=None, prior_count=0.125, start=None, nthreads=1):
    """Genewise Negative Binomial Generalized Linear Models

    Fit a negative binomial generalized log-linear model to the read counts for each gene.

    Parameters
    ----------
    y : DGEList
        The counts. Offsets (``offset`` or ``offset.prior``, else the effective library sizes),
        observation weights and the samples' technical and biological weights are taken from ``y``.
    design : DataFrame or ndarray, optional
        Design matrix for the genewise linear models. Must be of full column rank. Defaults to
        ``y["design"]``; if ``y`` has none, to the one-way layout defined by
        ``y["samples"]["group"]`` (R's ``model.matrix(~group)``) when the group has more than one
        level, and otherwise to a single column of ones.
    dispersion : float or array-like, optional
        Negative binomial dispersions: one value, one per gene, or a matrix the size of the counts.
        If ``None``, they are taken from ``y`` in this order of precedence: tagwise dispersions,
        trended dispersions, common dispersion. Raises ``ValueError`` if ``y`` has none.
    prior_count : float, default 0.125
        Average prior count to be added to each observation to shrink the estimated log-fold-changes
        towards zero.
    start : DataFrame or ndarray, optional
        Initial estimates for the linear model coefficients, genes by design columns.
    nthreads : int, default 1
        Number of OpenMP threads to use.

    Returns
    -------
    DGEGLM
        With components:

        ``coefficients``
            Estimated coefficients from the GLM fits, on the natural log scale; a DataFrame of genes
            by design columns.
        ``fitted.values``
            Fitted values from the GLM fits, the same size as the counts.
        ``deviance``
            Deviances, one for each gene.
        ``iter``, ``failed``, ``method``
            Iterations, convergence failures and the fitting method, as from R.
        ``counts``
            The counts from ``y``.
        ``unshrunk.coefficients``
            Estimated coefficients when no log-fold-change shrinkage is applied, on the natural log
            scale. Present only when ``prior_count`` is not 0.
        ``df.residual``
            Residual degrees of freedom, one for each gene.
        ``design``
            The design matrix.
        ``offset``
            The linear model offsets (a ``CompressedMatrix``).
        ``dispersion``
            The dispersions used for the fit.
        ``weights``
            The observation weights of ``y``, if any.
        ``technical.weights``
            The technical sample weights, if any. The ``deviance`` is then that of the scaled counts
            ``t*y``, while ``counts``, ``offset`` and ``fitted.values`` are on the original scale.
        ``biological.weights``
            The biological sample weights, if any. ``dispersion`` is stored as given; the dispersion
            of sample ``j`` used in the fit is ``dispersion / biological.weights[j]``.
        ``prior.count``
            The ``prior_count`` used.
        ``nthreads``
            The number of OpenMP threads used, stored so that downstream tests reuse it.

        and ``samples``, ``genes``, ``prior.df`` and ``AveLogCPM`` from ``y`` (``AveLogCPM`` is
        computed if ``y`` has none).

    Notes
    -----
    Implements generalized linear model (GLM) methods developed by McCarthy et al (2012).
    Specifically, ``glmFit`` fits genewise negative binomial GLMs, all with the same design matrix
    but possibly different dispersions, offsets and weights. When the design matrix defines a one-way
    layout, or can be re-parametrized to a one-way layout, the GLMs are fitted very quickly using
    Fisher scoring (same algorithm as ``mglmOneGroup``). For more complex design matrices, the GLMs
    are fitted using Fisher scoring with Levenberg-style damping to ensure convergence (same
    algorithm as ``mglmLevenberg``).

    Positive ``prior_count`` values cause the returned coefficients to be shrunk in such a way that
    fold-changes between the treatment conditions are decreased and infinite fold-changes are avoided
    (Phipson, 2013). Larger ``prior_count`` values cause more shrinkage. Coefficient shrinkage does
    not affect the likelihood ratio tests or p-values.

    R documentation by Davis McCarthy, Gordon Smyth, Yunshun Chen, Aaron Lun, Lizhong Chen; adapted
    for edgerpy.
    """
    if not isinstance(y, DGEList):
        raise TypeError("glmFit() needs a DGEList")
    if design is None:
        design = y.get("design")
        if design is None:
            group = y["samples"]["group"].array.remove_unused_categories()
            if len(group.categories) > 1:
                design = treatment_design(group, "group")
    if dispersion is None:
        dispersion = getDispersion(y)
    if dispersion is None:
        raise ValueError("No dispersion values found in DGEList object.")
    offset = getOffset(y)
    if y.get("AveLogCPM") is None:
        y = y.copy()
        y["AveLogCPM"] = aveLogCPM(y)
    s = y["samples"]
    fit = _glmFit_default(y["counts"], design=design, dispersion=dispersion, offset=offset, lib_size=None,
                          weights=y.get("weights"), technical_weights=s.get("technical.weights"),
                          biological_weights=s.get("biological.weights"), prior_count=prior_count, start=start,
                          nthreads=nthreads)
    fit["samples"] = s
    if y.get("genes") is not None:
        fit["genes"] = y["genes"]
    if y.get("prior.df") is not None:
        fit["prior.df"] = y["prior.df"]
    fit["AveLogCPM"] = y["AveLogCPM"]
    return fit


def _glmFit_default(y, design=None, dispersion=None, offset=None, lib_size=None, weights=None, technical_weights=None,
                    biological_weights=None, prior_count=0.125, start=None, nthreads=1):
    """R glmFit.default(): genewise NB GLM fits of a count matrix. Returns a DGEGLM."""
    counts = as_matrix(y)
    ym = mat(counts)
    if ym.dtype.kind not in "fi":
        raise TypeError("y is not a numeric matrix")
    ntag, nlib = ym.shape

    if design is None:
        design = _default_design(nlib, list(counts.columns))
    X, colnames, design_df = design_matrix(design, nlib)
    if X.shape[0] != nlib:
        raise ValueError("nrow(design) disagrees with ncol(y)")
    ne = nonEstimable(X, colnames)
    if ne is not None:
        raise ValueError("Design matrix not of full rank.  The following coefficients not estimable:\n " + " ".join(ne))

    if dispersion is None:
        raise ValueError("No dispersion values provided.")
    if isinstance(dispersion, CompressedMatrix):
        if np.isnan(dispersion.values).any():
            raise ValueError("NA dispersions not allowed")
        if dispersion.dims != ym.shape:
            raise ValueError("Dimensions of dispersion don't agree with dimensions of y")
    else:
        d = np.asarray(dispersion, dtype=np.float64)
        if np.isnan(d).any():
            raise ValueError("NA dispersions not allowed")
        if d.ndim <= 1 and d.size not in (1, ntag):
            raise ValueError("dispersion has wrong length. As a vector, it should agree with nrow(y)")
        if d.ndim == 2 and d.shape != ym.shape:
            raise ValueError("Dimensions of dispersion don't agree with dimensions of y")
    dispersion_mat = compressDispersions(ym, dispersion)

    offset_cm = compressOffsets(ym, offset=offset, lib_size=lib_size)
    weights_mat = compressWeights(ym, weights)
    technical_weights = checkSampleWeights(technical_weights, nlib)
    biological_weights = checkSampleWeights(biological_weights, nlib)

    st = None
    if start is not None:
        st = np.asfortranarray(np.asarray(getattr(start, "to_numpy", lambda: start)(), dtype=np.float64))
    fit = _C.fit_glm(ym, offset_cm.to_c(), dispersion_mat.to_c(), weights_mat.to_c(), technical_weights,
                     biological_weights, X, 250, 1e-6, st, int(nthreads))

    rn = list(counts.index)
    cn = colnames if colnames is not None else list(range(X.shape[1]))
    out = DGEGLM()
    out["coefficients"] = frame(fit["coefficients"], rn, cn)
    out["fitted.values"] = frame(fit["fitted.values"], rn, list(counts.columns))
    out["deviance"] = fit["deviance"]
    out["iter"] = fit["iter"]
    out["failed"] = fit["failed"].astype(bool)
    out["method"] = fit["method"]
    out["counts"] = counts
    if prior_count > 0:
        out["unshrunk.coefficients"] = out["coefficients"]
        priors = compressPrior(ym, prior_count)
        coef = _C.pred_fc(ym, offset_cm.to_c(), priors.to_c(), dispersion_mat.to_c(), weights_mat.to_c(),
                          technical_weights, biological_weights, X, 250, 1e-6, int(nthreads))
        out["coefficients"] = frame(coef, rn, cn)
    out["df.residual"] = np.full(ntag, nlib - X.shape[1], dtype=np.int32)
    out["design"] = design_df
    out["offset"] = offset_cm
    out["dispersion"] = dispersion
    if weights is not None:
        out["weights"] = weights
    if technical_weights is not None:
        out["technical.weights"] = technical_weights
    if biological_weights is not None:
        out["biological.weights"] = biological_weights
    out["prior.count"] = prior_count
    out["nthreads"] = nthreads
    return out
