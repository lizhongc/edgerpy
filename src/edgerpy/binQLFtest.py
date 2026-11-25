"""binFit, binQLFit, binQLFTest (R: binFit.R, binQLFtest.R)."""

import numpy as np

from . import _C, _r
from .aveLogCPM import _aveLogCPM_default, aveLogCPM
from .classes import DGEBIN, DGELRT, PCList
from .glmQLFTest import _coef_setup, _table
from .helpers import (as_matrix, compressOffsets, design_matrix, frame, makeCompressedMatrix, mat,
                   nonEstimable, treatment_design)
from ._squeezeVar import squeezeVar


def _check_yz_design(y, z, design):
    yc, zc = as_matrix(y), as_matrix(z)
    if yc.shape != zc.shape:
        raise ValueError("The dimensions of 'y' and 'z' are not matched")
    nlib = yc.shape[1]
    if design is None:
        import pandas as pd
        design = pd.DataFrame(np.ones((nlib, 1)), index=list(yc.columns), columns=["Intercept"], copy=False)
        X, cn, ddf = design_matrix(design, nlib)
    else:
        X, cn, ddf = design_matrix(design, nlib)
        if X.shape[0] != nlib:
            raise ValueError("nrow(design) disagrees with ncol(y)")
        ne = nonEstimable(X, cn)
        if ne is not None:
            raise ValueError("Design matrix not of full rank.  The following coefficients not estimable:\n " + " ".join(ne))
    return yc, zc, X, cn, ddf


def _design_default(y, design):
    if design is None:
        design = y.get("design")
        if design is None:
            group = y["samples"]["group"].array.remove_unused_categories()
            if len(group.categories) > 1:
                design = treatment_design(group, "y$samples$group")
    return design


def binFit(y, design=None, offset=None, prior_count=1, nthreads=1):
    """Genewise Binomial Generalized Linear Models

    Fit a binomial generalized log-linear model to the paired counts for each gene.

    Parameters
    ----------
    y : PCList
        The paired counts (``counts`` and ``counts2``). Observation weights are taken from
        ``y["weights"]``. Other objects raise ``TypeError``.
    design : DataFrame or ndarray, optional
        Design matrix for the genewise linear models. Must be of full column rank. Defaults to
        ``y["design"]``; if ``y`` has none, to the treatment-contrast design of
        ``y["samples"]["group"]`` (R's ``model.matrix(~group)``) when the group has more than one
        level, and otherwise to a single column of ones (the samples are replicates).
    offset : float or array-like, optional
        Offsets for the log-linear models: one value, one per sample, or a matrix the size of the
        counts. Defaults to zero for all observations.
    prior_count : float, default 1
        Average prior count to be added to each observation to shrink the estimated log-odds-ratios
        towards zero.
    nthreads : int, default 1
        Number of OpenMP threads to use for the genewise computations. Stored in the fit, so that the
        tests downstream use the same number.

    Returns
    -------
    DGEBIN
        A fit with the components:

        ``coefficients``
            DataFrame of estimated coefficients from the GLM fits, on the natural log scale, one row
            per gene and one column per design column.
        ``unshrunk.coefficients``
            DataFrame of estimated coefficients from the unshrunk GLM fits. Present only when
            ``prior_count > 0``.
        ``fitted.values``
            DataFrame of fitted values from the GLM fits, the size of the counts.
        ``deviance``
            Deviances, one for each gene.
        ``df.residual``
            Residual degrees of freedom, one for each gene.
        ``counts``
            The first matrix of counts.
        ``counts2``
            The second matrix of counts.
        ``design``
            The design matrix.
        ``weights``
            The weights, when ``y`` has them.
        ``nthreads``
            The number of OpenMP threads used for the fit.

        and ``iter``, ``failed`` and ``method`` from the fitting, and ``samples``, ``genes`` and
        ``AveLogCPM`` from ``y`` (``AveLogCPM`` is computed if ``y`` has none).

    Notes
    -----
    Implements binomial generalized linear model (GLM) methods developed by Lizhong Chen.
    Specifically, ``binFit`` fits genewise binomial GLMs, all with the same design matrix but possibly
    different offsets and weights. When the design matrix defines a one-way layout, or can be
    re-parametrized to a one-way layout, the GLMs are fit very quickly by a one-way method. Otherwise
    a standard iterated weighted least squares algorithm is used.

    R documentation by Lizhong Chen and Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(y, PCList):
        raise TypeError("binFit() needs a PCList")
    design = _design_default(y, design)
    if y.get("AveLogCPM") is None:
        y = y.copy()
        y["AveLogCPM"] = aveLogCPM(y)
    fit = _binFit_default(y["counts"], y["counts2"], design=design, weights=y.get("weights"), offset=offset,
                          prior_count=prior_count, nthreads=nthreads)
    fit["samples"] = y["samples"]
    if y.get("genes") is not None:
        fit["genes"] = y["genes"]
    fit["AveLogCPM"] = y["AveLogCPM"]
    return fit


def _binFit_default(y, z, design=None, weights=None, offset=None, prior_count=1, nthreads=1):
    """R binFit.default(): genewise binomial GLMs of count matrices y and z."""
    yc, zc, X, cn, ddf = _check_yz_design(y, z, design)
    ym, zm = mat(yc), mat(zc)
    ntag, nlib = ym.shape
    coverage = ym + zm
    weights0 = coverage if weights is None else np.asarray(mat(weights), dtype=np.float64) * coverage
    prop = ym / np.maximum(coverage, 1)
    offset0 = makeCompressedMatrix(0.0, ym.shape) if offset is None else compressOffsets(ym, offset)
    r = _C.bin_fit(np.asfortranarray(prop), offset0.to_c(), np.asfortranarray(weights0, dtype=np.float64), X,
                   50, 1e-8, 100, 1e-6, int(nthreads))
    fit = DGEBIN()
    _bin_core(fit, r, yc, cn, X)
    fit["counts"] = yc
    fit["counts2"] = zc
    if prior_count > 0:
        fit["unshrunk.coefficients"] = fit["coefficients"]
        add = np.ceil(prior_count) / 2
        fit["coefficients"] = _binFit_default(ym + add, zm + add, design=X if cn is None else ddf, offset=offset,
                                              weights=weights, prior_count=0)["coefficients"]
        fit["coefficients"].index = list(yc.index)
    fit["df.residual"] = np.full(ntag, nlib - X.shape[1], dtype=np.int32)
    fit["design"] = ddf
    if weights is not None:
        fit["weights"] = weights
    fit["nthreads"] = nthreads
    return fit


def _bin_core(fit, r, yc, cn, X):
    rn = list(yc.index)
    fit["coefficients"] = frame(r["coefficients"], rn, cn if cn is not None else list(range(X.shape[1])))
    fit["fitted.values"] = frame(r["fitted.values"], rn, list(yc.columns))
    fit["deviance"] = r["deviance"]
    fit["iter"] = r["iter"]
    fit["failed"] = r["failed"].astype(bool)
    fit["method"] = r["method"]


def binQLFit(y, design=None, abundance_trend=True, covariate_trend=None, robust=True, offset=None, prior_count=1,
             nthreads=1):
    """Quasi-Likelihood Binomial Generalized Linear Models

    Fit a binomial generalized log-linear model to the paired counts for each gene and estimate the
    empirical Bayes moderated quasi-dispersions.

    Parameters
    ----------
    y : PCList
        The paired counts (``counts`` and ``counts2``). Observation weights and ``AveLogCPM`` are
        taken from ``y``; ``AveLogCPM`` is computed by ``aveLogCPM(y)`` if ``y`` has none. Other
        objects raise ``TypeError``.
    design : DataFrame or ndarray, optional
        Design matrix for the genewise linear models. Must be of full column rank. Defaults to
        ``y["design"]``; if ``y`` has none, to the treatment-contrast design of
        ``y["samples"]["group"]`` (R's ``model.matrix(~group)``) when the group has more than one
        level, and otherwise to a single column of ones (the samples are replicates).
    abundance_trend : bool, default True
        Whether to allow an abundance trend for the quasi-dispersion prior.
    covariate_trend : array-like, optional
        One value per gene. If given, it is used instead of ``AveLogCPM`` as the covariate for the
        trended quasi-dispersion prior.
    robust : bool, default True
        Whether to estimate the prior distribution for the quasi-dispersions robustly.
    offset : float or array-like, optional
        Offsets for the log-linear models: one value, one per sample, or a matrix the size of the
        counts. Defaults to zero for all observations.
    prior_count : float, default 1
        Average prior count to be added to each observation to shrink the estimated log-odds-ratios
        towards zero.
    nthreads : int, default 1
        Number of OpenMP threads to use for the genewise computations. Stored in the fit, so that the
        tests downstream use the same number.

    Returns
    -------
    DGEBIN
        The components of a ``binFit`` fit plus:

        ``df.residual.adj``
            Adjusted residual degrees of freedom for each gene.
        ``deviance.adj``
            Adjusted deviances, one for each gene.
        ``df.prior``
            Prior degrees of freedom for the quasi-dispersions: one value per gene if
            ``robust=True``, otherwise a single value.
        ``s2.prior``
            Prior value for the quasi-dispersions: one value per gene if ``abundance_trend=True``,
            otherwise a single value.
        ``s2.post``
            Posterior genewise quasi-dispersions.
        ``average.ql.dispersion``
            Average quasi-dispersion, the overall overdispersion for binomial models.
        ``AveLogCPM``
            Average log2-counts per million for each gene.

    Notes
    -----
    ``binQLFit`` is the binomial analogue of ``glmQLFit``. It first fits genewise binomial GLMs (as
    done by ``binFit``), all with the same design matrix but possibly different offsets and weights,
    then estimates the genewise quasi-dispersions and moderates them towards a common prior by
    empirical Bayes using limma's ``squeezeVar``. When the design matrix defines a one-way layout, or
    can be re-parametrized to a one-way layout, the GLMs are fit very quickly by a one-way method.
    Otherwise a standard iterated weighted least squares algorithm is used.

    The moderated quasi-dispersions (``s2.post``) and adjusted degrees of freedom are required by
    ``binQLFTest`` to conduct the quasi-likelihood F-tests.

    R documentation by Lizhong Chen and Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(y, PCList):
        raise TypeError("binQLFit() needs a PCList")
    design = _design_default(y, design)
    if y.get("AveLogCPM") is None:
        y = y.copy()
        y["AveLogCPM"] = aveLogCPM(y)
    fit = _binQLFit_default(y["counts"], y["counts2"], design=design, weights=y.get("weights"),
                            AveLogCPM=y["AveLogCPM"], abundance_trend=abundance_trend,
                            covariate_trend=covariate_trend, robust=robust, offset=offset, prior_count=prior_count,
                            nthreads=nthreads)
    fit["samples"] = y["samples"]
    if y.get("genes") is not None:
        fit["genes"] = y["genes"]
    fit["AveLogCPM"] = y["AveLogCPM"]
    return fit


def _binQLFit_default(y, z, design=None, weights=None, offset=None, AveLogCPM=None, abundance_trend=True,
                      covariate_trend=None, robust=True, prior_count=1, nthreads=1):
    """R binQLFit.default(): binomial QL fits of count matrices y and z."""
    yc, zc, X, cn, ddf = _check_yz_design(y, z, design)
    ym, zm = mat(yc), mat(zc)
    ntag, nlib = ym.shape
    offset0 = makeCompressedMatrix(0.0, ym.shape) if offset is None else compressOffsets(ym, offset)
    w = None if weights is None else np.asfortranarray(np.asarray(mat(weights), dtype=np.float64))
    r = _C.bin_ql_fit(ym, zm, w, offset0.to_c(), X, 50, 1e-8, 100, 1e-6, int(nthreads))
    fit = DGEBIN()
    _bin_core(fit, r, yc, cn, X)
    fit["df.residual.adj"] = df_adj = r["df.residual.adj"]
    fit["deviance.adj"] = r["deviance.adj"]
    s2 = r["s2"]

    if AveLogCPM is None:
        AveLogCPM = _aveLogCPM_default(ym + zm, lib_size=_r.colSums(ym) + _r.colSums(zm), prior_count=2,
                                       dispersion=0, weights=weights)
    fit["AveLogCPM"] = np.asarray(AveLogCPM, dtype=np.float64)
    if covariate_trend is None:
        cov = AveLogCPM if abundance_trend else None
    else:
        cov = covariate_trend
    s2_fit = squeezeVar(s2, df=df_adj, covariate=cov, robust=robust)
    fit["df.prior"] = s2_fit["df.prior"]
    fit["s2.post"] = s2_fit["var.post"]
    fit["s2.prior"] = s2_fit["var.prior"]
    fit["average.ql.dispersion"] = np.array([_r.median(s2_fit["var.prior"])])
    fit["counts"] = yc
    fit["counts2"] = zc
    if prior_count > 0:
        fit["unshrunk.coefficients"] = fit["coefficients"]
        add = np.ceil(prior_count) / 2
        fit["coefficients"] = _binFit_default(ym + add, zm + add, design=ddf, offset=offset, weights=weights,
                                              prior_count=0)["coefficients"]
        fit["coefficients"].index = list(yc.index)
    fit["df.residual"] = np.full(ntag, nlib - X.shape[1], dtype=np.int32)
    fit["design"] = ddf
    if weights is not None:
        fit["weights"] = weights
    fit["nthreads"] = nthreads
    return fit


def binQLFTest(glmfit, coef=None, contrast=None, or_=1, lor=None, upshot=True):
    """Quasi-Likelihood F-Tests for Binomial Models

    Conduct genewise quasi-likelihood F-tests for a given coefficient or coefficient contrast, for
    binomial models fitted by ``binQLFit``.

    Parameters
    ----------
    glmfit : DGEBIN
        Usually output from ``binQLFit``. Must contain the empirical Bayes quasi-dispersions
        (``s2.post``), else ``ValueError`` is raised. Other objects raise ``TypeError``.
    coef : int, str or list, optional
        Which coefficients of the binomial linear model are to be tested equal to zero, by name or
        0-based position. Defaults to the last coefficient. Ignored if ``contrast`` is given.
        Positions out of range raise ``IndexError``.
    contrast : array-like, optional
        A vector, or a matrix with one column per contrast, specifying one or more contrasts of the
        linear model coefficients to be tested equal to zero.
    or_ : float or array-like, default 1
        Minimum odds ratio below which changes are not considered scientifically meaningful. R's
        ``or``, renamed because ``or`` is a Python keyword.
    lor : float or array-like, optional
        Minimum absolute log2 odds ratio below which changes are not considered scientifically
        meaningful. Defaults to ``log2(or_)``. If given, it takes precedence over ``or_``. Must be
        non-negative.
    upshot : bool, default True
        If ``True``, a more powerful test is conducted assuming that the true log2 odds ratios are
        uniformly distributed over the null interval from ``-lor`` to ``lor``. If ``False``, the
        standard TREAT p-values are computed assuming the worst-case scenario of true log2 odds ratio
        equal to ``lor``.

    Returns
    -------
    DGELRT
        The components of ``glmfit`` except ``counts`` and ``counts2``, plus:

        ``table``
            DataFrame with one row per gene and the columns ``logOR``, the log2 odds ratio for the
            tested coefficient or contrast (``logOR.<name>`` for each when several are tested);
            ``logCPM``, the average log2-counts per million; ``F``, the quasi-likelihood
            F-statistic; ``PValue``, the associated p-value.
        ``comparison``
            Describes the coefficient or contrast being tested.
        ``df.test``
            Numerator degrees of freedom for the F-test, equal to the number of coefficients tested.
        ``df.total``
            Denominator (total) degrees of freedom for the F-test, one per gene.
        ``lor``
            The log2 odds ratio threshold, when it is positive.

    Notes
    -----
    ``binQLFTest`` is the binomial analogue of ``glmQLFTest`` and is typically used after
    ``binQLFit``. It conducts empirical Bayes quasi-likelihood F-tests for the binomial generalized
    linear models fitted genewise to paired counts (for example methylated and unmethylated read
    counts) by ``binQLFit``.

    The recommended workflow is to build a ``PCList`` object, fit the models with ``binQLFit``, then
    test with ``binQLFTest``: ``binQLFit`` attaches the average log2-CPM (``AveLogCPM``) and the
    moderated quasi-dispersions that ``binQLFTest`` requires.

    The reported ``logOR`` is the log2 odds ratio for the tested coefficient or contrast, equal to the
    estimated coefficient divided by ``log(2)``. When ``binQLFit`` is run with ``prior_count > 0``
    (the default), ``logOR`` is reported on the shrunk coefficient scale, whereas the F-test and its
    deviances are computed from the unshrunk maximum-likelihood fit.

    When a nonzero threshold is specified via ``or_`` or ``lor``, a TREAT-style test relative to the
    threshold is performed. With ``upshot=True`` the more powerful UPSHOT p-values are computed by
    Gauss quadrature over the null interval.

    R documentation by Lizhong Chen and Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(glmfit, DGEBIN):
        raise TypeError("glmfit must be a DGEBIN object produced by binQLFit")
    if glmfit.get("s2.post") is None:
        raise ValueError("need to run binQLFit before binQLFTest")
    X, coef1, coef_name, logOR, unshrunk_logOR, lcols = _coef_setup(glmfit, coef, contrast)
    ncoef = len(coef1)
    if lor is None:
        lor = _r.log2(float(or_))
    lor = np.atleast_1d(np.asarray(lor, dtype=np.float64))
    if lor.size == 1:
        lor = np.repeat(lor, ncoef)
    if np.sum(lor < 0) > 0:
        raise ValueError("lor should be non-negative")
    if lor.size != ncoef and lor.size > 1:
        raise ValueError("'or' or 'lor' vector of wrong length")

    df_residual = glmfit["df.residual.adj"]
    df_total = glmfit["df.prior"] + df_residual
    df_residual_total = float(np.sum(np.asarray(glmfit["df.residual"], dtype=np.int64)))
    df_total = np.minimum(df_total, df_residual_total)

    logORt = unshrunk_logOR if unshrunk_logOR is not None else logOR
    w = glmfit.get("weights")
    w = None if w is None else np.asfortranarray(np.asarray(mat(w), dtype=np.float64))
    res = _C.bin_ftest(mat(glmfit["counts"]), mat(glmfit["counts2"]), w, X, np.asarray(coef1, dtype=np.int32),
                       np.asarray(glmfit["deviance"], dtype=np.float64), np.asarray(glmfit["s2.post"], dtype=np.float64),
                       np.asarray(df_total, dtype=np.float64), lor, np.asfortranarray(logORt), bool(upshot),
                       int(glmfit.get("nthreads") or 1))
    tab = _table(glmfit.rownames, "logOR", logOR, lcols, glmfit["AveLogCPM"], res["F"], res["PValue"])
    out = glmfit.copy()
    if np.any(lor > 0):
        out["lor"] = lor
    del out["counts"]
    del out["counts2"]
    out["table"] = tab
    out["comparison"] = coef_name
    out["df.test"] = ncoef
    out["df.total"] = df_total
    return DGELRT(out)
