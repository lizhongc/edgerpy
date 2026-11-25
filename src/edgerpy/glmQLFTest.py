"""glmQLFit and glmQLFTest, edgeR v4 (legacy=FALSE) only (R: glmQLFTest.R)."""

import numpy as np
import pandas as pd

from . import _C, _r
from .aveLogCPM import _aveLogCPM_default, aveLogCPM
from .classes import DGEGLM, DGEList, DGELRT
from .DGEList import getOffset
from .glmfit import _default_design, _glmFit_default, checkSampleWeights
from .helpers import (as_matrix, coef_index, compressDispersions, compressOffsets, compressWeights,
                   contrastAsCoef, design_matrix, mat, treatment_design)
from ._squeezeVar import squeezeVar


def glmQLFit(y, design=None, dispersion=None, abundance_trend=True, covariate_trend=None, robust=False,
             sample_weights=False, winsor_tail_p=(0.05, 0.1), legacy=False, nthreads=1):
    """Genewise Negative Binomial Generalized Linear Models with Quasi-Dispersion Estimation

    Fit a negative binomial generalized log-linear model to the read counts for each gene. Estimate
    the genewise quasi-dispersions with empirical Bayes moderation.

    Parameters
    ----------
    y : DGEList
        The counts. Offsets (``offset`` or ``offset.prior``, else the effective library sizes),
        observation weights, ``AveLogCPM`` and the samples' technical and biological weights are taken
        from ``y``.
    design : DataFrame or ndarray, optional
        Design matrix for the genewise linear models. Must be of full column rank. Defaults to
        ``y["design"]``; if ``y`` has none, to the treatment-contrast design of
        ``y["samples"]["group"]`` (R's ``model.matrix(~group)``) when the group has more than one
        level, and otherwise to a single column of ones (the samples are replicates).
    dispersion : float or array-like, optional
        Negative binomial dispersions: one value, one per gene, or a matrix the size of the counts.
        Usually left ``None``, in which case it is estimated (see Notes). Only preset ``dispersion``
        if you know what you are doing! Sample-wise dispersions are set through the biological
        sample weights, which, together with the technical weights, are estimated by
        ``sampleWeights``. Dispersion values greater than 4 are capped at 4.
    abundance_trend : bool, default True
        Whether to allow an abundance trend for the quasi-dispersion prior.
    covariate_trend : array-like, optional
        One value per gene. If given, it is used instead of ``AveLogCPM`` as the covariate for the
        trended quasi-dispersion prior.
    robust : bool, default False
        Whether to estimate the prior distribution for the quasi-dispersions robustly.
    sample_weights : bool, default False
        If ``True``, technical and biological sample weights are first estimated by
        ``sampleWeights(y, design, nthreads=nthreads)`` and stored in ``y["samples"]``, replacing any
        there, and the fit uses them. The weights are estimated with the Cox-Reid common dispersion,
        and the fit uses the same estimate, so a given ``dispersion`` is not used.
    winsor_tail_p : tuple of two floats, default (0.05, 0.1)
        Proportions to trim (Winsorize) from the lower and upper tails of the distribution of
        genewise deviances when estimating the hyperparameters of the quasi-dispersion prior. Used
        when ``robust=True``.
    legacy : bool, default False
        Only ``False`` (the edgeR v4 method) is available; ``True`` raises ``NotImplementedError``.
    nthreads : int, default 1
        Number of OpenMP threads to use.

    Returns
    -------
    DGEGLM
        The components of a ``glmFit`` fit (``coefficients``, ``fitted.values``, ``deviance``,
        ``df.residual``, ``dispersion``, ...), plus:

        ``df.residual.adj``
            Adjusted residual degrees of freedom for each gene.
        ``deviance.adj``
            Adjusted residual deviance for each gene.
        ``df.prior``
            Prior degrees of freedom for the quasi-dispersions: one value per gene if
            ``robust=True``, otherwise a single value.
        ``s2.prior``
            Prior value for the quasi-dispersions: one value per gene if ``abundance_trend=True``,
            otherwise a single value.
        ``s2.post``
            Posterior genewise quasi-dispersions.
        ``average.ql.dispersion``
            Average quasi-dispersion, used to scale the NB dispersion after estimating the
            quasi-dispersions.

        and ``samples``, ``genes`` and ``AveLogCPM`` from ``y``.

    Notes
    -----
    ``glmQLFit`` and ``glmQLFTest`` implement the quasi-likelihood (QL) methods of Lund et al (2012)
    with some enhancements and with slightly different GLM, trend and FDR methods. See Lun et al (2016)
    or Chen et al (2016) for tutorials describing the use of ``glmQLFit`` and ``glmQLFTest`` as part
    of a complete analysis pipeline.

    ``glmQLFit`` is similar to ``glmFit`` except that it also estimates a quasi-dispersion for each
    gene. It calls limma's ``squeezeVar`` to conduct empirical Bayes moderation of the genewise
    quasi-dispersions. If ``robust=True``, then the robust hyperparameter estimation features of
    ``squeezeVar`` are used (Phipson et al, 2016). If ``abundance_trend=True``, then a prior trend is
    estimated based on the average logCPMs. If ``covariate_trend`` is given, then a prior trend is
    estimated using the ``covariate_trend`` values as predictors.

    ``glmQLFit`` gives special attention to small counts and zero counts. It adjusts both the
    residual deviances and the residual degrees of freedom to improve the accuracy of the
    quasi-dispersion estimates even for very small counts (Chen et al 2025). With this method, the
    residual deviance is no longer equal to the usual residual deviance from generalized linear model
    theory and the residual degrees of freedom are no longer integers. ``glmQLFTest`` then gives good
    error rate control even for data with many small counts. This method was introduced in edgeR 4.0.0
    (Bioconductor 3.18).

    The NB dispersion is estimated by ``glmQLFit`` whenever ``dispersion=None``: it is the common
    dispersion of the most highly expressed genes by ``AveLogCPM``. The number of top genes used is
    ``min(ceiling(p * ngenes), 10000)``, where
    ``p = chooseLowessSpan(ngenes * sqrt(df.residual), small.n=20, min.span=0.02)``, ``ngenes`` is the
    number of genes and ``df.residual`` is the residual degrees of freedom, equal to the number of
    samples minus the number of design columns.

    A given ``dispersion`` must be based on a global model, that is, it must be trended or common
    dispersions. It is not correct to supply genewise dispersions because ``glmQLFit`` estimates
    gene-specific variability using the quasi-dispersions.

    R documentation by Yunshun Chen, Aaron Lun, Davis McCarthy, Lizhong Chen and Gordon Smyth; adapted
    for edgerpy.
    """
    if legacy:
        raise NotImplementedError("edgerpy implements the edgeR v4 pipeline only (legacy=False)")
    if not isinstance(y, DGEList):
        raise TypeError("glmQLFit() needs a DGEList")
    if design is None:
        design = y.get("design")
        if design is None:
            group = y["samples"]["group"].array.remove_unused_categories()
            if len(group.categories) > 1:
                design = treatment_design(group, "group")
    # sample weights are estimated by sampleWeights with its own Cox-Reid dispersion, which the fit then also uses
    if sample_weights:
        dispersion = None
    if y.get("AveLogCPM") is None:
        y = y.copy()
        y["AveLogCPM"] = aveLogCPM(y, nthreads=nthreads)
    if sample_weights:
        from .sampleWeights import sampleWeights
        y = sampleWeights(y, design, nthreads=nthreads)
    samples = y["samples"]
    fit = _glmQLFit_default(y["counts"], design=design, dispersion=dispersion, offset=getOffset(y), lib_size=None,
                            weights=y.get("weights"), technical_weights=samples.get("technical.weights"),
                            biological_weights=samples.get("biological.weights"), abundance_trend=abundance_trend,
                            AveLogCPM=y["AveLogCPM"], covariate_trend=covariate_trend, robust=robust,
                            winsor_tail_p=winsor_tail_p, nthreads=nthreads)
    fit["samples"] = y["samples"]
    if y.get("genes") is not None:
        fit["genes"] = y["genes"]
    fit["AveLogCPM"] = y["AveLogCPM"]
    return fit


def _glmQLFit_default(y, design=None, dispersion=None, offset=None, lib_size=None, weights=None,
                      technical_weights=None, biological_weights=None, abundance_trend=True, AveLogCPM=None, covariate_trend=None, robust=False,
                      winsor_tail_p=(0.05, 0.1), nthreads=1):
    counts = as_matrix(y)
    ym = mat(counts)
    if design is None:
        design = _default_design(ym.shape[1], list(counts.columns))
    X, _, _ = design_matrix(design, ym.shape[1])

    if AveLogCPM is None:
        AveLogCPM = _aveLogCPM_default(ym, offset=offset, lib_size=lib_size, weights=weights, dispersion=dispersion,
                                       nthreads=nthreads)
    # weights are GLM prior weights; technical and biological weights (one per sample, None means 1) are used
    # only by the v4 QL pipeline, not for AveLogCPM or the dispersion estimate
    weights = compressWeights(ym, weights)
    technical_weights = checkSampleWeights(technical_weights, ym.shape[1])
    biological_weights = checkSampleWeights(biological_weights, ym.shape[1])
    offset = compressOffsets(ym, offset, lib_size)

    AveLogCPM2 = AveLogCPM
    if covariate_trend is None:
        cov = AveLogCPM if abundance_trend else None
    else:
        cov = covariate_trend

    if dispersion is None:
        dispersion = _C.coxreid_disp_top(ym, offset.to_c(), weights.to_c(), X, np.asarray(AveLogCPM2, dtype=np.float64),
                                         0.0, float(_r.sqrt(2.0)), 1e-5, True, int(nthreads))
    dispersion_mat = compressDispersions(ym, dispersion)
    fit = _glmFit_default(counts, design=design, dispersion=dispersion_mat, offset=offset, lib_size=lib_size,
                          weights=weights, technical_weights=technical_weights, biological_weights=biological_weights,
                          nthreads=nthreads, prior_count=0)
    ave_ql_disp = _C.compute_ave_qd(ym, mat(fit["fitted.values"]), X, dispersion_mat.to_c(),
                                    np.asarray(AveLogCPM2, dtype=np.float64), weights.to_c(), technical_weights,
                                    biological_weights, int(nthreads))

    fit = _glmFit_default(counts, design=design, dispersion=np.asarray(dispersion) / ave_ql_disp, offset=offset,
                          lib_size=lib_size, weights=weights, technical_weights=technical_weights,
                          biological_weights=biological_weights, nthreads=nthreads)
    fit["dispersion"] = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))

    out = _C.compute_adj_vec(ym, mat(fit["fitted.values"]), X, dispersion_mat.to_c(), float(ave_ql_disp),
                             weights.to_c(), technical_weights, biological_weights, int(nthreads))
    s2 = out["s2"]
    fit["df.residual.adj"] = df_residual = out["df"]
    fit["deviance.adj"] = out["deviance"]
    fit["average.ql.dispersion"] = np.array([ave_ql_disp])

    s2_fit = squeezeVar(s2, df=df_residual, covariate=cov, robust=robust, winsor_tail_p=winsor_tail_p)

    fit["AveLogCPM"] = np.asarray(AveLogCPM2, dtype=np.float64)
    fit["df.prior"] = s2_fit["df.prior"]
    fit["s2.post"] = s2_fit["var.post"]
    fit["s2.prior"] = s2_fit["var.prior"]
    fit["nthreads"] = nthreads
    return fit


def _coef_setup(fit, coef, contrast):
    """Shared coef/contrast handling of glmQLFTest/binQLFTest/diffSplice."""
    X, colnames, _ = design_matrix(fit["design"], None)
    if X.shape[1] < 2:
        raise ValueError("Need at least two columns for design, usually the first is the intercept column")
    coef_names = colnames
    ln2 = _r.log(2.0)
    if contrast is None:
        coef1 = coef_index(coef, colnames, X.shape[1])
        if coef is not None and (isinstance(coef, str) or (isinstance(coef, (list, tuple)) and coef and all(isinstance(c, str) for c in coef))):
            coef_name = [coef] if isinstance(coef, str) else list(dict.fromkeys(coef))
        else:
            coef_name = [coef_names[c - 1] for c in coef1] if coef_names is not None else coef1
        if isinstance(coef_name, list) and len(coef_name) == 1:
            coef_name = coef_name[0]
        cols = [c - 1 for c in coef1]
        B = mat(fit["coefficients"])[:, cols] / ln2
        Bu = mat(fit["unshrunk.coefficients"])[:, cols] / ln2 if fit.get("unshrunk.coefficients") is not None else None
        lcols = [coef_names[c] if coef_names is not None else str(c + 1) for c in cols]
        return X, coef1, coef_name, B, Bu, lcols
    contrast = np.asarray(contrast, dtype=np.float64)
    if contrast.ndim == 1:
        contrast = contrast.reshape(-1, 1)
        ccn = None
    else:
        ccn = None
    reform = contrastAsCoef(X, contrast, contrast_colnames=ccn, first=True)
    coef1 = reform["coef"]
    Xr = reform["design"]
    if len(coef1) > 1:
        coef_name = f"LR test on {len(coef1)} degrees of freedom"
    else:
        cv = contrast[:, 0]
        i = cv != 0
        coef_name = " ".join(f"{_r.num2str(v)}*{n}" for v, n in zip(cv[i], np.asarray(coef_names, dtype=object)[i]))
    B = _r.matprod(mat(fit["coefficients"]), contrast) / ln2
    Bu = _r.matprod(mat(fit["unshrunk.coefficients"]), contrast) / ln2 \
        if fit.get("unshrunk.coefficients") is not None else None
    lcols = [str(k + 1) for k in range(contrast.shape[1])]
    return Xr, coef1, coef_name, B, Bu, lcols


def _table(rownames, effect_name, B, lcols, logCPM, F, P):
    rn = _r.make_unique(rownames) if rownames is not None else [str(k + 1) for k in range(len(F))]
    data = {}
    if B.shape[1] == 1:
        data[effect_name] = B[:, 0]
    else:
        for k, c in enumerate(lcols):
            data[f"{effect_name}.{c}"] = B[:, k]
    data["logCPM"] = np.asarray(logCPM, dtype=np.float64)
    data["F"] = F
    data["PValue"] = P
    return pd.DataFrame(data, index=rn)


def glmQLFTest(glmfit, coef=None, contrast=None, fc=1, lfc=None, poisson_bound=True, upshot=True):
    """Quasi-Likelihood F-Tests

    Conduct genewise quasi F-tests for a given coefficient or coefficient contrast.

    Parameters
    ----------
    glmfit : DGEGLM
        Output from ``glmQLFit``. Raises ``ValueError`` if it has no ``s2.post`` (e.g. a ``glmFit``
        fit).
    coef : int, str or list, optional
        Which coefficients of the linear model are to be tested equal to zero: 0-based column
        positions or column names of the design. Defaults to the last column. Ignored if
        ``contrast`` is given.
    contrast : array-like, optional
        One or more contrasts of the linear model coefficients to be tested equal to zero: a vector
        with one value per design column, or a matrix with one column per contrast.
    fc : float or array-like, default 1
        Minimum fold-change below which changes are not considered scientifically meaningful.
    lfc : float or array-like, optional
        Minimum log2-fold-change below which changes are not considered scientifically meaningful.
        Defaults to ``log2(fc)``. If given then it takes precedence over ``fc``. Must be
        non-negative.
    poisson_bound : bool, default True
        Not used: it is not relevant for the edgeR v4 QL pipeline, and R ignores it for fits from
        the edgeR v4 ``glmQLFit``.
    upshot : bool, default True
        If ``True``, then a more powerful test is conducted assuming that the true logFC are
        uniformly distributed over the null interval from ``-lfc`` to ``lfc``. If ``False``, then the
        standard TREAT p-values are computed assuming the worst-case scenario of true logFC equal to
        ``lfc``.

    Returns
    -------
    DGELRT
        The components of ``glmfit`` (except ``counts``), plus:

        ``table``
            A DataFrame with a row for each gene and columns ``logFC`` (log2-fold-change; one column
            ``logFC.<name>`` per coefficient or ``logFC.<k>`` per contrast when several are tested),
            ``logCPM`` (average log2-counts per million), ``F`` (quasi-likelihood F-statistic) and
            ``PValue``.
        ``comparison``
            The name of the coefficient or contrast tested.
        ``df.test``
            The numerator degrees of freedom of the test (the number of coefficients or contrasts).
        ``df.total``
            The denominator degrees of freedom for the F-test, equal to ``df.prior +
            df.residual.adj``, bounded by the total residual degrees of freedom.
        ``lfc``
            The minimum log2-fold-change, if positive.

    Notes
    -----
    ``glmQLFTest`` is typically used after ``glmQLFit``. The two functions implement the
    quasi-likelihood (QL) methods of Lund et al (2012) with some enhancements and with slightly
    different GLM, trend and FDR methods. See Lun et al (2016) or Chen et al (2016) for tutorials
    describing the use of ``glmQLFit`` and ``glmQLFTest`` as part of a complete analysis pipeline.

    ``glmQLFTest`` is similar to ``glmLRT`` except that it replaces likelihood ratio tests with
    empirical Bayes quasi-likelihood F-tests. The p-values from ``glmQLFTest`` are always greater
    than or equal to those that would be obtained from ``glmLRT`` using the same negative binomial
    dispersions.

    If ``lfc`` is positive, the test is against the null hypothesis that the absolute log2-fold-change
    is at most ``lfc`` (TREAT, or UPSHOT with ``upshot=True``).

    R documentation by Yunshun Chen, Aaron Lun, Davis McCarthy, Lizhong Chen and Gordon Smyth; adapted
    for edgerpy.
    """
    if not isinstance(glmfit, DGEGLM):
        raise TypeError("glmfit must be an DGEGLM object produced by glmQLFit")
    if glmfit.get("s2.post") is None:
        raise ValueError("need to run glmQLFit before glmQLFTest")
    X, coef1, coef_name, logFC, unshrunk_logFC, lcols = _coef_setup(glmfit, coef, contrast)
    ncoef = len(coef1)

    if glmfit.get("dispersion") is not None:
        if glmfit.get("average.ql.dispersion") is None:
            dispersion = glmfit["dispersion"]
        else:
            dispersion = glmfit["dispersion"] / glmfit["average.ql.dispersion"]
    else:
        dispersion = None

    if lfc is None:
        lfc = _r.log2(float(fc))
    lfc = np.atleast_1d(np.asarray(lfc, dtype=np.float64))
    if lfc.size == 1:
        lfc = np.repeat(lfc, ncoef)
    if np.sum(lfc < 0) > 0:
        raise ValueError("lfc should be non-negative")
    if lfc.size != ncoef and lfc.size > 1:
        raise ValueError("fc or lfc vector of wrong length")

    if glmfit.get("df.residual.zeros") is None:
        df_residual = glmfit["df.residual.adj"]
    else:
        raise NotImplementedError("legacy (df.residual.zeros) fits are not supported")

    df_total = glmfit["df.prior"] + df_residual
    df_residual_total = float(np.sum(np.asarray(glmfit["df.residual"], dtype=np.int64)))
    df_total = np.minimum(df_total, df_residual_total)

    ym = mat(glmfit["counts"])
    offset_mat = compressOffsets(ym, offset=glmfit["offset"])
    dispersion_mat = compressDispersions(ym, dispersion)
    weights_mat = compressWeights(ym, glmfit.get("weights"))
    logFCt = unshrunk_logFC if unshrunk_logFC is not None else logFC
    out_fit = glmfit.copy()
    if np.any(lfc > 0):
        out_fit["lfc"] = lfc
    res = _C.glm_ftest(ym, offset_mat.to_c(), dispersion_mat.to_c(), weights_mat.to_c(),
                       glmfit.get("technical.weights"), glmfit.get("biological.weights"), X,
                       np.asarray(coef1, dtype=np.int32), np.asarray(glmfit["deviance"], dtype=np.float64),
                       np.asarray(glmfit["s2.post"], dtype=np.float64), np.asarray(df_total, dtype=np.float64), lfc,
                       np.asfortranarray(logFCt), bool(upshot), int(glmfit.get("nthreads") or 1))

    tab = _table(glmfit.rownames, "logFC", logFC, lcols, glmfit["AveLogCPM"], res["F"], res["PValue"])
    del out_fit["counts"]
    out_fit["table"] = tab
    out_fit["comparison"] = coef_name
    out_fit["df.test"] = ncoef
    out_fit["df.total"] = df_total
    return DGELRT(out_fit)
