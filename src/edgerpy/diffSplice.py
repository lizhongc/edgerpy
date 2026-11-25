"""diffSplice for DGEGLM fits and topSplice for its output
(R: edgeR diffSplice.R, diffSplice.DGEGLM; limma topSplice.R)."""

import numbers

import numpy as np
import pandas as pd

from . import _C, _r
from .classes import DGEDTU, DGEGLM
from .glmfit import _glmFit_default
from .helpers import (compressDispersions, compressOffsets, compressWeights, contrastAsCoef,
                   design_matrix, frame, mat)
from ._squeezeVar import squeezeVar


def diffSplice(fit, coef=None, contrast=None, geneid=None, exonid=None, robust=None, nexons_approx=10, verbose=True):
    """Differential Transcript or Exon Usage

    Given an edgeR fit object with multiple features per gene, test for differential usage of those
    features within genes between experimental conditions. The features may be any set of
    isoform-identifying features such as transcripts, exons or exon-junctions.

    Parameters
    ----------
    fit : DGEGLM
        A fit produced by ``glmQLFit``. Rows should correspond to transcripts, for a DTU analysis, or
        to exons and exon-exon junctions for a DEU analysis. Other objects raise ``TypeError``; a fit
        without quasi-dispersions raises ``ValueError``.
    coef : int or str, optional
        Which coefficient of the generalized linear model is to be tested for differential usage, by
        name or 0-based position. Defaults to the last coefficient.
    contrast : array-like, optional
        The contrast of the linear model coefficients to be tested for differential usage. Length must
        equal the number of columns of the design. If given, it takes precedence over ``coef``.
    geneid : str or array-like
        Gene identifiers. Either a vector with one value per row of ``fit`` or the name of the column
        of ``fit["genes"]`` containing the gene identifiers. Rows with the same ID are assumed to
        belong to the same gene. Required: if missing, ``TypeError`` is raised, as R stops.
    exonid : str or array-like, optional
        Feature or exon identifiers. Either a vector with one value per row of ``fit`` or the name of
        the column of ``fit["genes"]`` containing the feature identifiers.
    robust : bool, optional
        Whether the estimation of the empirical Bayes prior parameters should be robustified against
        outlier variances. By default, the same setting is used as in the ``glmQLFit`` call that
        created ``fit``.
    nexons_approx : int, default 10
        Exact test statistics are computed for all genes with up to this number of features (or
        exons). For genes with more features, a computationally faster approximation is used.
    verbose : bool, default True
        If ``True``, some diagnostic information about the number of genes and exons is printed.

    Returns
    -------
    DGEDTU
        R's ``MArrayLM``, containing both exon-level and gene-level tests. Results are sorted by
        ``geneid`` and by ``exonid`` within gene. Genes with only one feature are dropped. The
        components include:

        ``coefficients``
            Single-column DataFrame of differential usage coefficients for the coefficient or
            contrast specified by ``coef`` or ``contrast``. The coefficients give the difference
            between the log-fold-change for that feature versus the average log-fold-change for all
            other features for the same gene.
        ``t``
            Single-column DataFrame of quasi t-statistics.
        ``p.value``
            Single-column DataFrame of p-values corresponding to the t-statistics.
        ``genes``
            DataFrame of annotation for the low-level features.
        ``genecolname``
            Name of the column of ``genes`` containing gene IDs.
        ``gene.F``
            Single-column DataFrame of quasi F-statistics, one row for each gene.
        ``gene.F.p.value``
            Single-column DataFrame of p-values corresponding to ``gene.F``.
        ``gene.simes.p.value``
            Single-column DataFrame of Simes adjusted p-values, one row for each gene.
        ``gene.bonferroni.p.value``
            Single-column DataFrame of Bonferroni adjusted p-values, one row for each gene.
        ``gene.genes``
            DataFrame of gene annotation, with the number of features of each gene in ``NExons``.

        and ``design``, ``comparison``, ``exoncolname`` (when ``exonid`` is given),
        ``nexons.approx``, ``gene.df.prior``, ``gene.df.residual``, ``gene.df.total``, ``gene.s2``,
        ``gene.s2.prior``, ``gene.s2.post``, ``gene.firstexon`` and ``gene.lastexon``, as in R.

    Notes
    -----
    Given an edgeR QL fit with multiple features per gene, test for differential usage of those
    features within genes between experimental conditions. The features can be transcripts for a
    differential transcript usage (DTU) analysis (Baldoni et al 2025), or can be a combination of
    exons and exon-exon junctions for a differential exon usage (DEU) analysis. Another possibility
    is for the features to be transcript-overlap equivalence classes (Cmero et al 2019). The output
    can be further explored using ``topSplice`` and ``plotSplice``.

    Testing for differential usage is equivalent to testing whether the log-fold-changes in the
    ``fit`` differ between features for the same gene. Two different tests are provided, one at the
    gene level and one at the feature level. In both cases, the tests are conducted using quasi-F
    test statistics based on deviance differences. The gene-level test is a quasi-F-test for
    differences between the log-fold-changes, equivalent to testing for interaction between the
    contrast comparison and the transcripts for each gene. This produces one F-statistic for each
    gene, where the numerator df is one less than the number of exons and the denominator is the
    residual df pooled over all exons for that gene. The null hypothesis is that all the exons have
    the same log-fold-change and the alternative is that the log-fold-changes are not all equal. The
    feature-level tests compare each feature to the other features for the same gene. Each
    feature-level test compares the model in which all the features have the same log-fold-change vs
    the alternative model that just the specified feature has a different log-fold-change. The null
    hypothesis is that the log-fold-change for the specified feature is equal to the consensus
    log-fold-change for the other features, and the alternative is that it is not. This produces one
    quasi F-statistic for each feature, where the numerator df is equal to one and the denominator is
    the same as for the gene-level test. The feature-level F-statistics are then converted into
    t-tests by taking square-roots and multiplying by the sign of log-fold-change difference.

    For genes with more than ``nexons_approx`` features, a fast approximation is used for the
    feature-level test, assuming that the consensus log-fold-change for the other features should be
    close to the consensus log-fold-change for all features for that gene. The null hypothesis for
    these genes is that the log-fold-change for the specified feature is equal to the consensus
    log-fold-change for all other features, and the alternative is that it is not. The approximation
    should be very slightly conservative.

    ``fit`` should be created by ``glmQLFit``, which in edgerpy always uses the edgeR v4
    bias-adjusted deviances. The number of OpenMP threads used for the genewise computations is taken
    from the ``nthreads`` value stored in ``fit`` by ``glmQLFit``, so ``diffSplice`` has no separate
    ``nthreads`` argument.

    The feature-level tests are converted into genewise tests by adjusting the p-values for the same
    gene by Simes method. As an alternative, the feature-level tests are also converted into
    genewise tests by adjusting the smallest p-value for each gene by Bonferroni's method.

    This function can be used on transcript level RNA-seq counts from Salmon, kallisto or RSEM, after
    pre-processing by ``catchSalmon`` (or ``catchKallisto`` or ``catchRSEM``), as described by
    Baldoni et al (2025). It can also be used on exon-level read counts.

    R documentation by Lizhong Chen, Yunshun Chen and Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(fit, DGEGLM):
        raise TypeError("fit must be a DGEGLM object produced by glmQLFit")
    if fit.get("s2.post") is None:
        raise ValueError("need to run glmQLFit before diffSplice")
    if fit.get("average.ql.dispersion") is None:
        raise NotImplementedError("legacy glmQLFit fits are not supported")
    if robust is None:
        robust = np.size(fit["df.prior"]) > 1
    if geneid is None:
        raise TypeError('argument "geneid" is missing, with no default')

    exon_genes = fit.get("genes")
    if exon_genes is None:
        exon_genes = pd.DataFrame({"ExonID": np.arange(1, fit.shape[0] + 1)}, index=fit.rownames)
    exon_genes = exon_genes.copy()
    if isinstance(geneid, str):
        genecolname = geneid
        geneid = list(exon_genes[genecolname])
    else:
        exon_genes["GeneID"] = list(geneid)
        genecolname = "GeneID"
        geneid = list(geneid)
    if exonid is not None:
        if isinstance(exonid, str):
            exoncolname = exonid
            exonid = list(exon_genes[exoncolname])
        else:
            exon_genes["ExonID"] = list(exonid)
            exoncolname = "ExonID"
            exonid = list(exonid)
    else:
        exoncolname = None

    isna = [k for k, g in enumerate(geneid) if g is None or (isinstance(g, float) and np.isnan(g))]
    for m, k in enumerate(isna):
        geneid[k] = f"NA{m + 1}"

    o = _order_ids(geneid) if exonid is None else _order_ids(geneid, exonid)
    exon_genes = exon_genes.iloc[o]
    geneid = [geneid[k] for k in o]
    fit = fit.subset(o, None)

    X, coef_names, _ = design_matrix(fit["design"], None)
    if X.shape[1] < 2:
        raise ValueError("Need at least two columns for design, usually the first is the intercept column")
    if contrast is None:
        if isinstance(coef, (list, tuple)):
            coef = coef[0]
        if coef is None:
            coef1 = X.shape[1]
        elif isinstance(coef, str):
            if coef_names is None or coef not in coef_names:
                raise ValueError("One or more named coef arguments do not match a column of the design matrix.")
            coef1 = coef_names.index(coef) + 1
        else:
            coef1 = int(coef) + 1
        coef_name = coef if isinstance(coef, str) else (coef_names[coef1 - 1] if coef_names is not None else str(coef1))
        beta = mat(fit["coefficients"])
        design = X
        design_cols = coef_names
    else:
        contrast = np.asarray(contrast, dtype=np.float64)
        if contrast.ndim == 2:
            contrast = contrast[:, 0]
        reform = contrastAsCoef(X, contrast.reshape(-1, 1), first=True)
        coef1 = 1
        nz = contrast != 0
        coef_name = " ".join(f"{_r.num2str(v)}*{n}" for v, n in zip(contrast[nz], np.asarray(coef_names, dtype=object)[nz]))
        design = reform["design"]
        design_cols = reform["colnames"]
        refit = _glmFit_default(fit["counts"], design=pd.DataFrame(design, columns=design_cols),
                                dispersion=fit["dispersion"], offset=fit["offset"], weights=fit.get("weights"),
                                technical_weights=fit.get("technical.weights"),
                                biological_weights=fit.get("biological.weights"))
        beta = mat(refit["coefficients"])

    nexons = fit.shape[0]
    gene_nexons_d, ugenes = _r.rowsum(np.ones(nexons), geneid)
    gene_nexons = gene_nexons_d[:, 0]
    if verbose:
        print("Total number of exons: ", nexons)
        print("Total number of genes: ", len(gene_nexons))
        print("Number of genes with 1 exon: ", int(np.sum(gene_nexons == 1)))
        print("Mean number of exons in a gene: ", int(np.round(_r.mean(gene_nexons))))
        print("Max number of exons in a gene: ", int(np.max(gene_nexons)))

    gene_keep = gene_nexons > 1
    ngenes = int(gene_keep.sum())
    if ngenes == 0:
        raise ValueError("No genes with more than one exon")
    exon_keep = np.repeat(gene_keep, gene_nexons.astype(np.int64))
    nexons = int(exon_keep.sum())
    keep_idx = np.nonzero(exon_keep)[0]
    geneid = [geneid[k] for k in keep_idx]
    exon_genes = exon_genes.iloc[keep_idx]
    gene_nexons = gene_nexons[gene_keep]
    fit = fit.subset(keep_idx, None)
    beta = np.asfortranarray(beta[keep_idx, :])

    ym = mat(fit["counts"])
    weights = compressWeights(ym, fit.get("weights"))
    offset = compressOffsets(ym, fit["offset"])
    dispersion = compressDispersions(ym, fit["dispersion"] / fit["average.ql.dispersion"])

    gene_lastexon = _r.cumsum(gene_nexons)
    gene_firstexon = gene_lastexon - gene_nexons + 1
    gnames = [geneid[int(k) - 1] for k in gene_firstexon]

    res = _C.fit_diff_splice(ym, offset.to_c(), dispersion.to_c(), weights.to_c(), fit.get("technical.weights"),
                             fit.get("biological.weights"), np.asfortranarray(design), int(coef1),
                             gene_nexons.astype(np.int32), (gene_firstexon - 1).astype(np.int32), int(nexons_approx),
                             250, 1e-6, beta, np.asarray(fit["deviance"], dtype=np.float64), int(fit.get("nthreads") or 1))

    exon_ids = list(exon_genes["ExonID"]) if "ExonID" in exon_genes else list(exon_genes.index)
    exon_LR = res["exon.LR"]
    exon_coef = res["exon.coef"]
    gene_LR = res["gene.LR"]

    exon_df_test = np.ones(nexons)
    gene_df_test = gene_nexons - 1
    adj = np.asarray(fit["df.residual.adj"], dtype=np.float64)
    exon_stat = np.column_stack([adj, np.asarray(fit["deviance.adj"], dtype=np.float64),
                                 adj * np.asarray(fit["AveLogCPM"], dtype=np.float64)])
    gene_sum, _ = _r.rowsum(exon_stat, geneid)
    gene_df_residual = gene_sum[:, 0]
    gene_s2 = gene_sum[:, 1] / gene_sum[:, 0]
    gene_AveLogCPM = gene_sum[:, 2] / gene_sum[:, 0]

    covariate = gene_AveLogCPM if np.size(fit["s2.prior"]) > 1 else None
    squeeze = squeezeVar(gene_s2, df=gene_df_residual, covariate=covariate, robust=robust)
    gene_df_total = gene_df_residual + squeeze["df.prior"]
    gene_df_total = np.minimum(gene_df_total, _r.rsum(gene_df_residual))
    gene_s2_post = squeeze["var.post"]
    gene_s2_prior = squeeze["var.prior"]

    reps = gene_nexons.astype(np.int64)
    exon_df_total = np.repeat(gene_df_total, reps)
    exon_s2_post = np.repeat(gene_s2_post, reps)
    exon_F = exon_LR / exon_df_test / exon_s2_post
    gene_F = gene_LR / gene_df_test / gene_s2_post
    exon_t = _r.sqrt(np.maximum(0, exon_F)) * _r.sign(exon_coef)
    exon_p = _r.pf(exon_F, exon_df_test, exon_df_total, lower_tail=False)
    gene_p = _r.pf(gene_F, gene_df_test, gene_df_total, lower_tail=False)

    col = [coef_name]
    out = DGEDTU()
    out["design"] = design
    out["comparison"] = (design_cols[coef1 - 1] if design_cols is not None else None)
    out["genes"] = exon_genes
    out["genecolname"] = genecolname
    if exoncolname is not None:
        out["exoncolname"] = exoncolname
    out["coefficients"] = frame(exon_coef.reshape(-1, 1), exon_ids, col)
    out["nexons.approx"] = nexons_approx
    out["t"] = frame(exon_t.reshape(-1, 1), exon_ids, col)
    out["p.value"] = frame(exon_p.reshape(-1, 1), exon_ids, col)
    out["gene.df.prior"] = squeeze["df.prior"]
    out["gene.df.residual"] = pd.Series(gene_df_residual, index=gnames)
    out["gene.df.total"] = pd.Series(gene_df_total, index=gnames)
    out["gene.s2"] = pd.Series(gene_s2, index=gnames)
    out["gene.s2.prior"] = gene_s2_prior
    out["gene.s2.post"] = gene_s2_post
    out["gene.F"] = frame(gene_F.reshape(-1, 1), gnames, col)
    out["gene.F.p.value"] = frame(gene_p.reshape(-1, 1), gnames, col)

    last = gene_lastexon.astype(np.int64) - 1
    first = gene_firstexon.astype(np.int64) - 1
    notfirst = np.ones(nexons, dtype=bool)
    notfirst[first] = False
    isgenelevel = [bool(exon_genes[c].duplicated().to_numpy()[notfirst].all()) for c in exon_genes.columns]
    gene_genes = exon_genes.iloc[last, [k for k, b in enumerate(isgenelevel) if b]].copy()
    gene_genes.index = [str(v) for v in gene_genes[genecolname]]
    gene_genes["NExons"] = gene_nexons
    out["gene.genes"] = gene_genes
    out["gene.firstexon"] = pd.Series(gene_firstexon, index=gnames)
    out["gene.lastexon"] = pd.Series(gene_lastexon, index=gnames)

    g = np.repeat(np.arange(1, ngenes + 1, dtype=np.float64), reps)
    o2 = _r.order(g, exon_p)
    p = exon_p[o2]
    q = np.ones(nexons)
    cq = _r.cumsum(q)
    r = cq - np.repeat(cq[last] - gene_nexons, reps)
    pp = p * np.repeat(gene_nexons, reps) / r
    oo = _r.order(-g, pp, decreasing=True)
    simes = pp[oo][last]
    bonf = np.minimum(p[first] * (gene_nexons - 1), 1)
    out["gene.simes.p.value"] = frame(simes.reshape(-1, 1), gnames, col)
    out["gene.bonferroni.p.value"] = frame(bonf.reshape(-1, 1), gnames, col)
    return out


def _order_ids(*keys):
    """R order() on id vectors (numeric order for numbers, C locale for strings)."""
    if all(all(isinstance(v, numbers.Number) and not isinstance(v, bool) for v in k) for k in keys):
        return _r.order(*[np.asarray(k, dtype=np.float64) for k in keys])
    return _r.order_str(*keys)


def topSplice(fit, coef=None, test="F", number=10, FDR=1, sort_by="p", treat_lfc=0):
    """Top Table of Differential Transcript Usage

    Top table of results from a ``diffSplice`` differential transcript usage analysis.

    Parameters
    ----------
    fit : DGEDTU
        Output of ``diffSplice``. An object without its gene-level results raises ``ValueError``.
    coef : int or str, optional
        The coefficient (column) of ``fit`` for which differential usage is assessed. ``diffSplice``
        output has a single column, so this argument is not used.
    test : {"F", "simes", "t"}, default "F"
        Which statistical test to apply. ``"F"`` gives F-tests for each gene. ``"t"`` gives t-tests for
        each transcript or exon. ``"simes"`` gives genewise p-values derived from the t-tests after
        Simes adjustment for each gene.
    number : int or float, default 10
        Maximum number of rows to output; ``numpy.inf`` for all.
    FDR : float, default 1
        Only show exons or genes with false discovery rate at most this cutoff.
    sort_by : {"p", "logFC", "NExons", "none"}, default "p"
        Which column to sort results by. ``"logFC"`` is only available if ``test="t"`` and
        ``"NExons"`` is only available if ``test="simes"`` or ``test="F"``.
    treat_lfc : float, default 0
        A treat-style log2-fold-change threshold for ``test="t"``. Only 0 is available; a positive
        value raises ``NotImplementedError``.

    Returns
    -------
    DataFrame
        The annotation columns found in ``fit`` (``fit["genes"]`` for ``test="t"``,
        ``fit["gene.genes"]`` otherwise) plus the following columns:

        ``logFC``
            log2-fold change of transcript vs other transcripts for the same gene (if ``test="t"``).
        ``t``
            Moderated t-statistic (if ``test="t"``).
        ``F``
            Moderated F-statistic (if ``test="F"``).
        ``P.Value``
            p-value.
        ``FDR``
            False discovery rate bound, computed by the Benjamini & Hochberg method.

    Notes
    -----
    This function ranks results from a differential transcript usage analysis conducted by
    ``diffSplice``. The output from ``diffSplice`` includes results at the gene level and results at
    the isoform level. The isoform-level results can be transcripts, if a DTU analysis was done, or
    can be exons and exon-junctions, if a DEU analysis was done.

    Choosing ``test="F"`` or ``test="simes"`` gives gene-level results, where the null hypothesis is
    that all transcripts or exons are used equally. Choosing ``test="t"`` gives transcript or exon
    level results.

    The F-statistic option chosen by ``test="F"`` conducts an ANOVA-like F-test of differential usage
    for each gene. The Simes method chosen by ``test="simes"`` processes the isoform-level p-values
    for each gene to give an overall call of differential usage for that gene. The Simes p-values are
    likely to be more powerful than F-tests when only one transcript of the gene is differentially
    expressed. The F-tests are likely to be more powerful than Simes when there are multiple
    transcripts of the same gene differentially expressed in different directions.

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    if fit.get("gene.genes") is None or "NExons" not in fit["gene.genes"]:
        raise ValueError("fit should be fit object produced by diffSplice")
    test = {"f": "F"}.get(test, test)
    if test not in ("simes", "F", "t"):
        raise ValueError("test must be one of 'simes', 'F', 't'")
    if test == "t" and treat_lfc > 0:
        raise NotImplementedError("topSplice treat test is not ported")
    if sort_by not in ("p", "none", "logFC", "NExons"):
        raise ValueError("invalid sort_by")
    if sort_by == "logFC" and test != "t":
        raise ValueError("Sorting by logFC only available with t test")
    if sort_by == "NExons" and test == "t":
        raise ValueError("Sorting by NExons only available with gene-level tests")

    if test == "t":
        out = fit["genes"].copy()
        out["logFC"] = np.asarray(fit["coefficients"]).reshape(-1)
        out["t"] = np.asarray(fit["t"]).reshape(-1)
        out["P.Value"] = np.asarray(fit["p.value"]).reshape(-1)
    elif test == "F":
        out = fit["gene.genes"].copy()
        out["F"] = np.asarray(fit["gene.F"]).reshape(-1)
        out["P.Value"] = np.asarray(fit["gene.F.p.value"]).reshape(-1)
    else:
        out = fit["gene.genes"].copy()
        out["P.Value"] = np.asarray(fit["gene.simes.p.value"]).reshape(-1)
    out["FDR"] = _r.p_adjust(out["P.Value"].to_numpy(), "BH")

    if FDR < 1:
        out = out[out["FDR"].to_numpy() <= FDR]
    number = min(number, len(out))
    if number <= 1:
        return out
    if sort_by == "p":
        o = _r.order(out["P.Value"].to_numpy())
    elif sort_by == "logFC":
        o = _r.order(np.abs(out["logFC"].to_numpy()), decreasing=True)
    elif sort_by == "NExons":
        o = _r.order(out["NExons"].to_numpy(dtype=np.float64), -out["P.Value"].to_numpy(), decreasing=True)
    else:
        o = np.arange(len(out))
    return out.iloc[o[:number]]
