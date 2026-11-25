"""decideTests for DGELRT objects (R: edgeR decidetestsDGE.R)."""

import numpy as np
import pandas as pd

from . import _r
from .classes import DGELRT, TestResults


def decideTests(object, adjust_method="BH", p_value=0.05, lfc=0):
    """Multiple Testing Across Genes and Contrasts

    Identify which genes are significantly differentially expressed from an edgeR test object
    containing p-values and test statistics.

    Parameters
    ----------
    object : DGELRT
        Test results from ``glmQLFTest`` (log-fold-changes ``logFC``) or ``binQLFTest`` (log-odds
        ratios ``logOR``), from which p-values and log-fold-changes can be extracted.
    adjust_method : str, default "BH"
        P-value adjustment method: ``"none"``, ``"BH"``, ``"fdr"``, ``"BY"``, ``"holm"``,
        ``"hochberg"``, ``"hommel"`` or ``"bonferroni"``, as R's ``p.adjust``.
    p_value : float, default 0.05
        Value between 0 and 1 giving the required family-wise error or false discovery rate.
    lfc : float, default 0
        Minimum absolute log2-fold-change required.

    Returns
    -------
    TestResults
        A single-column DataFrame of nullable integers (named after ``object["comparison"]``), with
        elements -1, 0 or 1 indicating whether each gene is classified as significantly
        down-regulated, not significant or significantly up-regulated for the comparison contained in
        ``object``. To be considered significant, genes need to have adjusted p-value below
        ``p_value`` and log2-fold-change at least ``lfc`` in absolute value. Missing p-values give
        missing calls. ``.summary()`` gives the number of genes in each class.

        If ``object`` contains F-tests for multiple coefficients or contrasts, then the genes are
        simply classified as significant (1) or not significant (0). In this case, the
        log2-fold-change threshold ``lfc`` has to be achieved by at least one of the contrasts for a
        gene to be significant.

    Notes
    -----
    This function applies a multiple testing procedure and significance level cutoff to the genewise
    tests contained in an edgeR test object.

    The function can optionally apply a logFC cutoff as well as a p-value or FDR cutoff (although
    logFC cutoffs are not recommended, see below). If the statistical tests are on 1 degree of
    freedom, then the logFC cutoff is applied to the absolute coefficient or contrast. If the
    statistical tests are on more than 1 degree of freedom, then the logFC cutoff will be satisfied
    if any of the coefficients or contrasts that define the test are greater than the cutoff.

    Although this function enables users to set p-value and logFC cutoffs simultaneously, this
    combination criterion is not recommended. logFC cutoffs tend to favor low expressed genes and
    thereby reduce rather than increase biological significance. Unless the fold changes and p-values
    are very highly correlated, the addition of a fold change cutoff can also increase the
    family-wise error rate or false discovery rate above the nominal level. Users wanting to use fold
    change thresholding should consider testing with a positive ``lfc`` in ``glmQLFTest`` (TREAT)
    instead and leaving ``lfc`` at the default value when using ``decideTests``.

    R documentation by Davis McCarthy, Gordon Smyth and the edgeR team; adapted for edgerpy.
    """
    if not isinstance(object, DGELRT):
        raise TypeError("decideTests() needs a DGELRT")
    tab = object["table"]

    # Apply multiple testing
    p = _r.p_adjust(tab["PValue"].to_numpy(dtype=np.float64), method=adjust_method)
    # R: as.integer(p < p.value); NA kept as NaN until the end (R's NA_integer_)
    isDE = np.where(np.isnan(p), np.nan, (p < p_value).astype(np.float64))

    # Extract logFC
    col = "logOR" if object.get("dispersion") is None else "logFC"
    FTest = col not in tab

    if FTest:
        # with multiple contrasts, apply the lfc threshold to the maximum logFC
        if lfc > 0:
            coef = tab[[c for c in tab.columns if c.startswith(col)]].to_numpy(dtype=np.float64)
            # R: rowSums(abs(logFC) >= lfc) == 0 is NA (no assignment) when a row has NA
            SmallFC = (np.sum(np.abs(coef) >= lfc, axis=1) == 0) & ~np.isnan(coef).any(axis=1)
            isDE[SmallFC] = 0
    else:
        # with a single contrast, apply directionality and the lfc threshold
        logFC = tab[col].to_numpy(dtype=np.float64)
        isDE[(isDE == 1) & (logFC < 0)] = -1
        isDE[np.abs(logFC) < lfc] = 0

    comparison = object["comparison"]
    comparison = [comparison] if isinstance(comparison, str) else list(comparison)
    calls = pd.array([None if np.isnan(v) else int(v) for v in isDE], dtype="Int32")
    out = TestResults({"-".join(reversed(comparison)): calls}, index=tab.index)
    if FTest:
        out.levels, out.labels = (0, 1), ("NotSig", "Sig")
    else:
        out.levels, out.labels = (-1, 0, 1), ("Down", "NotSig", "Up")
    return out
