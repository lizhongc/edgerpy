"""topTags (R: topTags.R)."""

import numpy as np
import pandas as pd

from . import _r
from .classes import TopTags

_FWER = ("holm", "hochberg", "hommel", "bonferroni")
_FDR = ("BH", "BY", "fdr")


def topTags(object, n=10, adjust_method="BH", sort_by="PValue", p_value=1):
    """Table of the Top Differentially Expressed Genes/Tags

    Extracts the most differentially expressed genes (or sequence tags) from a test object, ranked
    either by p-value or by absolute log-fold-change.

    Parameters
    ----------
    object : DGELRT
        Test statistics and p-values, from ``glmQLFTest`` or ``binQLFTest``.
    n : int or float, default 10
        Maximum number of genes/tags to return. ``n=np.inf`` returns all.
    adjust_method : str, default "BH"
        Method used to adjust p-values for multiple testing: ``"BH"``, ``"BY"``, ``"fdr"``,
        ``"holm"``, ``"hochberg"``, ``"hommel"``, ``"bonferroni"`` or ``"none"``, as R's
        ``p.adjust``.
    sort_by : {"PValue", "logFC", "none"}, default "PValue"
        Sort method: ``"PValue"`` for p-value, ``"logFC"`` for absolute log-fold change (log-odds
        ratio for binomial tests) or ``"none"`` for no sorting.
    p_value : float, default 1
        Cutoff value for adjusted p-values. Only tags with adjusted p-values equal or lower than
        specified are returned.

    Returns
    -------
    TopTags
        With components:

        ``table``
            A DataFrame containing differential expression results for the top genes in sorted
            order. The number of rows is the smaller of ``n`` and the number of genes with adjusted
            p-value less than or equal to ``p_value``. It includes all the annotation columns from
            ``object["genes"]`` and all statistic columns from ``object["table"]``, plus ``FDR``
            (false discovery rate, when ``adjust_method`` is ``"BH"``, ``"BY"`` or ``"fdr"``) or
            ``FWER`` (family-wise error rate, when ``adjust_method`` is ``"holm"``, ``"hochberg"``,
            ``"hommel"`` or ``"bonferroni"``).
        ``adjust.method``
            The method used to adjust p-values for multiple testing.
        ``comparison``
            The coefficient or contrast being tested.
        ``test``
            The name of the test (``"glm"``).

        If no gene passes ``p_value``, an empty DataFrame is returned instead.

    Notes
    -----
    This function is closely analogous to the ``topTable`` function in the limma package. It accepts
    a test statistic object created by ``glmQLFTest`` or ``binQLFTest`` and extracts a readable
    DataFrame of the most differentially expressed genes. The DataFrame collates the annotation and
    differential expression statistics for the top genes. It is wrapped in a ``TopTags`` output
    object that records the test statistic used and the multiple testing adjustment method.

    ``topTags`` permits ranking by fold-change but the authors do not recommend fold-change ranking
    or fold-change cutoffs for routine RNA-seq analysis. The p-value ranking is intended to be more
    biologically meaningful, especially if the p-values were computed with a positive ``lfc``
    (TREAT).

    The terms 'tag' and 'gene' are used synonymously on this page and refer to the rows of
    ``object``. In general, the rows might be genes, sequence tags, transcripts, exons or whatever
    type of genomic feature is appropriate for the analysis at hand.

    R documentation by Mark Robinson, Davis McCarthy, Yunshun Chen, Gordon Smyth; adapted for
    edgerpy.
    """
    tab0 = object.get("table")
    if tab0 is None:
        raise ValueError("Need to run exactTest or glmLRT first")
    test = "glm"
    multiple = tab0.shape[1] > 4
    n = min(n, len(tab0))
    if n < 1:
        raise ValueError("No rows to output")
    if adjust_method not in _FWER + _FDR + ("none",):
        raise ValueError(f"invalid adjust_method {adjust_method!r}")
    if sort_by == "p.value":
        sort_by = "PValue"
    if sort_by not in ("none", "PValue", "logFC"):
        raise ValueError(f"invalid sort_by {sort_by!r}")

    if multiple:
        alfc = np.abs(tab0.iloc[:, 0].to_numpy(dtype=np.float64))
    else:
        col = "logFC" if "logFC" in tab0 else "logOR"
        alfc = np.abs(tab0[col].to_numpy(dtype=np.float64))
    pv = tab0["PValue"].to_numpy(dtype=np.float64)
    if sort_by == "logFC":
        o = _r.order(alfc, decreasing=True)
    elif sort_by == "PValue":
        o = _r.order(pv, -alfc)
    else:
        o = np.arange(len(tab0))
    tab = tab0.iloc[o].copy()
    adj = _r.p_adjust(pv, adjust_method)
    if adjust_method != "none":
        tab["FWER" if adjust_method in _FWER else "FDR"] = adj[o]
    genes = object.get("genes")
    if genes is not None:
        if not isinstance(genes, pd.DataFrame):
            genes = pd.DataFrame({"ID": list(genes)})
        rn = tab.index
        g = genes.iloc[o].copy()
        g.index = rn
        tab = pd.concat([g, tab], axis=1)
    if p_value < 1:
        sig = adj[o] <= p_value
        sig[np.isnan(sig)] = False
        tab = tab[sig]
    if len(tab) < n:
        n = len(tab)
    if n < 1:
        return pd.DataFrame()
    return TopTags(table=tab.iloc[:n], **{"adjust.method": adjust_method,
                                          "comparison": object.get("comparison"), "test": test})
