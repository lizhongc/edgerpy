"""plotMDS for DGEList and PCList (R: edgeR plotMDS.DGEList.R, plotMDS.PCList.R;
limma plotMDS.R for the distance matrix, the scaling and the drawing)."""

import warnings

import numpy as np
import pandas as pd

from . import _r
from ._rplot import RPlot
from .aveLogCPM import cpm
from .classes import MDS, DGEList, PCList
from .helpers import mat


def plotMDS(x, top=500, labels=None, pch=None, cex=1, dim_plot=(1, 2), gene_selection="pairwise", xlab=None,
            ylab=None, prior_count=2, plot=True, var_explained=True, ax=None, **kwargs):
    """Multidimensional Scaling Plot of Distances between Digital Gene Expression Profiles

    Plot samples on a two-dimensional scatterplot so that distances on the plot approximate the
    typical log2 fold changes between the samples (DGEList) or the differences in M-values, log2 odds,
    between the samples (PCList).

    Parameters
    ----------
    x : DGEList or PCList
        The counts. Other objects raise ``TypeError``.
    top : int, default 500
        Number of top genes (features) used to calculate pairwise distances.
    labels : list of str, optional
        Sample names or labels. Defaults to the column names of the counts, unless ``pch`` is given.
    pch : int or list, optional
        Plotting symbol or symbols, as in R. Ignored if ``labels`` is given.
    cex : float, default 1
        Expansion of the plot symbols or text labels.
    dim_plot : tuple of two ints, default (1, 2)
        Which two dimensions should be plotted. 1-based, as in R.
    gene_selection : {"pairwise", "common"}, default "pairwise"
        ``"pairwise"`` to choose the top genes separately for each pairwise comparison between the
        samples, or ``"common"`` to select the same genes for all comparisons.
    xlab : str, optional
        x-axis label.
    ylab : str, optional
        y-axis label.
    prior_count : float, default 2
        Average prior count added to the counts to shrink the log-fold-changes (DGEList: log-CPM;
        PCList: M-values, log2 odds) of low counts towards zero.
    plot : bool, default True
        If ``True``, a plot is drawn. If ``False``, nothing is drawn and only the MDS object is
        returned.
    var_explained : bool, default True
        If ``True``, the percentage of variation explained is included in the axis labels.
    ax : matplotlib.axes.Axes, optional
        The axes to draw in. Defaults to a new 7 x 7 inch figure, like a new R graphics device.
    **kwargs
        Other arguments, such as ``col``, are passed to the plot, as R's ``...``.

    Returns
    -------
    MDS
        A list with the components:

        ``eigen.values``
            Eigen values.
        ``eigen.vectors``
            Eigen vectors.
        ``var.explained``
            Proportion of variance explained by each dimension.
        ``dim.plot``
            Dimensions plotted.
        ``distance.matrix.squared``
            DataFrame of squared pairwise distances between the samples.
        ``top``
            Number of top genes used.
        ``gene.selection``
            Gene selection method.
        ``axislabel``
            Stem of the axis labels.
        ``x``
            x-coordinates of the plotted points.
        ``y``
            y-coordinates of the plotted points.

    Notes
    -----
    For a DGEList, the counts are converted to log-counts-per-million by ``cpm`` and distances
    between samples are based on log2 fold changes. For a PCList, the paired counts are converted to
    M-values (log2 odds) and distances are based on log2 odds ratios. Only the ``"logFC"`` method of
    edgeR's DGEList method is available; the ``"bcv"`` method is not ported.

    This function uses multidimensional scaling (MDS) to produce a principal coordinate (PCoA) or
    principal component (PCA) plot showing the relationships between the expression profiles of the
    samples. If ``gene_selection="common"``, or if ``top`` is equal to or greater than the number of
    genes, then a PCA plot is constructed from the ``top`` genes with largest standard deviations
    across the samples.

    If ``gene_selection="pairwise"`` and ``top`` is less than the number of genes, then a PCoA plot is
    produced and distances on the plot represent the *leading log2-fold-changes*. The leading
    log-fold-change between a pair of samples is defined as the root-mean-square average of the
    ``top`` largest log2-fold-changes between those two samples. The PCA and PCoA plots produced by
    ``gene_selection="common"`` and ``gene_selection="pairwise"``, respectively, use similar distance
    measures but the PCA plot uses the same genes throughout whereas the PCoA plot potentially selects
    different genes to distinguish each pair of samples. The pairwise choice is the default. It
    potentially gives better resolution than a PCA plot if different molecular pathways are relevant
    for distinguishing different pairs of samples.

    The number of genes (``top``) should roughly correspond to the number of differentially expressed
    genes with materially large fold-changes. The default setting of 500 genes is widely effective and
    suitable for routine use, but a smaller value might be chosen when the samples are distinguished
    by a specific focused molecular pathway. Very large values (greater than 1000) are not usually so
    effective.

    If ``pch`` is not given, each sample is represented by a text label, defaulting to the sample
    names. If ``pch`` is given, plotting symbols are used.

    The plot follows R's base graphics: plotted values, axis limits, ticks, labels and colours are
    those of R.

    R documentation by Yunshun Chen, Mark Robinson, Lizhong Chen, Di Wu and Gordon Smyth; adapted for
    edgerpy.
    """
    if isinstance(x, DGEList):
        M = cpm(x, log=True, prior_count=prior_count)
        mds = _plotMDS_default(M, top=top, dim_plot=dim_plot, gene_selection=gene_selection)
    elif isinstance(x, PCList):
        c1, c2 = x["counts"], x["counts2"]
        M = pd.DataFrame(_r.log2(mat(c1) + prior_count) - _r.log2(mat(c2) + prior_count),
                         index=c1.index, columns=c1.columns)
        mds = _plotMDS_default(M, top=top, dim_plot=dim_plot, gene_selection=gene_selection)
        mds["axislabel"] = "Leading logOR dim"
    else:
        raise TypeError("plotMDS() needs a DGEList or PCList")
    if plot:
        _plotMDS_draw(mds, labels=labels, pch=pch, cex=cex, xlab=xlab, ylab=ylab, var_explained=var_explained,
                      **kwargs).render(ax)
    return mds


def _plotMDS_default(x, top=500, dim_plot=(1, 2), gene_selection="pairwise"):
    """limma plotMDS.default(plot=FALSE) for a matrix of log-expression values -> MDS."""
    cn = list(x.columns) if isinstance(x, pd.DataFrame) else None
    x = np.asarray(mat(x), dtype=np.float64)
    nsamples = x.shape[1]
    if nsamples < 3:
        raise ValueError(f"Only {nsamples} columns of data: need at least 3")
    # remove rows with missing or Inf values
    bad = np.isfinite(x).sum(axis=1) < nsamples
    if bad.any():
        x = x[~bad]
    nprobes = x.shape[0]
    top = min(top, nprobes)

    dim_plot = np.array(list(dict.fromkeys(int(d) for d in dim_plot)), dtype=np.int32)
    if dim_plot.size != 2:
        raise ValueError("dim.plot must specify two dimensions to plot")
    ndim = int(dim_plot.max())
    if ndim < 2:
        raise ValueError("Need at least two dim.plot")
    if nsamples < ndim:
        raise ValueError("ndim is greater than number of samples")
    if nprobes < ndim:
        raise ValueError("ndim is greater than number of rows of data")
    choices = [c for c in ("pairwise", "common") if c.startswith(gene_selection)] if gene_selection else []
    if len(choices) != 1:
        raise ValueError("'arg' should be one of \"pairwise\", \"common\"")
    gene_selection = choices[0]

    # distance matrix from pairwise leading fold changes
    dd = np.zeros((nsamples, nsamples), order="F")
    if gene_selection == "pairwise":
        # mean of the top squared deviations for each pair of samples
        topindex = int(nprobes - top + 1)
        for i in range(1, nsamples):
            for j in range(i):
                d2 = _r.pow(x[:, i] - x[:, j], 2.0)
                dd[i, j] = _r.mean(_r.psort(d2, topindex)[topindex - 1:])
        axislabel = "Leading logFC dim"
    else:
        # same genes for all comparisons
        if nprobes > top:
            s = _r.rowMeans(_r.pow(x - _r.rowMeans(x)[:, None], 2.0))
            o = _r.order(s, decreasing=True)
            x = x[o[:int(top)]]
        for i in range(1, nsamples):
            dd[i, :i] = _r.colMeans(_r.pow(x[:, [i]] - x[:, :i], 2.0))
        axislabel = "Principal Component"

    # multidimensional scaling
    dd = dd + dd.T
    rm = _r.rowMeans(dd)
    dd = dd - rm[:, None]
    dd = dd.T - (rm - _r.mean(rm))[:, None]
    e = _r.eigen_symmetric(-dd / 2)

    lam = np.maximum(e["values"], 0)
    mds = MDS()
    mds["eigen.values"] = e["values"]
    mds["eigen.vectors"] = e["vectors"]
    mds["var.explained"] = lam / _r.rsum(lam)
    mds["dim.plot"] = dim_plot
    mds["distance.matrix.squared"] = pd.DataFrame(np.asfortranarray(dd), index=cn, columns=cn)
    mds["top"] = float(top)
    mds["gene.selection"] = gene_selection
    mds["axislabel"] = axislabel
    for key, d in zip(("x", "y"), dim_plot):
        mds[key] = e["vectors"][:, d - 1] * _r.sqrt(lam[d - 1])
        if lam[d - 1] < 1e-13:
            warnings.warn(f"dimension {d} is degenerate or all zero")
    return mds


def _plotMDS_draw(x, labels=None, pch=None, cex=1, xlab=None, ylab=None, var_explained=True, **kwargs):
    """limma plotMDS.MDS(): draw an MDS object -> RPlot."""
    if labels is None and pch is None:
        labels = list(x["distance.matrix.squared"].columns)
    dim_plot = x["dim.plot"]

    axislabel = x.get("axislabel") or "Principal Coordinate"
    if xlab is None:
        xlab = f"{axislabel} {dim_plot[0]}"
    if ylab is None:
        ylab = f"{axislabel} {dim_plot[1]}"
    if var_explained:
        perc = [_perc(x["var.explained"][d - 1]) for d in dim_plot]
        xlab = f"{xlab} ({perc[0]}%)"
        ylab = f"{ylab} ({perc[1]}%)"

    p = RPlot()
    if labels is None:
        # symbols instead of text
        p.plot(x["x"], x["y"], pch=pch, xlab=xlab, ylab=ylab, cex=cex, **kwargs)
    else:
        # text; widen the axes by an estimate of the label widths in plot coordinates
        labels = [str(v) for v in labels]
        radius = 0.01 * cex * np.array([len(v) for v in labels], dtype=np.float64)
        left, right = x["x"] - radius, x["x"] + radius
        p.plot(np.concatenate([left, right]), np.concatenate([x["y"], x["y"]]), type="n", xlab=xlab, ylab=ylab,
               **kwargs)
        p.text(x["x"], x["y"], labels=labels, cex=cex, col=kwargs.get("col"))
    return p


def _perc(v):
    """R paste0(round(v * 100)): round half to even, printed as an integer."""
    r = np.rint(v * 100)
    return "NaN" if np.isnan(r) else str(int(r))
