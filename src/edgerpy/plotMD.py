"""plotMD for edgeR objects and limma's plotWithHighlights
(R: edgeR plotMD.R; limma plotWithHighlights.R)."""

import numbers

import numpy as np
import pandas as pd

from . import _r
from ._rplot import RPlot
from .aveLogCPM import cpm
from .classes import DGEBIN, DGEGLM, DGELRT, DGEList, TestResults
from .decideTests import decideTests
from .helpers import CompressedMatrix, mat

_MISSING = object()


def plotMD(object, column=None, coef=None, xlab=None, ylab=None, main=None, status=None, zero_weights=False,
           prior_count=3, contrast=0, adjust_method="BH", p_value=0.05, ax=None, **kwargs):
    """Mean-Difference Plot of Count Data

    Creates a mean-difference plot (aka MA plot) with color coding for highlighted points.

    Parameters
    ----------
    object : DGEList, DGEGLM, DGEBIN or DGELRT
        The data or fit to plot. Other objects raise ``TypeError``.
    column : int or str, optional
        Column of ``object`` to be plotted, by name or 0-based position: for a DGEList, the sample
        (default the first); for a DGEGLM or DGEBIN, the coefficient (default the last). Not used for
        a DGELRT.
    coef : int or str, optional
        Alternative to ``column`` for a DGEGLM or DGEBIN. If given, ``column`` is ignored.
    xlab : str, optional
        Label for the x-axis. Defaults to ``"Average log CPM (this sample and others)"`` for a
        DGEList and to ``"Average log CPM"`` otherwise.
    ylab : str, optional
        Label for the y-axis. Defaults to ``"log-ratio (this sample vs others)"`` for a DGEList, to
        ``"log-odds-ratio"`` for a DGEBIN or a binomial test result, and to ``"log-fold-change"``
        otherwise.
    main : str, optional
        Title for the plot. Defaults to the name of the column plotted, or for a DGELRT to its
        ``comparison`` (``"Contrast <name>"`` when several contrasts were tested).
    status : array-like, optional
        The status of each gene, of the same length as the number of genes. Defaults to
        ``object["genes"]["Status"]`` if present. If ``None`` for a DGEList, DGEGLM or DGEBIN, all
        points are plotted in the default color, symbol and size. If ``None`` for a DGELRT,
        ``decideTests`` is run to determine the status of all the genes: the up-regulated DE genes
        are highlighted in red and the down-regulated in blue.
    zero_weights : bool, default False
        Whether genes with zero or negative weights should be plotted. Not used for a DGELRT.
    prior_count : float, default 3
        The average prior count to be added to each observation for a DGEList. Larger values produce
        more shrinkage.
    contrast : int or str, default 0
        Which log-fold-change to plot when a DGELRT tested several contrasts, by name or 0-based
        position.
    adjust_method : str, default "BH"
        p-value adjustment method passed to ``decideTests``. Only used for a DGELRT when ``status`` is
        ``None``.
    p_value : float, default 0.05
        The desired size of the test, passed to ``decideTests``. Only used for a DGELRT when
        ``status`` is ``None``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw in. Defaults to a new 7 x 7 inch figure, like a new R graphics device.
    **kwargs
        Passed to limma's ``plotWithHighlights``, which draws the plot: ``values``, ``hl_pch``,
        ``hl_col``, ``hl_cex``, ``legend``, ``bg_pch``, ``bg_col``, ``bg_cex``, and plot arguments.

    Returns
    -------
    None
        A plot is drawn.

    Notes
    -----
    A mean-difference plot (MD-plot) is a plot of log fold changes (differences) versus average log
    values (means). The history of mean-difference plots and MA-plots is reviewed in Ritchie et al
    (2015).

    For a DGEList, a between-sample MD-plot is produced. Counts are first converted to log2-CPM
    values. An artificial sample is produced by averaging all the samples other than the sample
    specified. A mean-difference plot is then produced from the specified sample and the artificial
    sample. This procedure reduces to an ordinary mean-difference plot when there are just two
    samples in total.

    For a DGEGLM, the plot is a fitted model MD-plot in which the estimated coefficient is on the
    y-axis and the average logCPM value is on the x-axis. For a DGEBIN (a binomial fit from
    ``binFit`` or ``binQLFit``), the fitted log-odds-ratio coefficient is on the y-axis and the
    average logCPM of the total coverage is on the x-axis. For a DGELRT, the MD-plot displays the
    logFC (or logOR) against the logCPM values from the results table.

    The ``status`` vector can correspond to any grouping of the genes that is of interest. For a
    fitted model object, it is often used to indicate statistical significance, so that
    differentially expressed points are highlighted. The status can be included as the column
    ``Status`` of ``object["genes"]`` instead of being passed as an argument.

    By default, highlighted points are larger than background points and a different color is used
    for each distinct highlighted value. If ``values`` is not given, the most common value of
    ``status`` is assumed to correspond to background points and all other values are highlighted.

    The plot follows R's base graphics: plotted values, axis limits, ticks, labels, colours and
    legend are those of R.

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    _plotMD(object, column=column, coef=coef, xlab=xlab, ylab=ylab, main=main, status=status,
            zero_weights=zero_weights, prior_count=prior_count, contrast=contrast, adjust_method=adjust_method,
            p_value=p_value, **kwargs).render(ax)


def _col_index(column, names):
    """A name or 0-based integer -> 0-based position."""
    if isinstance(column, str):
        if column not in names:
            raise ValueError(f"column {column!r} not found")
        return list(names).index(column)
    return int(column)


def _genes_status(object):
    genes = object.get("genes")
    if isinstance(genes, pd.DataFrame) and "Status" in genes:
        return genes["Status"]
    return None


def _plotMD(object, column=None, coef=None, xlab=None, ylab=None, main=None, status=None, zero_weights=False,
            prior_count=3, contrast=0, adjust_method="BH", p_value=0.05, **kwargs):
    if status is None:
        status = _genes_status(object)

    if isinstance(object, DGEList):
        names = list(object["counts"].columns)
        nlib = len(names)
        if nlib < 2:
            raise ValueError("Need at least two columns")
        column = _col_index(0 if column is None else column, names)
        logCPM = mat(cpm(object, log=True, prior_count=prior_count))
        others = np.delete(np.arange(nlib), column)
        AveOfOthers = _r.rowMeans(logCPM[:, others], na_rm=True)
        Diff = logCPM[:, column] - AveOfOthers
        Mean = (logCPM[:, column] + AveOfOthers) / 2
        if not zero_weights and object.get("weights") is not None:
            w = _as_matrix(object["weights"])[:, column]
            Diff[np.isnan(w) | (w <= 0)] = np.nan
        return plotWithHighlights(
            Mean, Diff, status=status, main=names[column] if main is None else main,
            xlab="Average log CPM (this sample and others)" if xlab is None else xlab,
            ylab="log-ratio (this sample vs others)" if ylab is None else ylab, **kwargs)

    if isinstance(object, (DGEGLM, DGEBIN)):
        beta = object["coefficients"]
        names = list(beta.columns)
        if coef is not None:
            column = coef
        column = _col_index(len(names) - 1 if column is None else column, names)
        if object.get("AveLogCPM") is None:
            raise ValueError("AveLogCPM component is absent.")
        logFC = mat(beta)[:, column].copy()
        if not zero_weights and object.get("weights") is not None:
            w = _as_matrix(object["weights"])[:, column]
            logFC[np.isnan(w) | (w <= 0)] = np.nan
        default_ylab = "log-odds-ratio" if isinstance(object, DGEBIN) else "log-fold-change"
        return plotWithHighlights(
            np.asarray(object["AveLogCPM"], dtype=np.float64), logFC, status=status,
            main=names[column] if main is None else main,
            xlab="Average log CPM" if xlab is None else xlab, ylab=default_ylab if ylab is None else ylab, **kwargs)

    if isinstance(object, DGELRT):
        tab = object["table"]
        ylab = "log-fold-change" if ylab is None else ylab
        logFC = tab["logFC"] if "logFC" in tab else None
        if logFC is None:
            logFC = tab["logOR"] if "logOR" in tab else None
            if logFC is not None and ylab == "log-fold-change":
                ylab = "log-odds-ratio"
        FTest = logFC is None
        if status is None:
            status = decideTests(object, adjust_method=adjust_method, p_value=p_value)
        if main is None:
            comparison = object["comparison"]
            main = comparison if isinstance(comparison, str) else list(comparison)
        # several contrasts (F-test): `contrast` selects the logFC.* (or logOR.*) column
        if FTest:
            prefix = "logOR" if object.get("dispersion") is None else "logFC"
            cols = [c for c in tab.columns if c.startswith(prefix)]
            k = _col_index(contrast, [c[len(prefix) + 1:] for c in cols])
            if not 0 <= k < len(cols):
                raise ValueError("Selected contrast does not exist.")
            logFC = tab[cols[k]]
            if prefix == "logOR" and ylab == "log-fold-change":
                ylab = "log-odds-ratio"
            main = "Contrast " + cols[k].replace(prefix + ".", "", 1)
        return plotWithHighlights(
            tab["logCPM"].to_numpy(dtype=np.float64), logFC.to_numpy(dtype=np.float64), status=status, main=main,
            xlab="Average log CPM" if xlab is None else xlab, ylab=ylab, **kwargs)

    raise TypeError("plotMD() needs a DGEList, DGEGLM, DGEBIN or DGELRT")


def _status_strings(status):
    """R as.character(status): strings, None for NA."""
    out = []
    for v in (status.tolist() if hasattr(status, "tolist") else list(status)):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        elif isinstance(v, (bool, np.bool_)):
            out.append("TRUE" if v else "FALSE")
        elif isinstance(v, numbers.Number):
            out.append(_r.num2str(v))
        else:
            out.append(str(v))
    return out


def plotWithHighlights(x, y, status=None, values=None, hl_pch=_MISSING, hl_col=None, hl_cex=_MISSING,
                       legend="topright", bg_pch=16, bg_col="black", bg_cex=0.3, pch=None, col=None, cex=None,
                       **kwargs):
    """limma plotWithHighlights(): scatterplot with highlighted groups of points -> RPlot.

    `kwargs` (main, xlab, ylab, ...) go to the plot call, as R's `...`.
    """
    p = RPlot()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if isinstance(status, TestResults):
        if status.shape[1] > 1:
            raise ValueError("status has more than one column")
    elif status is not None and all(s is None for s in _status_strings(status)):
        status = None

    # no status information: plot all points normally
    if status is None:
        p.plot(x, y, pch=bg_pch, col=bg_col, cex=bg_cex, **kwargs)
        return p

    # values to highlight: from a TestResults (Up and Down, in red and blue), or every value
    # except the most frequent one, in decreasing order of frequency
    if values is None:
        if isinstance(status, TestResults):
            labels = {-1: "Down", 0: "NotSig", 1: "Up"}
            status = [None if v is pd.NA else labels.get(int(v)) for v in status.iloc[:, 0].tolist()]
            values = ["Up", "Down"]
            if hl_col is None:
                hl_col = ["red", "blue"]
        else:
            s = [v for v in _status_strings(status) if v is not None]
            counts = {}
            for v in sorted(s, key=lambda t: t.encode("utf-8")):      # table(): levels in C-locale order
                counts[v] = counts.get(v, 0) + 1
            ranked = sorted(counts, key=lambda v: -counts[v])           # stable, as R's sort(decreasing=TRUE)
            values = ranked[1:]

    status = _status_strings(status)
    values = _status_strings(values if isinstance(values, (list, tuple, np.ndarray, pd.Series)) else [values])

    # no values to highlight: plot all points normally
    nvalues = len(values)
    if nvalues == 0:
        p.plot(x, y, pch=bg_pch, col=bg_col, cex=bg_cex, **kwargs)
        return p

    # legacy names pch, col and cex for hl_pch, hl_col and hl_cex
    if hl_pch is _MISSING:
        hl_pch = 16 if pch is None else pch
    if hl_col is None and col is not None:
        hl_col = col
    if hl_cex is _MISSING:
        hl_cex = 1 if cex is None else cex

    # axes, then background points
    p.plot(x, y, type="n", **kwargs)
    bg = np.array([s not in values for s in status])
    nonhi = bool(bg.any())
    if nonhi:
        p.points(x[bg], y[bg], pch=_first(bg_pch), col=_first(bg_col), cex=_first(bg_cex))

    rep = lambda v: list(np.resize(np.asarray(v, dtype=object).reshape(-1), nvalues))
    hl_pch, hl_cex = rep(hl_pch), rep(hl_cex)
    hl_col = rep(list(range(int(nonhi) + 1, int(nonhi) + 1 + nvalues)) if hl_col is None else hl_col)

    # highlighted points; R's x[status == v] gives NA for NA status
    for i in range(nvalues):
        keep = [k for k, s in enumerate(status) if s is None or s == values[i]]
        na = np.array([status[k] is None for k in keep], dtype=bool)
        xs, ys = x[keep], y[keep]
        xs[na] = np.nan
        ys[na] = np.nan
        p.points(xs, ys, pch=hl_pch[i], cex=hl_cex[i], col=hl_col[i])

    # legend
    if isinstance(legend, (bool, np.bool_)):
        position = "topleft"
    else:
        position, legend = str(legend), True
    if legend:
        if nonhi:
            bg_value = list(dict.fromkeys(s for s, b in zip(status, bg) if b))
            bg_value = ["Other"] if len(bg_value) > 1 else ["NA" if v is None else v for v in bg_value]
            labels = bg_value + values
            lpch = [bg_pch] + hl_pch
            lcol = [bg_col] + hl_col
            lcex = [bg_cex] + hl_cex
        else:
            labels, lpch, lcol, lcex = values, hl_pch, hl_col, hl_cex
        lcex = [0.5 + 0.8 * (c - 0.5) if c > 0.5 else c for c in lcex]
        p.legend(position, legend=labels, pch=lpch, col=lcol, cex=0.9, pt_cex=lcex)
    return p


def _as_matrix(w):
    """R as.matrix(object$weights), also for a CompressedMatrix."""
    return np.asarray(w.as_matrix() if isinstance(w, CompressedMatrix) else mat(w), dtype=np.float64)


def _first(v):
    return v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
