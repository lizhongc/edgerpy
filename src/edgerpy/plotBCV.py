"""plotBCV (R: edgeR plotBCV.R, getDispersion.R)."""

import numpy as np

from . import _r
from ._rplot import RPlot
from .aveLogCPM import _aveLogCPM_default
from .classes import DGEList
from .DGEList import getOffset
from .helpers import mat


def plotBCV(y, xlab="Average log CPM", ylab="Biological coefficient of variation", pch=16, cex=0.2,
            col_common="red", col_trend="blue", col_tagwise="black", ax=None, **kwargs):
    """Plot Biological Coefficient of Variation

    Plot the genewise biological coefficient of variation (BCV) against gene abundance (in log2
    counts per million).

    Parameters
    ----------
    y : DGEList
        With dispersions, for example from ``estimateDisp``. Raises ``TypeError`` for other objects
        and ``ValueError`` if ``y`` has no dispersions.
    xlab : str, default "Average log CPM"
        Label for the x-axis.
    ylab : str, default "Biological coefficient of variation"
        Label for the y-axis.
    pch : int, default 16
        The plotting symbol, as in R.
    cex : float, default 0.2
        Plot symbol expansion factor.
    col_common : str, default "red"
        Color of the line showing the common dispersion.
    col_trend : str, default "blue"
        Color of the line showing the dispersion trend.
    col_tagwise : str, default "black"
        Color of the points showing the genewise dispersions. Note that "tag" and "gene" are
        synonymous here.
    ax : matplotlib.axes.Axes, optional
        The axes to draw in. Defaults to a new 7 x 7 inch figure, like a new R graphics device.
    **kwargs
        Other arguments are passed to the plot, as R's ``...``.

    Returns
    -------
    None
        A plot is drawn.

    Notes
    -----
    The BCV is the square root of the negative binomial dispersion. This function displays the
    common, trended and genewise BCV estimates, those that ``y`` has. The x-axis is
    ``y["AveLogCPM"]``, computed from the counts if absent.

    The plot follows R's base graphics: plotted values, axis limits, ticks, labels, colours and
    legend are those of R.

    R documentation by Davis McCarthy, Yunshun Chen and Gordon Smyth; adapted for edgerpy.
    """
    _plotBCV(y, xlab=xlab, ylab=ylab, pch=pch, cex=cex, col_common=col_common, col_trend=col_trend,
             col_tagwise=col_tagwise, **kwargs).render(ax)


def _getDispersion(y):
    """edgeR getDispersion(): the most complex dispersion present -> (values, type) or (None, None)."""
    for key, kind in (("tagwise.dispersion", "tagwise"), ("trended.dispersion", "trended"),
                      ("common.dispersion", "common")):
        if key in y:
            return y[key], kind
    return None, None


def _plotBCV(y, xlab="Average log CPM", ylab="Biological coefficient of variation", pch=16, cex=0.2,
             col_common="red", col_trend="blue", col_tagwise="black", **kwargs):
    if not isinstance(y, DGEList):
        raise TypeError("y must be a DGEList.")

    # AveLogCPM, computed if not in y
    A = y.get("AveLogCPM")
    if A is None:
        A = _aveLogCPM_default(mat(y["counts"]), offset=getOffset(y))
    A = np.asarray(A, dtype=np.float64)

    # points that determine the y axis limits
    disp, kind = _getDispersion(y)
    if disp is None:
        raise ValueError("No dispersions to plot")
    disp = np.asarray(disp, dtype=np.float64).reshape(-1)
    if kind == "common":
        disp = np.resize(disp, A.size)

    p = RPlot()
    p.plot(A, _r.sqrt(disp), xlab=xlab, ylab=ylab, type="n", **kwargs)
    labels, cols, lty, pt = [], [], [], []
    if y.get("tagwise.dispersion") is not None:
        p.points(A, _r.sqrt(y["tagwise.dispersion"]), pch=pch, cex=cex, col=col_tagwise)
        labels.append("Tagwise"), cols.append(col_tagwise), lty.append(-1), pt.append(pch)
    if y.get("common.dispersion") is not None:
        p.abline(h=_r.sqrt(y["common.dispersion"]), col=col_common, lwd=2)
        labels.append("Common"), cols.append(col_common), lty.append(1), pt.append(-1)
    if y.get("trended.dispersion") is not None:
        o = _r.order(A)
        p.lines(A[o], _r.sqrt(np.asarray(y["trended.dispersion"], dtype=np.float64))[o], col=col_trend, lwd=2)
        labels.append("Trend"), cols.append(col_trend), lty.append(1), pt.append(-1)
    p.legend("topright", legend=labels, lty=lty, pch=pt, pt_cex=cex, lwd=2, col=cols)
    return p
