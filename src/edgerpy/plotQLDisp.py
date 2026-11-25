"""plotQLDisp (R: edgeR glmQLFTest.R)."""

import numpy as np

from . import _r
from ._rplot import RPlot
from .classes import DGEBIN, DGEGLM


def plotQLDisp(glmfit, xlab="Average Log2 CPM", ylab="Quarter-Root Mean Deviance", pch=16, cex=0.2,
               col_shrunk="red", col_trend="blue", col_raw="black", ax=None, **kwargs):
    """Plot the Quasi-Dispersion

    Plot the genewise quasi-dispersion against the gene abundance (in log2 counts per million).

    Parameters
    ----------
    glmfit : DGEGLM or DGEBIN
        A fit from ``glmQLFit`` or ``binQLFit``. Other objects raise ``TypeError``; a fit without
        quasi-dispersions (``s2.post``) or without ``AveLogCPM`` raises ``ValueError``.
    xlab : str, default "Average Log2 CPM"
        Label for the x-axis.
    ylab : str, default "Quarter-Root Mean Deviance"
        Label for the y-axis.
    pch : int, default 16
        The plotting symbol, as in R.
    cex : float, default 0.2
        Plot symbol expansion factor.
    col_shrunk : str, default "red"
        Color of the points representing the squeezed quasi-dispersions.
    col_trend : str, default "blue"
        Color of the line showing the dispersion trend.
    col_raw : str, default "black"
        Color of the points showing the unshrunk dispersions.
    ax : matplotlib.axes.Axes, optional
        The axes to draw in. Defaults to a new 7 x 7 inch figure, like a new R graphics device.
    **kwargs
        Other arguments are passed to the plot, as R's ``...``.

    Returns
    -------
    dict
        ``{"x": ..., "y": ...}``: the plotted average log2 CPM and the quarter-root raw
        quasi-dispersions, as R returns them.

    Notes
    -----
    This function displays the quarter-root of the quasi-dispersions for all genes, before and after
    shrinkage towards a trend. If ``glmfit`` was constructed without an abundance trend, the function
    instead plots a horizontal line (of color ``col_trend``) at the common value towards which
    dispersions are shrunk. The quarter-root transformation is applied to improve visibility for
    dispersions around unity.

    The plot follows R's base graphics: plotted values, axis limits, ticks, labels, colours and
    legend are those of R.

    R documentation by Aaron Lun, Davis McCarthy, Gordon Smyth and Yunshun Chen; adapted for
    edgerpy.
    """
    p, out = _plotQLDisp(glmfit, xlab=xlab, ylab=ylab, pch=pch, cex=cex, col_shrunk=col_shrunk,
                         col_trend=col_trend, col_raw=col_raw, **kwargs)
    p.render(ax)
    return out


def _plotQLDisp(glmfit, xlab="Average Log2 CPM", ylab="Quarter-Root Mean Deviance", pch=16, cex=0.2,
                col_shrunk="red", col_trend="blue", col_raw="black", **kwargs):
    if not isinstance(glmfit, (DGEGLM, DGEBIN)):
        raise TypeError("plotQLDisp() needs a DGEGLM or DGEBIN")
    if glmfit.get("s2.post") is None:
        raise ValueError("need to run glmQLFit before plotQLDisp")
    A = glmfit.get("AveLogCPM")
    if A is None:
        raise ValueError("AveLogCPM component is absent.")
    A = np.asarray(A, dtype=np.float64)

    # older fits put the df adjusted for exact zeros in df.residual.zeros
    if glmfit.get("df.residual.zeros") is None:
        df_residual = np.asarray(glmfit["df.residual.adj"], dtype=np.float64)
        deviance = np.asarray(glmfit["deviance.adj"], dtype=np.float64)
    else:
        df_residual = np.asarray(glmfit["df.residual.zeros"], dtype=np.float64)
        deviance = np.asarray(glmfit["deviance"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):       # zero df: R gives NaN, set to 0 below
        s2 = deviance / df_residual
    s2[df_residual < 1e-8] = 0
    qr = lambda v: _r.sqrt(_r.sqrt(np.asarray(v, dtype=np.float64)))

    p = RPlot()
    y = qr(s2)
    p.plot(A, y, xlab=xlab, ylab=ylab, pch=pch, cex=cex, col=col_raw, **kwargs)
    p.points(A, qr(glmfit["s2.post"]), pch=pch, cex=cex, col=col_shrunk)
    s2_prior = np.asarray(glmfit["s2.prior"], dtype=np.float64).reshape(-1)
    if s2_prior.size == 1:
        p.abline(h=qr(s2_prior), col=col_trend)
    else:
        o = _r.order(A)
        p.lines(A[o], qr(s2_prior[o]), col=col_trend, lwd=2)
    p.legend("topright", lty=[-1, -1, 1], pch=[pch, pch, -1], col=[col_raw, col_shrunk, col_trend], pt_cex=0.7,
             lwd=2, legend=["Raw", "Squeezed", "Trend"])
    return p, {"x": A, "y": y}
