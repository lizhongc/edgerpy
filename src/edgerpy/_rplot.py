"""R base graphics for the plot functions (R: graphics package plot.default,
points, lines, text, abline, axis, mtext, title, legend).

A plot is first recorded as R's display list: one entry per low-level
graphics call (plot.window, plotXY, axis, box, title, text, abline, mtext,
legend) with the arguments R evaluates for it, named as in the
.External.graphics() calls of R's graphics package. tests/test_plots.py
compares the list with R's recordPlot() output; render() then draws it with
matplotlib.
"""

import numpy as np

from . import _r

# R's par() defaults used by the recorded calls
_PAR = {"pch": 1, "lty": "solid", "col": "black", "cex": 1.0, "lwd": 1.0}


def _dbl(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _range_finite(v):
    """R range(v[is.finite(v)])."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("need finite 'xlim' values")
    return np.array([v.min(), v.max()])


class RPlot:
    """One R plot: a list of graphics calls, recorded in R's order."""

    def __init__(self):
        self.ops = []

    def _add(self, op, **args):
        self.ops.append((op, args))

    # ---- high level: plot.default -------------------------------------------
    def plot(self, x, y=None, type="p", xlim=None, ylim=None, main=None, sub=None, xlab=None, ylab=None,
             pch=None, col=None, bg=None, cex=1.0, lty=None, lwd=None, **par):
        """R plot.default(x, y, ...): window, points, both axes, box and titles.

        Extra graphical parameters (`par`, e.g. xaxt="n") go to every call,
        as R's `...` does.
        """
        if y is None:                     # xy.coords(x, NULL): index plot
            y = _dbl(x)
            x = np.arange(1, y.size + 1, dtype=np.float64)
            if xlab is None:
                xlab = "Index"
        x, y = _dbl(x), _dbl(y)
        xlim = _range_finite(x) if xlim is None else _dbl(xlim)
        ylim = _range_finite(y) if ylim is None else _dbl(ylim)
        self._add("plot_window", xlim=xlim, ylim=ylim, **par)
        self._xy(x, y, type, pch, lty, col, bg, cex, lwd, **par)
        self._add("axis", side=1, at=None, labels=None, **par)
        self._add("axis", side=2, at=None, labels=None, **par)
        self._add("box", which=1, lty="solid", **par)
        self._add("title", main=main, sub=sub, xlab=xlab, ylab=ylab, **par)

    # ---- low level -------------------------------------------------------------
    def _xy(self, x, y, type, pch=None, lty=None, col=None, bg=None, cex=1.0, lwd=None, **par):
        """R plot.xy() -> C_plotXY with par() defaults filled in."""
        self._add("plotXY", x=_dbl(x), y=_dbl(y), type=type,
                  pch=_PAR["pch"] if pch is None else pch, lty=_PAR["lty"] if lty is None else lty,
                  col=_PAR["col"] if col is None else col, bg=bg, cex=cex,
                  lwd=_PAR["lwd"] if lwd is None else lwd, **par)

    def points(self, x, y, pch=None, col=None, cex=1.0, bg=None, lwd=None, type="p"):
        self._xy(x, y, type, pch=pch, col=col, cex=cex, bg=bg, lwd=lwd)

    def lines(self, x, y, col=None, lwd=None, lty=None):
        self._xy(x, y, "l", col=col, lwd=lwd, lty=lty)

    def text(self, x, y, labels, cex=1.0, col=None):
        self._add("text", x=_dbl(x), y=_dbl(y), labels=[str(v) for v in labels], cex=cex, col=col)

    def abline(self, h=None, v=None, col=None, lty=None, lwd=None):
        self._add("abline", h=None if h is None else _dbl(h), v=None if v is None else _dbl(v),
                  col=_PAR["col"] if col is None else col, lty=_PAR["lty"] if lty is None else lty,
                  lwd=_PAR["lwd"] if lwd is None else lwd)

    def axis(self, side, at=None, labels=None, **par):
        self._add("axis", side=side, at=None if at is None else _dbl(at),
                  labels=None if labels is None else [str(v) for v in labels], **par)

    def mtext(self, text, side=3, line=0.0, padj=None):
        self._add("mtext", text=text, side=side, line=line, padj=padj)

    def legend(self, x, legend, pch=None, col=None, lty=None, lwd=None, cex=1.0, pt_cex=None):
        """R legend(x, legend=, ...) at a keyword position, kept as one entry."""
        n = len(legend)
        rep = lambda v: None if v is None else list(np.resize(np.asarray(v, dtype=object), n))
        self._add("legend", x=x, legend=[str(v) for v in legend], pch=rep(pch), col=rep(col), lty=rep(lty),
                  lwd=rep(lwd), cex=cex, pt_cex=rep(cex if pt_cex is None else pt_cex))

    # ---- R's axis layout ---------------------------------------------------------
    def window(self):
        """xlim, ylim of the plot (from its plot.window call)."""
        for op, a in self.ops:
            if op == "plot_window":
                return a["xlim"], a["ylim"]
        raise ValueError("plot.new has not been called yet")

    def axes(self):
        """R's par("usr") and default tick positions of both axes (xaxs = yaxs = "r")."""
        xlim, ylim = self.window()
        return _r.axis_ticks(xlim[0], xlim[1], "r"), _r.axis_ticks(ylim[0], ylim[1], "r")

    # ---- drawing -------------------------------------------------------------------
    def render(self, ax=None):
        """Draw the recorded plot with matplotlib, laid out as R's default device.

        ax=None opens a new 7 x 7 in figure whose plot region has R's default
        margins (par("mar") = 5.1, 4.1, 4.1, 2.1 lines); otherwise the plot is
        drawn into `ax`. Returns the Axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:                                  # pragma: no cover
            raise ImportError("edgerpy's plot functions need matplotlib: pip install matplotlib") from e
        with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": [_font()]}):
            return self._render(ax)

    def _render(self, ax):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(figsize=(_DEV_IN, _DEV_IN))
            b, l, t, r = (m * _LINE / 72 / _DEV_IN for m in _MAR)
            ax = fig.add_axes([l, b, 1 - l - r, 1 - b - t])
        xa, ya = self.axes()
        ax.set_xlim(*xa["usr"])
        ax.set_ylim(*ya["usr"])
        ax.set_autoscale_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(False)
        for op, a in self.ops:
            getattr(self, "_draw_" + op)(ax, a, xa, ya)
        # matplotlib resolves a generic family when the figure is drawn, so name the font on every text
        font = _font()
        legend = ax.get_legend()
        for t in list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels() + \
                (legend.get_texts() if legend is not None else []):
            t.set_fontfamily(font)
        return ax

    def _draw_plot_window(self, ax, a, xa, ya):
        pass

    def _draw_plotXY(self, ax, a, xa, ya):
        x, y, t = a["x"], a["y"], a["type"]
        if t in ("l", "b"):
            segs = _gapped(ax, x, y, 0.5 * _CHAR_IN * 72 * _first(a["cex"])) if t == "b" else [(x, y)]
            for sx, sy in segs:
                ax.plot(sx, sy, color=_rcol(_first(a["col"])), lw=_LWD * _first(a["lwd"]),
                        linestyle=_lty(_first(a["lty"])), solid_capstyle="round")
        if t in ("p", "b"):
            _points(ax, x, y, a["pch"], a["col"], a["cex"], a.get("bg"), a["lwd"])

    def _draw_axis(self, ax, a, xa, ya):
        side = int(a["side"])
        horizontal = side in (1, 3)
        if a.get("xaxt" if horizontal else "yaxt") == "n" and a["at"] is None:
            return
        info = xa if horizontal else ya
        at = info["at"] if a["at"] is None else a["at"]
        labels = _axis_labels(at) if a["labels"] is None else a["labels"]
        size = _PS * float(a.get("cex.axis", 1.0))
        las = int(a.get("las", 0))
        # labels parallel to the axis (las 0 on both axes, 1 on x) or perpendicular to it
        parallel = las == 0 or (las == 1 and horizontal) or (las == 3 and not horizontal)
        tick = 0.5 * _LINE                                     # par("tcl") = -0.5 lines, outward
        if parallel and horizontal:     # baseline at line mgp[2] + 1 - ylbias; label top ~0.72 size above it
            pad = (1 + 1 - _YLBIAS) * _LINE - tick - 0.72 * size
        elif parallel:                  # rotated text: baseline at line mgp[2] + ylbias, glyphs away from the axis
            pad = (1 + _YLBIAS) * _LINE - tick
        else:                           # perpendicular text ends at line mgp[2]
            pad = 1 * _LINE - tick
        axis = ax.xaxis if horizontal else ax.yaxis
        axis.set_ticks(at)
        axis.set_ticklabels(labels)
        axis.set_tick_params(direction="out", length=tick, width=_LWD, labelsize=size, pad=pad)
        for lab in axis.get_ticklabels():
            lab.set_fontfamily(_font())
            lab.set_rotation(90 if parallel != horizontal else 0)
            if not horizontal and parallel:
                lab.set_verticalalignment("center")

    def _draw_box(self, ax, a, xa, ya):
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(_LWD)
            ax.spines[side].set_color("black")

    def _draw_title(self, ax, a, xa, ya):
        if a["main"] is not None:
            main = a["main"] if isinstance(a["main"], str) else " ".join(map(str, a["main"]))
            ax.annotate(main, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 0.5 * _MAR[2] * _LINE),
                        textcoords="offset points", ha="center", va="center", fontsize=1.2 * _PS, fontweight="bold")
        if a["xlab"]:
            ax.annotate(a["xlab"], xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -(3 + 1 - _YLBIAS) * _LINE),
                        textcoords="offset points", ha="center", va="baseline", fontsize=_PS)
        if a["ylab"]:
            ax.annotate(a["ylab"], xy=(0, 0.5), xycoords="axes fraction", xytext=(-(3 + _YLBIAS) * _LINE, 0),
                        textcoords="offset points", ha="center", va="baseline", fontsize=_PS, rotation=90,
                        rotation_mode="anchor")

    def _draw_text(self, ax, a, xa, ya):
        n = len(a["labels"])
        col = np.resize(np.asarray(["black"] if a["col"] is None else a["col"], dtype=object).reshape(-1), n)
        cex = np.resize(np.asarray(a["cex"], dtype=np.float64).reshape(-1), n)
        for xi, yi, s, c, k in zip(a["x"], a["y"], a["labels"], col, cex):
            ax.text(xi, yi, s, ha="center", va="center", fontsize=_PS * k, color=_rcol(c), zorder=3)

    def _draw_abline(self, ax, a, xa, ya):
        kw = dict(color=_rcol(a["col"]), lw=_LWD * a["lwd"], linestyle=_lty(a["lty"]))
        for h in ([] if a["h"] is None else a["h"]):
            ax.axhline(h, **kw)
        for v in ([] if a["v"] is None else a["v"]):
            ax.axvline(v, **kw)

    def _draw_mtext(self, ax, a, xa, ya):
        if int(a["side"]) != 1:
            raise NotImplementedError("mtext is drawn on side 1 only")
        down = (a["line"] + 1 - _YLBIAS) * _LINE + (a["padj"] or 0) * 0.72 * _PS
        ax.annotate(a["text"], xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -down), textcoords="offset points",
                    ha="center", va="baseline", fontsize=_PS)

    def _draw_legend(self, ax, a, xa, ya):
        from matplotlib.lines import Line2D
        n = len(a["legend"])
        handles = []
        for k in range(n):
            pch = None if a["pch"] is None else a["pch"][k]
            lty = None if a["lty"] is None else a["lty"][k]
            col = _rcol(a["col"][k] if a["col"] is not None else "black")
            marker, filled, size = _PCH.get(int(pch), _PCH[1]) if pch is not None and pch >= 0 else ("", True, 0)
            line = lty is not None and lty not in (0, -1, "blank")
            handles.append(Line2D([], [], color=col, marker=marker, markersize=size * float(a["pt_cex"][k]),
                                  markerfacecolor=col if filled else "none", markeredgewidth=0 if filled else _LWD,
                                  linestyle=_lty(lty) if line else "none",
                                  lw=_LWD * float(a["lwd"][k]) if line and a["lwd"] is not None else _LWD))
        has_line = a["lty"] is not None and any(v not in (0, -1, None) for v in a["lty"])
        ax.legend(handles, a["legend"], loc=_LEGEND_LOC[a["x"]], fontsize=_PS * a["cex"], frameon=True, fancybox=False,
                  edgecolor="black", framealpha=1, borderaxespad=0, handlelength=2 if has_line else 1,
                  handletextpad=0.5).get_frame().set_linewidth(_LWD)


# ---- R graphics constants and helpers ------------------------------------------------
_DEV_IN = 7.0                     # R's default device: 7 x 7 in
_PS = 12.0                        # par("ps")
_LINE = 1.2 * _PS                 # one margin line (cra[2] = 1.2 * ps), in points
_CHAR_IN = 0.2                    # par("cin")[2]: character height in inches
_MAR = (5.1, 4.1, 4.1, 2.1)       # par("mar"): bottom, left, top, right
_YLBIAS = 0.2                     # par("ylbias")
_LWD = 0.75                       # lwd = 1 is 1/96 in = 0.75 pt
_RADIUS = 0.375 * 0.5 * _LINE     # R's symbol radius at cex = 1: RADIUS * GSTR_0, GSTR_0 = cra[2] / 2 (points)
_FONTS = ("Helvetica", "Arial", "Nimbus Sans", "Liberation Sans")   # R's sans; DejaVu Sans if none installed

# R pch -> (matplotlib marker, filled, size in points at cex = 1)
_PCH = {0: ("s", False, 2 * _RADIUS), 1: ("o", False, 2 * _RADIUS), 2: ("^", False, 2.33 * _RADIUS),
        3: ("+", False, 2 * 1.414214 * _RADIUS), 4: ("x", False, 2 * _RADIUS), 5: ("D", False, 1.414214 * _RADIUS),
        6: ("v", False, 2.33 * _RADIUS), 8: ("*", False, 2 * _RADIUS), 15: ("s", True, 2 * _RADIUS),
        16: ("o", True, 2 * _RADIUS), 17: ("^", True, 2.33 * _RADIUS), 18: ("D", True, _RADIUS),
        19: ("o", True, 2 * _RADIUS + _LWD), 20: ("o", True, 2 * 0.25 * 0.5 * _LINE), 21: ("o", False, 2 * _RADIUS),
        22: ("s", False, 2 * _RADIUS * 0.886227), 23: ("D", False, 1.414214 * _RADIUS),
        24: ("^", False, 2.33 * _RADIUS), 25: ("v", False, 2.33 * _RADIUS)}

_LTY = {"solid": "-", 1: "-", "dashed": (0, (4, 4)), 2: (0, (4, 4)), "dotted": (0, (1, 3)), 3: (0, (1, 3)),
        "dotdash": (0, (1, 3, 4, 3)), 4: (0, (1, 3, 4, 3)), "longdash": (0, (7, 3)), 5: (0, (7, 3)),
        "twodash": (0, (2, 2, 6, 2)), 6: (0, (2, 2, 6, 2))}

_LEGEND_LOC = {"topright": "upper right", "topleft": "upper left", "bottomright": "lower right",
               "bottomleft": "lower left", "top": "upper center", "bottom": "lower center", "left": "center left",
               "right": "center right", "center": "center"}


def _font():
    """The first of R's sans fonts installed for matplotlib (DejaVu Sans otherwise)."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in _FONTS if f in names), "DejaVu Sans")


def _first(v):
    return np.asarray(v, dtype=object).reshape(-1)[0] if isinstance(v, (list, tuple, np.ndarray)) else v


def _lty(v):
    v = int(v) if isinstance(v, (float, np.floating)) else v
    return _LTY.get(v, "-")


def _rcol(c):
    """An R colour (name, "#RRGGBB", palette index or "transparent") -> matplotlib colour."""
    from ._rcolors import COLORS, PALETTE
    if c is None:
        return "black"
    if isinstance(c, (int, np.integer, float, np.floating)) or (isinstance(c, str) and c.isdigit()):
        i = int(c)
        return "#FFFFFF" if i == 0 else PALETTE[(i - 1) % len(PALETTE)]
    if c == "transparent":
        return "none"
    return COLORS.get(c, COLORS.get(c.replace(" ", "").lower(), c))


def _points(ax, x, y, pch, col, cex, bg, lwd):
    """R points(): one marker per point, grouped by (pch, col, cex) for matplotlib."""
    n = x.size
    rep = lambda v: np.resize(np.asarray(v, dtype=object).reshape(-1), n) if n else np.array([], dtype=object)
    pch, col, cex = rep(pch), rep(col), rep(cex)
    lw = _LWD * float(_first(lwd))
    groups = {}
    for k in range(n):
        groups.setdefault((int(pch[k]), str(col[k]), float(cex[k])), []).append(k)
    for (p, c, k), idx in groups.items():
        marker, filled, size = _PCH.get(p, _PCH[1])
        color = _rcol(c if not c.lstrip("-").isdigit() else int(c))
        face = color if filled else ("none" if bg is None or p < 21 else _rcol(_first(bg)))
        ax.plot(x[idx], y[idx], linestyle="none", marker=marker, markersize=size * k, markerfacecolor=face,
                markeredgecolor=color, markeredgewidth=0 if filled and p != 19 else lw)


def _gapped(ax, x, y, gap_pt):
    """R type = "b" lines: each segment shortened by gap_pt at both ends (skipped if too short)."""
    to_pt = ax.transData
    inv = to_pt.inverted()
    pts = to_pt.transform(np.column_stack([x, y])) * 72 / ax.figure.dpi
    segs = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d = np.hypot(x1 - x0, y1 - y0)
        if np.isfinite(d) and gap_pt < 0.5 * d:
            f = gap_pt / d
            a = np.array([[x0 + f * (x1 - x0), y0 + f * (y1 - y0)], [x1 + f * (x0 - x1), y1 + f * (y0 - y1)]])
            dat = inv.transform(a * ax.figure.dpi / 72)
            segs.append((dat[:, 0], dat[:, 1]))
    return segs


def _axis_labels(at):
    """R's default axis labels: formatReal() of the whole vector at 7 significant
    digits, fixed notation unless scientific is narrower."""
    at = np.asarray(at, dtype=np.float64)
    if at.size == 0:
        return []
    sig, left, right, mant = [], [], [], []
    for v in at:
        if v == 0:
            sig.append(1), left.append(1), right.append(0), mant.append(1)
            continue
        s = next(s for s in range(1, 8) if float(f"{v:.{s - 1}e}") == float(f"{v:.6e}"))
        kp = int(f"{v:.{s - 1}e}".split("e")[1])
        sig.append(s)
        left.append(kp + 1 if kp >= 0 else 1)
        right.append(max(0, s - 1 - kp))
        mant.append(s)
    neg = int((at < 0).any())
    rgt = max(right)
    wF = neg + max(left) + (rgt + 1 if rgt > 0 else 0)
    mF = max(mant)
    ewidth = 4 if max(abs(int(f"{v:e}".split("e")[1])) for v in at if v != 0) < 100 else 5
    wE = neg + (mF + 1 if mF > 1 else 1) + ewidth
    if wF <= wE:
        return [f"{v:.{rgt}f}" for v in at]
    return [f"{v:.{mF - 1}e}".replace("e+0", "e+0").replace("e-0", "e-0") for v in at]
