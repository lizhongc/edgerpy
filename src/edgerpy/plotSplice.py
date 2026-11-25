"""plotSplice for diffSplice output (R: limma plotSplice.R)."""

import re

import numpy as np

from . import _r
from ._rplot import RPlot
from .classes import DGEDTU
from .diffSplice import topSplice
from .helpers import mat


def plotSplice(fit, coef=None, geneid=None, genecolname=None, rank=1, FDR=0.05, xlab="Exon", ax=None):
    """Differential usage plot

    Plot relative log-fold changes by exons (or transcripts) for the specified gene and highlight
    differentially used exons (or transcripts).

    Parameters
    ----------
    fit : DGEDTU
        Output of ``diffSplice``. Other objects raise ``TypeError``.
    coef : int or str, optional
        The coefficient (column) of ``fit`` for which differential splicing is assessed, by name or
        0-based position. Defaults to the last coefficient.
    geneid : str, optional
        ID of the gene to plot. Raises ``ValueError`` if it is not found.
    genecolname : str or list of str, optional
        Column of ``fit["gene.genes"]`` containing gene IDs. Defaults to ``fit["genecolname"]``. With
        several columns, the plotted gene is titled by their values joined with ``"."``, and
        ``geneid`` is looked up in the first column only.
    rank : int, default 1
        If ``geneid`` is not given, the gene of this rank by the gene-level F-test is plotted:
        ``rank=1`` is the most significant gene. A rank, so it starts at 1, as in R.
    FDR : float, default 0.05
        Highlight exons as red dots with false discovery rate less than this cutoff. The FDR of the
        individual exon is calculated based on the exon-level t-statistics test for differences
        between each exon and all other exons for the same gene.
    xlab : str, default "Exon"
        x-axis label. Typical choices are ``"Exon"`` for DEU analyses or ``"Transcript"`` for DTU
        analyses.
    ax : matplotlib.axes.Axes, optional
        The axes to draw in. Defaults to a new 7 x 7 inch figure, like a new R graphics device.

    Returns
    -------
    None
        A plot is drawn.

    Notes
    -----
    Plot relative log2-fold-changes by exon (or transcript) for the specified gene. The relative
    logFC is the difference between that exon (or transcript)'s logFC and the overall logFC for the
    gene, as computed by ``diffSplice``. Individual exons (or transcripts) that are significantly
    differentially used are highlighted as red dots. The size of the red dots increases with the
    statistical significance of the exon (or transcript).

    The plot follows R's base graphics: plotted values, axis limits, ticks, labels and colours are
    those of R.

    R documentation by Gordon Smyth and Yifang Hu; adapted for edgerpy.
    """
    _plotSplice(fit, coef=coef, geneid=geneid, genecolname=genecolname, rank=rank, FDR=FDR, xlab=xlab).render(ax)


def _plotSplice(fit, coef=None, geneid=None, genecolname=None, rank=1, FDR=0.05, xlab="Exon"):
    if not isinstance(fit, DGEDTU) or fit.get("gene.firstexon") is None:
        raise TypeError("plotSplice() needs the output of diffSplice")
    beta = fit["coefficients"]
    coefficients = np.asarray(mat(beta), dtype=np.float64).reshape(len(beta), -1)
    ncoef = coefficients.shape[1]
    if coef is None:
        coef = ncoef - 1
    elif isinstance(coef, str):
        coef = list(beta.columns).index(coef)
    genecolname = fit["genecolname"] if genecolname is None else genecolname
    gcols = [genecolname] if isinstance(genecolname, str) else list(genecolname)
    gene_genes = fit["gene.genes"]

    if geneid is None:
        # gene from the specified rank
        pv = np.asarray(fit["gene.F.p.value"], dtype=np.float64).reshape(len(gene_genes), -1)[:, coef]
        i = _r.which_min(pv) if rank == 1 else _r.order(pv)[rank - 1]
        geneid = ".".join(str(gene_genes.iloc[i][c]) for c in gcols)
    else:
        # gene from the specified name
        geneid = str(geneid)
        hit = np.flatnonzero(gene_genes[gcols[0]].astype(str).to_numpy() == geneid)
        if hit.size == 0:
            raise ValueError(f"geneid {geneid} not found")
        i = int(hit[0])

    # rows of the gene's exons
    j = np.arange(int(fit["gene.firstexon"].iloc[i]) - 1, int(fit["gene.lastexon"].iloc[i]))
    exoncolname = fit.get("exoncolname")

    # strand, if available
    strcol = [c for c in gene_genes.columns if re.search("strand", str(c), re.I)]
    if strcol:
        geneid = f"{geneid} ({gene_genes.iloc[i][strcol[0]]})"

    ylab = f"logFC (this {xlab.lower()} vs rest)"
    y = coefficients[j, coef]
    p = RPlot()
    if exoncolname is None:
        p.plot(y, xlab=xlab, ylab=ylab, main=geneid, type="b")
    else:
        exon_id = fit["genes"][exoncolname].iloc[j].tolist()
        p.plot(y, xlab="", ylab=ylab, main=geneid, type="b", xaxt="n")
        p.axis(1, at=np.arange(1, j.size + 1), labels=exon_id, las=2, **{"cex.axis": 0.5})
        p.mtext(f"{xlab} {exoncolname}", side=1, padj=5.2)

    # mark the exons with FDR < FDR in the exon-level t-tests
    top = topSplice(fit, coef=coef, number=np.inf, test="t", sort_by="none")
    gene_values = set(gene_genes.iloc[i][gcols].astype(str))
    m = np.flatnonzero(top[gcols[0]].astype(str).isin(gene_values).to_numpy())
    fdr = top["FDR"].to_numpy(dtype=np.float64)[m]
    sig = fdr < FDR
    if sig.any():
        fdr_sig = fdr[sig]
        if np.unique(fdr_sig).size == 1:
            cex = 1.5
        else:
            abs_fdr = np.abs(_r.log10(fdr_sig))
            lo, hi = abs_fdr.min(), abs_fdr.max()
            cex = (abs_fdr - lo) / (hi - lo) * (2.0 - 1.0) + 1.0
        p.points(np.arange(1, j.size + 1, dtype=np.float64)[sig], y[sig], col="red", pch=16, cex=cex)
    p.abline(h=0, lty=2)
    return p
