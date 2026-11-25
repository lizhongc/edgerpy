"""sampleWeights (R: sampleWeights.R)."""

import numpy as np

from . import _C
from .aveLogCPM import aveLogCPM
from .classes import DGEList
from .DGEList import getOffset
from .glmfit import _default_design
from .helpers import compressOffsets, compressWeights, design_matrix, mat, nonEstimable, treatment_design


def sampleWeights(y, design=None, niter=5, tol=0.01, prior_n=50, nthreads=1):
    """Empirical Sample Quality Weights

    Estimate two-part sample weights from the adjusted unit deviances of a quasi-likelihood fit: a
    technical weight on the Poisson part of the variance and a biological weight on the negative
    binomial dispersion of each sample. The weights are stored in the ``samples`` DataFrame of the
    DGEList, where ``glmFit`` and ``glmQLFit`` use them.

    Parameters
    ----------
    y : DGEList
        The counts. Existing columns ``technical.weights`` and ``biological.weights`` of
        ``y["samples"]`` are ignored. Offsets and observation weights are taken from ``y``.
    design : DataFrame or ndarray, optional
        The design matrix. If ``None``, it is chosen as in ``glmQLFit``: ``y["design"]`` if present,
        else a one-way layout from ``y["samples"]["group"]`` if there is more than one group, else an
        intercept. Must be of full rank.
    niter : int, default 5
        Maximum number of updates of the weights, with a refit after each update except the last.
    tol : float, default 0.01
        The updates stop when the largest absolute change of ``log(t)`` and ``log(c)`` in an update
        is below ``tol``.
    prior_n : float, default 50
        Number of prior pseudo-genes for the technical weights and for the NB dispersion factors.
        Larger values squeeze ``t`` and ``c`` towards equality. Must be positive.
    nthreads : int, default 1
        Number of OpenMP threads.

    Returns
    -------
    DGEList
        A copy of ``y`` with two columns added to (or updated in) ``y["samples"]``, and the design
        used stored as ``y["design"]``:

        ``technical.weights``
            The technical weights ``t``, one per sample.
        ``biological.weights``
            The biological weights ``b = 1/c``, one per sample. Like the technical weights, larger
            values mean a less variable sample, and a GLM prior weight ``w`` corresponds
            approximately to ``t = b = w``.

        Nothing else in ``y`` is changed. ``glmFit`` and ``glmQLFit`` on the result then fit each
        sample with technical weight ``t`` and NB dispersion ``phi/b``, and ``glmQLFTest`` and
        ``diffSplice`` use the weights stored in the fit.

    Notes
    -----
    Each sample ``j`` has a technical weight ``t[j]``, which scales the Poisson part of the variance,
    and a factor ``c[j]``, which scales the NB dispersion ``phi``:
    ``var(y) = s2/s0 * (s0*mu/t + c*phi*mu^2)``, where ``s2`` is the quasi-dispersion of the gene and
    ``s0`` the average quasi-dispersion. Technical noise, such as counts from technical copies,
    lowers ``t``, while extra biological variation raises ``c``. Since the two parts dominate at
    different abundances, both can be estimated: genes where the Poisson part dominates inform ``t``
    and genes where the NB part dominates inform ``c``.

    The weights are estimated by REML. The adjusted unit deviance of each observation is treated as a
    scaled chi-square variable on its adjusted unit df, scaled by the genewise quasi-dispersion and
    by the change of variance implied by ``t`` and ``c``. Each update is one Fisher scoring step for
    ``log(t)`` and ``log(c)`` of all samples jointly, constrained so that ``t`` and ``c`` keep
    geometric mean 1, followed by a refit of the model as in ``glmQLFit``. The information matrix of
    the step is the working information of each sample with the genewise quasi-dispersion profiled
    out, as in limma's ``arrayWeights``, so that a change of variance common to all samples of a gene
    is absorbed by its quasi-dispersion. The Poisson share of each observation is evaluated at the
    leave-one-out mean of the other samples. As in ``arrayWeights``, a prior of ``prior_n``
    pseudo-genes, here genes with a pure Poisson variance, keeps ``t`` near 1 when the data carry
    little information about the Poisson part, for example when all genes have large counts. In the
    same way, ``prior_n`` pseudo-genes with a pure NB variance keep ``c`` near 1 when the data carry
    little information about the NB part, for example when all genes have small counts or when ``c``
    of a sample becomes small.

    The starting fit is that of ``glmQLFit(y, design)`` without sample weights: the common Cox-Reid
    NB dispersion ``phi`` on the most abundant genes, the average quasi-dispersion and the refitted
    means. Both ``t`` and ``c`` start at 1. The prior weights ``y["weights"]`` are taken into account.

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(y, DGEList):
        raise TypeError("y must be a DGEList")
    # the joint step needs the priors to be positive definite
    if not prior_n > 0:
        raise ValueError("prior.n must be positive")

    # design: as glmQLFit.DGEList, an intercept if None as glmQLFit.default, of full rank as glmFit.default requires
    counts = y["counts"]
    nlib = counts.shape[1]
    if design is None:
        design = y.get("design")
        if design is None:
            group = y["samples"]["group"].array.remove_unused_categories()
            if len(group.categories) > 1:
                design = treatment_design(group, "y$samples$group")
    if design is None:
        design = _default_design(nlib, list(counts.columns))
    X, colnames, design = design_matrix(design, nlib)
    if X.shape[0] != nlib:
        raise ValueError("nrow(design) disagrees with ncol(y)")
    ne = nonEstimable(X, colnames)
    if ne is not None:
        raise ValueError("Design matrix not of full rank.  The following coefficients not estimable:\n " + " ".join(ne))

    AveLogCPM = y.get("AveLogCPM")
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y)

    # starting fit as glmQLFit(legacy=False) without sample weights (existing weights in y$samples are not used), then
    # niter REML updates of t and c, each followed by a refit, stopping early when the largest change of log(t) and
    # log(c) is below tol
    ym = mat(counts)
    out = _C.sample_weights(ym, compressOffsets(ym, getOffset(y)).to_c(), compressWeights(ym, y.get("weights")).to_c(),
                            X, np.asarray(AveLogCPM, dtype=np.float64), int(niter), float(tol), float(prior_n),
                            int(nthreads))

    # store the technical weights t and the biological weights b = 1/c, one per sample, and the design
    y = y.copy()
    s = y["samples"].copy()
    s["technical.weights"] = out["t"]
    s["biological.weights"] = 1 / out["c"]
    y["samples"] = s
    y["design"] = design
    return y
