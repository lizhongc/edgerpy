"""filterByExpr (R: filterByExpr.R)."""

import numpy as np
import pandas as pd

from . import _r
from .aveLogCPM import _cpm_default
from .classes import DGEList
from .helpers import as_matrix, design_matrix, factor, mat

_PRESETS = {
    "stringent": dict(min_count=10, min_total_count=15, large_n=10, min_prop=0.7, max_n_samples=40),
    "moderate": dict(min_count=6, min_total_count=10, large_n=4, min_prop=0.5, max_n_samples=20),
    "lenient": dict(min_count=1, min_total_count=6, large_n=2, min_prop=0.5, max_n_samples=10),
}


def filterByExpr(y, design=None, group=None, min_sample_size=None, lib_size=None, min_count=10,
                 min_total_count=15, large_n=10, min_prop=0.7, max_n_samples=40, preset=None):
    """Filter Genes by Expression Level

    Determine which genes have sufficiently large counts to be retained in a statistical analysis.

    Parameters
    ----------
    y : DGEList
        The counts, with the samples' library sizes, normalization factors and groups.
    design : DataFrame or ndarray, optional
        Design matrix. Defaults to ``y["design"]``. Used to compute the minimum group sample size
        (``MinSampleSize``). Ignored if ``group`` or ``min_sample_size`` is given.
    group : array-like or pandas.Categorical, optional
        Group membership of the samples. An alternative way to specify the design matrix when there is
        only one explanatory factor. Defaults to ``y["samples"]["group"]`` if neither ``design`` nor
        ``group`` is given and ``y`` has no design. Ignored if ``min_sample_size`` is given.
    min_sample_size : float, optional
        Minimum group sample size, given directly instead of being computed from ``design`` or
        ``group``. If given, it takes precedence over both.
    lib_size : array-like, optional
        Library sizes. Defaults to the effective library sizes of ``y``
        (``lib.size * norm.factors``). Used to compute the CPM cutoff.
    min_count : float, default 10
        Minimum read count required for at least some samples. Used to compute the CPM cutoff.
    min_total_count : float, default 15
        Minimum total count required across all samples.
    large_n : float, default 10
        Number of samples that would be considered a "large" group. Used to modify ``MinSampleSize``
        for large datasets.
    min_prop : float, default 0.7
        Proportion of samples in large groups for which a gene needs to be expressed to be
        biologically interesting. Used to modify ``MinSampleSize`` for large datasets; see Notes.
    max_n_samples : float, default 40
        Upper bound on the number of samples in which a gene needs to be expressed.
    preset : {"stringent", "moderate", "lenient"}, optional
        A preset combination of the tuning arguments, going from heavy to light filtering. The
        defaults match ``"stringent"``. If given, it overrides the individual tuning arguments.

    Returns
    -------
    ndarray of bool
        One value per gene (row of ``y``): ``True`` for the genes to keep in the analysis.

    Notes
    -----
    This function implements the filtering strategy that was described informally by Chen et al
    (2016). Roughly speaking, the strategy keeps genes that have at least ``min_count`` reads in a
    worthwhile number of samples.

    More precisely, the filtering keeps genes that have ``CPM >= CPM.cutoff`` in ``MinSampleSize``
    samples, where ``CPM.cutoff = min_count / median(lib_size) * 1e6`` and ``MinSampleSize`` is the
    smallest group sample size or, more generally, the minimum inverse leverage computed from the
    design matrix.

    If all the group sample sizes are large, then ``MinSampleSize`` is reduced somewhat. If
    ``MinSampleSize > large_n``, then genes are kept if ``CPM >= CPM.cutoff`` in ``k`` samples where
    ``k = min(large_n + (MinSampleSize - large_n) * min_prop, max_n_samples)``. This rule requires
    that genes are expressed in at least ``min_prop * MinSampleSize`` samples when ``MinSampleSize``
    is large, with ``max_n_samples`` being the upper bound on the number of samples that a gene must
    be expressed in.

    In addition, each kept gene is required to have at least ``min_total_count`` reads across all
    the samples.

    The ``preset`` argument chooses a combination of tuning parameters instead of setting them
    individually. Choose ``"stringent"`` to keep only genes with enough reads across enough samples to
    have a good chance of being called as DE if they are genuinely DE. This corresponds to the
    historical defaults of ``filterByExpr``. Choose ``"lenient"`` to keep the maximum number of genes,
    including all genes that could possibly be called as DE if the true differences are large and the
    dispersion levels are not too high. ``"moderate"`` is in between.

    ``filterByExpr`` implements independent filtering in the sense that the filtering would be exactly
    the same even if the sample group labels or the rows of the design matrix were permuted. The
    function uses ``design`` and ``group`` only to identify the smallest group size
    (``MinSampleSize``). Non-independent filtering that used information about which sample belongs
    to each group, for example keeping genes expressed in a certain number of samples in at least one
    group, would be invalid because it would potentially cause false discoveries in subsequent
    differential expression analyses.

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    if not isinstance(y, DGEList):
        raise TypeError("filterByExpr() needs a DGEList")
    if design is None and group is None:
        design = y.get("design")
        if design is None:
            group = y["samples"]["group"].array
            if len(group.categories) == 1:
                print("Treating all samples as belonging to one group.")
    if lib_size is None:
        s = y["samples"]
        lib_size = s["lib.size"].to_numpy(dtype=np.float64) * s["norm.factors"].to_numpy(dtype=np.float64)

    ym = mat(as_matrix(y["counts"]))
    if preset is not None:
        p = _PRESETS[preset]
        min_count, min_total_count, large_n = p["min_count"], p["min_total_count"], p["large_n"]
        min_prop, max_n_samples = p["min_prop"], p["max_n_samples"]

    if min_sample_size is None:
        if group is None:
            if design is None:
                print("No group or design set. Assuming all samples belong to one group.")
                MinSampleSize = float(ym.shape[1])
            else:
                h = _r.hat(design_matrix(design, ym.shape[1])[0])
                MinSampleSize = 1 / np.max(h)
        else:
            if not isinstance(group, pd.Categorical):
                group = factor(group)
            n = np.bincount(np.asarray(group.codes)[np.asarray(group.codes) >= 0], minlength=len(group.categories))
            MinSampleSize = float(np.min(n[n > 0]))
    else:
        MinSampleSize = float(min_sample_size)

    if MinSampleSize > large_n:
        MinSampleSize = large_n + (MinSampleSize - large_n) * min_prop
    MinSampleSize = min(MinSampleSize, max_n_samples)

    MedianLibSize = _r.median(lib_size)
    CPM_Cutoff = min_count / MedianLibSize * 1e6
    CPM = _cpm_default(ym, lib_size=lib_size)
    tol = 1e-14
    keep_CPM = _r.rowSums(CPM >= CPM_Cutoff) >= (MinSampleSize - tol)
    keep_TotalCount = _r.rowSums(ym) >= (min_total_count - tol)
    return keep_CPM & keep_TotalCount
