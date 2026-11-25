"""DGEList, PCList and helpers (R: DGEList.R, PCList.R, dropEmptyLevels.R, getOffset.R)."""

import warnings

import numpy as np
import pandas as pd

from . import _r
from .classes import DGEList as _DGEList
from .classes import PCList as _PCList
from .helpers import as_matrix, factor, mat


def dropEmptyLevels(x):
    """R dropEmptyLevels(x): drop factor levels that do not occur."""
    if isinstance(x, pd.Series):
        x = x.array
    if isinstance(x, pd.Categorical):
        return x.remove_unused_categories()
    return factor(x)


def _default_dimnames(counts):
    nlib, ntags = counts.shape[1], counts.shape[0]
    cols = counts.columns
    if isinstance(cols, pd.RangeIndex):
        counts.columns = [f"Sample{k + 1}" for k in range(nlib)]
    else:
        counts.columns = [str(c) for c in cols]
    idx = counts.index
    if isinstance(idx, pd.RangeIndex):
        counts.index = [str(k + 1) for k in range(ntags)]
    else:
        counts.index = [str(r) for r in idx]
    return counts


def _check_counts(a, what="counts"):
    if a.size == 0:
        raise ValueError(f"'{what}' must contain at least one value")
    m = np.min(a)
    if np.isnan(m):
        raise ValueError("NA counts not allowed")
    if m < 0:
        raise ValueError("Negative counts not allowed")
    if np.isinf(np.max(a)):
        raise ValueError("Infinite counts not allowed")


def DGEList(counts, lib_size=None, norm_factors=None, samples=None, group=None, genes=None, remove_zeros=False):
    """Create a DGEList Object

    Assembles a DGEList object from its components, especially the table of counts.

    Parameters
    ----------
    counts : DataFrame or array-like
        Sequence read counts, with rows corresponding to genes (genomic features) and columns to
        samples. Negative, missing or infinite values are not allowed. Every column is taken as
        counts. Without row or column names (a plain array, or a DataFrame with a default
        ``RangeIndex``), the genes are named ``"1"``, ``"2"``, ... and the samples ``"Sample1"``,
        ``"Sample2"``, ...
    lib_size : array-like, optional
        Library sizes (sequencing depths) of the samples. Defaults to the column sums of ``counts``.
    norm_factors : array-like, optional
        Normalization factors that modify the library sizes. Defaults to ones.
    samples : DataFrame, optional
        Sample information, with a row for each sample. Its columns are appended to the ``samples``
        component of the DGEList. A ``group`` column is used as ``group`` if ``group`` is not given.
    group : array-like or pandas.Categorical, optional
        The experimental group or treatment condition of each sample. Defaults to a single group,
        ``"1"``. Levels that do not occur are dropped; plain values become a factor with sorted
        levels (C-locale order for strings).
    genes : DataFrame, optional
        Gene annotation, with a row for each gene.
    remove_zeros : bool, default False
        Whether to remove rows that have 0 total count.

    Returns
    -------
    DGEList
        With components ``counts`` (a DataFrame of the counts), ``samples`` (a DataFrame with columns
        ``group``, ``lib.size`` and ``norm.factors``, then the columns of ``samples``) and, if given,
        ``genes``.

    Notes
    -----
    Assembles a DGEList object from its components. The only compulsory argument is the table of
    counts.

    R documentation by edgeR team. Originally created by Mark Robinson; adapted for edgerpy.
    """
    counts = _default_dimnames(as_matrix(counts))
    a = mat(counts)
    _check_counts(a)
    nlib, ntags = a.shape[1], a.shape[0]

    if lib_size is None:
        lib_size = _r.colSums(a)
        if np.min(lib_size) <= 0:
            warnings.warn("At least one library size is zero")
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)
        if lib_size.size != nlib:
            raise ValueError("length of 'lib.size' must equal number of samples")
        m = np.min(lib_size)
        if np.isnan(m):
            raise ValueError("NA library sizes not allowed")
        if m < 0:
            raise ValueError("negative library sizes not allowed")
        if m == 0 and np.any((lib_size == 0) & (_r.colSums(a) > 0)):
            raise ValueError("library size set to zero but counts for that sample are nonzero")

    if norm_factors is None:
        norm_factors = np.ones(nlib)
    else:
        norm_factors = np.asarray(norm_factors, dtype=np.float64)
        if norm_factors.size != nlib:
            raise ValueError("Length of 'norm.factors' must equal number of columns in 'counts'")
        m = np.min(norm_factors)
        if np.isnan(m):
            raise ValueError("NA norm factors not allowed")
        if m <= 0:
            raise ValueError("norm factors must be positive")
        if abs(_r.rsum(_r.log(norm_factors))) > 1e-6:
            warnings.warn("norm factors don't multiply to 1")

    if samples is not None:
        samples = pd.DataFrame(samples).reset_index(drop=True)
        if len(samples) != nlib:
            raise ValueError("Number of rows in 'samples' must equal number of columns in 'counts'")
        if group is None and "group" in samples.columns:
            group = samples["group"]
            samples = samples.drop(columns="group")

    if group is None:
        group = pd.Categorical(["1"] * nlib, categories=["1"])
    else:
        if len(group) != nlib:
            raise ValueError("Length of 'group' must equal number of columns in 'counts'")
        group = dropEmptyLevels(group)

    sam = pd.DataFrame({"group": group, "lib.size": np.asarray(lib_size, dtype=np.float64),
                        "norm.factors": norm_factors})
    if samples is not None:
        sam = pd.concat([sam, samples], axis=1)
    cn = list(counts.columns)
    if len(set(cn)) < len(cn):
        print("Repeated column names found in count matrix")
        sam.index = [str(k + 1) for k in range(nlib)]
    else:
        sam.index = cn

    x = _DGEList(counts=counts, samples=sam)

    if genes is not None:
        genes = pd.DataFrame(genes).copy()
        if len(genes) != ntags:
            raise ValueError("Counts and genes have different numbers of rows")
        rn = list(counts.index)
        if len(set(rn)) < len(rn):
            warnings.warn("Count matrix has duplicated rownames")
        else:
            genes.index = rn
        x["genes"] = genes

    if remove_zeros:
        all_zeros = np.sum(a > 0, axis=1) == 0
        if all_zeros.any():
            x = x.subset(~all_zeros, None)
            print(f"Removing {int(all_zeros.sum())} rows with all zero counts")
    return x


def PCList(counts, counts2, samples=None, group=None, genes=None):
    """Create a PCList Object

    Assembles a PCList object from its components, especially the paired counts as two matrices.

    Parameters
    ----------
    counts : DataFrame or array-like
        Sequence read counts, with rows corresponding to genes (genomic features) and columns to
        samples. Negative, missing or infinite values are not allowed. Without row or column names,
        the genes are named ``"1"``, ``"2"``, ... and the samples ``"Sample1"``, ``"Sample2"``, ...
    counts2 : DataFrame or array-like
        The paired counts, of the same dimensions as ``counts``. Negative, missing or infinite values
        are not allowed.
    samples : DataFrame, optional
        Sample information, with a row for each sample. Its columns are appended to the ``samples``
        component of the PCList. A ``group`` column is used as ``group`` if ``group`` is not given.
    group : array-like or pandas.Categorical, optional
        The experimental group or treatment condition of each sample. Defaults to a single group,
        ``"1"``. Levels that do not occur are dropped.
    genes : DataFrame, optional
        Gene annotation, with a row for each gene.

    Returns
    -------
    PCList
        With components ``counts``, ``counts2``, ``samples`` (a DataFrame with column ``group``, then
        the columns of ``samples``) and, if given, ``genes``.

    Notes
    -----
    Assembles a PCList object from its components. The only compulsory arguments are the paired
    counts.

    R documentation by edgeR team. Originally created by Lizhong Chen; adapted for edgerpy.
    """
    counts = as_matrix(counts)
    counts2 = as_matrix(counts2)
    if mat(counts).size == 0:
        raise ValueError("'counts' must contain at least one value")
    if mat(counts2).size == 0:
        raise ValueError("'counts2' must contain at least one value")
    if counts.shape != counts2.shape:
        raise ValueError("The dimensions of 'counts' and 'counts2' are not matched")
    a, b = mat(counts), mat(counts2)
    m1, m2 = np.min(a), np.min(b)
    if np.isnan(m1) or np.isnan(m2):
        raise ValueError("NA counts not allowed")
    if m1 < 0 or m2 < 0:
        raise ValueError("Negative counts not allowed")
    if np.isinf(np.max(a)) or np.isinf(np.max(b)):
        raise ValueError("Infinite counts not allowed")
    nlib, ntags = a.shape[1], a.shape[0]
    if isinstance(counts.columns, pd.RangeIndex):
        names = [f"Sample{k + 1}" for k in range(nlib)]
        counts.columns = names
        counts2.columns = names
    if isinstance(counts.index, pd.RangeIndex):
        rn = [str(k + 1) for k in range(ntags)]
        counts.index = rn
        counts2.index = rn

    if samples is not None:
        samples = pd.DataFrame(samples).reset_index(drop=True)
        if len(samples) != nlib:
            raise ValueError("Number of rows in 'samples' must equal number of columns in 'counts'")
        if group is None and "group" in samples.columns:
            group = samples["group"]
            samples = samples.drop(columns="group")
    if group is None:
        group = pd.Categorical(["1"] * nlib, categories=["1"])
    else:
        if len(group) != nlib:
            raise ValueError("Length of 'group' must equal number of columns in 'counts'")
        group = dropEmptyLevels(group)
    sam = pd.DataFrame({"group": group})
    if samples is not None:
        sam = pd.concat([sam, samples], axis=1)
    cn = [str(c) for c in counts.columns]
    if len(set(cn)) < len(cn):
        print("Repeated column names found in count matrix")
        sam.index = [str(k + 1) for k in range(nlib)]
    else:
        sam.index = cn

    x = _PCList(counts=counts, counts2=counts2, samples=sam)
    if genes is not None:
        genes = pd.DataFrame(genes).copy()
        if len(genes) != ntags:
            raise ValueError("Counts and genes have different numbers of rows")
        rn = list(counts.index)
        if len(set(rn)) < len(rn):
            warnings.warn("Count matrix has duplicated rownames")
        else:
            genes.index = rn
        x["genes"] = genes
    return x


def getOffset(y):
    """Extract the Offset of a DGEList Object

    Returns offsets for the log-linear predictor that account for sequencing depth and possibly
    other normalization factors. Specifically it returns the matrix ``y["offset"]`` if that exists.
    Otherwise, it returns the log effective library sizes (``log(lib.size * norm.factors)``), added
    columnwise to the prior offset matrix ``y["offset.prior"]`` if that exists.

    Parameters
    ----------
    y : DGEList
        Containing at least the ``samples`` component with a ``lib.size`` column.

    Returns
    -------
    ndarray or DataFrame
        ``y["offset"]`` as it is stored if present; a matrix (genes by samples) if ``y`` has
        ``offset.prior``; otherwise a vector with one value per sample.

    Notes
    -----
    Raises ``ValueError`` if ``y["samples"]`` has no ``lib.size`` column.

    R documentation by Mark Robinson, Davis McCarthy, Gordon Smyth; adapted for edgerpy.
    """
    if "offset" in y:
        return y["offset"]
    s = y["samples"]
    if "lib.size" not in s:
        raise ValueError("y is not a valid DGEList object")
    lib_size = s["lib.size"].to_numpy(dtype=np.float64)
    if "norm.factors" in s:
        lib_size = lib_size * s["norm.factors"].to_numpy(dtype=np.float64)
    if "offset.prior" in y:
        op = mat(y["offset.prior"]).astype(np.float64)
        return op + _r.log(lib_size)[None, :]
    return _r.log(lib_size)
