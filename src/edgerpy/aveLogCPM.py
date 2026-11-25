"""aveLogCPM and cpm (R: aveLogCPM.R, cpm.R)."""

import warnings

import numpy as np

from . import _C, _r
from .classes import DGEGLM, DGELRT, DGEList, PCList
from .helpers import (as_matrix, compressDispersions, compressOffsets, compressPrior, compressWeights, frame,
                   makeCompressedMatrix, mat)


def _isAllZero(y):
    if y.size == 0:
        return False
    lo, hi = np.min(y), np.max(y)
    if np.isnan(lo):
        raise ValueError("NA counts not allowed")
    if lo < 0:
        raise ValueError("Negative counts not allowed")
    if np.isinf(hi):
        raise ValueError("Infinite counts not allowed")
    return hi == 0


def aveLogCPM(y, prior_count=2, dispersion=None, nthreads=1, normalized_lib_sizes=True):
    """Average Log Counts per Million

    Compute average log2 counts-per-million for each row of counts.

    Parameters
    ----------
    y : DGEList or PCList
        The counts. For a DGEList, the library sizes (``lib.size``, else the column sums of the
        counts), normalization factors, ``common.dispersion`` and observation ``weights`` are taken
        from ``y``; its ``offset`` and ``offset.prior`` are not used. For a PCList, the counts are
        ``counts + counts2``, the library sizes are their column sums and the dispersion is 0.
    prior_count : float or array-like, default 2
        The average value(s) to be added to each count to avoid infinite values on the log-scale: one
        value, or one per gene.
    dispersion : float or array-like, optional
        Negative binomial dispersions, one value or one per gene. Defaults to
        ``y["common.dispersion"]``, or 0.05 if ``y`` has none. DGEList only.
    nthreads : int, default 1
        Number of OpenMP threads to use for the genewise computations.
    normalized_lib_sizes : bool, default True
        Whether to use normalized library sizes (``lib.size * norm.factors``). DGEList only.

    Returns
    -------
    ndarray
        log2(AveCPM) for each row of ``y``.

    Notes
    -----
    This function uses ``mglmOneGroup`` to compute average counts-per-million (AveCPM) for each row
    of counts, and returns log2(AveCPM). An average value of ``prior_count`` is added to the counts
    before running ``mglmOneGroup``. If ``prior_count`` is a vector, each entry is added to all counts
    in the corresponding row of ``y``.

    This function is similar to ``log2(rowMeans(cpm(y)))``, but with the refinement that larger
    library sizes are given more weight in the average. The two versions will agree for large values
    of the dispersion.

    R documentation by Gordon Smyth. C++ version by Aaron Lun. C version by Lizhong Chen; adapted for
    edgerpy.
    """
    if isinstance(y, DGEList):
        lib = y["samples"]["lib.size"].to_numpy(dtype=np.float64) if "lib.size" in y["samples"] \
            else _r.colSums(mat(y["counts"]))
        if normalized_lib_sizes and "norm.factors" in y["samples"]:
            lib = lib * y["samples"]["norm.factors"].to_numpy(dtype=np.float64)
        if dispersion is None:
            dispersion = y.get("common.dispersion")
        return _aveLogCPM_default(y["counts"], lib_size=lib, prior_count=prior_count, dispersion=dispersion,
                                  weights=y.get("weights"), nthreads=nthreads)
    if isinstance(y, PCList):
        tot = mat(y["counts"]) + mat(y["counts2"])
        lib = _r.colSums(tot)
        return _aveLogCPM_default(tot, lib_size=lib, prior_count=prior_count, dispersion=0,
                                  weights=y.get("weights"), nthreads=nthreads)
    raise TypeError("aveLogCPM() needs a DGEList or PCList")


def _aveLogCPM_default(y, lib_size=None, offset=None, prior_count=2, dispersion=None, weights=None, nthreads=1):
    """R aveLogCPM.default(): average log2-CPM per row of a count matrix."""
    ym = mat(as_matrix(y))
    if ym.shape[0] == 0:
        return np.zeros(0)
    if _isAllZero(ym):
        if (lib_size is None or np.max(lib_size) == 0) and (offset is None or np.max(mat(offset)) == -np.inf):
            abundance = np.full(ym.shape[0], -_r.log(float(ym.shape[0])))
            return (abundance + _r.log(1e6)) / _r.log(2.0)

    if dispersion is None:
        dispersion = 0.05
    dispersion = np.asarray(dispersion, dtype=np.float64)
    isna = np.isnan(dispersion)
    if isna.all():
        dispersion = np.asarray(0.05)
    elif isna.any():
        dispersion = dispersion.copy()
        dispersion[isna] = _r.mean(dispersion, na_rm=True)
    dmat = compressDispersions(ym, dispersion)
    wmat = compressWeights(ym, weights)
    omat = compressOffsets(ym, offset, lib_size)
    pmat = compressPrior(ym, prior_count)
    # maxit and tol are formals(mglmOneGroup)$maxit / $tol
    return _C.ave_log_cpm(ym, omat.to_c(), pmat.to_c(), dmat.to_c(), wmat.to_c(), 50, 1e-10, int(nthreads))


def cpm(y, normalized_lib_sizes=True, log=False, prior_count=2, shrunk=True, nthreads=1):
    """Counts per Million

    Compute counts per million (CPM).

    Parameters
    ----------
    y : DGEList, DGEGLM or DGELRT
        A DGEList gives observed CPM values; a fitted model (DGEGLM from ``glmFit`` or ``glmQLFit``,
        DGELRT from ``glmQLFTest``) gives fitted CPM values.
    normalized_lib_sizes : bool, default True
        Whether to use normalized library sizes (``lib.size * norm.factors``). DGEList only.
    log : bool, default False
        If ``True`` then log2 values are returned.
    prior_count : float or array-like, default 2
        Average count to be added to each observation to avoid taking log of zero. Used only if
        ``log=True`` and ``y`` is a DGEList.
    shrunk : bool, default True
        Whether to use the shrunk coefficients of a fitted model. If ``True`` then the usual
        coefficients from the fitted object are used. If ``False`` then the unshrunk coefficients are
        used. DGEGLM and DGELRT only.
    nthreads : int, default 1
        Number of OpenMP threads to use for the genewise computations.

    Returns
    -------
    DataFrame
        CPM values, on the log2 scale if ``log=True``, of the same size as the counts, with the genes
        and samples as its index and columns.

    Notes
    -----
    CPM values are useful descriptive measures for the expression level of a gene. By default, the
    normalized library sizes are used in the computation.

    If log-values are computed, then a small count, given by ``prior_count`` but scaled to be
    proportional to the library size, is added to the counts to avoid taking the log of zero.

    For a DGEList with a prior offset matrix (``offset.prior``), observation-specific effective
    library sizes are obtained by adding ``offset.prior`` to the log library sizes. If the DGEList has
    an ordinary offset matrix (``offset``), then ``exp(offset)`` is used as the effective library
    sizes instead of ``lib.size``, ``norm.factors`` and ``offset.prior``.

    For a fitted model, the fitted CPM values come from the linear predictor ``coefficients`` times
    the transposed ``design``. If ``shrunk=True``, then the CPM values reflect the ``prior.count``
    input to the original linear model fit. If ``shrunk=False``, then they are computed with
    ``prior.count=0``. Note that the latter could result in taking the log of near-zero values if
    ``log=True``.

    ``aveLogCPM(y)``, ``rowMeans(cpm(y, log=True))`` and ``log2(rowMeans(cpm(y)))`` all give slightly
    different results.

    R documentation by Davis McCarthy, Gordon Smyth, Yunshun Chen, Aaron Lun; adapted for edgerpy.
    """
    if isinstance(y, DGEList):
        counts = y["counts"]
        if "offset" in y:
            out = _cpm_default(counts, offset=y["offset"], log=log, prior_count=prior_count)
        else:
            lib = y["samples"]["lib.size"].to_numpy(dtype=np.float64)
            if normalized_lib_sizes:
                lib = lib * y["samples"]["norm.factors"].to_numpy(dtype=np.float64)
            out = _cpm_default(counts, lib_size=lib, offset_prior=y.get("offset.prior"), log=log,
                               prior_count=prior_count, nthreads=nthreads)
        return frame(out, counts.index, counts.columns)
    if isinstance(y, (DGEGLM, DGELRT)):
        beta = y["coefficients"] if shrunk else y["unshrunk.coefficients"]
        design = y["design"]
        eta = _r.matprod(mat(beta), mat(design).T)
        if log:
            out = (eta + _r.log(1e6)) / _r.log(2.0)
        else:
            out = _r.exp(eta + _r.log(1e6))
        return frame(out, getattr(beta, "index", None), getattr(design, "index", None))
    raise TypeError("cpm() needs a DGEList, DGEGLM or DGELRT")


def _cpm_default(y, lib_size=None, offset=None, offset_prior=None, log=False, prior_count=2, nthreads=1):
    """R cpm.default() on a count matrix -> column-major ndarray (internal)."""
    ym = mat(as_matrix(y))
    if ym.size == 0:
        return ym
    ymin = np.min(ym)
    if np.isnan(ymin):
        raise ValueError("NA counts not allowed")
    if ymin < 0:
        raise ValueError("Negative counts not allowed")

    # offset takes precedence over lib.size and offset.prior; offset.prior adds to log(lib.size)
    if offset is None:
        if lib_size is None:
            lib_size = _r.colSums(ym)
        if offset_prior is not None:
            op = mat(offset_prior)
            if op.shape != ym.shape:
                raise ValueError("counts and offset.prior must have equal dimensions.")
            lr = np.broadcast_to(_r.log(lib_size).reshape(1, -1), ym.shape)
            lib_size = _r.exp(lr + op)
    else:
        offset = np.asarray(mat(offset), dtype=np.float64)
        if offset.ndim == 2:
            if offset.shape != ym.shape:
                raise ValueError("counts and offset must have equal dimensions")
        elif offset.size != ym.shape[1]:
            raise ValueError("if offset is a vector, its length must be the number of samples")
        if lib_size is not None:
            warnings.warn("lib.size is ignored in the presence of offset")
        if offset_prior is not None:
            warnings.warn("offset.prior is ignored in the presence of offset")
        lib_size = _r.exp(offset)

    lib_size = np.asarray(lib_size, dtype=np.float64)
    minlib = np.min(lib_size)
    if np.isnan(minlib):
        raise ValueError("NA library sizes not allowed")
    if minlib <= 0:
        raise ValueError("Library sizes should be greater than zero")
    lmat = makeCompressedMatrix(lib_size, ym.shape, byrow=True)

    if log:
        prior = compressPrior(ym, prior_count)
        return _C.calculate_cpm_log(ym, lmat.to_c(), prior.to_c(), int(nthreads))
    return _C.calculate_cpm_raw(ym, lmat.to_c(), int(nthreads))
