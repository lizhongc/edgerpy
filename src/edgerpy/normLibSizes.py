"""normLibSizes (R: normLibSizes.R)."""

import math
import warnings

import numpy as np

from . import _r
from .classes import DGEList
from .helpers import as_matrix, mat

METHODS = ("TMM", "TMMwsp", "RLE", "upperquartile", "none")


def normLibSizes(object, method="TMM", refColumn=None, logratioTrim=0.3, sumTrim=0.05, doWeighting=True,
                 Acutoff=-1e10, p=0.75, renormalize_offset=False):
    """Normalize Library Sizes

    Calculate scaling factors to convert the raw library sizes for a set of sequenced samples into
    normalized effective library sizes.

    Parameters
    ----------
    object : DGEList
        The counts and their library sizes (``y["samples"]["lib.size"]``). If it has an
        ``offset.prior`` matrix (offsets relative to the log library sizes), the TMM method uses it.
    method : {"TMM", "TMMwsp", "RLE", "upperquartile", "none"}, default "TMM"
        Normalization method to be used.
    refColumn : int, optional
        Column (0-based) to use as reference for ``method="TMM"`` and ``method="TMMwsp"``.
    logratioTrim : float, default 0.3
        The fraction (0 to 0.5) of observations to be trimmed from each tail of the distribution of
        log-ratios (M-values) before computing the mean. Used by ``method="TMM"`` for each pair of
        samples.
    sumTrim : float, default 0.05
        The fraction (0 to 0.5) of observations to be trimmed from each tail of the distribution of
        A-values before computing the mean. Used by ``method="TMM"`` for each pair of samples.
    doWeighting : bool, default True
        Whether to use (asymptotic binomial precision) weights when computing the mean M-values. Used
        by ``method="TMM"`` for each pair of samples.
    Acutoff : float, default -1e10
        Minimum cutoff applied to A-values. Count pairs with lower A-values are ignored. Used by
        ``method="TMM"`` for each pair of samples.
    p : float, default 0.75
        Which quantile (between 0 and 1) of the counts should be used by
        ``method="upperquartile"``.
    renormalize_offset : bool, default False
        Whether to proceed with library size normalization even if ``object`` contains an ``offset``
        matrix. In that case, the offset matrix is converted to a prior offset matrix
        (``offset.prior``) so that the library sizes can be normalized separately. Otherwise a
        DGEList with an ``offset`` matrix is returned unchanged, with a message.

    Returns
    -------
    DGEList
        A copy of ``object`` with the library scaling factors stored in
        ``object["samples"]["norm.factors"]``.

    Notes
    -----
    This function computes scaling factors to convert observed library sizes into normalized library
    sizes, also called "effective library sizes". The effective library sizes for use in downstream
    analysis are ``lib.size * norm.factors`` where ``lib.size`` contains the original library sizes
    and ``norm.factors`` is the vector of scaling factors computed by this function.

    The TMM method implements the trimmed mean of M-values method proposed by Robinson and Oshlack
    (2010). By default, the M-values are weighted according to inverse variances, as computed by the
    delta method for logarithms of binomial random variables. If ``refColumn`` is unspecified, then
    the column whose count-per-million upper quartile is closest to the mean upper quartile is set as
    the reference library.

    The TMMwsp method stands for "TMM with singleton pairing". This is a variant of TMM that is
    intended to have more stable performance when the counts have a high proportion of zeros. In the
    TMM method, genes that have zero count in either library are ignored when comparing pairs of
    libraries. In the TMMwsp method, the positive counts from such genes are reused to increase the
    number of features by which the libraries are compared. The singleton positive counts are paired
    up between the libraries in decreasing order of size and then a slightly modified TMM method is
    applied to the re-ordered libraries. If ``refColumn`` is unspecified, then the column with
    largest sum of square-root counts is used as the reference library.

    RLE is the scaling factor method proposed by Anders and Huber (2010). We call it "relative log
    expression", as the median library is calculated from the geometric mean of all columns and the
    median ratio of each sample to the median library is taken as the scale factor.

    The upperquartile method is the upper-quartile normalization method of Bullard et al (2010), in
    which the scale factors are calculated from the 75% quantile of the counts for each library,
    after removing genes that are zero in all libraries. The idea is generalized here to allow
    normalization by any quantile of the count distributions.

    If ``method="none"``, then the normalization factors are set to 1.

    For symmetry, normalization factors are adjusted to multiply to 1. Rows of ``object`` that have
    zero counts for all columns are removed before normalization factors are computed. The number of
    such rows does not affect the estimated normalization factors.

    ``offset.prior`` is used by the TMM method only; the other methods ignore it, with a message. An
    ``offset.prior`` together with genes that are zero in every sample raises
    ``NotImplementedError``: edgeR drops those rows from the counts but not from its library-size
    matrix. Filter them first.

    ``normLibSizes`` became the new name for ``calcNormFactors`` in edgeR 3.42.0.

    R documentation by Mark Robinson, Gordon Smyth, Yunshun Chen; adapted for edgerpy.
    """
    if not isinstance(object, DGEList):
        raise TypeError("normLibSizes() needs a DGEList")
    if "offset" in object:
        if "offset.prior" in object:
            raise ValueError("Object contains both offset and offset.prior matrices.")
        if renormalize_offset:
            print("Converting offset matrix to offset.prior with normalized library sizes")
            object = object.copy()
            lib = object["samples"]["lib.size"].to_numpy(dtype=np.float64)
            object["offset.prior"] = mat(object["offset"]) - _r.log(lib)[None, :]
        else:
            print("Data object contains offset matrix so skipping library size normalization")
            return object
    f = _normLibSizes_default(object["counts"], lib_size=object["samples"]["lib.size"].to_numpy(dtype=np.float64),
                              offset_prior=object.get("offset.prior"), method=method, refColumn=refColumn,
                              logratioTrim=logratioTrim, sumTrim=sumTrim, doWeighting=doWeighting, Acutoff=Acutoff,
                              p=p)
    out = object.copy()
    s = out["samples"].copy()
    s["norm.factors"] = np.asarray(f, dtype=np.float64)
    out["samples"] = s
    return out


def _normLibSizes_default(object, lib_size=None, offset_prior=None, method="TMM", refColumn=None, logratioTrim=0.3,
                          sumTrim=0.05, doWeighting=True, Acutoff=-1e10, p=0.75):
    """R normLibSizes.default(): the normalization factors of a count matrix."""
    x = mat(as_matrix(object))
    minx = np.min(x)
    if np.isnan(minx):
        raise ValueError("NA counts not allowed")
    if minx < 0:
        raise ValueError("Negative counts not allowed")
    ngenes, nsamples = x.shape
    if lib_size is None:
        lib_size = _r.colSums(x)
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64).reshape(-1)
        if np.isnan(lib_size).any():
            raise ValueError("NA lib.sizes not permitted")
        if lib_size.size != nsamples:
            if lib_size.size > 1:
                warnings.warn("normLibSizes: length(lib.size) doesn't match number of samples")
            lib_size = np.resize(lib_size, nsamples)

    # combine offset.prior with library sizes
    if offset_prior is not None:
        offset_prior = mat(as_matrix(offset_prior)).astype(np.float64)
        if offset_prior.shape != x.shape:
            raise ValueError("data object and offset.prior are not of same dimensions")
        lib_size_matrix = _r.exp(offset_prior + _r.log(lib_size)[None, :])

    if method == "TMMwzp":
        method = "TMMwsp"
        print("TMMwzp has been renamed to TMMwsp")
    if method not in METHODS:
        raise ValueError(f"'method' should be one of {METHODS}")
    if offset_prior is not None and method != "TMM":
        print(f"ignoring offset.prior for {method} method")

    allzero = np.sum(x > 0, axis=1) == 0
    if allzero.any():
        if offset_prior is not None and method == "TMM":
            # edgeR 4.99.12 removes the all-zero rows from x but not from lib.size.matrix, so R recycles vectors of
            # unequal length there; that result is not reproduced
            raise NotImplementedError("offset.prior with all-zero rows: edgeR does not drop these rows from the "
                                      "library size matrix; filter all-zero rows first")
        x = x[~allzero, :]
    if x.shape[0] == 0 or nsamples == 1:
        method = "none"

    if method == "TMM":
        if refColumn is None:
            if offset_prior is None:
                f75 = np.array([_r.quantile7(x[:, i], 0.75) / lib_size[i] for i in range(nsamples)])
            else:
                f75 = np.array([_r.quantile7(x[:, i] / lib_size_matrix[:, i], 0.75) for i in range(nsamples)])
            if _r.median(f75) < 1e-20:
                refColumn = _r.which_max(_r.colSums(_r.sqrt(x)))
            else:
                refColumn = _r.which_min(np.abs(f75 - _r.mean(f75)))
        Yref = x[:, refColumn]
        if offset_prior is None:
            Nref = lib_size[refColumn]
            f = np.array([_calcFactorTMM(x[:, i], Yref, lib_size[i], Nref, logratioTrim, sumTrim, doWeighting,
                                         Acutoff) for i in range(nsamples)])
        else:
            Nref = lib_size_matrix[:, refColumn]
            f = np.array([_calcFactorTMM(x[:, i], Yref, lib_size_matrix[:, i], Nref, logratioTrim, sumTrim,
                                         doWeighting, Acutoff) for i in range(nsamples)])
    elif method == "TMMwsp":
        if refColumn is None:
            refColumn = _r.which_max(_r.colSums(_r.sqrt(x)))
        f = np.array([_calcFactorTMMwsp(x[:, i], x[:, refColumn], lib_size[i], lib_size[refColumn],
                                        logratioTrim, sumTrim, doWeighting, Acutoff) for i in range(nsamples)])
    elif method == "RLE":
        f = _calcFactorRLE(x) / lib_size
    elif method == "upperquartile":
        f = _calcFactorQuantile(x, lib_size, p)
    else:
        f = np.ones(nsamples)

    return f / _r.exp(_r.mean(_r.log(f)))


def _calcFactorRLE(data):
    gm = _r.exp(_r.rowMeans(_r.log(data)))
    pos = gm > 0
    return np.array([_r.median(data[pos, j] / gm[pos]) for j in range(data.shape[1])])


def _calcFactorQuantile(data, lib_size, p=0.75):
    f = np.array([_r.quantile7(data[:, j], p) for j in range(data.shape[1])], dtype=np.float64)
    if np.min(f) == 0:
        warnings.warn("One or more quantiles are zero")
    return f / lib_size


def _calcFactorTMM(obs, ref, nO, nR, logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10):
    obs = np.asarray(obs, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logR = _r.log2((obs / nO) / (ref / nR))
        absE = (_r.log2(obs / nO) + _r.log2(ref / nR)) / 2
        v = (nO - obs) / nO / obs + (nR - ref) / nR / ref
    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > Acutoff)
    logR, absE, v = logR[fin], absE[fin], v[fin]
    if logR.size == 0 or np.max(np.abs(logR)) < 1e-6:
        return 1.0
    n = logR.size
    loL = math.floor(n * logratioTrim) + 1
    hiL = n + 1 - loL
    loS = math.floor(n * sumTrim) + 1
    hiS = n + 1 - loS
    rL = _r.rank_average(logR)
    rA = _r.rank_average(absE)
    keep = (rL >= loL) & (rL <= hiL) & (rA >= loS) & (rA <= hiS)
    if doWeighting:
        f = _r.rsum(logR[keep] / v[keep], na_rm=True) / _r.rsum(1 / v[keep], na_rm=True)
    else:
        f = _r.mean(logR[keep], na_rm=True)
    if math.isnan(f):
        f = 0.0
    return _r.pow(2.0, f)


def _calcFactorTMMwsp(obs, ref, libsize_obs, libsize_ref, logratioTrim=0.3, sumTrim=0.05, doWeighting=True,
                      Acutoff=-1e10):
    obs = np.asarray(obs, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    eps = 1e-14
    npos = 2 * (obs > eps).astype(np.int64) + (ref > eps).astype(np.int64)
    rm = (npos == 0) | np.isnan(obs) | np.isnan(ref)
    if rm.any():
        obs, ref, npos = obs[~rm], ref[~rm], npos[~rm]
    zero_obs = npos == 1
    zero_ref = npos == 2
    k = zero_obs | zero_ref
    n_single = min(int(zero_obs.sum()), int(zero_ref.sum()))
    if n_single > 0:
        refk = np.sort(ref[k])[::-1][:n_single]
        obsk = np.sort(obs[k])[::-1][:n_single]
        obs = np.concatenate([obs[~k], obsk])
        ref = np.concatenate([ref[~k], refk])
    else:
        obs, ref = obs[~k], ref[~k]
    n = obs.size
    if n == 0:
        return 1.0
    obs_p = obs / libsize_obs
    ref_p = ref / libsize_ref
    M = _r.log2(obs_p / ref_p)
    A = 0.5 * _r.log2(obs_p * ref_p)
    if np.max(np.abs(M)) < 1e-6:
        return 1.0
    obs_p_shrunk = (obs + 0.5) / (libsize_obs + 0.5)
    ref_p_shrunk = (ref + 0.5) / (libsize_ref + 0.5)
    M_shrunk = _r.log2(obs_p_shrunk / ref_p_shrunk)
    o_M = _r.order(M, M_shrunk)
    o_A = _r.order(A)
    loM = int(n * logratioTrim) + 1
    hiM = n + 1 - loM
    keep_M = np.zeros(n, dtype=bool)
    keep_M[o_M[loM - 1:hiM]] = True
    loA = int(n * sumTrim) + 1
    hiA = n + 1 - loA
    keep_A = np.zeros(n, dtype=bool)
    keep_A[o_A[loA - 1:hiA]] = True
    keep = keep_M & keep_A
    M = M[keep]
    if doWeighting:
        obs_p = obs_p[keep]
        ref_p = ref_p[keep]
        v = (1 - obs_p) / obs_p / libsize_obs + (1 - ref_p) / ref_p / libsize_ref
        w = (1 + 1e-6) / (v + 1e-6)
        TMM = _r.rsum(w * M) / _r.rsum(w)
    else:
        TMM = _r.mean(M)
    return _r.pow(2.0, TMM)
