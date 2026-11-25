"""R primitives with R's exact numerical semantics.

The ported edgeR/limma code calls these instead of NumPy wherever NumPy's
result can differ from R's: transcendental functions, R's ^, long-double
reductions, sorting/ranking rules and BLAS-backed products. Plain IEEE
operations (+, -, *, /, comparisons, abs, floor) are left to NumPy, which
gives the same bits.
"""

import math

import numpy as np

from . import _C

DBL_EPSILON = np.finfo(np.float64).eps


# ------------------------------------------------------------------
# elementwise

def _dbl(x):
    return np.asarray(x, dtype=np.float64)


def _map1(fun, x):
    a = _dbl(x)
    if a.ndim == 0:
        return float(fun(a.reshape(1))[0])
    return fun(a)


def log(x):
    """R log(x)."""
    return _map1(_C.r_log, x)


def log2(x):
    """R log2(x)."""
    return _map1(_C.r_log2, x)


def log10(x):
    """R log10(x)."""
    return _map1(_C.r_log10, x)


def exp(x):
    """R exp(x)."""
    return _map1(_C.r_exp, x)


def log1p(x):
    """R log1p(x)."""
    return _map1(_C.r_log1p, x)


def sqrt(x):
    """R sqrt(x)."""
    return _map1(_C.r_sqrt, x)


def lgamma(x):
    """R lgamma(x) (nmath lgammafn)."""
    return _map1(_C.r_lgamma, x)


def trigamma(x):
    """R trigamma(x)."""
    return _map1(_C.r_trigamma, x)


def _broadcast(*args):
    arrs = np.broadcast_arrays(*[_dbl(a) for a in args])
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a).reshape(-1) for a in arrs]
    return shape, flat


def pow(x, y):
    """R x ^ y."""
    shape, (a, b) = _broadcast(x, y)
    r = _C.r_pow(a, b).reshape(shape)
    return float(r) if r.ndim == 0 else r


def pf(q, df1, df2, lower_tail=True, log_p=False):
    """R pf(q, df1, df2, lower.tail, log.p)."""
    shape, (a, b, c) = _broadcast(q, df1, df2)
    r = _C.r_pf(a, b, c, bool(lower_tail), bool(log_p)).reshape(shape)
    return float(r) if r.ndim == 0 else r


def qf(p, df1, df2, lower_tail=True, log_p=False):
    """R qf(p, df1, df2, lower.tail, log.p)."""
    shape, (a, b, c) = _broadcast(p, df1, df2)
    r = _C.r_qf(a, b, c, bool(lower_tail), bool(log_p)).reshape(shape)
    return float(r) if r.ndim == 0 else r


def sign(x):
    """R sign(x)."""
    x = _dbl(x)
    return np.where(x > 0, 1.0, np.where(x == 0, 0.0, np.where(np.isnan(x), x, -1.0)))


# ------------------------------------------------------------------
# reductions (long double accumulation, column-major order as in R)

def _flatF(x):
    return np.ravel(_dbl(x), order="F")


def rsum(x, na_rm=False):
    """R sum(x, na.rm) for doubles."""
    return _C.r_sum(_flatF(x), bool(na_rm))


def mean(x, na_rm=False):
    """R mean(x, na.rm) for doubles."""
    x = _flatF(x)
    if na_rm:
        x = x[~np.isnan(x)]
    return _C.r_mean(x)


def cumsum(x):
    """R cumsum(x) for doubles."""
    return _C.r_cumsum(_flatF(x))


def _matrix(m):
    return np.asfortranarray(np.asarray(m, dtype=np.float64))


def colSums(m):
    return _C.r_colsum(_matrix(m), 0)


def colMeans(m):
    return _C.r_colsum(_matrix(m), 1)


def rowSums(m, na_rm=False):
    return _C.r_colsum(_matrix(m), 2, na_rm)


def rowMeans(m, na_rm=False):
    return _C.r_colsum(_matrix(m), 3, na_rm)


def rowsum(x, group):
    """R rowsum(x, group, reorder=FALSE) for doubles.

    Returns (sums, unique_groups) with groups in order of first appearance.
    """
    x = _dbl(x)
    vec = x.ndim == 1
    if vec:
        x = x.reshape(-1, 1)
    codes, uniq = factorize_first(group)
    out = _C.r_rowsum(np.asfortranarray(x), codes.astype(np.int32), len(uniq))
    return out, uniq


def factorize_first(values):
    """Integer codes and unique values in order of first appearance (R unique/match)."""
    seen = {}
    codes = np.empty(len(values), dtype=np.int64)
    uniq = []
    for i, v in enumerate(values):
        key = _key(v)
        c = seen.get(key)
        if c is None:
            c = len(uniq)
            seen[key] = c
            uniq.append(v)
        codes[i] = c
    return codes, uniq


def _key(v):
    if isinstance(v, float) and math.isnan(v):
        return ("NaN",)
    return v


def cummax(x):
    """R cummax(x) for doubles without NaN."""
    return np.maximum.accumulate(_dbl(x))


def pmin(x, y):
    return np.minimum(_dbl(x), _dbl(y))


def pmax(x, y):
    return np.maximum(_dbl(x), _dbl(y))


def which_min(x):
    """R which.min (0-based, first occurrence, NaN ignored); None if empty."""
    x = _dbl(x)
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    return int(idx[np.argmin(x[ok])])


def which_max(x):
    x = _dbl(x)
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    return int(idx[np.argmax(x[ok])])


# ------------------------------------------------------------------
# order statistics

def order(*keys, decreasing=False):
    """R order(...) (0-based): stable, NaN last, as R's radix order.

    decreasing may be a bool or one bool per key. Numeric keys only.
    """
    if isinstance(decreasing, bool):
        decreasing = [decreasing] * len(keys)
    cols = []
    for k, dec in zip(keys, decreasing):
        k = _dbl(k).reshape(-1)
        cols.append(-k if dec else k)
    # np.lexsort sorts by the last key first and is stable; NaN go last
    return np.lexsort(tuple(reversed(cols)))


def order_str(*keys):
    """R order() for character / mixed keys in the C locale (stable)."""
    n = len(keys[0])
    return np.array(sorted(range(n), key=lambda i: tuple(_skey(k[i]) for k in keys)), dtype=np.int64)


def _skey(v):
    if isinstance(v, str):
        return (0, v.encode("utf-8"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return (2, b"")
    return (1, float(v))


def rank_average(x):
    """R rank(x) with ties.method='average' (no NaN).

    Sort once, find the runs of equal values, and give every member of a run
    covering sorted positions first..last (1-based) the rank (first + last) / 2.
    Average ranks are exact halves or integers, so this equals R's rank()
    bit for bit; it is vectorized because rank() runs twice per sample in TMM.
    """
    x = _dbl(x)
    n = x.size
    if n < 2:
        return np.arange(1, n + 1, dtype=np.float64)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    last = np.append(np.flatnonzero(xs[1:] != xs[:-1]), n - 1)  # 0-based last position of each run
    first = np.concatenate(([0], last[:-1] + 1))
    r = np.empty(n, dtype=np.float64)
    r[o] = np.repeat((first + last + 2) / 2.0, last - first + 1)
    return r


def median(x):
    """R median.default(x) for doubles (NA if any NaN)."""
    x = _dbl(x).reshape(-1)
    if np.isnan(x).any():
        return float("nan")
    n = x.size
    if n == 0:
        return float("nan")
    half = (n + 1) // 2
    xs = np.sort(x)
    if n % 2 == 1:
        return float(xs[half - 1])
    return _C.r_mean(np.ascontiguousarray(xs[half - 1:half + 1]))


def psort(x, k):
    """R sort.int(x, partial=k) for one 1-based index k (NaNs removed, as na.last=NA)."""
    x = _dbl(x).reshape(-1)
    return _C.r_psort(x[~np.isnan(x)], int(k))


def quantile7(x, p):
    """R quantile(x, probs=p, type=7) for a single prob, no NaN."""
    x = np.sort(_dbl(x).reshape(-1))
    n = x.size
    index = 1 + max(n - 1, 0) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    qs = x[lo - 1]
    if index > lo and x[hi - 1] != qs:
        h = index - lo
        qs = (1 - h) * qs + h * x[hi - 1]
    return float(qs)


def p_adjust(p, method="BH"):
    """R p.adjust(p, method) for BH/fdr, BY, holm, hochberg, hommel, bonferroni, none."""
    p = _dbl(p).reshape(-1)
    out = p.copy()
    nna = ~np.isnan(p)
    pp = p[nna]
    n = lp = pp.size
    if n <= 1 or method == "none":
        return out
    if method == "fdr":
        method = "BH"
    if n == 2 and method == "hommel":
        method = "hochberg"
    if method == "bonferroni":
        res = np.minimum(1.0, n * pp)
    elif method == "holm":
        i = np.arange(1, lp + 1, dtype=np.float64)
        o = order(pp)
        ro = np.argsort(o, kind="stable")
        res = np.minimum(1.0, np.maximum.accumulate((n + 1 - i) * pp[o]))[ro]
    elif method in ("hochberg", "BH", "BY"):
        i = np.arange(lp, 0, -1, dtype=np.float64)
        o = order(pp, decreasing=True)
        ro = np.argsort(o, kind="stable")
        if method == "hochberg":
            v = (n + 1 - i) * pp[o]
        elif method == "BH":
            v = n / i * pp[o]
        else:
            q = rsum(1.0 / np.arange(1, n + 1, dtype=np.float64))
            v = q * n / i * pp[o]
        res = np.minimum(1.0, np.minimum.accumulate(v))[ro]
    elif method == "hommel":
        i = np.arange(1, n + 1, dtype=np.float64)
        o = order(pp)
        ps = pp[o]
        ro = np.argsort(o, kind="stable")
        q = np.full(n, np.min(n * ps / i))
        pa = q.copy()
        for j in range(n - 1, 1, -1):
            ij = np.arange(n - j + 1)
            i2 = np.arange(n - j + 1, n)
            q1 = np.min(j * ps[i2] / np.arange(2, j + 1, dtype=np.float64))
            q[ij] = np.minimum(j * ps[ij], q1)
            q[i2] = q[n - j]
            pa = np.maximum(pa, q)
        res = np.maximum(pa, ps)[ro]
    else:
        raise NotImplementedError(f"p.adjust method {method!r}")
    out[nna] = res
    return out


def make_unique(names, sep="."):
    """R make.unique(names)."""
    names = [str(n) for n in names]
    seen = set(names)
    cnt = {}
    out = []
    used = set()
    for nm in names:
        if nm not in used:
            used.add(nm)
            out.append(nm)
            continue
        k = cnt.get(nm, 1)
        while f"{nm}{sep}{k}" in seen or f"{nm}{sep}{k}" in used:
            k += 1
        new = f"{nm}{sep}{k}"
        cnt[nm] = k + 1
        used.add(new)
        out.append(new)
    return out


def num2str(x):
    """R as.character() of a double (15 significant digits)."""
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.15g}"


# ------------------------------------------------------------------
# linear algebra and smoothing through R's compiled routines

def matprod(x, y):
    """R x %*% y for numeric matrices (2-D arrays)."""
    return _C.r_matprod(_matrix(x), _matrix(y))


def eigen_symmetric(x):
    """R eigen(x, symmetric=TRUE): La_rs (LAPACK dsyevr), then values and vectors reversed."""
    z = _C.r_la_rs(_matrix(x))
    return {"values": z["values"][::-1].copy(), "vectors": np.asfortranarray(z["vectors"][:, ::-1])}


def axis_ticks(lo, hi, style="r"):
    """R's default axis for data range (lo, hi): dict(usr, axp, at) as par("usr"), par("xaxp"), axTicks()."""
    return _C.r_axis_ticks(float(lo), float(hi), style)


def qr(x, tol=1e-7):
    """R qr(x, tol) (LINPACK): dict(qr, rank, qraux, pivot (1-based))."""
    return _C.r_qr(_matrix(x), float(tol))


def qr_qty(q, y):
    return _C.r_qr_qy(q["qr"], q["qraux"], q["rank"], _matrix(y), True)


def qr_qy(q, y):
    return _C.r_qr_qy(q["qr"], q["qraux"], q["rank"], _matrix(y), False)


def backsolve(r, x, k=None, upper_tri=True, transpose=False):
    r = _matrix(r)
    x = _matrix(x)
    if k is None:
        k = r.shape[1]
    return _C.r_backsolve(r, x, int(k), bool(upper_tri), bool(transpose))


def hat(x, intercept=True):
    """stats::hat(x, intercept)."""
    x = _matrix(x)
    if intercept:
        x = np.asfortranarray(np.column_stack([np.ones(x.shape[0]), x]))
    n = x.shape[0]
    q = qr(x)
    I = np.asfortranarray(np.eye(n, q["rank"]))
    qy = qr_qy(q, I)
    return rowSums(qy * qy)


def optimize(f, lower, upper, tol=None):
    """R optimize(f, c(lower, upper), tol)$minimum (Brent, edgeR brent_fmin)."""
    if tol is None:
        tol = DBL_EPSILON ** 0.25
    return _C.r_brent_fmin(float(lower), float(upper), f, float(tol))


def lowess_sorted(x, y, f, iter, delta):
    """stats::lowess C call on x already sorted (y in the same order)."""
    return _C.r_lowess(np.ascontiguousarray(_dbl(x)), np.ascontiguousarray(_dbl(y)), float(f), int(iter), float(delta))


def weighted_lowess_c(x, y, w, span, iterations, delta):
    """limma .Call('weighted_lowess', ...): (fitted, robustness weights)."""
    return _C.r_weighted_lowess(np.ascontiguousarray(_dbl(x)), np.ascontiguousarray(_dbl(y)),
                                np.ascontiguousarray(_dbl(w)), float(span), int(iterations), float(delta))


def lm_wfit(x, y, w, tol=1e-7):
    """stats::lm.wfit for positive weights: dict(fitted, residuals, ...)."""
    x = _matrix(x)
    y = _dbl(y)
    w = _dbl(w)
    if np.any(w == 0):
        raise NotImplementedError("lm_wfit: zero weights are not used by the pipelines")
    wts = sqrt(w)
    z = _C.r_cdqrls(np.asfortranarray(x * wts[:, None]), np.ascontiguousarray(y * wts), float(tol))
    resid = z["residuals"] / wts
    z["residuals"] = resid
    z["fitted.values"] = y - resid
    return z
