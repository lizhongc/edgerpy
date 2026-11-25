"""Shared helpers of edgerpy's pipeline functions.

The Python counterpart of include/helper.h: building blocks used across the
edgeR and limma ports. Each section names the R source it ports; numerical
work goes through the R primitives in _r.py.

Sections:
  - Matrix, factor and design conversions (edgerpy (R's as.matrix, factor, model.matrix(~group), coef handling))
  - modelMatrix (R stats R/models.R, src/model.c)
  - CompressedMatrix and the .compress* helpers (edgeR R/makeCompressedMatrix.R)
  - subsetListOfArrays (limma R/subsetting.R)
  - nonEstimable (limma R/lmfit.R)
  - contrastAsCoef (limma R/contrastAsCoef.R)
  - loessFit, weightedLowess, chooseLowessSpan (limma R/loessFit.R, R/weightedLowess.R)
"""

import numbers
import re

import numpy as np
import pandas as pd

from . import _r
from .classes import _pos


# ==========================================================================
# Matrix, factor and design conversions
# R source: edgerpy (R's as.matrix, factor, model.matrix(~group), coef handling)
# ==========================================================================
# Matrix and factor conversions shared by the ported R functions.

def as_matrix(x, rownames=None, colnames=None):
    """R as.matrix() for count-like input -> DataFrame over one column-major block.

    Integer data is stored as int32 (as R stores integer counts) when the
    values fit; everything else becomes float64. Data that is already a
    column-major float64/int32 block is wrapped without copying.
    """
    if isinstance(x, pd.DataFrame):
        idx, cols = x.index, x.columns
        a = x.to_numpy()
    else:
        a = np.asarray(x)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        idx = pd.RangeIndex(a.shape[0]) if rownames is None else rownames
        cols = pd.RangeIndex(a.shape[1]) if colnames is None else colnames
    a = _block(a)
    return pd.DataFrame(a, index=idx, columns=cols, copy=False)


def _block(a):
    if a.dtype == np.int32 or a.dtype == np.float64:
        return a if a.flags.f_contiguous else np.asfortranarray(a)
    if a.dtype == bool:
        return np.asfortranarray(a.astype(np.int32))
    if a.dtype.kind in "iu":
        if a.size == 0 or (a.min() >= -2**31 + 1 and a.max() <= 2**31 - 1):
            return np.asfortranarray(a.astype(np.int32))
        return np.asfortranarray(a.astype(np.float64))
    return np.asfortranarray(a.astype(np.float64))


def mat(df):
    """The column-major array behind a matrix DataFrame (no copy)."""
    if isinstance(df, pd.DataFrame):
        return df.to_numpy()
    return np.asarray(df)


def frame(a, index, columns):
    """Wrap a column-major array as a matrix DataFrame without copying."""
    return pd.DataFrame(np.asfortranarray(a), index=index, columns=columns, copy=False)


def r_level(v):
    """R as.character() for a factor level value."""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, numbers.Number):
        return _r.num2str(v)
    return str(v)


def factor(x):
    """R factor(x): levels sorted (numeric order for numbers, C locale for strings)."""
    if isinstance(x, pd.Series):
        x = x.array
    if isinstance(x, pd.Categorical):
        cats = [r_level(c) for c in x.categories]
        return pd.Categorical([r_level(v) for v in x], categories=cats)
    vals = list(x)
    uniq = list(dict.fromkeys(vals))
    if all(isinstance(v, numbers.Number) and not isinstance(v, bool) for v in uniq):
        uniq = sorted(uniq)
    else:
        uniq = sorted((r_level(v) for v in uniq), key=lambda s: s.encode("utf-8"))
        vals = [r_level(v) for v in vals]
        return pd.Categorical(vals, categories=uniq)
    labels = [r_level(v) for v in uniq]
    lut = dict(zip(uniq, labels))
    return pd.Categorical([lut[v] for v in vals], categories=labels)


def treatment_design(group, prefix):
    """R model.matrix(~group) for one factor: '(Intercept)' + prefix+level columns."""
    g = factor(group) if not isinstance(group, pd.Categorical) else group
    cats = list(g.categories)
    codes = np.asarray(g.codes)
    n = len(codes)
    X = np.zeros((n, len(cats)), dtype=np.float64, order="F")
    X[:, 0] = 1.0
    for k in range(1, len(cats)):
        X[:, k] = (codes == k).astype(np.float64)
    cols = ["(Intercept)"] + [prefix + c for c in cats[1:]]
    return pd.DataFrame(X, index=[str(i + 1) for i in range(n)], columns=cols, copy=False)


def design_matrix(design, nlib):
    """as.matrix(design) -> (float64 column-major array, column names or None)."""
    if isinstance(design, pd.DataFrame):
        return np.asfortranarray(design.to_numpy(dtype=np.float64)), [str(c) for c in design.columns], design
    a = np.asfortranarray(np.asarray(design, dtype=np.float64))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    df = pd.DataFrame(a, copy=False)
    return a, None, df


def coef_index(coef, colnames, ncol):
    """Python coef (0-based int, name, or list of them) -> 1-based R indices and names."""
    if coef is None:
        coef = [ncol - 1]
    if isinstance(coef, (str, numbers.Integral)):
        coef = [coef]
    coef = list(dict.fromkeys(coef))
    out = []
    for c in coef:
        if isinstance(c, str):
            if colnames is None or c not in colnames:
                raise ValueError("One or more named coef arguments do not match a column of the design matrix.")
            out.append(colnames.index(c) + 1)
        else:
            # edgeR 4.99.16 crashes in C for coef <= 0 (R numbering); check the range before any native call
            if not 0 <= int(c) < ncol:
                raise IndexError(f"coef {c} out of range: the design matrix has {ncol} columns (0-based)")
            out.append(int(c) + 1)
    return out


# ==========================================================================
# modelMatrix
# R source: stats R/models.R (model.matrix.default), src/model.c (termsform, TermCode, modelmatrix)
# ==========================================================================
# R's formula algebra and design-matrix construction with treatment contrasts.

_FORMULA_NAME = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")
_FORMULA_TOKEN = re.compile(r"[A-Za-z.][A-Za-z0-9._]*|\d+|\S")


def _formula_terms(formula):
    """R terms(formula) -> (variables in order of appearance, terms, intercept).

    Terms are sorted variable-index tuples ordered by degree, otherwise in
    formula order (keep.order=FALSE). Supports + - * : ( ) and 0/1.
    """
    toks = _FORMULA_TOKEN.findall(formula)
    if toks[:1] == ["~"]:
        toks = toks[1:]
    variables, pos, intercept = [], 0, True

    def at(*ops):
        return pos < len(toks) and toks[pos] in ops

    def take():
        nonlocal pos
        if pos == len(toks):
            raise ValueError(f"incomplete formula {formula!r}")
        pos += 1
        return toks[pos - 1]

    def unique(terms):                      # TrimRepeats
        return list(dict.fromkeys(terms))

    def plus_minus(parity):                 # PlusTerms, DeleteTerms (parity flips under '-', as EncodeVars)
        terms = [] if at("-") else times(parity)
        while at("+", "-"):
            if take() == "+":
                terms = unique(terms + times(parity))
            else:
                drop = times(not parity)
                terms = [t for t in terms if t not in drop]
        return terms

    def times(parity):                      # CrossTerms
        terms = colon(parity)
        while at("*"):
            take()
            right = colon(parity)
            terms = unique(terms + right + [l | r for l in terms for r in right])
        return terms

    def colon(parity):                      # InteractTerms
        terms = atom(parity)
        while at(":"):
            take()
            right = atom(parity)
            terms = unique([l | r for l in terms for r in right])
        return terms

    def atom(parity):
        nonlocal intercept
        tok = take()
        if tok == "(":
            terms = plus_minus(parity)
            if take() != ")":
                raise ValueError(f"unbalanced parentheses in {formula!r}")
            return terms
        if tok in ("0", "1"):
            intercept = parity if tok == "1" else not parity
            return []
        if tok != "." and _FORMULA_NAME.fullmatch(tok):
            if tok not in variables:
                variables.append(tok)
            return [frozenset([variables.index(tok)])]
        raise ValueError(f"unsupported formula element {tok!r} in {formula!r}")

    terms = plus_minus(True)
    if pos != len(toks):
        raise ValueError(f"unsupported formula element {toks[pos]!r} in {formula!r}")
    terms = sorted((tuple(sorted(t)) for t in terms if t), key=len)
    return variables, terms, intercept


def modelMatrix(formula, data):
    """Construct Design Matrices

    Create a design (or model) matrix, by expanding factors to a set of dummy variables (treatment
    contrasts) and expanding interactions similarly, as R's ``model.matrix(formula, data)`` does.

    Parameters
    ----------
    formula : str
        A one-sided model formula such as ``"~ group + batch"`` or ``"~ 0 + group"``, built from
        variable names, the operators ``+ - * : ( )``, and ``0`` or ``1`` for the intercept.
        Function calls such as ``log(x)`` and the operators ``/``, ``^`` and ``.`` are not supported
        and raise ``ValueError``.
    data : DataFrame or dict of columns
        The variables named in ``formula``. Other columns may be present, in any order. Strings and
        unordered pandas Categoricals are factors, booleans are two-level factors (named as in R, so a
        column ``flag`` gives the design column ``flagTRUE``), and numbers are covariates.

    Returns
    -------
    DataFrame
        The design matrix, one row per row of ``data`` (with R's row names "1", "2", ... when
        ``data`` has a default index) and one column per coefficient. ``attrs["assign"]`` gives, for
        each column, the term in the formula that gave rise to it: 0 for the intercept (if any) and
        1, 2, ... for the terms in order. If there are factors, ``attrs["contrasts"]`` maps each
        factor name to ``"contr.treatment"``.

    Notes
    -----
    Factor levels are those of R's ``factor()``: strings are sorted in C-locale order (as R does with
    ``LC_COLLATE=C``), and a pandas Categorical keeps its categories, unused ones included. Pass a
    Categorical to fix the order yourself. Factors must have two or more levels; ordered factors
    (R's ``contr.poly`` coding) and missing values (which R's ``model.frame`` would drop) raise
    ``ValueError``.

    Column names are shorter than R's: ``Intercept``, factor levels without the variable name
    (``trt``, and ``trt:b2`` for an interaction, where R has ``grouptrt`` and ``grouptrt:batchb2``),
    and numeric covariates by name (``age``). Logicals keep R's form (``flagTRUE``), and so does a
    factor whose level names clash with another variable's column names or with ``Intercept``
    (``groupb1``).

    In an interaction term, the variable whose levels vary fastest is the first one to appear in the
    formula (and not in the term), so in ``~ a + b + b:a`` the interaction will have ``a`` varying
    fastest. Terms are ordered by degree (main effects first), otherwise in formula order. Within a
    term, a factor is coded by treatment contrasts when the term without it is the intercept or is
    contained in an earlier term, and otherwise by indicator (dummy) columns for all its levels; in a
    formula without an intercept, the first factor gets indicator columns.

    R documentation by the R Core Team; adapted for edgerpy.
    """
    variables, terms, intercept = _formula_terms(formula)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    n = len(data)
    index = [str(i + 1) for i in data.index] if isinstance(data.index, pd.RangeIndex) else data.index

    levels, values, logicals = {}, {}, set()
    for v, name in enumerate(variables):
        x = data[name]
        if x.isna().any():
            raise ValueError(f"missing values in '{name}' (R's model.frame would drop those rows)")
        if pd.api.types.is_bool_dtype(x):
            f = pd.Categorical(np.where(x, "TRUE", "FALSE"), categories=["FALSE", "TRUE"])
            logicals.add(v)
        elif pd.api.types.is_numeric_dtype(x):
            values[v] = np.asarray(x, dtype=np.float64)
            continue
        elif isinstance(x.dtype, pd.CategoricalDtype) and x.cat.ordered:
            raise ValueError(f"'{name}' is an ordered factor; its contr.poly coding is not supported")
        else:
            f = factor(x)
        if len(f.categories) < 2:
            raise ValueError("contrasts can be applied only to factors with 2 or more levels")
        levels[v] = (list(f.categories), np.asarray(f.codes))

    # factor pattern (TermCode): 1 = contrasts when the rest of the term is
    # empty or contained in a preceding term, otherwise 2 = indicators
    codes = [{v: 1 if not (set(t) - {v}) or any(set(t) - {v} <= set(p) for p in terms[:j]) else 2 for v in t}
             for j, t in enumerate(terms)]
    if not intercept:                       # first factor found gets indicator coding
        first = next(((j, v) for j, t in enumerate(terms) for v in t if v in levels), None)
        if first is not None:
            codes[first[0]][first[1]] = 2

    # column labels per variable level: short level names, R's variable+level
    # for logicals and for factors whose levels clash with other labels
    labels = {v: [name + l for l in levels[v][0]] if v in logicals else list(levels[v][0]) if v in levels
              else [name] for v, name in enumerate(variables)}
    clash = [v for v in levels if v not in logicals
             and set(labels[v]) & {"Intercept"}.union(*(labels[u] for u in labels if u != v))]
    for v in clash:
        labels[v] = [variables[v] + l for l in levels[v][0]]

    names, assign, cols = [], [], []
    if intercept:
        names.append("Intercept")
        assign.append(0)
        cols.append(np.ones(n))
    for j, (t, code) in enumerate(zip(terms, codes), start=1):
        tn, tx = [""], [None]
        for v in t:                         # the first variable varies fastest
            if v in levels:
                lev, idx = levels[v]
                keep = range(len(lev)) if code[v] == 2 else range(1, len(lev))
                vn = [labels[v][k] for k in keep]
                vx = [(idx == k).astype(np.float64) for k in keep]
            else:
                vn, vx = labels[v], [values[v]]
            tn = [a + ":" + b if a else b for b in vn for a in tn]
            tx = [c if a is None else c * a for c in vx for a in tx]
        names += tn
        assign += [j] * len(tn)
        cols += tx

    X = np.zeros((n, len(cols)), dtype=np.float64, order="F")
    for k, c in enumerate(cols):
        X[:, k] = c
    out = pd.DataFrame(X, index=index, columns=names, copy=False)
    out.attrs["assign"] = assign
    if levels:
        out.attrs["contrasts"] = {variables[v]: "contr.treatment" for v in levels}
    return out


# ==========================================================================
# CompressedMatrix and the .compress* helpers
# R source: edgeR R/makeCompressedMatrix.R
# ==========================================================================
# CompressedMatrix and the .compress* helpers (R: makeCompressedMatrix.R).
#
# A CompressedMatrix stores a scalar, a row vector, a column vector or a full
# matrix and stands for a matrix of size `dims`. The kernels read it in place
# (py_exports.c builds the same cmx struct as R's SEXPtocmx2).

class CompressedMatrix:
    """Compressed Matrix of Offsets or Weights

    A matrix stored in compressed form: a scalar, a row vector repeated down the rows, a column vector
    repeated across the columns, or a full matrix.

    Parameters
    ----------
    values : array-like
        The stored values: a 1 x 1 array (scalar), a 1 x ncol array (repeated down the rows), an
        nrow x 1 array (repeated across the columns) or the full nrow x ncol matrix.
    dims : tuple of two ints
        The dimensions (nrow, ncol) of the matrix represented.
    repeat_row : bool
        Whether the stored row is repeated down all rows.
    repeat_col : bool
        Whether the stored column is repeated across all columns.

    Attributes
    ----------
    values : ndarray
        The stored values (float64, column-major).
    dims : tuple of two ints
        The dimensions of the matrix represented; also available as ``shape``.
    repeat_row, repeat_col : bool
        As above. Both ``True`` for a scalar.

    Notes
    -----
    CompressedMatrix objects are used throughout edgeR to save space in storing offsets and (to a
    lesser extent) weights. This is because, for routine analyses, offsets are the same for all genes,
    so it makes little sense to expand them to the full dimensions of the count matrix. The fits store
    their ``offset`` and ``weights`` this way, and the compiled kernels read them in place.

    ``as_matrix()`` expands the full matrix; ``min()`` and ``max()`` give the smallest and largest
    stored values; ``subset_rows(rows)`` and ``subset_cols(cols)`` subset by integer positions,
    keeping the compression.

    R documentation by Aaron Lun; adapted for edgerpy.
    """

    __slots__ = ("values", "dims", "repeat_row", "repeat_col")

    def __init__(self, values, dims, repeat_row, repeat_col):
        self.values = np.asfortranarray(np.asarray(values, dtype=np.float64))
        self.dims = (int(dims[0]), int(dims[1]))
        self.repeat_row = bool(repeat_row)
        self.repeat_col = bool(repeat_col)

    @property
    def shape(self):
        return self.dims

    def to_c(self):
        """Argument tuple for edgerpy._C (values, nrow, ncol, repeat_row, repeat_col)."""
        return (self.values, self.dims[0], self.dims[1], self.repeat_row, self.repeat_col)

    def as_matrix(self):
        """R as.matrix(): expand to the full matrix."""
        nr, nc = self.dims
        v = self.values.reshape(-1, order="F")
        if self.repeat_row:
            return np.asfortranarray(np.tile(v, (nr, 1)) if not self.repeat_col else np.full((nr, nc), v[0]))
        if self.repeat_col:
            return np.asfortranarray(np.tile(v.reshape(-1, 1), (1, nc)))
        return np.asfortranarray(self.values)

    def min(self):
        return float(np.min(self.values)) if self.values.size else float("nan")

    def max(self):
        return float(np.max(self.values)) if self.values.size else float("nan")

    def subset_rows(self, rows):
        """R x[i, ] keeping compression (rows: integer positions)."""
        vals = self.values
        if not self.repeat_row:
            vals = vals[rows, :]
        return CompressedMatrix(vals, (len(rows), self.dims[1]), self.repeat_row, self.repeat_col)

    def subset_cols(self, cols):
        vals = self.values
        if not self.repeat_col:
            vals = vals[:, cols]
        return CompressedMatrix(vals, (self.dims[0], len(cols)), self.repeat_row, self.repeat_col)

    def __repr__(self):
        kind = {(True, True): "scalar", (True, False): "row-repeated",
                (False, True): "column-repeated", (False, False): "full"}[(self.repeat_row, self.repeat_col)]
        return f"CompressedMatrix {self.dims[0]} x {self.dims[1]} ({kind})\n{self.values}"


def makeCompressedMatrix(x, dims=None, byrow=True):
    """R makeCompressedMatrix(x, dims, byrow)."""
    if isinstance(x, CompressedMatrix):
        return x
    rrow = rcol = False
    a = np.asarray(getattr(x, "to_numpy", lambda: x)(), dtype=np.float64)
    if a.ndim == 2:
        if dims is not None:
            nr, nc = a.shape
            if nr == 1 and nc == 1:
                rrow = rcol = True
            elif nr == 1 and nc >= 2:
                if nc != dims[1]:
                    raise ValueError("'dims[2]' should equal length of row vector 'x'")
                if byrow:
                    rrow = True
                else:
                    dims = a.shape
            elif nr >= 2 and nc == 1:
                if nr != dims[0]:
                    raise ValueError("'dims[1]' should equal length of column vector 'x'")
                if not byrow:
                    rcol = True
                else:
                    dims = a.shape
            else:
                dims = a.shape
        else:
            dims = a.shape
    elif a.size == 1:
        rrow = rcol = True
        a = a.reshape(1, 1)
    else:
        a = a.reshape(-1)
        if not byrow:
            if dims[0] != a.size:
                raise ValueError("'dims[1]' should equal length of 'x'")
            a = a.reshape(-1, 1)
            rcol = True
        else:
            if dims[1] != a.size:
                raise ValueError("'dims[2]' should equal length of 'x'")
            a = a.reshape(1, -1)
            rrow = True
    return CompressedMatrix(a, dims, rrow, rcol)


def _min(x):
    if isinstance(x, CompressedMatrix):
        return x.min()
    a = np.asarray(x, dtype=np.float64)
    return float(np.min(a)) if a.size else float("nan")


def _dims(y):
    return tuple(y.shape)


def compressOffsets(y, offset=None, lib_size=None):
    """R .compressOffsets(y, offset, lib.size)."""
    if isinstance(offset, CompressedMatrix):
        return offset
    if offset is None:
        if lib_size is None:
            lib_size = _r.colSums(y)
        offset = _r.log(lib_size)
    offset = np.asarray(getattr(offset, "to_numpy", lambda: offset)(), dtype=np.float64)
    m = _min(offset)
    if not np.isfinite(m):
        raise ValueError("offsets must be finite values")
    return makeCompressedMatrix(offset, _dims(y), byrow=True)


def compressWeights(y, weights=None):
    """R .compressWeights(y, weights)."""
    if isinstance(weights, CompressedMatrix):
        return weights
    if weights is None:
        weights = 1.0
    weights = np.asarray(getattr(weights, "to_numpy", lambda: weights)(), dtype=np.float64)
    m = _min(weights)
    if np.isnan(m):
        raise ValueError("NA weights not allowed")
    if m <= 0:
        raise ValueError("Weights must be positive")
    return makeCompressedMatrix(weights, _dims(y), byrow=True)


def compressPrior(y, prior_count):
    """R .compressPrior(y, prior.count)."""
    if isinstance(prior_count, CompressedMatrix):
        return prior_count
    prior_count = np.asarray(prior_count, dtype=np.float64)
    m = _min(prior_count)
    if np.isnan(m):
        raise ValueError("NA prior counts not allowed")
    if m < 0:
        raise ValueError("Negative prior counts not allowed")
    return makeCompressedMatrix(prior_count, _dims(y), byrow=False)


def compressDispersions(y, dispersion):
    """R .compressDispersions(y, dispersion)."""
    if isinstance(dispersion, CompressedMatrix):
        return dispersion
    dispersion = np.asarray(getattr(dispersion, "to_numpy", lambda: dispersion)(), dtype=np.float64)
    m = _min(dispersion)
    if np.isnan(m):
        raise ValueError("NA dispersions not allowed")
    if m < 0:
        raise ValueError("Negative dispersions not allowed")
    return makeCompressedMatrix(dispersion, _dims(y), byrow=False)


# ==========================================================================
# subsetListOfArrays
# R source: limma R/subsetting.R
# ==========================================================================
# limma subsetListOfArrays (R: limma/R/subsetting.R) for edgerpy objects.

def _length(v):
    if v is None:
        return 0
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return v.size
    if isinstance(v, CompressedMatrix):
        return v.dims[0] * v.dims[1]
    if isinstance(v, str):
        return 1
    return np.asarray(v).size


def _rows(v, i):
    if isinstance(v, CompressedMatrix):
        return v.subset_rows(i)
    if isinstance(v, pd.DataFrame):
        return _frame_take(v, i, None)
    a = np.asarray(v)
    return np.asfortranarray(a[i]) if a.ndim == 2 else a[i]


def _cols(v, j):
    if isinstance(v, CompressedMatrix):
        return v.subset_cols(j)
    if isinstance(v, pd.DataFrame):
        return _frame_take(v, None, j)
    return np.asfortranarray(np.asarray(v)[:, j])


def _frame_take(df, i, j):
    """Row/column take that keeps a single column-major numeric block."""
    if df.shape[1] and all(df.dtypes == df.dtypes.iloc[0]) and df.dtypes.iloc[0].kind in "fiub":
        a = df.to_numpy()
        if i is not None:
            a = a[i, :]
        if j is not None:
            a = a[:, j]
        return pd.DataFrame(np.asfortranarray(a),
                            index=df.index if i is None else df.index[i],
                            columns=df.columns if j is None else df.columns[j], copy=False)
    return df.iloc[i if i is not None else slice(None), j if j is not None else slice(None)]


def subsetListOfArrays(obj, i, j, IJ, IX, I, JX):
    """Subset the recognised components of a list-like object (0-based i, j)."""
    out = obj.copy()
    I = [a for a in I if a in obj and _length(obj[a]) > 1]
    if i is None:
        IX, I = (), ()
        if j is None:
            IJ = ()
    else:
        i = _pos(i, obj.rownames)
    if j is None:
        JX = ()
    else:
        j = _pos(j, obj.colnames)
    for a in IJ:
        if a in out:
            v = out[a]
            if i is not None:
                v = _rows(v, i)
            if j is not None:
                v = _cols(v, j)
            out[a] = v
    for a in IX:
        if a in out:
            out[a] = _rows(out[a], i)
    for a in I:
        out[a] = np.asarray(out[a])[i]
    for a in JX:
        if a in out:
            out[a] = _rows(out[a], j)
    return out


# ==========================================================================
# nonEstimable
# R source: limma R/lmfit.R
# ==========================================================================
# limma nonEstimable (R: limma/R/lmfit.R).

def nonEstimable(x, colnames=None):
    """Names of non-estimable columns of x, or None if x has full column rank."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    p = x.shape[1]
    q = _r.qr(x)
    if q["rank"] < p:
        n = colnames if colnames is not None else [str(k + 1) for k in range(p)]
        idx = list(range(q["rank"], p))
        notest = [n[q["pivot"][k] - 1] for k in idx]
        return [s if s != "" else str(k + 1) for s, k in zip(notest, idx)]
    return None


# ==========================================================================
# contrastAsCoef
# R source: limma R/contrastAsCoef.R
# ==========================================================================
# limma contrastAsCoef (R: limma/R/contrastAsCoef.R).

def contrastAsCoef(design, contrast, colnames=None, contrast_colnames=None, first=True):
    """Reform a Design Matrix so that Contrasts Become Coefficients

    Reform a design matrix so that one or more coefficients from the new matrix correspond to
    specified contrasts of coefficients from the old matrix.

    Parameters
    ----------
    design : DataFrame or ndarray
        Numeric design matrix.
    contrast : array-like
        Numeric matrix with rows corresponding to columns of the design matrix (coefficients) and
        columns containing contrasts. May be a vector if there is only one contrast.
    colnames : list of str, optional
        Not used.
    contrast_colnames : list of str, optional
        Names for the columns of the new design matrix that hold the contrasts (R's
        ``colnames(contrast)``). Defaults to ``"C"`` followed by the column number of each retained
        contrast in ``contrast`` (e.g. ``C1``, ``C3``).
    first : bool, default True
        Whether the coefficients corresponding to contrasts should be the first columns (``True``)
        or the last columns (``False``) of the new design matrix.

    Returns
    -------
    dict
        With keys:

        ``"design"``
            The reformed design matrix (ndarray).
        ``"colnames"``
            Its column names: the contrast names, and ``Q1``, ``Q2``, ... for the other columns.
        ``"coef"``
            The columns of the design matrix which hold the meaningful coefficients, numbered from 1
            as in R.
        ``"qr"``
            QR decomposition of the contrast matrix (dict with ``qr``, ``rank``, ``qraux`` and
            ``pivot``, as R's ``qr()``).

    Notes
    -----
    If the contrasts contained in the columns of ``contrast`` are linearly dependent, then superfluous
    columns are dropped until the remaining matrix has full column rank. The number of retained
    contrasts is stored in ``qr["rank"]`` and the retained columns are given by ``qr["pivot"]``. A
    contrast whose length differs from the number of design columns, or a contrast matrix that is all
    zero, raises ``ValueError``.

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    design = np.asfortranarray(np.asarray(design, dtype=np.float64))
    contrast = np.asarray(contrast, dtype=np.float64)
    if contrast.ndim == 1:
        contrast = contrast.reshape(-1, 1)
    if design.shape[1] != contrast.shape[0]:
        raise ValueError("Length of contrast doesn't match ncol(design)")
    QRc = _r.qr(contrast)
    ncontrasts = QRc["rank"]
    if ncontrasts == 0:
        raise ValueError("contrast is all zero")
    coef = list(range(1, ncontrasts + 1))
    designT = _r.qr_qty(QRc, design.T)
    designT[:ncontrasts, :] = _r.backsolve(QRc["qr"], designT, k=ncontrasts)
    design = np.asfortranarray(designT.T)
    cn = [f"Q{k + 1}" for k in range(design.shape[1])]
    ccn = contrast_colnames
    if ccn is None:
        ccn = [f"C{QRc['pivot'][c - 1]}" for c in coef]
    for k, c in enumerate(coef):
        cn[c - 1] = ccn[k]
    if not first:
        nc = design.shape[1]
        idx = [k for k in range(nc) if k + 1 not in coef] + [c - 1 for c in coef]
        design = np.asfortranarray(design[:, idx])
        cn = [cn[k] for k in idx]
        coef = list(reversed([nc - c + 1 for c in coef]))
    return {"design": design, "colnames": cn, "coef": coef, "qr": QRc}


# ==========================================================================
# loessFit, weightedLowess, chooseLowessSpan
# R source: limma R/loessFit.R, R/weightedLowess.R
# ==========================================================================
# limma loessFit, weightedLowess, chooseLowessSpan (R: loessFit.R, weightedLowess.R).

def chooseLowessSpan(n=1000, small_n=50, min_span=0.3, power=1 / 3):
    """limma chooseLowessSpan()."""
    return min(min_span + (1 - min_span) * _r.pow(small_n / n, power), 1.0)


def weightedLowess(x, y, weights=None, delta=None, npts=200, span=0.3, iterations=4):
    """limma weightedLowess(..., output.style='loess'): dict(fitted, weights, residuals, delta)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    if delta is None:
        npts = int(npts + 0.5)
        if npts < 1:
            raise ValueError("number of points should be a positive integer")
        if npts >= len(xs):
            delta = 0.0
        else:
            dx = np.sort(np.diff(xs))
            cumrange = _r.cumsum(dx)
            numclusters = np.arange(npts, dtype=np.float64)
            idx = len(dx) - np.arange(npts)          # 1-based positions length(dx) - numclusters
            delta = float(np.min(cumrange[idx - 1] / (npts - numclusters)))
    fit, rob = _r.weighted_lowess_c(xs, y[o], weights[o], span, int(iterations), float(delta))
    fitted = np.empty_like(fit)
    fitted[o] = fit
    w = np.empty_like(rob)
    w[o] = rob
    return {"fitted": fitted, "weights": w, "residuals": y - fitted, "delta": delta}


def loessFit(y, x, weights=None, span=0.3, iterations=4, min_weight=1e-5, max_weight=1e5,
             equal_weights_as_null=True, method="weightedLowess"):
    """limma loessFit(): dict(fitted, residuals)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    if len(x) != n:
        raise ValueError("y and x have different lengths")
    fitted = np.full(n, np.nan)
    residuals = np.full(n, np.nan)
    obs = np.isfinite(y) & np.isfinite(x)
    xobs, yobs = x[obs], y[obs]
    nobs = len(yobs)
    if nobs == 0:
        return {"fitted": fitted, "residuals": residuals}
    if span < 1 / nobs:
        fitted[obs] = y[obs]
        residuals[obs] = 0
        return {"fitted": fitted, "residuals": residuals}
    if min_weight < 0:
        min_weight = 0
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError("y and weights have different lengths")
        wobs = weights[obs].copy()
        wobs[np.isnan(wobs)] = 0
        wobs = np.maximum(wobs, min_weight)
        wobs = np.minimum(wobs, max_weight)
        if equal_weights_as_null and np.max(wobs) - np.min(wobs) < 1e-15:
            weights = None
    if weights is None:
        o = np.argsort(xobs, kind="stable")
        xs = xobs[o]
        delta = 0.01 * (xs[-1] - xs[0])          # lowess() default 0.01 * diff(range(x))
        lo_y = _r.lowess_sorted(xs, yobs[o], span, iterations - 1, delta)
        fo = np.empty(nobs)
        fo[o] = lo_y
        fitted[obs] = fo
        residuals[obs] = yobs - fo
        return {"fitted": fitted, "residuals": residuals}
    nwobs = nobs if min_weight > 0 else int(np.sum(wobs > 0))
    if nwobs < 4 + 1 / span:
        if nwobs == 1:
            fitted[obs] = yobs[wobs > 0]
            residuals[obs] = yobs - fitted[obs]
        else:
            fit = _r.lm_wfit(np.column_stack([np.ones(nobs), xobs]), yobs, wobs)
            fitted[obs] = fit["fitted.values"]
            residuals[obs] = fit["residuals"]
        return {"fitted": fitted, "residuals": residuals}
    if method != "weightedLowess":
        raise NotImplementedError("loessFit: only method='weightedLowess' is ported")
    fit = weightedLowess(xobs, yobs, weights=wobs, span=span, iterations=iterations, npts=200)
    fitted[obs] = fit["fitted"]
    residuals[obs] = fit["residuals"]
    return {"fitted": fitted, "residuals": residuals}
