"""edgeR and limma data objects (R: classes.R, S3methods.R, subsetting.R).

Every edgeR class is an S4 class with representation("list"): a named list
whose components are added and removed freely. RList reproduces that:

  obj["s2.post"]       component by its exact R name (KeyError if absent)
  obj.s2_post          R's $: '_' maps to '.', None when absent
  obj.x = None         removes the component, as x$a <- NULL
  len(obj)             number of components
  obj.shape            R's dim();  obj.dimnames  R's dimnames()
  obj[i, j]            R's [ (subsetListOfArrays); i, j 0-based, mask or names
"""

import numpy as np
import pandas as pd


class RList(dict):
    """A named R list with S3-style $ access."""

    # components that subsetting knows about (subsetting.R): row x col,
    # row x anything, row only, col x anything
    _IJ = ()
    _IX = ()
    _I = ()
    _JX = ()
    _rows_only = False
    _dim_from = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        key = name if name in self else name.replace("_", ".")
        return self.get(key)

    def __setattr__(self, name, value):
        key = name if name in self else name.replace("_", ".")
        if value is None:
            self.pop(key, None)
        else:
            self[key] = value

    def __delattr__(self, name):
        self.pop(name if name in self else name.replace("_", "."), None)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("two subscripts required")
            return self.subset(*key)
        return dict.__getitem__(self, key)

    def copy(self):
        return type(self)(self)

    # --- R dim / dimnames ---
    def _dim_obj(self):
        return self.get(self._dim_from)

    @property
    def shape(self):
        m = self._dim_obj()
        return (0, 0) if m is None else tuple(m.shape)

    @property
    def dimnames(self):
        m = self._dim_obj()
        if m is None:
            return (None, None)
        return (list(m.index), list(m.columns))

    @property
    def rownames(self):
        return self.dimnames[0]

    @property
    def colnames(self):
        return self.dimnames[1]

    # --- R [ ---
    def subset(self, i=None, j=None):
        """R object[i, j]; None (or slice(None)) means 'missing'."""
        from .helpers import subsetListOfArrays
        i = None if isinstance(i, slice) and i == slice(None) else i
        j = None if isinstance(j, slice) and j == slice(None) else j
        if self._rows_only and j is not None:
            raise IndexError(f"Subsetting columns not allowed for {type(self).__name__} object.")
        return subsetListOfArrays(self, i, j, self._IJ, self._IX, self._I, self._JX)

    def __repr__(self):
        # limma LargeDataObject show(): class, then head of each component
        lines = [f'An object of class "{type(self).__name__}"']
        for k, v in self.items():
            lines.append(f"${k}")
            lines.append(_print_head(v))
            lines.append("")
        return "\n".join(lines)


def _print_head(x):
    if isinstance(x, pd.DataFrame):
        s = repr(x.head(5))
        if len(x) > 5:
            s += f"\n{len(x) - 5} more rows ..."
        return s
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.size > 5:
        return f"{np.array2string(x.reshape(-1)[:5])}\n{x.size - 5} more elements ..."
    return repr(x)


class DGEList(RList):
    """Digital Gene Expression Data - Class

    A list-based class for storing read counts and associated information from sequencing
    technologies. Objects are created by ``DGEList()`` and by the catch* readers.

    Attributes
    ----------
    counts : DataFrame
        Read counts, one row for each gene and one column for each sample.
    samples : DataFrame
        One row for each sample, with columns ``group``, ``lib.size`` and ``norm.factors`` containing
        the group labels (a Categorical), library sizes and normalization factors. Other columns can be
        added to give more detailed sample information. Optional columns ``technical.weights`` and
        ``biological.weights`` hold sample weights, one per sample, which are used by ``glmFit`` and
        ``glmQLFit`` (see ``sampleWeights``).
    genes : DataFrame, optional
        Annotation for the genes, one row for each gene.

    Other components may be present, for example ``offset.prior`` from the catch* readers and the
    dispersions from ``estimateDisp``.

    Notes
    -----
    Rows correspond to genomic features and columns to samples. The features are called genes, but
    the rows can correspond to any genomic feature for which read counts are meaningful, such as
    transcripts, tags, exons, exon-junctions, or genomic intervals.

    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    A DGEList can also be treated as a matrix: ``shape``, ``dimnames``, ``rownames`` and
    ``colnames`` are those of ``counts``. ``y[i, j]`` and ``y.subset(i, j, keep_lib_sizes=True)``
    subset genes and samples, as R's ``[``, with 0-based positions, boolean masks or names
    (``None`` or ``:`` keeps all). With ``keep_lib_sizes=False`` the library sizes are recomputed
    from the kept genes. Subsetting samples drops unused group levels, and the matrices in
    ``y["other"]`` are subset with ``counts``.

    R documentation by the edgeR team (first created by Mark Robinson); adapted for edgerpy.
    """
    _IJ = ("counts", "pseudo.counts", "offset", "offset.prior", "weights")
    _IX = ("genes",)
    _I = ("AveLogCPM", "trended.dispersion", "tagwise.dispersion", "prior.n", "prior.df")
    _JX = ("samples",)
    _dim_from = "counts"

    def subset(self, i=None, j=None, keep_lib_sizes=True):
        """R y[i, j, keep.lib.sizes] (i, j 0-based positions, masks or names)."""
        from . import _r
        from .DGEList import dropEmptyLevels
        i = None if isinstance(i, slice) and i == slice(None) else i
        j = None if isinstance(j, slice) and j == slice(None) else j
        out = RList.subset(self, i, j)
        if i is not None and not keep_lib_sizes:
            s = out["samples"].copy()
            s["lib.size"] = _r.colSums(out["counts"].to_numpy())
            out["samples"] = s
        if j is not None:
            s = out["samples"].copy()
            s["group"] = dropEmptyLevels(s["group"])
            out["samples"] = s
        # edgeR 4.99.16: each matrix in y$other (e.g. the catch* effective lengths and TPM) is subset as counts
        if isinstance(out.get("other"), dict) and (i is not None or j is not None):
            from .helpers import _cols, _pos, _rows
            ii = None if i is None else _pos(i, self.rownames)
            jj = None if j is None else _pos(j, self.colnames)
            other = {}
            for k, v in out["other"].items():
                if ii is not None:
                    v = _rows(v, ii)
                if jj is not None:
                    v = _cols(v, jj)
                other[k] = v
            out["other"] = other
        return out


class PCList(RList):
    """Paired Counts Data Object - Class

    A list-based class for storing paired counts and associated information from sequencing
    technologies. Objects are created by ``PCList()``.

    Attributes
    ----------
    counts : DataFrame
        Read counts, one row for each feature and one column for each sample.
    counts2 : DataFrame
        Paired read counts, one row for each feature and one column for each sample.
    samples : DataFrame
        One row for each sample, with column ``group`` containing the group labels (a Categorical).
        Other columns can be added to give more detailed sample information.
    genes : DataFrame, optional
        Annotation for the features, one row for each feature.

    Notes
    -----
    Rows correspond to genomic features and columns to samples. The features can be genes, genomic
    loci, or any feature for which paired read counts are meaningful.

    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    A PCList can also be treated as a matrix: ``shape``, ``dimnames``, ``rownames`` and
    ``colnames`` are those of ``counts``. ``y[i, j]`` subsets features and samples, as R's ``[``,
    with 0-based positions, boolean masks or names; subsetting samples drops unused group levels.

    R documentation by the edgeR team (first created by Lizhong Chen); adapted for edgerpy.
    """
    _IJ = ("counts", "counts2", "weights")
    _IX = ("genes",)
    _I = ("AveLogCPM",)
    _JX = ("samples",)
    _dim_from = "counts"

    def subset(self, i=None, j=None):
        from .DGEList import dropEmptyLevels
        out = RList.subset(self, i, j)
        if j is not None and not (isinstance(j, slice) and j == slice(None)):
            s = out["samples"].copy()
            s["group"] = dropEmptyLevels(s["group"])
            out["samples"] = s
        return out


_FIT_IX = ("counts", "counts2", "offset", "weights", "genes", "coefficients", "fitted.values", "unshrunk.coefficients")
_FIT_I = ("AveLogCPM", "dispersion", "prior.n", "prior.df", "s2.post", "s2.prior", "df.prior", "df.residual",
          "df.residual.zeros", "df.residual.adj", "deviance", "deviance.adj", "iter", "failed")


class DGEGLM(RList):
    """Digital Gene Expression Generalized Linear Model Results - Class

    A list-based class for storing the results of a negative binomial GLM fit to each gene, as
    produced by ``glmFit`` or ``glmQLFit``.

    Attributes
    ----------
    coefficients : DataFrame
        Coefficients from fitting the model defined by the design matrix to each gene, one column per
        design column. Coefficients are on the natural log scale.
    unshrunk.coefficients : DataFrame
        Coefficients of the fits without the prior count.
    fitted.values : DataFrame
        The fitted (expected) values for each gene and sample.
    deviance : ndarray
        The deviance from the model fit to each gene.
    df.residual : ndarray
        The residual degrees of freedom for each gene.
    iter, failed : ndarray
        Number of iterations, and whether the fit failed to converge, for each gene.
    design : DataFrame
        The design matrix.
    offset : CompressedMatrix
        Offsets included in the GLMs for each gene (log effective library sizes, or the DGEList's
        offsets).
    weights : CompressedMatrix, optional
        Weights used in the GLM fitting for each gene.
    dispersion : ndarray
        The negative binomial dispersion used for each gene (one value, or one per gene).
    prior.count : float
        The prior count added for the shrunk coefficients.
    AveLogCPM : ndarray
        Average log2 counts per million for each gene.
    counts : DataFrame
        The counts.
    samples, genes : DataFrame
        The sample and gene annotation of the DGEList.

    ``glmQLFit`` adds the quasi-likelihood components ``df.residual.adj``, ``deviance.adj``,
    ``df.prior``, ``s2.prior``, ``s2.post`` and ``average.ql.dispersion`` (see ``glmQLFit``).

    Notes
    -----
    Rows correspond to genomic features (genes, transcripts, exons, ...) and columns to the
    coefficients of the linear model.

    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    ``shape``, ``dimnames``, ``rownames`` and ``colnames`` are those of ``coefficients``.
    ``fit[i, :]`` subsets genes (0-based positions, boolean masks or names); subsetting columns
    raises ``IndexError``.

    R documentation by the edgeR team (first created by Davis McCarthy); adapted for edgerpy.
    """
    _IX = _FIT_IX
    _I = _FIT_I
    _rows_only = True
    _dim_from = "coefficients"


class DGEBIN(RList):
    """Digital Gene Expression Binomial Model Results - Class

    A list-based class for storing the results of a binomial generalized linear model fit to each
    gene, as produced by ``binFit`` or ``binQLFit``. It is a sibling of ``DGEGLM`` (the negative
    binomial GLM fit): a binomial fit is deliberately not a ``DGEGLM``, so that ``binQLFTest``
    accepts only a ``DGEBIN`` and the negative binomial tests do not.

    Attributes
    ----------
    coefficients : DataFrame
        Coefficients from fitting the model defined by the design matrix to each gene, one column per
        design column. Coefficients are on the natural log scale.
    unshrunk.coefficients : DataFrame
        Coefficients of the unshrunk fits; present only when ``prior_count > 0``.
    fitted.values : DataFrame
        Fitted (expected) proportions from the binomial GLM for each gene and sample.
    deviance : ndarray
        The deviance from the model fit to each gene.
    df.residual : ndarray
        The residual degrees of freedom for each gene.
    iter, failed : ndarray
        Number of iterations, and whether the fit failed to converge, for each gene.
    design : DataFrame
        The design matrix.
    counts, counts2 : DataFrame
        The two matrices of counts (successes and failures) as input.
    weights : DataFrame, optional
        Prior weights used in the GLM fitting.
    nthreads : int
        The number of OpenMP threads used for the fit.
    AveLogCPM : ndarray
        Average log2 counts per million of the total coverage for each gene.
    samples, genes : DataFrame
        The sample and gene annotation of the PCList.

    ``binQLFit`` adds the quasi-likelihood components ``df.residual.adj``, ``deviance.adj``,
    ``df.prior``, ``s2.prior``, ``s2.post`` and ``average.ql.dispersion`` (see ``binQLFit``).

    Notes
    -----
    Rows correspond to genomic features and columns to the coefficients of the linear model.

    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    ``shape``, ``dimnames``, ``rownames`` and ``colnames`` are those of ``coefficients``.
    ``fit[i, :]`` subsets genes (0-based positions, boolean masks or names); subsetting columns
    raises ``IndexError``.

    R documentation by Lizhong Chen and Gordon Smyth; adapted for edgerpy.
    """
    _IX = _FIT_IX
    _I = _FIT_I
    _rows_only = True
    _dim_from = "coefficients"


class DGELRT(RList):
    """Digital Gene Expression Likelihood Ratio Test Data and Results - Class

    A list-based class for storing the results of a GLM-based differential expression analysis, as
    produced by ``glmQLFTest`` and ``binQLFTest``.

    Attributes
    ----------
    table : DataFrame
        The test results, one row for each gene: the log-fold-change (``logFC``; ``logOR`` for
        ``binQLFTest``), or one column per coefficient for a test of several (``logFC.<name>``), the
        average abundance (``logCPM``), the F-statistic (``F``) and the p-value (``PValue``).
    comparison : str or list of str
        The coefficient(s) or contrast(s) tested.
    df.test : int
        Degrees of freedom of the test (the number of coefficients or contrasts tested).
    df.total : ndarray
        Total degrees of freedom for each gene (prior plus residual).

    The object also holds all the components of the fit that was tested (``coefficients``,
    ``s2.post``, ``samples``, ``genes``, ...).

    Notes
    -----
    Rows correspond to genomic features (genes, transcripts, exons, ...) and columns to statistics
    associated with the differential expression analysis.

    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    ``shape``, ``dimnames``, ``rownames`` and ``colnames`` are those of ``table``. ``res[i, :]``
    subsets genes (0-based positions, boolean masks or names); subsetting columns raises
    ``IndexError``.

    R documentation by the edgeR team (first created by Davis McCarthy); adapted for edgerpy.
    """
    _IX = _FIT_IX + ("table",)
    _I = _FIT_I + ("df.test", "df.total")
    _rows_only = True
    _dim_from = "table"


class TopTags(RList):
    """Table of the Top Differentially Expressed Genes - Class

    A list-based class holding the top genes of a test, as produced by ``topTags``.

    Attributes
    ----------
    table : DataFrame
        Differential expression results for the top genes in sorted order: the annotation columns of
        the result's ``genes``, the statistic columns of its ``table``, and the adjusted p-values
        (``FDR``, or ``FWER`` for the Bonferroni and Holm adjustments).
    adjust.method : str
        The method used to adjust p-values for multiple testing.
    comparison : str or list of str
        The coefficient(s) or contrast(s) tested.
    test : str
        The name of the test.

    Notes
    -----
    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    ``shape``, ``dimnames``, ``rownames`` and ``colnames`` are those of ``table``. ``tt[i, :]``
    subsets rows of the table (0-based positions, boolean masks or names). Printing shows the
    comparison and the table.

    R documentation by Mark Robinson, Davis McCarthy, Yunshun Chen and Gordon Smyth; adapted for
    edgerpy.
    """
    _dim_from = "table"
    _rows_only = True

    def subset(self, i=None, j=None):
        out = self.copy()
        if i is not None or j is not None:
            t = out["table"]
            out["table"] = t.iloc[_pos(i, t.index) if i is not None else slice(None),
                                  _pos(j, t.columns) if j is not None else slice(None)]
        return out

    def __repr__(self):
        if self.get("test") == "exact":
            head = "Comparison of groups:  " + "-".join(reversed(list(self["comparison"])))
        else:
            head = f"Coefficient:  {self['comparison']}"
        return head + "\n" + repr(self["table"])


class DGEDTU(RList):
    """Differential Usage Results - Class

    A list-based class for storing the results of ``diffSplice``: tests for differential exon or
    transcript usage, at exon (or transcript) level and at gene level. R stores these results in
    limma's ``MArrayLM`` class.

    Attributes
    ----------
    coefficients : DataFrame
        For each exon, the difference between the log-fold-change for that exon and the average
        log-fold-change for all other exons of the same gene.
    t : DataFrame
        Moderated t-statistics, same dimensions as ``coefficients``.
    p.value : DataFrame
        p-values corresponding to the t-statistics.
    genes : DataFrame
        Exon annotation, sorted by gene ID and by exon ID within gene.
    genecolname : str
        Name of the column of ``genes`` containing gene IDs.
    design : ndarray
        The design matrix of the fit.
    comparison : str
        The coefficient or contrast tested.
    gene.F, gene.F.p.value : DataFrame
        Moderated F-statistics and their p-values, one row for each gene.
    gene.simes.p.value, gene.bonferroni.p.value : DataFrame
        Simes and Bonferroni adjusted p-values, one row for each gene.
    gene.genes : DataFrame
        Gene annotation, with the number of exons (``NExons``).
    gene.firstexon, gene.lastexon : Series
        Positions of each gene's first and last exon in the exon-level results, numbered from 1 as
        in R.
    gene.df.prior, gene.df.residual, gene.df.total, gene.s2, gene.s2.prior, gene.s2.post
        The gene-level empirical Bayes quantities behind the tests.

    Notes
    -----
    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    ``shape``, ``dimnames``, ``rownames`` and ``colnames`` are those of ``coefficients``.

    R documentation by Gordon Smyth and Charity Law; adapted for edgerpy.
    """
    _dim_from = "coefficients"


class MDS(RList):
    """Multidimensional Scaling Results - Class

    A list-based class holding the multidimensional scaling computed by ``plotMDS``.

    Attributes
    ----------
    eigen.values : ndarray
        Eigenvalues.
    eigen.vectors : ndarray
        Eigenvectors.
    var.explained : ndarray
        Proportion of variance explained by each dimension.
    distance.matrix.squared : DataFrame
        Squared pairwise distances between the samples.
    dim.plot : ndarray
        The dimensions plotted.
    x, y : ndarray
        Coordinates of the plotted points.
    top : float
        Number of top genes used for each pairwise distance.
    gene.selection : str
        The gene selection method.
    axislabel : str
        Label used for the axes.

    Notes
    -----
    Components are accessed by their R names, ``obj["s2.post"]`` (``KeyError`` if absent), or like
    R's ``$`` with ``.`` written as ``_``, ``obj.s2_post`` (``None`` if absent). Setting a component
    to ``None`` (``obj.s2_post = None``) removes it. The object is a dict, so the components can also
    be listed and iterated over.

    R documentation by Di Wu and Gordon Smyth; adapted for edgerpy.
    """

    def __repr__(self):
        return "An object of class MDS\n" + "\n".join(f"${k}\n{_print_head(v)}\n" for k, v in self.items())


class TestResults(pd.DataFrame):
    """Matrix of Test Results - Class

    A matrix-based class for storing the results of simultaneous tests, as produced by
    ``decideTests``.

    Attributes
    ----------
    levels : tuple
        The possible values, e.g. ``(-1, 0, 1)``.
    labels : tuple of str
        Their labels, e.g. ``("Down", "NotSig", "Up")``.

    Notes
    -----
    A ``TestResults`` object is a pandas DataFrame of integer calls, one row for each gene and one
    column for each test. With levels ``(-1, 0, 1)``, zero represents acceptance of the null
    hypothesis, ``1`` indicates rejection in favor of the right tail alternative and ``-1`` indicates
    rejection in favor of the left tail alternative. Missing calls are pandas ``<NA>``.

    ``summary()`` counts the genes at each level for each column, as R's ``summary`` method.

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    _metadata = ["levels", "labels"]

    @property
    def _constructor(self):
        return TestResults

    def summary(self):
        """limma summary.TestResults: counts of each level per column."""
        levels = self.levels if self.levels is not None else (-1, 0, 1)
        labels = self.labels if self.labels is not None else [str(v) for v in levels]
        tab = {c: [int((self[c] == v).fillna(False).sum()) for v in levels] for c in self.columns}
        return pd.DataFrame(tab, index=list(labels), dtype=np.int64)


def _pos(idx, labels):
    """R subscript (0-based ints, bool mask or names) -> integer positions."""
    if isinstance(idx, slice):
        return np.arange(len(labels))[idx]
    a = np.asarray(idx)
    if a.dtype == bool:
        if a.size != len(labels):
            raise IndexError("logical subscript of wrong length")
        return np.nonzero(a)[0]
    if a.dtype.kind in "iu":
        return a.astype(np.int64).reshape(-1)
    lab = {str(k): n for n, k in reversed(list(enumerate(labels)))}
    try:
        return np.array([lab[str(v)] for v in np.atleast_1d(a)], dtype=np.int64)
    except KeyError as e:
        raise KeyError(f"Subscript not found: {e}") from None
