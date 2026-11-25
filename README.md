# edgerpy

The edgeR v4 quasi-likelihood pipelines for Python, running edgeR's own C code.

edgerpy ports three pipelines of edgeR 4.99.16:

- **NB QL:** differential expression with negative binomial quasi-likelihood.
- **Binomial QL:** differential proportions, e.g. methylation.
- **diffSplice:** differential exon or transcript usage.

All numerical work runs in compiled code taken verbatim from upstream: edgeR's C/Fortran kernels, limma's
weighted lowess, R 4.6.1's nmath, LINPACK and arithmetic helpers, and the Netlib BLAS/LAPACK 3.12.1 routines
they call. Only R's glue code is rewritten in Python.

**The results are R's.** On Linux x86_64 every intermediate and final object of the three pipelines is
bitwise identical to edgeR 4.99.16 / limma 3.99.1 / R 4.6.1 with the reference BLAS/LAPACK. On other
platforms the results agree with R to the last few bits, as R's own results do from one platform to
another. The page `docs/comparison/GSE60450-R-vs-edgerpy.html` runs a full RNA-seq analysis in R and in
edgerpy side by side and compares every object, value by value.

## Installation

edgerpy needs Python 3.13 or later and runs on Linux and macOS.

```sh
pip install edgerpy
pip install 'edgerpy[kallisto,oarfish]'   # also read kallisto and oarfish output
```

Building from source needs a C compiler, gfortran and OpenMP; no BLAS/LAPACK library is needed (the routines
are compiled in). With conda-forge's compilers:

```sh
conda create -n edgerpy -c conda-forge python=3.13 numpy pandas matplotlib meson-python meson ninja \
    c-compiler fortran-compiler llvm-openmp     # llvm-openmp: macOS only
conda activate edgerpy
pip install --no-build-isolation .            # from the source folder
```

## Getting started

A worked example, from Salmon output to differential expression and differential transcript usage, is the
notebook `docs/tutorial/gse60450-edgerpy.ipynb`. Every function has a help page: `help(ep.glmQLFit)`, or
`ep.glmQLFit?` in Jupyter.

## The three pipelines, side by side

**NB quasi-likelihood**

```r
y <- DGEList(counts, group=group, genes=genes)
keep <- filterByExpr(y)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- normLibSizes(y)
design <- model.matrix(~group)
fit <- glmQLFit(y, design, robust=TRUE)
res <- glmQLFTest(fit, coef=2)
topTags(res)
glmQLFTest(fit, contrast=c(0,-1,1))
glmQLFTest(fit, coef=2, fc=1.5)           # UPSHOT/TREAT
```

```python
import edgerpy as ep
y = ep.DGEList(counts, group=group, genes=genes)
keep = ep.filterByExpr(y)
y = y.subset(keep, None, keep_lib_sizes=False)   # or y[keep, :] to keep lib sizes
y = ep.normLibSizes(y)
design = ep.modelMatrix("~ group", y.samples)   # or any DataFrame / ndarray
fit = ep.glmQLFit(y, design, robust=True)
res = ep.glmQLFTest(fit, coef=1)                 # 0-based; or coef="B"
ep.topTags(res)
ep.glmQLFTest(fit, contrast=[0, -1, 1])
ep.glmQLFTest(fit, coef=1, fc=1.5)
```

**Binomial quasi-likelihood**

```r
y <- PCList(meth, unmeth, group=group)
fit <- binQLFit(y, design)
res <- binQLFTest(fit, coef=2, or=1.5)
topTags(res)
```

```python
y = ep.PCList(meth, unmeth, group=group)
fit = ep.binQLFit(y, design)
res = ep.binQLFTest(fit, coef=1, or_=1.5)        # `or` is a Python keyword
ep.topTags(res)
```

**Differential usage (diffSplice)**

```r
fit <- glmQLFit(y, design)
ds <- diffSplice(fit, geneid="GeneID", exonid="ExonID")
topSplice(ds, test="simes")
```

```python
fit = ep.glmQLFit(y, design)
ds = ep.diffSplice(fit, geneid="GeneID", exonid="ExonID")
ep.topSplice(ds, test="simes")
```

## Translating R calls

- **Function names** are the same as in R. The result of `diffSplice` (R's `MArrayLM`) is a `DGEDTU`.
- **Input is a DGEList (NB) or a PCList (binomial), never a count matrix.** The public functions are R's DGEList and PCList methods (`glmFit.DGEList`, `glmQLFit.DGEList`, `binQLFit.PCList`, ...). Offsets, weights, sample weights and `AveLogCPM` are read from the object, so arguments that only R's matrix methods have (`offset`, `lib.size`, `weights`, `AveLogCPM`, the `y, z` count pair) are not accepted. A matrix raises `TypeError`.
- **Arguments:** R's `.` becomes `_` (`abundance.trend` → `abundance_trend`, `keep.lib.sizes` → `keep_lib_sizes`). Defaults are R's.
- **Integer `coef` is 0-based.** Column names work as in R. The default is the last column, as in R.
- **Results are R lists.** `fit["s2.post"]` uses the exact R component name, and `fit.s2_post` works like R's `$` (it returns `None` when the component is absent). Components appear in R's order.
- **Data types:**
  - Matrices are `pandas.DataFrame`s over one column-major block, read by the C code in place with no copy.
  - Gene-level vectors are NumPy arrays.
  - `samples`, `genes` and result tables are DataFrames with R's column names.
  - Factors are `pandas.Categorical`.
- **Subsetting:** `y[i, j]` and `y.subset(i, j, keep_lib_sizes=False)` follow R's `[`. They accept 0-based positions, boolean masks or names.
- **Design matrices:** `ep.modelMatrix("~ group + batch", data)` builds the same matrix as R's `model.matrix` with treatment contrasts, including its `assign` and `contrasts` attributes (in `X.attrs`). Column names are shorter than R's: `Intercept`, and factor levels without the variable name (`trt`, `trt:b2` for R's `grouptrt`, `grouptrt:batchb2`). Logicals keep R's form (`flagTRUE`), and so does a factor whose level names clash with another variable's (`groupb1`). Formulas use `+ - * : ( )` and `0`/`1`. Function calls such as `log(x)`, and the `/`, `^` and `.` operators, are not supported.
- **Factor levels** from plain strings are sorted in C-locale order, as R does with `LC_COLLATE=C`. Pass a `pandas.Categorical` to fix the order yourself.

**Sample weights:** each sample can carry a technical weight t (on the Poisson part of the variance) and a biological weight b (dividing its NB dispersion), as in edgeR 4.99.12 and later. They are the columns `technical.weights` and `biological.weights` of `y.samples`, used by `glmFit`, `glmQLFit`, `glmQLFTest` and `diffSplice`. `ep.sampleWeights(y, design)` estimates both by REML and stores them in `y.samples`. `ep.glmQLFit(y, design, sample_weights=True)` does that first.

**Normalization with offset.prior:** a DGEList's `offset.prior` (TMM only) works as in R. A DGEList with an `offset` matrix is returned unchanged with a message, unless `renormalize_offset=True` converts the offset to `offset.prior` first. One case is not ported: `offset_prior` together with genes that are zero in every sample. There, edgeR 4.99.12 drops the zero rows from the counts but not from its library-size matrix, so R recycles vectors of unequal length. edgerpy raises an error; filter those genes first.

**Scope:** edgeR v4 (`legacy=FALSE`) only. The empirical Bayes squeezing always uses limma's `fitFDistUnequalDF1` method, as glmQLFit and binQLFit do in R. diffSplice's `squeezeVar(legacy=NULL)` differs only when every gene's pooled residual df is identical. `sinpi` always comes from R's own fallback code, on every platform, as R uses wherever the C library has none.

## Reading quantifier output (catch*)

`catchSalmon`, `catchKallisto`, `catchRSEM` and `catchOarfish` read transcript counts, and `catchSalmonGene`, `catchKallistoGene` and `catchOarfishGene` read gene counts (summed by the Gencode gene in the transcript names, or by a `tx2gene` table). Each returns a DGEList ready for `filterByExpr` → `normLibSizes` → `glmQLFit`. The counts are divided by an overdispersion estimated from the bootstrap or Gibbs samples, and `offset.prior` holds the log effective lengths. `splitGencodeTxNames` unpacks Gencode transcript IDs.

```python
y = ep.catchSalmonGene("salmon_out")                  # one subdirectory per sample
y = ep.catchSalmon(sample_dirs=["s1", "s2", "s3"])
y = ep.catchKallistoGene("kallisto_out", tx2gene=t2g)  # t2g: DataFrame, transcript and gene ID columns
```

- **Optional dependencies:** kallisto output needs `h5py` (`pip install 'edgerpy[kallisto]'`), and oarfish output needs `pyarrow` (`pip install 'edgerpy[oarfish]'`). Salmon and RSEM need nothing extra.
- **Numbers as R reads them.** readr's number parser is not correctly rounded: a 17-digit value can differ from Python's `float()` by many ULPs, and exponents beyond ±307 are clamped. edgerpy parses text output with the same algorithm, so every value is R's.
- **Only the DGEList form is ported** (R's `DGEList=FALSE` list is not).
- **Where edgeR 4.99.16 fails, edgerpy raises an error too:**
  - `catchOarfishGene(gene_length="simple")`;
  - a Salmon `meta_info.json` without `samp_type`.
- **Copied from edgeR 4.99.16 as is:**
  - `catchKallistoGene` with `tx2gene` keeps the version suffix of the gene IDs;
  - `catchOarfishGene(gene_length="tximport")` leaves the lengths equal to the counts when no gene is zero in all samples.
- **Subsetting:** as in R, `y[i, j]` also subsets the matrices in `y.other`.

## CPM, test calls and plots

These take edgeR objects only, as in the R methods they port:

| Function | Input | Returns |
|---|---|---|
| `cpm(y, log=False, prior_count=2, ...)` | DGEList (uses `offset` / `offset.prior` when present), DGEGLM, DGELRT (fitted CPM, `shrunk`) | DataFrame |
| `decideTests(res, adjust_method="BH", p_value=0.05, lfc=0)` | DGELRT from `glmQLFTest` or `binQLFTest` | `TestResults`; `.summary()` gives R's count table |
| `plotMDS(y, ...)` | DGEList (leading logFC), PCList (leading logOR); `method="bcv"` is not ported | `MDS` object (`plot=False` draws nothing) |
| `plotMD(object, ...)` | DGEList, DGEGLM, DGEBIN, DGELRT (coloured by `decideTests`) | `None` |
| `plotQLDisp(fit, ...)` | DGEGLM, DGEBIN | `{"x": ..., "y": ...}`, as R |
| `plotSplice(ds, ...)` | `diffSplice` output; `rank` is 1-based as in R | `None` |
| `estimateDisp(y, design, prior_df=None, tagwise=True, robust=False, span=None, min_row_sum=5, grid_length=21, grid_range=(-10, 10), nthreads=1)` | DGEList and a design (required) | the DGEList with R's components (common, trended, tagwise dispersions, AveLogCPM, prior.df, prior.n, span) |
| `plotBCV(y, ...)` | DGEList with dispersions | `None` |

**estimateDisp is simplified:**
- GLM mode only, so a design is required; the trend is always locfit;
- no `trend.method`, `legacy.span` or `winsor.tail.p`.

With `prior_df=None`, the prior df of the tagwise step is estimated as in R: a GLM fit with the trended dispersions, residual df reduced for exact zero fits (`.residDF`), then limma's `squeezeVar`. The one difference is that the squeeze always uses `fitFDistUnequalDF1`. limma makes the same choice whenever the residual df are unequal, which happens as soon as a kept gene has zero fitted values in a whole group. In that case every component equals R's `estimateDisp(y, design)`. When all residual df are equal (no zero fits at all), R uses limma's legacy `fitFDist` (`fitFDistRobustly` with `robust`). Then only the common and trended dispersions, AveLogCPM and span equal R's, and prior.df, prior.n and the tagwise dispersions differ. With a given `prior_df`, every component always equals R's `estimateDisp(y, design, prior.df=...)`.

`nthreads` is an edgerpy extension: R's `estimateDisp` has none, though the functions it calls do. It is passed to edgeR's gene-wise C kernels (the likelihood grid, the GLM fits, aveLogCPM) and to edgeR's locfit, which fits the grid columns in parallel. Results do not depend on it.

On 20,000 genes × 12 samples:

| Design | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| groups only | 0.87 s | 0.44 s | 0.42 s |
| groups and a covariate | 1.58 s | 0.60 s | 0.47 s |

With groups only, edgeR fits a one-way layout whose deviances and back-transform are serial, so the gain stops at about 2×. More than 8 threads is slower on a 16-core machine.

```python
import matplotlib.pyplot as plt
ep.plotMDS(y)                                  # a new 7 x 7 in figure, like an R device
plt.savefig("mds.png")

fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7))   # like par(mfrow=c(1,2))
ep.plotQLDisp(fit, ax=left)
ep.plotMD(res, ax=right)
print(ep.decideTests(res).summary())
```

The plots are drawn with matplotlib but follow R's base graphics: every plotted number, the axis limits (`par("usr")`), tick positions and labels, titles, legends, colours (R's own colour table and palette), symbols and their sizes are those of R. Fonts and anti-aliasing are matplotlib's.

## Testing

```sh
pip install 'edgerpy[test]'
python -m pytest tests                          # from the source folder
EDGERPY_TEST_BITWISE=1 python -m pytest tests   # require identical bits to R (Linux x86_64)
```

The tests compare edgerpy's results with R's on small data sets, to a relative tolerance of 1e-10, or
bitwise with `EDGERPY_TEST_BITWISE=1`.

## Source

```
c/          C: edgeR kernels, limma's weighted_lowess.c, R's src/main excerpts,
            and edgerpy's own py_exports.c, rcompat.c, rshim.c
  nmath/    R's math library (distribution and special functions)
fortran/    edgeR's lowess.f and spline.f; Netlib BLAS/LAPACK 3.12.1; R's LINPACK QR
include/    headers assembled from verbatim blocks of R, edgeR and limma headers
src/edgerpy/  the Python code
VENDORED.sha256   every vendored file with its origin and checksums (checked by tests/test_manifest.py)
```

In the vendored C files only `#include` lines differ from upstream.

## Citation

If you use edgerpy, please cite edgerpy (`CITATION.cff`) and the edgeR v4 paper:

Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful differential analysis of sequencing
data with expanded functionality and improved support for small counts and larger datasets. *Nucleic Acids
Research* 53(2), gkaf018. https://doi.org/10.1093/nar/gkaf018

## Licence

GPL (>= 2), as edgeR, limma and R (`LICENSE`). The Netlib BLAS/LAPACK routines are under a modified BSD
licence (`LICENSE-LAPACK`). `NOTICE.md` lists the vendored components and their copyright holders.
