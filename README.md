# edgerpy

The edgeR v4 quasi-likelihood pipelines for Python, running edgeR's own C code.

edgerpy brings [edgeR](https://bioconductor.org/packages/edgeR/) 4.99.16 to Python for differential
analysis of sequencing data. All numerical work runs in edgeR's compiled C/Fortran code, taken verbatim
from upstream, so the results are R's: on Linux x86_64 every result is bitwise identical to edgeR, and on
other platforms it agrees to the last few bits. The
[comparison page](https://lizhongc.github.io/edgerpy/comparison/GSE60450-R-vs-edgerpy.html) runs a full
RNA-seq analysis in R and in edgerpy and compares every object, value by value.

## Installation

```sh
pip install edgerpy
pip install 'edgerpy[kallisto,oarfish]'   # also read kallisto and oarfish output
```

edgerpy needs Python 3.13 or later and runs on Linux and macOS. To build from source, see the
[guide](https://github.com/lizhongc/edgerpy/blob/main/docs/guide.md#building-from-source).

## Quick start

```python
import edgerpy as ep

y = ep.DGEList(counts, group=group, genes=genes)
keep = ep.filterByExpr(y)
y = y.subset(keep, None, keep_lib_sizes=False)
y = ep.normLibSizes(y)
design = ep.modelMatrix("~ group", y.samples)
fit = ep.glmQLFit(y, design, robust=True)
res = ep.glmQLFTest(fit, coef=1)
ep.topTags(res)
```

Every function has a help page: `help(ep.glmQLFit)`, or `ep.glmQLFit?` in Jupyter.

## Features

- **NB quasi-likelihood** (`glmQLFit`, `glmQLFTest`): differential expression.
- **Binomial quasi-likelihood** (`PCList`, `binQLFit`, `binQLFTest`): differential proportions, e.g. methylation.
- **diffSplice** (`diffSplice`, `topSplice`): differential exon or transcript usage.
- **Quantifier readers** (`catchSalmon`, `catchKallisto`, `catchRSEM`, `catchOarfish` and gene-level versions).
- **Utilities**: `cpm`, `decideTests`, `estimateDisp`, `sampleWeights`.
- **Plots** that follow R's base graphics: `plotMDS`, `plotMD`, `plotQLDisp`, `plotBCV`, `plotSplice`.

## Coming from R

- Function names are the same as in R.
- Arguments use `_` instead of `.`: `keep.lib.sizes` → `keep_lib_sizes`. Defaults are R's.
- Integer `coef` is 0-based. Column names work as in R.
- The input is a DGEList or a PCList, never a count matrix.
- `or` is a Python keyword, so `binQLFTest` takes `or_`.

The [guide](https://github.com/lizhongc/edgerpy/blob/main/docs/guide.md) shows the three pipelines in R and
Python side by side and lists every difference.

## Documentation

- [Tutorial](https://github.com/lizhongc/edgerpy/blob/main/docs/tutorial/gse60450-edgerpy.ipynb): from
  Salmon output to differential expression and differential transcript usage.
- [Guide](https://github.com/lizhongc/edgerpy/blob/main/docs/guide.md): translating R calls, quantifier
  readers, plots, building from source, testing.
- [R vs edgerpy](https://lizhongc.github.io/edgerpy/comparison/GSE60450-R-vs-edgerpy.html): a full
  analysis in both, compared value by value.

## Citation

If you use edgerpy, please cite edgerpy
([CITATION.cff](https://github.com/lizhongc/edgerpy/blob/main/CITATION.cff)) and the edgeR v4 paper:

Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful differential analysis of sequencing
data with expanded functionality and improved support for small counts and larger datasets. *Nucleic Acids
Research* 53(2), gkaf018. https://doi.org/10.1093/nar/gkaf018

## Licence

GPL (>= 2), as edgeR, limma and R. The Netlib BLAS/LAPACK routines are under a modified BSD licence. See
[LICENSE](https://github.com/lizhongc/edgerpy/blob/main/LICENSE),
[LICENSE-LAPACK](https://github.com/lizhongc/edgerpy/blob/main/LICENSE-LAPACK) and
[NOTICE.md](https://github.com/lizhongc/edgerpy/blob/main/NOTICE.md).
