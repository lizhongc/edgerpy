# Notices

edgerpy is distributed under the GNU General Public License, version 2 or (at your option) any later
version (`LICENSE`), the licence of edgeR and limma.

edgerpy contains code from the projects below. Their files are vendored unchanged except, in C files, for
`#include` lines; `VENDORED.sha256` lists every vendored file with its origin and checksums.

| Component | Where in edgerpy | Licence | Copyright |
|---|---|---|---|
| edgeR 4.99.16 (Bioconductor): C kernels, `lowess.f`, `spline.f`, `locfit_lite.c` | `c/`, `fortran/` | GPL (>= 2) | the edgeR authors: Yunshun Chen, Lizhong Chen, Aaron TL Lun, Davis J McCarthy, Pedro Baldoni, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth |
| limma 3.99.1 (Bioconductor): `weighted_lowess.c` | `c/` | GPL (>= 2) | the limma authors (Gordon K Smyth et al.) |
| R 4.6.1: nmath (math library), excerpts of `src/main` (arithmetic, sorting, `pretty`), stats `lowess`, LINPACK QR routines, and header blocks | `c/`, `c/nmath/`, `fortran/linpack/`, `include/` | GPL (>= 2) | The R Core Team; The R Foundation; Ross Ihaka; Robert Gentleman; Morten Welinder |
| Netlib BLAS and LAPACK 3.12.1 (the routines used) | `fortran/blas/`, `fortran/lapack/` | modified BSD (`LICENSE-LAPACK`) | The University of Tennessee and The University of Tennessee Research Foundation; The University of California Berkeley; The University of Colorado Denver |

The Python code ports the R code of edgeR 4.99.16, limma 3.99.1 and statmod (`logmdigamma`), and R's
colour table and base-graphics layout rules for the plots; it is distributed under the same licence.
