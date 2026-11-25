#define EDGERPY_R_INTERNALS
#include "rapi.h"
#include "rmath.h"

/* ---- R-4.6.1/src/main/qsort.c lines 130-137 (verbatim) ---- */
#define qsort_Index
#define INTt int
#define INDt int

#define NUMERIC double
void R_qsort_I(double *v, int *I, int i, int j)
#include "qsort-body.c"
#undef NUMERIC

