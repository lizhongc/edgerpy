#define EDGERPY_R_INTERNALS
#include "rapi.h"
#include "rmath.h"

/* ---- R-4.6.1/src/main/sort.c lines 45-54 (verbatim) ---- */
static int rcmp(double x, double y, bool nalast)
{
    int nax = ISNAN(x), nay = ISNAN(y);
    if (nax && nay)	return 0;
    if (nax)		return nalast ? 1 : -1;
    if (nay)		return nalast ? -1 : 1;
    if (x < y)		return -1;
    if (x > y)		return 1;
    return 0;
}

/* ---- R-4.6.1/src/main/sort.c lines 343-357 (verbatim) ---- */
void rsort_with_index(double *x, int *indx, int n)
{
    double v;
    int i, j, h, iv;

    for (h = 1; h <= n / 9; h = 3 * h + 1);
    for (; h > 0; h /= 3)
	for (i = h; i < n; i++) {
	    v = x[i]; iv = indx[i];
	    j = i;
	    while (j >= h && rcmp(x[j - h], v, true) > 0)
		 { x[j] = x[j - h]; indx[j] = indx[j-h]; j -= h; }
	    x[j] = v; indx[j] = iv;
	}
}

/* ---- R-4.6.1/src/main/sort.c lines 668-681 (verbatim) ---- */
#define psort_body						\
    bool nalast=true;					\
    R_xlen_t L, R, i, j;					\
								\
    for (L = lo, R = hi; L < R; ) {				\
	v = x[k];						\
	for(i = L, j = R; i <= j;) {				\
	    while (TYPE_CMP(x[i], v, nalast) < 0) i++;		\
	    while (TYPE_CMP(v, x[j], nalast) < 0) j--;		\
	    if (i <= j) { w = x[i]; x[i++] = x[j]; x[j--] = w; }\
	}							\
	if (j < k) L = i;					\
	if (k < i) R = j;					\
    }

/* ---- R-4.6.1/src/main/sort.c lines 692-698 (verbatim) ---- */
static void rPsort2(double *x, R_xlen_t lo, R_xlen_t hi, R_xlen_t k)
{
    double v, w;
#define TYPE_CMP rcmp
    psort_body
#undef TYPE_CMP
}

/* ---- R-4.6.1/src/main/sort.c lines 724-727 (verbatim) ---- */
void rPsort(double *x, int n, int k)
{
    rPsort2(x, 0, n-1, k);
}

