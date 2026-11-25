/* ---- R-4.6.1/src/main/array.c lines 643-662 (verbatim) ---- */
static bool mayHaveNaNOrInf(double *x, R_xlen_t n)
{
    if ((n&1) != 0 && !R_FINITE(x[0]))
	return true;
    for (R_xlen_t i = n&1; i < n; i += 2)
	/* A precise version could use this condition:
	 *
	 * !R_FINITE(x[i]+x[i+1]) && (!R_FINITE(x[i]) || !R_FINITE(x[i+1]))
	 *
	 * The present imprecise version has been found to be faster
	 * with GCC and ICC in the common case when the sum of the two
	 * values is always finite.
	 *
	 * The present version is imprecise because the sum of two very
	 * large finite values (e.g. 1e308) may be infinite.
	 */
	if (!R_FINITE(x[i]+x[i+1]))
	    return true;
    return false;
}

/* ---- R-4.6.1/src/main/array.c lines 714-728 (verbatim) ---- */
static void internal_matprod(double *x, int nrx, int ncx,
                             double *y, int nry, int ncy, double *z)
{
    LDOUBLE sum;
#define MATPROD_BODY					\
    R_xlen_t NRX = nrx, NRY = nry;			\
    for (int i = 0; i < nrx; i++)			\
	for (int k = 0; k < ncy; k++) {			\
	    sum = 0.0;					\
	    for (int j = 0; j < ncx; j++)		\
		sum += x[i + j * NRX] * y[j + k * NRY];	\
	    z[i + k * NRX] = (double) sum;		\
	}
    MATPROD_BODY;
}

/* ---- R-4.6.1/src/main/array.c lines 730-735 (verbatim) ---- */
static void simple_matprod(double *x, int nrx, int ncx,
                           double *y, int nry, int ncy, double *z)
{
    double sum;
    MATPROD_BODY;
}

