/* ---- R-4.6.1/src/main/arithmetic.h lines 41-43 (verbatim) ---- */
static R_INLINE double R_log(double x) {
    return x > 0 ? log(x) : x == 0 ? R_NegInf : R_NaN;
}

/* ---- R-4.6.1/src/main/arithmetic.h lines 50-59 (verbatim) ---- */
static R_INLINE double logbase(double x, double base)
{
#ifdef HAVE_LOG10
    if(base == 10) return x > 0 ? log10(x) : x == 0 ? R_NegInf : R_NaN;
#endif
#ifdef HAVE_LOG2
    if(base == 2) return x > 0 ? log2(x) : x == 0 ? R_NegInf : R_NaN;
#endif
    return R_log(x) / R_log(base);
}

/* ---- R-4.6.1/src/main/arithmetic.c lines 1291-1327 (verbatim) ---- */
static double Rsqrt(double x)
{
    if (x == 0.) return x; // => sqrt(-0.) = -0.
    for(int i=1; i < 12; i++) {
	if (x == i*i) return i;
    }
    return sqrt(x);
}

static double Rexp(double x)
{
    /* exp(x) = 1 + x + x^2/2! + ...
     * should return 1+x for very small x i.e.,  x^2/2 < D_EPS / 2
     * <==> x^2 < D_EPS  <==> |x| < sqrt(D_EPS) : */
    return (fabs(x) <= sqrt(DBL_EPSILON)) ? 1. + x : exp(x);
}

/* should return x for very small x (e.g. expm1(x))*/
static double f_x_x(double x, double(*f)(double), double m)
{
    return (fabs(x) <= m) ? x : f(x);
}

static double Rexpm1(double x)
{
    /* expm1(x) = exp(x) - 1 =  x + x^2/2 + O(x^3)
     *                       =. x  when  x^2/2 < |x| * D_EPS/2
     *                             <==>  |x|   <  D_EPS */
    return f_x_x(x, expm1, DBL_EPSILON);
}

static double Rlog1p(double x)
{
    /* log1p(x) = log(1 + x) =  x - x^2/2  + O(x^3)
     *                       =. x  when  |x| < D_EPS, see Rexpm1() */
    return f_x_x(x, log1p, DBL_EPSILON);
}

