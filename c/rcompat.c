/* rcompat.c -- R semantics for the Python glue code.
 *
 * The edgeR/limma R code that edgerpy ports to Python does its arithmetic
 * through R primitives whose results are not those of NumPy:
 *   - log, exp, log1p, sqrt, log2, ^ use R wrappers around libm (R_log,
 *     Rexp, Rlog1p, Rsqrt, logbase, R_POW); NumPy may use its own SIMD code;
 *   - sum, mean, cumsum, colSums, rowSums, rowMeans accumulate in long double;
 *   - lgamma, trigamma, pf come from R's nmath;
 *   - %*% calls BLAS dgemv/dgemm (or a plain loop when NaN/Inf may be present);
 *   - optimize, lowess, qr, qr.qty, qr.qy, backsolve and lm.wfit call
 *     Brent_fmin, clowess, LINPACK and BLAS routines.
 * Each function below reproduces one of those primitives. Static helpers
 * come verbatim from R (the three .inc excerpts); the loops around them are
 * transcribed from the R 4.6.1 sources named in each comment. All are
 * checked bitwise against R in tests/test_rcompat.py.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL edgerpy_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define R_NO_REMAP
#define EDGERPY_R_INTERNALS   /* LDOUBLE, R_POW, ... from R's Defn.h/arithmetic.h (rapi.h) */
#include "edgerpy.h"

/* verbatim R excerpts: R_log, logbase, Rsqrt, Rexp, f_x_x, Rexpm1, Rlog1p;
 * mayHaveNaNOrInf, internal_matprod (defines MATPROD_BODY), simple_matprod */
#include "arith_math1.inc"
#include "array_matprod.inc"
/* stats' static fsquare/fcube would clash with edgeR's global ones (helper.h) */
#define fsquare stats_fsquare
#define fcube stats_fcube
#include "stats_lowess.inc"
#undef fsquare
#undef fcube

/* ------------------------------------------------------------------ */
/* helpers                                                              */

/* new float64 array (Fortran order) with the shape of a */
static PyArrayObject *new_like(PyArrayObject *a)
{
    return (PyArrayObject *)PyArray_NewLikeArray(a, NPY_FORTRANORDER, PyArray_DescrFromType(NPY_DOUBLE), 0);
}

/* any array-like -> float64 Fortran-contiguous array (new reference) */
static PyArrayObject *as_double(PyObject *o)
{
    return (PyArrayObject *)PyArray_FROMANY(o, NPY_DOUBLE, 0, 0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
}

/* ------------------------------------------------------------------ */
/* elementwise functions                                               */

/* math1() of src/main/arithmetic.c: y = f(x), incoming NaN preserved */
static PyObject *apply_math1(PyObject *args, double (*f)(double))
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(o);
    if (a == NULL)
    {
        return NULL;
    }
    PyArrayObject *r = new_like(a);
    if (r == NULL)
    {
        Py_DECREF(a);
        return NULL;
    }
    const double *x = PyArray_DATA(a);
    double *y = PyArray_DATA(r);
    npy_intp n = PyArray_SIZE(a);
    for (npy_intp i = 0; i < n; i++)
    {
        double xi = x[i];
        y[i] = f(xi);
        if (ISNAN(y[i]) && ISNAN(xi))
        {
            y[i] = xi;
        }
    }
    Py_DECREF(a);
    return (PyObject *)r;
}

/* math1_ari() of src/main/arithmetic.c: x == arg gives res exactly */
static PyObject *apply_math1_ari(PyObject *args, double (*f)(double), double arg, double res)
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(o);
    if (a == NULL)
    {
        return NULL;
    }
    PyArrayObject *r = new_like(a);
    if (r == NULL)
    {
        Py_DECREF(a);
        return NULL;
    }
    const double *x = PyArray_DATA(a);
    double *y = PyArray_DATA(r);
    npy_intp n = PyArray_SIZE(a);
    for (npy_intp i = 0; i < n; i++)
    {
        double xi = x[i];
        if (xi == arg)
        {
            y[i] = res;
        }
        else
        {
            y[i] = f(xi);
        }
        if (ISNAN(y[i]) && ISNAN(xi))
        {
            y[i] = xi;
        }
    }
    Py_DECREF(a);
    return (PyObject *)r;
}

static double log2_base(double x)
{
    return logbase(x, 2.0);
}

static double log10_base(double x)
{
    return logbase(x, 10.0);
}

/* R: log(x) */
static PyObject *r_log(PyObject *self, PyObject *args) { return apply_math1(args, R_log); }
/* R: sqrt(x) */
static PyObject *r_sqrt(PyObject *self, PyObject *args) { return apply_math1(args, Rsqrt); }
/* R: exp(x) */
static PyObject *r_exp(PyObject *self, PyObject *args) { return apply_math1_ari(args, Rexp, 0., 1.); }
/* R: log1p(x) */
static PyObject *r_log1p(PyObject *self, PyObject *args) { return apply_math1_ari(args, Rlog1p, 0., 0.); }
/* R: lgamma(x) */
static PyObject *r_lgamma(PyObject *self, PyObject *args) { return apply_math1(args, Rf_lgammafn); }
/* R: trigamma(x) */
static PyObject *r_trigamma(PyObject *self, PyObject *args) { return apply_math1(args, Rf_trigamma); }

/* R: log2(x) / log10(x) = math2(x, base, logbase) with if_NA_Math2_set (base is never NA) */
static PyObject *log_base_apply(PyObject *args, double (*f)(double))
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(o);
    if (a == NULL)
    {
        return NULL;
    }
    PyArrayObject *r = new_like(a);
    if (r == NULL)
    {
        Py_DECREF(a);
        return NULL;
    }
    const double *x = PyArray_DATA(a);
    double *y = PyArray_DATA(r);
    npy_intp n = PyArray_SIZE(a);
    for (npy_intp i = 0; i < n; i++)
    {
        if (ISNA(x[i]))
        {
            y[i] = NA_REAL;
        }
        else if (ISNAN(x[i]))
        {
            y[i] = R_NaN;
        }
        else
        {
            y[i] = f(x[i]);
        }
    }
    Py_DECREF(a);
    return (PyObject *)r;
}

static PyObject *r_log2(PyObject *self, PyObject *args) { return log_base_apply(args, log2_base); }
static PyObject *r_log10(PyObject *self, PyObject *args) { return log_base_apply(args, log10_base); }

/* R: x ^ y (real_binary POWOP: R_POW elementwise); x and y already broadcast */
static PyObject *r_pow(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;
    if (!PyArg_ParseTuple(args, "OO", &ox, &oy))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(ox), *b = as_double(oy), *r = NULL;
    if (a && b && PyArray_SIZE(a) == PyArray_SIZE(b))
    {
        r = new_like(a);
        if (r)
        {
            const double *x = PyArray_DATA(a), *e = PyArray_DATA(b);
            double *z = PyArray_DATA(r);
            npy_intp n = PyArray_SIZE(a);
            for (npy_intp i = 0; i < n; i++)
            {
                z[i] = e[i] == 2.0 ? x[i] * x[i] : R_pow(x[i], e[i]);
            }
        }
    }
    else if (a && b)
    {
        PyErr_SetString(PyExc_ValueError, "r_pow: x and y must have the same size");
    }
    Py_XDECREF(a);
    Py_XDECREF(b);
    return (PyObject *)r;
}

/* R: pf(q, df1, df2, lower.tail, log.p) = math3_2 of stats distn.c; arrays broadcast */
static PyObject *r_pf(PyObject *self, PyObject *args)
{
    PyObject *oq, *o1, *o2;
    int lower, logp;
    if (!PyArg_ParseTuple(args, "OOOpp", &oq, &o1, &o2, &lower, &logp))
    {
        return NULL;
    }
    PyArrayObject *q = as_double(oq), *d1 = as_double(o1), *d2 = as_double(o2), *r = NULL;
    if (q && d1 && d2 && PyArray_SIZE(q) == PyArray_SIZE(d1) && PyArray_SIZE(q) == PyArray_SIZE(d2))
    {
        r = new_like(q);
        if (r)
        {
            const double *a = PyArray_DATA(q), *b = PyArray_DATA(d1), *c = PyArray_DATA(d2);
            double *y = PyArray_DATA(r);
            npy_intp n = PyArray_SIZE(q);
            for (npy_intp i = 0; i < n; i++)
            {
                if (ISNA(a[i]) || ISNA(b[i]) || ISNA(c[i]))
                {
                    y[i] = NA_REAL;
                }
                else if (ISNAN(a[i]) || ISNAN(b[i]) || ISNAN(c[i]))
                {
                    y[i] = R_NaN;
                }
                else
                {
                    y[i] = Rf_pf(a[i], b[i], c[i], lower, logp);
                }
            }
        }
    }
    else if (q && d1 && d2)
    {
        PyErr_SetString(PyExc_ValueError, "r_pf: arguments must have the same size");
    }
    Py_XDECREF(q);
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return (PyObject *)r;
}

/* R: qf(p, df1, df2, lower.tail, log.p) = math3_2 of stats distn.c; arrays broadcast */
static PyObject *r_qf(PyObject *self, PyObject *args)
{
    PyObject *op, *o1, *o2;
    int lower, logp;
    if (!PyArg_ParseTuple(args, "OOOpp", &op, &o1, &o2, &lower, &logp))
    {
        return NULL;
    }
    PyArrayObject *p = as_double(op), *d1 = as_double(o1), *d2 = as_double(o2), *r = NULL;
    if (p && d1 && d2 && PyArray_SIZE(p) == PyArray_SIZE(d1) && PyArray_SIZE(p) == PyArray_SIZE(d2))
    {
        r = new_like(p);
        if (r)
        {
            const double *a = PyArray_DATA(p), *b = PyArray_DATA(d1), *c = PyArray_DATA(d2);
            double *y = PyArray_DATA(r);
            npy_intp n = PyArray_SIZE(p);
            for (npy_intp i = 0; i < n; i++)
            {
                if (ISNA(a[i]) || ISNA(b[i]) || ISNA(c[i]))
                {
                    y[i] = NA_REAL;
                }
                else if (ISNAN(a[i]) || ISNAN(b[i]) || ISNAN(c[i]))
                {
                    y[i] = R_NaN;
                }
                else
                {
                    y[i] = Rf_qf(a[i], b[i], c[i], lower, logp);
                }
            }
        }
    }
    else if (p && d1 && d2)
    {
        PyErr_SetString(PyExc_ValueError, "r_qf: arguments must have the same size");
    }
    Py_XDECREF(p);
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return (PyObject *)r;
}

/* readr::read_tsv's double parser (vroom 1.7.1, the BSD/Tcl strtod algorithm). It is not correctly rounded:
 * at most 18 mantissa digits are used (the rest are dropped), the mantissa is 1e9 * (first digits) + (last 9
 * digits), the decimal exponent is clamped to 307, and the power of ten is built by repeated squaring and then
 * multiplied or divided in. This reproduces readr bitwise (checked against R's catch* readers).
 * Leading and trailing blanks are trimmed; "" and "NA" give NA; inf, -inf and nan (any case) give Inf, -Inf, NaN;
 * a field that is not wholly a number gives NA, as readr does (with a warning there). */
static double readr_dbl(const char *s, Py_ssize_t n)
{
    static const double p10[] = {1e1, 1e2, 1e4, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256};
    const int maxexp = 307;
    while (n > 0 && (*s == ' ' || *s == '\t'))
    {
        s++;
        n--;
    }
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r'))
    {
        n--;
    }
    if (n == 0 || (n == 2 && s[0] == 'N' && s[1] == 'A'))
    {
        return NA_REAL;
    }
    Py_ssize_t i = 0;
    int neg = 0;
    if (s[i] == '+' || s[i] == '-')
    {
        neg = s[i] == '-';
        i++;
    }
    if (n - i == 3)
    {
        char a = (char)(s[i] | 0x20), b = (char)(s[i + 1] | 0x20), c = (char)(s[i + 2] | 0x20);
        if (a == 'i' && b == 'n' && c == 'f')
        {
            return neg ? R_NegInf : R_PosInf;
        }
        if (a == 'n' && b == 'a' && c == 'n')
        {
            return R_NaN;
        }
    }
    int digits[18], nd = 0, mant = 0, decpt = -1;
    for (; i < n && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.' && decpt < 0)); i++, mant++)
    {
        if (s[i] == '.')
        {
            decpt = mant;
        }
        else if (nd < 18)
        {
            digits[nd++] = s[i] - '0';
        }
    }
    if (decpt < 0)
    {
        decpt = mant;
    }
    else
    {
        mant--;
    }
    int fracexp;
    if (mant > 18)
    {
        fracexp = decpt - 18;
        mant = 18;
    }
    else
    {
        fracexp = decpt - mant;
    }
    if (mant == 0)
    {
        return (i == n) ? (neg ? -0.0 : 0.0) : NA_REAL;
    }
    int k = 0, frac1 = 0, frac2 = 0;
    for (; mant - k > 9; k++)
    {
        frac1 = 10 * frac1 + digits[k];
    }
    for (; k < mant; k++)
    {
        frac2 = 10 * frac2 + digits[k];
    }
    double fraction = (1.0e9 * frac1) + frac2;
    int exp = 0, esign = 0;
    if (i < n && (s[i] == 'e' || s[i] == 'E'))
    {
        i++;
        if (i < n && (s[i] == '+' || s[i] == '-'))
        {
            esign = s[i] == '-';
            i++;
        }
        if (i == n || s[i] < '0' || s[i] > '9')
        {
            return NA_REAL;
        }
        for (; i < n && s[i] >= '0' && s[i] <= '9'; i++)
        {
            if (exp < 100000)
            {
                exp = exp * 10 + (s[i] - '0');
            }
        }
    }
    if (i != n)
    {
        return NA_REAL;
    }
    exp = esign ? fracexp - exp : fracexp + exp;
    int eneg = exp < 0;
    if (eneg)
    {
        exp = -exp;
    }
    if (exp > maxexp)
    {
        exp = maxexp;
    }
    double dblexp = 1.0;
    for (int d = 0; exp != 0; exp >>= 1, d++)
    {
        if (exp & 1)
        {
            dblexp *= p10[d];
        }
    }
    fraction = eneg ? fraction / dblexp : fraction * dblexp;
    return neg ? -fraction : fraction;
}

/* readr's double parser over a sequence of str: r_parse_dbl(fields) -> float64 array */
static PyObject *r_parse_dbl(PyObject *self, PyObject *args)
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
    {
        return NULL;
    }
    PyObject *seq = PySequence_Fast(o, "r_parse_dbl: a sequence of str is required");
    if (!seq)
    {
        return NULL;
    }
    npy_intp n = PySequence_Fast_GET_SIZE(seq);
    PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    if (r)
    {
        double *y = PyArray_DATA(r);
        PyObject **items = PySequence_Fast_ITEMS(seq);
        for (npy_intp i = 0; i < n; i++)
        {
            Py_ssize_t len;
            const char *s = PyUnicode_AsUTF8AndSize(items[i], &len);
            if (!s)
            {
                Py_DECREF(r);
                r = NULL;
                break;
            }
            y[i] = readr_dbl(s, len);
        }
    }
    Py_DECREF(seq);
    return (PyObject *)r;
}

/* ------------------------------------------------------------------ */
/* reductions (long double accumulation as in R)                        */

/* R: sum(x, na.rm) for doubles = rsum() of src/main/summary.c */
static PyObject *r_sum(PyObject *self, PyObject *args)
{
    PyObject *o;
    int narm = 0;
    if (!PyArg_ParseTuple(args, "O|p", &o, &narm))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(o);
    if (a == NULL)
    {
        return NULL;
    }
    const double *x = PyArray_DATA(a);
    npy_intp n = PyArray_SIZE(a);
    LDOUBLE s = 0.0;
    for (npy_intp k = 0; k < n; k++)
    {
        if (!narm || !ISNAN(x[k]))
        {
            s += x[k];
        }
    }
    double value;
    if (s > DBL_MAX)
    {
        value = R_PosInf;
    }
    else if (s < -DBL_MAX)
    {
        value = R_NegInf;
    }
    else
    {
        value = (double) s;
    }
    Py_DECREF(a);
    return PyFloat_FromDouble(value);
}

/* R: mean(x) for doubles = real_mean() of src/main/summary.c */
static PyObject *r_mean(PyObject *self, PyObject *args)
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(o);
    if (a == NULL)
    {
        return NULL;
    }
    const double *dx = PyArray_DATA(a);
    npy_intp n = PyArray_SIZE(a);
    LDOUBLE s = 0.0;
    for (npy_intp k = 0; k < n; k++)
    {
        s += dx[k];
    }
    int finite_s = R_FINITE((double) s) != 0;
    if (finite_s)
    {
        s /= n;
    }
    else
    {
        s = 0.;
        for (npy_intp k = 0; k < n; k++)
        {
            s += dx[k] / n;
        }
    }
    if (finite_s && R_FINITE((double) s))
    {
        LDOUBLE t = 0.0;
        for (npy_intp k = 0; k < n; k++)
        {
            t += (dx[k] - s);
        }
        s += t / n;
    }
    else if (R_FINITE((double) s))
    {
        LDOUBLE t = 0.0;
        for (npy_intp k = 0; k < n; k++)
        {
            t += (dx[k] - s) / n;
        }
        s += t;
    }
    Py_DECREF(a);
    return PyFloat_FromDouble((double) s);
}

/* R: cumsum(x) for doubles = cumsum() of src/main/cum.c */
static PyObject *r_cumsum(PyObject *self, PyObject *args)
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
    {
        return NULL;
    }
    PyArrayObject *a = as_double(o);
    if (a == NULL)
    {
        return NULL;
    }
    PyArrayObject *r = new_like(a);
    if (r == NULL)
    {
        Py_DECREF(a);
        return NULL;
    }
    const double *rx = PyArray_DATA(a);
    double *rs = PyArray_DATA(r);
    LDOUBLE sum = 0.;
    for (npy_intp i = 0; i < PyArray_SIZE(a); i++)
    {
        sum += rx[i];
        rs[i] = (double) sum;
    }
    Py_DECREF(a);
    return (PyObject *)r;
}

/* R: colSums / colMeans (op 0/1) and rowSums / rowMeans (op 2/3) for a
 * double matrix = do_colsum() of src/main/array.c; na.rm = TRUE (optional
 * third argument) is supported for the row ops */
static PyObject *r_colsum(PyObject *self, PyObject *args)
{
    PyObject *o;
    int op, narm = 0;
    if (!PyArg_ParseTuple(args, "Oi|p", &o, &op, &narm))
    {
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(o, NPY_DOUBLE, 2, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
    if (a == NULL)
    {
        return NULL;
    }
    npy_intp n = PyArray_DIM(a, 0), p = PyArray_DIM(a, 1);
    const double *x = PyArray_DATA(a);
    npy_intp len = (op <= 1) ? p : n;
    PyObject *r = PyArray_ZEROS(1, &len, NPY_DOUBLE, 0);
    if (r == NULL)
    {
        Py_DECREF(a);
        return NULL;
    }
    double *ans = PyArray_DATA((PyArrayObject *)r);
    if (op <= 1)
    {
        for (npy_intp j = 0; j < p; j++)
        {
            const double *rx = x + n * j;
            LDOUBLE sum = 0.;
            for (npy_intp i = 0; i < n; i++)
            {
                sum += *rx++;
            }
            if (op == 1)
            {
                sum /= n;
            }
            ans[j] = (double) sum;
        }
    }
    else
    {
        LDOUBLE *rans = calloc(n > 0 ? n : 1, sizeof(LDOUBLE));
        int *Cnt = (narm && op == 3) ? calloc(n > 0 ? n : 1, sizeof(int)) : NULL;
        for (npy_intp j = 0; j < p; j++)
        {
            LDOUBLE *ra = rans;
            const double *rx = x + n * j;
            if (!narm)
            {
                for (npy_intp i = 0; i < n; i++)
                {
                    *ra++ += *rx++;
                }
            }
            else
            {
                for (npy_intp i = 0; i < n; i++, ra++, rx++)
                {
                    if (!ISNAN(*rx))
                    {
                        *ra += *rx;
                        if (op == 3)
                        {
                            Cnt[i]++;
                        }
                    }
                }
            }
        }
        if (op == 3)
        {
            if (!narm)
            {
                for (npy_intp i = 0; i < n; i++)
                {
                    rans[i] /= p;
                }
            }
            else
            {
                for (npy_intp i = 0; i < n; i++)
                {
                    rans[i] /= Cnt[i];
                }
            }
        }
        free(Cnt);
        for (npy_intp i = 0; i < n; i++)
        {
            ans[i] = (double) rans[i];
        }
        free(rans);
    }
    Py_DECREF(a);
    return r;
}

/* R: rowsum(x, group, reorder=FALSE) for a double matrix = rowsum() of
 * src/main/unique.c; g holds 0-based group indices (first-appearance order) */
static PyObject *r_rowsum(PyObject *self, PyObject *args)
{
    PyObject *ox, *og;
    int ng;
    if (!PyArg_ParseTuple(args, "OOi", &ox, &og, &ng))
    {
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 2, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
    PyArrayObject *g = (PyArrayObject *)PyArray_FROMANY(og, NPY_INT32, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
    PyObject *r = NULL;
    if (a && g)
    {
        npy_intp n = PyArray_DIM(a, 0), p = PyArray_DIM(a, 1);
        npy_intp dims[2] = {ng, p};
        r = PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
        if (r)
        {
            const double *x = PyArray_DATA(a);
            const int *gi = PyArray_DATA(g);
            double *pa = PyArray_DATA((PyArrayObject *)r);
            for (npy_intp i = 0; i < p; i++)
            {
                for (npy_intp j = 0; j < n; j++)
                {
                    pa[gi[j] + ng * i] += x[j + n * i];
                }
            }
        }
    }
    Py_XDECREF(a);
    Py_XDECREF(g);
    return r;
}

/* ------------------------------------------------------------------ */
/* optimize(): edgeR's brent_fmin (= R's Brent_fmin) with R's fcn1()    */

typedef struct
{
    PyObject *f;
    int failed;   /* the Python objective raised */
    int nwarn;    /* non-finite values replaced, as R warns */
} brent_info;

/* fcn1() of src/library/stats/src/optimize.c for a double result */
static double brent_call(double x, void *v)
{
    brent_info *info = v;
    if (info->failed)
    {
        return DBL_MAX;
    }
    PyObject *res = PyObject_CallFunction(info->f, "d", x);
    if (res == NULL)
    {
        info->failed = 1;
        return DBL_MAX;
    }
    double y = PyFloat_AsDouble(res);
    Py_DECREF(res);
    if (y == -1.0 && PyErr_Occurred())
    {
        info->failed = 1;
        return DBL_MAX;
    }
    if (!R_FINITE(y))
    {
        info->nwarn++;
        if (y == R_NegInf)
        {
            return -DBL_MAX;
        }
        return DBL_MAX;
    }
    return y;
}

/* R: optimize(f, c(lower, upper), tol)$minimum */
static PyObject *r_brent_fmin(PyObject *self, PyObject *args)
{
    double lower, upper, tol;
    brent_info info = {NULL, 0, 0};
    if (!PyArg_ParseTuple(args, "ddOd", &lower, &upper, &info.f, &tol))
    {
        return NULL;
    }
    double x = brent_fmin(lower, upper, brent_call, &info, tol);
    if (info.failed)
    {
        return NULL;
    }
    if (info.nwarn && PyErr_WarnEx(PyExc_RuntimeWarning, "NA/Inf replaced by maximum positive value", 1) < 0)
    {
        return NULL;
    }
    return PyFloat_FromDouble(x);
}

/* ------------------------------------------------------------------ */
/* lowess: stats::lowess (R's clowess, stats_lowess.inc); weightedLowess via limma */

/* R: lowess(x, y, f, iter, delta)$y with x already sorted (as stats::lowess
 * sorts before calling clowess) */
static PyObject *r_lowess(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;
    double f, delta;
    int iter;
    if (!PyArg_ParseTuple(args, "OOdid", &ox, &oy, &f, &iter, &delta))
    {
        return NULL;
    }
    PyArrayObject *x = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY);
    PyArrayObject *y = (PyArrayObject *)PyArray_FROMANY(oy, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY);
    PyObject *r = NULL;
    if (x && y && PyArray_SIZE(x) == PyArray_SIZE(y))
    {
        npy_intp n = PyArray_SIZE(x);
        r = PyArray_ZEROS(1, &n, NPY_DOUBLE, 0);
        double *rw = calloc(n > 0 ? n : 1, sizeof(double));
        double *res = calloc(n > 0 ? n : 1, sizeof(double));
        if (r && rw && res)
        {
            int nn = (int)n;
            clowess(PyArray_DATA(x), PyArray_DATA(y), nn, f, iter, delta, PyArray_DATA((PyArrayObject *)r), rw, res);
        }
        free(rw);
        free(res);
    }
    Py_XDECREF(x);
    Py_XDECREF(y);
    return r;
}

/* limma .Call("weighted_lowess", x, y, w, span, iter, delta): (fitted, weights) */
static PyObject *r_weighted_lowess(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy, *ow;
    double span, delta;
    int iter;
    if (!PyArg_ParseTuple(args, "OOOdid", &ox, &oy, &ow, &span, &iter, &delta))
    {
        return NULL;
    }
    PyArrayObject *x = as_double(ox), *y = as_double(oy), *w = as_double(ow);
    PyObject *fit = NULL, *rob = NULL, *ans = NULL;
    if (x && y && w)
    {
        npy_intp n = PyArray_SIZE(x);
        fit = PyArray_ZEROS(1, &n, NPY_DOUBLE, 0);
        rob = PyArray_ZEROS(1, &n, NPY_DOUBLE, 0);
        if (fit && rob)
        {
            lowess(PyArray_DATA(x), PyArray_DATA(y), PyArray_DATA(w), (int)n, span, iter, delta,
                   PyArray_DATA((PyArrayObject *)fit), PyArray_DATA((PyArrayObject *)rob));
            ans = Py_BuildValue("(OO)", fit, rob);
        }
    }
    Py_XDECREF(fit);
    Py_XDECREF(rob);
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(w);
    return ans;
}

/* ------------------------------------------------------------------ */
/* LINPACK QR and BLAS                                                  */

/* R: qr(x, tol)  (qr.default, LAPACK = FALSE): dict(qr, rank, qraux, pivot) */
static PyObject *r_qr(PyObject *self, PyObject *args)
{
    PyObject *ox;
    double tol;
    if (!PyArg_ParseTuple(args, "Od", &ox, &tol))
    {
        return NULL;
    }
    PyArrayObject *x = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    if (x == NULL)
    {
        return NULL;
    }
    int n = (int)PyArray_DIM(x, 0), p = (int)PyArray_DIM(x, 1), rank = 0;
    npy_intp np = p;
    PyObject *qraux = PyArray_ZEROS(1, &np, NPY_DOUBLE, 0);
    PyObject *pivot = PyArray_ZEROS(1, &np, NPY_INT32, 0);
    double *work = calloc(2 * (size_t)(p > 0 ? p : 1), sizeof(double));
    PyObject *ans = NULL;
    if (qraux && pivot && work)
    {
        int *piv = PyArray_DATA((PyArrayObject *)pivot);
        for (int i = 0; i < p; i++)
        {
            piv[i] = i + 1;
        }
        F77_CALL(dqrdc2)(PyArray_DATA(x), &n, &n, &p, &tol, &rank, PyArray_DATA((PyArrayObject *)qraux), piv, work);
        ans = Py_BuildValue("{s:O,s:i,s:O,s:O}", "qr", x, "rank", rank, "qraux", qraux, "pivot", pivot);
    }
    free(work);
    Py_DECREF(x);
    Py_XDECREF(qraux);
    Py_XDECREF(pivot);
    return ans;
}

/* R: qr.qty / qr.qy (transpose = 1 / 0) */
static PyObject *r_qr_qy(PyObject *self, PyObject *args)
{
    PyObject *oqr, *oqraux, *oy;
    int k, transpose;
    if (!PyArg_ParseTuple(args, "OOiOp", &oqr, &oqraux, &k, &oy, &transpose))
    {
        return NULL;
    }
    PyArrayObject *qr = (PyArrayObject *)PyArray_FROMANY(oqr, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY);
    PyArrayObject *qraux = (PyArrayObject *)PyArray_FROMANY(oqraux, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY);
    PyArrayObject *y = (PyArrayObject *)PyArray_FROMANY(oy, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    PyObject *ans = NULL;
    if (qr && qraux && y)
    {
        int n = (int)PyArray_DIM(qr, 0), ny = (int)PyArray_DIM(y, 1);
        PyArrayObject *out = (PyArrayObject *)PyArray_NewCopy(y, NPY_FORTRANORDER);
        if (out)
        {
            if (transpose)
            {
                F77_CALL(dqrqty)(PyArray_DATA(qr), &n, &k, PyArray_DATA(qraux), PyArray_DATA(y), &ny, PyArray_DATA(out));
            }
            else
            {
                F77_CALL(dqrqy)(PyArray_DATA(qr), &n, &k, PyArray_DATA(qraux), PyArray_DATA(y), &ny, PyArray_DATA(out));
            }
            ans = (PyObject *)out;
        }
    }
    Py_XDECREF(qr);
    Py_XDECREF(qraux);
    Py_XDECREF(y);
    return ans;
}

/* R: backsolve(r, b, k, upper.tri, transpose) = do_backsolve of src/main/array.c */
static PyObject *r_backsolve(PyObject *self, PyObject *args)
{
    PyObject *or_, *ob;
    int k, upper, trans;
    if (!PyArg_ParseTuple(args, "OOipp", &or_, &ob, &k, &upper, &trans))
    {
        return NULL;
    }
    PyArrayObject *r = (PyArrayObject *)PyArray_FROMANY(or_, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY_RO | NPY_ARRAY_FORCECAST);
    PyArrayObject *b = (PyArrayObject *)PyArray_FROMANY(ob, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY_RO | NPY_ARRAY_FORCECAST);
    PyObject *ans = NULL;
    if (r && b)
    {
        int nrr = (int)PyArray_DIM(r, 0), nrb = (int)PyArray_DIM(b, 0), ncb = (int)PyArray_DIM(b, 1);
        const double *rr = PyArray_DATA(r);
        int ok = 1;
        for (int i = 0; i < k; i++)
        {
            if (rr[i * (size_t)(nrr + 1)] == 0.0)
            {
                PyErr_Format(PyExc_ValueError, "singular matrix in 'backsolve'. First zero in diagonal [%d]", i + 1);
                ok = 0;
                break;
            }
        }
        npy_intp dims[2] = {k, ncb};
        PyObject *out = ok ? PyArray_ZEROS(2, dims, NPY_DOUBLE, 1) : NULL;
        if (out)
        {
            double *z = PyArray_DATA((PyArrayObject *)out);
            const double *bb = PyArray_DATA(b);
            for (int j = 0; j < ncb; j++)
            {
                memcpy(z + (size_t)j * k, bb + (size_t)j * nrb, (size_t)k * sizeof(double));
            }
            double one = 1.0;
            if (k > 0 && ncb > 0)
            {
                F77_CALL(dtrsm)("L", upper ? "U" : "L", trans ? "T" : "N", "N", &k, &ncb, &one,
                                (double *)rr, &nrr, z, &k FCONE FCONE FCONE FCONE);
            }
            ans = out;
        }
    }
    Py_XDECREF(r);
    Py_XDECREF(b);
    return ans;
}

/* R: x %*% y = matprod() of src/main/array.c with options(matprod="default") */
static PyObject *r_matprod(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;
    if (!PyArg_ParseTuple(args, "OO", &ox, &oy))
    {
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY_RO | NPY_ARRAY_FORCECAST);
    PyArrayObject *b = (PyArrayObject *)PyArray_FROMANY(oy, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY_RO | NPY_ARRAY_FORCECAST);
    PyObject *ans = NULL;
    if (a && b)
    {
        int nrx = (int)PyArray_DIM(a, 0), ncx = (int)PyArray_DIM(a, 1);
        int nry = (int)PyArray_DIM(b, 0), ncy = (int)PyArray_DIM(b, 1);
        if (ncx != nry)
        {
            PyErr_SetString(PyExc_ValueError, "non-conformable arguments");
        }
        else
        {
            npy_intp dims[2] = {nrx, ncy};
            ans = PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
            if (ans)
            {
                double *x = PyArray_DATA(a), *y = PyArray_DATA(b), *z = PyArray_DATA((PyArrayObject *)ans);
                R_xlen_t NRX = nrx, NRY = nry;
                if (nrx == 0 || ncx == 0 || nry == 0 || ncy == 0)
                {
                    /* zeros already */
                }
                else if (mayHaveNaNOrInf(x, NRX * ncx) || mayHaveNaNOrInf(y, NRY * ncy))
                {
                    simple_matprod(x, nrx, ncx, y, nry, ncy, z);
                }
                else
                {
                    double one = 1.0, zero = 0.0;
                    int ione = 1;
                    if (ncy == 1)
                    {
                        F77_CALL(dgemv)("N", &nrx, &ncx, &one, x, &nrx, y, &ione, &zero, z, &ione FCONE);
                    }
                    else if (nrx == 1)
                    {
                        F77_CALL(dgemv)("T", &nry, &ncy, &one, y, &nry, x, &ione, &zero, z, &ione FCONE);
                    }
                    else
                    {
                        F77_CALL(dgemm)("N", "N", &nrx, &ncy, &ncx, &one, x, &nrx, y, &nry, &zero, z, &nrx FCONE FCONE);
                    }
                }
            }
        }
    }
    Py_XDECREF(a);
    Py_XDECREF(b);
    return ans;
}

/* stats C_Cdqrls(x, y, tol, FALSE) used by lm.wfit: y is a vector */
static PyObject *r_cdqrls(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;
    double tol;
    if (!PyArg_ParseTuple(args, "OOd", &ox, &oy, &tol))
    {
        return NULL;
    }
    PyArrayObject *qr = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    PyArrayObject *y = (PyArrayObject *)PyArray_FROMANY(oy, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    PyObject *ans = NULL;
    if (qr && y)
    {
        int n = (int)PyArray_DIM(qr, 0), p = (int)PyArray_DIM(qr, 1), ny = 1, rank = 0;
        npy_intp np = p, nn = n;
        PyObject *coef = PyArray_ZEROS(1, &np, NPY_DOUBLE, 0);
        PyObject *resid = PyArray_NewCopy(y, NPY_CORDER);
        PyObject *eff = PyArray_NewCopy(y, NPY_CORDER);
        PyObject *pivot = PyArray_ZEROS(1, &np, NPY_INT32, 0);
        PyObject *qraux = PyArray_ZEROS(1, &np, NPY_DOUBLE, 0);
        double *work = calloc(2 * (size_t)(p > 0 ? p : 1), sizeof(double));
        if (coef && resid && eff && pivot && qraux && work && nn == PyArray_SIZE(y))
        {
            int *ip = PyArray_DATA((PyArrayObject *)pivot);
            for (int i = 0; i < p; i++)
            {
                ip[i] = i + 1;
            }
            F77_CALL(dqrls)(PyArray_DATA(qr), &n, &p, PyArray_DATA(y), &ny, &tol, PyArray_DATA((PyArrayObject *)coef),
                            PyArray_DATA((PyArrayObject *)resid), PyArray_DATA((PyArrayObject *)eff), &rank, ip,
                            PyArray_DATA((PyArrayObject *)qraux), work);
            int pivoted = 0;
            for (int i = 0; i < p; i++)
            {
                if (ip[i] != i + 1)
                {
                    pivoted = 1;
                    break;
                }
            }
            ans = Py_BuildValue("{s:O,s:O,s:O,s:O,s:i,s:O,s:O,s:d,s:O}", "qr", qr, "coefficients", coef, "residuals", resid,
                                "effects", eff, "rank", rank, "pivot", pivot, "qraux", qraux, "tol", tol,
                                "pivoted", pivoted ? Py_True : Py_False);
        }
        free(work);
        Py_XDECREF(coef);
        Py_XDECREF(resid);
        Py_XDECREF(eff);
        Py_XDECREF(pivot);
        Py_XDECREF(qraux);
    }
    Py_XDECREF(qr);
    Py_XDECREF(y);
    return ans;
}

/* ------------------------------------------------------------------ */
/* eigen, partial sort, axis ticks                                      */

/* R: La_rs(x, only.values = FALSE) of src/modules/lapack/Lapack.c, the
 * symmetric branch of eigen(): LAPACK dsyevr ("V", "A", "L", abstol 0) on a
 * copy of x after a workspace query. Values come back ascending; eigen()
 * reverses values and vectors in R code (_r.eigen_symmetric). */
static PyObject *r_la_rs(PyObject *self, PyObject *args)
{
    PyObject *ox;
    if (!PyArg_ParseTuple(args, "O", &ox))
    {
        return NULL;
    }
    PyArrayObject *x = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    if (x == NULL)
    {
        return NULL;
    }
    int n = (int)PyArray_DIM(x, 0);
    if (n != (int)PyArray_DIM(x, 1))
    {
        Py_DECREF(x);
        PyErr_SetString(PyExc_ValueError, "'x' must be a square numeric matrix");
        return NULL;
    }
    char jobv[2] = "V", uplo[2] = "L", range[2] = "A";
    double vl = 0.0, vu = 0.0, abstol = 0.0, tmp;
    int il = 0, iu = 0, m, info = 0, lwork = -1, liwork = -1, itmp;
    npy_intp nv = n, dims[2] = {n, n};
    PyObject *values = PyArray_ZEROS(1, &nv, NPY_DOUBLE, 0);
    PyObject *vectors = PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
    int *isuppz = malloc(2 * (size_t)(n > 0 ? n : 1) * sizeof(int));
    double *work = NULL;
    int *iwork = NULL;
    PyObject *ans = NULL;
    if (values && vectors && isuppz)
    {
        double *rx = PyArray_DATA(x), *rvalues = PyArray_DATA((PyArrayObject *)values),
               *rz = PyArray_DATA((PyArrayObject *)vectors);
        F77_CALL(dsyevr)(jobv, range, uplo, &n, rx, &n, &vl, &vu, &il, &iu, &abstol, &m, rvalues,
                         rz, &n, isuppz, &tmp, &lwork, &itmp, &liwork, &info FCONE FCONE FCONE);
        if (info == 0)
        {
            lwork = (int)tmp;
            liwork = itmp;
            work = malloc((size_t)lwork * sizeof(double));
            iwork = malloc((size_t)liwork * sizeof(int));
            if (work && iwork)
            {
                F77_CALL(dsyevr)(jobv, range, uplo, &n, rx, &n, &vl, &vu, &il, &iu, &abstol, &m, rvalues,
                                 rz, &n, isuppz, work, &lwork, iwork, &liwork, &info FCONE FCONE FCONE);
            }
            else
            {
                PyErr_NoMemory();
            }
        }
        if (!PyErr_Occurred())
        {
            if (info != 0)
            {
                PyErr_Format(PyExc_RuntimeError, "error code %d from Lapack routine '%s'", info, "dsyevr");
            }
            else
            {
                ans = Py_BuildValue("{s:O,s:O}", "values", values, "vectors", vectors);
            }
        }
    }
    else if (!PyErr_Occurred())
    {
        PyErr_NoMemory();
    }
    free(isuppz);
    free(work);
    free(iwork);
    Py_DECREF(x);
    Py_XDECREF(values);
    Py_XDECREF(vectors);
    return ans;
}

/* R: sort.int(x, partial = k) for a double vector without NAs and one 1-based
 * index k: do_psort -> Psort0 -> rPsort2(x, 0, n-1, k-1) on a copy
 * (src/main/sort.c); nothing to do when n < 2 */
static PyObject *r_psort(PyObject *self, PyObject *args)
{
    PyObject *ox;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &ox, &k))
    {
        return NULL;
    }
    PyArrayObject *x = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    if (x == NULL)
    {
        return NULL;
    }
    int n = (int)PyArray_DIM(x, 0);
    if (k < 1 || k > n)
    {
        Py_DECREF(x);
        PyErr_Format(PyExc_ValueError, "'partial' index %d outside bounds", k);
        return NULL;
    }
    if (n >= 2)
    {
        rPsort(PyArray_DATA(x), n, k - 1);
    }
    return (PyObject *)x;
}

/* R_ext/GraphicsEngine.h; defined in the verbatim excerpt c/engine_subset.c */
void GEPretty(double *lo, double *up, int *ndiv);

/* Default axis of a base-graphics plot for data range (min, max) and axis
 * style 'r' (extend by 4%) or 'i', as par("usr"), par("xaxp") and axTicks():
 * GScale (src/library/graphics/src/graphics.c, linear branch), GAxisPars
 * (src/main/graphics.c, linear branch, which calls GEPretty -> R_pretty)
 * with n = par("lab")[1] = 5, and CreateAtVector (src/main/plot.c, linear
 * branch). R_pretty and GEPretty are R's own code (verbatim excerpts). */
static PyObject *r_axis_ticks(PyObject *self, PyObject *args)
{
    double min, max;
    int style, n = 5;
    if (!PyArg_ParseTuple(args, "ddC", &min, &max, &style))
    {
        return NULL;
    }
    if (style != 'r' && style != 'i')
    {
        PyErr_Format(PyExc_ValueError, "axis style \"%c\" unimplemented", style);
        return NULL;
    }
    /* GScale */
#define EPS_FAC_1 16
    if (!R_FINITE(min))
    {
        min = .45 * ((min < 0) ? -DBL_MAX : DBL_MAX);
    }
    if (!R_FINITE(max))
    {
        max = .45 * ((max < 0) ? -DBL_MAX : DBL_MAX);
    }
    double temp = fmax2(fabs(max), fabs(min));
    if (temp == 0)
    {
        min = -1;
        max = 1;
    }
    else
    {
        double tf = (temp > 1) ? (temp * DBL_EPSILON) * EPS_FAC_1 : (temp * EPS_FAC_1) * DBL_EPSILON;
        if (tf == 0)
        {
            tf = DBL_MIN;
        }
        if (fabs(max - min) < tf)
        {
            temp *= (min == max) ? .4 : 1e-2;
            min -= temp;
            max += temp;
        }
    }
    if (style == 'r')
    {
        temp = (temp > 100 ? 0.04 * max - 0.04 * min : 0.04 * (max - min));
        double d;
        d = min - temp;
        if (R_FINITE(d))
        {
            min = d;
        }
        else
        {
            min = (d < 0) ? -DBL_MAX : DBL_MAX;
        }
        d = max + temp;
        if (R_FINITE(d))
        {
            max = d;
        }
        else
        {
            max = (d < 0) ? -DBL_MAX : DBL_MAX;
        }
    }
    double usr0 = min, usr1 = max;

    /* GAxisPars, linear */
#define EPS_FAC_2 16
    bool swap = min > max;
    if (swap)
    {
        double t = min;
        min = max;
        max = t;
    }
    double min_o = min, max_o = max;
    GEPretty(&min, &max, &n);
    double t_ = fmax2(fabs(max), fabs(min)),
           tf = (t_ > 1) ? (t_ * DBL_EPSILON) * EPS_FAC_2 : (t_ * EPS_FAC_2) * DBL_EPSILON;
    if (tf == 0)
    {
        tf = DBL_MIN;
    }
    if (fabs(max - min) <= tf)
    {
        min = min_o;
        max = max_o;
        double eps = .005 * (max - min);
        min += eps;
        max -= eps;
        n = 1;
    }
    if (swap)
    {
        double t = min;
        min = max;
        max = t;
    }

    /* CreateAtVector, linear */
#define SMALL_F 100.
    double axp[3] = {min, max, n};
    int nint = (int)(fabs(axp[2]) + 0.25);
    double dn = imax2(1, nint), rng = axp[1] - axp[0], small, a_i;
    npy_intp len = nint + 1;
    PyObject *at = PyArray_ZEROS(1, &len, NPY_DOUBLE, 0);
    if (at == NULL)
    {
        return NULL;
    }
    double *rat = PyArray_DATA((PyArrayObject *)at);
    if (!R_FINITE(rng))
    {
        double at_ = axp[0] / dn;
        rng = axp[1] / dn - at_;
        small = fabs(rng) / SMALL_F;
        int n2 = nint / 2, i;
        for (i = 0; i <= n2; i++)
        {
            a_i = axp[0] + i * rng;
            rat[i] = (fabs(a_i) < small) ? 0. : a_i;
        }
        for (int i2 = 0; i2 < nint - n2; i2++)
        {
            i = nint - i2;
            a_i = axp[1] - i2 * rng;
            rat[i] = (fabs(a_i) < small) ? 0. : a_i;
        }
    }
    else
    {
        small = fabs(rng) / SMALL_F / dn;
        for (int i = 0; i <= nint; i++)
        {
            a_i = axp[0] + (i / dn) * rng;
            rat[i] = (fabs(a_i) < small) ? 0. : a_i;
        }
    }
    PyObject *ans = Py_BuildValue("{s:(dd),s:(ddi),s:O}", "usr", usr0, usr1, "axp", axp[0], axp[1], n, "at", at);
    Py_DECREF(at);
    return ans;
}

/* ------------------------------------------------------------------ */

PyMethodDef rcompat_methods[] = {
    {"r_log", r_log, METH_VARARGS, "R log(x)"},
    {"r_log2", r_log2, METH_VARARGS, "R log2(x)"},
    {"r_log10", r_log10, METH_VARARGS, "R log10(x)"},
    {"r_exp", r_exp, METH_VARARGS, "R exp(x)"},
    {"r_log1p", r_log1p, METH_VARARGS, "R log1p(x)"},
    {"r_sqrt", r_sqrt, METH_VARARGS, "R sqrt(x)"},
    {"r_lgamma", r_lgamma, METH_VARARGS, "R lgamma(x) (nmath lgammafn)"},
    {"r_trigamma", r_trigamma, METH_VARARGS, "R trigamma(x)"},
    {"r_pow", r_pow, METH_VARARGS, "R x^y (R_POW)"},
    {"r_pf", r_pf, METH_VARARGS, "R pf(q, df1, df2, lower.tail, log.p)"},
    {"r_qf", r_qf, METH_VARARGS, "R qf(p, df1, df2, lower.tail, log.p)"},
    {"r_parse_dbl", r_parse_dbl, METH_VARARGS, "readr::read_tsv double parser (BSD strtod) over str fields"},
    {"r_sum", r_sum, METH_VARARGS, "R sum(x, na.rm)"},
    {"r_mean", r_mean, METH_VARARGS, "R mean(x)"},
    {"r_cumsum", r_cumsum, METH_VARARGS, "R cumsum(x)"},
    {"r_colsum", r_colsum, METH_VARARGS, "R colSums/colMeans/rowSums/rowMeans (op 0..3)"},
    {"r_rowsum", r_rowsum, METH_VARARGS, "R rowsum(x, group, reorder=FALSE) with group indices"},
    {"r_brent_fmin", r_brent_fmin, METH_VARARGS, "R optimize(f, c(lower, upper), tol)$minimum"},
    {"r_lowess", r_lowess, METH_VARARGS, "R lowess fitted values for sorted x (edgeR lowess.f)"},
    {"r_weighted_lowess", r_weighted_lowess, METH_VARARGS, "limma weighted_lowess"},
    {"r_qr", r_qr, METH_VARARGS, "R qr(x, tol) via LINPACK dqrdc2"},
    {"r_qr_qy", r_qr_qy, METH_VARARGS, "R qr.qty / qr.qy"},
    {"r_backsolve", r_backsolve, METH_VARARGS, "R backsolve(r, b, k, upper.tri, transpose)"},
    {"r_matprod", r_matprod, METH_VARARGS, "R x %*% y"},
    {"r_cdqrls", r_cdqrls, METH_VARARGS, "stats C_Cdqrls(x, y, tol)"},
    {"r_la_rs", r_la_rs, METH_VARARGS, "R La_rs(x): symmetric eigen via LAPACK dsyevr (ascending)"},
    {"r_psort", r_psort, METH_VARARGS, "R sort.int(x, partial = k) via rPsort2"},
    {"r_axis_ticks", r_axis_ticks, METH_VARARGS, "R par(usr), par(xaxp), axTicks for a data range (GScale, GEPretty)"},
    {NULL, NULL, 0, NULL},
};
