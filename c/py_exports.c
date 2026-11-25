/* py_exports.c -- Python entry points for the edgeR kernels.
 *
 * Each function mirrors the .Call wrapper of the same name in edgeR's
 * R_exports.c: same name, same argument order, same 1-based conventions,
 * same result names. Only the object marshalling differs:
 *
 *   R matrix (SEXPtocmx1)        2-D NumPy array. Read in place when it is
 *                                float64 or int32 and Fortran-contiguous, as
 *                                R aliases REAL()/INTEGER(); any other array
 *                                is copied to float64 (counted, see
 *                                copy_count()).
 *   CompressedMatrix (SEXPtocmx2) tuple (values, nrow, ncol, repeat_row,
 *                                repeat_col), values a float64 array holding
 *                                the stored row/column/scalar/matrix.
 *   numeric vector (REAL)        1-D float64 array (copied if not).
 *
 * Every kernel runs through rshim_run() with the GIL released; an R error()
 * becomes RuntimeError and R warnings become RuntimeWarning.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL edgerpy_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define R_NO_REMAP
#include "edgerpy.h"

/* number of inputs that could not be read in place since the last reset */
static Py_ssize_t n_copies = 0;

/* ------------------------------------------------------------------ */
/* argument conversion                                                  */

/* Holds the arrays an entry point borrows so they can be released. */
typedef struct
{
    PyObject *objs[24];
    int n;
} keep_list;

static void keep_release(keep_list *k)
{
    for (int i = 0; i < k->n; ++i)
    {
        Py_XDECREF(k->objs[i]);
    }
    k->n = 0;
}

static int keep_add(keep_list *k, PyObject *o)
{
    if (k->n >= (int)(sizeof(k->objs) / sizeof(k->objs[0])))
    {
        Py_DECREF(o);
        PyErr_SetString(PyExc_RuntimeError, "edgerpy: too many array arguments");
        return -1;
    }
    k->objs[k->n++] = o;
    return 0;
}

/* Is a an aligned, Fortran-contiguous array of the given type? */
static int usable_inplace(PyArrayObject *a, int type)
{
    return PyArray_TYPE(a) == type && PyArray_ISFARRAY_RO(a);
}

/* R matrix -> cmx (SEXPtocmx1). Double or int32 data is aliased. */
static int to_dense(PyObject *obj, const char *name, cmx *out, keep_list *k)
{
    PyArrayObject *a = NULL;
    if (PyArray_Check(obj) && PyArray_NDIM((PyArrayObject *)obj) == 2 &&
        (usable_inplace((PyArrayObject *)obj, NPY_DOUBLE) || usable_inplace((PyArrayObject *)obj, NPY_INT32)))
    {
        Py_INCREF(obj);
        a = (PyArrayObject *)obj;
    }
    else
    {
        a = (PyArrayObject *)PyArray_FROMANY(obj, NPY_DOUBLE, 2, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (a == NULL)
        {
            PyErr_Format(PyExc_TypeError, "%s: expected a 2-D numeric matrix", name);
            return -1;
        }
        n_copies++;
    }
    if (keep_add(k, (PyObject *)a) < 0)
    {
        return -1;
    }
    int nrow = (int)PyArray_DIM(a, 0), ncol = (int)PyArray_DIM(a, 1);
    if (PyArray_TYPE(a) == NPY_INT32)
    {
        *out = make_cmx(NULL, (int *)PyArray_DATA(a), nrow, ncol, 0, 1);
    }
    else
    {
        *out = make_cmx((double *)PyArray_DATA(a), NULL, nrow, ncol, 0, 0);
    }
    return 0;
}

/* CompressedMatrix tuple -> cmx (SEXPtocmx2): type = 2*repeat_row + repeat_col */
static int to_compressed(PyObject *obj, const char *name, cmx *out, keep_list *k)
{
    PyObject *values;
    int nrow, ncol, rrow, rcol;
    if (!PyTuple_Check(obj) || !PyArg_ParseTuple(obj, "Oiipp", &values, &nrow, &ncol, &rrow, &rcol))
    {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "%s: expected (values, nrow, ncol, repeat_row, repeat_col)", name);
        return -1;
    }
    PyArrayObject *a;
    if (PyArray_Check(values) && usable_inplace((PyArrayObject *)values, NPY_DOUBLE))
    {
        Py_INCREF(values);
        a = (PyArrayObject *)values;
    }
    else
    {
        a = (PyArrayObject *)PyArray_FROMANY(values, NPY_DOUBLE, 0, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (a == NULL)
        {
            return -1;
        }
        n_copies++;
    }
    if (keep_add(k, (PyObject *)a) < 0)
    {
        return -1;
    }
    npy_intp stored = PyArray_SIZE(a);
    npy_intp want = rrow ? (rcol ? 1 : ncol) : (rcol ? nrow : (npy_intp)nrow * ncol);
    if (stored != want)
    {
        PyErr_Format(PyExc_ValueError, "%s: CompressedMatrix stores %zd values, expected %zd", name, (Py_ssize_t)stored, (Py_ssize_t)want);
        return -1;
    }
    *out = make_cmx((double *)PyArray_DATA(a), NULL, nrow, ncol, 2 * rrow + rcol, 0);
    return 0;
}

/* numeric vector -> double pointer (REAL after coerceVector) */
static int to_dvec(PyObject *obj, const char *name, double **out, npy_intp *len, keep_list *k)
{
    PyArrayObject *a;
    if (PyArray_Check(obj) && PyArray_NDIM((PyArrayObject *)obj) <= 2 &&
        usable_inplace((PyArrayObject *)obj, NPY_DOUBLE))
    {
        Py_INCREF(obj);
        a = (PyArrayObject *)obj;
    }
    else
    {
        a = (PyArrayObject *)PyArray_FROMANY(obj, NPY_DOUBLE, 0, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (a == NULL)
        {
            PyErr_Format(PyExc_TypeError, "%s: expected a numeric vector", name);
            return -1;
        }
        n_copies++;
    }
    if (keep_add(k, (PyObject *)a) < 0)
    {
        return -1;
    }
    *out = (double *)PyArray_DATA(a);
    if (len)
    {
        *len = PyArray_SIZE(a);
    }
    return 0;
}

/* integer vector -> int pointer (INTEGER after coerceVector) */
static int to_ivec(PyObject *obj, const char *name, int **out, npy_intp *len, keep_list *k)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(obj, NPY_INT32, 0, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (a == NULL)
    {
        PyErr_Format(PyExc_TypeError, "%s: expected an integer vector", name);
        return -1;
    }
    if (keep_add(k, (PyObject *)a) < 0)
    {
        return -1;
    }
    *out = (int *)PyArray_DATA(a);
    *len = PyArray_SIZE(a);
    return 0;
}

/* sample weights (technical t, biological b): None (all 1) or a numeric vector */
typedef struct
{
    double *t, *b;
    npy_intp nt, nb;
} sw_args;

static int to_sample_weights(PyObject *t, PyObject *b, sw_args *s, keep_list *k)
{
    s->t = s->b = NULL;
    s->nt = s->nb = 0;
    if (t != Py_None && to_dvec(t, "technical.weights", &s->t, &s->nt, k) < 0)
    {
        return -1;
    }
    if (b != Py_None && to_dvec(b, "biological.weights", &s->b, &s->nb, k) < 0)
    {
        return -1;
    }
    return 0;
}

/* R_exports.c sample_vec_cmx: per-sample weights as a cmx with one value per
 * sample (type 2), without a copy; returns 0 when v is NULL (all 1). Called
 * inside rshim_run, so error() unwinds as in R. */
static int sample_vec_cmx(double *v, npy_intp len, int ntag, int nlib, cmx *out)
{
    if (v == NULL)
    {
        return 0;
    }
    if (len != nlib)
    {
        Rf_error("sample weights must be a double vector with one value per sample");
    }
    *out = make_cmx(v, NULL, ntag, nlib, 2, 0);
    return 1;
}

/* R_exports.c fold_biological: the dispersion of sample j becomes disp_j / b_j
 * (R_alloc'd, released at the end of rshim_run) */
static void fold_biological(cmx *dmx, const double *b, npy_intp len)
{
    if (b == NULL)
    {
        return;
    }
    int nlib = dmx->ncol;
    if (len != nlib)
    {
        Rf_error("biological weights must be a double vector with one value per sample");
    }
    if (dmx->type != 2 && dmx->type != 3)
    {
        Rf_error("biological.weights require a common or sample-wise dispersion");
    }
    double *pc = (double *)R_alloc(nlib, sizeof(double));
    get_row(dmx, 0, pc);
    for (int j = 0; j < nlib; ++j)
    {
        pc[j] /= b[j];
    }
    *dmx = make_cmx(pc, NULL, dmx->nrow, nlib, 2, 0);
}

/* ------------------------------------------------------------------ */
/* outputs                                                              */

static PyObject *new_dmat(npy_intp nrow, npy_intp ncol)
{
    npy_intp dims[2] = {nrow, ncol};
    return PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
}

static PyObject *new_vec(npy_intp n, int type)
{
    return PyArray_ZEROS(1, &n, type, 0);
}

#define DPTR(o) ((double *)PyArray_DATA((PyArrayObject *)(o)))
#define IPTR(o) ((int *)PyArray_DATA((PyArrayObject *)(o)))

/* Run a kernel with the GIL released; turn R error/warnings into Python ones. */
static int run_kernel(void (*fn)(void *), void *arg)
{
    rshim_status st;
    int failed;
    Py_BEGIN_ALLOW_THREADS
    failed = rshim_run(fn, arg, &st);
    Py_END_ALLOW_THREADS
    for (int i = 0; i < st.nwarn; ++i)
    {
        if (PyErr_WarnEx(PyExc_RuntimeWarning, st.warn[i], 1) < 0)
        {
            return -1;
        }
    }
    if (failed)
    {
        PyErr_SetString(PyExc_RuntimeError, st.error);
        return -1;
    }
    return 0;
}

/* Build a dict from (name, new reference) pairs; steals the references. */
static PyObject *make_dict(int n, const char **names, PyObject **vals)
{
    PyObject *d = PyDict_New();
    for (int i = 0; i < n; ++i)
    {
        if (d == NULL || vals[i] == NULL || PyDict_SetItemString(d, names[i], vals[i]) < 0)
        {
            Py_XDECREF(d);
            d = NULL;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        Py_XDECREF(vals[i]);
    }
    return d;
}

/* ------------------------------------------------------------------ */
/* ave_log_cpm(y, offsets, priors, disp, weights, maxit, tol, nthreads) */

typedef struct
{
    cmx y, o, p, d, w;
    int maxit, nthr;
    double tol;
    double *out;
} alc_args;

static void alc_run(void *v)
{
    alc_args *a = v;
    average_log_cpm(&a->y, &a->o, &a->p, &a->d, &a->w, a->maxit, a->tol, a->out, a->nthr);
}

static PyObject *py_ave_log_cpm(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *p, *d, *w, *ans = NULL;
    alc_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOidi", &y, &o, &p, &d, &w, &a.maxit, &a.tol, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "y", &a.y, &k) < 0 || to_compressed(o, "offsets", &a.o, &k) < 0 ||
        to_compressed(p, "priors", &a.p, &k) < 0 || to_compressed(d, "disp", &a.d, &k) < 0 ||
        to_compressed(w, "weights", &a.w, &k) < 0)
    {
        goto done;
    }
    ans = new_vec(a.y.nrow, NPY_DOUBLE);
    if (ans == NULL)
    {
        goto done;
    }
    a.out = DPTR(ans);
    if (run_kernel(alc_run, &a) < 0)
    {
        Py_CLEAR(ans);
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* coxreid_disp_top(y, offsets, weights, design, avelogcpm, lower, upper, tol, adjust, nthreads) */

typedef struct
{
    cmx y, o, w, g;
    double *ab, lower, upper, tol, res;
    int adjust, nthr;
} crt_args;

static void crt_run(void *v)
{
    crt_args *a = v;
    a->res = coxreid_disp_top_opt(&a->y, &a->o, &a->w, &a->g, a->ab, a->lower, a->upper, a->tol, a->adjust, a->nthr);
}

static PyObject *py_coxreid_disp_top(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *w, *g, *ab, *ans = NULL;
    crt_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOdddpi", &y, &o, &w, &g, &ab, &a.lower, &a.upper, &a.tol, &a.adjust, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "y", &a.y, &k) < 0 || to_compressed(o, "offsets", &a.o, &k) < 0 ||
        to_compressed(w, "weights", &a.w, &k) < 0 || to_dense(g, "design", &a.g, &k) < 0 ||
        to_dvec(ab, "avelogcpm", &a.ab, NULL, &k) < 0)
    {
        goto done;
    }
    if (run_kernel(crt_run, &a) == 0)
    {
        ans = PyFloat_FromDouble(a.res);
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* fit_glm(y, offsets, disp, weights, tweights, bweights, design, maxit, tol,
 *         coef_start, nthreads); tweights/bweights may be None */

typedef struct
{
    cmx y, o, d, w, g;
    sw_args sw;
    int maxit, nthr, method;
    double tol;
    double *start, *coef, *mu, *dev;
    int *iter, *failed;
} fg_args;

static void fg_run(void *v)
{
    fg_args *a = v;
    cmx ymx = a->y, omx = a->o, dmx = a->d;
    fold_biological(&dmx, a->sw.b, a->sw.nb);
    /* technical weights: fit the count-scaled data t*y with offset + log(t),
     * then return the fitted values on the original scale (mu = mu_scaled / t) */
    cmx tmx;
    int ntag = ymx.nrow, nlibs = ymx.ncol;
    int has_t = sample_vec_cmx(a->sw.t, a->sw.nt, ntag, nlibs, &tmx);
    if (has_t)
    {
        scale_technical(&ymx, &omx, &tmx, &ymx, &omx);
    }
    fit_glm_mat(&ymx, &omx, &dmx, &a->w, &a->g, a->maxit, a->tol, a->start, a->coef, a->mu, a->dev, a->iter, a->failed, &a->method, a->nthr);
    if (has_t)
    {
        double *tr = (double *)R_alloc(nlibs, sizeof(double));
        for (int tag = 0; tag < ntag; ++tag)
        {
            get_row(&tmx, tag, tr);
            for (int lib = 0; lib < nlibs; ++lib)
            {
                a->mu[tag + (npy_intp)lib * ntag] /= tr[lib];
            }
        }
    }
}

static PyObject *py_fit_glm(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *d, *w, *tw, *bw, *g, *start, *ans = NULL;
    fg_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOOOidOi", &y, &o, &d, &w, &tw, &bw, &g, &a.maxit, &a.tol, &start, &a.nthr))
    {
        return NULL;
    }
    a.start = NULL;
    if (to_dense(y, "y", &a.y, &k) < 0 || to_compressed(o, "offsets", &a.o, &k) < 0 ||
        to_compressed(d, "disp", &a.d, &k) < 0 || to_compressed(w, "weights", &a.w, &k) < 0 ||
        to_sample_weights(tw, bw, &a.sw, &k) < 0 || to_dense(g, "design", &a.g, &k) < 0 ||
        (start != Py_None && to_dvec(start, "coef_start", &a.start, NULL, &k) < 0))
    {
        goto done;
    }
    {
        npy_intp ntag = a.y.nrow, ncoef = a.g.ncol, nlib = a.y.ncol;
        PyObject *v[6] = {new_dmat(ntag, ncoef), new_dmat(ntag, nlib), new_vec(ntag, NPY_DOUBLE),
                          new_vec(ntag, NPY_INT32), new_vec(ntag, NPY_INT32), NULL};
        if (v[0] && v[1] && v[2] && v[3] && v[4])
        {
            a.coef = DPTR(v[0]);
            a.mu = DPTR(v[1]);
            a.dev = DPTR(v[2]);
            a.iter = IPTR(v[3]);
            a.failed = IPTR(v[4]);
            a.method = 0;
            if (run_kernel(fg_run, &a) == 0)
            {
                v[5] = PyUnicode_FromString(a.method == 0 ? "oneway" : "levenberg");
                const char *names[] = {"coefficients", "fitted.values", "deviance", "iter", "failed", "method"};
                ans = make_dict(6, names, v);
                goto done;
            }
        }
        for (int i = 0; i < 6; ++i)
        {
            Py_XDECREF(v[i]);
        }
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* fit_apl(y, offsets, disp, weights, design, maxit, tol, coef_start, adjust, nthreads)
 * = edgeR fit_apl (R_exports.c): fit_glm_mat, then the adjusted profile
 * likelihood from its fitted values; returns dict(apl, coefficients) */

typedef struct
{
    fg_args fg;
    int adjust;
    double *apl;
} apl_args;

static void apl_run(void *v)
{
    apl_args *a = v;
    fg_run(&a->fg);
    cmx umx = make_cmx(a->fg.mu, NULL, a->fg.y.nrow, a->fg.y.ncol, 0, 0);
    compute_adj_profile_ll(&a->fg.y, &umx, &a->fg.d, &a->fg.w, &a->fg.g, a->adjust, a->apl, a->fg.nthr);
}

static PyObject *py_fit_apl(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *d, *w, *g, *start, *ans = NULL;
    apl_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOidOpi", &y, &o, &d, &w, &g, &a.fg.maxit, &a.fg.tol, &start, &a.adjust,
                          &a.fg.nthr))
    {
        return NULL;
    }
    a.fg.start = NULL;
    a.fg.sw = (sw_args){NULL, NULL, 0, 0};
    if (to_dense(y, "y", &a.fg.y, &k) < 0 || to_compressed(o, "offsets", &a.fg.o, &k) < 0 ||
        to_compressed(d, "disp", &a.fg.d, &k) < 0 || to_compressed(w, "weights", &a.fg.w, &k) < 0 ||
        to_dense(g, "design", &a.fg.g, &k) < 0 ||
        (start != Py_None && to_dvec(start, "coef_start", &a.fg.start, NULL, &k) < 0))
    {
        goto done;
    }
    {
        npy_intp ntag = a.fg.y.nrow, ncoef = a.fg.g.ncol, nlib = a.fg.y.ncol;
        /* coefficients and apl are returned; mu feeds the APL; deviance/iter/failed are scratch (as R) */
        PyObject *v[2] = {new_vec(ntag, NPY_DOUBLE), new_dmat(ntag, ncoef)};
        PyObject *mu = new_dmat(ntag, nlib);
        a.fg.dev = calloc(ntag > 0 ? ntag : 1, sizeof(double));
        a.fg.iter = calloc(ntag > 0 ? ntag : 1, sizeof(int));
        a.fg.failed = calloc(ntag > 0 ? ntag : 1, sizeof(int));
        if (v[0] && v[1] && mu && a.fg.dev && a.fg.iter && a.fg.failed)
        {
            a.apl = DPTR(v[0]);
            a.fg.coef = DPTR(v[1]);
            a.fg.mu = DPTR(mu);
            a.fg.method = 0;
            if (run_kernel(apl_run, &a) == 0)
            {
                const char *names[] = {"apl", "coefficients"};
                ans = make_dict(2, names, v);      /* takes the references in v */
                v[0] = v[1] = NULL;
            }
        }
        else if (!PyErr_Occurred())
        {
            PyErr_NoMemory();
        }
        free(a.fg.dev);
        free(a.fg.iter);
        free(a.fg.failed);
        Py_XDECREF(mu);
        Py_XDECREF(v[0]);
        Py_XDECREF(v[1]);
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* locfit_by_col(y, x, w, span, degree, nthreads): edgeR locfitByCol after its
 * argument handling (c/locfit_lite.c); returns the fitted values as a new
 * matrix. rshim_run is not reentrant, so the GIL is held throughout. */

typedef struct
{
    double *y;
    const double *x, *w;
    int ntags, ncol, degree, nthreads;
    double span;
} lbc_args;

static void lbc_run(void *v)
{
    lbc_args *a = v;
    locfit_by_col_mat(a->y, a->ntags, a->ncol, a->x, a->w, a->span, a->degree, a->nthreads);
}

static PyObject *py_locfit_by_col(PyObject *self, PyObject *args)
{
    PyObject *oy, *ox, *ow;
    lbc_args a;
    if (!PyArg_ParseTuple(args, "OOOdii", &oy, &ox, &ow, &a.span, &a.degree, &a.nthreads))
    {
        return NULL;
    }
    PyArrayObject *y = (PyArrayObject *)PyArray_FROMANY(oy, NPY_DOUBLE, 2, 2, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY);
    PyArrayObject *x = (PyArrayObject *)PyArray_FROMANY(ox, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO);
    PyArrayObject *w = (PyArrayObject *)PyArray_FROMANY(ow, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO);
    PyObject *ans = NULL;
    if (y && x && w)
    {
        a.ntags = (int)PyArray_DIM(y, 0);
        a.ncol = (int)PyArray_DIM(y, 1);
        if (PyArray_DIM(x, 0) != a.ntags || PyArray_DIM(w, 0) != a.ntags)
        {
            PyErr_SetString(PyExc_ValueError, "x and weights must have one value per row of y");
        }
        else
        {
            rshim_status st;
            a.y = PyArray_DATA(y);
            a.x = PyArray_DATA(x);
            a.w = PyArray_DATA(w);
            int failed = rshim_run(lbc_run, &a, &st);
            for (int i = 0; i < st.nwarn && !PyErr_Occurred(); ++i)
            {
                PyErr_WarnEx(PyExc_RuntimeWarning, st.warn[i], 1);
            }
            if (failed && !PyErr_Occurred())
            {
                PyErr_SetString(PyExc_RuntimeError, st.error);
            }
            if (!PyErr_Occurred())
            {
                ans = Py_NewRef((PyObject *)y);
            }
        }
    }
    Py_XDECREF(y);
    Py_XDECREF(x);
    Py_XDECREF(w);
    return ans;
}

/* ------------------------------------------------------------------ */
/* maximize_interpolant(spline_pts, ll) = edgeR maximize_interpolant (R_exports.c) */

static PyObject *py_maximize_interpolant(PyObject *self, PyObject *args)
{
    PyObject *s, *l, *ans = NULL;
    keep_list k = {.n = 0};
    double *spts;
    cmx lmx;
    if (!PyArg_ParseTuple(args, "OO", &s, &l))
    {
        return NULL;
    }
    if (to_dvec(s, "spline_pts", &spts, NULL, &k) == 0 && to_dense(l, "ll", &lmx, &k) == 0)
    {
        ans = new_vec(lmx.nrow, NPY_DOUBLE);
        if (ans)
        {
            max_interpolant(spts, &lmx, DPTR(ans));
        }
    }
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* pred_fc(y, offsets, priors, disp, weights, tweights, bweights, design, maxit,
 *         tol, nthreads); tweights/bweights may be None */

typedef struct
{
    cmx y, o, p, d, w, g;
    sw_args sw;
    int maxit, nthr;
    double tol;
    double *coef;
} pfc_args;

static void pfc_run(void *v)
{
    pfc_args *a = v;
    cmx ymx = a->y, omx = a->o, dmx = a->d;
    fold_biological(&dmx, a->sw.b, a->sw.nb);
    /* technical weights: prior counts are added to the count-scaled data t*y */
    cmx tmx;
    if (sample_vec_cmx(a->sw.t, a->sw.nt, ymx.nrow, ymx.ncol, &tmx))
    {
        scale_technical(&ymx, &omx, &tmx, &ymx, &omx);
    }
    pred_fc_mat(&ymx, &omx, &a->p, &dmx, &a->w, &a->g, a->maxit, a->tol, a->coef, a->nthr);
}

static PyObject *py_pred_fc(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *p, *d, *w, *tw, *bw, *g, *ans = NULL;
    pfc_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOOOOidi", &y, &o, &p, &d, &w, &tw, &bw, &g, &a.maxit, &a.tol, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "y", &a.y, &k) < 0 || to_compressed(o, "offsets", &a.o, &k) < 0 ||
        to_compressed(p, "priors", &a.p, &k) < 0 || to_compressed(d, "disp", &a.d, &k) < 0 ||
        to_compressed(w, "weights", &a.w, &k) < 0 || to_sample_weights(tw, bw, &a.sw, &k) < 0 ||
        to_dense(g, "design", &a.g, &k) < 0)
    {
        goto done;
    }
    ans = new_dmat(a.y.nrow, a.g.ncol);
    if (ans == NULL)
    {
        goto done;
    }
    a.coef = DPTR(ans);
    if (run_kernel(pfc_run, &a) < 0)
    {
        Py_CLEAR(ans);
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* compute_ave_qd(ym, um, gm, dm, ag, wm, tm, bm, nthreads); tm/bm may be None */

typedef struct
{
    cmx y, u, g, d, w;
    sw_args sw;
    double *ag, res;
    int nthr;
} aqd_args;

static void aqd_run(void *v)
{
    aqd_args *a = v;
    cmx dmx = a->d;
    fold_biological(&dmx, a->sw.b, a->sw.nb);
    cmx tmx;
    int has_t = sample_vec_cmx(a->sw.t, a->sw.nt, a->y.nrow, a->y.ncol, &tmx);
    a->res = update_prior(&a->y, &a->u, &a->g, &dmx, &a->w, has_t ? &tmx : NULL, a->ag, a->nthr);
}

static PyObject *py_compute_ave_qd(PyObject *self, PyObject *args)
{
    PyObject *y, *u, *g, *d, *ag, *w, *tw, *bw, *ans = NULL;
    aqd_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOOOOi", &y, &u, &g, &d, &ag, &w, &tw, &bw, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "ym", &a.y, &k) < 0 || to_dense(u, "um", &a.u, &k) < 0 ||
        to_dense(g, "gm", &a.g, &k) < 0 || to_compressed(d, "dm", &a.d, &k) < 0 ||
        to_dvec(ag, "ag", &a.ag, NULL, &k) < 0 || to_compressed(w, "wm", &a.w, &k) < 0 ||
        to_sample_weights(tw, bw, &a.sw, &k) < 0)
    {
        goto done;
    }
    if (run_kernel(aqd_run, &a) == 0)
    {
        ans = PyFloat_FromDouble(a.res);
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* compute_adj_vec(ym, um, gm, dm, qd, wm, tm, bm, nthreads); tm/bm may be None */

typedef struct
{
    cmx y, u, g, d, w;
    sw_args sw;
    double prior, *df, *dv, *s2;
    int nthr;
} adj_args;

static void adj_run(void *v)
{
    adj_args *a = v;
    cmx dmx = a->d;
    fold_biological(&dmx, a->sw.b, a->sw.nb);
    /* technical weights: NULL means 1 */
    cmx tmx;
    int has_t = sample_vec_cmx(a->sw.t, a->sw.nt, a->y.nrow, a->y.ncol, &tmx);
    compute_adjust_vec(&a->y, &a->u, &a->g, &dmx, a->prior, &a->w, has_t ? &tmx : NULL, a->df, a->dv, a->s2, a->nthr);
}

static PyObject *py_compute_adj_vec(PyObject *self, PyObject *args)
{
    PyObject *y, *u, *g, *d, *w, *tw, *bw, *ans = NULL;
    adj_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOdOOOi", &y, &u, &g, &d, &a.prior, &w, &tw, &bw, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "ym", &a.y, &k) < 0 || to_dense(u, "um", &a.u, &k) < 0 ||
        to_dense(g, "gm", &a.g, &k) < 0 || to_compressed(d, "dm", &a.d, &k) < 0 ||
        to_compressed(w, "wm", &a.w, &k) < 0 || to_sample_weights(tw, bw, &a.sw, &k) < 0)
    {
        goto done;
    }
    {
        npy_intp n = a.y.nrow;
        PyObject *v[3] = {new_vec(n, NPY_DOUBLE), new_vec(n, NPY_DOUBLE), new_vec(n, NPY_DOUBLE)};
        if (v[0] && v[1] && v[2])
        {
            a.df = DPTR(v[0]);
            a.dv = DPTR(v[1]);
            a.s2 = DPTR(v[2]);
            if (run_kernel(adj_run, &a) == 0)
            {
                const char *names[] = {"df", "deviance", "s2"};
                ans = make_dict(3, names, v);
                goto done;
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            Py_XDECREF(v[i]);
        }
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* glm_ftest(counts, offset, dispersion, weights, tweights, bweights, design,
 *           coef, deviance, s2post, df_total, lfc, logFCt, upshot, nthreads);
 * coef 1-based; tweights/bweights may be None */

typedef struct
{
    cmx y, o, d, w, g, l, z;
    sw_args sw;
    int *coef, ncoef, upshot, nthr, has_w;
    double *dev, *s2, *dft, *lfc, *F, *P;
} ft_args;

static void glm_ft_run(void *v)
{
    ft_args *a = v;
    cmx ymx = a->y, omx = a->o, dmx = a->d;
    fold_biological(&dmx, a->sw.b, a->sw.nb);
    /* technical weights: refit the count-scaled data t*y with offset + log(t) */
    cmx tmx;
    if (sample_vec_cmx(a->sw.t, a->sw.nt, ymx.nrow, ymx.ncol, &tmx))
    {
        scale_technical(&ymx, &omx, &tmx, &ymx, &omx);
    }
    glm_ftest_mat(&ymx, &omx, &dmx, &a->w, &a->g, a->coef, a->ncoef, a->dev, a->s2, a->dft, a->lfc, &a->l, a->upshot, a->F, a->P, a->nthr);
}

static void bin_ft_run(void *v)
{
    ft_args *a = v;
    bin_ftest_mat(&a->y, &a->z, &a->w, a->has_w, &a->g, a->coef, a->ncoef, a->dev, a->s2, a->dft, a->lfc, &a->l, a->upshot, a->F, a->P, a->nthr);
}

/* shared tail of glm_ftest / bin_ftest: per-gene vectors, outputs, run */
static PyObject *ftest_finish(ft_args *a, void (*fn)(void *), PyObject *coef, PyObject *dev, PyObject *s2,
                              PyObject *dft, PyObject *lfc, PyObject *logfct, keep_list *k)
{
    npy_intp ncoef;
    if (to_ivec(coef, "coef", &a->coef, &ncoef, k) < 0 || to_dvec(dev, "deviance", &a->dev, NULL, k) < 0 ||
        to_dvec(s2, "s2post", &a->s2, NULL, k) < 0 || to_dvec(dft, "df_total", &a->dft, NULL, k) < 0 ||
        to_dvec(lfc, "lfc", &a->lfc, NULL, k) < 0 || to_dense(logfct, "logFCt", &a->l, k) < 0)
    {
        return NULL;
    }
    a->ncoef = (int)ncoef;
    npy_intp n = a->y.nrow;
    PyObject *v[2] = {new_vec(n, NPY_DOUBLE), new_vec(n, NPY_DOUBLE)};
    if (v[0] && v[1])
    {
        a->F = DPTR(v[0]);
        a->P = DPTR(v[1]);
        if (run_kernel(fn, a) == 0)
        {
            const char *names[] = {"F", "PValue"};
            return make_dict(2, names, v);
        }
    }
    Py_XDECREF(v[0]);
    Py_XDECREF(v[1]);
    return NULL;
}

static PyObject *py_glm_ftest(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *d, *w, *tw, *bw, *g, *coef, *dev, *s2, *dft, *lfc, *logfct, *ans = NULL;
    ft_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOpi", &y, &o, &d, &w, &tw, &bw, &g, &coef, &dev, &s2, &dft, &lfc, &logfct,
                          &a.upshot, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "counts", &a.y, &k) == 0 && to_compressed(o, "offset", &a.o, &k) == 0 &&
        to_compressed(d, "dispersion", &a.d, &k) == 0 && to_compressed(w, "weights", &a.w, &k) == 0 &&
        to_sample_weights(tw, bw, &a.sw, &k) == 0 && to_dense(g, "design", &a.g, &k) == 0)
    {
        ans = ftest_finish(&a, glm_ft_run, coef, dev, s2, dft, lfc, logfct, &k);
    }
    keep_release(&k);
    return ans;
}

/* bin_ftest(counts, counts2, weights, design, coef, deviance, s2post,
 *           df_total, lfc, logFCt, upshot, nthreads); weights may be None */
static PyObject *py_bin_ftest(PyObject *self, PyObject *args)
{
    PyObject *y, *z, *w, *g, *coef, *dev, *s2, *dft, *lfc, *logfct, *ans = NULL;
    ft_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOpi", &y, &z, &w, &g, &coef, &dev, &s2, &dft, &lfc, &logfct, &a.upshot, &a.nthr))
    {
        return NULL;
    }
    a.has_w = (w != Py_None);
    if (to_dense(y, "counts", &a.y, &k) == 0 && to_dense(z, "counts2", &a.z, &k) == 0 &&
        (!a.has_w || to_dense(w, "weights", &a.w, &k) == 0) && to_dense(g, "design", &a.g, &k) == 0)
    {
        ans = ftest_finish(&a, bin_ft_run, coef, dev, s2, dft, lfc, logfct, &k);
    }
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* bin_fit(y, offsets, weights, design, maxit_oneway, tol_oneway,
 *         maxit_iwls, tol_iwls, nthreads)
 * bin_ql_fit(y, z, weights, offsets, design, maxit_oneway, tol_oneway,
 *            maxit_iwls, tol_iwls, nthreads); weights may be None */

typedef struct
{
    cmx y, z, w, o, g;
    int has_w, maxit_ow, maxit_iw, nthr, method;
    double tol_ow, tol_iw;
    double *coef, *mu, *dev, *dfa, *dva, *s2;
    int *iter, *failed;
} bf_args;

static void bf_run(void *v)
{
    bf_args *a = v;
    bin_fit_mat(&a->y, &a->o, &a->w, &a->g, a->maxit_ow, a->tol_ow, a->maxit_iw, a->tol_iw, a->coef, a->mu, a->dev, a->iter, a->failed, &a->method, a->nthr);
}

static void bqf_run(void *v)
{
    bf_args *a = v;
    bin_ql_fit_mat(&a->y, &a->z, &a->w, a->has_w, &a->o, &a->g, a->maxit_ow, a->tol_ow, a->maxit_iw, a->tol_iw, a->coef, a->mu, a->dev, a->iter, a->failed, &a->method, a->dfa, a->dva, a->s2, a->nthr);
}

/* allocate outputs, run, and build the result dict; ql adds the QL vectors */
static PyObject *bin_finish(bf_args *a, int ql)
{
    npy_intp ntag = a->y.nrow, ncoef = a->g.ncol, nlib = a->y.ncol;
    PyObject *v[9] = {new_dmat(ntag, ncoef), new_dmat(ntag, nlib), new_vec(ntag, NPY_DOUBLE),
                      new_vec(ntag, NPY_INT32), new_vec(ntag, NPY_INT32), NULL,
                      ql ? new_vec(ntag, NPY_DOUBLE) : NULL, ql ? new_vec(ntag, NPY_DOUBLE) : NULL,
                      ql ? new_vec(ntag, NPY_DOUBLE) : NULL};
    int nv = ql ? 9 : 6;
    int ok = v[0] && v[1] && v[2] && v[3] && v[4] && (!ql || (v[6] && v[7] && v[8]));
    if (ok)
    {
        a->coef = DPTR(v[0]);
        a->mu = DPTR(v[1]);
        a->dev = DPTR(v[2]);
        a->iter = IPTR(v[3]);
        a->failed = IPTR(v[4]);
        if (ql)
        {
            a->dfa = DPTR(v[6]);
            a->dva = DPTR(v[7]);
            a->s2 = DPTR(v[8]);
        }
        a->method = 0;
        if (run_kernel(ql ? bqf_run : bf_run, a) == 0)
        {
            v[5] = PyUnicode_FromString(a->method == 0 ? "oneway" : "IWLS");
            const char *names[] = {"coefficients", "fitted.values", "deviance", "iter", "failed", "method",
                                   "df.residual.adj", "deviance.adj", "s2"};
            return make_dict(nv, names, v);
        }
    }
    for (int i = 0; i < 9; ++i)
    {
        Py_XDECREF(v[i]);
    }
    return NULL;
}

static PyObject *py_bin_fit(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *w, *g, *ans = NULL;
    bf_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOididi", &y, &o, &w, &g, &a.maxit_ow, &a.tol_ow, &a.maxit_iw, &a.tol_iw, &a.nthr))
    {
        return NULL;
    }
    a.has_w = 1;
    if (to_dense(y, "y", &a.y, &k) == 0 && to_compressed(o, "offsets", &a.o, &k) == 0 &&
        to_dense(w, "weights", &a.w, &k) == 0 && to_dense(g, "design", &a.g, &k) == 0)
    {
        ans = bin_finish(&a, 0);
    }
    keep_release(&k);
    return ans;
}

static PyObject *py_bin_ql_fit(PyObject *self, PyObject *args)
{
    PyObject *y, *z, *w, *o, *g, *ans = NULL;
    bf_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOididi", &y, &z, &w, &o, &g, &a.maxit_ow, &a.tol_ow, &a.maxit_iw, &a.tol_iw, &a.nthr))
    {
        return NULL;
    }
    a.has_w = (w != Py_None);
    if (to_dense(y, "y", &a.y, &k) == 0 && to_dense(z, "z", &a.z, &k) == 0 &&
        (!a.has_w || to_dense(w, "weights", &a.w, &k) == 0) && to_compressed(o, "offsets", &a.o, &k) == 0 &&
        to_dense(g, "design", &a.g, &k) == 0)
    {
        ans = bin_finish(&a, 1);
    }
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* fit_diff_splice(y, offsets, disp, weights, tweights, bweights, design,
 *                 coef_idx, gene_nexons, gene_firstexon, nexons_approx, maxit,
 *                 tol, coef_start, exon_dev, nthreads); coef_idx 1-based as in
 *                 R; tweights/bweights may be None */

typedef struct
{
    cmx y, o, d, w, g;
    sw_args sw;
    int ngenes, coef, *nex, *first, nex_approx, maxit, nthr;
    double tol, *start, *exdev, *exLR, *excoef, *geneLR;
} ds_args;

static void ds_run(void *v)
{
    ds_args *a = v;
    cmx ymx = a->y, omx = a->o, dmx = a->d;
    fold_biological(&dmx, a->sw.b, a->sw.nb);
    /* technical weights: fit the count-scaled data t*y with offset + log(t) */
    cmx tmx;
    if (sample_vec_cmx(a->sw.t, a->sw.nt, ymx.nrow, ymx.ncol, &tmx))
    {
        scale_technical(&ymx, &omx, &tmx, &ymx, &omx);
    }
    fit_diff_splice_mat(&ymx, &omx, &dmx, &a->w, &a->g, a->ngenes, ymx.ncol, a->g.ncol, a->coef, a->nex, a->first,
                        a->nex_approx, a->maxit, a->tol, a->start, a->exdev, a->exLR, a->excoef, a->geneLR, a->nthr);
}

static PyObject *py_fit_diff_splice(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *d, *w, *tw, *bw, *g, *nex, *first, *start, *exdev, *ans = NULL;
    ds_args a;
    int coef1;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOOOiOOiidOOi", &y, &o, &d, &w, &tw, &bw, &g, &coef1, &nex, &first, &a.nex_approx,
                          &a.maxit, &a.tol, &start, &exdev, &a.nthr))
    {
        return NULL;
    }
    a.coef = coef1 - 1; /* convert to 0-indexed, as R_exports.c does */
    a.start = NULL;
    npy_intp ngenes, nfirst;
    if (to_dense(y, "y", &a.y, &k) < 0 || to_compressed(o, "offsets", &a.o, &k) < 0 ||
        to_compressed(d, "disp", &a.d, &k) < 0 || to_compressed(w, "weights", &a.w, &k) < 0 ||
        to_sample_weights(tw, bw, &a.sw, &k) < 0 ||
        to_dense(g, "design", &a.g, &k) < 0 || to_ivec(nex, "gene_nexons", &a.nex, &ngenes, &k) < 0 ||
        to_ivec(first, "gene_firstexon", &a.first, &nfirst, &k) < 0 ||
        (start != Py_None && to_dvec(start, "coef_start", &a.start, NULL, &k) < 0) ||
        to_dvec(exdev, "exon_dev", &a.exdev, NULL, &k) < 0)
    {
        goto done;
    }
    a.ngenes = (int)ngenes;
    {
        PyObject *v[3] = {new_vec(a.y.nrow, NPY_DOUBLE), new_vec(a.y.nrow, NPY_DOUBLE), new_vec(ngenes, NPY_DOUBLE)};
        if (v[0] && v[1] && v[2])
        {
            a.exLR = DPTR(v[0]);
            a.excoef = DPTR(v[1]);
            a.geneLR = DPTR(v[2]);
            if (run_kernel(ds_run, &a) == 0)
            {
                const char *names[] = {"exon.LR", "exon.coef", "gene.LR"};
                ans = make_dict(3, names, v);
                goto done;
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            Py_XDECREF(v[i]);
        }
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* sample_weights(y, offset, weights, design, ave, niter, tol, priorn, nthreads):
 * two-part sample weights (sample_weights_nb); returns dict(t, c) */

typedef struct
{
    cmx y, o, w, g;
    double *ave, tol, priorn, *t, *c;
    int niter, nthr;
} sw_run_args;

static void sw_run(void *v)
{
    sw_run_args *a = v;
    int nupd = sample_weights_nb(&a->y, &a->g, &a->o, &a->w, a->ave, a->niter, a->tol, a->priorn, a->nthr, a->t, a->c);
    if (nupd < 0)
    {
        Rf_error("sampleWeights: the step matrix is not positive definite");
    }
}

static PyObject *py_sample_weights(PyObject *self, PyObject *args)
{
    PyObject *y, *o, *w, *g, *ave, *ans = NULL;
    sw_run_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOOOiddi", &y, &o, &w, &g, &ave, &a.niter, &a.tol, &a.priorn, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "y", &a.y, &k) < 0 || to_compressed(o, "offset", &a.o, &k) < 0 ||
        to_compressed(w, "weights", &a.w, &k) < 0 || to_dense(g, "design", &a.g, &k) < 0 ||
        to_dvec(ave, "ave", &a.ave, NULL, &k) < 0)
    {
        goto done;
    }
    {
        PyObject *v[2] = {new_vec(a.y.ncol, NPY_DOUBLE), new_vec(a.y.ncol, NPY_DOUBLE)};
        if (v[0] && v[1])
        {
            a.t = DPTR(v[0]);
            a.c = DPTR(v[1]);
            if (run_kernel(sw_run, &a) == 0)
            {
                const char *names[] = {"t", "c"};
                ans = make_dict(2, names, v);
                goto done;
            }
        }
        Py_XDECREF(v[0]);
        Py_XDECREF(v[1]);
    }
done:
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* calculate_cpm_raw(y, libsizes, nthreads) */

typedef struct
{
    cmx y, l;
    double *out;
    int nthr;
} cpm_args;

static void cpm_run(void *v)
{
    cpm_args *a = v;
    calc_cpm_raw(&a->y, &a->l, a->out, a->nthr);
}

static PyObject *py_calculate_cpm_raw(PyObject *self, PyObject *args)
{
    PyObject *y, *l, *ans = NULL;
    cpm_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOi", &y, &l, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "y", &a.y, &k) == 0 && to_compressed(l, "libsizes", &a.l, &k) == 0)
    {
        /* R: ans <- coerceVector(duplicate(y), REALSXP), then filled by the kernel */
        ans = PyArray_FROMANY(k.objs[0], NPY_DOUBLE, 2, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
        if (ans)
        {
            a.out = DPTR(ans);
            if (run_kernel(cpm_run, &a) < 0)
            {
                Py_CLEAR(ans);
            }
        }
    }
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* calculate_cpm_log(y, libsizes, priors, nthreads) */

typedef struct
{
    cmx y, l, p;
    double *out;
    int nthr;
} cpm_log_args;

static void cpm_log_run(void *v)
{
    cpm_log_args *a = v;
    calc_cpm_log(&a->y, &a->l, &a->p, a->out, a->nthr);
}

static PyObject *py_calculate_cpm_log(PyObject *self, PyObject *args)
{
    PyObject *y, *l, *p, *ans = NULL;
    cpm_log_args a;
    keep_list k = {.n = 0};
    if (!PyArg_ParseTuple(args, "OOOi", &y, &l, &p, &a.nthr))
    {
        return NULL;
    }
    if (to_dense(y, "y", &a.y, &k) == 0 && to_compressed(l, "libsizes", &a.l, &k) == 0 &&
        to_compressed(p, "priors", &a.p, &k) == 0)
    {
        /* R: ans <- coerceVector(duplicate(y), REALSXP), then filled by the kernel */
        ans = PyArray_FROMANY(k.objs[0], NPY_DOUBLE, 2, 2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
        if (ans)
        {
            a.out = DPTR(ans);
            if (run_kernel(cpm_log_run, &a) < 0)
            {
                Py_CLEAR(ans);
            }
        }
    }
    keep_release(&k);
    return ans;
}

/* ------------------------------------------------------------------ */
/* diagnostics                                                          */

static PyObject *py_copy_count(PyObject *self, PyObject *args)
{
    int reset = 0;
    if (!PyArg_ParseTuple(args, "|p", &reset))
    {
        return NULL;
    }
    Py_ssize_t n = n_copies;
    if (reset)
    {
        n_copies = 0;
    }
    return PyLong_FromSsize_t(n);
}

static PyMethodDef methods[] = {
    {"ave_log_cpm", py_ave_log_cpm, METH_VARARGS, "average log2-CPM per row (edgeR .cxx_ave_log_cpm)"},
    {"coxreid_disp_top", py_coxreid_disp_top, METH_VARARGS, "Cox-Reid common dispersion of the top genes"},
    {"fit_glm", py_fit_glm, METH_VARARGS, "genewise NB GLM fits (edgeR .cxx_fit_glm)"},
    {"fit_apl", py_fit_apl, METH_VARARGS, "GLM fits and Cox-Reid adjusted profile log-likelihoods (edgeR .cxx_fit_apl)"},
    {"locfit_by_col", py_locfit_by_col, METH_VARARGS, "edgeR locfitByCol (locfit's default 1-D fit)"},
    {"maximize_interpolant", py_maximize_interpolant, METH_VARARGS, "spline maximiser of a likelihood grid (edgeR .cxx_maximize_interpolant)"},
    {"pred_fc", py_pred_fc, METH_VARARGS, "shrunk NB GLM coefficients (edgeR .cxx_pred_fc)"},
    {"compute_ave_qd", py_compute_ave_qd, METH_VARARGS, "average quasi-dispersion"},
    {"compute_adj_vec", py_compute_adj_vec, METH_VARARGS, "adjusted deviance, df and s2"},
    {"glm_ftest", py_glm_ftest, METH_VARARGS, "NB QL F-test with TREAT/UPSHOT"},
    {"bin_fit", py_bin_fit, METH_VARARGS, "genewise binomial GLM fits"},
    {"bin_ql_fit", py_bin_ql_fit, METH_VARARGS, "binomial fit plus adjusted deviance, df and s2"},
    {"bin_ftest", py_bin_ftest, METH_VARARGS, "binomial QL F-test with TREAT/UPSHOT"},
    {"fit_diff_splice", py_fit_diff_splice, METH_VARARGS, "gene- and exon-level LRs for diffSplice"},
    {"sample_weights", py_sample_weights, METH_VARARGS, "two-part sample weights (edgeR .cxx_sample_weights)"},
    {"calculate_cpm_raw", py_calculate_cpm_raw, METH_VARARGS, "counts per million (edgeR .cxx_calculate_cpm_raw)"},
    {"calculate_cpm_log", py_calculate_cpm_log, METH_VARARGS, "log2 counts per million (edgeR .cxx_calculate_cpm_log)"},
    {"copy_count", py_copy_count, METH_VARARGS, "inputs copied (not read in place); copy_count(reset=False)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_C", "edgeR C kernels for edgerpy", -1, methods};

extern PyMethodDef rcompat_methods[];

PyMODINIT_FUNC PyInit__C(void)
{
    import_array();
    rshim_init();
    PyObject *m = PyModule_Create(&moduledef);
    if (m != NULL && PyModule_AddFunctions(m, rcompat_methods) < 0)
    {
        Py_CLEAR(m);
    }
    return m;
}
