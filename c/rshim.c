/* rshim.c -- R API symbols needed by the vendored code, implemented for a
 * Python host. The set is exactly what tools/vendor_sources.py reports as
 * unresolved after the vendored closures (RSHIM_SYMBOLS there).
 *
 * Behaviour follows R 4.6.1:
 *   R_chk_calloc / R_chk_free   src/main/memory.c (same checks and message)
 *   R_alloc / vmaxget / vmaxset src/main/memory.c: blocks live until vmaxset()
 *                                to an earlier mark or the end of the call
 *                                (rshim_run), as at the end of a .Call
 *   R_NaN, NA_REAL, R_PosInf ... set by InitArithmetic() from the verbatim
 *                                src/main/arithmetic.c excerpt (rmain)
 *   xerbla_                     src/main/print.c message; the copy into the
 *                                local buffer is bounded (R's version
 *                                inverts the bound, see findings.md)
 *   SEXP accessors              never called: they are referenced only by
 *                                SEXPtocmx1/2 in object.c, which the Python
 *                                entry points replace with make_cmx().
 *   Rprintf / REprintf          print to stdout / stderr, as R's console
 */
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "edgerpy.h"

void InitArithmetic(void);

/* R's arithmetic constants (declared in R_ext/Arith.h) */
double R_NaN;
double R_PosInf;
double R_NegInf;
double R_NaReal;
int R_NaInt;

/* the call currently running on this thread (NULL outside rshim_run) */
static _Thread_local jmp_buf *cur_env;
static _Thread_local rshim_status *cur_status;

/* R_alloc blocks of the running call, newest first; a vmaxget() mark is the top block */
typedef union ralloc_block
{
    union ralloc_block *prev;
    max_align_t align;
} ralloc_block;
static _Thread_local ralloc_block *ralloc_top;

void rshim_init(void)
{
    InitArithmetic();
}

int rshim_run(void (*fn)(void *), void *arg, rshim_status *status)
{
    jmp_buf env;
    status->error[0] = '\0';
    status->nwarn = 0;
    cur_env = &env;
    cur_status = status;
    if (setjmp(env) != 0)
    {
        vmaxset(NULL);
        cur_env = NULL;
        cur_status = NULL;
        return 1;
    }
    fn(arg);
    vmaxset(NULL);
    cur_env = NULL;
    cur_status = NULL;
    return 0;
}

/* error(): store the message and unwind to rshim_run(); never returns.
 * Outside rshim_run there is nowhere to unwind to, so abort loudly. */
void Rf_error(const char *format, ...)
{
    va_list ap;
    if (cur_env == NULL)
    {
        va_start(ap, format);
        fputs("edgerpy: R error() outside a kernel call: ", stderr);
        vfprintf(stderr, format, ap);
        va_end(ap);
        fputc('\n', stderr);
        abort();
    }
    va_start(ap, format);
    vsnprintf(cur_status->error, RSHIM_MSGLEN, format, ap);
    va_end(ap);
    longjmp(*cur_env, 1);
}

/* warning(): keep the first RSHIM_MAXWARN messages for the entry point */
void Rf_warning(const char *format, ...)
{
    va_list ap;
    if (cur_status == NULL || cur_status->nwarn >= RSHIM_MAXWARN)
    {
        return;
    }
    va_start(ap, format);
    vsnprintf(cur_status->warn[cur_status->nwarn], RSHIM_MSGLEN, format, ap);
    va_end(ap);
    cur_status->nwarn++;
}

/* As built on glibc, where configure defines HAVE_WORKING_CALLOC: R then
 * skips its nelem == 0 early return and calloc(0, ...) yields a valid,
 * non-NULL pointer. */
void *R_chk_calloc(size_t nelem, size_t elsize)
{
    void *p;
    p = calloc(nelem, elsize);
    if (!p)
    {
        Rf_error("'R_Calloc' could not allocate memory (%llu of %llu bytes)",
                 (unsigned long long)nelem, (unsigned long long)elsize);
    }
    return p;
}

void R_chk_free(void *ptr)
{
    if (ptr)
    {
        free(ptr);
    }
}

/* R_alloc(): zero bytes give NULL, as in R */
char *R_alloc(size_t nelem, int eltsize)
{
    double dsize = (double)nelem * eltsize;
    if (dsize <= 0)
    {
        return NULL;
    }
    ralloc_block *b = NULL;
    if (dsize < (double)(SIZE_MAX - sizeof(ralloc_block)))
    {
        b = malloc(sizeof(ralloc_block) + nelem * (size_t)eltsize);
    }
    if (!b)
    {
        Rf_error("cannot allocate vector of size %0.1f Gb", dsize / 1024.0 / 1024.0 / 1024.0);
    }
    b->prev = ralloc_top;
    ralloc_top = b;
    return (char *)(b + 1);
}

void *vmaxget(void)
{
    return ralloc_top;
}

/* free every block allocated after the mark */
void vmaxset(const void *mark)
{
    while (ralloc_top && ralloc_top != mark)
    {
        ralloc_block *b = ralloc_top;
        ralloc_top = b->prev;
        free(b);
    }
}

/* Fortran-callable LAPACK error routine (R: src/main/print.c) */
void F77_SUB(xerbla)(const char *srname, int *info, const FC_LEN_T srname_len)
{
    char buf[21];
    size_t len = srname_len < 20 ? srname_len : 20;
    memcpy(buf, srname, len);
    buf[len] = '\0';
    Rf_error("BLAS/LAPACK routine '%6s' gave error code %d", buf, -(*info));
}

/* ---- SEXP accessors referenced only by object.c's SEXPtocmx1/2 ---- */

static void sexp_unavailable(const char *what)
{
    fprintf(stderr, "edgerpy: R object API '%s' called; SEXP objects do not exist in edgerpy\n", what);
    abort();
}

SEXP R_DimSymbol = NULL;

double *(REAL)(SEXP x)
{
    (void) x;
    sexp_unavailable("REAL");
    return NULL;
}

int *(INTEGER)(SEXP x)
{
    (void) x;
    sexp_unavailable("INTEGER");
    return NULL;
}

int (TYPEOF)(SEXP x)
{
    (void) x;
    sexp_unavailable("TYPEOF");
    return 0;
}

SEXP Rf_getAttrib(SEXP x, SEXP name)
{
    (void) x;
    (void) name;
    sexp_unavailable("Rf_getAttrib");
    return NULL;
}

SEXP Rf_install(const char *name)
{
    (void) name;
    sexp_unavailable("Rf_install");
    return NULL;
}

int Rf_asLogical(SEXP x)
{
    (void) x;
    sexp_unavailable("Rf_asLogical");
    return 0;
}

/* ---- console output (R_ext/Print.h): R's console is stdout, its error stream stderr ---- */

void Rprintf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);
}

void REprintf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}
