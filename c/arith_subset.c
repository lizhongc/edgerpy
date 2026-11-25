#define EDGERPY_R_INTERNALS
#include "rapi.h"
#include "rmath.h"

/* ---- R-4.6.1/src/main/arithmetic.c lines 65-274 (verbatim) ---- */
typedef union
{
    double value;
    unsigned int word[2];
} ieee_double;

/* gcc had problems with static const on AIX and Solaris
   Solaris was for gcc 3.1 and 3.2 under -O2 32-bit on 64-bit kernel */
#ifdef _AIX
#define CONST
#elif defined(sparc) && defined (__GNUC__) && __GNUC__ == 3
#define CONST
#else
#define CONST const
#endif

#ifdef WORDS_BIGENDIAN
static CONST int hw = 0;
static CONST int lw = 1;
#else  /* !WORDS_BIGENDIAN */
static CONST int hw = 1;
static CONST int lw = 0;
#endif /* WORDS_BIGENDIAN */


static double R_ValueOfNA(void)
{
    /* The gcc (3.2.1?) shipping with Red Hat Linux 9 gets this wrong
     * without the volatile declaration. Thanks to Marc Schwartz. */
    volatile ieee_double x;
    x.word[hw] = 0x7ff00000;
    x.word[lw] = 1954;
    return x.value;
}

/* is a value known to be a NaN also an R NA? */
attribute_hidden int R_NaN_is_R_NA(double x)
{
    ieee_double y;
    y.value = x;
    return (y.word[lw] == 1954);
}

int R_IsNA(double x)
{
    return isnan(x) && R_NaN_is_R_NA(x);
}

int R_IsNaN(double x)
{
    return isnan(x) && ! R_NaN_is_R_NA(x);
}

/* ISNAN uses isnan, which is undefined by C++ headers
   This workaround is called only when ISNAN() is used
   in a user code in a file with __cplusplus defined */

int R_isnancpp(double x)
{
   return (isnan(x)!=0);
}


/* Mainly for use in packages */
int R_finite(double x)
{
#ifdef HAVE_WORKING_ISFINITE
    return isfinite(x);
#else
    return (!isnan(x) & (x != R_PosInf) & (x != R_NegInf));
#endif
}


/* Arithmetic Initialization */

attribute_hidden void InitArithmetic(void)
{
    R_NaInt = INT_MIN;
    R_NaReal = R_ValueOfNA();
// we assume C99, so
#ifndef OLD
    R_NaN = NAN;
    R_PosInf = INFINITY;
    R_NegInf = -INFINITY;
#else
    R_NaN = 0.0/R_Zero_Hack;
    R_PosInf = 1.0/R_Zero_Hack;
    R_NegInf = -1.0/R_Zero_Hack;
#endif
}


#if HAVE_LONG_DOUBLE && (SIZEOF_LONG_DOUBLE > SIZEOF_DOUBLE)
/*
# ifdef __powerpc__
 // PowerPC 64 (when gcc has -mlong-double-128) fails constant folding with LDOUBLE
 // Debian Bug#946836 shows it is needed also for 32-bit ppc, not just __PPC64__
 // NB: 1 / LDBL_EPSILON has been seen to overflow on 'ppc64el ...
 ==> use  eps instead of  1 / eps  (and one multiplication more)
*/
# define c_eps LDBL_EPSILON
#else
# define c_eps DBL_EPSILON
#endif

/* Keep myfmod() and myfloor() in step */
static double myfmod(double x1, double x2)
{
    if (x2 == 0.0) return R_NaN;
    if(fabs(x2) * c_eps > 1 && R_FINITE(x1) && fabs(x1) <= fabs(x2)) {
	return
	    (fabs(x1) == fabs(x2)) ? 0 :
	    ((x1 < 0 && x2 > 0) ||
	     (x2 < 0 && x1 > 0))
	     ? x1+x2  // differing signs
	     : x1   ; // "same" signs (incl. 0)
    }
    double q = x1 / x2;
    if(R_FINITE(q) && (fabs(q) * c_eps > 1))
	warning(_("probable complete loss of accuracy in modulus"));
    LDOUBLE tmp = (LDOUBLE)x1 - floor(q) * (LDOUBLE)x2;
    return (double) (tmp - floorl(tmp/x2) * x2);
}

static double myfloor(double x1, double x2)
{
    double q = x1 / x2;
    if (x2 == 0.0 || fabs(q) * c_eps > 1 || !R_FINITE(q))
	return q;
    if(fabs(q) < 1)
	return (q < 0) ? -1
	    : ((x1 < 0 && x2 > 0) ||
	       (x1 > 0 && x2 < 0) // differing signs
	       ? -1 : 0);
    LDOUBLE tmp = (LDOUBLE)x1 - floor(q) * (LDOUBLE)x2;
    return (double) (floor(q) + floorl(tmp/x2));
}

double R_pow(double x, double y) /* = x ^ y */
{
    /* squaring is the most common of the specially handled cases so
       check for it first. */
    if(y == 2.)
	return x * x;
    if(x == 1. || y == 0.)
	return(1.);
    if(x == 0.) {
	if(y > 0.) return(0.);
	else if(y < 0) return(R_PosInf);
	else return(y); /* NA or NaN, we assert */
    }
    if (x >= -11. && x <= 11.) {
	if(y == 4.)
	    return x * x * x * x;
	if(y == 3.)
	    return x * x * x;
    }
    if (R_FINITE(x) && R_FINITE(y)) {
	/* There was a special case for y == 0.5 here, but
	   gcc 4.3.0 -g -O2 mis-compiled it.  Showed up with
	   100^0.5 as 3.162278, example(pbirthday) failed. */
#ifdef USE_POWL_IN_R_POW
	// this is used only on 64-bit Windows (so has powl).
	return powl(x, y);
#else
	return pow(x, y);
#endif
    }
    if (ISNAN(x) || ISNAN(y))
	return(x + y);
    if(!R_FINITE(x)) {
	if(x > 0)		/* Inf ^ y */
	    return (y < 0.)? 0. : R_PosInf;
	else {			/* (-Inf) ^ y */
	    if(R_FINITE(y) && y == floor(y)) /* (-Inf) ^ n */
		return (y < 0.) ? 0. : (myfmod(y, 2.) != 0 ? x  : -x);
	}
    }
    if(!R_FINITE(y)) {
	if(x >= 0) {
	    if(y > 0)		/* y == +Inf */
		return (x >= 1) ? R_PosInf : 0.;
	    else		/* y == -Inf */
		return (x < 1) ? R_PosInf : 0.;
	}
    }
    return R_NaN; // all other cases: (-Inf)^{+-Inf, non-int}; (neg)^{+-Inf}
}

double R_pow_di(double x, int n)
{
    double xn = 1.;

    if (ISNAN(x)) return x;
    if (n == NA_INTEGER) return NA_REAL;

    if (n != 0) {
	if (!R_FINITE(x)) return R_POW(x, (double)n);

	bool is_neg = (n < 0);
	if(is_neg) n = -n;
	for(;;) {
	    if(n & 01) xn *= x;
	    if(n >>= 1) x *= x; else break;
	}
	if(is_neg) xn = 1. / xn;
    }
    return xn;
}

