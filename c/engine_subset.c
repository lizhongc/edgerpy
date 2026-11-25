#define EDGERPY_R_INTERNALS
#include "rapi.h"
#include "rmath.h"
#define USE_BASE_R_SUPPORT  /* as R's src/include/Defn.h line 27 */
#include "linalg.h"

/* ---- R-4.6.1/src/main/engine.c lines 2453-2513 (verbatim) ---- */
void GEPretty(double *lo, double *up, int *ndiv)
{
/*	Set scale and ticks for linear scales.
 *
 *	Pre:	    x1 == lo < up == x2      ;  ndiv >= 1
 *	Post: x1 <= y1 := lo < up =: y2 <= x2;	ndiv >= 1
 */
    if(*ndiv <= 0)
	error(_("invalid axis extents [GEPretty(.,.,n=%d)"), *ndiv);
    if(!R_FINITE(*lo) || !R_FINITE(*up)) // also catch NA etc
	error(_("non-finite axis extents [GEPretty(%g,%g, n=%d)]"), *lo, *up, *ndiv);

    // For *finite* boundaries, now allow (*up - *lo) = +/- inf  as R_pretty() now does
    double ns = *lo, nu = *up;
#ifdef DEBUG_axis
    double x1 = ns, x2 = nu;
#endif
    double unit, high_u_fact[3] = { .8, 1.7, 1.125 };
                             // =   (h, h5 , f_min) = (high.u.bias, u5.bias, f_min)
    // -> ../appl/pretty.c
    unit = R_pretty(&ns, &nu, ndiv, /* min_n = */ 1,
		    /* shrink_sml = */ 0.25,
		    high_u_fact,
		    2, /* do eps_correction in any case */
		    0 /* return (ns,nu) in  (lo,up) */);
#ifdef DEBUG_axis
    REprintf(" R_pretty() -> new (ns=%g, nu=%g, ndiv=%d)\n", ns, nu, *ndiv);
#endif
    // The following is ugly since it kind of happens already in R_pretty(..):
#define rounding_eps 1e-10 /* <- compatible to seq*(); was 1e-7 till 2017-08-14 */
    if(nu >= ns + 1) {
	int mod = 0;
	if(               ns * unit < *lo - rounding_eps*unit) { ns++; mod++; }
	if(nu > ns + 1 && nu * unit > *up + rounding_eps*unit) { nu--; mod++; }
	if(mod) *ndiv = (int)(nu - ns);
#ifdef DEBUG_axis
	if(mod) REprintf(" GEPretty(): _mod_ify -> new (ns=%g, nu=%g, ndiv=%d)\n", ns, nu, *ndiv);
#endif
    }
    *lo = ns * unit;
    *up = nu * unit;
#ifdef non_working_ALTERNATIVE
    if(ns * unit > *lo)
	*lo = ns * unit;
    if(nu * unit < *up)
	*up = nu * unit;
    if(nu - ns >= 1)
	*ndiv = nu - ns;
#endif

#ifdef DEBUG_axis
    REprintf(" .. GEPr final (lo=%g, up=%g, ndiv=%d)\n", *lo, *up, *ndiv);
#endif

#ifdef DEBUG_axis
    if(*lo < x1)
	warning(_(" new *lo = %g < %g = x1"), *lo, x1);
    if(*up > x2)
	warning(_(" new *up = %g > %g = x2"), *up, x2);
#endif
}

