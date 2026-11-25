#include "edgerpy.h"
#include <math.h>

/* ============================================================================
 * Two-part empirical sample weights from the adjusted unit deviances of a quasi-likelihood fit.
 *
 * Each sample j has a technical weight t_j on the Poisson part of the variance and a factor c_j on the NB part:
 *   var(y_gj) = s2_g/s0 * (s0*mu_gj/t_j + c_j*phi*mu_gj^2),
 * which is the fit glmQLFit(technical.weights=t, dispersion=c*phi). If t_j changes to t_j/x_j and c_j to c_j*z_j, the
 * variance of observation (g,j) changes by the factor r_gj = pi_gj*x_j + (1-pi_gj)*z_j, where pi_gj is its Poisson
 * share. The adjusted unit deviance is modelled as D_gj ~ s2_g*r_gj*chi2(nu_gj), with nu_gj the unit df, and one Fisher
 * scoring (REML) step for (log x_j, log z_j) at x = z = 1 solves the 2x2 system
 *   sum_g nu_gj a_gj a_gj' (log x_j, log z_j)' = sum_g (D_gj/s2_g - nu_gj) a_gj,   a_gj = (pi_gj, 1-pi_gj)',
 * where s2_g = sum_j D_gj / sum_j nu_gj is the gene-wise QL dispersion. That is the working information of one sample; the
 * samples are updated jointly (see the step in sample_weights_nb), with the gene scale profiled out of the information.
 * A prior on log t as in limma::arrayWeights adds
 * prior.n pseudo-genes with pi = 1 and one df each that agree with t = 1: prior.n to the (1,1) element and prior.n*(t_j-1)
 * to the first score. The same prior on log c adds prior.n pseudo-genes with pi = 0 that agree with c = 1: prior.n to the
 * (2,2) element and prior.n*(1/c_j-1) to the second score. pi_gj = 1/(1 + phi*c_j*t_j*m_gj/s0) is evaluated at the leave-one-out mean m_gj of the
 * library-scaled counts of the other samples, so that a spike does not move its own gene to a low-count share.
 * The update is the joint constrained step s = A^{-1}(g - E lambda) over all samples, with A the block-diagonal working
 * information with the priors minus the information of the gene scale, and lambda chosen so that log t and log c keep
 * mean 0 (geometric mean 1). Then t <- t*exp(-log x) and c <- c*exp(log z), and the model is refitted as in
 * glmQLFit(legacy=FALSE).
 * ========================================================================== */

/* Leave-one-out mean of the library-scaled counts, on the scale of each sample.
 * For gene g with offsets o_gj and mean offset om_g, the scaled counts are ys_gj = y_gj*exp(om_g - o_gj) and
 *   m_gj = (sum_k ys_gk - ys_gj)/(nlib-1) * exp(o_gj - om_g).
 * Equivalent R: ys <- y*exp(rowMeans(o)-o); m <- (rowSums(ys)-ys)/(n-1)*exp(o-rowMeans(o)).
 *   ymx, omx : counts and offsets
 *   mmat     : output, genes-by-samples, column-major (mmat[g+j*ntag])
 * OpenMP over genes; each thread has its own two row buffers. */
static void loo_mean(cmx *ymx, cmx *omx, int nthreads, double *mmat)
{
    int ntag = ymx->nrow;
    int nlib = ymx->ncol;
    int nth = clamp_threads(nthreads);
    double *buf = R_Calloc((size_t) 2*nth*nlib, double);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nth) schedule(static)
    #endif
    for(int g=0; g<ntag; ++g)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double *yrow = buf + (size_t) 2*tid*nlib;
        double *orow = yrow + nlib;
        get_row(ymx, g, yrow);
        get_row(omx, g, orow);

        double om = 0;
        for(int j=0; j<nlib; ++j)
        {
            om += orow[j];
        }
        om /= nlib;

        double sum = 0;
        for(int j=0; j<nlib; ++j)
        {
            yrow[j] *= exp(om - orow[j]);
            sum += yrow[j];
        }

        double *mptr = mmat + g;
        for(int j=0; j<nlib; ++j, mptr+=ntag)
        {
            (*mptr) = (sum - yrow[j]) / (nlib - 1) * exp(orow[j] - om);
        }
    }

    R_Free(buf);
}

/* Gene-wise REML residuals and Poisson shares for one update.
 * On input dvmat and dfmat hold the adjusted unit deviances and unit df from compute_adjust_mat. For each gene,
 *   s2_g = sum_j w_gj*dv_gj / sum_j df_gj,
 * and on output dvmat holds e_gj = w_gj*dv_gj/s2_g - df_gj and pimat holds pi_gj = 1/(1 + pc_j*t_j*m_gj/s0).
 * Genes with no positive df or deviance, or a non-finite deviance (mu = 0 when a group is all zero), get e = df = 0
 * so that they contribute nothing to the per-sample sums.
 *   wmx   : GLM prior weights
 *   pc, t : NB dispersion phi*c_j and technical weights t_j, one per sample
 *   mmat  : leave-one-out means (loo_mean)
 *   s0    : average quasi-dispersion of the current fit
 * OpenMP over genes; writes are to the rows of gene g only. */
static void reml_resid(cmx *wmx, const double *pc, const double *t, const double *mmat, double s0, int ntag, int nlib, int nthreads, double *dvmat, double *dfmat, double *pimat)
{
    int nth = clamp_threads(nthreads);
    double *buf = R_Calloc((size_t) nth*nlib, double);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nth) schedule(static)
    #endif
    for(int g=0; g<ntag; ++g)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double *wrow = buf + (size_t) tid*nlib;
        get_row(wmx, g, wrow);

        double *dvptr = dvmat + g;
        double *dfptr = dfmat + g;
        double sdv = 0;
        double sdf = 0;
        for(int j=0; j<nlib; ++j, dvptr+=ntag, dfptr+=ntag)
        {
            sdv += wrow[j] * (*dvptr);
            sdf += (*dfptr);
        }
        int ok = R_FINITE(sdv) && sdv > 0 && sdf > 0;
        double s2 = ok ? sdv/sdf : 1.0;

        dvptr = dvmat + g;
        dfptr = dfmat + g;
        double *piptr = pimat + g;
        const double *mptr = mmat + g;
        for(int j=0; j<nlib; ++j, dvptr+=ntag, dfptr+=ntag, piptr+=ntag, mptr+=ntag)
        {
            if(ok)
            {
                (*dvptr) = wrow[j]*(*dvptr)/s2 - (*dfptr);
            }
            else
            {
                (*dvptr) = 0;
                (*dfptr) = 0;
            }
            (*piptr) = 1 / (1 + pc[j]*t[j]*(*mptr)/s0);
        }
    }

    R_Free(buf);
}

/* One fit as in glmQLFit(legacy=FALSE) with technical weights t and NB dispersions pc = phi*c, one per sample:
 * fit t*y with offset o+log(t) and dispersion pc -> average quasi-dispersion s0 -> refit with pc/s0. At t = c = 1 it is the
 * fit of glmQLFit without sample weights (1*y = y, o + log 1 = o).
 *   ymx, omx, wmx, gmx : counts, compressed offsets, compressed GLM prior weights, design
 *   tmx, pcmx          : by-column cmx views over t and pc
 *   pc, t              : NB dispersions phi*c_j and technical weights t_j (nlib)
 *   pcs                : scratch (nlib), holds pc/s0 on return
 *   ave                : AveLogCPM (ntag), used by update_prior
 *   coef, dev, iter, failed : scratch outputs of fit_glm_mat
 *   mu                 : output, fitted values of the refit on the original scale (ntag x nlib, column-major)
 * Returns s0. The t and log(t) vectors of the scale_technical views are R_alloc'd and released here with vmaxset. */
static double sw_refit(cmx *ymx, cmx *omx, cmx *wmx, cmx *gmx, cmx *tmx, cmx *pcmx, const double *pc, const double *t, double *pcs, double *ave, double *coef, double *mu, double *dev, int *iter, int *failed, int nthreads)
{
    int ntag = ymx->nrow;
    int nlib = ymx->ncol;
    const void *vmax = vmaxget();
    cmx ys;
    cmx os;
    scale_technical(ymx, omx, tmx, &ys, &os);
    int method = 0;
    fit_glm_mat(&ys, &os, pcmx, wmx, gmx, 250, 1e-6, NULL, coef, mu, dev, iter, failed, &method, nthreads);
    double *muptr = mu;
    for(int j=0; j<nlib; ++j)
    {
        for(int g=0; g<ntag; ++g, ++muptr)
        {
            (*muptr) /= t[j];
        }
    }
    cmx mumx = make_cmx(mu, NULL, ntag, nlib, 0, 0);
    double s0 = update_prior(ymx, &mumx, gmx, pcmx, wmx, tmx, ave, nthreads);

    for(int j=0; j<nlib; ++j)
    {
        pcs[j] = pc[j]/s0;
    }
    cmx pcsmx = make_cmx(pcs, NULL, ntag, nlib, 2, 0);
    method = 0;
    fit_glm_mat(&ys, &os, &pcsmx, wmx, gmx, 250, 1e-6, NULL, coef, mu, dev, iter, failed, &method, nthreads);
    muptr = mu;
    for(int j=0; j<nlib; ++j)
    {
        for(int g=0; g<ntag; ++g, ++muptr)
        {
            (*muptr) /= t[j];
        }
    }
    vmaxset(vmax);
    return s0;
}

/* Two-part sample weights by REML on the adjusted unit deviances. The starting fit is that of glmQLFit(legacy=FALSE)
 * without sample weights: the common Cox-Reid dispersion phi on the most abundant genes, then sw_refit at t = c = 1.
 * Then up to niter updates with a refit after each update except the last; stops early when the largest |change| of
 * log t and log c in an update is below tol.
 *   ymx      : counts
 *   gmx      : design (full rank)
 *   omx      : compressed offsets
 *   wmx      : compressed GLM prior weights
 *   ave      : AveLogCPM (ntag), used to choose the genes for phi and by update_prior
 *   priorn   : prior.n, number of pseudo-genes of the prior on log t and of the prior on log c
 *   t        : output, final technical weights (nlib, geometric mean 1)
 *   c        : output, final NB dispersion factors (nlib, geometric mean 1)
 * Returns the number of updates done, or -1 if the step matrix is not positive definite. Scratch is R_Calloc'd and freed here. */
int sample_weights_nb(cmx *ymx, cmx *gmx, cmx *omx, cmx *wmx, double *ave, int niter, double tol, double priorn, int nthreads, double *t, double *c)
{
    int ntag = ymx->nrow;
    int nlib = ymx->ncol;
    int ncoef = gmx->ncol;
    int nth = clamp_threads(nthreads);
    size_t nn = (size_t) ntag*nlib;
    double *dfmat = R_Calloc(nn, double);
    double *dvmat = R_Calloc(nn, double);
    double *pimat = R_Calloc(nn, double);
    double *mmat  = R_Calloc(nn, double);
    double *mu    = R_Calloc(nn, double);
    double *coef  = R_Calloc((size_t) ntag*ncoef, double);
    double *dev   = R_Calloc(ntag, double);
    int *iter     = R_Calloc(ntag, int);
    int *failed   = R_Calloc(ntag, int);
    double *pc    = R_Calloc(nlib, double);
    double *pcs   = R_Calloc(nlib, double);
    double *ssum  = R_Calloc((size_t) 5*nlib, double);
    int n2 = 2*nlib;
    double *amat  = R_Calloc((size_t) n2*n2, double);
    double *rhs   = R_Calloc((size_t) 3*n2, double);
    double *invn  = R_Calloc(ntag, double);

    /* leave-one-out means */
    loo_mean(ymx, omx, nthreads, mmat);

    /* common NB dispersion: Cox-Reid on the most abundant genes, with the bounds 0 and sqrt(2) (interval c(0,4) on the scale
     * disp^0.25), tolerance 1e-5 and the Cox-Reid adjustment, as glmQLFit(legacy=FALSE) computes it */
    double phi = coxreid_disp_top_opt(ymx, omx, wmx, gmx, ave, 0, sqrt(2.0), 1e-5, 1, nthreads);

    /* per-sample vectors repeated by row: technical weights t and dispersion phi*c, starting at t = c = 1 */
    for(int j=0; j<nlib; ++j)
    {
        t[j] = 1.0;
        c[j] = 1.0;
        pc[j] = phi;
    }
    cmx tmx   = make_cmx(t, NULL, ntag, nlib, 2, 0);
    cmx pcmx  = make_cmx(pc, NULL, ntag, nlib, 2, 0);
    cmx mumx  = make_cmx(mu, NULL, ntag, nlib, 0, 0);

    /* starting fit, as glmQLFit without sample weights */
    double s0 = sw_refit(ymx, omx, wmx, gmx, &tmx, &pcmx, pc, t, pcs, ave, coef, mu, dev, iter, failed, nthreads);

    int it = 0;
    while(it < niter)
    {
        /* adjusted unit deviances and df at the current t, c, mu, s0; lvmat output goes to pimat, overwritten below */
        compute_adjust_mat(ymx, &mumx, gmx, &pcmx, s0, wmx, &tmx, dfmat, dvmat, pimat, nthreads);
        reml_resid(wmx, pc, t, mmat, s0, ntag, nlib, nthreads, dvmat, dfmat, pimat);

        /* per-sample sums m11, m12, m22, u1, u2 in a fixed gene order, so the result does not depend on nthreads */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nth) schedule(static)
        #endif
        for(int j=0; j<nlib; ++j)
        {
            const double *eptr = dvmat + (size_t) j*ntag;
            const double *fptr = dfmat + (size_t) j*ntag;
            const double *pptr = pimat + (size_t) j*ntag;
            double m11 = 0;
            double m12 = 0;
            double m22 = 0;
            double u1 = 0;
            double u2 = 0;
            for(int g=0; g<ntag; ++g, ++eptr, ++fptr, ++pptr)
            {
                double p = (*pptr);
                double q = 1 - p;
                m11 += (*fptr)*p*p;
                m12 += (*fptr)*p*q;
                m22 += (*fptr)*q*q;
                u1 += (*eptr)*p;
                u2 += (*eptr)*q;
            }
            double *sptr = ssum + (size_t) 5*j;
            sptr[0] = m11;
            sptr[1] = m12;
            sptr[2] = m22;
            sptr[3] = u1;
            sptr[4] = u2;
        }

        /* joint constrained scoring step. The parameters are stacked as (log t_1..log t_n, log b_1..log b_n), b = 1/c. The step
         * matrix is A = blockdiag(M_j + priorn*I) - S, with M_j = (m11 m12; m12 m22) the working information of sample j and
         * S = sum_g v_g v_g'/N_g the information of the gene scale s2_g, profiled out because the scores use the plug-in s2_g:
         * v_g = (nu_g1*pi_g1, .., nu_gn*pi_gn, nu_g1*(1-pi_g1), .., nu_gn*(1-pi_gn)), N_g = sum_j nu_gj. With the scores
         * g = (u1 + priorn*(t-1), u2 + priorn*(b-1)) and E = the two indicator columns of the log t and log b blocks, the step is
         * s = A^{-1}(g - E lambda), with lambda chosen so that the new log t and log b keep sum 0:
         *   (E'A^{-1}E) lambda = E'A^{-1}g - (sum log t, sum log b).
         * Its fixed points are the Lagrange points g_j = lambda for every sample, whatever A is.
         * Equivalent R (proto.R of 202610312028-exact-diag-constrained, info = "workingS"):
         *   V <- cbind(nu*p, nu*q)/sqrt(rowSums(nu)); A <- blockdiag(M) - crossprod(V) + priorn*diag(2n)
         *   lam <- solve(t(E) %*% solve(A, E), t(E) %*% solve(A, g) - c(sum(log t), sum(log b))); s <- solve(A, g - E %*% lam) */

        /* 1/N_g of every gene; genes left out by reml_resid have zero df and are skipped */
        for(int g=0; g<ntag; ++g)
        {
            double sdf = 0;
            const double *fptr = dfmat + g;
            for(int j=0; j<nlib; ++j, fptr+=ntag)
            {
                sdf += (*fptr);
            }
            invn[g] = sdf > 0 ? 1/sdf : 0;
        }

        /* upper triangle of S, element (a,b) = sum_g v_ga v_gb / N_g, each summed over genes in a fixed order */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nth) schedule(dynamic)
        #endif
        for(int bb=0; bb<n2; ++bb)
        {
            int jb = bb < nlib ? bb : bb - nlib;
            const double *fb = dfmat + (size_t) jb*ntag;
            const double *pb = pimat + (size_t) jb*ntag;
            for(int aa=0; aa<=bb; ++aa)
            {
                int ja = aa < nlib ? aa : aa - nlib;
                const double *fa = dfmat + (size_t) ja*ntag;
                const double *pa = pimat + (size_t) ja*ntag;
                double sab = 0;
                for(int g=0; g<ntag; ++g)
                {
                    double va = fa[g]*(aa < nlib ? pa[g] : 1 - pa[g]);
                    double vb = fb[g]*(bb < nlib ? pb[g] : 1 - pb[g]);
                    sab += va*vb*invn[g];
                }
                amat[aa + (size_t) bb*n2] = -sab;
            }
        }

        /* add the block-diagonal working information with the priors, and set up the right-hand sides g, E_t, E_b */
        double slt = 0;
        double slb = 0;
        for(int j=0; j<nlib; ++j)
        {
            const double *sptr = ssum + (size_t) 5*j;
            int jt = j;
            int jb = nlib + j;
            amat[jt + (size_t) jt*n2] += sptr[0] + priorn;
            amat[jt + (size_t) jb*n2] += sptr[1];
            amat[jb + (size_t) jb*n2] += sptr[2] + priorn;
            rhs[jt] = sptr[3] + priorn*(t[j] - 1);
            rhs[jb] = sptr[4] + priorn*(1/c[j] - 1);
            slt += log(t[j]);
            slb -= log(c[j]);
        }
        for(int k=0; k<n2; ++k)
        {
            rhs[n2 + k] = k < nlib ? 1 : 0;
            rhs[2*n2 + k] = k < nlib ? 0 : 1;
        }

        /* dpotrf: LAPACK Cholesky factorisation A = U'U of the upper triangle (UPLO = 'U') of the 2n x 2n matrix amat, which is
         * overwritten by U; INFO > 0 means A is not positive definite (it is when priorn > 0, since S is a Schur complement
         * of a positive semidefinite matrix and the block-diagonal part dominates it). dpotrs: the matching solves of
         * A X = B for the three right-hand sides (g, E_t, E_b) in rhs, overwritten by A^{-1}g, A^{-1}E_t, A^{-1}E_b.
         * FCONE passes the hidden Fortran length of UPLO. Equivalent R: X <- solve(A, cbind(g, E)). */
        char uplo = 'U';
        int info = 0;
        int nrhs = 3;
        F77_CALL(dpotrf)(&uplo, &n2, amat, &n2, &info FCONE);
        if(info != 0)
        {
            it = -1;
            break;
        }
        F77_CALL(dpotrs)(&uplo, &n2, &nrhs, amat, &n2, rhs, &n2, &info FCONE);
        if(info != 0)
        {
            it = -1;
            break;
        }

        /* lambda from the 2 x 2 system (E'A^{-1}E) lambda = E'A^{-1}g - (sum log t, sum log b) */
        double k11 = 0;
        double k12 = 0;
        double k22 = 0;
        double r1 = 0;
        double r2 = 0;
        for(int j=0; j<nlib; ++j)
        {
            k11 += rhs[n2 + j];
            k12 += rhs[2*n2 + j];
            k22 += rhs[2*n2 + nlib + j];
            r1 += rhs[j];
            r2 += rhs[nlib + j];
        }
        r1 -= slt;
        r2 -= slb;
        double dk = k11*k22 - k12*k12;
        double lam1 = (k22*r1 - k12*r2) / dk;
        double lam2 = (k11*r2 - k12*r1) / dk;

        /* step s = A^{-1}g - A^{-1}E lambda; log t <- log t - s_t and log c <- log c + s_b, then centring, which only removes
         * rounding because the constrained step keeps both sums at 0 */
        double mt = 0;
        double mc = 0;
        double *sptr = ssum;
        for(int j=0; j<nlib; ++j, sptr+=5)
        {
            double lx = rhs[j] - rhs[n2 + j]*lam1 - rhs[2*n2 + j]*lam2;
            double lz = rhs[nlib + j] - rhs[n2 + nlib + j]*lam1 - rhs[2*n2 + nlib + j]*lam2;
            sptr[0] = log(t[j]) - lx;
            sptr[1] = log(c[j]) + lz;
            mt += sptr[0];
            mc += sptr[1];
        }
        mt /= nlib;
        mc /= nlib;
        double dl = 0;
        sptr = ssum;
        for(int j=0; j<nlib; ++j, sptr+=5)
        {
            double ltn = sptr[0] - mt;
            double lcn = sptr[1] - mc;
            dl = fmax2(dl, fabs(ltn - log(t[j])));
            dl = fmax2(dl, fabs(lcn - log(c[j])));
            t[j] = exp(ltn);
            c[j] = exp(lcn);
            pc[j] = phi*c[j];
        }
        ++it;
        if(it == niter || dl < tol)
        {
            break;
        }

        /* refit as in glmQLFit(legacy=FALSE) with technical weights t and dispersion phi*c */
        s0 = sw_refit(ymx, omx, wmx, gmx, &tmx, &pcmx, pc, t, pcs, ave, coef, mu, dev, iter, failed, nthreads);
    }

    R_Free(dfmat);
    R_Free(dvmat);
    R_Free(pimat);
    R_Free(mmat);
    R_Free(mu);
    R_Free(coef);
    R_Free(dev);
    R_Free(iter);
    R_Free(failed);
    R_Free(pc);
    R_Free(pcs);
    R_Free(ssum);
    R_Free(amat);
    R_Free(rhs);
    R_Free(invn);

    return it;
}
