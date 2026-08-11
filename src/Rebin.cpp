#include "specfit/Rebin.hpp"
#include <algorithm>
#include <numeric>

namespace specfit {

/* -------------------------------------------------------------- *
 *  helper: linear interpolation of y(x) for a *monotonic* array  *
 * -------------------------------------------------------------- */
static double interp(const Vector& x, const Vector& y, double xi)
{
    if (xi <= x[0])               return y[0];
    if (xi >= x[x.size() - 1])    return y[y.size() - 1];

    const auto it = std::lower_bound(x.data(), x.data() + x.size(), xi);
    const int  hi = static_cast<int>(it - x.data());
    const int  lo = hi - 1;
    const double w = (xi - x[lo]) / (x[hi] - x[lo]);
    return y[lo] * (1.0 - w) + y[hi] * w;
}

/* -------------------------------------------------------------- *
 *  build pixel edges from centres:                               *
 *      e_0 , c_0 , e_1 , c_1 , …                                 *
 * -------------------------------------------------------------- */
static Vector make_edges(const Vector& centres)
{
    const int N = centres.size();
    Vector edges(N + 1);

    // Use symmetric spacing for first and last edges
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0]);
    for (int i = 1; i < N; ++i)
        edges[i] = 0.5 * (centres[i - 1] + centres[i]);
    edges[N] = centres[N - 1] + 0.5 * (centres[N - 1] - centres[N - 2]);
    
    // Ensure edges are monotonic and don't create negative widths
    for (int i = 1; i <= N; ++i) {
        if (edges[i] <= edges[i-1]) {
            edges[i] = edges[i-1] + 1e-10;  // small epsilon
        }
    }

    return edges;
}

/* -------------------------------------------------------------- *
 *  integrate f(λ) once → F(λ)  and interpolate that integral      *
 * -------------------------------------------------------------- */
static Vector cumulative_trapz(const Vector& x, const Vector& y)
{
    const int N = x.size();
    Vector F(N);
    F[0] = 0.0;
    for (int i = 1; i < N; ++i)
        F[i] = F[i - 1] +
               0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    return F;
}

/* ==============================================================
 *  public interface                                             
 * =============================================================*/
Vector trapezoidal_rebin(const Vector& lam_in,
                         const Vector& flux_in,
                         const Vector& lam_out)
{
    /* ---- derive pixel edges ---- */
    const Vector out_edges = make_edges(lam_out);

    /* ---- cumulative integral of input spectrum ---- */
    const Vector F = cumulative_trapz(lam_in, flux_in);

    const double* const xin  = lam_in.data();
    const double* const fin  = flux_in.data();
    const Eigen::Index  nin  = lam_in.size();
    const double        xlo  = xin[0];
    const double        xhi  = xin[nin - 1];

    /*  helper that returns ∫ f dλ  from λ_0  to  xi
     *
     *  Two things used to make this the hottest loop in the fit.  It called
     *  interp() for f(xi), which repeated the very binary search that had
     *  just been done here -- the bracket is already known, so the linear
     *  interpolation is written out instead.  And it binary-searched a
     *  ~10 000-point model grid per edge, although the edges arrive in
     *  ascending order: `cursor` walks forward and lands on exactly the index
     *  std::lower_bound would return.  Both give the same brackets and the
     *  same arithmetic, so every value is unchanged.                        */
    Eigen::Index cursor = 0;
    auto integral_at = [&](double xi) -> double {
        if (xi <= xlo) return 0.0;
        if (xi >= xhi) return F[nin - 1];

        while (cursor < nin && xin[cursor] < xi) ++cursor;   // == lower_bound
        const Eigen::Index hi = cursor;
        const Eigen::Index lo = hi - 1;

        /* partial area from lam_in[lo]  to  xi                    */
        const double f_lo = fin[lo];
        const double w    = (xi - xin[lo]) / (xin[hi] - xin[lo]);
        const double f_hi = fin[lo] * (1.0 - w) + fin[hi] * w;
        const double dx   = xi - xin[lo];
        const double area = 0.5 * (f_lo + f_hi) * dx;

        return F[lo] + area;
    };

    /* ---- rebin by evaluating the integral at the edges ---- */
    Vector out(lam_out.size());
    for (int i = 0; i < lam_out.size(); ++i) {
        // Clamp edges to input data range
        const double lo = std::max(out_edges[i], xlo);
        const double hi = std::min(out_edges[i + 1], xhi);

        if (hi <= lo) {
            // Output bin is completely outside input range
            // Use nearest neighbor extrapolation
            if (out_edges[i + 1] < xlo) {
                out[i] = fin[0];
            } else if (out_edges[i] > xhi) {
                out[i] = fin[nin - 1];
            } else {
                out[i] = interp(lam_in, flux_in, lam_out[i]);
            }
        } else {
            /*  lo before hi: the cursor above only ever moves forward, and
             *  the arguments are non-decreasing in exactly this order.  (The
             *  original `integral_at(hi) - integral_at(lo)` left the order of
             *  the two calls up to the compiler.)                           */
            const double I_lo = integral_at(lo);
            const double I_hi = integral_at(hi);
            const double w    = hi - lo;
            out[i] = (I_hi - I_lo) / w;
        }
    }
    return out;
}

} // namespace specfit