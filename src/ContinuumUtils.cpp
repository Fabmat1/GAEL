#include "specfit/ContinuumUtils.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>       // std::abs

namespace specfit {


/* ---------------------------------------------------------------------- */
Vector anchors_from_intervals(
        const std::vector<std::tuple<double,double,double>>& intervals,
        const Spectrum& spectrum)
{
    /* ---------- spectrum sanity checks -------------------------------- */
    if (spectrum.lambda.size() == 0)
        throw std::runtime_error("anchors_from_intervals(): spectrum has size 0");

    if (!spectrum.ignoreflag.empty() &&
        static_cast<Eigen::Index>(spectrum.ignoreflag.size()) != spectrum.lambda.size())
        throw std::runtime_error(
            "anchors_from_intervals(): ignoreflag vector has different length "
            "than spectrum.lambda");

    if (!spectrum.lambda.allFinite())
        throw std::runtime_error("anchors_from_intervals(): spectrum.lambda contains NaN/Inf");

    /* ---------- anchor x list, exactly as ISIS builds it --------------- *
     *  spectroscopy_automated.sl:
     *      l[id] = union(bin_lo[0]-1, bin_lo[-1]+1, cspline_anchorpoints);
     *      l[id] = l[id][where(bin_lo[0]-2 <= l[id] <= bin_lo[-1]+2)];
     *
     *  So the two boundary anchors sit 1 Angstrom *outside* the data on each
     *  side, and the user's anchors are clipped to a 2 Angstrom collar around
     *  it.  GAEL used to put its boundary anchors ten pixels *inside* the
     *  data instead, which leaves the spline's end segments determined by the
     *  first interior anchor alone and makes the fitted continuum at the edges
     *  depend on the pixel sampling.
     *
     *  The bounds come from the whole data array, not from the noticed pixels:
     *  ISIS's bin_lo covers every bin of the dataset regardless of what is
     *  ignored, and after the waveCut trim the data array already *is* the fit
     *  window.
     *
     *  bin_lo, not the bin centre.  GAEL stores pixel centres; ISIS converts
     *  them to lower bin boundaries before define_counts, so its first anchor
     *  reference is half a pixel bluer than the first centre.  Half a pixel
     *  sounds ignorable but it decides whether a user anchor that sits just
     *  below the first centre (3600 A, with data starting at 3600.4) falls
     *  inside the collar and whether the boundary anchor has any noticed pixel
     *  next to it -- i.e. whether the freeze rule in solve_stage pins the blue
     *  edge of the continuum.                                                */
    const Eigen::Index n_lam = spectrum.lambda.size();
    double min_lambda = spectrum.lambda[0];
    double max_lambda = spectrum.lambda[n_lam - 1];
    if (min_lambda > max_lambda) std::swap(min_lambda, max_lambda);
    if (n_lam >= 2) {
        const double half_lo = 0.5 * (spectrum.lambda[1] - spectrum.lambda[0]);
        const double half_hi = 0.5 * (spectrum.lambda[n_lam - 1] -
                                      spectrum.lambda[n_lam - 2]);
        min_lambda -= std::abs(half_lo);
        max_lambda -= std::abs(half_hi);   // bin_lo of the *last* bin
    }

    const double clip_lo = min_lambda - 2.0;
    const double clip_hi = max_lambda + 2.0;

    std::vector<double> xs;
    xs.reserve(32 + intervals.size() * 16);   // heuristic

    xs.push_back(min_lambda - 1.0);
    xs.push_back(max_lambda + 1.0);

    /* user supplied intervals ----------------------------------------- */
    for (const auto& tpl : intervals)
    {
        const double lo   = std::get<0>(tpl);
        const double hi   = std::get<1>(tpl);
        const double step = std::get<2>(tpl);

        if (step <= 0)
            throw std::runtime_error("anchors_from_intervals(): step ≤ 0 in interval");
        if (hi < lo)
            throw std::runtime_error("anchors_from_intervals(): hi < lo in interval");

        /* iterate with a small epsilon so that “hi” itself is included */
        for (double x = lo; x <= hi + 1e-6; x += step)
            if (x >= clip_lo && x <= clip_hi) xs.push_back(x);

        /* ensure hi is present if in range */
        if (hi >= clip_lo && hi <= clip_hi) xs.push_back(hi);
    }

    /* ---------- final clean-up ---------------------------------------- */
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    if (xs.size() < 2)
        throw std::runtime_error(
            "anchors_from_intervals(): fewer than two anchor points produced");

    /* ---------- return an owning Eigen vector ------------------------- */
    Vector out(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) out[i] = xs[i];
    return out;           // owns its data, safe outside the function
}

/* ------------------------------------------------------------------ */

Vector interp_linear(const Vector& x_in,
                     const Vector& y_in,
                     const Vector& x_out)
{
    const Eigen::Index n_in  = x_in.size();
    const Eigen::Index n_out = x_out.size();

    Vector out(n_out);

    // -------- 1. Pre-compute slopes -----------------------------------------
    Vector slope(n_in - 1);
    for (Eigen::Index i = 0; i < n_in - 1; ++i) {
        const double dx = x_in[i+1] - x_in[i];
        slope[i] = (std::abs(dx) < 1e-12) ? 0.0
                                          : (y_in[i+1] - y_in[i]) / dx;
    }

    // -------- 2. Single forward scan through the already-sorted x_out -------
    Eigen::Index seg = 0;                 // left border of current interval

    Eigen::Index k = 0;
    // left tail
    while (k < n_out && x_out[k] <= x_in[0])
        out[k++] = y_in[0];

    // interior region
    for (; k < n_out; ++k) {
        const double x = x_out[k];

        // right tail encountered → fill remainder and stop
        if (x >= x_in[n_in-1]) {
            for (; k < n_out; ++k) out[k] = y_in[n_in-1];
            break;
        }

        // advance segment until x_in[seg] ≤ x < x_in[seg+1]
        while (x_in[seg+1] < x) ++seg;

        out[k] = y_in[seg] + slope[seg] * (x - x_in[seg]);
    }
    return out;
}

} // namespace specfit