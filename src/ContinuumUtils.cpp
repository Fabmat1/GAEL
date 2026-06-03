#include "specfit/ContinuumUtils.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>       // std::abs
#include <cstdlib>     // std::getenv

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

    /* ---------- collect wavelengths used to bound the anchor list ------ *
     *  Default: only non-ignored (fit-window) points  ->  anchors live
     *  inside the fit window.  ISIS instead spans the full DATA range and
     *  adds exterior boundary anchors, giving a denser, edge-constrained
     *  continuum.  DIGGA_ANCHORS_FULLRANGE=1 reproduces the ISIS behaviour
     *  (clip anchors to the full data range, ignore the mask for bounds). */
    const char* full_env = std::getenv("DIGGA_ANCHORS_FULLRANGE");
    const bool  full_range = full_env && std::string(full_env) == "1";

    std::vector<double> good_lambda;
    good_lambda.reserve(spectrum.lambda.size());

    for (Eigen::Index i = 0; i < spectrum.lambda.size(); ++i)
    {
        bool keep = (full_range || spectrum.ignoreflag.empty())
                        ? true
                        : (spectrum.ignoreflag[i] == 1);
        if (keep) good_lambda.push_back(spectrum.lambda[i]);
    }

    if (good_lambda.empty())
        throw std::runtime_error(
            "anchors_from_intervals(): after masking, no wavelength point is left");

    std::sort(good_lambda.begin(), good_lambda.end());
    const double min_lambda = good_lambda.front();
    const double max_lambda = good_lambda.back();

    /* ---------- build anchor list ------------------------------------- */
    std::vector<double> xs;
    xs.reserve(32 + intervals.size() * 16);   // heuristic

    /* two boundary anchors (or the 10th / n-10th if enough points) */
    if (good_lambda.size() >= 20)
    {
        xs.push_back(good_lambda[9]);
        xs.push_back(good_lambda[good_lambda.size() - 10]);
    }
    else
    {
        xs.push_back(min_lambda);
        xs.push_back(max_lambda);
    }

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
            if (x >= min_lambda && x <= max_lambda) xs.push_back(x);

        /* ensure hi is present if in range */
        if (hi >= min_lambda && hi <= max_lambda) xs.push_back(hi);
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