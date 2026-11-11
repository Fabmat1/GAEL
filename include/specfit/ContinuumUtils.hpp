#pragma once
#include "Types.hpp"
#include "Spectrum.hpp"
#include <array>
#include <vector>
#include <tuple>

namespace specfit {

/**
 * Build continuum–spline anchor positions from the triples that appear in the
 * JSON configuration:
 *
 *     [ start , end , step ]
 *
 * For every triple the function returns      start , start+step , … , end
 * (end is enforced to be contained ‑ no duplicates).
 */
Vector anchors_from_intervals(
        const std::vector<std::tuple<double, double, double>>& intervals,
        const Spectrum& spectrum);

/**
 * Anchor intervals for one continuum-jitter refit: every step is scaled by
 * `scale` and every start moves `phase` (0..1) of the scaled step to the red.
 *
 * The start never moves past the end.  A range narrower than the shift (one
 * trimmed to the spectrum's edge, say) keeps only its end anchor, which is
 * what [start, end, step] means when no start + k*step falls inside it.
 */
std::vector<std::array<double, 3>> jitter_anchor_intervals(
        std::vector<std::array<double, 3>> intervals,
        double scale, double phase);

/**
 * Helper – simple linear interpolation y(x_out)  (no extrapolation).
 */
Vector interp_linear(const Vector& x_in,
                     const Vector& y_in,
                     const Vector& x_out);

} // namespace specfit
