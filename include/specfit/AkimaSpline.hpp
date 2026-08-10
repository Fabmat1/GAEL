#pragma once

#include "Types.hpp"
#include <memory>
#include <vector>
#include <boost/math/interpolators/makima.hpp>

namespace specfit {

/*
 *  The continuum spline through the cspline anchors.
 *
 *  Two variants are implemented:
 *
 *    modified Akima (boost `makima`)   -- the default
 *    classical Akima (Akima 1970)      -- GAEL_SPLINE=akima
 *
 *  ISIS uses classical Akima (`gsl->interp_akima`, see
 *  miscellaneous/initialize_cspline.sl), so matching it looks like the
 *  obviously right thing to do -- and through fixed anchors the two curves
 *  differ by only 0.16 % rms / 0.74 % peak.  It was measured anyway, and it
 *  makes agreement with ISIS *worse*, not better: over the 100 real_single
 *  cases, with everything else held fixed,
 *
 *      makima            Teff NMAD 108 K   log g NMAD 0.0214   He NMAD 0.0156
 *      classical Akima   Teff NMAD 135 K   log g NMAD 0.0208   He NMAD 0.0229
 *
 *  The likely reason is that GAEL fits the anchor ordinates through a
 *  finite-difference Jacobian.  Classical Akima weights its slopes with
 *  |m_{i+1} - m_i| alone, which vanishes wherever two secants agree, so the
 *  curve's dependence on an ordinate has kinks and the FD column is noisy.
 *  makima's extra |m_{i+1} + m_i| term keeps the weights away from zero.
 *
 *  So the default stays makima on measured agreement, and the classical
 *  implementation is kept, tested (it reproduces scipy's Akima1DInterpolator
 *  to 2e-11) and one env var away.
 *
 *  Classical-Akima coefficients follow GSL's akima.c exactly, including its
 *  boundary slopes (m[-1] = 2m[0]-m[1], m[-2] = 3m[0]-2m[1], mirrored at the
 *  top end).  Fewer than three knots degenerate to linear (or constant).
 *
 *  Outside [x_min, x_max] the spline is continued linearly with the end
 *  derivative.  With ISIS's anchor layout the boundary anchors sit 1 Angstrom
 *  outside the data, so this only ever guards against rounding.
 */
class AkimaSpline {
public:
    AkimaSpline(const Vector& x, const Vector& y);

    Real   operator()(Real x) const;
    /*  x must be ascending; evaluated with a single merged scan. */
    Vector operator()(const Vector& x) const;

private:
    /*  Non-null when the modified-Akima variant is in use (the default). */
    using Makima = decltype(boost::math::interpolators::makima(
        std::vector<Real>(), std::vector<Real>()));
    std::shared_ptr<Makima> makima_;

    std::vector<Real> x_, y_;      // knots
    std::vector<Real> b_, c_, d_;  // per-interval cubic coefficients
    Real x_min_ = 0.0, x_max_ = 0.0;
    Real y_min_ = 0.0, y_max_ = 0.0;
    Real deriv_min_ = 0.0, deriv_max_ = 0.0;

    Real eval_interval(std::size_t i, Real x) const;
};

} // namespace specfit
