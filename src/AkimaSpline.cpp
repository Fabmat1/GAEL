#include "specfit/AkimaSpline.hpp"

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace specfit {

namespace {
/*  GAEL_SPLINE=akima selects classical Akima; anything else keeps makima. */
bool use_classical_akima()
{
    static const bool v = [] {
        const char* e = std::getenv("GAEL_SPLINE");
        return e && std::strcmp(e, "akima") == 0;
    }();
    return v;
}
} // namespace

AkimaSpline::AkimaSpline(const Vector& x, const Vector& y)
{
    const std::size_t n = static_cast<std::size_t>(x.size());
    if (n == 0)
        throw std::runtime_error("AkimaSpline: no knots");
    if (static_cast<std::size_t>(y.size()) != n)
        throw std::runtime_error("AkimaSpline: x and y differ in length");

    x_.assign(x.data(), x.data() + n);
    y_.assign(y.data(), y.data() + n);

    x_min_ = x_.front();
    x_max_ = x_.back();
    y_min_ = y_.front();
    y_max_ = y_.back();

    if (n >= 4 && !use_classical_akima()) {
        makima_ = std::make_shared<Makima>(boost::math::interpolators::makima(
            std::vector<Real>(x_), std::vector<Real>(y_)));
        const Real h = 1e-6;
        y_min_ = (*makima_)(x_min_);
        y_max_ = (*makima_)(x_max_);
        deriv_min_ = ((*makima_)(x_min_ + h) - y_min_) / h;
        deriv_max_ = (y_max_ - (*makima_)(x_max_ - h)) / h;
        return;
    }

    if (n == 1) {                       // constant
        deriv_min_ = deriv_max_ = 0.0;
        return;
    }

    b_.assign(n - 1, 0.0);
    c_.assign(n - 1, 0.0);
    d_.assign(n - 1, 0.0);

    /* ---- secant slopes, with GSL's two virtual slopes at each end ----- *
     *  m_ is indexed 0..n+2 and corresponds to GSL's _m[-2] .. _m[n].     */
    std::vector<Real> m_(n + 3, 0.0);
    auto m = [&](std::ptrdiff_t i) -> Real& {
        return m_[static_cast<std::size_t>(i + 2)];
    };

    for (std::size_t i = 0; i + 1 < n; ++i) {
        const Real h = x_[i + 1] - x_[i];
        m(static_cast<std::ptrdiff_t>(i)) =
            (h != 0.0) ? (y_[i + 1] - y_[i]) / h : 0.0;
    }

    if (n == 2) {                       // a single secant: straight line
        b_[0] = m(0);
        deriv_min_ = deriv_max_ = m(0);
        return;
    }

    const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(n);
    m(-1)    = 2.0 * m(0)     - m(1);
    m(-2)    = 3.0 * m(0)     - 2.0 * m(1);
    m(N - 1) = 2.0 * m(N - 2) - m(N - 3);
    m(N)     = 3.0 * m(N - 2) - 2.0 * m(N - 3);

    /* ---- GSL akima.c: akima_calc() ------------------------------------ */
    for (std::ptrdiff_t i = 0; i < N - 1; ++i) {
        const Real NE = std::abs(m(i + 1) - m(i)) + std::abs(m(i - 1) - m(i - 2));
        if (NE == 0.0) {
            b_[static_cast<std::size_t>(i)] = m(i);
            c_[static_cast<std::size_t>(i)] = 0.0;
            d_[static_cast<std::size_t>(i)] = 0.0;
            continue;
        }

        const Real h_i = x_[static_cast<std::size_t>(i) + 1]
                       - x_[static_cast<std::size_t>(i)];
        const Real NE_next =
            std::abs(m(i + 2) - m(i + 1)) + std::abs(m(i) - m(i - 1));

        const Real alpha_i = std::abs(m(i - 1) - m(i - 2)) / NE;

        Real tL_ip1;
        if (NE_next == 0.0) {
            tL_ip1 = m(i);
        } else {
            const Real alpha_next = std::abs(m(i) - m(i - 1)) / NE_next;
            tL_ip1 = (1.0 - alpha_next) * m(i) + alpha_next * m(i + 1);
        }

        const Real bi = (1.0 - alpha_i) * m(i - 1) + alpha_i * m(i);
        b_[static_cast<std::size_t>(i)] = bi;
        c_[static_cast<std::size_t>(i)] = (3.0 * m(i) - 2.0 * bi - tL_ip1) / h_i;
        d_[static_cast<std::size_t>(i)] = (bi + tL_ip1 - 2.0 * m(i)) / (h_i * h_i);
    }

    deriv_min_ = b_.front();
    {
        const std::size_t last = n - 2;
        const Real h = x_[n - 1] - x_[last];
        deriv_max_ = b_[last] + h * (2.0 * c_[last] + 3.0 * d_[last] * h);
    }
}

Real AkimaSpline::eval_interval(std::size_t i, Real x) const
{
    const Real dx = x - x_[i];
    return y_[i] + dx * (b_[i] + dx * (c_[i] + d_[i] * dx));
}

Real AkimaSpline::operator()(Real x) const
{
    if (x_.size() == 1)   return y_min_;
    if (x < x_min_)       return y_min_ + deriv_min_ * (x - x_min_);
    if (x > x_max_)       return y_max_ + deriv_max_ * (x - x_max_);
    if (makima_)          return (*makima_)(x);

    /* last knot inclusive: upper_bound gives the interval to its left */
    const auto it = std::upper_bound(x_.begin(), x_.end(), x);
    std::size_t i = static_cast<std::size_t>(it - x_.begin());
    i = (i == 0) ? 0 : i - 1;
    if (i >= b_.size()) i = b_.size() - 1;
    return eval_interval(i, x);
}

Vector AkimaSpline::operator()(const Vector& x) const
{
    Vector out(x.size());
    if (x_.size() == 1) { out.setConstant(y_min_); return out; }
    if (makima_) {
        for (Eigen::Index k = 0; k < x.size(); ++k) out[k] = (*this)(x[k]);
        return out;
    }

    /* The evaluation grid is the (sorted) observed wavelength array, so one
       merged scan replaces a binary search per pixel.  Fall back to the
       scalar path for any point that breaks monotonicity.                 */
    std::size_t i = 0;
    Real prev = -std::numeric_limits<Real>::infinity();
    for (Eigen::Index k = 0; k < x.size(); ++k) {
        const Real xv = x[k];
        if (xv < prev) { out[k] = (*this)(xv); continue; }
        prev = xv;

        if (xv < x_min_)      { out[k] = y_min_ + deriv_min_ * (xv - x_min_); continue; }
        if (xv > x_max_)      { out[k] = y_max_ + deriv_max_ * (xv - x_max_); continue; }

        while (i + 1 < b_.size() && x_[i + 1] <= xv) ++i;
        out[k] = eval_interval(i, xv);
    }
    return out;
}

} // namespace specfit
