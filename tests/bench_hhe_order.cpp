/* ------------------------------------------------------------------ *
 *  Model-level accuracy of the metal-free surface-spectrum pipeline.
 *
 *  Compares, on the Nyquist grid a fit actually sees, against a reference
 *  computed as accurately as the native grid allows:
 *
 *    REF   interpolate raw corners -> rotate (n_kernel taps, native grid)
 *          -> degrade (native) -> rebin
 *    OLD   ModelGrid::load_spectrum as it stands (each corner rotated with
 *          81 taps, then degraded, then interpolated) -> rebin
 *    NEW   interpolate raw -> degrade onto a log grid at `os` per R
 *          -> rotate there -> rebin
 *
 *  Usage: bench_hhe_order <grid_base> <rel> T g z he xi resOffset resSlope
 *                         [n_kernel of the reference]
 * ------------------------------------------------------------------ */
#include "specfit/ModelGrid.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/Resolution.hpp"
#include "specfit/RotationalConvolution.hpp"
#include "specfit/NyquistGrid.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace specfit;

static Vector log_grid(double lo, double hi, double R, double os)
{
    const double span = std::log(hi / lo);
    const Eigen::Index n = Eigen::Index(std::ceil(span * os * R)) + 1;
    Vector g(n);
    const double step = span / double(n - 1);
    for (Eigen::Index i = 0; i < n; ++i) g[i] = lo * std::exp(double(i) * step);
    g[0] = lo; g[n - 1] = hi;
    return g;
}

struct Stat { double maxd, rms; };
static Stat diff(const Vector& a, const Vector& b)
{
    const Vector d = a - b;
    return { d.cwiseAbs().maxCoeff(), std::sqrt(d.squaredNorm() / d.size()) };
}

int main(int argc, char** argv)
{
    if (argc < 8) { std::fprintf(stderr, "usage\n"); return 2; }
    const std::string base = argv[1], rel = argv[2];
    const double T = std::atof(argv[3]), g = std::atof(argv[4]),
                 z = std::atof(argv[5]), he = std::atof(argv[6]),
                 xi = std::atof(argv[7]);
    const double ro = std::atof(argv[8]), rs = std::atof(argv[9]);
    const int nref = argc > 10 ? std::atoi(argv[10]) : 1001;

    ModelGrid grid({base}, rel);
    const double obs_lo = 3625, obs_hi = 9100;
    grid.set_wavelength_window(obs_lo / 1.007, obs_hi * 1.007);

    const Spectrum raw = grid.load_spectrum(T, g, z, he, xi, 0.0, 0.0, 0.0);
    std::printf("native points in window: %ld\n", long(raw.lambda.size()));

    struct Res { double ro, rs; const char* name; };
    const std::vector<Res> res = { {ro, rs, "R"} };
    const std::vector<double> vs = { 2, 10, 30, 60, 100, 140 };
    const std::vector<double> oss = { 24, 48, 96 };

    for (const auto& r : res) {
        /*  Nyquist grid exactly as GaelAPI builds it, nq_eff 2.7, trimmed
         *  well inside the window so edge effects stay out.                */
        const Vector obs = build_nyquist_grid(obs_lo + 20, obs_hi - 20,
                                              r.ro, r.rs, 2.7);
        std::printf("\n=== R = %g + %g*lam  (obs %ld px) ===\n", r.ro, r.rs, long(obs.size()));
        std::printf("%6s | %-21s | %-21s", "vsini", "OLD-REF max / rms",
                    "NATIVE-REF max/rms");
        for (double os : oss) std::printf(" | os=%-3g raw / corr ", os);
        std::printf("\n");

        for (double v : vs) {
            auto t0 = std::chrono::steady_clock::now();
            const Spectrum old = grid.load_spectrum(T, g, z, he, xi, v,
                                                    r.ro, r.rs);
            const double t_old = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            const Vector f_old = trapezoidal_rebin(old.lambda, old.flux, obs);

            const Eigen::Index nn = raw.lambda.size();
            const double Rmax = std::max(r.ro + r.rs * raw.lambda[0],
                                         r.ro + r.rs * raw.lambda[nn - 1]);
            /*  REF: rotate on the native grid, degrade onto a log grid at
             *  200 per R and rebin from there -- the rebin from the native
             *  grid carries an h^2 error as large as the effects measured.  */
            const Vector lf = log_grid(raw.lambda[0], raw.lambda[nn - 1], Rmax, 200);
            const Vector rot_ref = rotational_broaden(raw.lambda, raw.flux,
                                                      raw.lambda, v, 0.6, nref);
            const Vector f_ref = trapezoidal_rebin(
                lf, degrade_resolution(raw.lambda, rot_ref, lf, r.ro, r.rs), obs);
            /*  NAT: the same on the native grid throughout -- what OLD
             *  computes, up to the order of the corner sum.                 */
            const Vector f_nat = trapezoidal_rebin(
                raw.lambda, degrade_resolution(raw.lambda, rot_ref, r.ro, r.rs), obs);

            const Stat so = diff(f_old, f_ref), s2 = diff(f_nat, f_ref);
            std::printf("%6g | %9.2e / %9.2e | %9.2e / %9.2e", v,
                        so.maxd, so.rms, s2.maxd, s2.rms);

            {   /*  NEW on the native grid: degrade, then rotate (81 taps). */
                const Vector d = degrade_resolution(raw.lambda, raw.flux, r.ro, r.rs);
                const Vector rot = rotational_broaden(raw.lambda, d, raw.lambda, v, 0.6, 81);
                const Vector f_nat2 = trapezoidal_rebin(raw.lambda, rot, obs);
                const Stat sn = diff(f_nat2, f_ref), so2 = diff(f_nat2, f_old);
                std::printf(" | nat %8.1e (vs OLD %8.1e / rms %8.1e)", sn.maxd, so2.maxd, so2.rms);
            }
            for (double os : oss) {
                /*  Never finer than the native grid's typical step. */
                const double nat = std::log(raw.lambda[nn - 1] / raw.lambda[0]) / double(nn - 1);
                const double os_eff = std::min(os, 1.0 / (nat * Rmax));
                const Vector lc = log_grid(raw.lambda[0], raw.lambda[nn - 1],
                                           Rmax, os_eff);
                const double h = std::log(lc[1] / lc[0]);
                const int nk = std::max(81, 4 * int(std::ceil(2.0 * v / 299792.458 / h)) + 1);
                auto t1 = std::chrono::steady_clock::now();
                const Vector d = degrade_resolution(raw.lambda, raw.flux, lc,
                                                    r.ro, r.rs);
                const Vector rot = rotational_broaden(lc, d, lc, v, 0.6, nk);
                const Vector f_new = trapezoidal_rebin(lc, rot, obs);
                const double t_new = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t1).count();
                /*  Euler-Maclaurin pre-correction for the trapezoid rebin. */
                Vector c = rot;
                for (Eigen::Index i = 1; i + 1 < c.size(); ++i)
                    c[i] -= (rot[i + 1] - 2.0 * rot[i] + rot[i - 1]) / 12.0;
                const Vector f_cor = trapezoidal_rebin(lc, c, obs);
                const Stat sn = diff(f_new, f_ref), sc = diff(f_cor, f_ref);
                std::printf(" | %8.1e %8.1e", sn.maxd, sc.maxd);
                if (v == vs.front()) std::fprintf(stderr, "os %g -> %g, %ld pts\n", os, os_eff, long(lc.size()));
                (void)t_new;
            }
            std::printf("   [old %.2fs]\n", t_old);
        }
    }
    return 0;
}
