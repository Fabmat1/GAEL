/* ===================================================================== *
 *  src/SyntheticModel.cpp      –– two-level ultra-fast caching
 * ===================================================================== */

#include "specfit/SyntheticModel.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/Resolution.hpp"
#include "specfit/RotationalConvolution.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/ContinuumUtils.hpp"
#include <algorithm>
#include <vector>

#include <ankerl/unordered_dense.h>   // hash mixing constant
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <utility>
#include <iostream>

namespace specfit {

/* ------------------------------------------------------------------ *
 *  Tiny helper  –– boost-like hash_combine                           *
 * ------------------------------------------------------------------ */
namespace {

template<typename T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    seed ^= std::hash<T>{}(v) +
            0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

/* Hash for the FINAL synthetic spectrum (everything) */
inline std::size_t make_hash_full(const StellarParams& p,
                                  double               lam_min,
                                  double               lam_max,
                                  std::size_t          lam_sz,
                                  double               resOffset,
                                  double               resSlope,
                                  bool                 with_continuum)
{
    std::size_t seed = 0xF00DBAA5FULL;  // domain separator

    /*  Two different spectra live under otherwise identical parameters: the
     *  normalised one and the calibrated-flux-plus-continuum pair.          */
    hash_combine(seed, with_continuum);

    hash_combine(seed, p.vrad);
    hash_combine(seed, p.vsini);
    hash_combine(seed, p.zeta);
    hash_combine(seed, p.teff);
    hash_combine(seed, p.logg);
    hash_combine(seed, p.xi);
    hash_combine(seed, p.z);
    hash_combine(seed, p.he);
    for (double a : p.abundances) hash_combine(seed, a);

    hash_combine(seed, lam_min);
    hash_combine(seed, lam_max);
    hash_combine(seed, lam_sz);

    hash_combine(seed, resOffset);
    hash_combine(seed, resSlope);

    return seed;
}

/* Hash for the *surface* spectrum that comes from grid.load_spectrum(...)
 * (includes rotation and resolution degradation)
 */
inline std::size_t make_hash_surface(const StellarParams& p,
                                     double               resOffset,
                                     double               resSlope,
                                     std::size_t          window_key,
                                     bool                 with_continuum)
{
    std::size_t seed = 0xBADA555EULL;   // different domain separator

    hash_combine(seed, with_continuum);

    hash_combine(seed, p.teff);
    hash_combine(seed, p.logg);
    hash_combine(seed, p.z);
    hash_combine(seed, p.he);
    hash_combine(seed, p.xi);
    hash_combine(seed, p.vsini);    // Now included in surface hash
    for (double a : p.abundances) hash_combine(seed, a);

    hash_combine(seed, resOffset);
    hash_combine(seed, resSlope);

    /*  A surface spectrum is only valid for the wavelength window its grid
     *  was sliced to -- without this a fit with a narrow window would poison
     *  the cache for a later, wider one.                                    */
    hash_combine(seed, window_key);

    return seed;
}

/* ------------------------------------------------------------------ *
 *  ISIS ignores an element whose abundance is positive -- "Metals with
 *  positive abundances are ignored during the fitting process. This can be
 *  used to exclude metals from being fitted by assigning a positive abundance
 *  and freezing the corresponding parameter" (spectroscopic_fitting.sl, where
 *  the test is written `abundance[i] < 10`).
 * ------------------------------------------------------------------ */
inline bool element_active(double abundance) { return abundance < 10.0; }

std::vector<int> active_species(const ModelGrid& grid, const StellarParams& p)
{
    std::vector<int> out;
    const std::size_t n = std::min(p.abundances.size(), grid.species().size());
    for (std::size_t s = 0; s < n; ++s)
        if (element_active(p.abundances[s])) out.push_back(static_cast<int>(s));
    return out;
}

/* ------------------------------------------------------------------ *
 *  acc[i] *= interp(x_in, y_in)(x_out[i]).
 *
 *  Identical rule to interp_linear -- forward scan, edge clamping -- fused
 *  into the product so that multiplying ~25 species onto the union grid does
 *  not allocate, fill and free a 600 k-point temporary (and a slope table)
 *  per species per model evaluation.
 * ------------------------------------------------------------------ */
void multiply_interp(const Vector& x_in, const Vector& y_in,
                     const Vector& x_out, Vector& acc)
{
    const Eigen::Index n_in  = x_in.size();
    const Eigen::Index n_out = x_out.size();
    if (n_in == 0 || n_out == 0) return;

    const double* xi = x_in.data();
    const double* yi = y_in.data();
    const double* xo = x_out.data();
    double*       a  = acc.data();

    Eigen::Index k = 0;
    while (k < n_out && xo[k] <= xi[0]) { a[k] *= yi[0]; ++k; }

    if (n_in == 1) {
        for (; k < n_out; ++k) a[k] *= yi[0];
        return;
    }

    Eigen::Index seg = 0, slope_seg = -1;
    double slope = 0.0;
    for (; k < n_out; ++k) {
        const double x = xo[k];
        if (x >= xi[n_in - 1]) {
            for (; k < n_out; ++k) a[k] *= yi[n_in - 1];
            return;
        }
        while (xi[seg + 1] < x) ++seg;
        if (seg != slope_seg) {                 // only on a segment change
            const double dx = xi[seg + 1] - xi[seg];
            slope     = (std::abs(dx) < 1e-12) ? 0.0 : (yi[seg + 1] - yi[seg]) / dx;
            slope_seg = seg;
        }
        a[k] *= yi[seg] + slope * (x - xi[seg]);
    }
}

/* ------------------------------------------------------------------ *
 *  The grid the convolved metal model is reported on.
 *
 *  The union grid exists so that the *product* of the species' line-ratio
 *  spectra is formed at full resolution; the convolved result cannot carry
 *  structure narrower than lambda/R, so reporting it on all 600 k of those
 *  points is an order of magnitude more samples than it has information.
 *  Both convolutions therefore integrate over the whole union grid -- none of
 *  the input is discarded -- but land here instead.
 *
 *  Log-uniform at `oversample` samples per resolving power, so the sampling
 *  follows lambda/R across the range.  ISIS builds the same kind of grid in
 *  convolve_syn (`clambda`) at oversample = 2, calling that "as rough as
 *  possible but as fine as necessary".  The default here is twelve times
 *  finer than that.  Measured on Feros_3 with all 24 species on, R = 5400,
 *  vsini = 20 km/s, against the same model convolved on the full 380 k-point
 *  union grid and rebinned to 0.15 A bins:
 *
 *      oversample    max |d|     rms       per surface build
 *          4         5.9e-3    2.4e-4          0.02 s
 *          8         1.6e-3    6.3e-5          0.03 s
 *         24         1.8e-4    6.8e-6          0.04 s
 *         48         4.6e-5    1.8e-6          0.07 s
 *        none        2.7e-5    1.1e-6          0.46 s
 *
 *  The error falls as 1/oversample^2 down to a floor of 2.7e-5, which is not
 *  the sampling at all but the order swap below plus plain rounding.  At 24
 *  the worst pixel in the spectrum is off by 1.8e-4 of the continuum -- some
 *  fifty times under the noise of even a S/N = 100 spectrum -- and it costs
 *  almost nothing over the coarser settings, because by then the convolution
 *  is no longer what the build spends its time on.  GAEL_METAL_NYQ overrides.
 * ------------------------------------------------------------------ */
double metal_oversample()
{
    static const double v = [] {
        const char* e = std::getenv("GAEL_METAL_NYQ");
        if (!e || !*e) return 24.0;
        const double x = std::atof(e);
        return (x >= 1.0) ? x : 24.0;
    }();
    return v;
}

/*  Empty when the union grid should simply be kept: either the requested
 *  sampling is not coarser than it already is, or R is degenerate over the
 *  range and there is no sensible sampling to derive.                       */
Vector make_conv_grid(double lam_lo, double lam_hi,
                      double resOffset, double resSlope,
                      Eigen::Index n_union)
{
    const double R = std::max(resOffset + resSlope * lam_lo,
                              resOffset + resSlope * lam_hi);
    const double span = std::log(lam_hi / lam_lo);
    if (!(R > 0.0) || !(span > 0.0)) return Vector();

    const double dln = 1.0 / (metal_oversample() * R);
    const double raw = std::ceil(span / dln) + 1.0;
    /*  Compared before allocating: a caller asking for an absurd resolving
     *  power must not be able to size a vector by it.                       */
    if (!(raw >= 2.0) || raw >= static_cast<double>(n_union)) return Vector();

    const Eigen::Index n = static_cast<Eigen::Index>(raw);
    Vector g(n);
    const double step = span / static_cast<double>(n - 1);
    for (Eigen::Index i = 0; i < n; ++i)
        g[i] = lam_lo * std::exp(static_cast<double>(i) * step);
    g[0]     = lam_lo;      // exact endpoints: the convolution clamps outside
    g[n - 1] = lam_hi;
    return g;
}

/*  Memoised: the grid depends only on the union range and the spectrograph,
 *  both fixed for the whole fit.                                            */
const Vector& conv_grid(const Vector& lam_u, double resOffset, double resSlope)
{
    struct Entry { double lo, hi, ro, rs; Vector g; };
    static std::vector<Entry> cache;
    static std::mutex         mtx;

    const double lo = lam_u[0];
    const double hi = lam_u[lam_u.size() - 1];

    std::lock_guard<std::mutex> lk(mtx);
    for (const auto& e : cache)
        if (e.lo == lo && e.hi == hi && e.ro == resOffset && e.rs == resSlope)
            return e.g;

    Vector g = make_conv_grid(lo, hi, resOffset, resSlope, lam_u.size());
    if (g.size() == 0) g = lam_u;      // keep the union grid
    cache.push_back({lo, hi, resOffset, resSlope, std::move(g)});
    return cache.back().g;
}

} // unnamed namespace

/* ------------------------------------------------------------------ *
 *  Surface spectrum of a metal-bearing model.
 *
 *      f = f_HHE(T,g,z,He,xi) * prod_S f_S(T,g,z,He,xi,A_S)
 *
 *  ISIS forms this product on the union of the species' wavelength grids and
 *  only then convolves (spectroscopic_fitting.sl, the `params.metals` loop
 *  followed by convolve_syn).  The order is forced: the corner-level
 *  broadening the metal-free path relies on works because convolution is
 *  linear in the flux and therefore commutes with the interpolation weights,
 *  and a *product* of two fluxes does not.  So here the corners are read
 *  unbroadened, multiplied, and the result is broadened once -- which is why
 *  a metal fit costs materially more per model evaluation than an HHE one,
 *  in ISIS just as much as here.
 *
 *  The metals multiply the flux only.  In the multi-component representation
 *  the continuum stays the HHE continuum, exactly as ISIS builds `c[k]` from
 *  HHE before the metal loop runs.
 * ------------------------------------------------------------------ */
static Spectrum build_metal_surface(const ModelGrid&       grid,
                                    const StellarParams&   pars,
                                    const std::vector<int>& metals,
                                    double                 resOffset,
                                    double                 resSlope,
                                    bool                   with_continuum)
{
    /* ---- HHE, unbroadened, on the union grid ----------------------- */
    Spectrum hhe = grid.load_spectrum(pars.teff, pars.logg, pars.z, pars.he,
                                      pars.xi,
                                      /*vsini=*/0.0, /*resOffset=*/0.0,
                                      /*resSlope=*/0.0, with_continuum);

    const Vector& lam_u = grid.union_lambda(metals);

    Vector prod = interp_linear(hhe.lambda, hhe.flux, lam_u);

    /* ---- multiply in every active element -------------------------- */
    for (int s : metals) {
        const Vector fs = grid.load_element_factor(
            s, pars.teff, pars.logg, pars.z, pars.he, pars.xi,
            pars.abundances[static_cast<std::size_t>(s)]);
        multiply_interp(grid.species_lambda(s), fs, lam_u, prod);
    }

    /* ---- now, and only now, broaden -------------------------------- *
     *  Both convolutions integrate over the full union grid, but they report
     *  on `lam_c`, which is sampled to the *convolved* model's own content
     *  rather than to the product's -- see conv_grid().  The instrumental
     *  profile goes first so that the rotational one runs on the small grid:
     *  the two are convolutions with kernels of constant width in velocity,
     *  so they commute (ISIS convolves with their combined profile in a
     *  single pass for exactly that reason, convolve_syn's res_slope == 0
     *  branch), and doing it the other way round would hand the coarse grid a
     *  spectrum still carrying unresolved metal lines whenever vsini is
     *  small.
     *
     *  Without any instrumental degradation there is nothing to band-limit
     *  the model, so the union grid has to be kept.                        */
    const bool degrade = (resOffset != 0.0 || resSlope != 0.0);
    const Vector& lam_c = degrade ? conv_grid(lam_u, resOffset, resSlope)
                                  : lam_u;

    Vector flux = degrade ? degrade_resolution(lam_u, prod, lam_c,
                                               resOffset, resSlope)
                          : std::move(prod);

    if (pars.vsini >= 0.1)
        flux = rotational_broaden(lam_c, flux, lam_c, pars.vsini);

    Spectrum out;
    out.lambda = lam_c;
    out.flux   = std::move(flux);
    out.sigma  = Vector::Ones(lam_c.size());

    /*  ISIS never broadens the continuum: it divides the broadened flux by
     *  the unbroadened continuum resampled onto the convolved grid.  Here
     *  that grid is lam_c, so the HHE continuum is interpolated straight
     *  onto it.                                                            */
    if (with_continuum)
        out.cont = interp_linear(hhe.lambda, hhe.cont, lam_c);

    return out;
}

/* ------------------------------------------------------------------ *
 *                         main routine                               *
 * ------------------------------------------------------------------ */
SpectrumPtr compute_synthetic_cached(const ModelGrid&    grid,
                                     const StellarParams& pars,
                                     const Vector&        lambda_obs,
                                     double               resOffset,
                                     double               resSlope,
                                     bool                 with_continuum)
{
    //std::cout << "[CompSynth] Entering Function" << std::endl;  
    //std::cout << "[CompSynth] Making Hash." << std::endl;               
    /* ------------------------- FULL key ---------------------------- */
    const double      lam_min  = lambda_obs.minCoeff();
    const double      lam_max  = lambda_obs.maxCoeff();
    const std::size_t lam_size = static_cast<std::size_t>(lambda_obs.size());

    std::size_t full_key =
        make_hash_full(pars, lam_min, lam_max, lam_size, resOffset, resSlope,
                       with_continuum);
    hash_combine(full_key, grid.window_key());

    //std::cout << "[CompSynth] Made Hash. Spectrum Cache." << std::endl;     
    /* ===== 2nd-level cache (final spectrum) ======================== */
    SpectrumPtr final_sp =
        SpectrumCache::instance().insert_if_absent(full_key, [&] {

            /* ===== 1st-level cache (surface spectrum with rotation) === */
            const std::size_t surf_key =
                make_hash_surface(pars, resOffset, resSlope, grid.window_key(),
                                  with_continuum);

            SpectrumPtr surf_sp = SpectrumCache::instance()
                .insert_if_absent(surf_key, [&]{
                    const std::vector<int> metals = active_species(grid, pars);
                    if (metals.empty()) {
                        /*  The ordinary path, unchanged: each hypercube corner
                         *  is cached *already broadened*, which is sound
                         *  because convolution and the corner interpolation
                         *  are both linear and so commute.                  */
                        return grid.load_spectrum(pars.teff, pars.logg,
                                                   pars.z,   pars.he,
                                                   pars.xi,  pars.vsini,
                                                   resOffset, resSlope,
                                                   with_continuum);
                    }
                    return build_metal_surface(grid, pars, metals,
                                               resOffset, resSlope,
                                               with_continuum);
                });
            const Spectrum& surf = *surf_sp;      // safe reference
            //std::cout << "[CompSynth] Got Cached Spectrum. Doppler Shift." << std::endl;   
            /* ---------- remaining operations -------------------- */
            /* Rotational broadening is now already applied in grid.load_spectrum */
            
            /* 1) Doppler shift (depends on vrad) */
            constexpr double c = 299'792.458;           // km/s
            const double     factor = 1.0 + pars.vrad / c;
            Vector           lam_shift = surf.lambda * factor;
            
            //std::cout << "[CompSynth] Doppler Shifted. Interpolating onto wl grid." << std::endl;   

            /* 2) interpolate onto observed wavelength grid */
            //Vector interp = interp_linear(lam_shift, surf.flux, lambda_obs);
            Vector interp = trapezoidal_rebin(lam_shift, surf.flux, lambda_obs);

            //std::cout << "[CompSynth] Interpolated. Finishing up." << std::endl;
            /* 3) pack the final synthetic spectrum */
            Spectrum out;
            out.lambda = lambda_obs;
            out.flux   = std::move(interp);
            out.sigma  = Vector::Ones(lambda_obs.size());

            /*  The continuum rides along through the same shift and rebin, so
             *  that a caller mixing components can weight flux and continuum
             *  consistently bin by bin.                                      */
            if (with_continuum)
                out.cont = trapezoidal_rebin(lam_shift, surf.cont, lambda_obs);

            return out;    // moved into cache (as shared_ptr target)
        });

    return final_sp;
}

Spectrum compute_synthetic(const ModelGrid&    grid,
                           const StellarParams& pars,
                           const Vector&        lambda_obs,
                           double               resOffset,
                           double               resSlope,
                           bool                 with_continuum)
{
    return *compute_synthetic_cached(grid, pars, lambda_obs,
                                     resOffset, resSlope, with_continuum);
}

/* ------------------------------------------------------------------ *
 *  compute_synthetic_pure  –– raw interpolated spectrum, no degradation
 * ------------------------------------------------------------------ */
Spectrum compute_synthetic_pure(const ModelGrid&     grid,
                                const StellarParams& pars)
{
    /* 
     * We want the pure interpolated spectrum without:
     *   - Rotational broadening (vsini)
     *   - Macroturbulence (zeta)  
     *   - Instrumental resolution degradation
     *   - Radial velocity shift
     *
     * Call load_spectrum with:
     *   - vsini = 0 (no rotational broadening)
     *   - resOffset = 1e9, resSlope = 0 (effectively infinite resolution = no degradation)
     */
    
    constexpr double NO_VSINI     = 0.0;
    constexpr double HIGH_RES     = 0.0;   // R ~ infinity, no instrumental broadening
    constexpr double NO_RES_SLOPE = 0.0;
    
    Spectrum surf = grid.load_spectrum(
        pars.teff,
        pars.logg,
        pars.z,
        pars.he,
        pars.xi,
        NO_VSINI,
        HIGH_RES,
        NO_RES_SLOPE
    );
    
    // The spectrum from load_spectrum should already be normalized
    // Return it directly on the native wavelength grid
    return surf;
}

} // namespace specfit