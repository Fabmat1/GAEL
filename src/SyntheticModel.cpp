/* ===================================================================== *
 *  src/SyntheticModel.cpp      –– two-level ultra-fast caching
 * ===================================================================== */

#include "specfit/SyntheticModel.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/Resolution.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/ContinuumUtils.hpp"

#include <ankerl/unordered_dense.h>   // hash mixing constant
#include <cmath>
#include <cstddef>
#include <functional>
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

    hash_combine(seed, resOffset);
    hash_combine(seed, resSlope);

    /*  A surface spectrum is only valid for the wavelength window its grid
     *  was sliced to -- without this a fit with a narrow window would poison
     *  the cache for a later, wider one.                                    */
    hash_combine(seed, window_key);

    return seed;
}

} // unnamed namespace

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
                    // Now includes vsini for rotational broadening
                    return grid.load_spectrum(pars.teff, pars.logg,
                                               pars.z,   pars.he,
                                               pars.xi,  pars.vsini,
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