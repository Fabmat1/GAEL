#pragma once
#include "Spectrum.hpp"
#include "ModelGrid.hpp"
#include "ContinuumModel.hpp"
#include "SpectrumCache.hpp"

namespace specfit {

struct StellarParams {
    double vrad;
    double vsini;
    double zeta;
    double teff;
    double logg;
    double xi;
    double z;
    double he;

    /*  Ratio of this component's effective surface area to the first
     *  component's (ISIS: `cN_sur_ratio`).  It mixes the components *after*
     *  each synthetic spectrum has been built, so -- unlike every field above
     *  -- it is deliberately not part of the spectrum cache keys.           */
    double sur_ratio = 1.0;
};

/*  Shared handle on the cached spectrum.  The hot path (one call per spectrum
 *  per residual evaluation) only reads it, and copying three vectors the
 *  length of the observed grid out of the cache each time was pure overhead.
 *
 *  With `with_continuum` the result carries the calibrated flux in `flux` and
 *  the component continuum in `cont`, which is what a multi-component fit
 *  combines; see ModelGrid::load_spectrum.                                  */
SpectrumPtr compute_synthetic_cached(const ModelGrid& grid,
                                     const StellarParams& pars,
                                     const Vector& lambda_obs,
                                     double resOffset,
                                     double resSlope,
                                     bool   with_continuum = false);

/*  Copying convenience wrapper, for callers that want to own the result. */
Spectrum compute_synthetic(const ModelGrid& grid,
                           const StellarParams& pars,
                           const Vector& lambda_obs,
                           double resOffset,
                           double resSlope,
                           bool   with_continuum = false);


Spectrum compute_synthetic_pure(const ModelGrid& grid, const StellarParams& params);

} // namespace specfit