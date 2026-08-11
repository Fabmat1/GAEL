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
};

/*  Shared handle on the cached spectrum.  The hot path (one call per spectrum
 *  per residual evaluation) only reads it, and copying three vectors the
 *  length of the observed grid out of the cache each time was pure overhead. */
SpectrumPtr compute_synthetic_cached(const ModelGrid& grid,
                                     const StellarParams& pars,
                                     const Vector& lambda_obs,
                                     double resOffset,
                                     double resSlope);

/*  Copying convenience wrapper, for callers that want to own the result. */
Spectrum compute_synthetic(const ModelGrid& grid,
                           const StellarParams& pars,
                           const Vector& lambda_obs,
                           double resOffset,
                           double resSlope);


Spectrum compute_synthetic_pure(const ModelGrid& grid, const StellarParams& params);

} // namespace specfit