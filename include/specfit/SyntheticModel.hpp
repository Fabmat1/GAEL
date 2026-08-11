#pragma once
#include "Spectrum.hpp"
#include "ModelGrid.hpp"
#include "ContinuumModel.hpp"
#include "SpectrumCache.hpp"
#include "ParameterSpec.hpp"
#include "ParameterIndexer.hpp"

#include <cstddef>
#include <vector>

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

    /*  One entry per element axis of this component's grid, in the grid's own
     *  species order; empty for an HHE-only grid.  ISIS multiplies one
     *  interpolated `f_element/f_HHE` ratio spectrum per entry onto the model
     *  (spectroscopic_fitting.sl, the `params.metals` loop).                */
    std::vector<double> abundances;
};

/* ------------------------------------------------------------------------- *
 *  Bridge between the descriptor table and the struct above, so that the
 *  places which copy a component's parameters out of (or into) the global
 *  vector can iterate the grid's parameter list instead of naming all nine
 *  fields.  Every such loop used to be nine hand-written lines, repeated in
 *  five functions.
 * ------------------------------------------------------------------------- */
inline void set_stellar_param(StellarParams& sp, const ParamSpec& ps, double v)
{
    switch (ps.kind) {
        case ParamKind::Vrad:     sp.vrad      = v; break;
        case ParamKind::Vsini:    sp.vsini     = v; break;
        case ParamKind::Zeta:     sp.zeta      = v; break;
        case ParamKind::Teff:     sp.teff      = v; break;
        case ParamKind::Logg:     sp.logg      = v; break;
        case ParamKind::Xi:       sp.xi        = v; break;
        case ParamKind::Z:        sp.z         = v; break;
        case ParamKind::He:       sp.he        = v; break;
        case ParamKind::SurRatio: sp.sur_ratio = v; break;
        case ParamKind::Abundance:
            if (ps.species >= 0) {
                if (sp.abundances.size() <= static_cast<std::size_t>(ps.species))
                    sp.abundances.resize(static_cast<std::size_t>(ps.species) + 1);
                sp.abundances[static_cast<std::size_t>(ps.species)] = v;
            }
            break;
    }
}

inline double get_stellar_param(const StellarParams& sp, const ParamSpec& ps)
{
    switch (ps.kind) {
        case ParamKind::Vrad:     return sp.vrad;
        case ParamKind::Vsini:    return sp.vsini;
        case ParamKind::Zeta:     return sp.zeta;
        case ParamKind::Teff:     return sp.teff;
        case ParamKind::Logg:     return sp.logg;
        case ParamKind::Xi:       return sp.xi;
        case ParamKind::Z:        return sp.z;
        case ParamKind::He:       return sp.he;
        case ParamKind::SurRatio: return sp.sur_ratio;
        case ParamKind::Abundance:
            return (ps.species >= 0 &&
                    static_cast<std::size_t>(ps.species) < sp.abundances.size())
                   ? sp.abundances[static_cast<std::size_t>(ps.species)] : 0.0;
    }
    return 0.0;
}

/*  Gather one component's parameters for one dataset out of the global vector.
 *  Templated on the container so the LM's Eigen::VectorXd and the workflow's
 *  std::vector<double> share one implementation.                            */
template<typename ParamVecT>
inline StellarParams stellar_params_from(const ParameterIndexer& indexer,
                                         const ParamVecT&        p,
                                         int                     comp,
                                         int                     dataset)
{
    StellarParams sp{};
    const auto& table = indexer.params(comp);
    for (std::size_t s = 0; s < table.size(); ++s)
        set_stellar_param(sp, table[s],
                          p[indexer.get(comp, dataset, static_cast<int>(s))]);
    return sp;
}

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