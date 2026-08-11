#pragma once
/* ------------------------------------------------------------------------- *
 *  Description of one fitted parameter of one stellar component.
 *
 *  This list used to be hard-coded: nine parameters, fixed order, the names
 *  spelled out again in eight translation units and the count baked into a
 *  `static constexpr int kNStellarParams`.  A grid that carries per-element
 *  abundance axes has a *variable* number of parameters per component -- and
 *  which ones depends on the grid, not on the code -- so the list became data.
 *
 *  The order here is the order of the global parameter vector, and therefore
 *  of everything derived from it: the uncertainties, the free mask, and the
 *  flat walk that writes fit_parameters.csv.  It keeps GAEL's historical
 *  placement of vrad/vsini/zeta and slots element abundances in between `he`
 *  and `sur_ratio` -- exactly where ISIS puts its metals within a component
 *  (spectroscopic_fitting.sl, the `stellar_default` switch) -- so a grid
 *  without element axes produces the identical layout it always did.
 * ------------------------------------------------------------------------- */

#include <string>
#include <vector>

namespace specfit {

/*  What a parameter *means* to the model.  The name alone cannot say it:
 *  element names come from the grid, so `Abundance` is the open-ended case
 *  and `species` says which element axis it drives.                         */
enum class ParamKind {
    Vrad, Vsini, Zeta, Teff, Logg, Xi, Z, He,
    Abundance,
    SurRatio
};

struct ParamSpec {
    std::string name;           // "vrad" .. "sur_ratio", or an element: "FE"
    ParamKind   kind    = ParamKind::Abundance;
    int         species = -1;   // index into the grid's species list (Abundance only)
};

/*  The eight parameters every component has, whatever grid it uses. */
inline const std::vector<ParamSpec>& canonical_params()
{
    static const std::vector<ParamSpec> v = {
        {"vrad",  ParamKind::Vrad },
        {"vsini", ParamKind::Vsini},
        {"zeta",  ParamKind::Zeta },
        {"teff",  ParamKind::Teff },
        {"logg",  ParamKind::Logg },
        {"xi",    ParamKind::Xi   },
        {"z",     ParamKind::Z    },
        {"he",    ParamKind::He   },
    };
    return v;
}

/*  Full parameter list of one component: the canonical eight, then one
 *  abundance per element the grid resolves, then the surface ratio.  Pass the
 *  grid's species list (empty for an HHE-only grid).                        */
inline std::vector<ParamSpec>
component_params(const std::vector<std::string>& species = {})
{
    std::vector<ParamSpec> v = canonical_params();
    v.reserve(v.size() + species.size() + 1);
    for (std::size_t s = 0; s < species.size(); ++s)
        v.push_back({species[s], ParamKind::Abundance, static_cast<int>(s)});
    v.push_back({"sur_ratio", ParamKind::SurRatio});
    return v;
}

} // namespace specfit
