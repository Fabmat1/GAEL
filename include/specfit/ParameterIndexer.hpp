#pragma once
/*
 * Helper that maps  (component  ,  dataset  ,  parameter)
 * to one global position in the unified parameter vector that
 * Levenberg–Marquardt operates on.
 *
 * If a parameter is “tied” the very same global index is stored
 * for each dataset, otherwise every spectrum receives its own slot.
 *
 *  Which parameters a component has -- and how many -- comes from the grid,
 *  not from this header; see ParameterSpec.hpp.  A component built on an
 *  HHE-only grid has the historical nine:
 *
 *      0 vrad   1 vsini   2 zeta   3 teff
 *      4 logg   5 xi      6 z      7 he
 *      8 sur_ratio
 *
 *  and a grid with element axes has one extra slot per element between `he`
 *  and `sur_ratio`.  Address parameters by *name* wherever the code cares
 *  which one it is; the integer slot is for iteration.
 *
 *  `sur_ratio` is the surface-area ratio of this component to the first one
 *  and only does anything in a multi-component fit; component 1's is pinned
 *  to 1 by definition.  It sits last, matching ISIS's ordering within a
 *  component, and is tied across datasets by default like the rest.
 */

#include "ParameterSpec.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace specfit {

class ParameterIndexer {
public:
    /* idx[component][dataset][slot] → global index */
    std::vector<std::vector<std::vector<int>>> idx;

    /* total number of stellar parameters in the global vector */
    int total_stellar_params = 0;

    int get(int comp, int dataset, int slot) const {
        return idx[comp][dataset][slot];
    }

    /*  By name; -1 when this component's grid has no such parameter.  Kept a
     *  distinct name rather than an overload of get(): a string literal is one
     *  user-defined conversion away from std::string, and an overload set that
     *  differs only in the third argument's type is a trap.                  */
    int index_of(int comp, int dataset, const std::string& name) const {
        const int s = slot_of(comp, name);
        return s < 0 ? -1 : idx[comp][dataset][s];
    }

    int slot_of(int comp, const std::string& name) const {
        const auto& t = specs_[static_cast<std::size_t>(comp)];
        for (std::size_t s = 0; s < t.size(); ++s)
            if (t[s].name == name) return static_cast<int>(s);
        return -1;
    }

    const std::vector<ParamSpec>& params(int comp) const {
        return specs_[static_cast<std::size_t>(comp)];
    }
    int n_params(int comp) const {
        return static_cast<int>(specs_[static_cast<std::size_t>(comp)].size());
    }
    int n_components() const { return static_cast<int>(specs_.size()); }

    /* -------------------------------------------------------- */
    /*  build complete mapping                                  */
    /* -------------------------------------------------------- */
    /*  `per_component[c]` is the parameter list of component c, normally
     *  built by component_params(grid.species()).  Components may legitimately
     *  differ: a binary can mix a grid that resolves iron with one that does
     *  not.                                                                  */
    template<typename UntieListT>
    void build(const std::vector<std::vector<ParamSpec>>& per_component,
               int n_datasets,
               const UntieListT& untie_params)
    {
        const auto is_untied = [&](const std::string& n)->bool {
            return std::find(untie_params.begin(),
                             untie_params.end(),
                             n) != untie_params.end();
        };

        specs_ = per_component;
        const int n_components = static_cast<int>(specs_.size());

        idx.assign(static_cast<std::size_t>(n_components), {});
        for (int c = 0; c < n_components; ++c)
            idx[static_cast<std::size_t>(c)]
                .assign(static_cast<std::size_t>(n_datasets),
                        std::vector<int>(specs_[static_cast<std::size_t>(c)].size(), -1));

        total_stellar_params = 0;
        for (int c = 0; c < n_components; ++c)
            for (int p = 0; p < n_params(c); ++p)
            {
                bool u = is_untied(specs_[static_cast<std::size_t>(c)]
                                       [static_cast<std::size_t>(p)].name);
                int shared_idx = -1;
                for (int d = 0; d < n_datasets; ++d)
                {
                    int gidx = u ? total_stellar_params++
                                 : (shared_idx>=0 ? shared_idx
                                                  : (shared_idx=total_stellar_params++));
                    idx[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)]
                       [static_cast<std::size_t>(p)] = gidx;
                }
            }
    }

    /*  Convenience for callers that only have HHE-only grids (tests, tools). */
    template<typename UntieListT>
    void build(int n_components, int n_datasets, const UntieListT& untie_params)
    {
        build(std::vector<std::vector<ParamSpec>>(
                  static_cast<std::size_t>(n_components), component_params()),
              n_datasets, untie_params);
    }

private:
    std::vector<std::vector<ParamSpec>> specs_;
};

} // namespace specfit
