#pragma once
/*
 * Helper that maps  (component  ,  dataset  ,  stellar-parameter)
 * to one global position in the unified parameter vector that
 * Levenberg–Marquardt operates on.
 *
 * If a parameter is “tied” the very same global index is stored
 * for each dataset, otherwise every spectrum receives its own slot.
 *
 *  Stellar parameter order:
 *      0 vrad   1 vsini   2 zeta   3 teff
 *      4 logg   5 xi      6 z      7 he
 *      8 sur_ratio
 *
 *  `sur_ratio` is the surface-area ratio of this component to the first one
 *  and only does anything in a multi-component fit; component 1's is pinned
 *  to 1 by definition.  It sits last, matching ISIS's ordering within a
 *  component, and is tied across datasets by default like the rest.
 */

#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cstddef>

namespace specfit {

class ParameterIndexer {
public:
    static constexpr int kNStellarParams = 9;

    /* idx[component][dataset][param] → global index */
    std::vector<std::vector<std::array<int, kNStellarParams>>> idx;

    /* total number of stellar parameters in the global vector */
    int total_stellar_params = 0;

    int get(int comp, int dataset, int par) const {
        return idx[comp][dataset][par];
    }

    /* -------------------------------------------------------- */
    /*  build complete mapping                                  */
    /* -------------------------------------------------------- */
    template<typename UntieListT>
    void build(int n_components,
               int n_datasets,
               const UntieListT& untie_params)
    {
        const auto is_untied = [&](const std::string& n)->bool {
            return std::find(untie_params.begin(),
                             untie_params.end(),
                             n) != untie_params.end();
        };

        const char* names[kNStellarParams] =
            { "vrad","vsini","zeta","teff","logg","xi","z","he","sur_ratio" };

        idx.resize(n_components);
        for (int c = 0; c < n_components; ++c)
            idx[c].resize(n_datasets);

        total_stellar_params = 0;
        for (int c = 0; c < n_components; ++c)
            for (int p = 0; p < kNStellarParams; ++p)
            {
                bool u = is_untied(names[p]);
                int shared_idx = -1;
                for (int d = 0; d < n_datasets; ++d)
                {
                    int gidx = u ? total_stellar_params++
                                 : (shared_idx>=0 ? shared_idx
                                                  : (shared_idx=total_stellar_params++));
                    idx[c][d][p] = gidx;
                }
            }
    }
};

} // namespace specfit