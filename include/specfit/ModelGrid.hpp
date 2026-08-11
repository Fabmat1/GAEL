#pragma once
#include "Spectrum.hpp"
#include "Types.hpp"
#include "ParameterSpec.hpp"
#include <cstddef>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace specfit {

struct GridAxis {
    std::string name;
    Vector      values;
};

class ModelGrid {
public:
    // Resolve <base_path>/<rel_path>/grid.fits; first hit wins.
    ModelGrid(const std::vector<std::string>& base_paths,
              const std::string& rel_path);
    explicit ModelGrid(std::string abs_path);

    /*  `with_continuum` switches to the representation a multi-component fit
     *  needs (ISIS's `nonorm` path): the returned `flux` is the *calibrated*
     *  surface flux (FITS column "c", broadened) instead of the normalised
     *  one, and `cont` carries the component's continuum, interp(c)/interp(f)
     *  over the hypercube corners.  Single-component fits leave it false and
     *  get exactly what they always got.                                    */
    Spectrum load_spectrum(double teff,
                           double logg,
                           double z,
                           double he,
                           double xi,
                           double vsini,     // Added vsini parameter
                           double resOffset,
                           double resSlope,
                           bool   with_continuum = false) const;

    const std::vector<GridAxis>& axes() const { return axes_; }

    /*  Grid axis by name ("t", "g", "HHE", "FE", ...), or nullptr. */
    const GridAxis* axis(const std::string& name) const
    {
        for (const auto& a : axes_) if (a.name == name) return &a;
        return nullptr;
    }

    /*  Names of the element axes this grid resolves, in grid.fits column
     *  order -- everything that is not one of the five stellar axes
     *  (t, g, x, z, HHE / HE3 / HE4).  Empty for an HHE-only grid, which is
     *  what every grid shipped so far is.  Each name is both the fit
     *  parameter's name (ISIS: `cN_FE`) and the sub-directory the element's
     *  ratio spectra live in.                                               */
    const std::vector<std::string>& species() const { return species_; }

    static bool is_stellar_axis(const std::string& name)
    {
        return name == "t" || name == "g" || name == "x" || name == "z" ||
               name == "HHE" || name == "HE3" || name == "HE4";
    }

    /* ------------------------------------------------------------------ *
     *  Restrict every corner spectrum to the wavelength range that the fit
     *  can actually see.
     *
     *  The sdB grid spans 3000-13218 A in 20309 points; a 3600-5250 A fit
     *  interpolated, rotationally convolved, degraded and rebinned all of
     *  them on every residual evaluation.  ISIS instead slices to the
     *  observed range plus a +-2000 km/s buffer immediately before
     *  convolve_syn (spectroscopic_fitting.sl).  Call this once with the
     *  union of the observed ranges before fitting; the buffer for the
     *  Doppler shift is added here.
     *
     *  The window becomes part of every cache key, so a grid sliced for one
     *  fit can never serve a short spectrum to a later, wider one.
     * ------------------------------------------------------------------ */
    /* ------------------------------------------------------------------ *
     *  One element's line-correction spectrum: the ratio f_S/f_HHE that
     *  ISIS's grids store per element (make_fit_grid.sl divides the metal
     *  flux by the corresponding HHE flux and clips it to exactly 1 where it
     *  lies within 0.999-1.001).  Multiplying these onto the HHE flux is how
     *  ISIS builds a metal-bearing model without computing every combination
     *  of abundances -- see Irrgang et al. 2014, A&A 565, A63.
     *
     *  Interpolated over the same five stellar axes as the HHE cube plus this
     *  element's own abundance axis, and returned *unbroadened* on the
     *  element's native wavelength grid: the product has to be formed before
     *  the convolution, because f_HHE * f_S is not linear and so does not
     *  commute with it the way the corner interpolation does.
     * ------------------------------------------------------------------ */
    Vector load_element_factor(int    species_index,
                               double teff,
                               double logg,
                               double z,
                               double he,
                               double xi,
                               double abundance) const;

    /*  The wavelength grid one species is tabulated on (<S>/lambda.fits),
     *  restricted to the active window.  Species grids differ a lot -- on the
     *  Feros grids HHE has 14 822 points where NI has 135 834 -- which is why
     *  ISIS interpolates everything onto their union before multiplying.    */
    const Vector& species_lambda(int species_index) const;
    const Vector& hhe_lambda() const;

    /*  HHE's grid merged with those of the given species, memoised.
     *
     *  This is the grid the metal product has to be formed on, and it is the
     *  same one for the whole fit: it depends only on which elements are
     *  switched on, never on the parameter values.  Rebuilding it per model
     *  evaluation -- 24 merges into a 600 k-point vector -- was pure repeat
     *  work.                                                                */
    const Vector& union_lambda(const std::vector<int>& species) const;

    void set_wavelength_window(double lambda_min, double lambda_max);

    /*  Hash of the active window; 0 when the whole grid is in use. */
    std::size_t window_key() const { return window_key_; }

    struct ParameterBounds {
        double teff_min = -std::numeric_limits<double>::max();
        double teff_max = std::numeric_limits<double>::max();
        double logg_min = -std::numeric_limits<double>::max();
        double logg_max = std::numeric_limits<double>::max();
        double z_min = -std::numeric_limits<double>::max();
        double z_max = std::numeric_limits<double>::max();
        double he_min = -std::numeric_limits<double>::max();
        double he_max = std::numeric_limits<double>::max();
        double xi_min = -std::numeric_limits<double>::max();
        double xi_max = std::numeric_limits<double>::max();
        
        // Convert to vectors for LM solver (order: teff, logg, z, he, xi)
        std::vector<double> get_lower_bounds() const {
            return {teff_min, logg_min, z_min, he_min, xi_min};
        }
        
        std::vector<double> get_upper_bounds() const {
            return {teff_max, logg_max, z_max, he_max, xi_max};
        }
        
        // Check if parameters are at boundary (within tolerance)
        std::vector<bool> at_lower_boundary(const Eigen::VectorXd& params, 
                                            double tol = 1e-6) const {
            std::vector<bool> at_boundary(5, false);
            if (params.size() >= 5) {
                at_boundary[0] = (params[0] - teff_min) < tol * std::abs(teff_min);
                at_boundary[1] = (params[1] - logg_min) < tol * std::abs(logg_min);
                at_boundary[2] = (params[2] - z_min) < tol * std::abs(z_min);
                at_boundary[3] = (params[3] - he_min) < tol * std::abs(he_min);
                at_boundary[4] = (params[4] - xi_min) < tol * std::abs(xi_min);
            }
            return at_boundary;
        }
        
        std::vector<bool> at_upper_boundary(const Eigen::VectorXd& params,
                                            double tol = 1e-6) const {
            std::vector<bool> at_boundary(5, false);
            if (params.size() >= 5) {
                at_boundary[0] = (teff_max - params[0]) < tol * std::abs(teff_max);
                at_boundary[1] = (logg_max - params[1]) < tol * std::abs(logg_max);
                at_boundary[2] = (z_max - params[2]) < tol * std::abs(z_max);
                at_boundary[3] = (he_max - params[3]) < tol * std::abs(he_max);
                at_boundary[4] = (xi_max - params[4]) < tol * std::abs(xi_max);
            }
            return at_boundary;
        }

        /*  The grid's own limits on one parameter, or nullopt when the grid
         *  says nothing about it (vrad, vsini, zeta, sur_ratio -- those are
         *  fit policy, not grid facts, and live in UnifiedFitWorkflow).
         *
         *  Note the asymmetry for `xi`: an absent axis leaves teff/logg/z/he
         *  at +-DBL_MAX and those are still handed to the solver, while xi
         *  falls back to the caller's default instead.  That is the behaviour
         *  the hard-coded version had; every grid shipped so far carries all
         *  five axes, so the two have never differed in practice.           */
        std::optional<std::pair<double,double>> for_kind(ParamKind k) const
        {
            switch (k) {
                case ParamKind::Teff: return std::make_pair(teff_min, teff_max);
                case ParamKind::Logg: return std::make_pair(logg_min, logg_max);
                case ParamKind::Z:    return std::make_pair(z_min,    z_max);
                case ParamKind::He:   return std::make_pair(he_min,   he_max);
                case ParamKind::Xi:
                    if (xi_min > -1e9) return std::make_pair(xi_min, xi_max);
                    return std::nullopt;
                default: return std::nullopt;
            }
        }
    };

    ParameterBounds get_parameter_bounds() const;

    /*  Most restrictive bounds over several grids -- a binary's components can
     *  sit on grids with different coverage, and a parameter tied across them
     *  has to stay inside both.                                             */
    static ParameterBounds intersect(const std::vector<ParameterBounds>& b);

private:
    std::string              base_;
    std::vector<GridAxis>    axes_;
    std::vector<std::string> species_;

    /*  Per-species wavelength grid, sliced to the active window; filled
     *  lazily because reading 25 of them costs more than most fits need.
     *  Guarded by a file-local mutex in ModelGrid.cpp -- a mutex *member*
     *  would make ModelGrid non-movable, and SharedModel keeps its grids in
     *  a std::vector.                                                      */
    mutable std::vector<Vector> species_lambda_;
    mutable Vector              hhe_lambda_;

    /*  Memoised union grids, keyed by the active-species list. */
    mutable std::vector<std::pair<std::vector<int>, Vector>> union_lambda_;

    double      window_lo_  = -std::numeric_limits<double>::infinity();
    double      window_hi_  =  std::numeric_limits<double>::infinity();
    std::size_t window_key_ = 0;

    Spectrum read_fits(const std::string& path, bool with_continuum) const;
    Spectrum read_element_fits(const std::string& path, int species_index) const;
    Vector   slice_to_window(const Vector& lam) const;
};

} // namespace specfit