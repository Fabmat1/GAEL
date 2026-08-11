#pragma once
/* ------------------------------------------------------------------------- *
 *  Telluric transmission as a fitted multiplicative component.
 *
 *  ISIS models the Earth's atmosphere with a separate fit function -- the
 *  model is `cspline * stellar * telluric` (initialize_telluric.sl) -- which
 *  interpolates a pre-computed library of 45 transmission spectra over
 *  airmass and precipitable water vapour, degrades it to the instrument's
 *  resolution, shifts it by the barycentric correction and rebins it onto the
 *  data.  The library is ESO's (Moehler et al. 2014, A&A 568, A9), downloaded
 *  from ftp://ftp.eso.org/pub/dfs/pipelines/skytools/telluric_libs/.
 *
 *  Unlike everything else GAEL fits, the telluric parameters belong to a
 *  *spectrum*, not to a stellar component: two stars observed through the
 *  same air share one atmosphere, and one star observed twice does not.
 * ------------------------------------------------------------------------- */

#include "Types.hpp"
#include "Spectrum.hpp"
#include "SpectrumCache.hpp"

#include <string>
#include <utility>
#include <vector>

namespace specfit {

/*  Vacuum -> air wavelength (Angstroem) for dry air at 15 C, 101.325 kPa and
 *  450 ppm CO2; Ciddor 1996, Applied Optics 35, 1566, eq. 1.  Outside
 *  2300-17000 A the refractive index is held at the edge value, exactly as
 *  ISIS's vacuum_to_air.sl does, rather than extrapolated.                  */
double vacuum_to_air(double lambda_vac);
Vector vacuum_to_air(const Vector& lambda_vac);

/*  The three fitted telluric parameters of one spectrum, in the order they
 *  occupy the global parameter vector.  `res_offset`/`res_slope` are ISIS
 *  parameters too but are frozen at the observation's resolution, so they are
 *  carried on the dataset rather than fitted.                              */
enum class TelluricParam { Airmass = 0, Pwv = 1, Barycorr = 2 };
constexpr int kNTelluricParams = 3;

const char* telluric_param_name(int i);      // "airmass", "pwv", "barycorr"

/*  Solver limits, from ISIS's telluric_default (initialize_telluric.sl). */
std::pair<double,double> telluric_param_limits(int i);

class TelluricGrid {
public:
    /*  First <base>/telluric/ that holds the library; throws if none does. */
    static std::string resolve(const std::vector<std::string>& base_paths);

    explicit TelluricGrid(std::string dir);

    /*  Interpret `pwv` the way ISIS's interpol_telluric.sl does -- scaled by
     *  100 against a library whose nodes are PWV*10, i.e. in tenths of a
     *  millimetre rather than the millimetres it documents.  Off by default;
     *  only useful for reproducing ISIS's numbers, since over most of ISIS's
     *  own allowed range it extrapolates past the library into negative
     *  transmission.  See the comment in transmission().                    */
    void set_isis_pwv_scale(bool on) { isis_pwv_scale_ = on; }
    bool isis_pwv_scale() const { return isis_pwv_scale_; }

    /* ------------------------------------------------------------------ *
     *  Transmission sampled on `lambda_obs`.
     *
     *  `airmass` below 1e-5 returns all ones -- ISIS's way of switching the
     *  component off for a spectrum whose tellurics were already removed.
     *
     *  Two caches sit behind this.  The library spectra are sliced to the
     *  fitted window and degraded to R(lambda) = resOffset + resSlope*lambda
     *  once per (corner, window, resolution); then each corner is shifted and
     *  rebinned onto the observed grid once per (corner, barycorr, grid).
     *  Since rebinning is linear, combining the four corners *after* the
     *  rebin is identical to ISIS combining them before it, and it moves the
     *  per-evaluation work from ~500 000 library points down to the few
     *  thousand of the observed grid -- which matters because a finite
     *  difference re-evaluates this for every free parameter.
     * ------------------------------------------------------------------ */
    Vector transmission(const Vector& lambda_obs,
                        double airmass,
                        double pwv,
                        double barycorr,
                        double resOffset,
                        double resSlope) const;

    /*  Library node values.  Stored scaled to integers because that is how
     *  the file names encode them: airmass*10 and pwv*100.                 */
    static const std::vector<int>& airmass_nodes();
    static const std::vector<int>& pwv_nodes();

    const std::string& directory() const { return dir_; }

private:
    std::string dir_;
    bool        isis_pwv_scale_ = false;

    std::string corner_path(int a10, int w100) const;

    /*  One library spectrum, sliced and degraded, on its native grid. */
    SpectrumPtr corner(int a10, int w100,
                       double lmin, double lmax,
                       double resOffset, double resSlope) const;

    /*  ... the same corner shifted by `barycorr` and rebinned onto `lam`. */
    SpectrumPtr corner_on_grid(int a10, int w100,
                               const Vector& lam,
                               double barycorr,
                               double resOffset, double resSlope) const;
};

} // namespace specfit
