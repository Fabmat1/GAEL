#pragma once
#include "Types.hpp"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace specfit { class UnifiedFitWorkflow; }   // forward decl in parent ns

namespace specfit::api {

enum class Status {
    Ok,
    InvalidInput,        // config was rejected before any work started
    PreprocessingFailed, // every spectrum got filtered/failed to load
    FitFailed,           // workflow threw during the LM/Powell stages
    InternalError        // unexpected C++ exception
};

// ---------- Global settings (one per GaelSession) -----------------------
struct GlobalSettings {
    std::vector<std::string> base_paths;       // model-grid roots

    // reporting / gui-only
    double xrange         = 500.0;

    // spectrum rejection
    double filter_snr     = 0.0;   // reject if SNR <= filter_snr (0 disables)
    double require_blue   = 0.0;   // reject if lambda_min >= require_blue (0 disables)

    // stage behaviour
    bool   auto_freeze_vsini = true;
    std::vector<std::string> untie_params = {"vrad"};

    /*  Fit the Earth's atmosphere as a multiplicative component, as ISIS's
     *  `add_telluric_model` qualifier does.  Needs the ESO transmission
     *  library under <basePath>/telluric/.  Off by default, and pointless
     *  blueward of ~5700 A where there is nothing to model.                */
    bool   add_telluric_model = false;

    /*  Reproduce ISIS's factor-10 slip in the telluric PWV scaling; see
     *  TelluricGrid::set_isis_pwv_scale.  Only for direct comparison with
     *  ISIS -- it puts most of the pwv range outside the library.          */
    bool   telluric_isis_pwv_scale = false;

    /*  Multi-component fits: ISIS's auto_freeze_sur_ratio, which drops a
     *  second grid whose initial surface ratio is already below threshold and
     *  retires a secondary the converged fit cannot detect.  ISIS has this on
     *  by default; here it is opt-in, so that "two grids" always means "fit
     *  two components" unless the config says otherwise.                    */
    bool   auto_freeze_sur_ratio = false;
    double sur_ratio_thres       = 5.0;    // ISIS: sur_ratio_thres
    double c2_detection_thres    = 0.05;   // ISIS: c2_detection_thres

    // iterative-noise / outlier rejection (stage 6)
    int    nit_noise_max  = 5;
    int    nit_fit_max    = 5;
    int    width_box_px   = 50;   // ISIS: width_of_box_filter_in_pixels
    double outlier_sigma_lo = 3.0;
    double outlier_sigma_hi = 3.0;
    double conv_range_lo  = 0.9;
    double conv_range_hi  = 1.1;
    double conv_fraction  = 0.9;

    // Continuum-placement systematic: refit cont_jitter_K times with jittered
    // cspline anchors and fold the stellar-parameter scatter into the reported
    // errors in quadrature.  On by default; set 0 to disable (faster fitting).
    int    cont_jitter_K  = 6;

    bool   verbose        = true;
    bool   debug_plots    = false;

    std::function<void(int stage_index,
        const ::specfit::UnifiedFitWorkflow& wf)>
    on_stage_complete;

    // Special members defined out-of-line in GaelAPI.cpp so that the
    // std::vector / std::function destructors run inside GAELcore's TU.
    GlobalSettings();
    ~GlobalSettings();
    GlobalSettings(const GlobalSettings&);
    GlobalSettings(GlobalSettings&&) noexcept;
    GlobalSettings& operator=(const GlobalSettings&);
    GlobalSettings& operator=(GlobalSettings&&) noexcept;
};

// ---------- Per-fit input -------------------------------------------------
struct StellarComponentInit {
    std::string grid_relative_path;        // e.g. "sdB/processed/"
    double vrad=0, vsini=0, zeta=0;
    double teff=0, logg=0, xi=0, z=0, he=0;
    bool freeze_vrad=false, freeze_vsini=false, freeze_zeta=true;
    bool freeze_teff=false, freeze_logg=false, freeze_xi=true;
    bool freeze_z=false,    freeze_he=false;

    /*  Ratio of this component's effective surface area to component 1's.
     *  Component 1's is 1 and frozen by definition (as in ISIS); the defaults
     *  here are ISIS's stellar_default for cN, N>1.  Only used when more than
     *  one grid is given.                                                    */
    double sur_ratio = 1.0;
    bool   freeze_sur_ratio = false;

    /*  Element abundances, keyed by the grid's own species name ("FE", "SI",
     *  ...) and given as log10 of the fractional particle number, exactly as
     *  ISIS's cN_<ELEMENT> parameters are.
     *
     *  Every element the grid resolves is *modelled* whether or not it
     *  appears here -- leaving it out would drop its lines from the model
     *  entirely -- but only the ones named here with freeze=false are
     *  *fitted*.  Anything absent is frozen at the middle of its grid axis,
     *  which is ISIS's stellar_default value.  (ISIS additionally leaves them
     *  all free; with 24 elements per grid that is 24 extra Jacobian columns,
     *  so GAEL makes fitting opt-in.)
     *
     *  An abundance of 10 or more switches the element off completely, which
     *  is ISIS's convention for excluding one from the model.               */
    std::map<std::string, double> abundances;
    std::map<std::string, bool>   freeze_abundances;
};

struct SpectrumFileInput {
    std::string filename;
    std::string spectype;                  // "ASCII_with_2_columns", ...
    double resOffset = 0.0;
    double resSlope  = 0.0;

    /*  Barycentric correction in km/s.  The spectra GAEL is given are already
     *  corrected, so this does not move the *data*; it seeds the telluric
     *  component's own shift, because the telluric lines sit in the
     *  observatory's frame and the correction moved them by exactly this
     *  much.  ISIS seeds it the same way (spectroscopy_automated.sl ~716).  */
    double barycorr  = 0.0;

    /*  Telluric seeds; ISIS's defaults are airmass 1, pwv 1 mm.  Only used
     *  when settings.addTelluricModel is on.  Setting airmass to 0 switches
     *  the component off for this spectrum, which is what ISIS does for a
     *  spectrum whose tellurics have already been divided out.             */
    double airmass   = 1.0;
    double pwv       = 1.0;
    bool   fit_telluric = true;

    std::optional<std::array<double,2>>                  waveCut;
    std::optional<std::vector<std::array<double,2>>>     ignore;
    std::optional<std::vector<std::array<double,3>>>     cspline_anchorpoints;

    SpectrumFileInput();
    ~SpectrumFileInput();
    SpectrumFileInput(const SpectrumFileInput&);
    SpectrumFileInput(SpectrumFileInput&&) noexcept;
    SpectrumFileInput& operator=(const SpectrumFileInput&);
    SpectrumFileInput& operator=(SpectrumFileInput&&) noexcept;
};

struct ObservationInput {
    std::vector<SpectrumFileInput> files;
    std::array<double,2> waveCut{ -1e300, 1e300 };
    std::vector<std::array<double,2>> ignore;
    std::vector<std::array<double,3>> cspline_anchorpoints;

    ObservationInput();
    ~ObservationInput();
    ObservationInput(const ObservationInput&);
    ObservationInput(ObservationInput&&) noexcept;
    ObservationInput& operator=(const ObservationInput&);
    ObservationInput& operator=(ObservationInput&&) noexcept;
};

struct FitInput {
    std::vector<StellarComponentInit> components;
    std::vector<ObservationInput>     observations;
    std::string                       output_path;

    FitInput();
    ~FitInput();
    FitInput(const FitInput&);
    FitInput(FitInput&&) noexcept;
    FitInput& operator=(const FitInput&);
    FitInput& operator=(FitInput&&) noexcept;
};

// ---------- Results -------------------------------------------------------
struct StellarParamResult {
    double value  = 0.0;
    double error  = 0.0;       // 0 if frozen
    bool   frozen = false;
    bool   at_boundary = false;

    /*  Which side of its allowed range the value is pinned against: -1 at the
     *  lower limit, +1 at the upper one, 0 in the interior.  `at_boundary` is
     *  exactly `boundary_side != 0`; the side is what tells a consumer whether
     *  an abundance ran into the bottom of its grid axis (the line is not
     *  detected -- an upper limit) or the top of it (a lower limit).  Only
     *  free parameters are checked: a frozen one sits where it was put.      */
    int    boundary_side = 0;
    // Trivially destructible — leave implicit.
};

struct ComponentResult {
    // each of these may be 1 (tied) or n_spectra (untied) long
    std::vector<StellarParamResult> vrad, vsini, zeta, teff, logg, xi, z, he,
                                    sur_ratio;

    /*  Element abundances keyed by the grid's species name ("FE", "SI", ...).
     *  Present for every element the grid resolves, frozen ones included.  */
    std::map<std::string, std::vector<StellarParamResult>> abundances;

    /*  This component's parameter names in the order they occupy the global
     *  parameter vector -- which is also the order raw_params /
     *  raw_uncertainties / raw_free_mask are laid out in, so a consumer
     *  walking those flat arrays can stay in step without knowing which
     *  parameters the grid happened to provide.                             */
    std::vector<std::string> param_order;

    /*  Look a parameter up by the name that appears in `param_order`;
     *  nullptr if this component has no such parameter.                     */
    std::vector<StellarParamResult>*       find(const std::string& name);
    const std::vector<StellarParamResult>* find(const std::string& name) const;

    ComponentResult();
    ~ComponentResult();
    ComponentResult(const ComponentResult&);
    ComponentResult(ComponentResult&&) noexcept;
    ComponentResult& operator=(const ComponentResult&);
    ComponentResult& operator=(ComponentResult&&) noexcept;
};

struct SpectrumResult {
    std::string source_filename;

    // rebinned observed spectrum on the Nyquist grid (what the fit saw)
    Vector lambda;
    Vector flux;
    Vector sigma;
    std::vector<int> ignoreflag;   // 1 = used, 0 = ignored

    // synthetic model (continuum * stellar) on the same lambda grid
    Vector model;
    // fitted continuum spline evaluated on the same lambda grid
    Vector continuum;

    /*  Each component's own model on the same lambda grid, in component order:
     *  the fitted continuum (and telluric, when one was fitted) times that
     *  component's normalised flux alone, i.e. what the spectrum would look
     *  like if that component were the only star in it.  A component's lines
     *  therefore appear at full depth here, undiluted by the other's light,
     *  which is what makes the two curves comparable when overplotted.
     *
     *  One entry per component always; for a single-component fit the single
     *  entry equals `model`.                                                 */
    std::vector<Vector> component_models;

    // continuum-spline anchors
    Vector cont_x;
    Vector cont_y;

    /*  Fitted telluric transmission on the same lambda grid, empty when this
     *  spectrum had no telluric component.  `model` already includes it.    */
    Vector telluric;
    // airmass, pwv, barycorr; empty unless the component was fitted
    std::vector<StellarParamResult> telluric_params;

    SpectrumResult();
    ~SpectrumResult();
    SpectrumResult(const SpectrumResult&);
    SpectrumResult(SpectrumResult&&) noexcept;
    SpectrumResult& operator=(const SpectrumResult&);
    SpectrumResult& operator=(SpectrumResult&&) noexcept;
};

struct FitResult {
    bool   converged   = false;
    int    iterations  = 0;
    double final_chi2  = 0.0;
    int    n_free_parameters = 0;
    int    n_data_points     = 0;
    // number of continuum-jitter refits folded into the reported errors (0=off)
    int    cont_jitter_K     = 0;

    std::vector<ComponentResult> components;
    std::vector<SpectrumResult>  spectra;

    // Files rejected during preprocessing (SNR / blue-cut / load failure)
    std::vector<std::string> rejected_files;

    // raw flat parameter vector + uncertainties (for advanced consumers)
    std::vector<double> raw_params;
    std::vector<double> raw_uncertainties;
    std::vector<bool>   raw_free_mask;

    Status                   status = Status::Ok;
    std::string              error_message;     // empty if status == Ok
    std::vector<std::string> warnings;

    FitResult();
    ~FitResult();
    FitResult(const FitResult&);
    FitResult(FitResult&&) noexcept;
    FitResult& operator=(const FitResult&);
    FitResult& operator=(FitResult&&) noexcept;
};

// ---------- The session object --------------------------------------------
class GaelSession {
public:
    GaelSession();
    ~GaelSession();

    GaelSession(const GaelSession&)            = delete;
    GaelSession& operator=(const GaelSession&) = delete;
    GaelSession(GaelSession&&) noexcept;
    GaelSession& operator=(GaelSession&&) noexcept;

    void set_global_settings(const GlobalSettings& gs);
    void set_fit_input      (const FitInput& fi);

    void set_num_threads(int n);      // 0 = hardware_concurrency

    // Optional progress callback: stage name + fractional progress [0,1]
    using ProgressFn = std::function<void(const std::string& stage, double frac)>;
    void set_progress_callback(ProgressFn cb);

    // Optional log-line callback
    using LogFn = std::function<void(const std::string& line)>;
    void set_log_callback(LogFn cb);

    // Runs preprocessing + UnifiedFitWorkflow + collects results.
    // Reports failures via FitResult::status / FitResult::error_message.
    FitResult run();

    // Optional report writer (requires GAELreport).
    void write_report(const FitResult& r, const std::string& out_dir,
                      bool make_plots = true, bool make_pdf = true) const;

private:
    FitResult run_impl();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------- JSON adapters (defined in GaelAPI_json.cpp) ------------------
GlobalSettings global_settings_from_json_file(const std::string& path);
FitInput       fit_input_from_json_file      (const std::string& path);

} // namespace specfit::api