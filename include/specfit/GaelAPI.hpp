#pragma once
#include "Types.hpp"
#include "FitProgress.hpp"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace specfit { class UnifiedFitWorkflow; }   // forward decl in parent ns
namespace specfit { class GridDataProvider;  }   // see GridDataProvider.hpp

namespace specfit::api {

enum class Status {
    Ok,
    InvalidInput,        // config was rejected before any work started
    PreprocessingFailed, // every spectrum got filtered/failed to load
    FitFailed,           // workflow threw during the LM/Powell stages
    InternalError,       // unexpected C++ exception
    Aborted              // the progress callback asked the fit to stop
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

    /*  Scales the Levenberg-Marquardt iteration budget of every fitting stage
     *  (100-200 iterations each at 1.0).  A fit that hits a stage's budget is
     *  reported as not converged; joint fits of many spectra often do.      */
    double lm_iteration_scale = 1.0;

    // Continuum-placement systematic: refit cont_jitter_K times with jittered
    // cspline anchors and fold the stellar-parameter scatter into the reported
    // errors in quadrature.  On by default; set 0 to disable (faster fitting).
    int    cont_jitter_K  = 6;

    bool   verbose        = true;
    bool   debug_plots    = false;

    std::function<void(int stage_index,
        const ::specfit::UnifiedFitWorkflow& wf)>
    on_stage_complete;

    /*  Optional source for grid files that are not on the local filesystem
     *  (base_paths entries with a scheme prefix such as "ssh://...").
     *  GaelSession::run() installs it into the process-wide slot for the
     *  duration of the fit; see GridDataProvider.hpp.                      */
    std::shared_ptr<::specfit::GridDataProvider> grid_provider;

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

    /*  Position of this spectrum in the caller's submission order, counted
     *  over every file of every observation.  Rejected spectra are absent
     *  from `spectra`, so its index there is *not* the submitted position:
     *  map results back with this, never by position.                        */
    int input_index = -1;

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

    /*  Each component's share of the light on the same lambda grid, in
     *  component order: w_k = s_k C_k / sum_j s_j C_j from the surface ratios
     *  and the flux-calibrated model continua -- the weights the composite is
     *  built with.  They sum to 1 at every point, and
     *  sum_k light_fractions[k] * component_models[k] == model, which is what
     *  a consumer needs to draw each component as its contribution.  One
     *  vector of ones for a single-component fit.                            */
    std::vector<Vector> light_fractions;

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

/*  One missing model file a fit's models were interpolated across: the
 *  missing node took the value linear interpolation along `axis` between
 *  the existing nodes `from` and `to` gives it (see "Holes in a grid" in
 *  ModelGrid.cpp).                                                          */
struct GridGapResult {
    int                           component = 0;   // index into `components`
    std::string                   file;            // the missing file
    std::map<std::string, double> point;           // its grid coordinates
    std::string                   axis;            // axis bridged along
    double                        from = 0.0;
    double                        to   = 0.0;

    GridGapResult();
    ~GridGapResult();
    GridGapResult(const GridGapResult&);
    GridGapResult(GridGapResult&&) noexcept;
    GridGapResult& operator=(const GridGapResult&);
    GridGapResult& operator=(GridGapResult&&) noexcept;
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
    // Their submission positions, in the same order as `rejected_files`, so a
    // caller can name the rejected spectra without matching on file names.
    std::vector<int>         rejected_indices;

    /*  Model files missing from a grid that some model of this fit was
     *  interpolated across, each once.  Empty for a grid without holes.  A
     *  fit listed here is not wrong, but part of it rests on a model that was
     *  linearly interpolated over twice the usual node spacing; each also
     *  appears in `warnings` as one line of text.                          */
    std::vector<GridGapResult> grid_gaps;

    // raw flat parameter vector + uncertainties (for advanced consumers)
    std::vector<double> raw_params;
    std::vector<double> raw_uncertainties;
    std::vector<bool>   raw_free_mask;

    Status                   status = Status::Ok;
    std::string              error_message;     // empty if status == Ok
    std::vector<std::string> warnings;

    /*  What the fit cost the process that ran it.  Filled by the CLI, which
     *  is the whole process; a library caller shares its process with other
     *  work, so for it these stay at -1 ("not measured").  Remote schedulers
     *  size memory and time limits from these numbers, which is why they are
     *  measured by the worker itself rather than read from a scheduler's
     *  accounting, which not every cluster keeps.                           */
    struct Resources {
        double    wall_seconds   = -1.0;
        double    cpu_seconds    = -1.0;   // user + system
        long long peak_rss_bytes = -1;
        int       threads        = 0;
    } resources;

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

    /*  Ceiling on the memory GAEL's caches may hold, in bytes; 0 (the
     *  default) sizes it from what the machine has free.
     *
     *  Worth setting from a host application, because the automatic figure is
     *  read from MemAvailable and a GUI that is already holding the same
     *  spectra is invisible to it -- GAEL would size itself as though the
     *  memory the caller is using were free, and the two together would go
     *  past what the machine has.                                           */
    void set_memory_budget(std::size_t bytes);

    /*  Optional progress callback.  Called from the fitting thread, several
     *  times a second while the fit runs: once per spectrum during
     *  preprocessing, once per LM iteration inside every stage, and at each
     *  phase boundary.  `ProgressReport::fraction` is a monotone estimate of
     *  the share of the whole run -- continuum-jitter ensemble included --
     *  that is done; see FitProgress.hpp for how it is arrived at.
     *
     *  Return false to ask the fit to stop.  It unwinds at the next LM
     *  iteration boundary and run() returns a FitResult with
     *  status == Status::Aborted and no parameters.                        */
    using ProgressFn = ::specfit::ProgressFn;
    void set_progress_callback(ProgressFn cb);

    /*  Convenience overload for consumers that only want a label and a
     *  fraction, and never abort.                                          */
    using SimpleProgressFn =
        std::function<void(const std::string& stage, double frac)>;
    void set_progress_callback(SimpleProgressFn cb);

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

/*  Writers, the exact inverse of the readers above: a file written here and
 *  read back yields an equal GlobalSettings / FitInput.  Note the readers
 *  expand ${VAR} references, so values written verbatim must not contain
 *  literal "${" sequences that are not meant as environment references.    */
void global_settings_to_json_file(const GlobalSettings& gs,
                                  const std::string& path);
void fit_input_to_json_file(const FitInput& fi, const std::string& path);

/*  Complete, lossless FitResult (de)serialization for running fits out of
 *  process (defined in GaelAPI_result_json.cpp).  The schema carries
 *  "gael_result_version"; the reader rejects versions it does not know.    */
void      fit_result_to_json_file(const FitResult& r, const std::string& path);
FitResult fit_result_from_json_file(const std::string& path);

} // namespace specfit::api