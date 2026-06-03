#pragma once
#include "Types.hpp"
#include <array>
#include <functional>
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

// ---------- Global settings (one per DiggaSession) -----------------------
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

    // iterative-noise / outlier rejection (stage 6)
    int    nit_noise_max  = 5;
    int    nit_fit_max    = 5;
    int    width_box_px   = 5;
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

    // Special members defined out-of-line in DiggaAPI.cpp so that the
    // std::vector / std::function destructors run inside DIGGAcore's TU.
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
    // Trivially destructible (only POD + std::string). Implicit special
    // members are fine.
};

struct SpectrumFileInput {
    std::string filename;
    std::string spectype;                  // "ASCII_with_2_columns", ...
    double resOffset = 0.0;
    double resSlope  = 0.0;
    double barycorr  = 0.0;

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
    // Trivially destructible — leave implicit.
};

struct ComponentResult {
    // each of these may be 1 (tied) or n_spectra (untied) long
    std::vector<StellarParamResult> vrad, vsini, zeta, teff, logg, xi, z, he;

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

    // continuum-spline anchors
    Vector cont_x;
    Vector cont_y;

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
class DiggaSession {
public:
    DiggaSession();
    ~DiggaSession();

    DiggaSession(const DiggaSession&)            = delete;
    DiggaSession& operator=(const DiggaSession&) = delete;
    DiggaSession(DiggaSession&&) noexcept;
    DiggaSession& operator=(DiggaSession&&) noexcept;

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

    // Optional report writer (requires DIGGAreport).
    void write_report(const FitResult& r, const std::string& out_dir,
                      bool make_plots = true, bool make_pdf = true) const;

private:
    FitResult run_impl();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------- JSON adapters (defined in DiggaAPI_json.cpp) ------------------
GlobalSettings global_settings_from_json_file(const std::string& path);
FitInput       fit_input_from_json_file      (const std::string& path);

} // namespace specfit::api