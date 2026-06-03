#include "specfit/DiggaAPI.hpp"
#include "specfit/UnifiedFitWorkflow.hpp"
#include "specfit/CommonTypes.hpp"
#include "specfit/ModelGrid.hpp"
#include "specfit/SpectrumLoaders.hpp"
#include "specfit/NyquistGrid.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "specfit/AkimaSpline.hpp"
#include "specfit/ParameterIndexer.hpp"
#include <Eigen/Core>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <thread>
#include <random>
#include <sstream>
#include <array>
#include <map>

namespace specfit::api {

// ───────────────────────────────────────────────────────────────────────────
// Out-of-line special members for the public types declared in DiggaAPI.hpp.
// MUST live at namespace scope (not inside `namespace { … }`) so the linker
// can resolve the references from other TUs (DiggaBackend.cpp etc.).
// ───────────────────────────────────────────────────────────────────────────
GlobalSettings::GlobalSettings()                                          = default;
GlobalSettings::~GlobalSettings()                                         = default;
GlobalSettings::GlobalSettings(const GlobalSettings&)                     = default;
GlobalSettings::GlobalSettings(GlobalSettings&&) noexcept                 = default;
GlobalSettings& GlobalSettings::operator=(const GlobalSettings&)          = default;
GlobalSettings& GlobalSettings::operator=(GlobalSettings&&) noexcept      = default;

SpectrumFileInput::SpectrumFileInput()                                    = default;
SpectrumFileInput::~SpectrumFileInput()                                   = default;
SpectrumFileInput::SpectrumFileInput(const SpectrumFileInput&)            = default;
SpectrumFileInput::SpectrumFileInput(SpectrumFileInput&&) noexcept        = default;
SpectrumFileInput& SpectrumFileInput::operator=(const SpectrumFileInput&) = default;
SpectrumFileInput& SpectrumFileInput::operator=(SpectrumFileInput&&) noexcept = default;

ObservationInput::ObservationInput()                                      = default;
ObservationInput::~ObservationInput()                                     = default;
ObservationInput::ObservationInput(const ObservationInput&)               = default;
ObservationInput::ObservationInput(ObservationInput&&) noexcept           = default;
ObservationInput& ObservationInput::operator=(const ObservationInput&)    = default;
ObservationInput& ObservationInput::operator=(ObservationInput&&) noexcept = default;

FitInput::FitInput()                                                      = default;
FitInput::~FitInput()                                                     = default;
FitInput::FitInput(const FitInput&)                                       = default;
FitInput::FitInput(FitInput&&) noexcept                                   = default;
FitInput& FitInput::operator=(const FitInput&)                            = default;
FitInput& FitInput::operator=(FitInput&&) noexcept                        = default;

ComponentResult::ComponentResult()                                        = default;
ComponentResult::~ComponentResult()                                       = default;
ComponentResult::ComponentResult(const ComponentResult&)                  = default;
ComponentResult::ComponentResult(ComponentResult&&) noexcept              = default;
ComponentResult& ComponentResult::operator=(const ComponentResult&)       = default;
ComponentResult& ComponentResult::operator=(ComponentResult&&) noexcept   = default;

SpectrumResult::SpectrumResult()                                          = default;
SpectrumResult::~SpectrumResult()                                         = default;
SpectrumResult::SpectrumResult(const SpectrumResult&)                     = default;
SpectrumResult::SpectrumResult(SpectrumResult&&) noexcept                 = default;
SpectrumResult& SpectrumResult::operator=(const SpectrumResult&)          = default;
SpectrumResult& SpectrumResult::operator=(SpectrumResult&&) noexcept      = default;

FitResult::FitResult()                                                    = default;
FitResult::~FitResult()                                                   = default;
FitResult::FitResult(const FitResult&)                                    = default;
FitResult::FitResult(FitResult&&) noexcept                                = default;
FitResult& FitResult::operator=(const FitResult&)                         = default;
FitResult& FitResult::operator=(FitResult&&) noexcept                     = default;

// ───────────────────────────────────────────────────────────────────────────
// Pimpl
// ───────────────────────────────────────────────────────────────────────────
struct DiggaSession::Impl {
    GlobalSettings gs;
    FitInput       fi;
    int            nthreads = 0;
    ProgressFn     progress;
    LogFn          logger;
};

DiggaSession::DiggaSession()  : impl_(std::make_unique<Impl>()) {}
DiggaSession::~DiggaSession() = default;
DiggaSession::DiggaSession(DiggaSession&&) noexcept = default;
DiggaSession& DiggaSession::operator=(DiggaSession&&) noexcept = default;

void DiggaSession::set_global_settings(const GlobalSettings& gs) { impl_->gs = gs; }
void DiggaSession::set_fit_input     (const FitInput& fi)        { impl_->fi = fi; }
void DiggaSession::set_num_threads(int n)                        { impl_->nthreads = n; }
void DiggaSession::set_progress_callback(ProgressFn cb)          { impl_->progress = std::move(cb); }
void DiggaSession::set_log_callback(LogFn cb)                    { impl_->logger  = std::move(cb); }

// ---------------------------------------------------------------------------
namespace {

bool preprocess_one(const SpectrumFileInput& f,
                    const ObservationInput&  obs,
                    const GlobalSettings&    gs,
                    DataSet&                 out,
                    std::string&             reject_reason)
{
    Spectrum raw;
    try { raw = load_spectrum(f.filename, f.spectype); }
    catch (const std::exception& e) {
        reject_reason = std::string("load failed: ") + e.what();
        return false;
    }
    if (raw.lambda.size() == 0) { reject_reason = "empty spectrum"; return false; }

    const double wmin = raw.lambda.minCoeff();

    double snr = 0.0;
    if (gs.filter_snr > 0.0) {
        try { snr = raw.estimate_snr_der().snr; } catch (...) { snr = 0.0; }
        if (snr <= gs.filter_snr) {
            reject_reason = "SNR=" + std::to_string(snr) +
                            " <= filter_snr=" + std::to_string(gs.filter_snr);
            return false;
        }
    }
    if (gs.require_blue > 0.0 && wmin >= gs.require_blue) {
        reject_reason = "lambda_min=" + std::to_string(wmin) +
                        " >= require_blue=" + std::to_string(gs.require_blue);
        return false;
    }

    /* ---- experiment knobs (env vars) -------------------------------- *
     *  DIGGA_NQ_EFF = "native"  -> fit on the raw data grid (no rebin),
     *                              like ISIS; also avoids rebinning sigma.
     *  DIGGA_NQ_EFF = <number>  -> use that nq_eff for the Nyquist grid
     *                              (default 2.7).                         */
    const char* nq_env = std::getenv("DIGGA_NQ_EFF");
    Spectrum rb;
    if (nq_env && std::string(nq_env) == "native") {
        rb.lambda = raw.lambda;
        rb.flux   = raw.flux;
        rb.sigma  = raw.sigma;
    } else {
        const double nq_eff = nq_env ? std::atof(nq_env) : 2.7;
        Vector nyq = build_nyquist_grid(raw.lambda.minCoeff(),
                                        raw.lambda.maxCoeff(),
                                        f.resOffset, f.resSlope, nq_eff);
        rb.lambda = nyq;
        rb.flux   = trapezoidal_rebin(raw.lambda, raw.flux,  nyq);
        rb.sigma  = trapezoidal_rebin(raw.lambda, raw.sigma, nyq);
    }

    const auto wcut   = f.waveCut.value_or(obs.waveCut);
    const auto ignore = f.ignore.value_or(obs.ignore);

    std::vector<int> flags(rb.lambda.size(), 1);
    for (Eigen::Index j = 0; j < rb.lambda.size(); ++j) {
        const double wl = rb.lambda[j];
        if (wl < wcut[0] || wl > wcut[1]) { flags[j] = 0; continue; }
        for (const auto& r : ignore)
            if (wl >= r[0] && wl <= r[1]) { flags[j] = 0; break; }
    }
    rb.ignoreflag = flags;

    const auto anchors = f.cspline_anchorpoints.value_or(obs.cspline_anchorpoints);
    std::vector<std::tuple<double,double,double>> intervals;
    intervals.reserve(anchors.size());
    for (const auto& a : anchors)
        intervals.emplace_back(a[0], a[1], a[2]);

    Vector cont_x = anchors_from_intervals(intervals, rb);
    Vector cy_raw = interp_linear(rb.lambda, rb.flux, cont_x);

    out.name      = f.filename;
    out.obs       = std::move(rb);
    out.cont_x    = cont_x;
    out.cont_y.assign(cont_x.size(), 0.0);
    for (Eigen::Index i = 0; i < cont_x.size(); ++i)
        out.cont_y[i] = std::max(cy_raw[i], 1e-6);
    out.resOffset = f.resOffset;
    out.resSlope  = f.resSlope;
    out.keep      = std::move(flags);
    return true;
}

inline void push_param(std::vector<StellarParamResult>& v,
                       const std::vector<double>& all_p,
                       const std::vector<double>& all_err,
                       const std::vector<bool>&   free_mask,
                       int gidx)
{
    StellarParamResult s;
    s.value       = all_p[gidx];
    s.frozen      = !free_mask[gidx];
    s.error       = (gidx < (int)all_err.size()) ? all_err[gidx] : 0.0;
    s.at_boundary = false;                       // set below if applicable
    v.push_back(s);
}

// ---------------------------------------------------------------------------
// Re-seat one dataset's cspline anchors *in place* from the given intervals,
// reusing its already-loaded/rebinned spectrum, and re-seed the heights from
// the data. Used by the continuum-jitter error ensemble so the spectra are
// never reloaded between refits.
// ---------------------------------------------------------------------------
void reanchor(DataSet& ds, const std::vector<std::array<double,3>>& intervals)
{
    std::vector<std::tuple<double,double,double>> iv;
    iv.reserve(intervals.size());
    for (const auto& a : intervals) iv.emplace_back(a[0], a[1], a[2]);
    Vector cx = anchors_from_intervals(iv, ds.obs);
    Vector cy = interp_linear(ds.obs.lambda, ds.obs.flux, cx);
    ds.cont_x = cx;
    ds.cont_y.assign(static_cast<std::size_t>(cx.size()), 0.0);
    for (Eigen::Index i = 0; i < cx.size(); ++i)
        ds.cont_y[static_cast<std::size_t>(i)] = std::max(cy[i], 1e-6);
}

} // anonymous

FitResult DiggaSession::run()
{
    const auto& gs = impl_->gs;
    const auto& fi = impl_->fi;

    int nt = impl_->nthreads > 0
           ? impl_->nthreads
           : static_cast<int>(std::thread::hardware_concurrency());
    omp_set_num_threads(nt);
    Eigen::setNbThreads(nt);

    auto log = [&](const std::string& s){ if (impl_->logger) impl_->logger(s); };

    FitResult R;

    // ---- build grids & initial params ---------------------------------------
    SharedModel model;
    for (const auto& c : fi.components)
        model.grids.emplace_back(gs.base_paths, c.grid_relative_path);

    model.params.resize(fi.components.size());
    std::vector<std::map<std::string,bool>> frozen(fi.components.size());

    for (std::size_t c = 0; c < fi.components.size(); ++c) {
        const auto& ci = fi.components[c];
        auto&       sp = model.params[c];
        sp.vrad=ci.vrad;   sp.vsini=ci.vsini; sp.zeta=ci.zeta;
        sp.teff=ci.teff;   sp.logg=ci.logg;   sp.xi=ci.xi;
        sp.z   =ci.z;      sp.he  =ci.he;

        frozen[c]["vrad"] =ci.freeze_vrad;   frozen[c]["vsini"]=ci.freeze_vsini;
        frozen[c]["zeta"] =ci.freeze_zeta;   frozen[c]["teff"] =ci.freeze_teff;
        frozen[c]["logg"] =ci.freeze_logg;   frozen[c]["xi"]   =ci.freeze_xi;
        frozen[c]["z"]    =ci.freeze_z;      frozen[c]["he"]   =ci.freeze_he;
    }

    // ---- preprocess every file ---------------------------------------------
    std::vector<DataSet> datasets;
    // anchor intervals per accepted dataset (for the continuum-jitter ensemble)
    std::vector<std::vector<std::array<double,3>>> ds_intervals;
    for (const auto& obs : fi.observations) {
        for (const auto& f : obs.files) {
            DataSet ds;
            std::string why;
            if (preprocess_one(f, obs, gs, ds, why)) {
                datasets.push_back(std::move(ds));
                ds_intervals.push_back(
                    f.cspline_anchorpoints.value_or(obs.cspline_anchorpoints));
            } else {
                R.rejected_files.push_back(f.filename);
                log("rejected: " + f.filename + "  (" + why + ")");
            }
        }
    }
    if (datasets.empty())
        throw std::runtime_error("No spectra passed the quality filters");

    // ---- workflow config ----------------------------------------------------
    ::specfit::UnifiedFitWorkflow::Config wcfg;
    wcfg.verbose           = gs.verbose;
    wcfg.debug_plots       = gs.debug_plots;
    wcfg.untie_params      = gs.untie_params;
    wcfg.nit_noise_max     = gs.nit_noise_max;
    wcfg.nit_fit_max       = gs.nit_fit_max;
    wcfg.width_box_px      = gs.width_box_px;
    wcfg.outlier_sigma_lo  = gs.outlier_sigma_lo;
    wcfg.outlier_sigma_hi  = gs.outlier_sigma_hi;
    wcfg.conv_range_lo     = gs.conv_range_lo;
    wcfg.conv_range_hi     = gs.conv_range_hi;
    wcfg.conv_fraction     = gs.conv_fraction;
    wcfg.on_stage_complete = gs.on_stage_complete;


    ::specfit::UnifiedFitWorkflow wf(datasets, model, wcfg, frozen, nt);
    if (impl_->progress) impl_->progress("fitting", 0.0);
    wf.run();
    if (impl_->progress) impl_->progress("fitting", 1.0);

    // ---- raw parameter vector ----------------------------------------------
    R.raw_params        = wf.get_parameters();
    R.raw_uncertainties = wf.get_uncertainties();
    R.raw_free_mask     = wf.get_free_mask();
    R.final_chi2        = wf.get_final_chi2();

    const auto& sum     = wf.get_summary();
    R.converged         = sum.converged;
    R.iterations        = sum.iterations;
    R.n_free_parameters = (int)std::count(R.raw_free_mask.begin(),
                                          R.raw_free_mask.end(), true);

    // ---- components (per-parameter, per-dataset entries) -------------------
    const auto& idx    = wf.get_indexer();
    const int   n_comp = wf.n_components();
    const int   n_ds   = (int)datasets.size();

    auto is_untied = [&](const std::string& n){
        return std::find(gs.untie_params.begin(), gs.untie_params.end(), n)
               != gs.untie_params.end();
    };

    // same parameter ordering as ParameterIndexer (vrad, vsini, zeta, teff, logg, xi, z, he)
    const char* pnames[8] = { "vrad","vsini","zeta","teff","logg","xi","z","he" };

    R.components.assign(n_comp, {});
    for (int c = 0; c < n_comp; ++c) {
        auto& cr = R.components[c];
        std::array<std::vector<StellarParamResult>*, 8> slots = {
            &cr.vrad, &cr.vsini, &cr.zeta, &cr.teff,
            &cr.logg, &cr.xi,    &cr.z,    &cr.he
        };
        for (int p = 0; p < 8; ++p) {
            const int reps = is_untied(pnames[p]) ? n_ds : 1;
            for (int d = 0; d < reps; ++d) {
                const int gidx = idx.get(c, d, p);
                push_param(*slots[p], R.raw_params, R.raw_uncertainties,
                           R.raw_free_mask, gidx);
            }
        }
    }

    // ---- per-spectrum result (rebinned obs + synthetic + continuum) ---------
    const int total_cont        = wf.n_continuum_params();
    const int cont_block_start  = (int)R.raw_params.size() - total_cont;

    int cs = 0, ndata = 0;
    for (std::size_t d = 0; d < datasets.size(); ++d) {
        const auto& ds = datasets[d];

        SpectrumResult S;
        S.source_filename = ds.name;
        S.lambda          = ds.obs.lambda;
        S.flux            = ds.obs.flux;
        S.sigma           = ds.obs.sigma;
        S.ignoreflag      = ds.obs.ignoreflag;

        // full model (continuum * stellar) on the rebinned grid
        S.model = wf.get_model_for_dataset(d);

        Eigen::Map<const Vector> cy(
            R.raw_params.data() + cont_block_start + cs,
            (Eigen::Index)ds.cont_y.size());

        S.continuum = AkimaSpline(ds.cont_x, cy)(ds.obs.lambda);
        S.cont_x    = ds.cont_x;
        S.cont_y    = cy;                     // materialises a copy
        cs += (int)ds.cont_y.size();

        ndata += std::count(ds.obs.ignoreflag.begin(),
                            ds.obs.ignoreflag.end(), 1);
        R.spectra.push_back(std::move(S));
    }
    R.n_data_points = ndata;

    // ---- continuum-placement systematic via anchor-jitter ensemble ---------
    // Warm-started: reuse the already-preprocessed spectra and the converged
    // solution; each refit only re-seats the continuum at jittered anchors and
    // does a short continuum+stellar solve (no progressive stages, no
    // iterative-noise rejection, no Powell). Far cheaper than a full re-fit.
    if (gs.cont_jitter_K > 0 && !R.components.empty()) {
        using Slot = std::vector<StellarParamResult> ComponentResult::*;
        const std::array<Slot,8> slots = {
            &ComponentResult::vrad, &ComponentResult::vsini,
            &ComponentResult::zeta, &ComponentResult::teff,
            &ComponentResult::logg, &ComponentResult::xi,
            &ComponentResult::z,    &ComponentResult::he };

        std::vector<std::array<std::vector<std::vector<double>>,8>>
            acc(R.components.size());
        for (std::size_t c = 0; c < R.components.size(); ++c)
            for (int p = 0; p < 8; ++p)
                acc[c][p].resize((R.components[c].*slots[p]).size());

        const std::vector<StellarParams> warm = model.params;  // converged fit
        const char* pn[8] = {"vrad","vsini","zeta","teff","logg","xi","z","he"};

        // Freeze in the ensemble exactly what the main fit ended frozen on
        // (e.g. vsini auto-frozen in stage 5), so the refits match the solution.
        std::vector<std::map<std::string,bool>> frozen_ens = frozen;
        for (std::size_t c = 0; c < R.components.size() && c < frozen_ens.size(); ++c)
            for (int p = 0; p < 8; ++p) {
                const auto& vec = (R.components[c].*slots[p]);
                if (!vec.empty()) frozen_ens[c][pn[p]] = vec[0].frozen;
            }

        std::mt19937 rng(20260604u);   // fixed seed: reproducible errors
        std::uniform_real_distribution<double> uscale(0.8,1.4), uphase(0.0,1.0);

        log("[cont-jitter] continuum-placement systematic over "
            + std::to_string(gs.cont_jitter_K) + " warm-started refits");

        // silence the workflow's per-stage stdout during the ensemble
        std::ostringstream sink;
        std::streambuf* old = std::cout.rdbuf(sink.rdbuf());
        try {
            for (int k = 0; k < gs.cont_jitter_K; ++k) {
                const double sc = uscale(rng), ph = uphase(rng);
                for (std::size_t d = 0; d < datasets.size(); ++d) {
                    auto iv = ds_intervals[d];
                    for (auto& a : iv) { const double st = a[2]*sc;
                                         a[2] = st; a[0] = a[0] + ph*st; }
                    reanchor(datasets[d], iv);
                }
                model.params = warm;                       // warm start
                ::specfit::UnifiedFitWorkflow jwf(datasets, model, wcfg, frozen_ens, nt);
                jwf.quick_refit();
                const auto p = jwf.get_parameters();
                const auto& ji = jwf.get_indexer();
                auto is_untied = [&](const char* n){
                    return std::find(gs.untie_params.begin(), gs.untie_params.end(),
                                     std::string(n)) != gs.untie_params.end(); };
                for (std::size_t c = 0; c < acc.size(); ++c)
                    for (int pp = 0; pp < 8; ++pp) {
                        const int reps = is_untied(pn[pp]) ? (int)datasets.size() : 1;
                        for (int d = 0; d < reps && (std::size_t)d < acc[c][pp].size(); ++d)
                            acc[c][pp][d].push_back(p[ji.get((int)c,d,pp)]);
                    }
            }
        } catch (...) { std::cout.rdbuf(old); throw; }
        std::cout.rdbuf(old);

        // fold the ensemble scatter into the reported errors (quadrature)
        for (std::size_t c = 0; c < R.components.size(); ++c)
            for (int p = 0; p < 8; ++p)
                for (std::size_t d = 0; d < acc[c][p].size(); ++d) {
                    const auto& v = acc[c][p][d];
                    if (v.size() < 2) continue;
                    double m = 0.0; for (double x : v) m += x; m /= v.size();
                    double s = 0.0; for (double x : v) s += (x-m)*(x-m);
                    s = std::sqrt(s / v.size());
                    auto& pr = (R.components[c].*slots[p])[d];
                    pr.error = std::sqrt(pr.error*pr.error + s*s);
                }
        R.cont_jitter_K = gs.cont_jitter_K;
    }

    return R;
}

void DiggaSession::write_report(const FitResult&,
                                const std::string&,
                                bool, bool) const
{
#ifndef DIGGA_HAVE_REPORT
    throw std::runtime_error(
        "DIGGA was built without the reporting component "
        "(DIGGA_BUILD_REPORT=OFF). Rebuild with the DIGGAreport target "
        "or call generate_results() directly.");
#else
    // Forward to the existing report generator – minimal wrapper.
    // If you want, re-expose specfit::generate_results() here using
    // the raw workflow; however, a clean API-only report can be added
    // later since ASTRA will normally plot results in the GUI itself.
    throw std::runtime_error("write_report() not yet implemented in the API.");
#endif
}

} // namespace specfit::api