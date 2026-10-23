#include "specfit/CommonTypes.hpp"
#include "specfit/GaelAPI.hpp"
#include "specfit/JsonUtils.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/MemoryBudget.hpp"
#include "specfit/Platform.hpp"
#include "specfit/SyntheticModel.hpp"
#include "specfit/UnifiedFitWorkflow.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <process.h>        // _getpid
#else
#include <unistd.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#ifndef GAEL_VERSION_STRING
#define GAEL_VERSION_STRING "unknown"
#endif

#ifdef GAEL_HAVE_REPORT
#include "specfit/ReportUtils.hpp"
#endif

namespace fs = std::filesystem;
using namespace specfit;
namespace api = specfit::api;

namespace {

/* ------------------------------------------------------------------ */
/*  Reproduce the old ReportUtils fit_parameters.csv straight from     */
/*  the FitResult — no GAELreport dependency.                         */
/* ------------------------------------------------------------------ */
void write_fit_parameters_csv(const std::string    &out_dir,
                              const api::FitResult &result) {
    fs::create_directories(out_dir);
    const std::string path = out_dir + "/fit_parameters.csv";
    std::ofstream     csv(path);
    if (!csv) {
        std::cerr << "Warning: could not open " << path << " for writing\n";
        return;
    }

    csv << "parameter,value,error\n";
    csv << "final_chi2," << std::setprecision(10) << result.final_chi2
        << ",0.0\n";

    /* ---- stellar parameters (tied / untied) ----------------------- *
     *  ComponentResult::param_order is the order these occupy the global
     *  parameter vector, so walking it keeps `idx` -- which continues into
     *  the continuum anchors below -- in step with raw_uncertainties.      */
    std::size_t idx = 0; // flat index into raw_uncertainties
    for (std::size_t c = 0; c < result.components.size(); ++c) {
        const auto &comp = result.components[c];
        for (const auto &tag : comp.param_order) {
            const auto *vec = comp.find(tag);
            if (!vec) continue;
            const bool untied = vec->size() > 1;
            for (std::size_t d = 0; d < vec->size(); ++d, ++idx) {
                std::string name = "c" + std::to_string(c + 1) + "_" + tag;
                if (untied)
                    name += "_d" + std::to_string(d + 1);
                csv << name << ',' << std::setprecision(10) << (*vec)[d].value
                    << ',' << std::setprecision(10) << (*vec)[d].error << '\n';
            }
        }
    }

    /* ---- telluric parameters --------------------------------------- *
     *  These sit between the stellar block and the continuum anchors in the
     *  global vector, so they have to be walked here to keep `idx` aligned
     *  with raw_uncertainties.                                             */
    for (const auto &sp : result.spectra) {
        if (sp.telluric_params.empty()) continue;
        const std::string stem = fs::path(sp.source_filename).stem().string();
        for (std::size_t k = 0; k < sp.telluric_params.size(); ++k, ++idx)
            csv << stem << "_" << specfit::telluric_param_name((int)k) << ','
                << std::setprecision(10) << sp.telluric_params[k].value << ','
                << std::setprecision(10) << sp.telluric_params[k].error << '\n';
    }

    /* ---- continuum anchor amplitudes (cont_y, with errors) -------- */
    for (const auto &sp : result.spectra) {
        const std::string stem = fs::path(sp.source_filename).stem().string();
        for (Eigen::Index a = 0; a < sp.cont_y.size(); ++a, ++idx) {
            const double err = (idx < result.raw_uncertainties.size())
                                   ? result.raw_uncertainties[idx]
                                   : 0.0;
            csv << stem << "_cont" << a << ',' << std::setprecision(10)
                << sp.cont_y[a] << ',' << std::setprecision(10) << err << '\n';
        }
    }

    /* ---- continuum anchor positions (cont_x, error = 0) ----------- */
    for (const auto &sp : result.spectra) {
        const std::string stem = fs::path(sp.source_filename).stem().string();
        for (Eigen::Index a = 0; a < sp.cont_x.size(); ++a)
            csv << stem << "_contX" << a << ',' << std::setprecision(10)
                << sp.cont_x[a] << ",0\n";
    }

    std::cout << "Written: " << path << '\n';
}

/* ------------------------------------------------------------------ */
/*  Dump per-spectrum arrays (data, model, continuum) for diagnostics. */
/*  One file per spectrum: <stem>.model.dat with columns               */
/*    lambda  flux  sigma  model  continuum  ignoreflag                */
/*  plus <stem>.anchors.dat with  cont_x  cont_y.                      */
/* ------------------------------------------------------------------ */
void write_model_dats(const std::string &out_dir, const api::FitResult &result) {
    fs::create_directories(out_dir);
    for (const auto &sp : result.spectra) {
        const std::string stem = fs::path(sp.source_filename).stem().string();
        std::ofstream m(out_dir + "/" + stem + ".model.dat");
        m << "# lambda flux sigma model continuum ignoreflag\n";
        m << std::scientific << std::setprecision(8);
        for (Eigen::Index i = 0; i < sp.lambda.size(); ++i)
            m << sp.lambda[i] << ' ' << sp.flux[i] << ' ' << sp.sigma[i] << ' '
              << sp.model[i] << ' ' << sp.continuum[i] << ' '
              << (i < (Eigen::Index)sp.ignoreflag.size() ? sp.ignoreflag[i] : 1)
              << '\n';
        std::ofstream a(out_dir + "/" + stem + ".anchors.dat");
        a << "# cont_x cont_y\n" << std::scientific << std::setprecision(8);
        for (Eigen::Index i = 0; i < sp.cont_x.size(); ++i)
            a << sp.cont_x[i] << ' ' << sp.cont_y[i] << '\n';
    }
    std::cout << "Written model dumps to: " << out_dir << '\n';
}

/*  SIGTERM asks the fit to stop the way the progress callback does: the CLI's
 *  callback returns false once this flips, the LM loop unwinds at the next
 *  iteration boundary, and run() comes back with Status::Aborted -- so a
 *  remote host that kills the worker still gets a well-formed result file. */
std::atomic<bool> g_terminate{false};

/*  Peak resident set and CPU time of this process so far, for the result's
 *  `resources` block.  POSIX only: elsewhere the fields stay "not measured",
 *  which callers already have to handle.  ru_maxrss is in kilobytes on Linux
 *  and the BSDs but in bytes on macOS.                                      */
void measure_process(specfit::api::FitResult::Resources& res)
{
#if defined(__unix__) || defined(__APPLE__)
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return;
#if defined(__APPLE__)
    res.peak_rss_bytes = static_cast<long long>(ru.ru_maxrss);
#else
    res.peak_rss_bytes = static_cast<long long>(ru.ru_maxrss) * 1024LL;
#endif
    auto secs = [](const timeval& t) { return t.tv_sec + t.tv_usec * 1e-6; };
    res.cpu_seconds = secs(ru.ru_utime) + secs(ru.ru_stime);
#else
    (void)res;
#endif
}
extern "C" void gael_on_sigterm(int) { g_terminate.store(true); }

int process_id()
{
#if defined(_WIN32)
    return ::_getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

/*  Machine-readable heartbeat for out-of-process monitoring: a small JSON
 *  file rewritten atomically (tmp + rename), at most once a second while the
 *  fit runs and once more with the final state.  Timestamps are this
 *  machine's clock; a monitor on another machine must compare against this
 *  machine's clock, never its own.                                          */
class StatusFile {
public:
    explicit StatusFile(std::string path) : path_(std::move(path)) {}

    void tick(const specfit::ProgressReport& r)
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mtx_);
        if (now - last_ < std::chrono::seconds(1)) return;
        last_ = now;
        write("running", r.fraction, r.phase, r.detail, r.eta_seconds, -1);
    }

    void finish(const std::string& state, int exit_code,
                const std::string& detail = {})
    {
        std::lock_guard<std::mutex> lk(mtx_);
        write(state, state == "done" ? 1.0 : -1.0, state, detail, -1.0,
              exit_code);
    }

private:
    void write(const std::string& state, double fraction,
               const std::string& stage, const std::string& detail,
               double eta_s, int exit_code)
    {
        nlohmann::json j;
        j["pid"]          = process_id();
        j["state"]        = state;
        j["stage"]        = stage;
        j["detail"]       = detail;
        j["fraction"]     = fraction;
        j["eta_s"]        = eta_s;
        j["updated_unix"] = static_cast<long long>(std::time(nullptr));
        if (exit_code >= 0) j["exit_code"] = exit_code;

        const std::string tmp = path_ + ".tmp";
        {
            std::ofstream out(tmp);
            if (!out) return;              // monitoring must never kill a fit
            out << j.dump() << '\n';
        }
        std::error_code ec;
        fs::rename(tmp, path_, ec);
    }

    std::string path_;
    std::mutex  mtx_;
    std::chrono::steady_clock::time_point last_{};
};

std::string find_global_settings() {
    std::vector<std::string> candidates = {
        "global_settings.json",
        "../global_settings.json",
        "../../global_settings.json",
    };
    if (const auto exe = specfit::executable_path(); !exe.empty())
        candidates.insert(candidates.begin(),
                          (exe.parent_path() / "global_settings.json").string());
    for (const auto &p : candidates)
        if (fs::exists(p)) {
            std::cout << "Loaded config from: " << p << '\n';
            return p;
        }
    throw std::runtime_error("global_settings.json not found");
}

int run_synthetic_only(const api::FitInput &fi, const api::GlobalSettings &gs) {
    fs::create_directories(fi.output_path);
    for (std::size_t c = 0; c < fi.components.size(); ++c) {
        ModelGrid     grid(gs.base_paths, fi.components[c].grid_relative_path);
        StellarParams sp{};
        const auto   &ci    = fi.components[c];
        sp.teff             = ci.teff;
        sp.logg             = ci.logg;
        sp.xi               = ci.xi;
        sp.z                = ci.z;
        sp.he               = ci.he;
        Spectrum      synth = compute_synthetic_pure(grid, sp);
        std::string   out   = fi.output_path + "/synthetic.dat";
        std::ofstream ofs(out);
        ofs << std::scientific << std::setprecision(8);
        for (Eigen::Index i = 0; i < synth.lambda.size(); ++i)
            ofs << synth.lambda[i] << "  " << synth.flux[i] << '\n';
        std::cout << "Written: " << out << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    auto t0 = std::chrono::steady_clock::now();

    /*  Outside the try block so the catch handler can still report through
     *  them: a monitor watching only these files must see every outcome.   */
    std::unique_ptr<StatusFile> status;
    std::string                 result_json_path;

    auto write_result_json = [&](const api::FitResult& r) {
        if (result_json_path.empty()) return;
        const std::string part = result_json_path + ".part";
        api::fit_result_to_json_file(r, part);
        fs::rename(part, result_json_path);
    };

    try {
        cxxopts::Options opts("GAEL",
                              "Multi-dataset stellar spectrum fitting");
        opts.add_options()("fit", "Fit configuration JSON",
                           cxxopts::value<std::string>())(
            "threads", "Number of threads",
            cxxopts::value<int>()->default_value("0"))(
            "output-synthetic", "Only write undegraded synthetic spectra")(
            /*  The old fixed default of 100 entries was thrashing badly once
                the continuum-jitter ensemble is on: a 5-spectrum joint fit
                walks through far more than 100 distinct corner spectra, and
                every eviction costs a FITS read plus a rotational and an
                instrumental convolution.  Measured on a real 5-spectrum case,
                bit-identical output throughout:

                    100 entries  22.1 s   120 MB RSS
                    200 entries  10.0 s   142 MB RSS
                    500 entries  10.1 s   179 MB RSS
                   2000 entries  10.2 s   319 MB RSS

                An entry count is the wrong knob, though: entries range from
                ~40 kB (a synthetic spectrum on the observed grid) to ~490 kB
                (an unsliced model surface spectrum), and both scale with the
                fit window.  The default is therefore a memory budget; 0 here
                means "no entry cap, let the budget decide".

                On the budget: once the solver stopped re-deriving residuals it
                did not need, the working set collapsed and the cache stopped
                being the bottleneck.  Same 5-spectrum case, bit-identical
                output at every setting:

                      8 MiB   1.09 s    94 MB RSS
                     32 MiB   1.13 s   118 MB
                    128 MiB   1.14 s   219 MB
                    512 MiB   1.22 s   601 MB

                Bigger is now mildly *worse* -- more allocator and hash
                pressure for spectra that are never asked for again.  128 MiB
                is several times the largest working set measured while still
                leaving room for a wider wavelength window or more spectra,
                and it sits below what the old 100-entry default peaked at.   */
            "cache-size", "Cache entry cap (0 = use --cache-mem only)",
            cxxopts::value<int>()->default_value("0"))(
            "cache-mem", "Model-spectrum cache budget in MiB",
            cxxopts::value<int>()->default_value("128"))(
            /*  The ceiling on everything GAEL caches, as opposed to
                --cache-mem, which is only the model-spectrum cache.  The two
                weight tables underneath it are each capable of outgrowing it
                by an order of magnitude -- a rotational table on a grid
                sampled as finely as synthe is ~600 MB, and vsini is part of
                its key, so a fit builds a fresh one per solver iteration.
                Left alone this is read from MemAvailable and any cgroup
                limit, which is what a container or a systemd scope will
                actually kill the process at.                                */
            "mem-budget",
            "Ceiling on all GAEL caches, e.g. 4G or 2048M (default: auto from "
            "available RAM)",
            cxxopts::value<std::string>())(
            "debug-plots", "Write per-stage debug plots")(
            "no-plots", "Skip per-spectrum summary plots")(
            "cont-jitter",
            "Override contJitterK: refit K times with jittered cspline anchors "
            "and fold the continuum-placement scatter into the reported errors "
            "(default from config, typically 6; 0 disables for faster fitting)",
            cxxopts::value<int>())(
            "no-pdf", "Do not run pdflatex")(
            "global", "Global settings JSON (default: search near the binary)",
            cxxopts::value<std::string>())(
            "result-json",
            "Write the complete fit result as JSON to this path (written as "
            "<path>.part and renamed, so its existence means it is complete; "
            "also written for failed and aborted fits)",
            cxxopts::value<std::string>())(
            "status-file",
            "Maintain a small machine-readable status JSON at this path "
            "(atomically rewritten while the fit runs; final state done, "
            "failed or aborted)",
            cxxopts::value<std::string>())(
            "version", "Print version and capability flags, then exit")(
            "h,help", "Show help");
        auto cli = opts.parse(argc, argv);
        if (cli.count("version")) {
            std::cout << "GAEL " << GAEL_VERSION_STRING
                      << " api=1 result_json=1 resources=1\n";
            return 0;
        }
        if (cli.count("help") || !cli.count("fit")) {
            std::cout << opts.help() << '\n';
            return 0;
        }

        /*  Before the caches, so that the first thing either of them asks for
         *  is already measured against the right ceiling.                    */
        if (cli.count("mem-budget"))
            MemoryBudget::instance().set_total(
                MemoryBudget::parse_size(cli["mem-budget"].as<std::string>()));

        SpectrumCache::instance().set_capacity(
            std::max(0, cli["cache-size"].as<int>()));
        SpectrumCache::instance().set_memory_budget(
            static_cast<std::size_t>(std::max(1, cli["cache-mem"].as<int>()))
                << 20);

        if (cli.count("status-file"))
            status = std::make_unique<StatusFile>(
                cli["status-file"].as<std::string>());
        if (cli.count("result-json"))
            result_json_path = cli["result-json"].as<std::string>();
        std::signal(SIGTERM, gael_on_sigterm);

        auto gs = api::global_settings_from_json_file(
            cli.count("global") ? cli["global"].as<std::string>()
                                : find_global_settings());
        auto fi = api::fit_input_from_json_file(cli["fit"].as<std::string>());

        // CLI override of the continuum-jitter systematic (else use config)
        if (cli.count("cont-jitter"))
            gs.cont_jitter_K = cli["cont-jitter"].as<int>();
        if (cli.count("debug-plots")) {
#ifdef GAEL_HAVE_REPORT
            gs.debug_plots       = true;
            gs.on_stage_complete = [&](int stage_idx,
                                       const specfit::UnifiedFitWorkflow &wf) {
                namespace fs = std::filesystem;
                fs::create_directories("debug");
                specfit::MultiPanelPlotter P(1.0, false);
                // you'll need access to datasets here — see note below
            };
#else
            std::cerr << "--debug-plots requires GAEL_BUILD_REPORT=ON\n";
#endif
        }

        if (cli.count("output-synthetic"))
            return run_synthetic_only(fi, gs);

        api::GaelSession session;
        session.set_global_settings(gs);
        session.set_fit_input(fi);
        session.set_num_threads(cli["threads"].as<int>());
        session.set_log_callback(
            [](const std::string &line) { std::cout << line << '\n'; });

        /*  GAEL_PROGRESS=1 puts the phase-by-phase progress report on stderr,
         *  one line per update, so it stays out of the piped stdout that the
         *  test harness reads.  It is also how the tracker's calibration gets
         *  checked against a real fit.
         *
         *  The callback is installed whenever anything needs per-iteration
         *  hooks: the stderr report, the status file, or SIGTERM handling
         *  (abort only unwinds at a callback boundary).                     */
        const bool progress_stderr = [] {
            const char *p = std::getenv("GAEL_PROGRESS");
            return p && std::string(p) == "1";
        }();
        if (progress_stderr || status || !result_json_path.empty()) {
            /*  GAEL_PROGRESS_ABORT_AFTER=N exercises the cancellation path
             *  (the callback's return value) from the CLI: the fit stops at
             *  the Nth report and run() comes back with Status::Aborted.   */
            const char *a  = std::getenv("GAEL_PROGRESS_ABORT_AFTER");
            const int abort_after = a ? std::atoi(a) : 0;
            auto seen = std::make_shared<int>(0);

            session.set_progress_callback(
                [abort_after, seen, progress_stderr,
                 &status](const specfit::ProgressReport &r) {
                    if (progress_stderr) {
                        std::cerr << "[progress] " << std::fixed
                                  << std::setw(6) << std::setprecision(2)
                                  << 100.0 * r.fraction << "%  "
                                  << std::setprecision(2) << r.elapsed_seconds
                                  << "s";
                        if (r.eta_seconds >= 0.0)
                            std::cerr << " (eta " << r.eta_seconds << "s)";
                        std::cerr << "  " << r.phase;
                        if (!r.detail.empty()) std::cerr << "  --  " << r.detail;
                        std::cerr << '\n';
                    }
                    if (status) status->tick(r);
                    if (g_terminate.load()) return false;
                    return !(abort_after > 0 && ++*seen >= abort_after);
                });
        }

        const auto run_started = std::chrono::steady_clock::now();
        auto result = session.run();
        result.resources.wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          run_started).count();
        {
            const int t = cli["threads"].as<int>();
            result.resources.threads =
                t > 0 ? t
                      : static_cast<int>(std::thread::hardware_concurrency());
        }
        measure_process(result.resources);

        /*  The result file is written for every outcome, so a remote monitor
         *  can rely on its appearance alone; the status inside says how the
         *  fit actually ended.                                              */
        write_result_json(result);

        if (result.status == api::Status::Aborted) {
            if (status) status->finish("aborted", 2, result.error_message);
            std::cerr << "Aborted: " << result.error_message << '\n';
            return 2;
        }
        if (result.status != api::Status::Ok) {
            if (status) status->finish("failed", 1, result.error_message);
            std::cerr << "Fit failed: " << result.error_message << '\n';
            return 1;
        }
        if (status) status->finish("done", 0);

        /* ---- always emit fit_parameters.csv from the CLI ------------- */
        write_fit_parameters_csv(fi.output_path, result);
        /* diagnostic only: per-spectrum model dump, opt-in via env var */
        if (const char* d = std::getenv("GAEL_DUMP_MODELS"); d && std::string(d) == "1")
            write_model_dats(fi.output_path + "/models", result);

        std::cout << "\nFit completed successfully.\n"
                  << "  chi2            = " << result.final_chi2 << '\n'
                  << "  iterations      = " << result.iterations << '\n'
                  << "  free parameters = " << result.n_free_parameters << '\n'
                  << "  data points     = " << result.n_data_points << '\n'
                  << "  spectra used    = " << result.spectra.size() << '\n'
                  << "  spectra rejected= " << result.rejected_files.size()
                  << '\n';
        for (const auto& w : result.warnings)
            std::cerr << "Warning: " << w << '\n';

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        if (status) status->finish("failed", 1, e.what());
        try {
            api::FitResult r;
            r.status        = api::Status::InternalError;
            r.error_message = e.what();
            write_result_json(r);
        } catch (...) { /* reporting must not mask the original error */ }
        return 1;
    }

    auto dur = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    std::cout << "Took: " << dur / 3600 << "h " << (dur % 3600) / 60 << "m "
              << dur % 60 << "s\n";
    return 0;
}
