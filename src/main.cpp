#include "specfit/CommonTypes.hpp"
#include "specfit/DiggaAPI.hpp"
#include "specfit/JsonUtils.hpp"
#include "specfit/SpectrumCache.hpp"
#include "specfit/SyntheticModel.hpp"
#include "specfit/UnifiedFitWorkflow.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

#ifdef DIGGA_HAVE_REPORT
#include "specfit/ReportUtils.hpp"
#endif

namespace fs = std::filesystem;
using namespace specfit;
namespace api = specfit::api;

namespace {

/* ------------------------------------------------------------------ */
/*  Reproduce the old ReportUtils fit_parameters.csv straight from     */
/*  the FitResult — no DIGGAreport dependency.                         */
/* ------------------------------------------------------------------ */
void write_fit_parameters_csv(const std::string    &out_dir,
                              const api::FitResult &result) {
    using PMember =
        std::vector<api::StellarParamResult> api::ComponentResult::*;
    static const std::array<std::pair<const char *, PMember>, 8> tags = {{
        {"vrad", &api::ComponentResult::vrad},
        {"vsini", &api::ComponentResult::vsini},
        {"zeta", &api::ComponentResult::zeta},
        {"teff", &api::ComponentResult::teff},
        {"logg", &api::ComponentResult::logg},
        {"xi", &api::ComponentResult::xi},
        {"z", &api::ComponentResult::z},
        {"he", &api::ComponentResult::he},
    }};

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

    /* ---- stellar parameters (tied / untied) ----------------------- */
    std::size_t idx = 0; // flat index into raw_uncertainties
    for (std::size_t c = 0; c < result.components.size(); ++c) {
        const auto &comp = result.components[c];
        for (const auto &[tag, mem] : tags) {
            const auto &vec    = comp.*mem;
            const bool  untied = vec.size() > 1;
            for (std::size_t d = 0; d < vec.size(); ++d, ++idx) {
                std::string name = "c" + std::to_string(c + 1) + "_" + tag;
                if (untied)
                    name += "_d" + std::to_string(d + 1);
                csv << name << ',' << std::setprecision(10) << vec[d].value
                    << ',' << std::setprecision(10) << vec[d].error << '\n';
            }
        }
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

std::string find_global_settings() {
    std::vector<std::string> candidates = {
        "global_settings.json",
        "../global_settings.json",
        "../../global_settings.json",
    };
    try {
        auto exe = fs::canonical("/proc/self/exe").parent_path();
        candidates.insert(candidates.begin(),
                          (exe / "global_settings.json").string());
    } catch (...) {
    }
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
    try {
        cxxopts::Options opts("DIGGA",
                              "Multi-dataset stellar spectrum fitting");
        opts.add_options()("fit", "Fit configuration JSON",
                           cxxopts::value<std::string>())(
            "threads", "Number of threads",
            cxxopts::value<int>()->default_value("0"))(
            "output-synthetic", "Only write undegraded synthetic spectra")(
            "cache-size", "Cache entries",
            cxxopts::value<int>()->default_value("100"))(
            "debug-plots", "Write per-stage debug plots")(
            "no-plots", "Skip per-spectrum summary plots")(
            "cont-jitter",
            "Override contJitterK: refit K times with jittered cspline anchors "
            "and fold the continuum-placement scatter into the reported errors "
            "(default from config, typically 6; 0 disables for faster fitting)",
            cxxopts::value<int>())(
            "no-pdf", "Do not run pdflatex")("h,help", "Show help");
        auto cli = opts.parse(argc, argv);
        if (cli.count("help") || !cli.count("fit")) {
            std::cout << opts.help() << '\n';
            return 0;
        }

        SpectrumCache::instance().set_capacity(cli["cache-size"].as<int>());

        auto gs = api::global_settings_from_json_file(find_global_settings());
        auto fi = api::fit_input_from_json_file(cli["fit"].as<std::string>());

        // CLI override of the continuum-jitter systematic (else use config)
        if (cli.count("cont-jitter"))
            gs.cont_jitter_K = cli["cont-jitter"].as<int>();
        if (cli.count("debug-plots")) {
#ifdef DIGGA_HAVE_REPORT
            gs.debug_plots       = true;
            gs.on_stage_complete = [&](int stage_idx,
                                       const specfit::UnifiedFitWorkflow &wf) {
                namespace fs = std::filesystem;
                fs::create_directories("debug");
                specfit::MultiPanelPlotter P(1.0, false);
                // you'll need access to datasets here — see note below
            };
#else
            std::cerr << "--debug-plots requires DIGGA_BUILD_REPORT=ON\n";
#endif
        }

        if (cli.count("output-synthetic"))
            return run_synthetic_only(fi, gs);

        api::DiggaSession session;
        session.set_global_settings(gs);
        session.set_fit_input(fi);
        session.set_num_threads(cli["threads"].as<int>());
        session.set_log_callback(
            [](const std::string &line) { std::cout << line << '\n'; });

        auto result = session.run();

        /* ---- always emit fit_parameters.csv from the CLI ------------- */
        write_fit_parameters_csv(fi.output_path, result);
        /* diagnostic only: per-spectrum model dump, opt-in via env var */
        if (const char* d = std::getenv("DIGGA_DUMP_MODELS"); d && std::string(d) == "1")
            write_model_dats(fi.output_path + "/models", result);

        std::cout << "\nFit completed successfully.\n"
                  << "  chi2            = " << result.final_chi2 << '\n'
                  << "  iterations      = " << result.iterations << '\n'
                  << "  free parameters = " << result.n_free_parameters << '\n'
                  << "  data points     = " << result.n_data_points << '\n'
                  << "  spectra used    = " << result.spectra.size() << '\n'
                  << "  spectra rejected= " << result.rejected_files.size()
                  << '\n';

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    auto dur = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    std::cout << "Took: " << dur / 3600 << "h " << (dur % 3600) / 60 << "m "
              << dur % 60 << "s\n";
    return 0;
}
