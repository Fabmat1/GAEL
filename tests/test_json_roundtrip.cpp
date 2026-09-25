/*  Round-trip tests for the JSON writers that ship fits to remote workers:
 *  write -> read -> write must reproduce the first file byte for byte
 *  (nlohmann::json orders keys deterministically), and a handful of spot
 *  checks pin the semantics the double-dump trick cannot see.
 */
#include "specfit/GaelAPI.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
namespace api = specfit::api;

static int g_failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " << msg << " (" << #cond << ")\n";          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static std::string slurp(const fs::path& p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static api::FitInput make_fit_input()
{
    api::FitInput fi;
    fi.output_path = "results";

    api::StellarComponentInit c1;
    c1.grid_relative_path = "sdB/processed/";
    c1.teff = 28000; c1.logg = 5.5; c1.he = -2.0; c1.vsini = 7.0;
    c1.freeze_vsini = true; c1.freeze_xi = true;
    c1.sur_ratio = 1.0; c1.freeze_sur_ratio = true;
    c1.abundances["FE"] = -4.5;
    c1.freeze_abundances["FE"] = false;
    c1.abundances["SI"] = -5.0;
    c1.freeze_abundances["SI"] = true;

    api::StellarComponentInit c2 = c1;
    c2.grid_relative_path = "bt-settl/";
    c2.teff = 5000; c2.sur_ratio = 0.3; c2.freeze_sur_ratio = false;
    c2.abundances.clear(); c2.freeze_abundances.clear();

    fi.components = {c1, c2};

    api::ObservationInput oi;
    oi.waveCut = {3600.0, 5250.0};
    oi.ignore  = {{3930.0, 3940.0}, {4855.0, 4870.0}};
    oi.cspline_anchorpoints = {{3600.0, 5250.0, 50.0}};

    api::SpectrumFileInput f1;
    f1.filename = "spec_000.txt";
    f1.spectype = "ASCII_with_2_columns";
    f1.resOffset = 0.5; f1.resSlope = 0.37; f1.barycorr = 12.25;
    f1.airmass = 1.3; f1.pwv = 2.5; f1.fit_telluric = true;

    api::SpectrumFileInput f2 = f1;
    f2.filename = "spec_001.txt";
    f2.airmass = 0.0; f2.fit_telluric = false;
    f2.waveCut = {{3700.0, 5000.0}};
    f2.ignore  = {{{4090.0, 4110.0}}};
    f2.cspline_anchorpoints = {{{3700.0, 5000.0, 40.0}}};

    oi.files = {f1, f2};
    fi.observations = {oi};
    return fi;
}

static api::FitResult make_fit_result()
{
    api::FitResult r;
    r.converged = true;
    r.iterations = 42;
    r.final_chi2 = 1.0625;
    r.n_free_parameters = 29;
    r.n_data_points = 91000;
    r.cont_jitter_K = 6;
    r.status = api::Status::Ok;
    r.warnings = {"one warning"};

    api::ComponentResult c;
    auto p = [](double v, double e, bool fr, int side) {
        api::StellarParamResult q;
        q.value = v; q.error = e; q.frozen = fr;
        q.boundary_side = side; q.at_boundary = side != 0;
        return q;
    };
    c.teff  = {p(28123.5, 150.25, false, 0)};
    c.logg  = {p(5.5, 0.05, false, 0)};
    c.vsini = {p(7.0, 0.0, true, 0)};
    c.zeta  = {p(0.0, 0.0, true, 0)};
    c.xi    = {p(0.0, 0.0, true, 0)};
    c.z     = {p(0.0, 0.0, true, 0)};
    c.he    = {p(-2.05, 0.02, false, -1)};
    c.vrad  = {p(12.5, 0.5, false, 0), p(13.5, 0.5, false, 0)};   // untied
    c.sur_ratio = {p(1.0, 0.0, true, 0)};
    c.abundances["FE"] = {p(-4.5, 0.25, false, 1)};
    c.param_order = {"vrad", "vsini", "zeta", "teff", "logg",
                     "xi", "z", "he", "FE"};
    r.components = {c};

    api::SpectrumResult s;
    s.source_filename = "spec_000.txt";
    s.input_index = 0;
    auto vec = [](std::initializer_list<double> v) {
        specfit::Vector out(static_cast<Eigen::Index>(v.size()));
        Eigen::Index i = 0;
        for (double x : v) out[i++] = x;
        return out;
    };
    s.lambda     = vec({4000.0, 4001.0, 4002.0});
    s.flux       = vec({1.0, 0.98, 0.97});
    s.sigma      = vec({0.01, 0.01, 0.011});
    s.ignoreflag = {1, 1, 0};
    s.model      = vec({0.99, 0.985, 0.972});
    s.continuum  = vec({1.0, 1.0, 1.0});
    s.component_models = {vec({0.99, 0.985, 0.972})};
    s.cont_x     = vec({4000.0, 4002.0});
    s.cont_y     = vec({1.0, 1.0});
    s.telluric   = vec({});
    s.telluric_params = {p(1.3, 0.1, false, 0), p(2.5, 0.4, false, 0),
                         p(12.25, 0.0, true, 0)};
    r.spectra = {s};

    api::SpectrumResult s2 = s;
    s2.source_filename = "spec_001.txt";
    s2.input_index = 2;                    // spectrum 1 was rejected
    s2.telluric_params.clear();
    r.spectra.push_back(s2);

    r.rejected_files   = {"spec_bad.txt"};
    r.rejected_indices = {1};
    r.raw_params        = {12.5, 13.5, 28123.5};
    r.raw_uncertainties = {0.5, 0.5, 150.25};
    r.raw_free_mask     = {true, true, true};
    r.resources.wall_seconds   = 12.25;
    r.resources.cpu_seconds    = 40.5;
    r.resources.peak_rss_bytes = 3221225472LL;   // 3 GiB, past int32
    r.resources.threads        = 4;
    return r;
}

int main()
{
    const fs::path dir =
        fs::temp_directory_path() /
        ("gael_json_roundtrip_" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);

    /* ---- FitInput --------------------------------------------------- */
    {
        const auto a = (dir / "fit_a.json").string();
        const auto b = (dir / "fit_b.json").string();
        api::fit_input_to_json_file(make_fit_input(), a);
        const api::FitInput back = api::fit_input_from_json_file(a);
        api::fit_input_to_json_file(back, b);
        CHECK(slurp(a) == slurp(b), "FitInput write/read/write is stable");

        CHECK(back.components.size() == 2, "both components survive");
        CHECK(back.components[0].abundances.at("SI") == -5.0,
              "abundance value survives");
        CHECK(back.components[0].freeze_abundances.at("SI"),
              "abundance freeze survives");
        CHECK(back.components[1].sur_ratio == 0.3, "sur_ratio survives");
        const auto& f2 = back.observations[0].files[1];
        CHECK(f2.waveCut && (*f2.waveCut)[0] == 3700.0,
              "per-file waveCut survives");
        CHECK(f2.ignore && f2.ignore->size() == 1,
              "per-file ignore survives");
        CHECK(!back.observations[0].files[0].waveCut,
              "absent per-file waveCut stays absent");
        CHECK(f2.airmass == 0.0 && !f2.fit_telluric,
              "telluric switches survive");
    }

    /* ---- GlobalSettings --------------------------------------------- */
    {
        api::GlobalSettings gs;
        gs.base_paths = {"/data/grids", "ssh://gridserver/data/grids"};
        gs.filter_snr = 5.0;
        gs.untie_params = {"vrad", "vsini"};
        gs.cont_jitter_K = 3;
        gs.add_telluric_model = true;

        const auto a = (dir / "gs_a.json").string();
        const auto b = (dir / "gs_b.json").string();
        api::global_settings_to_json_file(gs, a);
        const auto back = api::global_settings_from_json_file(a);
        api::global_settings_to_json_file(back, b);
        CHECK(slurp(a) == slurp(b), "GlobalSettings write/read/write stable");
        CHECK(back.base_paths.size() == 2 &&
                  back.base_paths[1] == "ssh://gridserver/data/grids",
              "scheme base paths survive");
        CHECK(back.untie_params.size() == 2, "untie params survive");
    }

    /* ---- FitResult --------------------------------------------------- */
    {
        const auto a = (dir / "res_a.json").string();
        const auto b = (dir / "res_b.json").string();
        api::fit_result_to_json_file(make_fit_result(), a);
        const api::FitResult back = api::fit_result_from_json_file(a);
        api::fit_result_to_json_file(back, b);
        CHECK(slurp(a) == slurp(b), "FitResult write/read/write is stable");

        CHECK(back.status == api::Status::Ok, "status survives");
        CHECK(back.components[0].vrad.size() == 2, "untied vrad survives");
        CHECK(back.components[0].he[0].boundary_side == -1 &&
                  back.components[0].he[0].at_boundary,
              "boundary side reconstructs at_boundary");
        CHECK(back.components[0].abundances.at("FE")[0].boundary_side == 1,
              "abundance boundary survives");
        CHECK(back.spectra[1].input_index == 2, "input_index survives");
        CHECK(back.spectra[0].telluric_params.size() == 3 &&
                  back.spectra[1].telluric_params.empty(),
              "telluric params survive per spectrum");
        CHECK(back.spectra[0].lambda.size() == 3 &&
                  back.spectra[0].lambda[2] == 4002.0,
              "spectrum arrays survive");
        CHECK(back.rejected_indices == std::vector<int>{1},
              "rejected indices survive");
        CHECK(back.raw_free_mask.size() == 3, "raw mask survives");
        CHECK(back.resources.peak_rss_bytes == 3221225472LL &&
                  back.resources.wall_seconds == 12.25 &&
                  back.resources.cpu_seconds == 40.5 &&
                  back.resources.threads == 4,
              "resources survive");

        /*  Failed results serialize too (the CLI writes them on error).  */
        api::FitResult bad;
        bad.status = api::Status::InternalError;
        bad.error_message = "boom";
        const auto c = (dir / "res_c.json").string();
        api::fit_result_to_json_file(bad, c);
        const auto badBack = api::fit_result_from_json_file(c);
        CHECK(badBack.status == api::Status::InternalError &&
                  badBack.error_message == "boom",
              "failed result round-trips");
        CHECK(badBack.resources.peak_rss_bytes == -1 &&
                  badBack.resources.wall_seconds < 0.0,
              "a result without resources reads back as not measured");
    }

    fs::remove_all(dir);

    if (g_failures) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "json roundtrip test passed\n";
    return 0;
}
