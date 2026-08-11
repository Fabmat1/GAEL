/* ===================================================================== *
 *  gen_mock_xshooter --  synthetic X-Shooter-like spectra with known truth
 *
 *  mock_data_generator covers the single-component, metal-free case.  The
 *  two features that came after it -- element abundances and more than one
 *  stellar component -- have no generator, and neither can be tested against
 *  ISIS on the grids that ship with GAEL (the sdB grids are HHE-only, and the
 *  metal-bearing Feros grids have no ISIS reference fits cached).  So the
 *  test is a recovery test: build a spectrum from known parameters with
 *  GAEL's own forward model, then check the fit finds them again.
 *
 *  That is weaker than the GAEL-vs-ISIS suites -- it cannot catch an error
 *  shared by the forward model and the fit -- but it does catch everything
 *  between the parameter vector and the residuals: indexing, bounds,
 *  freezing, the Jacobian, and the caches.
 *
 *  Two arms are written per star, mimicking X-Shooter's UVB and VIS:
 *      UVB  3600-5600 A  R = 5400
 *      VIS  5500-9400 A  R = 8900
 *  so the joint multi-spectrum path is exercised too, and (with --telluric)
 *  the VIS arm carries real telluric absorption.
 * ===================================================================== */
#include "specfit/ModelGrid.hpp"
#include "specfit/SyntheticModel.hpp"
#include "specfit/TelluricGrid.hpp"

#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace specfit;
using nlohmann::json;

namespace {

struct Arm {
    const char* name;
    double lo, hi;      // Angstroem
    double R;           // resolving power (res_offset, res_slope = 0)
    double sampling;    // Angstroem per pixel
};

/*  X-Shooter's real pixel scale is ~0.15-0.2 A, i.e. 4-5 pixels per
 *  resolution element.  That oversampling is not cosmetic here: GAEL rebins
 *  the data onto its own Nyquist grid before fitting, so a mock written at
 *  ~1 pixel per resolution element gets bin-averaged a second time, and the
 *  fit absorbs the extra smoothing as rotation -- it inflated a vsini of
 *  12 km/s to 21 km/s (15 sigma) before this was oversampled.             */
const std::vector<Arm> kArms = {
    {"uvb", 3600.0, 5600.0, 5400.0, 0.15},
    {"vis", 5500.0, 9400.0, 8900.0, 0.15},
};

Vector linspace(double lo, double hi, double step)
{
    const Eigen::Index n = static_cast<Eigen::Index>((hi - lo) / step) + 1;
    Vector v(n);
    for (Eigen::Index i = 0; i < n; ++i) v[i] = lo + step * static_cast<double>(i);
    return v;
}

/*  A smooth, obviously-not-flat continuum, so the cspline has real work to
 *  do and cannot hide a model error by absorbing it.                      */
Vector continuum_of(const Vector& lam, double level)
{
    const double l0 = lam[0], l1 = lam[lam.size() - 1];
    Vector c(lam.size());
    for (Eigen::Index i = 0; i < lam.size(); ++i) {
        const double t = (lam[i] - l0) / (l1 - l0);
        c[i] = level * (1.0 + 0.45 * t - 0.30 * t * t);
    }
    return c;
}

} // namespace

int main(int argc, char** argv)
{
    cxxopts::Options opts("gen_mock_xshooter",
        "Synthetic X-Shooter-like spectra with known parameters");
    opts.add_options()
        ("grid",   "Grid path of component 1", cxxopts::value<std::string>()
                       ->default_value("Feros_3/processed/"))
        ("grid2",  "Grid path of component 2 (enables a binary)",
                       cxxopts::value<std::string>()->default_value(""))
        ("base",   "Model-grid base path", cxxopts::value<std::string>()
                       ->default_value("/home/fabian/ISIS_models/"))
        ("teff",   "Component 1 Teff",  cxxopts::value<double>()->default_value("23500"))
        ("logg",   "Component 1 log g", cxxopts::value<double>()->default_value("5.3"))
        ("he",     "Component 1 log n(He)", cxxopts::value<double>()->default_value("-2.0"))
        ("vsini",  "Component 1 vsini", cxxopts::value<double>()->default_value("12"))
        ("vrad",   "Component 1 vrad",  cxxopts::value<double>()->default_value("18"))
        ("teff2",  "Component 2 Teff",  cxxopts::value<double>()->default_value("24500"))
        ("logg2",  "Component 2 log g", cxxopts::value<double>()->default_value("5.0"))
        ("he2",    "Component 2 log n(He)", cxxopts::value<double>()->default_value("-1.3"))
        ("vsini2", "Component 2 vsini", cxxopts::value<double>()->default_value("30"))
        ("vrad2",  "Component 2 vrad",  cxxopts::value<double>()->default_value("-45"))
        ("sur-ratio", "Component 2 surface ratio", cxxopts::value<double>()->default_value("0.6"))
        ("abundance", "Element truth as NAME=value (repeatable)",
                       cxxopts::value<std::vector<std::string>>()->default_value(""))
        ("fit-abundance", "Elements to leave free in the fit config (repeatable)",
                       cxxopts::value<std::vector<std::string>>()->default_value(""))
        ("telluric", "Imprint tellurics on the VIS arm",
                       cxxopts::value<bool>()->default_value("false"))
        ("airmass", "Telluric airmass", cxxopts::value<double>()->default_value("1.45"))
        ("pwv",     "Telluric PWV [mm]", cxxopts::value<double>()->default_value("2.6"))
        ("barycorr","Barycentric correction [km/s]", cxxopts::value<double>()->default_value("14"))
        ("snr",    "Signal-to-noise per pixel", cxxopts::value<double>()->default_value("120"))
        ("seed",   "RNG seed", cxxopts::value<unsigned>()->default_value("20260811"))
        ("o,output", "Output directory", cxxopts::value<std::string>()
                       ->default_value("./mock_xshooter"))
        ("h,help", "Print usage");

    auto cli = opts.parse(argc, argv);
    if (cli.count("help")) { std::cout << opts.help() << '\n'; return 0; }

    const std::string out_dir = cli["output"].as<std::string>();
    fs::create_directories(out_dir);

    const std::vector<std::string> bases = { cli["base"].as<std::string>() };
    const std::string grid1 = cli["grid"].as<std::string>();
    const std::string grid2 = cli["grid2"].as<std::string>();
    const bool binary = !grid2.empty();

    std::vector<ModelGrid> grids;
    grids.emplace_back(bases, grid1);
    if (binary) grids.emplace_back(bases, grid2);

    /* ---- truth parameters ------------------------------------------- */
    std::vector<StellarParams> truth(binary ? 2 : 1);
    truth[0].teff  = cli["teff"].as<double>();
    truth[0].logg  = cli["logg"].as<double>();
    truth[0].he    = cli["he"].as<double>();
    truth[0].vsini = cli["vsini"].as<double>();
    truth[0].vrad  = cli["vrad"].as<double>();
    truth[0].z = 0.0; truth[0].xi = 0.0; truth[0].zeta = 0.0;
    truth[0].sur_ratio = 1.0;
    if (binary) {
        truth[1].teff  = cli["teff2"].as<double>();
        truth[1].logg  = cli["logg2"].as<double>();
        truth[1].he    = cli["he2"].as<double>();
        truth[1].vsini = cli["vsini2"].as<double>();
        truth[1].vrad  = cli["vrad2"].as<double>();
        truth[1].z = 0.0; truth[1].xi = 0.0; truth[1].zeta = 0.0;
        truth[1].sur_ratio = cli["sur-ratio"].as<double>();
    }

    /*  Every element the grid resolves gets a value.  Unless named on the
     *  command line it is switched off (99), so a metal test isolates the
     *  elements it actually cares about instead of fitting 24 at once.    */
    std::map<std::string,double> want;
    for (const auto& s : cli["abundance"].as<std::vector<std::string>>()) {
        const auto eq = s.find('=');
        if (eq == std::string::npos) continue;
        want[s.substr(0, eq)] = std::stod(s.substr(eq + 1));
    }
    for (std::size_t c = 0; c < truth.size(); ++c) {
        const auto& sp = grids[c].species();
        truth[c].abundances.assign(sp.size(), 99.0);
        for (std::size_t s = 0; s < sp.size(); ++s) {
            auto it = want.find(sp[s]);
            if (it != want.end()) truth[c].abundances[s] = it->second;
        }
    }

    /* ---- the fitted range, so the grids can be sliced --------------- */
    double lam_lo = 1e30, lam_hi = -1e30;
    for (const auto& a : kArms) { lam_lo = std::min(lam_lo, a.lo);
                                  lam_hi = std::max(lam_hi, a.hi); }
    for (auto& g : grids) g.set_wavelength_window(lam_lo / 1.007, lam_hi * 1.007);

    std::unique_ptr<TelluricGrid> tell;
    if (cli["telluric"].as<bool>())
        tell = std::make_unique<TelluricGrid>(TelluricGrid::resolve(bases));

    std::mt19937 rng(cli["seed"].as<unsigned>());
    const double snr = cli["snr"].as<double>();

    json files = json::array();

    for (const auto& arm : kArms) {
        const Vector lam = linspace(arm.lo, arm.hi, arm.sampling);

        /*  Composite exactly as MultiDatasetCost builds it: calibrated
         *  fluxes weighted by the surface ratios over summed continua.    */
        Vector model;
        if (!binary) {
            model = compute_synthetic(grids[0], truth[0], lam, arm.R, 0.0).flux;
        } else {
            Vector num = Vector::Zero(lam.size()), den = Vector::Zero(lam.size());
            for (std::size_t c = 0; c < truth.size(); ++c) {
                Spectrum s = compute_synthetic(grids[c], truth[c], lam,
                                               arm.R, 0.0, /*with_continuum=*/true);
                num += truth[c].sur_ratio * s.flux;
                den += truth[c].sur_ratio * s.cont;
            }
            model = Vector(lam.size());
            for (Eigen::Index i = 0; i < lam.size(); ++i)
                model[i] = (den[i] > 0.0) ? num[i] / den[i] : 0.0;
        }

        Vector flux = model.cwiseProduct(continuum_of(lam, 1.0e4));

        const bool arm_has_tell = tell && std::string(arm.name) == "vis";
        if (arm_has_tell)
            flux = flux.cwiseProduct(tell->transmission(
                       lam, cli["airmass"].as<double>(), cli["pwv"].as<double>(),
                       cli["barycorr"].as<double>(), arm.R, 0.0));

        Vector sigma = flux / snr;
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (Eigen::Index i = 0; i < flux.size(); ++i)
            flux[i] += sigma[i] * gauss(rng);

        const std::string path = out_dir + "/spectrum_" + arm.name + ".txt";
        std::ofstream os(path);
        os << std::scientific << std::setprecision(10);
        for (Eigen::Index i = 0; i < lam.size(); ++i)
            os << lam[i] << ' ' << flux[i] << ' ' << sigma[i] << '\n';

        json f;
        f["filename"]  = fs::absolute(path).string();
        f["spectype"]  = "ASCII_with_3_columns";
        f["resOffset"] = arm.R;
        f["resSlope"]  = 0.0;
        f["waveCut"]   = { arm.lo + 5.0, arm.hi - 5.0 };
        if (tell) {
            f["barycorr"]    = arm_has_tell ? cli["barycorr"].as<double>() : 0.0;
            f["fitTelluric"] = arm_has_tell;
            f["airmass"]     = 1.0;     // deliberately wrong seed
            f["pwv"]         = 1.0;
        }
        files.push_back(std::move(f));
    }

    /* ---- fit configuration, seeded away from the truth -------------- */
    json ig;
    /*  Seeded *away* from the truth, but per component.  Two components
     *  started at the same point are a degenerate configuration -- the chi2
     *  gradient is symmetric under swapping them, so neither can separate,
     *  and the fit settles with both at the same temperature.  This is a
     *  recovery test for the forward model and the solver plumbing, not for
     *  the global optimiser, so each component starts in its own basin.    */
    auto seed_component = [&](int c, const StellarParams& t) {
        const std::string p = "c" + std::to_string(c + 1) + "_";
        const double sgn = (c == 0) ? +1.0 : -1.0;
        ig[p + "vrad"]  = {{"value", t.vrad  + sgn * 12.0}, {"freeze", false}};
        ig[p + "vsini"] = {{"value", std::max(1.0, t.vsini * 0.6)},
                                                            {"freeze", false}};
        ig[p + "zeta"]  = {{"value", 0.0},                  {"freeze", true }};
        ig[p + "teff"]  = {{"value", t.teff  + sgn * 900.0},{"freeze", false}};
        ig[p + "logg"]  = {{"value", t.logg  - sgn * 0.12}, {"freeze", false}};
        ig[p + "xi"]    = {{"value", 0.0},                  {"freeze", true }};
        ig[p + "z"]     = {{"value", 0.0},                  {"freeze", true }};
        ig[p + "HE"]    = {{"value", t.he    + sgn * 0.25}, {"freeze", false}};
        if (c > 0)
            ig[p + "sur_ratio"] = {{"value", t.sur_ratio * 1.5}, {"freeze", false}};

        /*  Elements: switched-off ones must stay switched off, or the fit
         *  would model lines the data does not contain.  The ones named with
         *  --fit-abundance start at the middle of the truth's neighbourhood
         *  and are free.                                                   */
        const auto& sp = grids[static_cast<std::size_t>(c)].species();
        const auto  freeit = cli["fit-abundance"].as<std::vector<std::string>>();
        for (std::size_t s = 0; s < sp.size(); ++s) {
            const bool wanted = want.count(sp[s]) > 0;
            const bool freed  = std::find(freeit.begin(), freeit.end(), sp[s])
                                != freeit.end();
            if (!wanted) { ig[p + sp[s]] = {{"value", 99.0}, {"freeze", true}}; }
            else if (freed) {
                const double* ax = nullptr;
                const GridAxis* a = grids[static_cast<std::size_t>(c)].axis(sp[s]);
                (void)ax;
                double start = t.abundances[s];
                if (a && a->values.size() > 1)      // start a whole node away
                    start = 0.5 * (a->values[0] + a->values[a->values.size()-1]);
                ig[p + sp[s]] = {{"value", start}, {"freeze", false}};
            } else {
                ig[p + sp[s]] = {{"value", t.abundances[s]}, {"freeze", true}};
            }
        }
    };
    for (std::size_t c = 0; c < truth.size(); ++c)
        seed_component(static_cast<int>(c), truth[c]);

    json obs;
    obs["files"] = files;
    obs["csplineAnchorpoints"] = json::array({ json::array({3000, 9500, 60}) });

    json cfg;
    cfg["grids"] = binary ? json::array({grid1, grid2}) : json::array({grid1});
    cfg["initialGuess"] = ig;
    cfg["observations"] = json::array({obs});
    cfg["outputPath"] = fs::absolute(out_dir + "/results").string();
    std::ofstream(out_dir + "/input.json") << cfg.dump(2) << '\n';

    /* ---- truth file, for the checker -------------------------------- */
    json tr;
    for (std::size_t c = 0; c < truth.size(); ++c) {
        const std::string p = "c" + std::to_string(c + 1) + "_";
        tr[p + "teff"]  = truth[c].teff;
        tr[p + "logg"]  = truth[c].logg;
        tr[p + "he"]    = truth[c].he;
        tr[p + "vsini"] = truth[c].vsini;
        tr[p + "vrad"]  = truth[c].vrad;
        if (c > 0) tr[p + "sur_ratio"] = truth[c].sur_ratio;
        const auto& sp = grids[c].species();
        for (std::size_t s = 0; s < sp.size(); ++s)
            if (want.count(sp[s])) tr[p + sp[s]] = truth[c].abundances[s];
    }
    if (tell) {
        tr["telluric_airmass"]  = cli["airmass"].as<double>();
        tr["telluric_pwv"]      = cli["pwv"].as<double>();
        tr["telluric_barycorr"] = cli["barycorr"].as<double>();
    }
    std::ofstream(out_dir + "/truth.json") << tr.dump(2) << '\n';

    std::cout << "wrote " << out_dir << " ("
              << (binary ? "2 components" : "1 component")
              << (want.empty() ? "" : ", metals")
              << (tell ? ", telluric" : "") << ")\n";
    return 0;
}
