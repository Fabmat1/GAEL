/*  Hermetic unit test for holes in a model grid.
 *
 *  Builds small synthetic HHE grids on disk:
 *
 *    full      t = 10000..16000 step 1000, g = 4.0..5.5 step 0.5, every file
 *    gappy     the same plus t = 14500 in grid.fits with no files at all, and
 *              the interior node (t = 12000, g = 4.5) deleted
 *    edgehole  `full` without (10000, 4.0): a hole with no neighbour pair
 *    corrupt   `full` with (13000, 4.5) overwritten by garbage
 *    metal     t x FE cube for one element, with (11000, -4.5) deleted
 *
 *  and checks that a hole is interpolated across by the documented rule,
 *  that the result is continuous, that cells away from a hole are
 *  bit-identical to a complete grid, that each hole is warned about once,
 *  that a fit through a hole completes and reports it, and that a file
 *  which exists but cannot be read -- locally or through a provider -- still
 *  fails, both a direct evaluation and a whole fit.
 */
#include "specfit/GaelAPI.hpp"
#include "specfit/GridDataProvider.hpp"
#include "specfit/ModelGrid.hpp"

#include <CCfits/CCfits>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <valarray>
#include <vector>

namespace fs  = std::filesystem;
namespace api = specfit::api;
using specfit::ModelGrid;
using specfit::Spectrum;
using specfit::Vector;

/*  Failures go to stdout: stderr is captured while holes are being met, to
 *  count the warnings.                                                     */
static int g_failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cout << "FAIL: " << msg << " (" << #cond << ")\n";          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ *
 *  Grid construction
 * ------------------------------------------------------------------ */
static const std::vector<double> kT     = {10000, 11000, 12000, 13000,
                                           14000, 15000, 16000};
static const std::vector<double> kG     = {4.0, 4.5, 5.0, 5.5};
static constexpr double          kLam0  = 3900.0;
static constexpr double          kDLam  = 0.5;
static constexpr int             kNlam  = 3401;          // 3900..5600 A

static std::vector<double> model_lambda()
{
    std::vector<double> l(kNlam);
    for (int i = 0; i < kNlam; ++i) l[std::size_t(i)] = kLam0 + i * kDLam;
    return l;
}

/*  Lines whose depth and width depend non-linearly on t and g, so that
 *  interpolating over a wider interval visibly differs from a real node.  */
static std::vector<double> model_flux(double t, double g)
{
    static const double lines[] = {4101.7, 4340.5, 4471.5,
                                   4861.3, 4921.9, 5015.7};
    std::vector<double> f(kNlam, 1.0);
    for (int i = 0; i < kNlam; ++i) {
        const double lam = kLam0 + i * kDLam;
        for (int k = 0; k < 6; ++k) {
            const bool balmer = (k == 0 || k == 1 || k == 3);
            const double depth =
                balmer ? 0.6 * std::exp(-std::pow((t - 9000.0) / 6000.0, 2))
                       : 0.05 + 0.3 * std::pow((t - 10000.0) / 6000.0, 2);
            const double width = balmer ? 1.5 + 4.0 * (g - 3.5)
                                        : 0.8 + 0.2 * (g - 4.0);
            f[std::size_t(i)] -=
                depth * std::exp(-0.5 * std::pow((lam - lines[k]) / width, 2));
        }
    }
    return f;
}

static void write_table(const std::string& path,
                        const std::vector<std::string>& names,
                        const std::vector<std::vector<double>>& cols)
{
    std::vector<std::string> forms(names.size(), "1D");
    std::vector<std::string> units(names.size(), "");
    CCfits::FITS f("!" + path, CCfits::Write);
    CCfits::Table* t = f.addTable("DATA", long(cols[0].size()), names, forms,
                                  units);
    for (std::size_t c = 0; c < names.size(); ++c)
        t->column(names[c]).write(cols[c], 1);
}

/*  grid.fits: one row, a vector cell per varied axis, a scalar per pinned. */
static void write_grid_fits(
    const fs::path& path,
    const std::vector<std::pair<std::string, std::vector<double>>>& axes)
{
    std::vector<std::string> names, forms, units;
    for (const auto& [n, v] : axes) {
        names.push_back(n);
        forms.push_back(std::to_string(v.size()) + "D");
        units.push_back("");
    }
    CCfits::FITS f("!" + path.string(), CCfits::Write);
    CCfits::Table* t = f.addTable("AXES", 1, names, forms, units);
    for (const auto& [n, v] : axes) {
        if (v.size() == 1) { t->column(n).write(v, 1); continue; }
        std::vector<std::valarray<double>> cell = {
            std::valarray<double>(v.data(), v.size())};
        t->column(n).writeArrays(cell, 1);
    }
}

static fs::path hhe_file(const fs::path& grid, double t, double g)
{
    char sub[128];
    std::snprintf(sub, sizeof sub, "HHE/Z0.00/HE-1.000/X0.00/G%.3f/T%d.fits",
                  g, int(t));
    return grid / sub;
}

/*  An HHE grid whose index lists `t_index` but which only has files for
 *  the values in kT, minus `skip`.                                        */
static void build_hhe_grid(const fs::path& dir,
                           const std::vector<double>& t_index,
                           const std::set<std::pair<double, double>>& skip)
{
    fs::create_directories(dir / "HHE");
    write_grid_fits(dir / "grid.fits", {{"t", t_index},
                                        {"g", kG},
                                        {"x", {0.0}},
                                        {"z", {0.0}},
                                        {"HHE", {-1.0}}});
    write_table((dir / "HHE" / "lambda.fits").string(), {"l"},
                {model_lambda()});
    for (double t : kT)
        for (double g : kG) {
            if (skip.count({t, g})) continue;
            const fs::path p = hhe_file(dir, t, g);
            fs::create_directories(p.parent_path());
            const auto fl = model_flux(t, g);
            write_table(p.string(), {"f", "c"}, {fl, fl});
        }
}

/* ------------------------------------------------------------------ *
 *  Comparisons
 * ------------------------------------------------------------------ */
static bool identical(const Vector& a, const Vector& b)
{
    if (a.size() != b.size()) return false;
    for (Eigen::Index i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

static double max_abs_diff(const Vector& a, const Vector& b)
{
    if (a.size() != b.size()) return 1e300;
    return (a - b).cwiseAbs().maxCoeff();
}

static Spectrum at(const ModelGrid& g, double t, double lg)
{
    return g.load_spectrum(t, lg, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0);
}

static int count_of(const std::string& hay, const std::string& needle)
{
    int n = 0;
    for (std::size_t p = hay.find(needle); p != std::string::npos;
         p = hay.find(needle, p + needle.size()))
        ++n;
    return n;
}

static bool has_gap(const std::vector<specfit::GridGap>& gaps,
                    const std::string& file_suffix, const std::string& axis,
                    double from, double to)
{
    for (const auto& g : gaps)
        if (g.file.size() >= file_suffix.size() &&
            g.file.compare(g.file.size() - file_suffix.size(),
                           file_suffix.size(), file_suffix) == 0 &&
            g.axis == axis && g.from == from && g.to == to)
            return true;
    return false;
}

/* ------------------------------------------------------------------ *
 *  cp://<abs path> provider: copies files into a cache directory, the way a
 *  remote one would.  Counts probes and records prefetch batches; can be
 *  told to fail for a path the way a dropped connection would.
 * ------------------------------------------------------------------ */
class CpProvider : public specfit::GridDataProvider {
public:
    explicit CpProvider(fs::path cacheDir) : cache_(std::move(cacheDir)) {}

    bool handles(const std::string& path) const override
    { return path.rfind("cp://", 0) == 0; }

    bool exists(const std::string& vp) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        ++exists_calls;
        if (!unreachable.empty() && vp.find(unreachable) != std::string::npos)
            throw std::runtime_error("host unreachable");
        return fs::exists(strip(vp));
    }

    std::string localize(const std::string& vp) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!broken.empty() && vp.find(broken) != std::string::npos)
            throw std::runtime_error("connection lost");
        const fs::path src = strip(vp);
        const fs::path dst = cache_ / src.relative_path();
        if (!fs::exists(dst)) {
            fs::create_directories(dst.parent_path());
            fs::copy_file(src, dst);       // throws for a missing source
        }
        return dst.string();
    }

    void prefetch(const std::vector<std::string>& paths) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        last_batch = paths;
    }

    int                      exists_calls = 0;
    std::vector<std::string> last_batch;
    std::string              broken;        // localize() fails for these
    std::string              unreachable;   // exists() fails for these

private:
    static fs::path strip(const std::string& vp) { return vp.substr(5); }
    fs::path   cache_;
    std::mutex mtx_;
};

/* ------------------------------------------------------------------ *
 *  A fit on one synthetic spectrum
 * ------------------------------------------------------------------ */
static fs::path write_observation(const fs::path& dir, const Spectrum& model)
{
    std::mt19937 rng(4242u);
    std::normal_distribution<double> noise(0.0, 0.005);
    const fs::path p = dir / "obs.txt";
    std::ofstream out(p);
    out.precision(10);
    for (double lam = 4000.0; lam < 5500.0; lam += 0.8) {
        const Eigen::Index i =
            Eigen::Index(std::floor((lam - kLam0) / kDLam));
        const double a = (lam - model.lambda[i]) /
                         (model.lambda[i + 1] - model.lambda[i]);
        const double f = (1 - a) * model.flux[i] + a * model.flux[i + 1];
        const double cont = 1.0 + 1e-4 * (lam - 4750.0);
        out << lam << ' ' << f * cont + noise(rng) << ' ' << 0.005 << '\n';
    }
    return p;
}

static api::FitResult run_fit(const fs::path& grids, const std::string& grid,
                              const fs::path& obs, double teff, double logg)
{
    api::GlobalSettings gs;
    gs.base_paths    = {grids.string()};
    gs.verbose       = false;
    gs.cont_jitter_K = 1;

    api::StellarComponentInit c;
    c.grid_relative_path = grid;
    c.teff = teff; c.logg = logg; c.he = -1.0; c.z = 0.0; c.xi = 0.0;
    c.freeze_vsini = true; c.freeze_z = true; c.freeze_he = true;
    c.freeze_xi = true;

    api::SpectrumFileInput f;
    f.filename  = obs.string();
    f.spectype  = "ASCII_with_3_columns";
    f.resOffset = 2000.0;

    api::ObservationInput o;
    o.files                = {f};
    o.waveCut              = {4000.0, 5500.0};
    o.cspline_anchorpoints = {{4000.0, 5500.0, 100.0}};

    api::FitInput fi;
    fi.components   = {c};
    fi.observations = {o};
    fi.output_path  = (grids.parent_path() / "out").string();

    api::GaelSession s;
    s.set_global_settings(gs);
    s.set_fit_input(fi);
    s.set_num_threads(4);       // the Jacobian's columns run in parallel
    return s.run();
}

int main()
{
    const fs::path root =
        fs::temp_directory_path() /
        ("gael_grid_gaps_test_" + std::to_string(std::random_device{}()));
    const fs::path grids = root / "grids";

    std::vector<double> t_gappy = kT;
    t_gappy.push_back(14500.0);
    std::sort(t_gappy.begin(), t_gappy.end());

    build_hhe_grid(grids / "full",     kT,      {});
    build_hhe_grid(grids / "gappy",    t_gappy, {{12000.0, 4.5}});
    build_hhe_grid(grids / "edgehole", kT,      {{10000.0, 4.0}});
    build_hhe_grid(grids / "corrupt",  kT,      {});
    {
        std::ofstream junk(hhe_file(grids / "corrupt", 13000.0, 4.5),
                           std::ios::trunc);
        junk << "this is not a FITS file";
    }

    ModelGrid full ({grids.string()}, "full");
    ModelGrid gappy({grids.string()}, "gappy");

    /*  Holes print their warning on std::cerr; keep it to count them.     */
    std::ostringstream warnings;
    std::streambuf* old_cerr = std::cerr.rdbuf(warnings.rdbuf());

    /* ---- 1. away from the holes: bit-identical, before any is known --- */
    CHECK(identical(at(gappy, 10500, 4.25).flux, at(full, 10500, 4.25).flux),
          "cell away from the holes is bit-identical (no hole known yet)");
    CHECK(!ModelGrid::any_gaps_known(), "no hole is known before one is met");

    /* ---- 2. the deleted interior node --------------------------------- *
     *  Along t the hole is bridged over 2000 K = 2.33 mean t-spacings (the
     *  t axis has 8 entries over 6000 K), along g over 1.0 dex = 2 mean
     *  g-spacings: g is the narrower bridge.                              */
    const Vector n_12_40 = at(full, 12000, 4.0).flux;
    const Vector n_12_50 = at(full, 12000, 5.0).flux;
    const Vector n_13_45 = at(full, 13000, 4.5).flux;
    const Vector n_13_50 = at(full, 13000, 5.0).flux;
    const Vector imputed = 0.5 * n_12_40 + 0.5 * n_12_50;

    CHECK(max_abs_diff(at(gappy, 12000, 4.5).flux, imputed) < 1e-12,
          "the hole itself takes the value interpolated along g");

    const double at_ = 0.3, ag = 0.2;               // (12300, 4.6)
    const Vector expect = (1 - at_) * (1 - ag) * imputed +
                          at_ * (1 - ag) * n_13_45 +
                          (1 - at_) * ag * n_12_50 + at_ * ag * n_13_50;
    const Spectrum in_cell = at(gappy, 12300, 4.6);
    CHECK(max_abs_diff(in_cell.flux, expect) < 1e-12,
          "a cell touching the hole interpolates over the imputed node");
    CHECK(max_abs_diff(in_cell.flux, at(full, 12300, 4.6).flux) > 1e-4,
          "and that differs from the complete grid, so the test has teeth");
    CHECK(ModelGrid::any_gaps_known(), "the hole is now known");

    /* ---- 3. away from the holes: still bit-identical ------------------ */
    for (auto [t, g] : std::vector<std::pair<double, double>>{
             {10500.0, 4.25}, {15500.0, 5.2}, {11000.0, 5.5}, {13600.0, 5.1}})
        CHECK(identical(at(gappy, t, g).flux, at(full, t, g).flux),
              "cell away from the holes is bit-identical at t=" +
                  std::to_string(t) + " g=" + std::to_string(g));

    /* ---- 4. an index value without files is as if it were not listed -- */
    for (double t : {14000.0, 14200.0, 14500.0, 14777.0, 15000.0})
        CHECK(max_abs_diff(at(gappy, t, 4.7).flux, at(full, t, 4.7).flux)
                  < 1e-12,
              "t=14500 without files behaves like no t=14500 node, at t=" +
                  std::to_string(t));

    /* ---- 5. no step anywhere ------------------------------------------ */
    struct Edge { double t0, g0, dt, dg; };
    for (const Edge& e : {Edge{12000, 4.3, 1e-6, 0}, Edge{12000, 4.5, 1e-6, 0},
                          Edge{11700, 4.5, 0, 1e-9}, Edge{12000, 4.5, 0, 1e-9},
                          Edge{14500, 4.7, 1e-6, 0}, Edge{14000, 4.5, 1e-6, 0}})
        CHECK(max_abs_diff(at(gappy, e.t0 - e.dt, e.g0 - e.dg).flux,
                           at(gappy, e.t0 + e.dt, e.g0 + e.dg).flux) < 1e-7,
              "continuous across the cell boundary at t=" +
                  std::to_string(e.t0) + " g=" + std::to_string(e.g0));

    /* ---- 6. what the grid reports ------------------------------------- */
    const auto used = gappy.gaps_used();
    CHECK(has_gap(used, "G4.500/T12000.fits", "g", 4.0, 5.0),
          "the interior hole is reported, bridged along g");
    CHECK(has_gap(used, "G4.500/T14500.fits", "t", 14000.0, 15000.0),
          "a node of the empty index value is reported, bridged along t");
    CHECK(full.gaps_used().empty(), "the complete grid reports nothing");

    /* ---- 7. a hole with no pair of neighbours fails, naming it -------- */
    {
        ModelGrid edge({grids.string()}, "edgehole");
        std::string what;
        try { (void)at(edge, 10200, 4.1); }
        catch (const std::exception& e) { what = e.what(); }
        CHECK(what.find("G4.000/T10000.fits") != std::string::npos &&
                  what.find("cannot be interpolated across") != std::string::npos &&
                  what.find("t=10000") != std::string::npos,
              "an unbridgeable hole fails with the grid point: " + what);
    }

    /* ---- 8. unreadable is not missing --------------------------------- */
    {
        ModelGrid bad({grids.string()}, "corrupt");
        std::string what;
        try { (void)at(bad, 12500, 4.7); }
        catch (const std::exception& e) { what = e.what(); }
        CHECK(what.find("cannot read model file") != std::string::npos &&
                  what.find("T13000.fits") != std::string::npos,
              "a corrupt file fails the evaluation: " + what);
        CHECK(bad.gaps_used().empty(), "and is not bridged as a hole");
    }

    /* ---- 9. through a provider ---------------------------------------- */
    {
        auto prov = std::make_shared<CpProvider>(root / "cache");
        specfit::set_grid_data_provider(prov);

        ModelGrid remote({"cp://" + grids.string()}, "gappy");
        const Spectrum r1 = at(remote, 12300, 4.6);
        CHECK(identical(r1.flux, in_cell.flux),
              "a hole met through a provider gives the local result");

        const int probes = prov->exists_calls;
        (void)at(remote, 12400, 4.7);
        (void)at(remote, 11500, 4.6);
        CHECK(prov->exists_calls == probes,
              "a known hole is not probed again (" +
                  std::to_string(prov->exists_calls - probes) + " probes)");

        bool hole_in_batch = false, substitute_in_batch = false;
        for (const auto& p : prov->last_batch) {
            if (p.find("G4.500/T12000.fits") != std::string::npos)
                hole_in_batch = true;
            if (p.find("G4.000/T12000.fits") != std::string::npos)
                substitute_in_batch = true;
        }
        CHECK(!hole_in_batch && substitute_in_batch,
              "prefetch skips the hole and asks for its substitute");

        /*  The transfer of an existing file fails: an error, not a hole.  */
        prov->broken = "full/HHE/Z0.00/HE-1.000/X0.00/G5.000/T13000.fits";
        ModelGrid remote_full({"cp://" + grids.string()}, "full");
        std::string what;
        try { (void)at(remote_full, 12500, 4.7); }
        catch (const std::exception& e) { what = e.what(); }
        CHECK(what.find("connection lost") != std::string::npos,
              "a failed transfer fails the evaluation: " + what);
        CHECK(remote_full.gaps_used().empty(), "and is not bridged");

        /*  A provider that cannot tell whether a missing file exists: an
         *  error, not a hole.                                             */
        prov->broken.clear();
        prov->unreachable = "edgehole/HHE/Z0.00/HE-1.000/X0.00/G4.000/T10000";
        ModelGrid remote_edge({"cp://" + grids.string()}, "edgehole");
        what.clear();
        try { (void)at(remote_edge, 10200, 4.1); }
        catch (const std::exception& e) { what = e.what(); }
        CHECK(what.find("host unreachable") != std::string::npos,
              "an unknown existence fails the evaluation: " + what);

        specfit::set_grid_data_provider(nullptr);
    }

    /* ---- 10. holes in an element cube --------------------------------- */
    {
        const fs::path dir = grids / "metal";
        const std::vector<double> T3 = {10000, 11000, 12000};
        const std::vector<double> FE = {-5.0, -4.5, -4.0};
        fs::create_directories(dir / "FE");
        write_grid_fits(dir / "grid.fits", {{"t", T3},
                                            {"g", {4.5}},
                                            {"x", {0.0}},
                                            {"z", {0.0}},
                                            {"HHE", {-1.0}},
                                            {"FE", FE}});
        write_table((dir / "FE" / "lambda.fits").string(), {"l"},
                    {model_lambda()});
        auto efile = [&](double t, double a) {
            char sub[160];
            std::snprintf(sub, sizeof sub,
                          "FE/Z0.00/HE-1.000/X0.00/G4.500/T%d/%.3f.fits",
                          int(t), a);
            return dir / sub;
        };
        auto eflux = [&](double t, double a) {
            auto f = model_flux(t, 4.5);
            for (auto& v : f) v = 1.0 - (1.0 - v) * (a + 6.0);
            return f;
        };
        for (double t : T3)
            for (double a : FE) {
                if (t == 11000.0 && a == -4.5) continue;
                fs::create_directories(efile(t, a).parent_path());
                write_table(efile(t, a).string(), {"f"}, {eflux(t, a)});
            }

        ModelGrid metal({grids.string()}, "metal");
        const Vector got =
            metal.load_element_factor(0, 11000, 4.5, 0.0, -1.0, 0.0, -4.5);
        /*  Both bridges span 2 mean spacings: the tie goes to t, which
         *  comes first in the element cube's (directory) axis order.      */
        const auto lo = eflux(10000, -4.5), hi = eflux(12000, -4.5);
        double worst = 0.0;
        for (Eigen::Index i = 0; i < got.size(); ++i)
            worst = std::max(worst, std::abs(got[i] - 0.5 * (lo[std::size_t(i)] +
                                                             hi[std::size_t(i)])));
        CHECK(got.size() == kNlam && worst < 1e-12,
              "an element-cube hole is interpolated along t");
        CHECK(has_gap(metal.gaps_used(), "T11000/-4.500.fits", "t",
                      10000.0, 12000.0),
              "and reported");
    }

    /* ---- 11. a whole fit through the holes ---------------------------- */
    {
        const Spectrum truth = gappy.load_spectrum(12300, 4.6, 0.0, -1.0, 0.0,
                                                   0.0, 2000.0, 0.0);
        const fs::path obs = write_observation(root, truth);

        api::FitResult R;
        std::string    failed;
        try { R = run_fit(grids, "gappy", obs, 12500.0, 4.4); }
        catch (const std::exception& e) { failed = e.what(); }
        CHECK(failed.empty() && R.status == api::Status::Ok,
              "a fit through a hole completes: " + failed);

        bool listed = false;
        for (const auto& g : R.grid_gaps)
            if (g.file.find("G4.500/T12000.fits") != std::string::npos &&
                g.axis == "g" && g.from == 4.0 && g.to == 5.0 &&
                g.point.at("t") == 12000.0)
                listed = true;
        CHECK(listed, "the fit result lists the hole it used");
        CHECK(!R.warnings.empty() && R.warnings.size() == R.grid_gaps.size(),
              "and has one warning line per hole");
        if (!R.components.empty() && !R.components[0].teff.empty())
            CHECK(std::abs(R.components[0].teff[0].value - 12300.0) < 300.0,
                  "and lands near the truth, teff = " +
                      std::to_string(R.components[0].teff[0].value));
    }

    /* ---- 12. a fit that meets a broken file fails, cleanly ------------- *
     *  Started exactly on the node (12000, 4.5), the residuals never touch
     *  t = 13000; only the parallel Jacobian's teff column steps into the
     *  cell with the corrupt file.  That used to std::terminate.          */
    {
        const Spectrum truth = full.load_spectrum(12000, 4.5, 0.0, -1.0, 0.0,
                                                  0.0, 2000.0, 0.0);
        const fs::path obs = write_observation(root, truth);

        std::string failed;
        try {
            const auto R = run_fit(grids, "corrupt", obs, 12000.0, 4.5);
            if (R.status != api::Status::Ok) failed = R.error_message;
        } catch (const std::exception& e) { failed = e.what(); }
        CHECK(failed.find("cannot read model file") != std::string::npos,
              "a fit meeting an unreadable file fails with its name: " +
                  failed);
    }

    std::cerr.rdbuf(old_cerr);

    /* ---- 13. each hole was warned about exactly once ------------------ */
    const std::string w = warnings.str();
    const std::string hole =
        hhe_file(grids / "gappy", 12000.0, 4.5).string();
    CHECK(count_of(w, "model file " + hole + " is missing") == 1,
          "the interior hole is warned about once, got " +
              std::to_string(count_of(w, "model file " + hole + " is missing")));
    for (double g : kG) {
        const std::string empty =
            hhe_file(grids / "gappy", 14500.0, g).string();
        CHECK(count_of(w, "model file " + empty + " is missing") <= 1,
              "a node of the empty index value is warned about at most once");
    }

    fs::remove_all(root);

    if (g_failures) {
        std::cout << g_failures << " check(s) failed\n"
                  << "captured stderr:\n" << w;
        return 1;
    }
    std::cout << "grid gaps test passed\n";
    return 0;
}
