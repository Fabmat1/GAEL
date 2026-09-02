/*  Hermetic unit test for the GridDataProvider hook.
 *
 *  Builds a tiny synthetic HHE grid on disk, then loads it twice: once via a
 *  plain local base path and once through a cp:// provider that materialises
 *  files into a cache directory the way a remote (ssh://) provider would.
 *  The two must be bit-identical, provider misses must only happen once per
 *  file, and load_spectrum must announce its corner files via prefetch().
 */
#include "specfit/GridDataProvider.hpp"
#include "specfit/ModelGrid.hpp"

#include <CCfits/CCfits>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <valarray>
#include <vector>

namespace fs = std::filesystem;
using specfit::ModelGrid;
using specfit::Vector;

static int g_failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " << msg << " (" << #cond << ")\n";          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ *
 *  Synthetic grid: axes t = {10000, 20000}, g = {4, 5}, x/z/HHE pinned.
 *  Four point files whose flux encodes (t, g) so a wrong file is loud.
 * ------------------------------------------------------------------ */
static constexpr int kNlam = 64;

static void write_scalar_table(const std::string& path,
                               const std::vector<std::string>& names,
                               const std::vector<std::vector<double>>& cols,
                               long nrows)
{
    std::vector<std::string> forms(names.size(), "1D");
    std::vector<std::string> units(names.size(), "");
    CCfits::FITS f("!" + path, CCfits::Write);
    CCfits::Table* t = f.addTable("DATA", nrows, names, forms, units);
    for (std::size_t c = 0; c < names.size(); ++c)
        t->column(names[c]).write(cols[c], 1);
}

static std::vector<double> point_flux(double teff, double logg)
{
    std::vector<double> f(kNlam);
    for (int i = 0; i < kNlam; ++i)
        f[static_cast<std::size_t>(i)] = teff * 1e-4 + logg * 0.01 + i * 1e-6;
    return f;
}

static void build_mini_grid(const fs::path& gridDir)
{
    fs::create_directories(gridDir / "HHE");

    /*  grid.fits: one column per axis; vector cells for the varied axes,
     *  scalar cells for the pinned ones.                                   */
    {
        std::vector<std::string> names = {"t", "g", "x", "z", "HHE"};
        std::vector<std::string> forms = {"2D", "2D", "1D", "1D", "1D"};
        std::vector<std::string> units(names.size(), "");
        CCfits::FITS f("!" + (gridDir / "grid.fits").string(), CCfits::Write);
        CCfits::Table* t = f.addTable("AXES", 1, names, forms, units);
        using VA = std::valarray<double>;
        std::vector<VA> tv = {VA{10000.0, 20000.0}};
        std::vector<VA> gv = {VA{4.0, 5.0}};
        t->column("t").writeArrays(tv, 1);
        t->column("g").writeArrays(gv, 1);
        t->column("x").write(std::vector<double>{0.0}, 1);
        t->column("z").write(std::vector<double>{0.0}, 1);
        t->column("HHE").write(std::vector<double>{-1.0}, 1);
    }

    std::vector<double> lam(kNlam);
    for (int i = 0; i < kNlam; ++i)
        lam[static_cast<std::size_t>(i)] = 4000.0 + i * 10.0;
    write_scalar_table((gridDir / "HHE" / "lambda.fits").string(),
                       {"l"}, {lam}, kNlam);

    for (double teff : {10000.0, 20000.0})
        for (double logg : {4.0, 5.0}) {
            char sub[128];
            std::snprintf(sub, sizeof sub, "Z0.00/HE-1.000/X0.00/G%.3f", logg);
            fs::path dir = gridDir / "HHE" / sub;
            fs::create_directories(dir);
            char fn[32];
            std::snprintf(fn, sizeof fn, "T%d.fits", int(teff));
            auto fl = point_flux(teff, logg);
            write_scalar_table((dir / fn).string(), {"f", "c"},
                               {fl, fl}, kNlam);
        }
}

/* ------------------------------------------------------------------ *
 *  cp:// provider: "cp://<abs path>" is served by copying the file into a
 *  cache directory once; repeat requests hit the copy.
 * ------------------------------------------------------------------ */
class CpProvider : public specfit::GridDataProvider {
public:
    CpProvider(fs::path cacheDir) : cache_(std::move(cacheDir)) {}

    bool handles(const std::string& path) const override
    { return path.rfind("cp://", 0) == 0; }

    bool exists(const std::string& vp) override
    { return fs::exists(strip(vp)); }

    std::string localize(const std::string& vp) override
    {
        const fs::path src = strip(vp);
        const fs::path dst = cache_ / src.relative_path();
        std::lock_guard<std::mutex> lk(mtx_);
        if (!fs::exists(dst)) {
            fs::create_directories(dst.parent_path());
            fs::copy_file(src, dst);
            ++copies;
        }
        return dst.string();
    }

    void prefetch(const std::vector<std::string>& paths) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (paths.size() > maxBatch) maxBatch = paths.size();
        ++prefetches;
    }

    int         copies     = 0;
    int         prefetches = 0;
    std::size_t maxBatch   = 0;

private:
    static fs::path strip(const std::string& vp)
    { return fs::path(vp.substr(5)); }

    fs::path   cache_;
    std::mutex mtx_;
};

static bool same_vector(const Vector& a, const Vector& b)
{
    if (a.size() != b.size()) return false;
    for (Eigen::Index i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main()
{
    const fs::path root =
        fs::temp_directory_path() /
        ("gael_grid_provider_test_" + std::to_string(std::random_device{}()));
    const fs::path base  = root / "grids";
    const fs::path cache = root / "cache";
    build_mini_grid(base / "mini");

    /* ---- reference: plain local read ------------------------------ */
    ModelGrid local({base.string()}, "mini");
    const auto ref = local.load_spectrum(15000.0, 4.5, 0.0, -1.0, 0.0,
                                         0.0, 0.0, 0.0);
    CHECK(ref.lambda.size() == kNlam, "local read covers the full grid");

    /* ---- same grid through the provider --------------------------- */
    auto prov = std::make_shared<CpProvider>(cache);
    specfit::set_grid_data_provider(prov);

    ModelGrid remote({"cp://" + base.string()}, "mini");
    const auto got = remote.load_spectrum(15000.0, 4.5, 0.0, -1.0, 0.0,
                                          0.0, 0.0, 0.0);

    CHECK(same_vector(ref.lambda, got.lambda), "lambda identical via provider");
    CHECK(same_vector(ref.flux,   got.flux),   "flux identical via provider");

    /*  grid.fits + lambda.fits + 4 corners = 6 files materialised.     */
    CHECK(prov->copies == 6, "each file transferred exactly once, got "
                                 + std::to_string(prov->copies));
    CHECK(prov->prefetches >= 1, "load_spectrum announced its corners");
    CHECK(prov->maxBatch == 4, "all 4 corner files in one prefetch batch");

    /*  A second read with a different vsini bypasses the RAM cache (new
     *  key) but must be served from the already-localized files.        */
    const int before = prov->copies;
    (void)remote.load_spectrum(15000.0, 4.5, 0.0, -1.0, 0.0,
                               5.0, 0.0, 0.0);
    CHECK(prov->copies == before, "second read hits the disk cache");

    /*  Unhandled (local) paths must never touch the provider.          */
    ModelGrid local2({base.string()}, "mini");
    (void)local2.load_spectrum(12000.0, 4.2, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0);
    CHECK(prov->copies == before, "local reads bypass the provider");

    specfit::set_grid_data_provider(nullptr);
    fs::remove_all(root);

    if (g_failures) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "grid provider test passed\n";
    return 0;
}
