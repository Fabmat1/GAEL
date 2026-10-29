/*  Hermetic stress test: many fits in one process open the same FITS files.
 *
 *  cfitsio shares one FITSfile between every handle a process has open on
 *  the same file name, and that shared state is not safe to touch from two
 *  threads at once.  ASTRA runs fits side by side, so N threads construct a
 *  ModelGrid from the same grid.fits and read the same corner file at the
 *  same moment; that aborted a 12-fit run with "malloc(): unaligned tcache
 *  chunk detected".  Here a few dozen threads do exactly that in a loop.
 *
 *  Note what this does and does not show.  It passed on the code before
 *  FitsLock.hpp too -- 1280 evaluations here and 57 600 constructions on
 *  synthe's real grid.fits without a fault -- so it does not reproduce that
 *  crash, and the lock is not proven to be its fix.  It stays as a guard
 *  that concurrent grid opens and corner reads keep working.
 */
#include "specfit/ModelGrid.hpp"

#include <CCfits/CCfits>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <valarray>
#include <vector>

namespace fs = std::filesystem;
using specfit::ModelGrid;
using specfit::Spectrum;

static constexpr int kNlam = 20000;

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

/*  Two-node t axis, so the grid.fits read covers a vector cell too. */
static void build_grid(const fs::path& dir)
{
    fs::create_directories(dir / "HHE");
    {
        const std::vector<std::string> names = {"t", "g", "x", "z", "HHE"};
        const std::vector<std::string> forms = {"2D", "1D", "1D", "1D", "1D"};
        const std::vector<std::string> units(5, "");
        CCfits::FITS f("!" + (dir / "grid.fits").string(), CCfits::Write);
        CCfits::Table* t = f.addTable("AXES", 1, names, forms, units);
        std::vector<std::valarray<double>> tt = {std::valarray<double>{10000.0, 11000.0}};
        t->column("t").writeArrays(tt, 1);
        t->column("g").write(std::vector<double>{4.0}, 1);
        t->column("x").write(std::vector<double>{0.0}, 1);
        t->column("z").write(std::vector<double>{0.0}, 1);
        t->column("HHE").write(std::vector<double>{-1.0}, 1);
    }
    std::vector<double> lam(kNlam);
    for (int i = 0; i < kNlam; ++i) lam[std::size_t(i)] = 4000.0 + 0.1 * i;
    write_table((dir / "HHE" / "lambda.fits").string(), {"l"}, {lam});

    for (int t : {10000, 11000}) {
        std::vector<double> f(kNlam);
        for (int i = 0; i < kNlam; ++i)
            f[std::size_t(i)] = 1.0 - 0.5 * std::exp(-0.5 * std::pow(
                (lam[std::size_t(i)] - 4861.3) / (t / 2000.0), 2));
        char sub[128];
        std::snprintf(sub, sizeof sub, "HHE/Z0.00/HE-1.000/X0.00/G4.000/T%d.fits", t);
        fs::create_directories((dir / sub).parent_path());
        write_table((dir / sub).string(), {"f"}, {f});
    }
}

/*  test_fits_concurrency <base> <rel>: only construct ModelGrid on an
 *  existing grid, as fast as possible -- for reproducing against a real
 *  grid.fits, which has more HDUs than the one built here.               */
static int hammer_ctor(const std::string& base, const std::string& rel)
{
    constexpr int kThreads = 64, kIter = 300;
    std::atomic<int> bad{0};
    std::vector<std::thread> pool;
    for (int k = 0; k < kThreads; ++k)
        pool.emplace_back([&] {
            for (int it = 0; it < kIter; ++it) {
                try {
                    ModelGrid g({base}, rel);
                    if (g.axes().empty()) ++bad;
                } catch (const std::exception& e) {
                    if (bad++ == 0) std::cout << "FAIL: " << e.what() << '\n';
                }
            }
        });
    for (auto& t : pool) t.join();
    std::cout << (bad ? "FAIL: " : "ok: ") << bad << " bad of "
              << kThreads * kIter << " constructions\n";
    return bad ? 1 : 0;
}

int main(int argc, char** argv)
{
    if (argc == 3) return hammer_ctor(argv[1], argv[2]);

    const fs::path root = fs::temp_directory_path() /
        ("gael_fits_concurrency_" + std::to_string(std::random_device{}()));
    build_grid(root / "grids" / "g");

    constexpr int kThreads = 32, kIter = 40;
    std::atomic<int> bad{0};
    std::vector<std::thread> pool;
    for (int k = 0; k < kThreads; ++k)
        pool.emplace_back([&, k] {
            for (int it = 0; it < kIter; ++it) {
                try {
                    ModelGrid g({(root / "grids").string()}, "g");
                    /*  A vsini unique to this call misses every cache, so
                     *  each call reads both corner files again.         */
                    const double vsini = 1.0 + k + 1e-3 * it;
                    const Spectrum s = g.load_spectrum(10500, 4.0, 0.0, -1.0,
                                                       0.0, vsini, 0.0, 0.0);
                    if (s.flux.size() != kNlam || !std::isfinite(s.flux.sum()))
                        ++bad;
                } catch (const std::exception& e) {
                    if (bad++ == 0) std::cout << "FAIL: " << e.what() << '\n';
                }
            }
        });
    for (auto& t : pool) t.join();

    std::error_code ec;
    fs::remove_all(root, ec);
    if (bad) {
        std::cout << bad << " of " << kThreads * kIter << " evaluations failed\n";
        return 1;
    }
    std::cout << "ok: " << kThreads * kIter << " concurrent grid opens and reads\n";
    return 0;
}
