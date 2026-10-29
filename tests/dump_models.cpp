/* ------------------------------------------------------------------ *
 *  Write a fixed set of model spectra to a binary file, so two builds can
 *  be compared bit for bit (tests/compare_models.py).
 *
 *  Usage: dump_models <grid_base> <rel> <out.bin> T g z he xi
 *
 *  For each resolution x vsini x {normalised, with_continuum}, and at the
 *  given point plus two shifted ones inside the same cell, writes
 *  lambda, flux and (with_continuum) cont of ModelGrid::load_spectrum and
 *  the Nyquist-rebinned flux compute_synthetic hands the fit.
 * ------------------------------------------------------------------ */
#include "specfit/ModelGrid.hpp"
#include "specfit/NyquistGrid.hpp"
#include "specfit/SyntheticModel.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace specfit;

static void put(std::FILE* f, const Vector& v)
{
    const long n = long(v.size());
    std::fwrite(&n, sizeof n, 1, f);
    std::fwrite(v.data(), sizeof(double), std::size_t(n), f);
}

int main(int argc, char** argv)
{
    if (argc < 9) { std::fprintf(stderr, "usage\n"); return 2; }
    const std::string base = argv[1], rel = argv[2];
    std::FILE* out = std::fopen(argv[3], "wb");
    const double T = std::atof(argv[4]), g = std::atof(argv[5]),
                 z = std::atof(argv[6]), he = std::atof(argv[7]),
                 xi = std::atof(argv[8]);
    const bool cont_ok = argc < 10 || std::string(argv[9]) != "nocont";

    ModelGrid grid({base}, rel);
    grid.set_wavelength_window(3625.0 / 1.007, 9100.0 * 1.007);

    struct Res { double ro, rs; };
    const std::vector<Res>    res = { {1800, 0}, {7500, 0}, {0, 0.455} };
    const std::vector<double> vs  = { 0, 2, 10, 30, 60, 100, 140 };
    const std::vector<double> dT  = { 0, 37, 81 };

    for (const auto& r : res) {
        const Vector obs = build_nyquist_grid(3650, 9080, r.ro, r.rs, 2.7);
        for (double v : vs)
            for (double d : dT)
                for (int wc = 0; wc < (cont_ok ? 2 : 1); ++wc) {
                    const Spectrum s = grid.load_spectrum(T + d, g + 0.001 * d,
                                                          z, he, xi, v,
                                                          r.ro, r.rs, wc != 0);
                    put(out, s.lambda);
                    put(out, s.flux);
                    if (wc) put(out, s.cont);

                    StellarParams p;
                    p.teff = T + d; p.logg = g + 0.001 * d; p.z = z;
                    p.he = he; p.xi = xi; p.vsini = v; p.vrad = 12.3; p.zeta = 0.0;
                    const Spectrum m = compute_synthetic(grid, p, obs, r.ro,
                                                         r.rs, wc != 0);
                    put(out, m.flux);
                    if (wc) put(out, m.cont);
                }
    }
    std::fclose(out);
    return 0;
}
