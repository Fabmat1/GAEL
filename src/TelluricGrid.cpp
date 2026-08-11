#include "specfit/TelluricGrid.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/Resolution.hpp"

#include <CCfits/CCfits>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace specfit {

/* ===================================================================== *
 *  Ciddor 1996 vacuum -> air
 * ===================================================================== */
namespace {

constexpr double kC0 = 2.380185e-06;
constexpr double kC1 = 5.792105e-10;
constexpr double kC2 = 5.7362e-07;
constexpr double kC3 = 1.67917e-11;

/*  Validity range of the fit; ISIS clamps the *divisor* to the edge value
 *  outside it instead of extrapolating a formula that diverges.           */
constexpr double kAirLo = 2300.0;
constexpr double kAirHi = 17000.0;

inline double air_divisor(double l)
{
    const double inv2 = 1.0 / (l * l);
    return kC1 / (kC0 - inv2) + kC3 / (kC2 - inv2) + 1.0;
}

/* ---------- hashing helpers, matching the style used elsewhere -------- */
inline std::size_t hash_combine(std::size_t seed, std::size_t v) noexcept
{
    seed ^= v + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
    return seed;
}
inline std::size_t hash_double(double x) noexcept
{
    std::uint64_t bits; std::memcpy(&bits, &x, sizeof bits);
    return std::hash<std::uint64_t>{}(bits);
}
inline std::size_t hash_string(const std::string& s) noexcept
{
    return std::hash<std::string>{}(s);
}

/* ------------------------------------------------------------------ *
 *  Bracketing node pair, reproducing ISIS's telluric_get_min_max:
 *  the first node strictly above x, with the pair clamped so that a
 *  value below the first (or above the last) node uses the first (last)
 *  interval -- i.e. it extrapolates linearly rather than saturating.
 * ------------------------------------------------------------------ */
std::pair<int,int> bracket(double x, const std::vector<int>& nodes)
{
    const int len = static_cast<int>(nodes.size());
    if (len <= 1) return {nodes[0], nodes[0]};

    int i = -1;
    for (int k = 0; k < len; ++k)
        if (x < nodes[k]) { i = k; break; }

    if (i == 0)       i = 1;          // below the first node
    else if (i < 0)   i = len - 1;    // at or above the last node
    return {nodes[i - 1], nodes[i]};
}

} // namespace

double vacuum_to_air(double l)
{
    const double lc = std::clamp(l, kAirLo, kAirHi);
    return l / air_divisor(lc);
}

Vector vacuum_to_air(const Vector& lv)
{
    Vector out(lv.size());
    for (Eigen::Index i = 0; i < lv.size(); ++i) out[i] = vacuum_to_air(lv[i]);
    return out;
}

/* ===================================================================== *
 *  Parameter metadata
 * ===================================================================== */
const char* telluric_param_name(int i)
{
    switch (i) {
        case 0: return "airmass";
        case 1: return "pwv";
        case 2: return "barycorr";
        default: return "?";
    }
}

std::pair<double,double> telluric_param_limits(int i)
{
    /*  initialize_telluric.sl, telluric_default: airmass min=1 max=3,
     *  pwv min=0 max=20 (mm), barycorr min=-500 max=500 (km/s).
     *
     *  The pwv floor is the library's own first node (0.5 mm) rather than
     *  ISIS's 0.  Below 0.5 mm there is nothing to interpolate, so ISIS
     *  extrapolates and can return a transmission above one; keeping the
     *  solver inside the library avoids that without introducing the kink a
     *  clamp would put in the middle of a fitted parameter's range.         */
    switch (i) {
        case 0: return { 1.0,   3.0 };
        case 1: return { 0.5,  20.0 };
        case 2: return {-500.0, 500.0};
        default: return {0.0, 0.0};
    }
}

/* ===================================================================== *
 *  Library layout
 * ===================================================================== */
const std::vector<int>& TelluricGrid::airmass_nodes()
{
    static const std::vector<int> v = {10, 15, 20, 25, 30};        // airmass*10
    return v;
}
const std::vector<int>& TelluricGrid::pwv_nodes()
{
    static const std::vector<int> v = {5, 10, 15, 25, 35, 50, 75, 100, 200};
    return v;                                                      // pwv*100
}

std::string TelluricGrid::corner_path(int a10, int w100) const
{
    std::ostringstream s;
    s << dir_ << "/LBL_A" << a10 << "_s0_w" << std::setw(3) << std::setfill('0')
      << w100 << "_R0300000_T.fits";
    return s.str();
}

std::string TelluricGrid::resolve(const std::vector<std::string>& bases)
{
    /*  The probe file is the one ISIS checks for, so a directory that passes
     *  here is one ISIS would also accept.                                 */
    for (const auto& b : bases) {
        fs::path p = fs::path(b) / "telluric";
        if (fs::exists(p / "LBL_A10_s0_w005_R0300000_T.fits")) return p.string();
    }
    throw std::runtime_error(
        "Telluric library not found: no <basePath>/telluric/ contains "
        "LBL_A10_s0_w005_R0300000_T.fits");
}

TelluricGrid::TelluricGrid(std::string dir) : dir_(std::move(dir)) {}

/* ===================================================================== *
 *  One library corner: read, air-convert, slice, degrade
 * ===================================================================== */
SpectrumPtr TelluricGrid::corner(int a10, int w100,
                                 double lmin, double lmax,
                                 double resOffset, double resSlope) const
{
    std::size_t key = hash_string(dir_);
    key = hash_combine(key, 0x7E11A21CULL);
    key = hash_combine(key, static_cast<std::size_t>(a10));
    key = hash_combine(key, static_cast<std::size_t>(w100) << 8);
    key = hash_combine(key, hash_double(lmin));
    key = hash_combine(key, hash_double(lmax));
    key = hash_combine(key, hash_double(resOffset));
    key = hash_combine(key, hash_double(resSlope));

    return SpectrumCache::instance().insert_if_absent(key, [&] {
        /*  Every library file shares one wavelength column, so it is read
         *  and air-converted once -- ISIS caches it the same way
         *  (cache->telluric_spectra.lambda).  The conversion is 1.4 M
         *  divisions, which is worth doing exactly once per process.      */
        static Vector           lam_air;
        static std::mutex       lam_mtx;
        {
            std::lock_guard<std::mutex> lk(lam_mtx);
            if (lam_air.size() == 0) {
                CCfits::FITS f(corner_path(airmass_nodes()[0], pwv_nodes()[0]),
                               CCfits::Read);
                CCfits::ExtHDU& e = f.extension(1);
                std::vector<Real> um;
                e.column("lam").read(um, 1, e.rows());
                Vector vac(um.size());
                for (std::size_t i = 0; i < um.size(); ++i)
                    vac[static_cast<Eigen::Index>(i)] = um[i] * 10000.0; // um -> A
                lam_air = vacuum_to_air(vac);
            }
        }
        const Vector& lam = lam_air;

        std::vector<Real> tr;
        {
            CCfits::FITS f(corner_path(a10, w100), CCfits::Read);
            CCfits::ExtHDU& e = f.extension(1);
            e.column("trans").read(tr, 1, e.rows());
        }
        if (static_cast<Eigen::Index>(tr.size()) != lam.size())
            throw std::runtime_error(
                "Telluric library file " + corner_path(a10, w100) +
                " does not share the wavelength grid of the library");

        /* ---- slice to the fitted window, one point of margin each side -- */
        const Real* b = lam.data();
        const Real* e = lam.data() + lam.size();
        Eigen::Index i0 = std::lower_bound(b, e, lmin) - b;
        Eigen::Index i1 = std::upper_bound(b, e, lmax) - b;
        if (i0 > 0)          --i0;
        if (i1 < lam.size()) ++i1;
        if (i1 <= i0) { i0 = 0; i1 = lam.size(); }

        const Eigen::Index n = i1 - i0;

        Spectrum sp;
        sp.lambda = lam.segment(i0, n);
        sp.flux   = Eigen::Map<const Vector>(tr.data(), tr.size()).segment(i0, n);
        sp.sigma  = Vector::Ones(n);

        /*  ISIS convolves with convolve_syn(vsini=0) and then thins the grid
         *  with optimize_wavegrid; the thinning is a speed measure, so only
         *  the convolution is reproduced here.                             */
        if (resOffset != 0.0 || resSlope != 0.0)
            sp.flux = degrade_resolution(sp.lambda, sp.flux, resOffset, resSlope);

        return sp;
    });
}

/* ===================================================================== *
 *  ... shifted and rebinned onto one observed grid
 * ===================================================================== */
SpectrumPtr TelluricGrid::corner_on_grid(int a10, int w100,
                                         const Vector& lam,
                                         double barycorr,
                                         double resOffset,
                                         double resSlope) const
{
    constexpr double c_kms = 299792.458;

    /*  The window has to cover the observed range *after* the shift, and the
     *  shift is itself fitted, so the slice is widened by the largest one the
     *  solver can reach.  ISIS slices to the data range and shifts afterwards,
     *  which lets the very edge pixels fall off the library; this only ever
     *  adds library coverage, never changes a value inside it.             */
    const double margin = 1.0 + 500.0 / c_kms;
    const double lmin   = lam.minCoeff() / margin;
    const double lmax   = lam.maxCoeff() * margin;

    std::size_t key = hash_string(dir_);
    key = hash_combine(key, 0x7E11B21DULL);
    key = hash_combine(key, static_cast<std::size_t>(a10));
    key = hash_combine(key, static_cast<std::size_t>(w100) << 8);
    key = hash_combine(key, hash_double(barycorr));
    key = hash_combine(key, hash_double(resOffset));
    key = hash_combine(key, hash_double(resSlope));
    key = hash_combine(key, hash_double(lam.minCoeff()));
    key = hash_combine(key, hash_double(lam.maxCoeff()));
    key = hash_combine(key, static_cast<std::size_t>(lam.size()));

    return SpectrumCache::instance().insert_if_absent(key, [&] {
        SpectrumPtr raw = corner(a10, w100, lmin, lmax, resOffset, resSlope);

        const double factor = 1.0 + barycorr / c_kms;
        Vector shifted = raw->lambda * factor;

        Spectrum out;
        out.lambda = lam;
        out.flux   = trapezoidal_rebin(shifted, raw->flux, lam);
        out.sigma  = Vector::Ones(lam.size());
        return out;
    });
}

/* ===================================================================== *
 *  Bilinear interpolation over (airmass, pwv)
 * ===================================================================== */
Vector TelluricGrid::transmission(const Vector& lam,
                                  double airmass, double pwv, double barycorr,
                                  double resOffset, double resSlope) const
{
    /*  ISIS: "If the airmass is set to zero, no telluric spectrum but a
     *  constant factor of one will be returned."                          */
    if (airmass < 1e-5) return Vector::Ones(lam.size());

    /* ------------------------------------------------------------------ *
     *  Node units.  The library encodes airmass*10 and PWV*10 in its file
     *  names (LBL_A15_s0_w025 is airmass 1.5, PWV 2.5 mm; see its README).
     *
     *  ISIS scales PWV by 100 instead (interpol_telluric.sl:
     *  `values = [airmass*10., pwv*100.]`), so its `pwv` is really tenths of
     *  a millimetre: pwv=1 returns the 10 mm spectrum bit-for-bit, and
     *  anything above pwv=2 runs off the end of the library and extrapolates
     *  to *negative* transmission -- over most of the range ISIS itself
     *  allows (min=0, max=20).  Its airmass scaling has no such problem, and
     *  its own bounds [1,3] match the library exactly, which is what marks
     *  the PWV factor as a slip rather than a unit convention.
     *
     *  GAEL uses millimetres, as documented.  Set `isis_pwv_scale` to
     *  reproduce ISIS's numbers when comparing the two codes directly.
     * ------------------------------------------------------------------ */
    const double av = airmass * 10.0;
    const double wv = pwv * (isis_pwv_scale_ ? 100.0 : 10.0);

    const auto [a_lo, a_hi] = bracket(av, airmass_nodes());
    const auto [w_lo, w_hi] = bracket(wv, pwv_nodes());

    /*  Weights of the two-point linear interpolation ISIS performs in each
     *  dimension: (y1*(max-v) + y2*(v-min)) / (max-min).  A value that lands
     *  exactly on a node collapses the pair, and then only that node is read
     *  -- which is also why the degenerate case must not divide by zero.   */
    const double fa = (a_hi == a_lo) ? 0.0
                    : (av - a_lo) / static_cast<double>(a_hi - a_lo);
    const double fw = (w_hi == w_lo) ? 0.0
                    : (wv - w_lo) / static_cast<double>(w_hi - w_lo);

    struct Corner { int a, w; double weight; };
    const Corner corners[4] = {
        { a_lo, w_lo, (1.0 - fa) * (1.0 - fw) },
        { a_hi, w_lo,        fa  * (1.0 - fw) },
        { a_lo, w_hi, (1.0 - fa) *        fw  },
        { a_hi, w_hi,        fa  *        fw  },
    };

    Vector out = Vector::Zero(lam.size());
    for (const auto& cn : corners) {
        if (cn.weight == 0.0) continue;      // never read a file we don't need
        out += cn.weight *
               corner_on_grid(cn.a, cn.w, lam, barycorr, resOffset, resSlope)->flux;
    }
    return out;
}

} // namespace specfit
