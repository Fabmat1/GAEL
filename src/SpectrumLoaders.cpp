#include "specfit/SpectrumLoaders.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfit {
namespace {

Vector linear_interpolate(const Vector& x_new,
                          const Vector& x_old,
                          const Vector& y_old)
{
    if (x_old.size() != y_old.size() || x_old.size() < 2)
        throw std::invalid_argument("linear_interpolate: invalid input table.");

    Vector y_new(x_new.size());

    for (Eigen::Index i = 0; i < x_new.size(); ++i)
    {
        const double x = x_new[i];

        if (x <= x_old[0]) {
            y_new[i] = y_old[0];
            continue;
        }
        if (x >= x_old[x_old.size() - 1]) {
            y_new[i] = y_old[y_old.size() - 1];
            continue;
        }

        // binary search
        const auto* first = x_old.data();
        const auto* last  = x_old.data() + x_old.size();
        const auto* it    = std::upper_bound(first, last, x);
        const Eigen::Index hi = static_cast<Eigen::Index>(it - first);
        const Eigen::Index lo = hi - 1;

        const double t =
            (x - x_old[lo]) / (x_old[hi] - x_old[lo]);      // 0 … 1
        y_new[i] = (1.0 - t) * y_old[lo] + t * y_old[hi];
    }
    return y_new;
}

inline double median(Vector v)               // pass *by value*  → cheap copy
{
    const Eigen::Index n = v.size();
    if (n == 0)
        throw std::runtime_error("median(): empty vector");

    Eigen::Index k = n / 2;
    std::nth_element(v.data(), v.data() + k, v.data() + n);   // k-th element

    double m = v[k];
    if ((n & 1) == 0) {                                       // even length
        const double max_lo = *std::max_element(v.data(), v.data() + k);
        m = 0.5 * (m + max_lo);
    }
    return m;
}

// ----------------------------------------------------------------------------
//  Read an ASCII table with N columns of doubles, skip comment lines
// ----------------------------------------------------------------------------
std::vector<std::array<double, 3>>
read_ascii_table(const std::string& path,
                 int  ncols,                        // 2 or 3
                 char comment_char = '#')
{
    if (ncols < 2 || ncols > 3)
        throw std::invalid_argument("read_ascii_table: ncols must be 2 or 3");

    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open '" + path + "'");

    std::vector<std::array<double,3>> rows;
    std::string line;
    while (std::getline(in, line))
    {
        // trim leading whitespace
        auto it  = std::find_if_not(line.begin(), line.end(), ::isspace);
        if (it == line.end()) continue;           // blank line
        if (*it == comment_char) continue;        // comment

        std::istringstream ss(line);
        std::array<double,3> row{0.0,0.0,std::numeric_limits<double>::quiet_NaN()};

        if (ncols == 2)
        {
            if (!(ss >> row[0] >> row[1])) continue;
        } else {
            if (!(ss >> row[0] >> row[1] >> row[2])) continue;
        }
        if (!(std::isfinite(row[0]) && std::isfinite(row[1]) &&
            (ncols == 2 || std::isfinite(row[2]))))
            continue;                                      // silently drop bad rows
        rows.push_back(row);
    }
    if (rows.empty())
        throw std::runtime_error("File '" + path + "' contains no valid data");

    return rows;
}

// ----------------------------------------------------------------------------
//  Sort a vector of <λ, …> rows by λ ascending and move into Eigen vectors
// ----------------------------------------------------------------------------
template <int NC>
void to_eigen(const std::vector<std::array<double,NC>>& rows,
              Vector& l_out,
              Vector& f_out,
              Vector* s_out = nullptr)
{
    const std::size_t n = rows.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t i, std::size_t j)
              { return rows[i][0] < rows[j][0]; });

    l_out.resize(n);
    f_out.resize(n);
    if (s_out) s_out->resize(n);

    for (std::size_t k = 0; k < n; ++k)
    {
        const auto& r = rows[idx[k]];
        l_out[k] = r[0];
        f_out[k] = r[1];
        if (s_out) (*s_out)[k] = r[2];
    }
}

} // unnamed namespace
// ============================================================================
//  Public loader implementations
// ============================================================================

Spectrum load_ascii_2col(const std::string& path)
{
    const auto rows = read_ascii_table(path, /*ncols=*/2);

    Vector lambda, flux;
    to_eigen<3>(rows, lambda, flux);
    const Eigen::Index len = lambda.size();

    // --- Defensive input validation (never reach the SNR code with bad data) -
    if (len < 10)
        throw std::runtime_error("load_ascii_2col: spectrum too short ("
                                 + std::to_string(len) + " points)");
    if (!lambda.allFinite() || !flux.allFinite())
        throw std::runtime_error("load_ascii_2col: non-finite values");
    for (Eigen::Index i = 1; i < len; ++i)
        if (!(lambda[i] > lambda[i-1]))
            throw std::runtime_error("load_ascii_2col: wavelengths not "
                                     "strictly ascending");

    // --- Normalise first, so spec_out ends up normalised -------------------
    const double med = median(flux);
    if (!std::isfinite(med) || med == 0.0)
        throw std::runtime_error("load_ascii_2col: invalid flux median");
    flux.array() /= med;

    // --- Prepare spec_out fully before calling any SNR routine -------------
    Spectrum spec_out;
    spec_out.lambda     = lambda;
    spec_out.flux       = flux;
    spec_out.sigma      = Vector::Ones(len);                 // placeholder
    spec_out.ignoreflag = std::vector<int>(len, 1);          // all good

    // --- Choose a safe window size ----------------------------------------
    int npix_box = std::clamp<int>(static_cast<int>(std::round(len / 10.0)),
                                   50,
                                   static_cast<int>(len / 4));
    if (npix_box < 50) npix_box = std::max<int>(50, int(len / 2));

    // --- SNR curve with a fallback (DER_SNR is the known-crashy code) ------
    Vector sigma_out;
    try {
        SNRCurve snr = spec_out.estimate_snr_curve("der_snr", npix_box);

        if (snr.lambda.size() < 2 ||
            !snr.noise.allFinite())
            throw std::runtime_error("bad SNR curve");

        Vector x_old(snr.lambda.size() + 2);
        Vector y_old(snr.noise .size() + 2);
        x_old[0] = lambda[0];
        x_old[x_old.size()-1] = lambda[len-1];
        y_old[0] = snr.noise[0];
        y_old[y_old.size()-1] = snr.noise[snr.noise.size()-1];
        x_old.segment(1, snr.lambda.size()) = snr.lambda;
        y_old.segment(1, snr.noise .size()) = snr.noise;

        sigma_out = linear_interpolate(lambda, x_old, y_old);
        if (!sigma_out.allFinite()) throw std::runtime_error("bad interp");
    }
    catch (const std::exception& e) {
        // Fallback: robust global sigma from MAD of first differences.
        // Scaled by 1/(sqrt(2)) because var(x_{i+1}-x_i)=2 var(x_i) for white noise.
        Vector d(len - 1);
        for (Eigen::Index i = 0; i < len - 1; ++i) d[i] = flux[i+1] - flux[i];
        std::vector<double> abs_d(d.size());
        for (Eigen::Index i = 0; i < d.size(); ++i) abs_d[i] = std::abs(d[i]);
        std::nth_element(abs_d.begin(),
                         abs_d.begin() + abs_d.size()/2,
                         abs_d.end());
        double mad   = abs_d[abs_d.size()/2];
        double sigma = (mad > 0 ? mad : 1e-3) / std::sqrt(2.0) * 1.4826;
        sigma_out    = Vector::Constant(len, sigma);
    }

    spec_out.sigma = sigma_out;
    return spec_out;
}

// ----------------------------------------------------------------------------
Spectrum load_ascii_3col(const std::string& path)
{
    // -------------------- 1. read file -------------------------------------------------
    const auto rows = read_ascii_table(path, /*ncols=*/3);

    // -------------------- 2. move & sort ----------------------------------------------
    Vector lambda, flux, sigma;
    to_eigen<3>(rows, lambda, flux, &sigma);

    // Normalisation and the flux/error repairs are format-independent and live
    // in sanitize_spectrum(), which the fit pipeline applies to every loader.
    return Spectrum{lambda, flux, sigma};
}

// ----------------------------------------------------------------------------
//  ISIS's post-read clean-up (spectroscopy_automated.sl, right after
//  read_spectrum):
//
//      ind = where(f>0); l = l[ind]; f = f[ind]; err = err[ind];
//      ind = where(err<=0, &temp); err[ind] = interpol(l[ind], l[temp], err[temp]);
//      temp = median(f); f /= temp; err /= temp;
//
//  Non-positive fluxes are unphysical pixels, and a non-positive error carries
//  no weight information -- GAEL used to keep both and simply drop them from
//  chi2, which still let them into the neighbourhood statistics of the
//  iterative-noise stage and biased the local mean and sdev low.  The median
//  normalisation additionally puts the continuum anchors near 1 instead of in
//  raw instrument counts, which matters because they share a parameter vector
//  (and now a scaled convergence test) with Teff.
// ----------------------------------------------------------------------------
Spectrum sanitize_spectrum(const Spectrum& in)
{
    const Eigen::Index n_in = in.lambda.size();
    if (n_in == 0) return in;

    /* ---- 1. keep only strictly positive, finite flux ------------------ */
    std::vector<Eigen::Index> keep;
    keep.reserve(static_cast<std::size_t>(n_in));
    for (Eigen::Index i = 0; i < n_in; ++i)
        if (std::isfinite(in.lambda[i]) && std::isfinite(in.flux[i]) &&
            in.flux[i] > 0.0)
            keep.push_back(i);

    if (keep.empty())
        throw std::runtime_error("sanitize_spectrum: no positive flux values");

    const Eigen::Index n = static_cast<Eigen::Index>(keep.size());
    Spectrum out;
    out.lambda.resize(n);
    out.flux.resize(n);
    out.sigma.resize(n);
    const bool have_sigma = in.sigma.size() == n_in;
    for (Eigen::Index k = 0; k < n; ++k) {
        const Eigen::Index i = keep[static_cast<std::size_t>(k)];
        out.lambda[k] = in.lambda[i];
        out.flux[k]   = in.flux[i];
        out.sigma[k]  = have_sigma ? in.sigma[i] : 0.0;
    }
    if (!in.ignoreflag.empty() &&
        static_cast<Eigen::Index>(in.ignoreflag.size()) == n_in) {
        out.ignoreflag.resize(static_cast<std::size_t>(n));
        for (Eigen::Index k = 0; k < n; ++k)
            out.ignoreflag[static_cast<std::size_t>(k)] =
                in.ignoreflag[static_cast<std::size_t>(keep[static_cast<std::size_t>(k)])];
    }

    /* ---- 2. repair non-positive errors by interpolating the good ones -- */
    std::vector<Eigen::Index> bad, good;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (std::isfinite(out.sigma[i]) && out.sigma[i] > 0.0) good.push_back(i);
        else                                                   bad.push_back(i);
    }
    if (!bad.empty() && good.size() >= 2) {
        Vector lg(static_cast<Eigen::Index>(good.size()));
        Vector sg(static_cast<Eigen::Index>(good.size()));
        for (std::size_t k = 0; k < good.size(); ++k) {
            lg[static_cast<Eigen::Index>(k)] = out.lambda[good[k]];
            sg[static_cast<Eigen::Index>(k)] = out.sigma [good[k]];
        }
        Vector lb(static_cast<Eigen::Index>(bad.size()));
        for (std::size_t k = 0; k < bad.size(); ++k)
            lb[static_cast<Eigen::Index>(k)] = out.lambda[bad[k]];

        const Vector repaired = linear_interpolate(lb, lg, sg);
        for (std::size_t k = 0; k < bad.size(); ++k)
            out.sigma[bad[k]] = repaired[static_cast<Eigen::Index>(k)];
    }
    // With fewer than two usable errors there is nothing to interpolate from;
    // those pixels keep their non-positive sigma and are dropped from chi2 by
    // the sigma <= 0 guard, exactly as before.

    /* ---- 3. normalise flux and error to a median flux of one ---------- */
    const double med = median(out.flux);
    if (std::isfinite(med) && med > 0.0) {
        out.flux .array() /= med;
        out.sigma.array() /= med;
    }

    return out;
}

// ============================================================================
//  Dispatcher / auto-detection  (unchanged stubs follow)
// ============================================================================

// keep previously generated stub macro for the remaining formats -------------
#define SPECFIT_MAKE_STUB(name)                                  \
Spectrum name(const std::string& path)                           \
{                                                                \
    throw std::runtime_error(#name ": loader not implemented");  \
}

SPECFIT_MAKE_STUB(load_sdss)
SPECFIT_MAKE_STUB(load_sdss_v)
SPECFIT_MAKE_STUB(load_lamost)
SPECFIT_MAKE_STUB(load_lamost_dr8)
SPECFIT_MAKE_STUB(load_feros)
SPECFIT_MAKE_STUB(load_feros_phase3)
SPECFIT_MAKE_STUB(load_uves)
SPECFIT_MAKE_STUB(load_xshooter)
SPECFIT_MAKE_STUB(load_xshooter_esoreflex)
SPECFIT_MAKE_STUB(load_fuse)
SPECFIT_MAKE_STUB(load_4most)
SPECFIT_MAKE_STUB(load_iraf)
SPECFIT_MAKE_STUB(load_muse)

#undef SPECFIT_MAKE_STUB

// ----------------------------------------------------------------------------
// Dispatcher / auto-detection -------------------------------------------------
// (same code as before – left unchanged)
// ----------------------------------------------------------------------------
static const std::unordered_map<std::string, SpectrumLoader> kLoaderMap = {
    {"ASCII_with_2_columns",  load_ascii_2col},
    {"ASCII_with_3_columns",  load_ascii_3col},
    {"SDSS",                  load_sdss},
    {"SDSSV",                 load_sdss_v},
    {"LAMOST",                load_lamost},
    {"LAMOST_DR8",            load_lamost_dr8},
    {"FEROS",                 load_feros},
    {"FEROS_phase3",          load_feros_phase3},
    {"UVES",                  load_uves},
    {"XSHOOTER",              load_xshooter},
    {"XSHOOTER_esoreflex",    load_xshooter_esoreflex},
    {"FUSE",                  load_fuse},
    {"4MOST",                 load_4most},
    {"IRAF",                  load_iraf},
    {"MUSE",                  load_muse}
};

Spectrum load_spectrum(const std::string& path, const std::string& format)
{
    if (format == "auto") {
        for (const auto& kv : kLoaderMap) {
            try { return kv.second(path); }
            catch (const std::exception&) { /* try next */ }
        }
        throw std::runtime_error("load_spectrum(auto): none of the "
                                 "registered readers could load '" + path + "'");
    }

    auto it = kLoaderMap.find(format);
    if (it == kLoaderMap.end())
        throw std::runtime_error("Unsupported spectrum format: " + format);

    return it->second(path);
}

} // namespace specfit