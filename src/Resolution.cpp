#include "specfit/Resolution.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace specfit {

using Vector = Eigen::VectorXd;

namespace {
constexpr double SIGMA_FROM_FWHM = 1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
constexpr double KERNEL_RADIUS   = 5.0;

// Structure to hold precomputed weights for one output point
struct WeightSegment {
    std::size_t jStart;
    std::size_t jEnd;
    std::vector<double> weights;  // Pre-normalized weights including dLam
};

/*  Signature of one wavelength grid, cheap enough to compare on every call.
 *
 *  Endpoints and length alone do not identify a grid: a metal fit convolves
 *  on the union of the species grids, and a binary's two components can have
 *  unions that share a range without sharing their interior.  A false hit
 *  there would convolve with another grid's weights, so two interior samples
 *  are carried as well -- O(1), and the whole point of the entry is that it
 *  survives for the rest of the fit.                                        */
struct GridSig {
    double      start;
    double      end;
    double      mid1;
    double      mid2;
    std::size_t nPoints;

    bool operator==(const GridSig& o) const {
        return std::abs(start - o.start) < 1e-10 &&
               std::abs(end - o.end) < 1e-10 &&
               std::abs(mid1 - o.mid1) < 1e-10 &&
               std::abs(mid2 - o.mid2) < 1e-10 &&
               nPoints == o.nPoints;
    }
};

static GridSig grid_sig(const Vector& lam) {
    const Eigen::Index n = lam.size();
    GridSig s;
    s.start   = lam[0];
    s.end     = lam[n - 1];
    s.mid1    = lam[n / 3];
    s.mid2    = lam[(2 * n) / 3];
    s.nPoints = static_cast<std::size_t>(n);
    return s;
}

// Cache key: input + output grid signature + resolution parameters
struct CacheKey {
    GridSig in;
    GridSig out;
    double resOffset;
    double resSlope;

    bool operator==(const CacheKey& other) const {
        return in == other.in && out == other.out &&
               std::abs(resOffset - other.resOffset) < 1e-10 &&
               std::abs(resSlope - other.resSlope) < 1e-10;
    }
};

// Hash function for cache key
struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const {
        auto h1 = std::hash<double>{}(k.in.start);
        auto h2 = std::hash<double>{}(k.in.end);
        auto h3 = std::hash<double>{}(k.in.mid1);
        auto h4 = std::hash<std::size_t>{}(k.in.nPoints);
        auto h5 = std::hash<double>{}(k.resOffset);
        auto h6 = std::hash<double>{}(k.resSlope);
        auto h7 = std::hash<std::size_t>{}(k.out.nPoints);
        auto h8 = std::hash<double>{}(k.out.end);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5)
                  ^ (h7 << 6) ^ (h8 << 7);
    }
};

/*  Weights are handed out as a shared_ptr, not as a reference into the map:
 *  corner spectra are convolved from several OpenMP threads at once, and a
 *  concurrent eviction would otherwise pull the vector out from under a
 *  thread that was still summing over it.                                   */
using WeightsPtr = std::shared_ptr<const std::vector<WeightSegment>>;

static std::size_t weights_bytes(const std::vector<WeightSegment>& w) {
    std::size_t b = w.size() * sizeof(WeightSegment);
    for (const auto& s : w) b += s.weights.capacity() * sizeof(double);
    return b;
}

// Global cache (thread-safe for reads after initial computation)
static std::unordered_map<CacheKey, std::pair<WeightsPtr, std::size_t>,
                          CacheKeyHash> g_weightCache;
static std::list<CacheKey> g_lruList;
static std::mutex g_cacheMutex;
static std::size_t g_cacheBytes = 0;

/*  Keys a thread is currently computing, so that the others wait for it
 *  instead of computing the same thing.  Without this, the first Jacobian of
 *  a metal fit has every thread miss the same key at once and each build its
 *  own copy of a weight set that runs to half a gigabyte -- the work is
 *  thrown away and the peak memory is not.                                  */
static std::vector<CacheKey>      g_inflight;
static std::condition_variable   g_inflightCv;

/*  A budget rather than an entry count.  One weight set is `n_out` segments of
 *  however many input points fall inside 5 sigma, so its size swings by three
 *  orders of magnitude between the grids this code sees: ~1 MB for a
 *  14 822-point HHE corner spectrum, but ~340 MB for a metal model, where the
 *  integral runs over a 600 k-point union grid and the kernel spans ~750 of
 *  its points.  Twenty-five of the latter is how a metal fit reached 21 GB.
 *
 *  On the size: a fit cycles through one entry per (grid, spectrograph) pair
 *  and does so round-robin, which is the one access pattern LRU handles
 *  worst -- with a budget too small by a single entry, every call evicts the
 *  set the next call wants and rebuilds it, and rebuilding means an exp() per
 *  weight.  Measured on the two-arm X-Shooter metal case, whose two sets come
 *  to 657 MB together: at 512 MiB the fit spent 53 % of its time in exp and
 *  took 363 s; holding both took it to 71 s.  So the budget has to clear a
 *  realistic joint fit rather than a single spectrum.  These entries only
 *  ever get large on the metal path -- an HHE-only fit stays around 25 MB
 *  whatever this says.                                                       */
static constexpr std::size_t MAX_CACHE_BYTES = 3ull << 30;
static constexpr std::size_t MAX_CACHE_SIZE  = 25;

// Moves key to front (most recently used)
static void touch_key(const CacheKey& key) {
    // Remove any existing occurrence
    g_lruList.remove(key);
    g_lruList.push_front(key);
}

// Inserts key and evicts if needed
static WeightsPtr insert_cache_entry(const CacheKey& key,
                                     std::vector<WeightSegment>&& value) {
    /*  Two threads can race to compute the same weights; keep the first set
     *  rather than adding a second LRU entry for the same key.              */
    if (auto it = g_weightCache.find(key); it != g_weightCache.end()) {
        touch_key(key);
        return it->second.first;
    }

    const std::size_t nbytes = weights_bytes(value);
    auto ptr = std::make_shared<const std::vector<WeightSegment>>(std::move(value));
    g_weightCache[key] = {ptr, nbytes};
    g_lruList.push_front(key);
    g_cacheBytes += nbytes;

    while (g_weightCache.size() > 1 &&
           (g_weightCache.size() > MAX_CACHE_SIZE ||
            g_cacheBytes > MAX_CACHE_BYTES)) {
        const CacheKey victim = g_lruList.back();
        if (auto it = g_weightCache.find(victim); it != g_weightCache.end()) {
            g_cacheBytes -= std::min(g_cacheBytes, it->second.second);
            g_weightCache.erase(it);
        }
        g_lruList.pop_back();
    }
    return ptr;
}


/*  Weights taking `lam_in` to `lam_out`.  With lam_out == lam_in this is the
 *  original single-grid routine, operation for operation.                    */
std::vector<WeightSegment> compute_weights(const Vector& lam,
                                          const Vector& lam_out,
                                          double resOffset,
                                          double resSlope)
{
    const std::size_t n     = lam.size();
    const std::size_t n_out = lam_out.size();
    std::vector<WeightSegment> weights(n_out);

    // Precompute bin widths
    Vector dLam(n);
    dLam[0]     = lam[1]     - lam[0];
    dLam[n - 1] = lam[n - 1] - lam[n - 2];
    for (std::size_t j = 1; j < n - 1; ++j)
        dLam[j] = 0.5 * (lam[j + 1] - lam[j - 1]);

    const double* lamData  = lam.data();
    const double* outData  = lam_out.data();
    const double* dLamData = dLam.data();

    // Compute weights for each output point
    #pragma omp parallel for schedule(dynamic, 32) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_out); ++i)
    {
        const double λ_i   = outData[i];
        const double R     = resOffset + resSlope * λ_i;
        const double sigma = (λ_i / R) * SIGMA_FROM_FWHM;

        const double lamMin = λ_i - KERNEL_RADIUS * sigma;
        const double lamMax = λ_i + KERNEL_RADIUS * sigma;

        const std::size_t jStart = std::lower_bound(lamData, lamData + n, lamMin) - lamData;
        const std::size_t jEnd   = std::upper_bound(lamData + jStart, lamData + n, lamMax) - lamData;

        weights[i].jStart = jStart;
        weights[i].jEnd = jEnd;
        weights[i].weights.resize(jEnd - jStart);

        const double inv2σ2 = 1.0 / (2.0 * sigma * sigma);
        double wSum = 0.0;

        // Compute normalized weights
        for (std::size_t j = jStart; j < jEnd; ++j) {
            double delta = lamData[j] - λ_i;
            double w = std::exp(-delta * delta * inv2σ2) * dLamData[j];
            weights[i].weights[j - jStart] = w;
            wSum += w;
        }

        // Normalize weights
        const double invWSum = 1.0 / wSum;
        for (auto& w : weights[i].weights) {
            w *= invWSum;
        }
    }

    return weights;
}

// Create cache key from wavelength grids and resolution parameters
CacheKey make_cache_key(const Vector& lam, const Vector& lam_out,
                        double resOffset, double resSlope) {
    CacheKey key;
    key.in        = grid_sig(lam);
    key.out       = grid_sig(lam_out);
    key.resOffset = resOffset;
    key.resSlope  = resSlope;
    return key;
}

} // anonymous namespace

// Fast application of precomputed weights
static Vector apply_weights(const Vector& flux, const std::vector<WeightSegment>& weights) {
    const std::size_t n = weights.size();
    Vector out(n);
    const double* fluxData = flux.data();
    
    #pragma omp parallel for schedule(static) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const auto& seg = weights[i];
        double sum = 0.0;
        
        // Simple dot product with precomputed normalized weights
        #pragma omp simd reduction(+:sum)
        for (std::size_t j = 0; j < seg.weights.size(); ++j) {
            sum += seg.weights[j] * fluxData[seg.jStart + j];
        }
        
        out[i] = sum;
    }
    
    return out;
}

/* ------------------------------------------------------------------ *
 *  Exact replica of ISIS' convolve_instrument / c_functions->convolve *
 *  (stellar_isisscripts). For each output point i:                    *
 *      out_i = sum_j gw_j * f_interp( lam_i - sig_i * gx_j )           *
 *  with gx_j a UNIFORM grid on [-3,3] (sigma units, spacing<=0.015A,   *
 *  >=15 pts/sigma), gw_j = exp(-gx_j^2/2) normalised to sum 1, and     *
 *  f_interp linear with edge-clamping.  This is the uniform-sigma-     *
 *  kernel + linear-interp scheme; GAEL's default instead sums over    *
 *  native points with dLam weights out to 5 sigma.  Selected via       *
 *  GAEL_CONV=isis (diagnostic toggle).                                */
static Vector degrade_resolution_isis(const Vector& lam,
                                      const Vector& flux,
                                      double resOffset,
                                      double resSlope)
{
    const std::ptrdiff_t n = lam.size();
    if (n < 2) return flux;

    constexpr double fwhm_to_sigma = 0.4246609001440095;
    constexpr double max_spacing   = 0.015;

    // numerical conditioning: divide by median during convolution (as ISIS)
    Vector tmp = flux;
    std::nth_element(tmp.data(), tmp.data() + n/2, tmp.data() + n);
    double med = tmp[n/2];
    if (!(med > 0.0)) med = 1.0;

    // per-point sigma (wavelength units)
    Vector sig(n);
    double sig_max = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const double R = resOffset + resSlope * lam[i];
        sig[i] = lam[i] / R * fwhm_to_sigma;
        if (sig[i] > sig_max) sig_max = sig[i];
    }

    // fixed Gaussian kernel on a uniform grid over [-3,3] sigma
    int Ng = static_cast<int>(std::lround(sig_max / max_spacing));
    if (Ng < 15) Ng = 15;
    const std::ptrdiff_t lg = 6 * Ng + 1;
    Vector gx(lg), gw(lg);
    double gsum = 0.0;
    for (std::ptrdiff_t j = 0; j < lg; ++j) {
        gx[j] = -3.0 + 6.0 * static_cast<double>(j) / static_cast<double>(lg - 1);
        gw[j] = std::exp(-0.5 * gx[j] * gx[j]);
        gsum += gw[j];
    }
    for (std::ptrdiff_t j = 0; j < lg; ++j) gw[j] /= gsum;

    const double* fx = lam.data();
    const double* fy = flux.data();
    Vector out(n);

    #pragma omp parallel for schedule(dynamic, 64) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        double sum = 0.0;
        const double s = sig[i];
        // x increases as j decreases (gx ascending), mirror as in ISIS
        for (std::ptrdiff_t j = lg - 1; j >= 0; --j) {
            const double x = lam[i] - s * gx[j];
            if (x <= fx[0]) {
                sum += fy[0] * gw[j];
            } else if (x >= fx[n - 1]) {
                sum += fy[n - 1] * gw[j];
            } else {
                const std::ptrdiff_t k =
                    std::upper_bound(fx, fx + n, x) - fx - 1;   // fx[k] <= x < fx[k+1]
                const double t = (x - fx[k]) / (fx[k + 1] - fx[k]);
                sum += (fy[k] + (fy[k + 1] - fy[k]) * t) * gw[j];
            }
        }
        out[i] = sum;
    }
    (void)med;   // flux/med then *med cancels exactly for this linear operator
    return out;
}

#ifdef GAEL_USE_CUDA
/*  The GPU kernel is algorithmically identical to the CPU one but recomputes
 *  the Gaussian weights on every call and pays a full device synchronise per
 *  corner spectrum, while the CPU path caches the weights and then only does
 *  a ~1 MFLOP sparse mat-vec.  Measured on a 5-spectrum joint fit, the CPU
 *  path won every configuration tried -- by 35 % when the spectrum cache was
 *  thrashing and by a few per cent once it was not -- and CUDA context setup
 *  costs another ~0.5 s per process, which is what a whole single-spectrum
 *  fit takes.  The back-end is still built; GAEL_GPU=1 selects it.          */
static bool use_gpu_convolution()
{
    static const bool v = [] {
        const char* e = std::getenv("GAEL_GPU");
        return e && *e && std::string(e) != "0";
    }();
    return v;
}
#endif

// Main resolution degradation function with caching
Vector degrade_resolution(const Vector& lam,
                         const Vector& flux,
                         double resOffset,
                         double resSlope)
{
    {
        const char* cv = std::getenv("GAEL_CONV");
        if (cv && std::string(cv) == "isis")
            return degrade_resolution_isis(lam, flux, resOffset, resSlope);
    }
#ifdef GAEL_USE_CUDA
    if (use_gpu_convolution())
        return degrade_resolution_cuda(lam, flux, resOffset, resSlope);
#endif
    return degrade_resolution(lam, flux, lam, resOffset, resSlope);
}

/*  Same convolution, reported on `lam_out`.  The GAEL_CONV=isis and CUDA
 *  back-ends above are single-grid diagnostics and are deliberately not
 *  reachable from here.                                                      */
Vector degrade_resolution(const Vector& lam,
                          const Vector& flux,
                          const Vector& lam_out,
                          double resOffset,
                          double resSlope)
{
    if (lam.size() < 2 || lam_out.size() == 0) return flux;

    // Create cache key
    CacheKey key = make_cache_key(lam, lam_out, resOffset, resSlope);

    /*  Single-flight: one thread computes a given weight set, the others wait
     *  for it rather than each building their own half-gigabyte copy.  With
     *  the Jacobian's columns running in parallel, a cold key is missed by
     *  every thread at the same moment.                                     */
    {
        std::unique_lock<std::mutex> lock(g_cacheMutex);
        for (;;) {
            auto it = g_weightCache.find(key);
            if (it != g_weightCache.end()) {
                WeightsPtr hit = it->second.first;   // keep alive past the lock
                touch_key(key);
                lock.unlock();
                return apply_weights(flux, *hit);
            }
            if (std::find(g_inflight.begin(), g_inflight.end(), key)
                    == g_inflight.end()) {
                g_inflight.push_back(key);           // this thread computes it
                break;
            }
            g_inflightCv.wait(lock);
        }
    }

    auto drop_inflight = [&] {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        auto it = std::find(g_inflight.begin(), g_inflight.end(), key);
        if (it != g_inflight.end()) g_inflight.erase(it);
        g_inflightCv.notify_all();
    };

    std::vector<WeightSegment> weights;
    try {
        weights = compute_weights(lam, lam_out, resOffset, resSlope);
    } catch (...) {
        drop_inflight();
        throw;
    }

    // Insert into cache with LRU enforcement
    WeightsPtr w;
    {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        w = insert_cache_entry(key, std::move(weights));
        auto it = std::find(g_inflight.begin(), g_inflight.end(), key);
        if (it != g_inflight.end()) g_inflight.erase(it);
        g_inflightCv.notify_all();
    }

    // Apply weights
    return apply_weights(flux, *w);
}

// Optional: Function to precompute weights for known grids
void precompute_weights(const Vector& lam, double resOffset, double resSlope) {
    CacheKey key = make_cache_key(lam, lam, resOffset, resSlope);

    std::lock_guard<std::mutex> lock(g_cacheMutex);
    if (g_weightCache.find(key) == g_weightCache.end()) {
        insert_cache_entry(key, compute_weights(lam, lam, resOffset, resSlope));
    }
}

// Optional: Clear cache if memory becomes an issue
void clear_weight_cache() {
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    g_weightCache.clear();
    g_lruList.clear();
    g_cacheBytes = 0;
}

// Optional: Get cache statistics
std::size_t get_cache_size() {
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    return g_weightCache.size();
}

} // namespace specfit