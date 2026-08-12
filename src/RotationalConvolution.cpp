// RotationalConvolution.cpp
#include "specfit/RotationalConvolution.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <cstdint>

namespace specfit {

/* ------------- Gray (2005) rotational profile ------------------------ */
static inline double rot_profile(double x, double eps)
{
    if (std::abs(x) >= 1.0) return 0.0;
    const double t   = 1.0 - x*x;
    const double num = 2.0*(1.0-eps)*std::sqrt(t) + 0.5*M_PI*eps*t;
    const double den = M_PI*(1.0 - eps/3.0);
    return num / den;
}

/* ------------- Cached weight structure for fast application ---------- */
struct RotationalWeights {
    struct WeightEntry {
        std::size_t idx;     // Index in input flux array
        double weight;       // Interpolation weight
    };
    
    // For each output point, store which input points contribute and their weights
    std::vector<std::vector<WeightEntry>> weights_per_point;
};

/* ------------- Cache key including wavelength grid signature --------- */
struct GridCacheKey {
    double lam_start;
    double lam_end;
    double avg_delta;
    std::size_t n_points;
    double vsini_kms;
    double epsilon;
    int n_kernel;
    
    bool operator==(const GridCacheKey& other) const {
        return std::abs(lam_start - other.lam_start) < 1e-10 &&
               std::abs(lam_end - other.lam_end) < 1e-10 &&
               std::abs(avg_delta - other.avg_delta) < 1e-10 &&
               n_points == other.n_points &&
               std::abs(vsini_kms - other.vsini_kms) < 1e-10 &&
               std::abs(epsilon - other.epsilon) < 1e-10 &&
               n_kernel == other.n_kernel;
    }
};

struct GridCacheKeyHash {
    std::size_t operator()(const GridCacheKey& k) const {
        auto h1 = std::hash<double>{}(k.lam_start);
        auto h2 = std::hash<double>{}(k.lam_end);
        auto h3 = std::hash<double>{}(k.avg_delta);
        auto h4 = std::hash<std::size_t>{}(k.n_points);
        auto h5 = std::hash<double>{}(k.vsini_kms);
        auto h6 = std::hash<double>{}(k.epsilon);
        auto h7 = std::hash<int>{}(k.n_kernel);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
    }
};

/* ------------- LRU Cache implementation ------------------------------ */
class LRUCache {
private:
    static constexpr std::size_t MAX_CACHE_SIZE = 25;
    
    // List to maintain LRU order (most recently used at front)
    using KeyList = std::list<GridCacheKey>;
    KeyList lru_list;
    
    /*  Shared ownership, not a value: `get` used to hand back a full copy of
     *  the weight structure (one small vector per model pixel) on every cache
     *  hit, which showed up as ~12 % of a fit.  A shared_ptr also makes
     *  eviction safe while a caller is still applying the weights.          */
    using WeightsPtr = std::shared_ptr<const RotationalWeights>;

    // Map from key to (weights, iterator in lru_list)
    std::unordered_map<GridCacheKey,
                       std::pair<WeightsPtr, KeyList::iterator>,
                       GridCacheKeyHash> cache_map;
    
    mutable std::mutex mutex;
    
    void move_to_front(const GridCacheKey& key, KeyList::iterator it) {
        // Move this key to front (most recently used)
        lru_list.splice(lru_list.begin(), lru_list, it);
    }
    
    void evict_lru() {
        // Remove least recently used (back of list)
        if (!lru_list.empty()) {
            auto lru_key = lru_list.back();
            cache_map.erase(lru_key);
            lru_list.pop_back();
        }
    }
    
public:
    // Try to get weights from cache; nullptr on a miss
    WeightsPtr get(const GridCacheKey& key) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = cache_map.find(key);
        if (it == cache_map.end()) {
            return nullptr;  // Cache miss
        }

        // Cache hit - update LRU order
        move_to_front(key, it->second.second);
        return it->second.first;
    }

    // Put weights into cache
    void put(const GridCacheKey& key, WeightsPtr weights) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = cache_map.find(key);
        if (it != cache_map.end()) {
            // Key already exists - update and move to front
            it->second.first = std::move(weights);
            move_to_front(key, it->second.second);
            return;
        }

        // Check if cache is full
        if (cache_map.size() >= MAX_CACHE_SIZE) {
            evict_lru();
        }

        // Insert new entry at front of LRU list
        lru_list.push_front(key);
        cache_map[key] = {std::move(weights), lru_list.begin()};
    }
    
    // Clear all cache entries
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        cache_map.clear();
        lru_list.clear();
    }
    
    // Get current cache size
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return cache_map.size();
    }
    
    // Check if key exists (for precompute)
    bool contains(const GridCacheKey& key) const {
        std::lock_guard<std::mutex> lock(mutex);
        return cache_map.find(key) != cache_map.end();
    }
};

/* ------------- Global LRU cache -------------------------------------- */
static LRUCache g_weightCache;

/* ------------- Create cache key from wavelength grid ----------------- */
static GridCacheKey make_grid_key(const Vector& lam, 
                                  double vsini_kms, 
                                  double epsilon, 
                                  int n_kernel)
{
    GridCacheKey key;
    key.lam_start = lam[0];
    key.lam_end = lam[lam.size() - 1];
    key.n_points = lam.size();
    key.avg_delta = (key.lam_end - key.lam_start) / (key.n_points - 1);
    key.vsini_kms = vsini_kms;
    key.epsilon = epsilon;
    key.n_kernel = n_kernel;
    return key;
}

/* ------------- Compute interpolation weights for a grid -------------- */
static RotationalWeights compute_weights(const Vector& lam,
                                        double vsini_kms,
                                        double epsilon,
                                        int n_kernel)
{
    const std::ptrdiff_t N = lam.size();
    RotationalWeights weights;
    weights.weights_per_point.resize(N);
    
    if (n_kernel <= 0) n_kernel = 81;
    if (n_kernel % 2 == 0) n_kernel++;
    
    // Compute velocity kernel
    Vector vel_shift(n_kernel);
    Vector vel_kernel(n_kernel);
    
    for (int i = 0; i < n_kernel; ++i) {
        double x = -1.0 + 2.0 * i / (n_kernel - 1);
        vel_shift[i] = x * vsini_kms;
        vel_kernel[i] = rot_profile(x, epsilon);
    }
    
    // Normalize kernel
    double kernel_sum = vel_kernel.sum();
    if (kernel_sum > 0) {
        vel_kernel /= kernel_sum;
    }
    
    const double c_km = 299792.458;  // Speed of light in km/s
    
    // Precompute interpolation weights for each output wavelength
    #pragma omp parallel for schedule(dynamic, 32)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        double lam_center = lam[i];
        std::vector<RotationalWeights::WeightEntry> point_weights;
        point_weights.reserve(n_kernel * 2);  // Estimate
        
        double sum_weight = 0.0;
        
        // For each kernel point
        for (int k = 0; k < n_kernel; ++k) {
            double dlam = lam_center * vel_shift[k] / c_km;
            double lam_k = lam_center + dlam;
            
            // Find where lam_k falls in the wavelength grid
            if (lam_k >= lam[0] && lam_k <= lam[N-1]) {
                // Binary search for interpolation position
                auto it = std::lower_bound(lam.begin(), lam.end(), lam_k);
                std::ptrdiff_t j = std::distance(lam.begin(), it);
                
                if (j == 0) j = 1;
                if (j >= N) j = N - 1;
                
                // Linear interpolation weights
                double t = (lam_k - lam[j-1]) / (lam[j] - lam[j-1]);
                double w1 = vel_kernel[k] * (1.0 - t);
                double w2 = vel_kernel[k] * t;
                
                // Store non-zero weights
                if (w1 > 1e-12) {
                    point_weights.push_back({static_cast<std::size_t>(j-1), w1});
                    sum_weight += w1;
                }
                if (w2 > 1e-12) {
                    point_weights.push_back({static_cast<std::size_t>(j), w2});
                    sum_weight += w2;
                }
            }
        }
        
        // Normalize weights
        if (sum_weight > 0) {
            for (auto& w : point_weights) {
                w.weight /= sum_weight;
            }
        } else {
            // Fallback: identity at this point
            point_weights.push_back({static_cast<std::size_t>(i), 1.0});
        }
        
        // Consolidate weights for same indices
        std::sort(point_weights.begin(), point_weights.end(), 
                 [](const auto& a, const auto& b) { return a.idx < b.idx; });
        
        std::vector<RotationalWeights::WeightEntry> consolidated;
        for (const auto& w : point_weights) {
            if (!consolidated.empty() && consolidated.back().idx == w.idx) {
                consolidated.back().weight += w.weight;
            } else {
                consolidated.push_back(w);
            }
        }
        
        weights.weights_per_point[i] = std::move(consolidated);
    }
    
    return weights;
}

/* ------------- Fast application of precomputed weights --------------- */
static Vector apply_weights(const Vector& flux, const RotationalWeights& weights)
{
    const std::ptrdiff_t N = weights.weights_per_point.size();
    assert(flux.size() >= N && "Flux array too small for weights!");

    Vector out(N);
    const double* flux_data = flux.data();
    
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        double sum = 0.0;
        
        // Simple dot product with precomputed weights
        for (const auto& w : weights.weights_per_point[i]) {
            assert(w.idx < flux.size() && "Weight index out of bounds!");
            sum += w.weight * flux_data[w.idx];
        }
        
        out[i] = sum;
    }
    
    return out;
}

/* ------------------- Main interface with LRU caching ----------------- */
Vector rotational_broaden(const Vector& lam,
                         const Vector& flux,
                         double vsini_kms,
                         double epsilon,
                         int n_kernel)
{
    if (lam.size() == 0 || vsini_kms <= 0.0) return flux;
    assert(flux.size() == lam.size() && "Flux and lambda size mismatch!");
    assert(flux.data() != nullptr && "Flux data is null!");
    
    // Create cache key including wavelength grid signature
    GridCacheKey key = make_grid_key(lam, vsini_kms, epsilon, n_kernel);
    
    // Try to get from cache
    if (auto cached = g_weightCache.get(key)) {
        // Cache hit! Just apply precomputed weights
        return apply_weights(flux, *cached);
    }

    // Cache miss - compute weights
    auto weights = std::make_shared<const RotationalWeights>(
        compute_weights(lam, vsini_kms, epsilon, n_kernel));

    // Store in cache (will handle LRU eviction if needed)
    g_weightCache.put(key, weights);

    // Apply weights
    return apply_weights(flux, *weights);
}

/* ===================================================================== *
 *  Two-grid variant: integrate over `lam_in`, report on `lam_out`.
 *
 *  Storage is one contiguous run of input indices per output point, held in
 *  a single flat array.  The kernel taps of one output point are monotone in
 *  wavelength, so the indices they hit are monotone too: a cursor walks them
 *  once, which removes both the per-tap binary search and the per-point sort
 *  that dominate the single-grid routine on a 600 k-point grid.
 * ===================================================================== */
namespace {

struct RotSegments {
    std::vector<std::uint32_t> start;   // n_out : first input index of the run
    std::vector<std::size_t>   off;     // n_out + 1 : offsets into w
    std::vector<double>        w;       // flat weights

    /* ---- shift-invariant core (see compute_segments) -------------------- *
     *  When the two grids are the same log-uniform grid, every output point
     *  in [core_lo, core_hi) has the identical run of weights at the identical
     *  offset from itself, so one copy is kept instead of n_out of them and
     *  `w` only carries the two edges.                                      */
    bool           uniform  = false;
    std::ptrdiff_t core_lo  = 0;        // first output point using `core_w`
    std::ptrdiff_t core_hi  = 0;        // one past the last
    std::ptrdiff_t core_off = 0;        // input index of core_w[0] is i+core_off
    std::vector<double> core_w;

    std::size_t bytes() const {
        return start.capacity() * sizeof(std::uint32_t)
             + off.capacity()   * sizeof(std::size_t)
             + w.capacity()     * sizeof(double)
             + core_w.capacity()* sizeof(double);
    }
};

/* ------------------------------------------------------------------ *
 *  Is `lam` uniform in ln(lambda)?
 *
 *  conv_grid() (SyntheticModel.cpp) builds the metal path's convolution grid
 *  as lam_lo * exp(i * step), which is what makes the answer yes there.  The
 *  test is deliberately strict -- a grid that is only nearly log-uniform must
 *  take the general path, because the whole point of the fast one is that one
 *  set of weights stands for every output point.
 * ------------------------------------------------------------------ */
bool is_log_uniform(const Vector& lam, double& step_out)
{
    const std::ptrdiff_t n = lam.size();
    if (n < 3 || lam[0] <= 0.0) return false;

    const double step = std::log(lam[n - 1] / lam[0]) / double(n - 1);
    if (!(step > 0.0)) return false;

    /*  Relative tolerance on each interval.  Rounding in exp() leaves a few
     *  ULP; 1e-9 of the step is far tighter than that and far looser than
     *  anything that would let a genuinely non-uniform grid through.        */
    const double tol = 1e-9 * step;
    for (std::ptrdiff_t i = 1; i < n; ++i) {
        if (!(lam[i] > lam[i - 1])) return false;
        if (std::abs(std::log(lam[i] / lam[i - 1]) - step) > tol) return false;
    }
    step_out = step;
    return true;
}

RotSegments compute_segments(const Vector& lam_in,
                             const Vector& lam_out,
                             double vsini_kms,
                             double epsilon,
                             int    n_kernel)
{
    const std::ptrdiff_t M = lam_in.size();
    const std::ptrdiff_t N = lam_out.size();

    if (n_kernel <= 0) n_kernel = 81;
    if (n_kernel % 2 == 0) n_kernel++;

    /*  Same profile as the single-grid routine: taps uniform in x = v/vsini
     *  over [-1, 1], so they are ascending in wavelength.                   */
    std::vector<double> vel_shift(n_kernel), vel_kernel(n_kernel);
    for (int k = 0; k < n_kernel; ++k) {
        const double x = -1.0 + 2.0 * k / (n_kernel - 1);
        vel_shift[k]  = x * vsini_kms;
        vel_kernel[k] = rot_profile(x, epsilon);
    }
    double ksum = 0.0;
    for (double v : vel_kernel) ksum += v;
    if (ksum > 0.0) for (double& v : vel_kernel) v /= ksum;

    constexpr double c_km = 299792.458;
    const double f_lo = 1.0 + vel_shift[0] / c_km;
    const double f_hi = 1.0 + vel_shift[n_kernel - 1] / c_km;

    const double*  li = lam_in.data();
    const double*  lo = lam_out.data();
    const double   lam_first = li[0];
    const double   lam_last  = li[M - 1];

    RotSegments S;
    S.start.resize(static_cast<std::size_t>(N));
    S.off.resize(static_cast<std::size_t>(N) + 1);

    /*  lower_bound index, clamped exactly as the single-grid routine does
     *  before it forms the two-point interpolation.                         */
    auto lb = [&](double x) -> std::ptrdiff_t {
        return std::lower_bound(li, li + M, x) - li;
    };

    /* ---- pass 1: the input-index run each output point touches --------- */
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        const double a = std::max(lo[i] * f_lo, lam_first);
        const double b = std::min(lo[i] * f_hi, lam_last);

        std::ptrdiff_t ja = lb(a); if (ja == 0) ja = 1; if (ja >= M) ja = M - 1;
        std::ptrdiff_t jb = lb(b); if (jb == 0) jb = 1; if (jb >= M) jb = M - 1;
        if (jb < ja) jb = ja;

        S.start[static_cast<std::size_t>(i)] =
            static_cast<std::uint32_t>(ja - 1);
        S.off[static_cast<std::size_t>(i) + 1] =
            static_cast<std::size_t>(jb - ja + 2);
    }

    /*  Reciprocal input spacings.  Every one of the n_kernel taps of every
     *  output point interpolates inside one input interval, so this turns
     *  ~n_out * n_kernel divisions into that many multiplies for M extra.   */
    std::vector<double> inv_dl(static_cast<std::size_t>(M), 0.0);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t j = 1; j < M; ++j)
        inv_dl[static_cast<std::size_t>(j)] = 1.0 / (li[j] - li[j - 1]);

    /*  The taps of output point i, accumulated into `wp` (which must hold the
     *  point's whole run, zeroed, starting at input index `s`).  Written once
     *  and used for the per-point storage and for the shared core alike, so
     *  the two cannot drift apart.                                          */
    auto fill_point = [&](std::ptrdiff_t i, std::ptrdiff_t s, std::size_t len,
                          double* wp) {
        std::ptrdiff_t cur = s + 1;          // running lower_bound cursor
        double sum = 0.0;

        for (int k = 0; k < n_kernel; ++k) {
            const double lam_k = lo[i] * (1.0 + vel_shift[k] / c_km);
            if (lam_k < lam_first || lam_k > lam_last) continue;

            while (cur < M && li[cur] < lam_k) ++cur;
            std::ptrdiff_t j = cur;
            if (j == 0) j = 1;
            if (j >= M) j = M - 1;

            const double t  = (lam_k - li[j - 1]) * inv_dl[static_cast<std::size_t>(j)];
            const double w1 = vel_kernel[k] * (1.0 - t);
            const double w2 = vel_kernel[k] * t;

            if (w1 > 1e-12) { wp[j - 1 - s] += w1; sum += w1; }
            if (w2 > 1e-12) { wp[j     - s] += w2; sum += w2; }
        }

        if (sum > 0.0) {
            const double inv = 1.0 / sum;
            for (std::size_t q = 0; q < len; ++q) wp[q] *= inv;
        } else {
            /*  Every tap fell outside the input grid.  Cannot happen while
             *  lam_out lies inside lam_in, but keep the single-grid routine's
             *  identity fallback rather than emitting a zero.               */
            wp[0] = 1.0;
        }
    };

    /* ---- shift-invariant core ------------------------------------------ *
     *  On one grid that is uniform in ln(lambda), the taps of output point i
     *  sit at ln(lam_i) + ln(1 + v_k/c): a fixed offset in log space, hence a
     *  fixed offset in index, with fixed interpolation fractions.  Every
     *  interior point therefore has the *same* run of weights at the same
     *  offset from itself, and the per-point table is n_out identical copies
     *  of it -- 1.07 GB of them for an 18-arm UVES metal fit, which is four
     *  times the cache budget, so it was being rebuilt on nearly every call
     *  and then streamed in full on every application.
     *
     *  Which points qualify is read off pass 1 rather than derived: the core
     *  is the run of points whose window has the same shape and the same
     *  offset as the middle one's, which excludes exactly the two ends, where
     *  taps fall off the grid and the normalisation differs.  If the grid is
     *  not log-uniform this collapses to nothing and the general path below
     *  runs unchanged.                                                       */
    double log_step = 0.0;
    const bool same_grid = (M == N) && (li == lo);
    if (same_grid && is_log_uniform(lam_in, log_step)) {
        const std::ptrdiff_t mid = N / 2;
        const std::ptrdiff_t s_mid   = S.start[static_cast<std::size_t>(mid)];
        const std::size_t    len_mid = S.off[static_cast<std::size_t>(mid) + 1];

        std::ptrdiff_t lo_i = mid, hi_i = mid + 1;
        auto same_shape = [&](std::ptrdiff_t i) {
            return S.off[static_cast<std::size_t>(i) + 1] == len_mid &&
                   static_cast<std::ptrdiff_t>(
                       S.start[static_cast<std::size_t>(i)]) - i == s_mid - mid;
        };
        while (lo_i > 0        && same_shape(lo_i - 1)) --lo_i;
        while (hi_i < N        && same_shape(hi_i))     ++hi_i;

        /*  Only worth the second representation if it covers most of the grid;
         *  otherwise the edges would carry the cost twice.                   */
        if (hi_i - lo_i > N / 2) {
            S.uniform  = true;
            S.core_lo  = lo_i;
            S.core_hi  = hi_i;
            S.core_off = s_mid - mid;
            S.core_w.assign(len_mid, 0.0);
            fill_point(mid, s_mid, len_mid, S.core_w.data());

            /*  Flat storage for the two edges only.  Interior points keep an
             *  empty run so that `off` stays a valid prefix sum.             */
            for (std::ptrdiff_t i = lo_i; i < hi_i; ++i)
                S.off[static_cast<std::size_t>(i) + 1] = 0;
        }
    }

    /* ---- prefix sum -> flat offsets ------------------------------------ */
    S.off[0] = 0;
    for (std::ptrdiff_t i = 0; i < N; ++i)
        S.off[static_cast<std::size_t>(i) + 1] +=
            S.off[static_cast<std::size_t>(i)];
    S.w.assign(S.off[static_cast<std::size_t>(N)], 0.0);

    /* ---- pass 2: accumulate the taps ----------------------------------- */
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        const std::size_t base = S.off[static_cast<std::size_t>(i)];
        const std::size_t len  = S.off[static_cast<std::size_t>(i) + 1] - base;
        if (len == 0) continue;                    // covered by the core
        fill_point(i, S.start[static_cast<std::size_t>(i)], len,
                   S.w.data() + base);
    }

    return S;
}

Vector apply_segments(const Vector& flux, const RotSegments& S)
{
    const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(S.start.size());
    Vector out(N);
    const double* f = flux.data();

    /*  The shift-invariant interior: one kernel, small enough to sit in L1,
     *  slid along the flux.  The general branch below reads a distinct run of
     *  weights per output point, which on the metal path means streaming a
     *  gigabyte of table to produce five megabytes of spectrum.             */
    if (S.uniform) {
        const double*      kw  = S.core_w.data();
        const std::size_t  klen = S.core_w.size();
        const std::ptrdiff_t koff = S.core_off;

        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = S.core_lo; i < S.core_hi; ++i) {
            const double* fp = f + i + koff;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (std::size_t q = 0; q < klen; ++q) sum += kw[q] * fp[q];
            out[i] = sum;
        }
    }

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        if (S.uniform && i >= S.core_lo && i < S.core_hi) continue;
        const std::size_t b   = S.off[static_cast<std::size_t>(i)];
        const std::size_t e   = S.off[static_cast<std::size_t>(i) + 1];
        const double*     fp  = f + S.start[static_cast<std::size_t>(i)];
        const double*     wp  = S.w.data() + b;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (std::size_t q = 0; q < e - b; ++q) sum += wp[q] * fp[q];
        out[i] = sum;
    }
    return out;
}

/* ---- byte-budgeted LRU over (input grid, output grid, vsini, ...) ----- */
/*  As in Resolution.cpp: the input grid here is the union of the species
 *  grids, and two components of a binary can have unions that agree on their
 *  range but not their interior, so an interior sample goes into the key.   */
struct SegKey {
    GridCacheKey in;
    double       in_mid;
    std::size_t  n_out;
    double       out_start;
    double       out_end;

    bool operator==(const SegKey& o) const {
        return in == o.in && n_out == o.n_out &&
               std::abs(in_mid    - o.in_mid)    < 1e-10 &&
               std::abs(out_start - o.out_start) < 1e-10 &&
               std::abs(out_end   - o.out_end)   < 1e-10;
    }
};
struct SegKeyHash {
    std::size_t operator()(const SegKey& k) const {
        return GridCacheKeyHash{}(k.in) ^ (std::hash<std::size_t>{}(k.n_out) << 1)
             ^ (std::hash<double>{}(k.out_start) << 2)
             ^ (std::hash<double>{}(k.out_end)   << 3)
             ^ (std::hash<double>{}(k.in_mid)    << 4);
    }
};

using SegPtr = std::shared_ptr<const RotSegments>;

std::unordered_map<SegKey, std::pair<SegPtr, std::size_t>, SegKeyHash> g_segCache;
std::list<SegKey> g_segLru;
std::mutex        g_segMtx;
std::size_t       g_segBytes = 0;
std::vector<SegKey>     g_segInflight;
std::condition_variable g_segInflightCv;
constexpr std::size_t SEG_MAX_BYTES = 256ull << 20;

} // anonymous namespace

Vector rotational_broaden(const Vector& lam_in,
                          const Vector& flux,
                          const Vector& lam_out,
                          double vsini_kms,
                          double epsilon,
                          int    n_kernel)
{
    if (lam_in.size() < 2 || lam_out.size() == 0) return flux;
    if (vsini_kms <= 0.0) {
        /*  No rotation: still has to land on lam_out, so interpolate. */
        Vector out(lam_out.size());
        const double* li = lam_in.data();
        const std::ptrdiff_t M = lam_in.size();
        std::ptrdiff_t j = 1;
        for (Eigen::Index i = 0; i < lam_out.size(); ++i) {
            const double x = lam_out[i];
            while (j < M - 1 && li[j] < x) ++j;
            const double t = (x - li[j - 1]) / (li[j] - li[j - 1]);
            out[i] = flux[j - 1] + t * (flux[j] - flux[j - 1]);
        }
        return out;
    }

    SegKey key;
    key.in        = make_grid_key(lam_in, vsini_kms, epsilon, n_kernel);
    key.in_mid    = lam_in[lam_in.size() / 3];
    key.n_out     = static_cast<std::size_t>(lam_out.size());
    key.out_start = lam_out[0];
    key.out_end   = lam_out[lam_out.size() - 1];

    /*  Single-flight, as in Resolution.cpp: with the Jacobian's columns in
     *  parallel every thread misses a cold key at the same moment, and each
     *  would otherwise build its own copy of the segments.                  */
    {
        std::unique_lock<std::mutex> lk(g_segMtx);
        for (;;) {
            auto it = g_segCache.find(key);
            if (it != g_segCache.end()) {
                g_segLru.remove(key);
                g_segLru.push_front(key);
                SegPtr hit = it->second.first;   // keep alive past the lock
                lk.unlock();
                return apply_segments(flux, *hit);
            }
            if (std::find(g_segInflight.begin(), g_segInflight.end(), key)
                    == g_segInflight.end()) {
                g_segInflight.push_back(key);
                break;
            }
            g_segInflightCv.wait(lk);
        }
    }

    struct FlightGuard {
        SegKey key; bool armed = true;
        ~FlightGuard() {
            if (!armed) return;
            std::lock_guard<std::mutex> lk(g_segMtx);
            auto it = std::find(g_segInflight.begin(), g_segInflight.end(), key);
            if (it != g_segInflight.end()) g_segInflight.erase(it);
            g_segInflightCv.notify_all();
        }
    } flight{key};

    auto seg = std::make_shared<const RotSegments>(
        compute_segments(lam_in, lam_out, vsini_kms, epsilon, n_kernel));

    {
        std::lock_guard<std::mutex> lk(g_segMtx);
        flight.armed = false;
        {
            auto it = std::find(g_segInflight.begin(), g_segInflight.end(), key);
            if (it != g_segInflight.end()) g_segInflight.erase(it);
            g_segInflightCv.notify_all();
        }
        if (g_segCache.find(key) == g_segCache.end()) {
            const std::size_t nb = seg->bytes();
            g_segCache[key] = {seg, nb};
            g_segLru.push_front(key);
            g_segBytes += nb;
            /*  Two entries are kept whatever the budget says, not one.
             *  vsini is a fitted parameter, so every Jacobian asks for the
             *  weights at vsini and again at vsini+h; with room for a single
             *  entry those two evict each other and each gets rebuilt from
             *  scratch.  That costs nothing on the shift-invariant path (the
             *  tables are ~10 MB there) and is the difference between a cache
             *  and a treadmill on an input grid that is not log-uniform.    */
            while (g_segCache.size() > 2 && g_segBytes > SEG_MAX_BYTES) {
                const SegKey victim = g_segLru.back();
                if (auto it = g_segCache.find(victim); it != g_segCache.end()) {
                    g_segBytes -= std::min(g_segBytes, it->second.second);
                    g_segCache.erase(it);
                }
                g_segLru.pop_back();
            }
        }
    }

    return apply_segments(flux, *seg);
}

/* ------------- Optional: Precompute weights for known grids ---------- */
void precompute_rotational_weights(const Vector& lam,
                                   double vsini_kms,
                                   double epsilon,
                                   int n_kernel)
{
    GridCacheKey key = make_grid_key(lam, vsini_kms, epsilon, n_kernel);
    
    if (!g_weightCache.contains(key)) {
        g_weightCache.put(key, std::make_shared<const RotationalWeights>(
            compute_weights(lam, vsini_kms, epsilon, n_kernel)));
    }
}

/* ------------- Optional: Clear cache --------------------------------- */
void clear_rotational_cache()
{
    g_weightCache.clear();
}

/* ------------- Optional: Get cache statistics ------------------------ */
std::size_t get_rotational_cache_size()
{
    return g_weightCache.size();
}

} // namespace specfit