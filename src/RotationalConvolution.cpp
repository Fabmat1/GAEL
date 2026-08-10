// RotationalConvolution.cpp
#include "specfit/RotationalConvolution.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <list>
#include <memory>
#include <mutex>
#include <cassert>

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