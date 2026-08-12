/* ===================================================================== *
 *  src/SpectrumCache.cpp
 * ===================================================================== */
#include "specfit/SpectrumCache.hpp"

#include <algorithm>

namespace specfit {

/* -------- singleton -------------------------------------------------- */
SpectrumCache& SpectrumCache::instance()
{
    static SpectrumCache inst;
    return inst;
}

/* -------- simple helpers -------------------------------------------- */
void SpectrumCache::reserve(std::size_t n)
{
    std::unique_lock lk(mtx_);
    cache_.reserve(n);
}
void SpectrumCache::set_capacity(std::size_t n)
{
    std::unique_lock lk(mtx_);
    max_entries_ = n;                 // 0 == memory budget decides alone
    evict_if_needed_();
}
void SpectrumCache::set_memory_budget(std::size_t bytes)
{
    std::unique_lock lk(mtx_);
    max_bytes_ = bytes;
    evict_if_needed_();
}
void SpectrumCache::ensure_memory_budget(std::size_t bytes)
{
    std::unique_lock lk(mtx_);
    if (bytes > max_bytes_) max_bytes_ = bytes;
}
void SpectrumCache::clear()
{
    std::unique_lock lk(mtx_);
    cache_.clear();
    lru_.clear();
    bytes_ = 0;
}
std::size_t SpectrumCache::size() const
{
    std::shared_lock lk(mtx_);
    return cache_.size();
}
std::size_t SpectrumCache::bytes_used() const
{
    std::shared_lock lk(mtx_);
    return bytes_;
}

/* -------- try_get ---------------------------------------------------- */
SpectrumPtr SpectrumCache::try_get(std::size_t hash) const
{
    std::unique_lock lk(mtx_);
    auto it = cache_.find(hash);
    if (it == cache_.end()) return nullptr;
    touch_(it);                      // update LRU even on read
    return it->second.sp;
}

/* -------- back-compat helpers --------------------------------------- */
void SpectrumCache::insert(std::size_t hash, Spectrum spec)
{
    insert_if_absent(hash, [&]{ return std::move(spec); });
}
bool SpectrumCache::fetch(std::size_t hash, Spectrum& out) const
{
    if (auto sp = try_get(hash)) {
        out = *sp;                   // deep copy for legacy callers
        return true;
    }
    return false;
}

/* ===================================================================== *
 *            internal L-R-U helpers (private)
 * ===================================================================== */
std::size_t SpectrumCache::footprint_(const Spectrum& s)
{
    return sizeof(Spectrum)
         + static_cast<std::size_t>(s.lambda.size()   +
                                    s.flux.size()     +
                                    s.sigma.size()    +
                                    s.cont.size()     +
                                    s.cont_den.size()) * sizeof(Real)
         + s.ignoreflag.size() * sizeof(int);
}

void SpectrumCache::touch_(typename Map::iterator it) const
{
    lru_.splice(lru_.begin(), lru_, it->second.lru_pos);
}
void SpectrumCache::evict_if_needed_()
{
    /*  Always keep at least one entry, whatever the budget says: evicting the
     *  spectrum a caller is about to use would turn the cache into a slow
     *  no-op rather than a small one.                                       */
    auto over_budget = [&] {
        if (cache_.size() <= 1) return false;
        if (max_entries_ && cache_.size() > max_entries_) return true;
        return bytes_ > max_bytes_;
    };

    while (over_budget()) {
        std::size_t victim = lru_.back();
        lru_.pop_back();
        if (auto it = cache_.find(victim); it != cache_.end()) {
            bytes_ -= std::min(bytes_, it->second.bytes);
            cache_.erase(it);        // shared_ptr keeps data alive
        }
    }
}

} // namespace specfit