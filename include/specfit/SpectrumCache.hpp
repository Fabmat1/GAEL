/* ===================================================================== *
 *  include/specfit/SpectrumCache.hpp   ––  bounded L-R-U cache (safe)
 * ===================================================================== */
#pragma once
#include "Spectrum.hpp"

#include <ankerl/unordered_dense.h>
#include <condition_variable>
#include <list>
#include <mutex>
#include <memory>
#include <optional>
#include <iostream>
#include <shared_mutex>
#include <unordered_set>

namespace specfit {

using SpectrumPtr = std::shared_ptr<const Spectrum>;

/*
 * Thread–safe bounded cache with   L-R-U eviction  and
 *                                   shared ownership.
 *
 * Public API:
 *   • SpectrumPtr try_get(key)
 *   • SpectrumPtr insert_if_absent(key, producer)
 *   • legacy bool fetch(key, Spectrum&)   (unchanged behaviour)
 *
 * A caller that needs the old  “const Spectrum&”  can simply dereference
 * the pointer it gets back:
 *
 *     const auto sp = cache.insert_if_absent(h, …);
 *     use(*sp);                      // operator*  gives a  const&.
 */
class SpectrumCache
{
public:
    static SpectrumCache& instance();

    /* ------------ zero-copy read (shared ownership) ---------------- */
    SpectrumPtr try_get(std::size_t hash) const;

    /* ------------ insert-or-get (preferred) ------------------------ */
    template<typename Producer>
    SpectrumPtr insert_if_absent(std::size_t hash, Producer&& make);

    /* ------------ legacy helpers ----------------------------------- */
    [[deprecated("Use try_get() or insert_if_absent()")]]
    bool fetch(std::size_t hash, Spectrum& out) const;
    void insert(std::size_t hash, Spectrum spec);

    /* ------------ house-keeping ------------------------------------ */
    void reserve(std::size_t n);

    /*  Hard cap on the number of entries.  0 restores the automatic mode,
     *  in which only the memory budget below decides when to evict.       */
    void set_capacity(std::size_t n);

    /*  Cap on the bytes held by the cached spectra.  Entries differ in size
     *  by more than an order of magnitude -- a sliced model surface spectrum
     *  is ~10 000 points, a final synthetic one only as long as the observed
     *  grid -- so a fixed entry count is either wasteful or far too small
     *  depending on the fit.  Budgeting bytes makes one default correct for
     *  both.                                                               */
    void set_memory_budget(std::size_t bytes);

    /*  Raise the budget to `bytes` if it is currently lower, leaving a larger
     *  one (a --cache-mem on the command line) alone.  The caller that knows
     *  how many working sets will be live at once is the fit, not the CLI:
     *  evaluating the Jacobian's columns in parallel puts one model surface
     *  plus one synthetic spectrum per fitted arm in flight *per thread*, and
     *  a budget sized for a single one of those turns the cache into a
     *  treadmill -- every column rebuilding the surface it evicted while
     *  rebinning the previous arm.                                          */
    void ensure_memory_budget(std::size_t bytes);

    void clear();

    /* ------------ diagnostics -------------------------------------- */
    std::size_t size()       const;   // entries currently held
    std::size_t bytes_used() const;

private:
    SpectrumCache() = default;

    /* ---------- internal L-R-U bookkeeping ------------------------- */
    using LruList = std::list<std::size_t>;                 // keys
    struct Node {
        SpectrumPtr       sp;       // shared ownership
        LruList::iterator lru_pos;  // position in the list
        std::size_t       bytes;    // footprint of *sp when it was inserted
    };
    using Map = ankerl::unordered_dense::map<std::size_t, Node>;

    static std::size_t footprint_(const Spectrum& s);

    void touch_(typename Map::iterator it) const;   // header
    void evict_if_needed_();

    /* ---------- data members --------------------------------------- */
    mutable std::shared_mutex mtx_;
    mutable Map     cache_;
    mutable LruList lru_;

    /*  Keys some thread is building right now, and who to wake when it is
     *  done.  See insert_if_absent().                                      */
    mutable std::unordered_set<std::size_t>   inflight_;
    mutable std::condition_variable_any       inflight_cv_;
    std::size_t     bytes_       = 0;
    std::size_t     max_entries_ = 0;              // 0 == no entry cap
    std::size_t     max_bytes_   = 128ull << 20;   // see --cache-mem in main.cpp
};

/* ===================================================================== *
 *  template implementation
 * ===================================================================== */
/*  Single-flight: at most one thread builds any given key, the rest wait for
 *  it and take its result.
 *
 *  The build deliberately happens outside the lock -- it is the expensive part
 *  and holding the cache during it would serialise every unrelated lookup --
 *  but that alone lets N threads that miss the same key simultaneously each
 *  build their own copy and then throw all but one away.  That was harmless
 *  while the Jacobian was evaluated one column at a time; with its columns in
 *  parallel it is not.  Every dataset of a joint fit wants the *same* model
 *  surface, so the first thing the cost function does is have every thread
 *  miss the same key at once: on an 18-arm metal fit that meant sixteen
 *  concurrent builds of one ~700 k-point surface, sixteen times the work and
 *  sixteen times the transient memory.                                      */
template<typename Producer>
SpectrumPtr SpectrumCache::insert_if_absent(std::size_t hash,
                                            Producer&&   make)
{
    {
        std::unique_lock lk(mtx_);
        for (;;) {
            auto it = cache_.find(hash);
            if (it != cache_.end()) {
                touch_(it);
                return it->second.sp;      //  hit (possibly after waiting)
            }
            if (inflight_.insert(hash).second) break;   // this thread builds it
            inflight_cv_.wait(lk);                      // someone else is on it
        }
    }

    /*  Clears the in-flight marker however this scope is left, so a producer
     *  that throws wakes the waiters instead of deadlocking them.           */
    struct FlightGuard {
        const SpectrumCache* self; std::size_t key; bool armed = true;
        ~FlightGuard() {
            if (!armed) return;
            std::unique_lock lk(self->mtx_);
            self->inflight_.erase(key);
            self->inflight_cv_.notify_all();
        }
    } guard{this, hash};

    /* ---------- build spectrum outside any lock -------------------- */
    // NB: we build the *shared_ptr* directly
    SpectrumPtr new_sp = std::make_shared<Spectrum>(
                             std::forward<Producer>(make)() );

    /* ---------- insertion ------------------------------------------ */
    std::unique_lock lk(mtx_);
    guard.armed = false;                   // released under this same lock
    inflight_.erase(hash);

    auto it = cache_.find(hash);
    if (it != cache_.end()) {              // evicted-and-refilled in between
        touch_(it);
        inflight_cv_.notify_all();
        return it->second.sp;
    }

    auto lru_it = lru_.insert(lru_.begin(), hash);          // MRU front
    const std::size_t nbytes = footprint_(*new_sp);
    cache_.try_emplace(hash, Node{new_sp, lru_it, nbytes});
    bytes_ += nbytes;
    evict_if_needed_();
    inflight_cv_.notify_all();
    return new_sp;
}

} // namespace specfit