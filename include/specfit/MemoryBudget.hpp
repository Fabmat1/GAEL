#pragma once
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <string>

namespace specfit {

/* ===================================================================== *
 *  One ceiling for everything GAEL caches.
 *
 *  Every cache in the fit holds the same kind of thing -- a table that was
 *  expensive to build and is cheap to rebuild -- so each of them was given
 *  its own limit, and each limit was chosen against the fit its author had
 *  in front of them.  The instrumental weights got a byte budget after a
 *  metal fit reached 21 GB on them (see Resolution.cpp); the rotational
 *  weights kept an entry count, and an entry there is ~600 MB on a grid
 *  sampled as finely as synthe, so twenty-five of them is 15 GB that nothing
 *  in the process was accounting for.  Three independent limits also cannot
 *  add up to anything: they were 3 GB + unbounded + one sized per thread,
 *  on a machine whose total is 31 GB.
 *
 *  So the limit lives here instead, once, and the caches ask for their share
 *  of it.  It is derived from what the machine actually has free -- including
 *  the cgroup limit, because that is what kills the process in a container --
 *  and can be overridden by the CLI, the API or GAEL_MEM_BUDGET when the
 *  caller knows better (ASTRA holds the same spectra in its own GUI).
 *
 *  Deliberately *not* a shared pool with cross-cache eviction: that needs the
 *  caches to call into one another while holding their own mutexes, and this
 *  code convolves from several OpenMP threads at once.  Fixed shares of a
 *  common total give up some capacity and buy back a lock order that is
 *  obviously acyclic.
 * ===================================================================== */
class MemoryBudget {
public:
    static MemoryBudget& instance();

    /*  set_total and set_reserved are setup-time calls -- the CLI before the
     *  fit starts, GaelAPI once the datasets are known -- and are not
     *  synchronised against the readers.  Everything the caches touch while
     *  a fit is running is read-only or atomic.                            */

    /*  Bytes all GAEL caches together may hold, after the fit's own
     *  non-cache footprint (see set_reserved) has been taken off.          */
    std::size_t total() const;

    /*  Explicit override; 0 restores the automatic value.  Set from
     *  --mem-budget, from GaelSession, or from GAEL_MEM_BUDGET.            */
    void set_total(std::size_t bytes);

    /*  What the fit will hold *outside* the caches and cannot shrink: the
     *  Jacobian, the per-thread residual scratch, the observed spectra.
     *  Known once the datasets and the free-parameter set are (GaelAPI.cpp),
     *  and subtracted from the total, which is the difference between a
     *  cache budget that leaves room for the solver and one that does not.  */
    void set_reserved(std::size_t bytes);
    std::size_t reserved() const;

    /* ---- per-cache shares ------------------------------------------- *
     *  Weights, not guarantees: a cache that never fills its share simply
     *  leaves the memory unused.  Model surfaces get the largest slice
     *  because a joint fit keeps one live per thread per arm; the two weight
     *  tables are each one entry per (grid, spectrograph) pair.            */
    std::size_t model_spectra()      const;   // SpectrumCache
    std::size_t rotational_weights() const;   // RotationalConvolution
    std::size_t resolution_weights() const;   // Resolution

    /*  Physical memory this process may actually use: the smallest of
     *  MemAvailable and any cgroup limit, in bytes.  Swap deliberately does
     *  not count -- a fit that swaps has already cost more than rebuilding
     *  every table it holds.  0 when nothing could be determined.          */
    static std::size_t available_ram();

    /*  "4G", "512M", "2048" -- a bare number is MiB, matching --cache-mem.
     *  0 when the text is not a size, which the caller reports rather than
     *  silently treating as "no limit".                                     */
    static std::size_t parse_size(const std::string& text);

    /*  One line for the log, so a fit that runs lean says why.            */
    std::string describe() const;

    /*  Factor the shares are currently scaled by, 1 when nothing is being
     *  held back.  Only for reporting; see the pressure valve below.       */
    double pressure_scale() const;

private:
    MemoryBudget();

    std::size_t auto_total_() const;

    /* ---- pressure valve --------------------------------------------- *
     *  Everything above is a prediction, and the predictions do not cover
     *  the whole process: Eigen's temporaries, the corner spectra a thread
     *  is holding while it interpolates them, whatever the allocator has
     *  not returned to the OS.  Measured on the 8-arm UVES fit, peak RSS
     *  ran about 3 GB above the sum of the cache budgets, which is the
     *  difference between a fit that fits in memory and one that does not.
     *
     *  So the static figure is a starting point and this is the correction:
     *  re-read how much memory the *machine* has left, and if it is running
     *  out, scale the shares down so the caches give memory back.  It
     *  watches free memory rather than this process's RSS on purpose --
     *  what matters to the user is whether the desktop survives, and on the
     *  machine this was found on GAEL was not the only thing running.
     *
     *  Rate-limited: it reads /proc/meminfo, and the caches consult the
     *  budget on every insert.                                            */
    void   refresh_pressure_() const;

    std::size_t detected_    = 0;   // available_ram() at construction
    std::size_t explicit_    = 0;   // 0 == automatic
    std::size_t reserved_    = 0;

    /*  Mutable and atomic: the valve is an implementation detail of reading
     *  the budget, every reader is a const accessor, and the readers are the
     *  cache mutexes' critical sections on several OpenMP threads.  Relaxed
     *  throughout -- a reader that sees last tick's scale simply evicts on
     *  the next insert instead of this one.                                */
    mutable std::atomic<double>        scale_          {1.0};
    mutable std::atomic<std::uint64_t> last_sample_ms_ {0};
};

/* ---------------------------------------------------------------------- *
 *  What each cache is actually holding, for GAEL_MEM_STATS=1.
 *
 *  Defined next to the caches themselves rather than here, because each one
 *  keeps its own running total under its own mutex.  Reported at the end of a
 *  fit, they are how one tells a fit that is short of cache from a fit whose
 *  memory is somewhere the budget does not reach -- the two look identical
 *  from the outside and want opposite fixes.
 * ---------------------------------------------------------------------- */
std::size_t rotational_cache_bytes();
std::size_t rotational_cache_entries();
std::size_t resolution_cache_bytes();
std::size_t model_cache_bytes();

/*  One line naming all three plus the total.  Empty when GAEL_MEM_STATS is
 *  not set, so callers can log it unconditionally.                        */
std::string memory_stats_line();

} // namespace specfit
