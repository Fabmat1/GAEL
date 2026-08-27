#include "specfit/MemoryBudget.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace specfit {

namespace {

constexpr std::size_t kMiB = 1ull << 20;

/*  Never shrink the caches below this, whatever the machine says.  A fit that
 *  cannot hold one model surface plus the weights that made it rebuilds both
 *  on every residual evaluation, which is slower by orders of magnitude
 *  rather than by a constant -- better to let the OS decide than to guarantee
 *  a fit that never finishes.                                               */
constexpr std::size_t kFloor = 192 * kMiB;

/*  Of what the machine has free, the share GAEL's caches may take.  The rest
 *  covers the allocations no cache knows about (Eigen temporaries, the FITS
 *  reader's buffers, the report writer) and, on a desktop, the fact that
 *  MemAvailable is an estimate rather than a promise.                       */
constexpr double kFractionOfAvailable = 0.55;

/*  An absolute ceiling.  Past a few GB these tables stop earning their keep:
 *  the entries a fit reuses are the handful it cycles through per iteration,
 *  and holding more only adds allocator and page-fault pressure (measured in
 *  main.cpp's --cache-mem note, where 512 MiB was already mildly worse than
 *  128 MiB on a five-spectrum fit).                                         */
constexpr std::size_t kCeiling = 8192 * kMiB;

std::size_t read_first_number(const std::string& path)
{
    std::ifstream f(path);
    if (!f) return 0;
    std::string tok;
    if (!(f >> tok)) return 0;
    if (tok == "max") return 0;
    errno = 0;
    const unsigned long long v = std::strtoull(tok.c_str(), nullptr, 10);
    return errno ? 0 : static_cast<std::size_t>(v);
}

/*  MemAvailable, in bytes.  It is the kernel's own estimate of what can be
 *  handed out without swapping, which is exactly the question here -- MemFree
 *  would ignore the reclaimable page cache and make every desktop look full. */
std::size_t mem_available()
{
    std::ifstream f("/proc/meminfo");
    if (!f) return 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream is(line.substr(13));
        unsigned long long kb = 0;
        if (is >> kb) return static_cast<std::size_t>(kb) * 1024ull;
        return 0;
    }
    return 0;
}

/*  The cgroup's limit, which is what actually kills the process in a
 *  container or under a systemd scope -- MemAvailable there still reports the
 *  host's.  v2 first, then v1.                                              */
std::size_t cgroup_limit()
{
    std::size_t best = 0;
    auto take = [&best](std::size_t v) {
        /*  memory.max on a machine with no limit reads as a number close to
         *  SIZE_MAX rather than "max" on some kernels; treat anything past a
         *  petabyte as "no limit".                                          */
        if (v == 0 || v > (1ull << 50)) return;
        if (best == 0 || v < best) best = v;
    };

    /*  cgroup v2: the process's own cgroup, and every ancestor, since the
     *  tightest of them is the one that applies.                            */
    std::ifstream self("/proc/self/cgroup");
    std::string line;
    while (std::getline(self, line)) {
        const auto pos = line.rfind("::");
        if (pos == std::string::npos) continue;
        std::string rel = line.substr(pos + 2);
        while (!rel.empty()) {
            take(read_first_number("/sys/fs/cgroup" + rel + "/memory.max"));
            take(read_first_number("/sys/fs/cgroup" + rel + "/memory.high"));
            const auto slash = rel.rfind('/');
            if (slash == std::string::npos || slash == 0) break;
            rel = rel.substr(0, slash);
        }
        take(read_first_number("/sys/fs/cgroup/memory.max"));
    }

    /*  cgroup v1. */
    take(read_first_number("/sys/fs/cgroup/memory/memory.limit_in_bytes"));

    return best;
}

} // namespace

/*  "4G", "512M", "2048" (MiB, matching --cache-mem), or 0. */
std::size_t MemoryBudget::parse_size(const std::string& s)
{
    if (s.empty()) return 0;
    char*             end = nullptr;
    const double      v   = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v <= 0.0) return 0;
    while (end && *end && std::isspace(static_cast<unsigned char>(*end))) ++end;
    const char unit = (end && *end) ? static_cast<char>(std::toupper(*end)) : 'M';
    switch (unit) {
        case 'G': return static_cast<std::size_t>(v * 1024.0 * kMiB);
        case 'K': return static_cast<std::size_t>(v * 1024.0);
        case 'B': return static_cast<std::size_t>(v);
        case 'M':
        default:  return static_cast<std::size_t>(v * kMiB);
    }
}

std::size_t MemoryBudget::available_ram()
{
    const std::size_t avail = mem_available();
    const std::size_t cg    = cgroup_limit();
    if (avail && cg) return std::min(avail, cg);
    return avail ? avail : cg;
}

MemoryBudget::MemoryBudget()
    : detected_(available_ram())
{
    if (const char* e = std::getenv("GAEL_MEM_BUDGET"))
        explicit_ = parse_size(e);
}

MemoryBudget& MemoryBudget::instance()
{
    static MemoryBudget inst;
    return inst;
}

void MemoryBudget::set_total(std::size_t bytes)    { explicit_ = bytes; }
void MemoryBudget::set_reserved(std::size_t bytes) { reserved_ = bytes; }
std::size_t MemoryBudget::reserved() const         { return reserved_; }

std::size_t MemoryBudget::auto_total_() const
{
    if (detected_ == 0) return 1024 * kMiB;      // could not tell: stay modest
    const auto share =
        static_cast<std::size_t>(static_cast<double>(detected_) *
                                 kFractionOfAvailable);
    return std::clamp(share, kFloor, kCeiling);
}

/* ---- pressure valve (see the header) -------------------------------- *
 *  Below this much free memory the machine is in trouble whatever GAEL
 *  thinks its budget is, so the shares start coming down.  Above the upper
 *  mark they are allowed back up.  The gap between the two is deliberate:
 *  a single threshold would oscillate, because shrinking the caches frees
 *  memory and immediately makes the fit look healthy again.               */
constexpr std::size_t kPressureLow  = 1536 * kMiB;   // start shrinking
constexpr std::size_t kPressureHigh = 3072 * kMiB;   // allow recovery

/*  Shrink fast, recover slowly.  Getting this wrong in the shrinking
 *  direction costs the machine; getting it wrong in the recovering
 *  direction costs a few rebuilt tables.                                  */
constexpr double kShrink  = 0.5;
constexpr double kRecover = 1.25;
constexpr double kMinScale = 0.05;

void MemoryBudget::refresh_pressure_() const
{
    using namespace std::chrono;
    const auto now_ms = static_cast<std::uint64_t>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
            .count());

    /*  At most four times a second: this reads /proc/meminfo and is called
     *  from inside the caches' critical sections.                          */
    auto last = last_sample_ms_.load(std::memory_order_relaxed);
    if (now_ms - last < 250) return;
    if (!last_sample_ms_.compare_exchange_strong(last, now_ms,
                                                 std::memory_order_relaxed))
        return;                         // another thread is doing it

    const std::size_t free_now = available_ram();
    if (free_now == 0) return;          // nothing to go on: leave it alone

    double s = scale_.load(std::memory_order_relaxed);
    if      (free_now < kPressureLow)  s = std::max(kMinScale, s * kShrink);
    else if (free_now > kPressureHigh) s = std::min(1.0,       s * kRecover);
    else                               return;      // in the dead band
    scale_.store(s, std::memory_order_relaxed);
}

double MemoryBudget::pressure_scale() const
{
    return scale_.load(std::memory_order_relaxed);
}

std::size_t MemoryBudget::total() const
{
    refresh_pressure_();

    const std::size_t gross = explicit_ ? explicit_ : auto_total_();
    /*  The solver's own arrays come off the top: they are allocated whatever
     *  the caches do, so a budget that ignored them would be describing a
     *  machine with more memory than this one has.                          */
    const std::size_t net =
        (gross > reserved_ + kFloor) ? gross - reserved_ : kFloor;

    const auto scaled = static_cast<std::size_t>(
        static_cast<double>(net) * scale_.load(std::memory_order_relaxed));
    return std::max(kFloor, scaled);
}

/*  Shares.  Model surfaces are the largest because a joint fit keeps one live
 *  per thread while it rebins onto each arm in turn; the two weight tables are
 *  one entry per (grid, spectrograph) pair and are reused across iterations,
 *  so they need enough to hold a full round-robin but no more.               */
std::size_t MemoryBudget::model_spectra() const      { return total() * 40 / 100; }
std::size_t MemoryBudget::rotational_weights() const { return total() * 30 / 100; }
std::size_t MemoryBudget::resolution_weights() const { return total() * 30 / 100; }

std::string MemoryBudget::describe() const
{
    auto mib = [](std::size_t b) { return std::to_string(b / kMiB) + " MiB"; };
    std::ostringstream os;
    os << "[memory] budget " << mib(total())
       << (explicit_ ? " (set)" : " (auto)");
    if (detected_) os << ", " << mib(detected_) << " available";
    if (reserved_) os << ", " << mib(reserved_) << " reserved by the solver";
    if (const double sc = pressure_scale(); sc < 0.999)
        os << ", scaled to " << static_cast<int>(sc * 100.0)
           << "% by memory pressure";
    os << "  --  models " << mib(model_spectra())
       << ", rotation "   << mib(rotational_weights())
       << ", resolution " << mib(resolution_weights());
    return os.str();
}

/*  Off unless asked for: it takes three cache mutexes, and the interesting
 *  moment to call it is the end of a fit, not the inside of one.           */
std::string memory_stats_line()
{
    const char* e = std::getenv("GAEL_MEM_STATS");
    if (!(e && *e && std::string(e) != "0")) return {};

    auto mib = [](std::size_t b) { return std::to_string(b / kMiB) + " MiB"; };
    const std::size_t rot = rotational_cache_bytes();
    const std::size_t res = resolution_cache_bytes();
    const std::size_t mod = model_cache_bytes();

    std::ostringstream os;
    os << "[memory] held at end of fit: rotation " << mib(rot)
       << " in " << rotational_cache_entries() << " entries"
       << ", resolution " << mib(res)
       << ", models " << mib(mod)
       << "  --  " << mib(rot + res + mod) << " of "
       << mib(MemoryBudget::instance().total()) << " budgeted";
    return os.str();
}

} // namespace specfit
