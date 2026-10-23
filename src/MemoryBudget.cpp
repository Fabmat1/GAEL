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

/*  An absolute ceiling, per fit in flight.  Past a few GB these tables stop
 *  earning their keep for one fit:
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

/*  Value of `key` in a memory.stat file (one "key value" pair per line). */
std::size_t read_stat(const std::string& path, const std::string& key)
{
    std::ifstream f(path);
    std::string k;
    unsigned long long v = 0;
    while (f >> k >> v)
        if (k == key) return static_cast<std::size_t>(v);
    return 0;
}

/*  memory.max on a machine with no limit reads as a number close to SIZE_MAX
 *  rather than "max" on some kernels, and cgroup v1 always does; treat
 *  anything past a petabyte as "no limit".                                   */
bool is_limit(std::size_t v) { return v != 0 && v <= (1ull << 50); }

} // namespace

/*  The memory cgroups that bound this process: its own and every ancestor,
 *  since the tightest of them is the one that applies, for whichever of the
 *  two hierarchies carries the memory controller.
 *
 *  v2 is the "0::<path>" line.  v1 has one line per hierarchy, and the
 *  memory controller's is "<n>:memory:<path>" (possibly co-mounted, as in
 *  "memory,cpu").  Reading only the v1 root, as this once did, sees the
 *  host's unlimited root group: under Slurm on a v1 host the job's limit
 *  sits five levels down (slurm/uid_<u>/job_<j>/step_<s>/task_<t>), and the
 *  budget was sized as if the job had the whole node.                       */
MemoryBudget::CgroupMemory
MemoryBudget::cgroup_memory(const std::string& proc_cgroup,
                            const std::string& sysfs)
{
    CgroupMemory best;

    auto consider = [&best](const std::string& dir, bool v2) {
        std::size_t limit = 0;
        auto tighter = [&limit](std::size_t v) {
            if (is_limit(v) && (limit == 0 || v < limit)) limit = v;
        };
        if (v2) {
            tighter(read_first_number(dir + "/memory.max"));
            tighter(read_first_number(dir + "/memory.high"));
        } else {
            tighter(read_first_number(dir + "/memory.limit_in_bytes"));
        }
        if (limit == 0) return;

        /*  Usage counts the page cache, and a fit that streams its grid
         *  through the page cache would look full long before it is.  The
         *  inactive file pages are what the kernel reclaims first, so they
         *  come off, the way `docker stats` reports a container's memory.    */
        const std::size_t usage = read_first_number(
            dir + (v2 ? "/memory.current" : "/memory.usage_in_bytes"));
        const std::size_t inactive =
            v2 ? read_stat(dir + "/memory.stat", "inactive_file")
               : read_stat(dir + "/memory.stat", "total_inactive_file");
        const std::size_t used     = usage > inactive ? usage - inactive : 0;
        const std::size_t headroom = limit > used ? limit - used : 0;

        if (best.limit == 0 || headroom < best.headroom) {
            best.limit    = limit;
            best.headroom = headroom;
        }
    };

    std::ifstream self(proc_cgroup);
    std::string line;
    while (std::getline(self, line)) {
        const auto c1 = line.find(':');
        const auto c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
        if (c2 == std::string::npos) continue;
        const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1);
        std::string rel = line.substr(c2 + 1);

        bool v2 = controllers.empty();
        if (!v2) {
            bool memory = false;
            std::istringstream is(controllers);
            for (std::string c; std::getline(is, c, ',');)
                memory = memory || c == "memory";
            if (!memory) continue;
        }
        const std::string base = sysfs + (v2 ? "" : "/memory");

        while (!rel.empty() && rel != "/") {
            consider(base + rel, v2);
            const auto slash = rel.rfind('/');
            if (slash == std::string::npos || slash == 0) break;
            rel = rel.substr(0, slash);
        }
        consider(base, v2);
    }
    return best;
}

namespace {

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
    const CgroupMemory cg   = cgroup_memory();
    if (avail && cg.limit) return std::min(avail, cg.headroom);
    return cg.limit ? cg.headroom : avail;
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

void MemoryBudget::set_total(std::size_t bytes) { explicit_.store(bytes); }
std::size_t MemoryBudget::reserved() const      { return reserved_.load(); }

/* ---- fits in flight (see the header) -------------------------------- */
MemoryBudget::Fit::Fit()  { instance().fits_.fetch_add(1); }
MemoryBudget::Fit::~Fit()
{
    MemoryBudget& b = instance();
    b.reserved_.fetch_sub(reserved_);
    b.fits_.fetch_sub(1);
}
void MemoryBudget::Fit::reserve(std::size_t bytes)
{
    MemoryBudget& b = instance();
    b.reserved_.fetch_add(bytes);
    b.reserved_.fetch_sub(reserved_);
    reserved_ = bytes;
}

int MemoryBudget::fits() const { return std::max(1, fits_.load()); }

std::size_t MemoryBudget::entry_cap(std::size_t per_fit) const
{
    return per_fit * static_cast<std::size_t>(fits());
}

/*  The share of what is free, between one fit's floor and one ceiling per
 *  fit.  For one fit that is the old clamp(share, floor, ceiling) exactly.
 *  For N it is what N gael-worker processes would each have taken, except
 *  that the share of free memory is the machine's and not one per fit:
 *  separate processes would each have read the same MemAvailable and
 *  promised themselves 55 % of it, which only the pressure valve kept off
 *  the swap.                                                               */
std::size_t MemoryBudget::auto_total(std::size_t available, int fits)
{
    if (available == 0) return 1024 * kMiB;      // could not tell: stay modest
    const auto share =
        static_cast<std::size_t>(static_cast<double>(available) *
                                 kFractionOfAvailable);
    const auto n = static_cast<std::size_t>(std::max(1, fits));
    return std::clamp(share, kFloor, n * kCeiling);
}

/*  kFloor is one fit's surface plus the weights that made it, so N fits want
 *  N of them -- but never more than the gross budget, which is where the
 *  machine (or whoever set it explicitly) draws the line.  For one fit this
 *  is kFloor.                                                              */
std::size_t MemoryBudget::floor_(std::size_t gross) const
{
    const auto n = static_cast<std::size_t>(fits());
    return std::max(kFloor, std::min(n * kFloor, gross));
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

    const std::size_t avail = mem_available();
    const CgroupMemory cg   = cgroup_memory();
    const std::size_t free_now =
        cg.limit ? (avail ? std::min(avail, cg.headroom) : cg.headroom) : avail;
    if (free_now == 0 && cg.limit == 0) return;   // nothing to go on

    /*  The marks are for a desktop.  In a 4 GB batch slot "1.5 GB free" is
     *  over a third of everything the job may hold, and a valve that opened
     *  there would never close; so under a cgroup limit they shrink with it,
     *  to the same fraction of the limit they are of a typical 10 GB box.    */
    std::size_t low = kPressureLow, high = kPressureHigh;
    if (cg.limit) {
        low  = std::min(low,  cg.limit * 15 / 100);
        high = std::min(high, cg.limit * 30 / 100);
    }

    double s = scale_.load(std::memory_order_relaxed);
    if      (free_now < low)  s = std::max(kMinScale, s * kShrink);
    else if (free_now > high) s = std::min(1.0,       s * kRecover);
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

    const std::size_t set   = explicit_.load();
    const std::size_t gross = set ? set : auto_total(detected_, fits());
    const std::size_t floor = floor_(gross);
    /*  The solver's own arrays come off the top: they are allocated whatever
     *  the caches do, so a budget that ignored them would be describing a
     *  machine with more memory than this one has.                          */
    const std::size_t res = reserved_.load();
    const std::size_t net = (gross > res + floor) ? gross - res : floor;

    const auto scaled = static_cast<std::size_t>(
        static_cast<double>(net) * scale_.load(std::memory_order_relaxed));
    return std::max(floor, scaled);
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
       << (explicit_.load() ? " (set)" : " (auto)");
    if (const int n = fits(); n > 1) os << " for " << n << " fits in flight";
    if (detected_) os << ", " << mib(detected_) << " available";
    if (const std::size_t r = reserved())
        os << ", " << mib(r) << " reserved by the solver";
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
