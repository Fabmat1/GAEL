/*  Hermetic unit test for MemoryBudget's cgroup detection.
 *
 *  Builds fake /proc/self/cgroup files and /sys/fs/cgroup trees on disk: a
 *  cgroup v1 host with a Slurm job's limit five levels below the root (what
 *  the hpc cluster looks like, where the root-only read found no limit), a
 *  cgroup v2 systemd scope, and a host with no limit at all.  Headroom must
 *  be the tightest group's limit minus its usage, with inactive page cache
 *  counted as free.
 *
 *  And the fits-in-flight scaling: N fits in one process (ASTRA's in-process
 *  mass fits) must raise the automatic total, the weight caches' entry caps
 *  and the model cache's per-fit budget N-fold, within what the machine has
 *  free, while one fit must leave every figure exactly where it was.
 */
#include "specfit/MemoryBudget.hpp"
#include "specfit/SpectrumCache.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;
using specfit::MemoryBudget;
using specfit::SpectrumCache;

static int g_failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " << msg << " (" << #cond << ")\n";          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

constexpr std::size_t kMiB = 1ull << 20;

static void put(const fs::path& p, const std::string& text)
{
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

static std::string num(std::size_t v) { return std::to_string(v) + "\n"; }

/* ---- fits in flight ------------------------------------------------ */
static void check_fit_scaling()
{
    constexpr std::size_t kGiB = 1024 * kMiB;

    /* -- the automatic total, on machines this one is not -------------- */
    //  A laptop: one fit is held to the 8 GiB ceiling, eight are held to
    //  55 % of what is free -- more than one fit, far less than eight
    //  processes' worth.
    CHECK(MemoryBudget::auto_total(16 * kGiB, 1) == 8 * kGiB,
          "laptop, one fit: the per-fit ceiling");
    CHECK(MemoryBudget::auto_total(16 * kGiB, 8) ==
              static_cast<std::size_t>(16.0 * kGiB * 0.55),
          "laptop, eight fits: the machine's share, not 8 ceilings");
    //  A batch node: room for every fit's ceiling, until the share runs out.
    CHECK(MemoryBudget::auto_total(400 * kGiB, 1) == 8 * kGiB,
          "node, one fit: unchanged");
    CHECK(MemoryBudget::auto_total(400 * kGiB, 8) == 64 * kGiB,
          "node, eight fits: one ceiling each");
    CHECK(MemoryBudget::auto_total(300 * kGiB, 24) ==
              static_cast<std::size_t>(300.0 * kGiB * 0.55),
          "node, 24 fits: capped by the share of free memory");
    //  Zero or one fit in flight is the same machine.
    CHECK(MemoryBudget::auto_total(400 * kGiB, 0) ==
              MemoryBudget::auto_total(400 * kGiB, 1),
          "no fit counts as one");
    //  Nothing known: stay modest, however many fits.
    CHECK(MemoryBudget::auto_total(0, 8) == MemoryBudget::auto_total(0, 1),
          "unknown RAM does not scale");
    CHECK(MemoryBudget::auto_total(100 * kMiB, 8) == 192 * kMiB,
          "starved machine: the floor");

    /* -- the live budget ------------------------------------------------ */
    MemoryBudget& mb = MemoryBudget::instance();
    auto calm = [&] { return mb.pressure_scale() > 0.999; };

    const std::size_t t0  = mb.total();
    const std::size_t m0  = mb.model_spectra();
    const std::size_t rw0 = mb.rotational_weights();
    const std::size_t sw0 = mb.resolution_weights();
    CHECK(mb.fits() == 1, "outside any fit, one fit");
    CHECK(mb.entry_cap(25) == 25, "outside any fit, the per-fit cap");

    {
        MemoryBudget::Fit one;
        CHECK(mb.fits() == 1, "one fit in flight");
        CHECK(mb.entry_cap(25) == 25, "one fit: entry cap unchanged");
        if (calm()) {
            CHECK(mb.total() == t0, "one fit: total unchanged");
            CHECK(mb.model_spectra() == m0 && mb.rotational_weights() == rw0 &&
                      mb.resolution_weights() == sw0,
                  "one fit: every share unchanged");
        }
    }

    {
        MemoryBudget::Fit eight[8];
        CHECK(mb.fits() == 8, "eight fits in flight");
        CHECK(mb.entry_cap(25) == 200, "eight fits: entry caps eightfold");
        if (calm()) {
            const std::size_t avail = MemoryBudget::available_ram();
            CHECK(mb.total() >= t0, "eight fits never get less");
            CHECK(mb.model_spectra() >= m0 && mb.rotational_weights() >= rw0 &&
                      mb.resolution_weights() >= sw0,
                  "eight fits: no share shrinks");
            //  Whenever the machine has the room, the fits get it.
            if (avail &&
                MemoryBudget::auto_total(avail, 8) >
                    MemoryBudget::auto_total(avail, 1) + 256 * kMiB) {
                CHECK(mb.total() > t0, "eight fits: a larger total");
                CHECK(mb.rotational_weights() > rw0 &&
                          mb.resolution_weights() > sw0,
                      "eight fits: larger weight caches");
            }
            //  ...and never past what is free.
            if (avail)
                CHECK(mb.total() <= std::max<std::size_t>(
                                        192 * kMiB,
                                        MemoryBudget::auto_total(avail, 8) +
                                            256 * kMiB),
                      "eight fits: inside the machine's share");
        }
    }
    CHECK(mb.fits() == 1, "finished fits are uncounted");
    if (calm()) CHECK(mb.total() == t0, "the total comes back down");

    /* -- reservations add up across fits and go with them -------------- */
    {
        MemoryBudget::Fit a;
        a.reserve(100 * kMiB);
        CHECK(mb.reserved() == 100 * kMiB, "one reservation");
        {
            MemoryBudget::Fit b;
            b.reserve(50 * kMiB);
            CHECK(mb.reserved() == 150 * kMiB, "two fits' reservations add");
            a.reserve(10 * kMiB);
            CHECK(mb.reserved() == 60 * kMiB, "re-reserving replaces, not adds");
        }
        CHECK(mb.reserved() == 10 * kMiB, "a finished fit's reservation goes");
    }
    CHECK(mb.reserved() == 0, "all reservations released");

    /* -- the model cache holds its per-fit budget once per fit ---------- */
    {
        SpectrumCache& sc = SpectrumCache::instance();
        sc.clear();
        sc.set_memory_budget(1 * kMiB);      // per fit; 64 KiB entries below

        std::size_t key = 1;
        auto fill = [&](int n) {
            for (int i = 0; i < n; ++i) {
                specfit::Spectrum s;
                s.flux = specfit::Vector::Zero(8192);
                sc.insert(key++, std::move(s));
            }
        };

        fill(100);
        const std::size_t one = sc.size();
        CHECK(one >= 10 && one <= 16, "one fit: ~1 MiB of 64 KiB spectra");
        CHECK(sc.bytes_used() <= 1 * kMiB, "one fit: inside its budget");
        {
            MemoryBudget::Fit four[4];
            fill(100);
            CHECK(sc.size() >= 4 * one - 4 && sc.size() <= 4 * one + 4,
                  "four fits: four fits' worth held");
            CHECK(sc.bytes_used() <= 4 * kMiB, "four fits: inside 4x");
        }
        fill(1);
        CHECK(sc.size() <= one, "the next insert after they finish trims back");
        sc.clear();
        sc.set_memory_budget(128 * kMiB);
    }
}

int main()
{
    //  The fits-in-flight checks need the automatic budget, which an
    //  override in the environment would replace.
#ifdef _WIN32
    ::_putenv_s("GAEL_MEM_BUDGET", "");
#else
    ::unsetenv("GAEL_MEM_BUDGET");
#endif
    check_fit_scaling();

    const fs::path root =
        fs::temp_directory_path() /
        ("gael_memory_budget_test_" + std::to_string(std::random_device{}()));
    const std::string unlimited = "9223372036854771712\n";

    /* ---- cgroup v1, Slurm job: the limit is on the task's group ------- */
    {
        const fs::path sys  = root / "v1" / "sys";
        const fs::path proc = root / "v1" / "cgroup";
        const std::string job = "/slurm/uid_131038/job_1926534/step_0/task_0";
        put(proc,
            "13:pids:/system.slice/slurmd.service\n"
            "6:memory:" + job + "\n"
            "4:cpu,cpuacct:" + job + "\n"
            "1:name=systemd:/system.slice/slurmd.service\n"
            "0::/system.slice/slurmd.service\n");
        put(sys / "memory" / "memory.limit_in_bytes", unlimited);
        put(sys / "memory/slurm/memory.limit_in_bytes", unlimited);
        put(sys / ("memory" + job) / "memory.limit_in_bytes", num(4096 * kMiB));
        put(sys / ("memory" + job) / "memory.usage_in_bytes", num(3000 * kMiB));
        put(sys / ("memory" + job) / "memory.stat",
            "cache 900\ninactive_file 1\ntotal_inactive_file " +
            std::to_string(500 * kMiB) + "\n");

        const auto cg = MemoryBudget::cgroup_memory(proc.string(), sys.string());
        CHECK(cg.limit == 4096 * kMiB, "v1: the job's limit, not the root's");
        CHECK(cg.headroom == (4096 - 2500) * kMiB,
              "v1: limit minus usage, inactive page cache counted as free");
    }

    /* ---- cgroup v1, memory co-mounted with cpu ------------------------ */
    {
        const fs::path sys  = root / "v1co" / "sys";
        const fs::path proc = root / "v1co" / "cgroup";
        put(proc, "3:cpu,memory:/batch/job7\n");
        put(sys / "memory/batch/job7/memory.limit_in_bytes", num(2048 * kMiB));
        put(sys / "memory/batch/job7/memory.usage_in_bytes", num(2048 * kMiB));

        const auto cg = MemoryBudget::cgroup_memory(proc.string(), sys.string());
        CHECK(cg.limit == 2048 * kMiB, "v1: memory in a co-mounted controller list");
        CHECK(cg.headroom == 0, "v1: a full group has no headroom");
    }

    /* ---- cgroup v2: the tightest ancestor applies --------------------- */
    {
        const fs::path sys  = root / "v2" / "sys";
        const fs::path proc = root / "v2" / "cgroup";
        put(proc, "0::/user.slice/app.slice/fit.scope\n");
        put(sys / "memory.max", "max\n");
        put(sys / "user.slice/memory.max", num(16384 * kMiB));
        put(sys / "user.slice/memory.current", num(15000 * kMiB));
        put(sys / "user.slice/memory.stat", "anon 1\ninactive_file 0\n");
        put(sys / "user.slice/app.slice/memory.max", "max\n");
        put(sys / "user.slice/app.slice/fit.scope/memory.max", num(8192 * kMiB));
        put(sys / "user.slice/app.slice/fit.scope/memory.high", "max\n");
        put(sys / "user.slice/app.slice/fit.scope/memory.current", num(1024 * kMiB));
        put(sys / "user.slice/app.slice/fit.scope/memory.stat",
            "inactive_file " + std::to_string(24 * kMiB) + "\n");

        const auto cg = MemoryBudget::cgroup_memory(proc.string(), sys.string());
        CHECK(cg.limit == 16384 * kMiB,
              "v2: the slice with less left wins over the scope's lower limit");
        CHECK(cg.headroom == (16384 - 15000) * kMiB, "v2: headroom of the slice");
    }

    /* ---- no limit anywhere -------------------------------------------- */
    {
        const fs::path sys  = root / "none" / "sys";
        const fs::path proc = root / "none" / "cgroup";
        put(proc, "0::/\n");
        put(sys / "memory.max", "max\n");

        const auto cg = MemoryBudget::cgroup_memory(proc.string(), sys.string());
        CHECK(cg.limit == 0, "no limit is reported as none");
        CHECK(MemoryBudget::cgroup_memory((root / "missing").string(),
                                          sys.string()).limit == 0,
              "an unreadable /proc/self/cgroup is no limit, not an error");
    }

    std::error_code ec;
    fs::remove_all(root, ec);

    if (g_failures) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "memory_budget: all checks passed\n";
    return 0;
}
