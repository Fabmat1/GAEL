/*  Hermetic unit test for MemoryBudget's cgroup detection.
 *
 *  Builds fake /proc/self/cgroup files and /sys/fs/cgroup trees on disk: a
 *  cgroup v1 host with a Slurm job's limit five levels below the root (what
 *  the hpc cluster looks like, where the root-only read found no limit), a
 *  cgroup v2 systemd scope, and a host with no limit at all.  Headroom must
 *  be the tightest group's limit minus its usage, with inactive page cache
 *  counted as free.
 */
#include "specfit/MemoryBudget.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;
using specfit::MemoryBudget;

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

int main()
{
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
