/*  Hermetic unit test for the continuum-jitter anchor intervals.
 *
 *  The jitter refits scale every anchor step by up to 1.4 and move each
 *  range's start by up to one scaled step.  A range trimmed to the edge of the
 *  spectrum can be narrower than that shift (ASTRA left [3850, 4050] at 95 A on
 *  one spectrum, against a shift of up to 140 A), and the start used to land
 *  past the end, which anchors_from_intervals rejects and the whole fit failed.
 */
#include "specfit/ContinuumUtils.hpp"

#include <array>
#include <iostream>
#include <vector>

using specfit::anchors_from_intervals;
using specfit::jitter_anchor_intervals;

static int g_failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " << msg << " (" << #cond << ")\n";          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

int main()
{
    /* the spectrum starts at 3955 A, so [3850, 4050] has been trimmed to
     * [3955, 4050]: 95 A wide, with a 100 A step                           */
    const std::vector<std::array<double, 3>> iv = {
        {3955.0, 4050.0, 100.0},
        {4050.0, 5000.0, 100.0},
    };

    /* worst case: largest step scale and the largest phase */
    const auto j = jitter_anchor_intervals(iv, 1.4, 0.9999);
    CHECK(j.size() == iv.size(), "interval count kept");
    for (std::size_t i = 0; i < j.size(); ++i) {
        CHECK(j[i][0] <= j[i][1], "start <= end for interval " << i);
        CHECK(j[i][1] == iv[i][1], "end unchanged for interval " << i);
        CHECK(j[i][2] == 140.0, "step scaled for interval " << i);
    }
    CHECK(j[0][0] == 4050.0, "narrow range collapses onto its end");
    CHECK(j[1][0] > 4050.0 + 139.0, "wide range is shifted as before");

    /* the shifted ranges must build anchors on a spectrum covering them */
    specfit::Spectrum sp;
    const int n = 1046;  // 3955 .. 5000 A at 1 A
    sp.lambda.resize(n);
    sp.flux  .resize(n);
    sp.sigma .resize(n);
    for (int i = 0; i < n; ++i) {
        sp.lambda[i] = 3955.0 + i;
        sp.flux  [i] = 1.0;
        sp.sigma [i] = 0.01;
    }
    sp.ignoreflag.assign(n, 1);

    std::vector<std::tuple<double, double, double>> tpl;
    for (const auto& a : j) tpl.emplace_back(a[0], a[1], a[2]);
    try {
        const auto x = anchors_from_intervals(tpl, sp);
        CHECK(x.size() >= 3, "anchors built");
        for (Eigen::Index i = 1; i < x.size(); ++i)
            CHECK(x[i] > x[i - 1], "anchors strictly increasing");
    } catch (const std::exception& e) {
        CHECK(false, "anchors_from_intervals threw: " << e.what());
    }

    /* every phase/scale the refits can draw keeps start <= end */
    for (double sc = 0.8; sc <= 1.4 + 1e-12; sc += 0.05)
        for (double ph = 0.0; ph < 1.0; ph += 0.05)
            for (const auto& a : jitter_anchor_intervals(iv, sc, ph))
                CHECK(a[0] <= a[1], "start <= end at sc=" << sc << " ph=" << ph);

    if (g_failures) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "continuum_jitter: all checks passed\n";
    return 0;
}
