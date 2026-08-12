#pragma once
/* -------------------------------------------------------------------------- *
 *  Progress reporting for a whole fit.
 *
 *  A fit is a sequence of *phases* -- load a spectrum, solve a stage, run one
 *  continuum-jitter refit -- and the tracker below turns that sequence into a
 *  single monotone fraction in [0,1] plus a human-readable label.
 *
 *  Why the fraction is time-calibrated rather than structural
 *  ---------------------------------------------------------
 *  What a stage costs is dominated by how many LM iterations it happens to
 *  need, and that is not predictable from the configuration.  Measured on two
 *  of the repository's own inputs (one thread, warm caches, K = 6):
 *
 *                        preprocess  stage1  stage4  st.5/6/7  jitter x6
 *      test_star             0.02 s  0.06 s  2.04 s    0.02 s     0.07 s
 *      test_star_multifile   0.06 s  0.15 s  0.19 s    0.15 s     1.45 s
 *
 *  Stage 4 is 96 % of one fit and 9 % of the other; the jitter ensemble is 3 %
 *  of one and 72 % of the other.  No fixed weighting can describe both, so
 *  each phase carries only a *prior* weight (see UnifiedFitWorkflow's cost
 *  model) and the tracker rescales the priors of everything still to come by
 *  the seconds-per-weight-unit it has actually measured so far -- per phase
 *  `key`, so that jitter refit 1 predicts refits 2..K, falling back to the
 *  global rate for a key that has not run yet.  The reported fraction is then
 *
 *      elapsed / (elapsed + estimated_remaining)
 *
 *  which is an estimate of the share of the *wall clock* that is done.  It is
 *  clamped monotone: a worsening estimate stalls the bar, it never rewinds.
 *
 *  Not thread-safe: every call site is on the fit's own thread (the LM loop,
 *  the stage driver, the preprocessing loop), never inside an OpenMP region.
 * -------------------------------------------------------------------------- */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specfit {

/*  One progress update.  Everything except `fraction` is descriptive; a
 *  consumer that only drives a bar needs `fraction`, `phase` and `detail`.  */
struct ProgressReport {
    std::string phase;            // "Stage 4/7 - first full fit"
    std::string detail;           // "LM iteration 37/200 - chi2 = 2413.7"

    double fraction        = 0.0; // [0,1] of the whole fit, monotone

    int    phase_index     = 0;   // 1-based position in the plan
    int    phase_count     = 0;   // phases in the plan (conditionals included)

    int    iteration       = -1;  // LM iteration inside the current solve, or -1
    int    max_iterations  = -1;
    double chi2            = std::numeric_limits<double>::quiet_NaN();

    double elapsed_seconds = 0.0;
    double eta_seconds     = -1.0;  // < 0 when there is nothing to base it on
};

/*  Return false to ask the fit to stop.  The request is honoured at the next
 *  LM iteration boundary; the fit then unwinds by throwing FitAborted.      */
using ProgressFn = std::function<bool(const ProgressReport&)>;

struct FitAborted : std::runtime_error {
    FitAborted() : std::runtime_error("fit aborted on request") {}
};

/* -------------------------------------------------------------------------- *
 *  How far into a solve iteration `it` of a budget of `max_it` is.
 *
 *  Not it/max_it: LM stops on its convergence tolerances, essentially never on
 *  the budget, so the budget is an upper bound that a typical stage uses a
 *  fifth of.  Reporting the linear fraction leaves the dominant stage of a
 *  single-spectrum fit -- 96 % of its wall clock, converging around iteration
 *  40 of 200 -- crawling to 20 % and then jumping, which is the failure this
 *  whole file exists to avoid.
 *
 *  What is actually known about the stopping iteration is that each iteration
 *  has a roughly constant chance of being the last, i.e. it is geometrically
 *  distributed, whose progress curve is 1 - exp(-it/tau).  tau is a fifth of
 *  the budget, floored so that the three-iteration solves of stage 6 -- which
 *  really do run to their budget -- still report sensibly (39/63/78 %) instead
 *  of saturating at the first step.
 *
 *  The curve never reaches 1; the phase ending is what closes it out.
 * -------------------------------------------------------------------------- */
inline double lm_iteration_progress(int it, int max_it)
{
    const double tau = std::max(2.0, 0.2 * std::max(1, max_it));
    return 1.0 - std::exp(-static_cast<double>(it) / tau);
}

class FitProgressTracker {
public:
    struct PhaseSpec {
        /*  Calibration bucket.  Phases sharing a key share their measured
         *  seconds-per-weight, which is what lets one continuum-jitter refit
         *  predict the remaining K-1.                                       */
        std::string key;
        std::string label;
        double      weight = 1.0;   // prior cost, arbitrary units
    };

    /*  Default-constructed trackers are inert: every call is a cheap no-op,
     *  so the fit code does not need to test for a null sink.               */
    FitProgressTracker() = default;
    explicit FitProgressTracker(ProgressFn sink);

    bool active()  const { return static_cast<bool>(sink_); }
    bool aborted() const { return aborted_; }
    void throw_if_aborted() const { if (aborted_) throw FitAborted(); }

    /* ---- planning ---------------------------------------------------- *
     *  Phases are addressed by a stable id, not by position, because
     *  expand() renumbers positions but leaves ids alone.                 */
    int  add(PhaseSpec p);

    /*  Replace phase `id` (not yet begun) with the given sub-phases, in
     *  place, and return their ids.  This is how UnifiedFitWorkflow turns
     *  the session's single "fit" placeholder into its own stage ladder
     *  without the session having to know what the ladder looks like.     */
    std::vector<int> expand(int id, const std::vector<PhaseSpec>& sub);

    /*  Revise a prior once a conditional resolves; 0 means "will not run".  */
    void set_weight(int id, double w);
    void drop(int id) { set_weight(id, 0.0); }

    /* ---- running ----------------------------------------------------- */
    void begin(int id, const std::string& detail = {});
    /*  Sub-progress `sub` in [0,1] of the phase that is running.  */
    void update(double sub,
                const std::string& detail = {},
                int iteration = -1, int max_iterations = -1,
                double chi2 = std::numeric_limits<double>::quiet_NaN());
    void end(int id);

    /*  Emit a final 100 % report.  */
    void finish(const std::string& label = {});

    double elapsed_seconds() const;

private:
    using clock = std::chrono::steady_clock;

    struct Phase {
        int         id;
        std::string key;
        std::string label;
        double      weight;
        bool        done    = false;
        double      seconds = 0.0;
    };

    int  find(int id) const;
    /*  Measured seconds per weight unit for `key`: that key's own history if
     *  it has any, else every completed phase's, else a neutral guess (which
     *  only matters before the first phase finishes, when `elapsed` is still
     *  ~0 and the fraction is ~0 whatever the rate is).                    */
    double rate(const std::string& key) const;
    void   publish(bool force);

    ProgressFn         sink_;
    std::vector<Phase> phases_;
    int                next_id_ = 1;
    int                current_ = -1;      // index into phases_, -1 = none
    double             sub_     = 0.0;

    clock::time_point  t0_        = clock::now();
    clock::time_point  t_phase_   = clock::now();
    clock::time_point  last_emit_ = clock::time_point::min();

    double high_water_ = 0.0;
    bool   aborted_    = false;

    // key -> (measured seconds, weight those seconds covered)
    std::map<std::string, std::pair<double,double>> calib_;
    double calib_seconds_ = 0.0;
    double calib_weight_  = 0.0;

    // carried into every report so the consumer can render a stable line
    std::string last_detail_;
    int    last_iteration_ = -1;
    int    last_max_iter_  = -1;
    double last_chi2_      = std::numeric_limits<double>::quiet_NaN();
};

} // namespace specfit
