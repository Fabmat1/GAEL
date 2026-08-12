#include "specfit/FitProgress.hpp"

#include <algorithm>
#include <cmath>

namespace specfit {

namespace {
/*  Reports are throttled to this interval so that a stage whose iterations
 *  are microseconds apart (a warm-started refit that converges in two) does
 *  not flood a GUI's event queue.  Phase changes bypass it.               */
constexpr double kMinEmitInterval = 0.04;   // seconds

/*  Never report a full bar from update(): 1.0 is finish()'s to give, so a
 *  consumer can treat it as "done" without a second signal.               */
constexpr double kMaxRunningFraction = 0.995;

/*  How far into a phase the tracker starts trusting that phase's own
 *  measured rate over its prior.  Below this it has seen too little of the
 *  phase for the extrapolation t/sub to mean anything.                    */
constexpr double kExtrapolationTrustAt = 0.2;

/*  Seconds per weight unit before anything has been measured, i.e. during the
 *  first phase only.  Deliberately ~50x the rate a real fit measures (1.7e-5
 *  s/unit on this machine, single-threaded): the two ways of being wrong are
 *  not symmetric.  Over-estimating what is left holds the bar near zero until
 *  something has actually been timed, which is what an honest bar does when
 *  it knows nothing; under-estimating it sends the bar past 70 % before the
 *  fit has started, and the monotone clamp then has to hold it there.     */
constexpr double kBootstrapRate = 1e-3;
} // namespace

FitProgressTracker::FitProgressTracker(ProgressFn sink)
    : sink_(std::move(sink))
{}

int FitProgressTracker::find(int id) const
{
    for (std::size_t i = 0; i < phases_.size(); ++i)
        if (phases_[i].id == id) return static_cast<int>(i);
    return -1;
}

int FitProgressTracker::add(PhaseSpec p)
{
    Phase ph;
    ph.id     = next_id_++;
    ph.key    = std::move(p.key);
    ph.label  = std::move(p.label);
    ph.weight = std::max(0.0, p.weight);
    phases_.push_back(std::move(ph));
    return phases_.back().id;
}

std::vector<int> FitProgressTracker::expand(int id,
                                            const std::vector<PhaseSpec>& sub)
{
    std::vector<int> ids;
    const int at = find(id);
    if (at < 0) return ids;

    std::vector<Phase> made;
    made.reserve(sub.size());
    for (const auto& s : sub) {
        Phase ph;
        ph.id     = next_id_++;
        ph.key    = s.key;
        ph.label  = s.label;
        ph.weight = std::max(0.0, s.weight);
        ids.push_back(ph.id);
        made.push_back(std::move(ph));
    }

    /*  current_ is a position, so it has to move with the splice.  Expanding
     *  a phase that is already running is not a supported use, but keeping
     *  the index sane costs one comparison.                                */
    phases_.erase(phases_.begin() + at);
    phases_.insert(phases_.begin() + at, made.begin(), made.end());
    if (current_ > at)
        current_ += static_cast<int>(made.size()) - 1;

    return ids;
}

void FitProgressTracker::set_weight(int id, double w)
{
    const int i = find(id);
    if (i >= 0) phases_[static_cast<std::size_t>(i)].weight = std::max(0.0, w);
}

void FitProgressTracker::begin(int id, const std::string& detail)
{
    const int i = find(id);
    if (i < 0) return;
    current_        = i;
    sub_            = 0.0;
    t_phase_        = clock::now();
    last_detail_    = detail;
    last_iteration_ = -1;
    last_max_iter_  = -1;
    last_chi2_      = std::numeric_limits<double>::quiet_NaN();
    publish(/*force=*/true);
}

void FitProgressTracker::update(double sub, const std::string& detail,
                                int iteration, int max_iterations, double chi2)
{
    if (current_ < 0) return;
    /*  Sub-progress only ever moves forward inside a phase: a stage that
     *  restarts its solver (stage 7's boundary refit) must not rewind the
     *  bar it already advanced.                                            */
    sub_ = std::clamp(std::max(sub_, sub), 0.0, 1.0);
    if (!detail.empty())     last_detail_    = detail;
    if (iteration >= 0)      last_iteration_ = iteration;
    if (max_iterations >= 0) last_max_iter_  = max_iterations;
    if (std::isfinite(chi2)) last_chi2_      = chi2;
    publish(/*force=*/false);
}

void FitProgressTracker::end(int id)
{
    const int i = find(id);
    if (i < 0) return;
    auto& ph = phases_[static_cast<std::size_t>(i)];

    const double secs =
        std::chrono::duration<double>(clock::now() - t_phase_).count();
    ph.seconds = secs;
    ph.done    = true;

    /*  Only a phase with a positive prior teaches anything about the rate;
     *  a zero-weight phase would divide by zero and, more to the point, was
     *  never claimed to cost anything.                                     */
    if (ph.weight > 0.0) {
        auto& c = calib_[ph.key];
        c.first  += secs;
        c.second += ph.weight;
        calib_seconds_ += secs;
        calib_weight_  += ph.weight;
    }

    /*  Report the phase at 100 % of itself *before* dropping it, so the
     *  update that closes a phase still carries that phase's label rather
     *  than an empty one.  A done phase contributes nothing to `remaining`
     *  either way: sub_ == 1 zeroes its term.                             */
    if (current_ == i) {
        sub_ = 1.0;
        publish(/*force=*/true);
        current_ = -1;
        sub_     = 0.0;
    } else {
        publish(/*force=*/true);
    }
}

double FitProgressTracker::rate(const std::string& key) const
{
    const auto it = calib_.find(key);
    if (it != calib_.end() && it->second.second > 0.0)
        return it->second.first / it->second.second;
    if (calib_weight_ > 0.0) return calib_seconds_ / calib_weight_;
    return kBootstrapRate;
}

double FitProgressTracker::elapsed_seconds() const
{
    return std::chrono::duration<double>(clock::now() - t0_).count();
}

void FitProgressTracker::publish(bool force)
{
    if (!sink_ || aborted_) return;

    const clock::time_point now = clock::now();
    if (!force && last_emit_ != clock::time_point::min() &&
        std::chrono::duration<double>(now - last_emit_).count() <
            kMinEmitInterval)
        return;
    last_emit_ = now;

    const double elapsed = std::chrono::duration<double>(now - t0_).count();

    /* ---- what is left ------------------------------------------------- */
    double remaining = 0.0;

    if (current_ >= 0) {
        const auto&  ph      = phases_[static_cast<std::size_t>(current_)];
        const double t_cur   = std::chrono::duration<double>(now - t_phase_).count();
        const double prior   = rate(ph.key) * ph.weight;
        /*  This phase's own rate, once enough of it has run to extrapolate
         *  from; blended in gradually so the estimate does not lurch at the
         *  first iteration of a long stage.                                */
        const double extrap  = (sub_ > 1e-3) ? t_cur / sub_ : prior;
        const double trust   = std::min(1.0, sub_ / kExtrapolationTrustAt);
        const double total   = (1.0 - trust) * prior + trust * extrap;
        remaining += std::max(0.0, total * (1.0 - sub_));
    }

    bool seen_current = current_ < 0;
    for (std::size_t i = 0; i < phases_.size(); ++i) {
        if (static_cast<int>(i) == current_) { seen_current = true; continue; }
        const auto& ph = phases_[i];
        if (ph.done || !seen_current) continue;     // already paid for
        remaining += rate(ph.key) * ph.weight;
    }

    /* ---- fraction ----------------------------------------------------- */
    double frac = 0.0;
    if (elapsed + remaining > 0.0) frac = elapsed / (elapsed + remaining);
    frac = std::min(frac, kMaxRunningFraction);
    high_water_ = std::max(high_water_, frac);

    ProgressReport r;
    r.fraction        = high_water_;
    r.phase_count     = static_cast<int>(phases_.size());
    r.phase_index     = (current_ >= 0) ? current_ + 1 : r.phase_count;
    r.phase           = (current_ >= 0)
                      ? phases_[static_cast<std::size_t>(current_)].label
                      : std::string();
    r.detail          = last_detail_;
    r.iteration       = last_iteration_;
    r.max_iterations  = last_max_iter_;
    r.chi2            = last_chi2_;
    r.elapsed_seconds = elapsed;
    r.eta_seconds     = (high_water_ > 1e-3) ? remaining : -1.0;

    if (!sink_(r)) aborted_ = true;
}

void FitProgressTracker::finish(const std::string& label)
{
    if (!sink_ || aborted_) return;
    high_water_ = 1.0;

    ProgressReport r;
    r.fraction        = 1.0;
    r.phase           = label;
    r.phase_count     = static_cast<int>(phases_.size());
    r.phase_index     = r.phase_count;
    r.elapsed_seconds = elapsed_seconds();
    r.eta_seconds     = 0.0;
    sink_(r);
}

} // namespace specfit
