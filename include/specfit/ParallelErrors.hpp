#pragma once
#include <atomic>
#include <exception>
#include <mutex>

namespace specfit {

/*  An exception must not leave an OpenMP parallel region: the runtime calls
 *  std::terminate, which is how one unreadable model file used to take down
 *  a whole worker process (exit 134) instead of failing its fit.
 *
 *  Run each iteration's body through run(), which keeps the first exception
 *  and skips the bodies still to come, then call rethrow() once the region
 *  has ended.                                                               */
class ParallelErrors {
public:
    template <typename Body>
    void run(Body&& body) noexcept
    {
        if (failed_.load(std::memory_order_relaxed)) return;
        try {
            body();
        } catch (...) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!first_) first_ = std::current_exception();
            failed_.store(true, std::memory_order_relaxed);
        }
    }

    bool failed() const noexcept
    {
        return failed_.load(std::memory_order_relaxed);
    }

    void rethrow()
    {
        if (first_) std::rethrow_exception(first_);
    }

private:
    std::mutex         mtx_;
    std::exception_ptr first_;
    std::atomic<bool>  failed_{false};
};

} // namespace specfit
