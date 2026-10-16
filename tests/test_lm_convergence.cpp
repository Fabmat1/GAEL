/*  Hermetic unit test for the Levenberg-Marquardt stopping rules.
 *
 *  1. A clean least-squares problem converges, at the truth.
 *  2. A parameter held at its bound by its own gradient (vsini at 0 in a real
 *     fit) must not keep the fit from converging, and must stay on the bound
 *     while the others reach their constrained optimum.
 *  3. The absolute chi2 floor ends a solve the relative tests cannot end
 *     (here they are switched off), without stopping early.
 */
#include "specfit/SimpleLM.hpp"

#include <cmath>
#include <iostream>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " << msg << " (" << #cond << ")\n";          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

namespace {

/*  y = a * exp(-x / b) + c on m points, sigma = 1, so chi2 counts in units of
 *  sigma like a real fit.  `jitter` adds a non-smooth term the analytic
 *  Jacobian does not know about.                                             */
struct DecayProblem {
    int    m;
    double jitter;
    double a0 = 300.0, b0 = 40.0, c0 = 20.0;
    std::vector<double> xs, ys;

    DecayProblem(int m_, double jitter_) : m(m_), jitter(jitter_)
    {
        for (int i = 0; i < m; ++i) {
            const double x = 200.0 * i / (m - 1);
            xs.push_back(x);
            // Fixed pseudo-noise of unit sigma, so chi2/m ~ 1 at the truth.
            ys.push_back(a0 * std::exp(-x / b0) + c0 + std::sin(12.9898 * i) * 1.2);
        }
    }

    void operator()(const Eigen::VectorXd& p, Eigen::VectorXd* r, Eigen::MatrixXd* J) const
    {
        const double a = p[0], b = p[1], c = p[2];
        // Changes with the parameters at a scale far below any step's size.
        const double wobble = jitter * std::sin(1e7 * (a + b + c));
        if (r) {
            r->resize(m);
            for (int i = 0; i < m; ++i)
                (*r)[i] = a * std::exp(-xs[i] / b) + c - ys[i] +
                          wobble * std::cos(3.1 * i);
        }
        if (J) {
            J->resize(m, 3);
            for (int i = 0; i < m; ++i) {
                const double e = std::exp(-xs[i] / b);
                (*J)(i, 0) = e;
                (*J)(i, 1) = a * e * xs[i] / (b * b);
                (*J)(i, 2) = 1.0;
            }
        }
    }
};

specfit::LMSolverSummary solve(const DecayProblem& pb, Eigen::VectorXd& x,
                               specfit::LMSolverOptions opt, double cMax = 1e3)
{
    return specfit::levenberg_marquardt(
        [&pb](const Eigen::VectorXd& p, Eigen::VectorXd* r, Eigen::MatrixXd* J) { pb(p, r, J); },
        x, { true, true, true }, { 0.0, 1.0, -1e3 }, { 1e4, 1e3, cMax }, opt);
}

Eigen::VectorXd start() { Eigen::VectorXd x(3); x << 100.0, 10.0, 0.0; return x; }

bool atTruth(const DecayProblem& pb, const Eigen::VectorXd& x)
{
    return std::abs(x[0] - pb.a0) < 3.0 && std::abs(x[1] - pb.b0) < 1.0 &&
           std::abs(x[2] - pb.c0) < 1.0;
}

} // namespace

int main()
{
    specfit::LMSolverOptions opt;
    opt.max_iterations = 600;

    // 1. Clean problem: converges, at the truth.
    {
        DecayProblem pb(400, 0.0);
        Eigen::VectorXd x = start();
        const auto s = solve(pb, x, opt);
        CHECK(s.converged, "clean problem converges");
        CHECK(atTruth(pb, x), "clean problem reaches the truth");
    }

    // 2. The offset is capped below its true value (20): it ends on the cap,
    //    pushed there by its gradient, and the fit still converges.
    {
        DecayProblem pb(400, 0.0);
        Eigen::VectorXd x = start();
        const auto s = solve(pb, x, opt, 15.0);
        std::cout << "bound: converged=" << s.converged << " after " << s.iterations
                  << " iterations, c = " << x[2] << "\n";
        CHECK(s.converged, "a parameter held at its bound does not block convergence");
        CHECK(x[2] == 15.0, "... and stays on the bound");
        CHECK(s.iterations < 100, "... without running to the iteration limit");
    }

    // 3. Relative tests off: only the absolute floor can end the solve (without
    //    it, it stalls on rejected steps and reports no convergence).
    {
        DecayProblem pb(40000, 0.0);
        specfit::LMSolverOptions o = opt;
        o.gradient_tolerance = o.step_tolerance = o.chi2_tolerance = 1e-300;
        Eigen::VectorXd xOff = start(), xOn = start();
        o.chi2_abs_tolerance = 0.0;
        const auto sOff = solve(pb, xOff, o);
        o.chi2_abs_tolerance = 1e-2;
        const auto sOn = solve(pb, xOn, o);
        std::cout << "floor: off converged=" << sOff.converged << " after " << sOff.iterations
                  << ", on converged=" << sOn.converged << " after " << sOn.iterations << "\n";
        CHECK(!sOff.converged, "without the floor nothing ends the solve");
        CHECK(sOn.converged, "the absolute floor ends it");
        CHECK(sOn.iterations <= sOff.iterations, "... no later than the solve stalls without it");
        CHECK(atTruth(pb, xOn), "... without stopping early");
        CHECK(std::abs(sOn.final_chi2 - sOff.final_chi2) < 0.05,
              "... at the chi2 a solve without the floor reaches");
    }

    if (g_failures) { std::cerr << g_failures << " failure(s)\n"; return 1; }
    std::cout << "lm_convergence: all passed\n";
    return 0;
}
