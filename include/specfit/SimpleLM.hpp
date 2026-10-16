#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>

namespace specfit {

/* ---------------------------  user visible bits  --------------------------- */
/*  A value ≤ 0 means "determine automatically".                              */

/*
 *  Convergence is tested per free parameter, never on a single "loudest" one.
 *
 *  The parameters of a spectral fit carry incommensurate units (K, dex, km/s,
 *  continuum counts), so a raw   max_j |g_j| < tol   is decided entirely by
 *  whichever parameter happens to have the largest gradient in its own units;
 *  log g could be left far from its stationary point because Teff had already
 *  reached the shared threshold.  Every test below is therefore made
 *  dimensionless by the column norm  d_j = sqrt((JᵀJ)_jj)  -- the same Fletcher
 *  scaling the damping already uses -- and must hold for *all* free parameters:
 *
 *    gradient :  |g_j| / (d_j ||r||)              < gradient_tolerance
 *                (the cosine of the angle between r and column j of J)
 *    step     :  d_j |dx_j| / ||r||               < step_tolerance
 *                (the model change along j, in units of sigma, relative to r)
 *    chi2     :  |predicted reduction| / max(1,chi2) < chi2_tolerance
 *
 *  The relative chi2 test alone cannot end a large joint fit: with chi2 ~ 1e5
 *  it asks for steps that improve chi2 by less than 1e-5, below the noise of
 *  the model interpolation, so such fits ran to the iteration limit and were
 *  reported as not converged however long they ran.  The residuals are in
 *  units of sigma, so chi2 itself has an absolute scale: a change of 1 is one
 *  standard deviation of a single parameter.  Hence also
 *
 *    flat     :  chi2_abs_steps accepted steps in a row, each improving chi2
 *                by less than chi2_abs_tolerance (actual and predicted)
 *    stalled  :  the solver gives up after max_consecutive_rejects rejected
 *                steps, and its last accepted step was already flat
 *
 *  both of which count as converged: the parameters are then fixed to far
 *  better than their own uncertainties.  A non-positive chi2_abs_tolerance
 *  turns both off.
 *
 *  A parameter held at one of its bounds by its own gradient (vsini at 0, a
 *  velocity at the edge of its range, an abundance at the grid's edge) is
 *  *active*: its descent direction points out of the box, so its gradient can
 *  never vanish and its unconstrained step is always clipped away.  Such a
 *  parameter is held fixed in that iteration's step and left out of both
 *  per-parameter tests -- the bound-constrained (KKT) form of the tests.
 *  Before, one active parameter kept every fit with it from ever passing
 *  either test, and the clipped step also spoiled the predicted reduction,
 *  so good steps were rejected.
 *
 *  d_j is a natural scale for parameter j and is finite even when x_j == 0,
 *  which a plain relative test on |x_j| is not (vrad and z legitimately sit at
 *  zero).  A column that is identically zero means the parameter cannot move
 *  the model at all; it is treated as converged rather than as a blocker.
 */
/*
 *  Structural sparsity of the Jacobian, if the caller knows it.
 *
 *  In a joint fit of D spectra the continuum anchors of spectrum d can only
 *  move that spectrum's residuals: their Jacobian columns are identically
 *  zero everywhere else.  Forming JᵀJ as one dense product spends almost all
 *  of its time multiplying those zeros -- 57 MFLOP per iteration on a
 *  5-spectrum fit, against 6 MFLOP for the blocks that can be non-zero.
 *
 *  Each block claims a half-open range of parameters (in the *full* parameter
 *  numbering) and the half-open range of residual rows outside which every
 *  one of those columns vanishes.  Blocks must be disjoint, ascending and
 *  cover every parameter; leaving the vector empty keeps the dense product.
 *
 *  GAEL_LM_CHECK_BLOCKS=1 recomputes JᵀJ densely and reports the largest
 *  disagreement, which is what a wrong block declaration looks like.
 */
struct LMColumnBlock {
    int col_begin, col_end;      // parameters   [begin, end)
    int row_begin, row_end;      // residual rows[begin, end)
};

struct LMSolverOptions {
    int    max_iterations        = 200;      // hard upper limit
    double gradient_tolerance    = 0;        // auto
    double step_tolerance        = 0;        // auto
    double chi2_tolerance        = 0;        // auto
    double chi2_abs_tolerance    = 1e-2;     // absolute floor, chi2 units; <= 0 off
    int    chi2_abs_steps        = 3;        // flat accepted steps in a row
    double initial_lambda        = 0;        // auto
    int    max_consecutive_rejects = 12;     // give up when LM cannot progress
    bool   verbose               = false;    // chatty?
    std::vector<LMColumnBlock> column_blocks;   // empty == dense JᵀJ

    /*  The functor already writes only the free columns, in the solver's own
     *  reduced numbering (see MultiDatasetCost::set_column_map), so the solver
     *  can work on that matrix directly instead of allocating a second one and
     *  copying the free columns into it every time the Jacobian is rebuilt --
     *  1.3 GB allocated, zeroed and copied per accepted step on an 18-arm
     *  metal fit, to drop about twenty columns of it.
     *
     *  A caller setting this MUST give the functor the matching column map:
     *  the flag and the map describe the same matrix.                       */
    bool   reduced_jacobian      = false;

    /*  Called once per iteration, before the step is computed, with the
     *  1-based iteration number, the iteration budget and the chi2 the
     *  solver is currently sitting on.  Returning false stops the solve at
     *  that boundary and sets LMSolverSummary::aborted; the parameter vector
     *  is left at the last accepted point, so the caller can unwind on a
     *  consistent state.  An empty callback costs one branch per iteration.
     *
     *  Iterations are the only structural handle a caller has on how far a
     *  solve has got, which is what the fit's progress reporting is built
     *  on -- see FitProgress.hpp.                                          */
    std::function<bool(int iteration, int max_iterations, double chi2)>
        iteration_callback;
};

/*  Defaults for the scaled tolerances above.  They are dimensionless, so --
 *  unlike the old 1e-4 * max|g| -- they do not have to be re-derived from the
 *  problem and mean the same thing in every stage.                          */
inline constexpr double kLMGradientTol = 1e-6;
inline constexpr double kLMStepTol     = 1e-8;
inline constexpr double kLMChi2Tol     = 1e-10;

/*  How far, and how often, the step test may back the damping off before it
 *  accepts that the starting point really is stationary.
 *
 *  The retry stops at the *first* lambda that produces a meaningful step, so a
 *  gentle factor lands on the least reduction that works rather than dropping
 *  straight to an undamped Gauss-Newton step on a matrix that has just been
 *  shown to be badly scaled.  Three decades a time clears the worst start
 *  observed (lambda = 1.1e13 where 1e-3 was well scaled) in three retries, and
 *  eight of them bound the cost: each retry is one LDLT solve, which on a
 *  2500-parameter joint fit is not free.                                    */
inline constexpr double kLMLambdaBackoff     = 1e-3;
inline constexpr int    kLMMaxLambdaBackoffs = 8;

struct LMSolverSummary {
    int    iterations         = 0;
    double initial_chi2       = 0.0;
    double final_chi2         = 0.0;
    bool   converged          = false;
    bool   aborted            = false;         // iteration_callback said stop
    std::vector<double> param_uncertainties;   // 1-σ; 0 = fixed
};

/* --------------------  internal helper (column selection)  ----------------- */

inline
void build_free_index(const std::vector<bool>& mask,
                      Eigen::VectorXi&         map_full_to_reduced,
                      int&                     n_free)
{
    const int n = static_cast<int>(mask.size());
    map_full_to_reduced.resize(n);
    n_free = 0;
    for (int j = 0; j < n; ++j) {
        if (mask.empty() || mask[j])
            map_full_to_reduced[j] = n_free++;
        else
            map_full_to_reduced[j] = -1;
    }
}

/*  A column block translated into the reduced (free-parameters-only)
 *  numbering that the normal equations actually use.                        */
struct LMReducedBlock { int col, ncol, row, nrow; };

inline std::vector<LMReducedBlock>
reduce_column_blocks(const std::vector<LMColumnBlock>& blocks,
                     const Eigen::VectorXi&            col_index,
                     int                               n_free,
                     std::ptrdiff_t                    m)
{
    std::vector<LMReducedBlock> out;
    if (blocks.empty()) return out;

    int covered = 0;
    for (const auto& b : blocks) {
        int first = -1, count = 0;
        for (int j = b.col_begin; j < b.col_end; ++j) {
            if (j < 0 || j >= col_index.size()) continue;
            const int c = col_index[j];
            if (c < 0) continue;
            if (first < 0) first = c;
            ++count;
        }
        if (count == 0) continue;
        /*  The reduced numbering keeps the original order, so the free
         *  columns of a block stay contiguous.  If that ever stops being
         *  true the declaration is unusable -- fall back to dense.          */
        if (first + count > n_free) return {};
        const int r0 = std::max(0, b.row_begin);
        const int r1 = std::min<int>(static_cast<int>(m), b.row_end);
        if (r1 <= r0) return {};
        out.push_back({first, count, r0, r1 - r0});
        covered += count;
    }
    if (covered != n_free) return {};      // incomplete cover: not usable

    /*  Every block reaching every row means there is no structure to exploit
     *  -- a single spectrum, typically.  Splitting the product up would then
     *  only trade one rank update for several smaller ones.                 */
    bool any_restricted = false;
    for (const auto& b : out)
        if (b.row != 0 || b.nrow != static_cast<int>(m)) any_restricted = true;
    if (!any_restricted) return {};

    return out;
}

/*  g = Jfᵀ r and JTJ = JfᵀJf, skipping the structurally-zero blocks.        */
inline void normal_equations(const Eigen::MatrixXd&             Jf,
                             const Eigen::VectorXd&             r,
                             const std::vector<LMReducedBlock>& blocks,
                             Eigen::MatrixXd&                   JTJ,
                             Eigen::VectorXd&                   g)
{
    JTJ.setZero();

    if (blocks.empty()) {                  // dense: original behaviour
        g.noalias() = Jf.transpose() * r;
        JTJ.selfadjointView<Eigen::Lower>().rankUpdate(Jf.adjoint(), 1.0);
        JTJ.template triangularView<Eigen::StrictlyUpper>() = JTJ.transpose();
        return;
    }

    for (const auto& b : blocks)
        g.segment(b.col, b.ncol).noalias() =
            Jf.block(b.row, b.col, b.nrow, b.ncol).transpose()
                * r.segment(b.row, b.nrow);

    /*  Plain products throughout, including on the diagonal blocks.  A rank
     *  update would halve their arithmetic, but Eigen cannot route one into
     *  a Block destination through the BLAS back-end and the generic fallback
     *  measured slower than just doing the full product (1.63 s vs 1.35 s on
     *  a 5-spectrum fit).                                                    */
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const auto& bi = blocks[i];
            const auto& bj = blocks[j];
            const int r0 = std::max(bi.row, bj.row);
            const int r1 = std::min(bi.row + bi.nrow, bj.row + bj.nrow);
            if (r1 <= r0) continue;        // no shared rows: block is zero

            /* blocks are ascending in column, so this lands strictly below
               the diagonal and the symmetrise below fills its mirror */
            JTJ.block(bi.col, bj.col, bi.ncol, bj.ncol).noalias() =
                Jf.block(r0, bi.col, r1 - r0, bi.ncol).transpose()
                    * Jf.block(r0, bj.col, r1 - r0, bj.ncol);
        }
    }

    JTJ.template triangularView<Eigen::StrictlyUpper>() = JTJ.transpose();
}

struct LMWorkspace
{
    Eigen::MatrixXd Jf;
    Eigen::MatrixXd JTJ;
    Eigen::MatrixXd JTJ0;      // undamped, reused while the Jacobian stands
    Eigen::VectorXd diag_JTJ;
    Eigen::VectorXd g;
    Eigen::VectorXd dx_free;
    Eigen::VectorXd dx;

    /*  `need_Jf` is false when the cost functor already produces the reduced
     *  Jacobian, in which case this workspace's copy of it is never read and
     *  must not be allocated -- it is the largest array here by far.  Every
     *  other array is still sized from n_free.                              */
    void resize(std::ptrdiff_t m,
                std::ptrdiff_t n_free,
                std::ptrdiff_t n_full,
                bool           need_Jf = true)
    {
        /* grow when necessary … */
        if (need_Jf && (Jf.rows() < m || Jf.cols() < n_free))
            Jf.resize  (std::max<std::ptrdiff_t>(Jf.rows(),  m),
                        std::max<std::ptrdiff_t>(Jf.cols(), n_free));

        if (JTJ.rows() < n_free || JTJ.cols() < n_free)
            JTJ.resize (n_free, n_free);            // square, so grow once
        if (JTJ0.rows() < n_free || JTJ0.cols() < n_free)
            JTJ0.resize(n_free, n_free);

        if (diag_JTJ.size() < n_free) diag_JTJ.resize(n_free);
        if (g.size()        < n_free) g.resize       (n_free);
        if (dx_free.size()  < n_free) dx_free.resize (n_free);
        if (dx.size()       < n_full) dx.resize      (n_full);

        /* … and then shrink logically to the exact size that is
           required for the *current* problem instance (no re-alloc): */
        if (need_Jf) Jf.conservativeResize(m, n_free);
        else         Jf.resize(0, 0);
        JTJ.conservativeResize (n_free, n_free);
        JTJ0.conservativeResize(n_free, n_free);
        diag_JTJ.conservativeResize(n_free);
        g.conservativeResize       (n_free);
        dx_free.conservativeResize (n_free);
        dx.conservativeResize      (n_full);
    }
};

/* -------------------  Levenberg–Marquardt driver routine  ------------------ */

template<typename Functor>
LMSolverSummary
levenberg_marquardt(Functor&&                    func,
                    Eigen::VectorXd&             x,
                    const std::vector<bool>&     free_mask,
                    const std::vector<double>&   lower,
                    const std::vector<double>&   upper,
                    const LMSolverOptions&       user_opt = {},
                    LMWorkspace*                 work      = nullptr)
{
    LMSolverSummary summ;
    const int n = static_cast<int>(x.size());
    LMSolverOptions opt = user_opt;               // mutable copy

    if (opt.verbose){
        std::cout << "[LM] Entered LevMarq Function" << std::endl;
        std::cout << "[LM] Mapping parameter vector" << std::endl;
    }
        

    /* --------------------------------------------------------------- */
    /*  map full parameter vector  ->  free (variable) parameters      */
    /* --------------------------------------------------------------- */
    Eigen::VectorXi col_index;
    int n_free = 0;
    build_free_index(free_mask, col_index, n_free);
    if (n_free == 0) {                            // nothing to fit
        std::cout << "[LM]  Warning: All parameters are frozen! There is nothing to fit..." << std::endl;
        summ.converged  = true;
        summ.final_chi2 = 0.0;
        return summ;
    }

    if (opt.verbose)
        std::cout << "[LM] Mapped Parameter Vector. First model evaluation" << std::endl;
    
    /* --------------------------------------------------------------- */
    /*  first model evaluation                                         */
    /* --------------------------------------------------------------- */
    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    if (opt.verbose)
            std::cout << "[LM] Function Evaluation" << std::endl;
    func(x, &r, &J);
    if (opt.verbose)
            std::cout << "[LM] Evaluation done." << std::endl;

    const std::size_t m = static_cast<std::size_t>(r.size());
    double chi2 = r.squaredNorm();
    summ.initial_chi2 = chi2;

    if (opt.verbose)
        std::cout << "[LM] Evaluated Model once. Determining initial λ and tolerances." << std::endl;

    /* --------------------------------------------------------------- */
    /*  automatic tolerances and initial λ                             */
    /* --------------------------------------------------------------- */
    const double eps   = std::numeric_limits<double>::epsilon();

    /* All three tolerances are dimensionless (see LMSolverOptions). */
    if (opt.gradient_tolerance <= 0.0) opt.gradient_tolerance = kLMGradientTol;
    if (opt.step_tolerance     <= 0.0) opt.step_tolerance     = kLMStepTol;
    if (opt.chi2_tolerance     <= 0.0) opt.chi2_tolerance     = kLMChi2Tol;

    if (opt.initial_lambda  <= 0.0) {
        /*  Only the diagonal of JᵀJ is wanted, and the diagonal is just the
         *  squared norm of each column.  Forming the whole product to throw
         *  n_free*(n_free-1) of its entries away costs m*n^2 flops instead of
         *  m*n -- 65 GFLOP against 0.24 on an 18-spectrum metal fit, once per
         *  levenberg_marquardt() call, plus an n x n temporary.              */
        Eigen::VectorXd diag = J.colwise().squaredNorm();
        opt.initial_lambda   = 1e-3 * diag.maxCoeff();
        if (opt.initial_lambda == 0.0) opt.initial_lambda = 1e-3;
    }
    double lambda = opt.initial_lambda;

    if (opt.verbose)
        std::cout << "[LM] Determinined initial λ and tolerances. One time allocations." << std::endl;


    /* --------------------------------------------------------------- */
    /*  one-time allocations                                           */
    /* --------------------------------------------------------------- */
    Eigen::MatrixXd Jf_local, JTJ_local, JTJ0_local;
    Eigen::VectorXd diag_local, g_local, dx_free_local, dx_local;

    /* pick the storage that will actually be used */
    Eigen::MatrixXd &Jf        = work ? work->Jf        : Jf_local;
    Eigen::MatrixXd &JTJ       = work ? work->JTJ       : JTJ_local;
    Eigen::MatrixXd &JTJ0      = work ? work->JTJ0      : JTJ0_local;
    Eigen::VectorXd &diag_JTJ  = work ? work->diag_JTJ  : diag_local;
    Eigen::VectorXd &g         = work ? work->g         : g_local;
    Eigen::VectorXd &dx_free   = work ? work->dx_free   : dx_free_local;
    Eigen::VectorXd &dx        = work ? work->dx        : dx_local;

    /*  With a reduced Jacobian, `J` *is* the reduced matrix: Jf would be an
     *  exact second copy of it, so it is left empty and every use below reads
     *  `Jr` instead.                                                        */
    const bool reduced = opt.reduced_jacobian;

    /* make sure the arrays are big enough (may allocate once) */
    if (work) work->resize(m, n_free, n, /*need_Jf=*/!reduced);
    else {                 // original behaviour
        if (!reduced) Jf.resize(m, n_free);
        JTJ.resize (n_free, n_free);
        JTJ0.resize(n_free, n_free);
        diag_JTJ.resize(n_free);
        g.resize(n_free);
        dx_free.resize(n_free);
        dx.resize(n);
    }

    /*  The matrix the normal equations are actually formed from. */
    Eigen::MatrixXd& Jr = reduced ? J : Jf;

    if (opt.verbose) {
        std::cout << "[LM] One time allocations complete." << std::endl;
        std::cout << "[LM]  Entering iteration loop..." << std::endl;
    }

    /*  Structural sparsity, if the caller declared any (see LMColumnBlock). */
    const std::vector<LMReducedBlock> blocks =
        reduce_column_blocks(opt.column_blocks, col_index, n_free,
                             static_cast<std::ptrdiff_t>(m));
    const bool check_blocks = [] {
        const char* e = std::getenv("GAEL_LM_CHECK_BLOCKS");
        return e && *e && std::string(e) != "0";
    }();

    int consecutive_rejects = 0;

    std::vector<char> active;          // per free parameter: held at a bound
    Eigen::VectorXd   g_step;          // g with the active entries zeroed

    /*  The absolute chi2 test (see header): accepted steps in a row that
     *  were flat, and the improvement of the last accepted step.           */
    int    flat_steps = 0;
    double last_gain  = std::numeric_limits<double>::infinity();

    /*  Whether the solver has ever managed to move, and how often the damping
     *  has been backed off because it had not.  See the step test below.    */
    bool accepted_any    = false;
    int  lambda_backoffs = 0;

    /*  A rejected step leaves J and r exactly as they were, so JᵀJ and Jᵀr
     *  come out identical and only the damping changes.  Rebuilding them
     *  anyway was the single most expensive thing the solver did once the
     *  cost function had been sped up (57 MFLOP per iteration on a
     *  5-spectrum fit, half of them for steps that were thrown away).      */
    bool jacobian_is_stale = true;

    /* --------------------------------------------------------------- */
    /*  main iteration loop                                            */
    /* --------------------------------------------------------------- */
    for (int it = 0; it < opt.max_iterations; ++it) {
        summ.iterations = it + 1;

        /*  Progress / cancellation.  Reported before the step so a caller
         *  sees iteration 1 of a long stage immediately rather than only
         *  once the first Jacobian and candidate are done.                */
        if (opt.iteration_callback &&
            !opt.iteration_callback(summ.iterations, opt.max_iterations, chi2)) {
            summ.aborted = true;
            break;
        }

        if (jacobian_is_stale) {
            if (opt.verbose)
                std::cout << "[LM] Building Reduced Jacobian." << std::endl;

            /* ----- build reduced Jacobian (copy only the free columns) -- */
            if (!reduced)
                for (int j = 0; j < n; ++j) {
                    int col = col_index[j];
                    if (col >= 0) Jf.col(col).noalias() = J.col(j);
                }

            if (opt.verbose)
                std::cout << "[LM] Built Reduced Jacobian. Transposing." << std::endl;

            /* --------------- g = Jᵀ r  and  JTJ0 = JᵀJ ----------------- */
            normal_equations(Jr, r, blocks, JTJ0, g);

            if (check_blocks && !blocks.empty()) {
                Eigen::MatrixXd JTJ_dense(n_free, n_free);
                Eigen::VectorXd g_dense(n_free);
                normal_equations(Jr, r, {}, JTJ_dense, g_dense);
                std::cout << "[LM] block check: max|dJTJ| = "
                          << (JTJ_dense - JTJ0).cwiseAbs().maxCoeff()
                          << "  max|dg| = "
                          << (g_dense - g).cwiseAbs().maxCoeff() << '\n';
            }

            diag_JTJ = JTJ0.diagonal();

            jacobian_is_stale = false;
        }

        /* --------- per-parameter gradient test (see header) --------- *
         *  |g_j| / (sqrt(JᵀJ_jj) ||r||) is the cosine of the angle between
         *  the residual vector and column j; it is dimensionless, so the
         *  same threshold is meaningful for Teff, log g and a continuum
         *  anchor alike, and *every* free parameter has to pass it.      */
        /*  Active bounds (see header): on a bound, with the descent
         *  direction -g pointing out of the box.                         */
        active.assign(n_free, 0);
        for (int j = 0; j < n; ++j) {
            const int c = col_index[j];
            if (c < 0) continue;
            const auto on = [&](double b) {
                return std::abs(x[j] - b) <= 1e-12 * std::max(1.0, std::abs(b));
            };
            const bool at_lo = !lower.empty() && on(lower[j]);
            const bool at_hi = !upper.empty() && on(upper[j]);
            if ((at_lo && g[c] > 0.0) || (at_hi && g[c] < 0.0)) active[c] = 1;
        }

        const double rnorm = std::sqrt(chi2);
        double cos_max = 0.0;
        if (rnorm > 0.0) {
            for (int c = 0; c < n_free; ++c) {
                if (active[c]) continue;
                const double dj = std::sqrt(diag_JTJ[c]);
                if (!(dj > 0.0)) continue;         // column ≡ 0: cannot move
                cos_max = std::max(cos_max, std::abs(g[c]) / (dj * rnorm));
            }
        }
        if (cos_max < opt.gradient_tolerance) {    // all gradients small
            if (summ.iterations == 1)
                std::cout << "[LM]  Warning: starting point already satisfies the "
                             "gradient tolerance – solver stopped without iterating\n";
            summ.converged = true;
            break;
        }

        if (opt.verbose)
            std::cout << "[LM] Multiplied Matrices. Diagonalizing and solving." << std::endl;

        /* ------- (JTJ + λ D) Δx = −g   (D = diag(JTJ)) -------------- */
        JTJ = JTJ0;                                // damp a copy, keep JᵀJ
        JTJ.diagonal().array() +=
            lambda * (diag_JTJ.array() + 1e-20);   // Fletcher scaling

        /*  Active parameters stay where they are: their rows and columns
         *  are decoupled and their right-hand side zeroed, so dx_c = 0.   */
        g_step = g;
        for (int c = 0; c < n_free; ++c) {
            if (!active[c]) continue;
            JTJ.row(c).setZero();
            JTJ.col(c).setZero();
            JTJ(c, c) = 1.0;
            g_step[c] = 0.0;
        }

        dx_free = -JTJ.ldlt().solve(g_step);       // SPD solve

        if (dx_free.hasNaN() || !dx_free.allFinite()){   // numerical failure
            std::cout << "[LM]  Warning: numerical failure! Inf/NaN in solver. Aborting iteration...\n";
            break;
        }

        if (opt.verbose)
            std::cout << "[LM] Diagonalized. Copying step." << std::endl;

        /* --------------- copy step into full parameter vector ------- */
        dx.setZero();
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) dx[j] = dx_free[col];
        }

        /* ---- per-parameter step test (see header) ------------------ *
         *  d_j |dx_j| / ||r|| is how much parameter j alone would move the
         *  model, in units of sigma, relative to the residual that is left.
         *  Converged only when that is negligible for *every* free
         *  parameter, so a small step in Teff cannot end the fit while a
         *  continuum anchor or log g is still travelling.                */
        {
            double step_max = 0.0;
            const double rnorm_s = std::max(std::sqrt(chi2), eps);
            for (int c = 0; c < n_free; ++c)
                if (!active[c])
                    step_max = std::max(step_max,
                                        std::sqrt(diag_JTJ[c]) * std::abs(dx_free[c])
                                            / rnorm_s);
            if (step_max < opt.step_tolerance) {
                /*  A step this small means "x is stationary" only once the
                 *  solver has actually been somewhere.  At the starting
                 *  point, with lambda still at the value derived from the
                 *  largest column of J, it far more often means the damping
                 *  is simply too large for this problem: lambda multiplies
                 *  diag(JtJ) again (Fletcher scaling), so one ill-conditioned
                 *  column -- two continuum knots that landed on top of each
                 *  other, an anchor with no noticed pixel beside it -- damps
                 *  every *other* parameter into immobility with it.  On a
                 *  115-spectrum joint fit that turned one bad anchor into a
                 *  fit that reported convergence after a single iteration
                 *  with every parameter still at its initial value.
                 *
                 *  So before believing it, back the damping off and re-solve.
                 *  The Jacobian is not stale, so this costs one LDLT solve;
                 *  a fit that was already moving never reaches this branch
                 *  (its first step is orders of magnitude above the
                 *  tolerance), which is what keeps the guard inert.         */
                if (!accepted_any && lambda_backoffs < kLMMaxLambdaBackoffs) {
                    lambda *= kLMLambdaBackoff;
                    lambda  = std::max(lambda, 1e-18);
                    ++lambda_backoffs;
                    if (opt.verbose)
                        std::cout << "[LM]  first step below step tolerance ("
                                  << step_max << " < " << opt.step_tolerance
                                  << ") -- reducing lambda to " << lambda
                                  << " and retrying\n";
                    continue;
                }
                if (!accepted_any)
                    std::cout << "[LM]  Warning: no step above the step "
                                 "tolerance even at lambda = " << lambda
                              << " - solver stopped without iterating\n";
                summ.converged = true;
                break;
            }
        }

        if (opt.verbose)
            std::cout << "[LM] Copied step. Calculating candidate." << std::endl;

        /* --------------------- candidate point ---------------------- */
        Eigen::VectorXd x_try = x + dx;

        /* ---------- simple bound constraints (project) -------------- */
        for (int j = 0; j < n; ++j) {
            if (!lower.empty()) x_try[j] = std::max(x_try[j], lower[j]);
            if (!upper.empty()) x_try[j] = std::min(x_try[j], upper[j]);
        }

        /* -------- recompute dx_free based on actual step taken ------ */
        Eigen::VectorXd dx_actual = x_try - x;
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) dx_free[col] = dx_actual[j];
        }

        /*  Residuals only.  Roughly half of all LM steps are rejected (measured:
         *  482 of 930 on a 5-spectrum joint fit), and a rejected step's Jacobian
         *  is discarded unread -- it used to be computed anyway, at ~14 residual
         *  evaluations plus a continuum spline per anchor.  The Jacobian is now
         *  built only once the step is known to be kept, which costs one extra
         *  residual evaluation per accepted step and saves a whole Jacobian per
         *  rejected one.  Every value below is unchanged.                      */
        Eigen::VectorXd r_try;
        if (opt.verbose)
            std::cout << "[LM] Function Evaluation" << std::endl;
        func(x_try, &r_try, nullptr);
        if (opt.verbose)
            std::cout << "[LM] Evaluation done." << std::endl;
        double chi2_try = r_try.squaredNorm();

        if (opt.verbose)
            std::cout << "[LM] Calculated candidate. Powell test." << std::endl;

        /* ------------------- Powell's ρ test ------------------------ */
        Eigen::VectorXd tmp = lambda * (diag_JTJ.array() * dx_free.array()).matrix() - g_step;
        double pred_red = 0.5 * dx_free.dot(tmp);
        if (pred_red <= 0.0) pred_red = eps;

        const double gain = chi2 - chi2_try;
        double rho     = gain / pred_red;
        bool   accept  = rho > 0.0 && chi2_try < chi2;

        if (accept) {
            /* --------------- successful iteration ------------------ */
            x.swap(x_try);
            r.swap(r_try);
            chi2 = chi2_try;
            consecutive_rejects = 0;
            accepted_any        = true;

            /*  Jacobian at the point we just moved to: needed by the next
             *  iteration, and by the uncertainty block if this is the last.  */
            func(x, nullptr, &J);
            jacobian_is_stale = true;

            /* adaptive λ (MINPACK style) ---------------------------- */
            double fac = std::max(1.0/3.0,
                                  1.0 - std::pow(2.0*rho - 1.0, 3.0));
            lambda *= fac;
            lambda  = std::max(lambda, 1e-18);

            if (opt.verbose)
                std::cout << "[LM]  iter " << it
                          << "  ρ="  << std::fixed << std::setprecision(2) << rho
                          << "  χ²=" << std::fixed << std::setprecision(2) << chi2
                          << "  λ="  << std::scientific << std::setprecision(2) << lambda
                          << "  (accepted)\n";

            if (std::abs(pred_red) < opt.chi2_tolerance * std::max(1.0, chi2)) {
                summ.converged = true;
                break;
            }
            last_gain = gain;
            if (opt.chi2_abs_tolerance > 0.0 && gain < opt.chi2_abs_tolerance &&
                std::abs(pred_red) < opt.chi2_abs_tolerance) {
                if (++flat_steps >= std::max(1, opt.chi2_abs_steps)) {
                    summ.converged = true;
                    break;
                }
            } else {
                flat_steps = 0;
            }
        } else {
            /* ------------------- rejected step --------------------- */
            lambda *= 2.0;
            ++consecutive_rejects;
            if (opt.verbose)
                std::cout << "[LM]  iter " << it
                          << "  ρ="  << std::fixed << std::setprecision(2) << rho
                          << "  χ²=" << std::fixed << std::setprecision(2) << chi2_try
                          << "  λ="  << std::scientific << std::setprecision(2) << lambda
                          << "  (rejected)\n";

            /* Damping has been raised this many times without a single
               improvement: the model cannot be improved from here (usually
               finite-difference noise in the Jacobian), so stop rather than
               burn the whole iteration budget on rejected steps.        */
            if (consecutive_rejects >= opt.max_consecutive_rejects) {
                /*  Stalled where the last accepted step was already flat:
                    the minimum is pinned to the precision the Jacobian
                    allows, which is all a further step could add.         */
                if (opt.chi2_abs_tolerance > 0.0 && accepted_any &&
                    last_gain < opt.chi2_abs_tolerance)
                    summ.converged = true;
                break;
            }
        }
    }

    summ.final_chi2 = chi2;

    if (opt.verbose)
            std::cout << "[LM] Iter complete, getting uncertainties." << std::endl;

    /* ---------------------  propagate uncertainties  ---------------- */
    summ.param_uncertainties.assign(n, 0.0);
    if (n_free > 0) {
        /* reuse Jf and JTJ already allocated ------------------------- */
        if (!reduced)
            for (int j = 0; j < n; ++j) {
                int col = col_index[j];
                if (col >= 0) Jf.col(col).noalias() = J.col(j);
            }
        normal_equations(Jr, r, blocks, JTJ, g);

        const double dof = std::max<std::size_t>(m - n_free, 1);
        const double var = r.squaredNorm() / dof;             // σ² ≈ χ²/dof

        Eigen::MatrixXd cov =
            JTJ.ldlt().solve(Eigen::MatrixXd::Identity(n_free, n_free));
        cov *= var;

        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0)
                summ.param_uncertainties[j] =
                    std::sqrt(std::max(0.0, cov(col, col)));
        }
    }

    if (opt.verbose)
            std::cout << "[LM] LevMarq done." << std::endl;
    return summ;
}

} // namespace specfit