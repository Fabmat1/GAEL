
#include "specfit/UnifiedFitWorkflow.hpp"
#include "specfit/MultiDatasetCost.hpp"
#include "specfit/SimpleLM.hpp"
#include "specfit/Powell.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <string>

using Eigen::ArrayXd;

namespace specfit {

/* ------------------------------------------------------------------------- *
 *  Investigation knobs (env vars, off by default)
 *    GAEL_STAGE_TIMING=1   -> print wall time + chi2 + parameters per stage
 *    GAEL_STAGES=<variant> -> run a different stage ladder:
 *        noladder (default) 1,    4,5,6,7
 *        full               1,2,3,4,5,6,7  (the legacy ladder)
 *        direct                  4,5,6,7   (no continuum-only pre-fit either)
 *        nonoise            1,    4,5,  7  (drop iterative noise rescaling)
 *
 *  Stages 2 (continuum+vrad) and 3 (continuum+vrad+teff/logg/z) were measured
 *  to cost ~32 % of the wall time of a 5-spectrum fit while changing the log g
 *  bias by 0.0007 dex and the Teff scatter by 1 K over 100 real fits, so the
 *  ladder skips them by default.  Stage 1 stays (without it stage 4 starts far
 *  from the continuum solution and needs more full-Jacobian iterations than
 *  stages 2+3 cost) and so does stage 6 (dropping it inflates the Teff scatter
 *  1.8x for no time saving).
 * ------------------------------------------------------------------------- */
namespace {
std::string stage_variant()
{
    const char* v = std::getenv("GAEL_STAGES");
    return v ? std::string(v) : std::string("noladder");
}
bool stage_timing()
{
    const char* v = std::getenv("GAEL_STAGE_TIMING");
    return v && std::string(v) == "1";
}
} // namespace

/* ------------------------------------------------------------------------- */
/*  constructor                                                              */
/* ------------------------------------------------------------------------- */
UnifiedFitWorkflow::UnifiedFitWorkflow(
        std::vector<DataSet>& datasets,
        SharedModel&          model,
        const Config&         config,
        const std::vector<std::map<std::string,bool>>& frozen_status,
        int                   nthreads)
    : datasets_(datasets)
    , model_(model)
    , config_(config)
    , frozen_status_(frozen_status)
    , nthreads_(nthreads)
{
    /* ================================================================ */
    /*  0.  build stellar-parameter indexer                             */
    /* ================================================================ */
    const int n_components = static_cast<int>(model_.params.size());
    const int n_datasets   = static_cast<int>(datasets_.size());

    /*  Which parameters a component has comes from its grid: the canonical
     *  eight, one per element axis the grid resolves, then sur_ratio.  An
     *  HHE-only grid gives the historical nine.                            */
    std::vector<std::vector<ParamSpec>> specs;
    specs.reserve(static_cast<std::size_t>(n_components));
    for (int c = 0; c < n_components; ++c)
        specs.push_back(component_params(
            c < static_cast<int>(model_.grids.size()) ? model_.grids[c].species()
                                                      : std::vector<std::string>{}));

    indexer_.build(specs, n_datasets, config_.untie_params);

    /* ----------  collect initial parameters into one big vector -------- */
    unified_params_.resize(indexer_.total_stellar_params);
    for (int c = 0; c < n_components; ++c) {
        const auto& sp    = model_.params[c];
        const auto& table = indexer_.params(c);
        for (int d = 0; d < n_datasets; ++d)
            for (int s = 0; s < static_cast<int>(table.size()); ++s)
                unified_params_[ indexer_.get(c,d,s) ] =
                    get_stellar_param(sp, table[static_cast<std::size_t>(s)]);
    }
    /* ----------  telluric block, then the continuum block --------------- *
     *  Layout of the global vector: [stellar][telluric][continuum anchors].
     *  The telluric parameters belong to a spectrum rather than to a stellar
     *  component, so they cannot live in the indexer; they sit in their own
     *  block, and the continuum stays last so that everything which locates
     *  it by counting back from the end keeps working.                     */
    telluric_offset_ = indexer_.total_stellar_params;
    n_telluric_      = 0;
    frozen_telluric_.assign(datasets_.size(), {false, false, false});
    for (const auto& ds : datasets_) {
        if (!(ds.telluric_enabled && model_.telluric)) continue;
        for (int k = 0; k < kNTelluricParams; ++k)
            unified_params_.push_back(ds.telluric[static_cast<std::size_t>(k)]);
        n_telluric_ += kNTelluricParams;
    }

    for (const auto& ds : datasets_)
        unified_params_.insert(unified_params_.end(),
                               ds.cont_y.begin(), ds.cont_y.end());
}

/*  Absolute index of dataset d's first telluric parameter, or -1. */
int UnifiedFitWorkflow::telluric_param_offset(std::size_t d) const
{
    if (!model_.telluric) return -1;
    int off = telluric_offset_;
    for (std::size_t i = 0; i < datasets_.size(); ++i) {
        if (!datasets_[i].telluric_enabled) continue;
        if (i == d) return off;
        off += kNTelluricParams;
    }
    return -1;
}

/* ------------------------------------------------------------------------- *
 *  Grid coverage, intersected over every component's grid.  A parameter tied
 *  across two components has to stay inside both.
 * ------------------------------------------------------------------------- */
ModelGrid::ParameterBounds UnifiedFitWorkflow::grid_bounds() const
{
    std::vector<ModelGrid::ParameterBounds> all;
    all.reserve(model_.grids.size());
    for (const auto& g : model_.grids) all.push_back(g.get_parameter_bounds());
    return ModelGrid::intersect(all);
}

/* ------------------------------------------------------------------------- *
 *  Solver limits on one parameter.  The grid answers for the axes it has
 *  (teff, logg, xi, z, he and, on a metal grid, each element abundance); the
 *  rest is fit policy and matches ISIS's stellar_set_ranges.
 * ------------------------------------------------------------------------- */
std::pair<double,double>
UnifiedFitWorkflow::param_limits(const ParamSpec& ps, int comp,
                                 const ModelGrid::ParameterBounds& gb) const
{
    constexpr double kWide = 1.0e10;      // "unbounded", as the solver sees it

    switch (ps.kind) {
        case ParamKind::Vrad:  return { -1000.0, 1000.0 };
        case ParamKind::Vsini: return {     0.0,  500.0 };
        case ParamKind::Zeta:  return {  -kWide,  kWide };   // deliberately free
        case ParamKind::SurRatio:
            /*  Component 1 defines the scale and is pinned to 1; the others
             *  get ISIS's range (stellar_set_ranges: 0 .. 1500).            */
            return (comp == 0) ? std::make_pair(1.0, 1.0)
                               : std::make_pair(0.0, 1500.0);
        case ParamKind::Abundance: {
            /*  Each element is bounded by its own axis in this component's
             *  grid -- unlike the stellar axes there is nothing to intersect,
             *  since an abundance is never shared between components.       */
            if (comp < static_cast<int>(model_.grids.size()))
                if (const GridAxis* ax = model_.grids[comp].axis(ps.name))
                    if (ax->values.size() > 0)
                        return { ax->values[0],
                                 ax->values[ax->values.size() - 1] };
            return { -kWide, kWide };
        }
        default: {
            /*  Bound a component by *its own* grid.  Every component has its
             *  own teff/logg/xi/z/he slot -- tying is across datasets, not
             *  across components -- so the intersection over all grids
             *  over-constrains a binary whose components sit on different
             *  grids: with Feros_3 (21-26 kK) and Feros_5 (25-29 kK) the
             *  intersection is 25-26 kK, which excluded both components'
             *  true temperatures and collapsed the fit.  `gb` remains the
             *  fallback for a component with no grid of its own.           */
            if (comp < static_cast<int>(model_.grids.size())) {
                const auto own = model_.grids[comp].get_parameter_bounds();
                if (auto b = own.for_kind(ps.kind)) return *b;
            }
            if (auto b = gb.for_kind(ps.kind)) return *b;
            return { -kWide, kWide };
        }
    }
}

/* ------------------------------------------------------------------------- *
 *  Push the current solution back into model_.params, so callers reading the
 *  model see the fit.  Dataset 0 is the representative one: a tied parameter
 *  has the same value everywhere, and an untied one (vrad) has never had a
 *  single value to report.
 * ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::sync_model_params()
{
    for (int c = 0; c < static_cast<int>(model_.params.size()); ++c)
        model_.params[static_cast<std::size_t>(c)] =
            stellar_params_from(indexer_, unified_params_, c, 0);
}

/* ------------------------------------------------------------------------- */
/*  helper that performs one optimisation stage                              */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::solve_stage(const std::set<std::string>& free_params,
                                     int max_iterations,
                                     bool add_powell)
{
    /* ---- a) gather bookkeeping info ----------------------------------- */
    static int dbg_stage_counter = 0;
    const int n_components  = static_cast<int>(model_.params.size());
    const int stellar_total = indexer_.total_stellar_params;

    std::vector<DatasetInfo> ds_infos;
    std::vector<ModelGrid*>  grid_ptrs;
    int total_residuals = 0;
    int cont_offset     = 0;
    
    for (auto& ds : datasets_) {
        DatasetInfo info;
        info.lambda            = ds.obs.lambda;
        info.flux              = ds.obs.flux;
        info.sigma             = ds.obs.sigma;
        info.ignoreflag        = ds.obs.ignoreflag;
        info.cont_x            = ds.cont_x;
        info.resOffset         = ds.resOffset;
        info.resSlope          = ds.resSlope;
        info.cont_param_offset = cont_offset;
        info.cont_param_count  = ds.cont_y.size();

        if (ds.telluric_enabled && model_.telluric) {
            info.telluric              = model_.telluric.get();
            info.telluric_param_offset =
                telluric_param_offset(ds_infos.size());
        }

        ds_infos.push_back(std::move(info));
        const int n_kept = std::accumulate(ds.obs.ignoreflag.begin(),
                               ds.obs.ignoreflag.end(), 0);
        total_residuals += n_kept;
        cont_offset     += ds.cont_y.size();
    }
    for (auto& g : model_.grids) grid_ptrs.push_back(&g);

    MultiDatasetCost cost(ds_infos, grid_ptrs, n_components,
                              indexer_,
                               total_residuals, cont_offset, n_telluric_);

    const int Npar      = stellar_total + n_telluric_ + cont_offset;
    const int cont_base = stellar_total + n_telluric_;   // first anchor

    /* ---- b)   build lower / upper bounds ------------------------------ */
    std::vector<double> lo(Npar, -1.0e10);
    std::vector<double> hi(Npar,  1.0e10);
    
    /*  Grid coverage, intersected over the components' grids; everything the
     *  grid has no opinion about keeps the wide default set above.          */
    const ModelGrid::ParameterBounds gb = grid_bounds();

    for (int c = 0; c < n_components; ++c) {
        const auto& table = indexer_.params(c);
        for (std::size_t s = 0; s < table.size(); ++s) {
            auto lim = param_limits(table[s], c, gb);
            for (std::size_t d = 0; d < datasets_.size(); ++d) {
                const int idx = indexer_.get(c, static_cast<int>(d),
                                             static_cast<int>(s));
                /*  An abundance of 10 or more means "this element is not in
                 *  the model" (ISIS: hard_max=_Inf so such a value survives).
                 *  Clamping it back onto the grid axis would silently switch
                 *  the element on again, so pin it where the user put it.   */
                if (table[s].kind == ParamKind::Abundance &&
                    unified_params_[idx] >= 10.0) {
                    lo[idx] = hi[idx] = unified_params_[idx];
                    continue;
                }
                lo[idx] = lim.first;
                hi[idx] = lim.second;
            }
        }
    }

    /* ---- continuum anchor bounds, as ISIS sets them ------------------- *
     *  set_par("cspline(1).d%d_y*"; min=0, max=2*max(get_data_counts(id).value))
     *  A continuum height is a flux: negative is unphysical, and no anchor
     *  should have to climb past twice the brightest pixel of its own
     *  spectrum.  GAEL left these unbounded (+-1e10), so a poorly constrained
     *  anchor could run away and drag the stellar parameters with it.        */
    /*  Telluric limits, as ISIS's telluric_default sets them. */
    for (std::size_t d = 0; d < datasets_.size(); ++d) {
        const int base = ds_infos[d].telluric_param_offset;
        if (base < 0) continue;
        for (int k = 0; k < kNTelluricParams; ++k) {
            const auto lim = telluric_param_limits(k);
            lo[base + k] = lim.first;
            hi[base + k] = lim.second;
        }
    }

    for (std::size_t d = 0; d < datasets_.size(); ++d) {
        const auto& ds   = datasets_[d];
        const int   base = cont_base + ds_infos[d].cont_param_offset;
        const double fmax = ds.obs.flux.size() ? ds.obs.flux.maxCoeff() : 0.0;
        const double cap  = (std::isfinite(fmax) && fmax > 0.0) ? 2.0 * fmax : 1.0e10;
        for (std::size_t i = 0; i < ds.cont_y.size(); ++i) {
            lo[base + static_cast<int>(i)] = 0.0;
            hi[base + static_cast<int>(i)] = cap;
        }
    }


    /* ---- c)   decide which parameters are free ------------------------ */
    std::vector<bool> free_mask(Npar, false);

    auto mark_free = [&](int global_idx) { free_mask[global_idx] = true; };
    
    const bool all_requested = free_params.count("all") > 0;

    // stellar params component-wise
    for (int c = 0; c < n_components; ++c)
    {
        auto& frz = frozen_status_[c];

        auto token = [&](const std::string& name){ return "c"+std::to_string(c+1)+"_"+name; };
        const auto& table = indexer_.params(c);
        for (std::size_t d = 0; d < datasets_.size(); ++d)
            for (std::size_t p = 0; p < table.size(); ++p)
            {
                const auto& ps = table[p];
                if (!(all_requested ||
                      free_params.count(token(ps.name)))) continue;
                if (frz.at(ps.name)) continue;
                /*  c1_sur_ratio *defines* the scale the other components are
                 *  measured against, so it is 1 by construction, never a
                 *  degree of freedom -- ISIS pins it the same way
                 *  (min=max=1, frozen).  A single-component fit therefore has
                 *  no free surface ratio at all, which is what keeps its
                 *  otherwise identically-zero Jacobian column out of JtJ.   */
                if (ps.kind == ParamKind::SurRatio && c == 0) continue;
                mark_free(indexer_.get(c,static_cast<int>(d),
                                       static_cast<int>(p)));
            }
    }

    /* ---- continuum anchors -------------------------------------------- *
     *  ISIS freezes anchor i when no noticed pixel lies between anchors i-1
     *  and i+1 (spectroscopy_automated.sl, "Freeze continuum points in
     *  ignored regions").  Such an anchor cannot change chi2 at all: its
     *  Jacobian column is identically zero and JTJ is singular in that
     *  direction, so leaving it free only feeds noise into the solve and into
     *  the reported uncertainties.  With wide anchor spacing and narrow masks
     *  nothing is frozen, which is why this stayed latent; on a heavily
     *  masked spectrum it is the difference between a solve and a guess.     */
    if (free_params.count("continuum")) {
        for (std::size_t d = 0; d < datasets_.size(); ++d) {
            const auto& ds  = datasets_[d];
            const auto& cx  = ds.cont_x;
            const int   na  = static_cast<int>(ds.cont_y.size());
            const int   nx  = static_cast<int>(cx.size());
            const int   base = cont_base + ds_infos[d].cont_param_offset;

            for (int i = 0; i < na; ++i) {
                const double x_lo = cx[std::max(0, i - 1)];
                const double x_hi = cx[std::min(i + 1, nx - 1)];

                /*  Compared against bin_lo, as ISIS does -- the anchor x list
                    is built on the same convention (see ContinuumUtils).     */
                bool constrained = false;
                const Eigen::Index np = ds.obs.lambda.size();
                for (Eigen::Index j = 0; j < np; ++j) {
                    if (!ds.obs.ignoreflag[j]) continue;
                    const Eigen::Index k = (j == 0) ? 1 : j;
                    const double bin_lo = ds.obs.lambda[j] -
                        0.5 * std::abs(ds.obs.lambda[k] - ds.obs.lambda[k - 1]);
                    if (bin_lo > x_lo && bin_lo < x_hi) { constrained = true; break; }
                }
                if (constrained) mark_free(base + i);
            }
        }
    }

    /* ---- telluric parameters ------------------------------------------ *
     *  ISIS leaves airmass, pwv and barycorr free (telluric_default) and
     *  freezes all three when the pipeline is told the spectrum has already
     *  had its tellurics divided out -- which here is expressed by the
     *  spectrum simply not enabling the component.  They join the fit from
     *  the same stage the stellar parameters do, since a continuum-only
     *  stage cannot see them.                                              */
    if (all_requested || free_params.count("telluric")) {
        for (std::size_t d = 0; d < datasets_.size(); ++d) {
            const int base = ds_infos[d].telluric_param_offset;
            if (base < 0) continue;
            for (int k = 0; k < kNTelluricParams; ++k) {
                if (frozen_telluric_[d][static_cast<std::size_t>(k)]) continue;
                mark_free(base + k);
            }
        }
    }

    /*  Tell the cost function which columns will actually be read: LM only
     *  ever assembles the free ones, so a frozen parameter's finite
     *  difference is thrown away.  In the continuum-only stage that is every
     *  stellar parameter -- twelve full residual evaluations per Jacobian on
     *  a five-spectrum fit, for nothing.                                    */
    cost.set_free_mask(free_mask);

    /*  Structural sparsity of the Jacobian: a stellar parameter reaches every
     *  residual, but a continuum anchor of spectrum d reaches only spectrum
     *  d's rows.  Telling LM so turns one dense JᵀJ into a handful of small
     *  blocks -- ~10x fewer multiplications on a five-spectrum joint fit,
     *  where the anchors are 107 of the 115 free parameters.                */
    std::vector<LMColumnBlock> col_blocks;
    {
        const auto& roff = cost.row_offsets();
        const auto& rcnt = cost.row_counts();
        col_blocks.push_back({0, stellar_total, 0, total_residuals});
        for (std::size_t d = 0; d < datasets_.size(); ++d) {
            /*  A spectrum's telluric parameters reach only that spectrum's
             *  rows, exactly like its continuum anchors.                    */
            const int tb = ds_infos[d].telluric_param_offset;
            if (tb >= 0)
                col_blocks.push_back({tb, tb + kNTelluricParams,
                                      roff[d], roff[d] + rcnt[d]});

            const int cb = cont_base + ds_infos[d].cont_param_offset;
            col_blocks.push_back({cb, cb + ds_infos[d].cont_param_count,
                                  roff[d], roff[d] + rcnt[d]});
        }
    }

    /* ---- d)   run Levenberg–Marquardt -------------------------------- */
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(unified_params_.data(), Npar);
    
    // Ensure initial parameters are within bounds
    for (int i = 0; i < Npar; ++i) {
        x[i] = std::clamp(x[i], lo[i], hi[i]);
    }
    
    LMSolverOptions lm_opt;
    lm_opt.max_iterations = max_iterations;
    lm_opt.verbose        = false;
    lm_opt.column_blocks  = std::move(col_blocks);
    
    // Create a wrapper functor for the cost function
    auto cost_functor = [&cost](const Eigen::VectorXd& p,
                                Eigen::VectorXd* r,
                                Eigen::MatrixXd* J) {
        cost(p, r, J);
    };
    
    // Use the standard LM with proper bounds (the bounds now match grid exactly)
    summary_ = levenberg_marquardt(cost_functor, x, free_mask, lo, hi, lm_opt);
    
    // Check for parameters at boundaries and adjust uncertainties
    const double boundary_tol = 1e-6;
    for (int i = 0; i < Npar; ++i) {
        if (!free_mask[i]) continue;
        
        bool at_lower = (x[i] - lo[i]) < boundary_tol * std::abs(lo[i] + 1.0);
        bool at_upper = (hi[i] - x[i]) < boundary_tol * std::abs(hi[i] + 1.0);
        
        if (at_lower || at_upper) {
            // Inflate uncertainty for boundary parameters
            summary_.param_uncertainties[i] *= 2.0;
            
            int comp = -1; std::string param_name;
            // Decode which parameter this is
            for (int c = 0; c < n_components && comp < 0; ++c) {
                const auto& table = indexer_.params(c);
                for (int d = 0; d < static_cast<int>(datasets_.size()); ++d) {
                    for (std::size_t p = 0; p < table.size(); ++p) {
                        if (indexer_.get(c, d, static_cast<int>(p)) == i) {
                            comp = c; param_name = table[p].name;
                            break;
                        }
                    }
                    if (comp >= 0) break;
                }
            }

            if (comp >= 0) {
                std::cout << "  Warning: Component " << (comp+1)
                            << " parameter " << param_name
                            << " at " << (at_lower ? "lower" : "upper")
                            << " grid boundary (" << x[i] << ")\n";
            }
        }
    }
    
    last_free_mask_ = free_mask;

    if (add_powell){
        /* ---- e)   refine with Powell's method ----------------------------- */
        PowellSolverOptions powell_opt;
        //powell_opt.max_iterations     = std::max(50, max_iterations / 4);  // Fewer iterations for Powell
        //powell_opt.max_function_evals = max_iterations * 2;
        powell_opt.relative_tolerance = 1e-5;
        powell_opt.absolute_tolerance = 1e-10;
        powell_opt.verbose            = config_.verbose;


        PowellSolverSummary powell_summary = powell(
                [&cost](const Eigen::VectorXd& p,
                        Eigen::VectorXd*       r,
                        Eigen::MatrixXd*       J)
                { cost(p, r, J); },
                x, free_mask, lo, hi, powell_opt);


        /* ---- f)   update summary with combined results -------------------- */
        // Keep LM's parameter uncertainties, but update chi2 if Powell improved
        if (powell_summary.final_value < summary_.final_chi2) {
            summary_.final_chi2 = powell_summary.final_value;
            summary_.converged = summary_.converged || powell_summary.converged;
            summary_.iterations += powell_summary.iterations;
        }
    }

    Eigen::Map<Eigen::VectorXd>(unified_params_.data(), Npar) = x;
    
    if (config_.on_stage_complete)
        config_.on_stage_complete(dbg_stage_counter, *this);
    ++dbg_stage_counter;
}

/* ------------------------------------------------------------------------- */
/*  small wrappers for the six stages                                        */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::stage1_continuum_only() {
    /*  ISIS's "First guess for the continuum" freezes only `stellar(1).*`
     *  and fits everything else that is free -- which includes the telluric
     *  parameters (spectroscopy_automated.sl ~972).                        */
    solve_stage( { "continuum", "telluric" }, 100 );
}

void UnifiedFitWorkflow::stage2_continuum_vrad() {
    std::set<std::string> fp = { "continuum" };
    for (std::size_t c = 0; c < model_.params.size(); ++c)
        fp.insert("c"+std::to_string(c+1)+"_vrad");

    solve_stage(fp, 100);
}


void UnifiedFitWorkflow::stage3_continuum_vrad_teff_logg_z() {
    std::set<std::string> fp = { "continuum" };
    for (std::size_t c = 0; c < model_.params.size(); ++c) {
        fp.insert("c"+std::to_string(c+1)+"_vrad");
        fp.insert("c"+std::to_string(c+1)+"_teff");
        fp.insert("c"+std::to_string(c+1)+"_logg");
        fp.insert("c"+std::to_string(c+1)+"_z");
    }
    solve_stage(fp, 150);
}


void UnifiedFitWorkflow::stage4_full(bool add_powell) {
    std::set<std::string> fp = { "all", "continuum" };
    solve_stage(fp, 200, add_powell);
}


void UnifiedFitWorkflow::quick_refit(int max_iterations)
{
    // Continuum is freshly re-seeded for the jittered anchors; settle it at the
    // (warm-started) stellar values, then do one joint continuum+stellar solve.
    stage1_continuum_only();
    solve_stage({ "all", "continuum" }, max_iterations, /*add_powell=*/false);

    // propagate to model.params so callers reading the model see the refit
    sync_model_params();
}

double UnifiedFitWorkflow::chi2_current() const
{
    double chi2 = 0.0;

    for (std::size_t d = 0; d < datasets_.size(); ++d)
    {
        Vector model      = get_model_for_dataset(d);
        Vector residuals  = (model - datasets_[d].obs.flux)
                            .cwiseQuotient(datasets_[d].obs.sigma);
        chi2 += residuals.dot(residuals);  // dot product with itself
    }
    return chi2;
}

void UnifiedFitWorkflow::stage5_auto_freeze_vsini() 
{
    // Calculate vsini threshold based on spectral resolution
    double vsini_thres = 50.0;  // Initial maximum threshold
    
    for (const auto& ds : datasets_) {
        // Calculate median wavelength
        Vector lambda_sorted = ds.obs.lambda;
        std::sort(lambda_sorted.data(), lambda_sorted.data() + lambda_sorted.size());
        double wave_cen = lambda_sorted[lambda_sorted.size() / 2];
        
        // Calculate resolution at center wavelength
        double res_cen = ds.resOffset + wave_cen * ds.resSlope;
        
        // Calculate vsini threshold for this dataset (c/R/15)
        double dataset_thres = 2.99792458e+05 / res_cen / 15.0;
        vsini_thres = std::min(vsini_thres, dataset_thres);
    }
    
    // Enforce minimum threshold of 0.5 km/s
    vsini_thres = std::max(vsini_thres, 0.5);
    std::cout << "vsini_thres = " << vsini_thres << std::endl;
    
    // Check if any component has a *free* vsini below threshold.  ISIS only
    // ever touches free parameters here (its `free = freeParameters` list), so
    // a vsini the user deliberately froze must be left at its value.
    std::vector<int> to_freeze;
    const int n_components = static_cast<int>(model_.params.size());

    for (int c = 0; c < n_components; ++c) {
        if (frozen_status_[c].at("vsini")) continue;
        if (unified_params_[indexer_.index_of(c, 0, "vsini")] < vsini_thres)
            to_freeze.push_back(c);
    }

    if (!to_freeze.empty()) {
        std::cout << "[Stage 5] Freezing vsini = 0 km/s (below threshold of "
                  << std::fixed << std::setprecision(3) << vsini_thres
                  << " km/s)\n";

        for (int c : to_freeze) {
            frozen_status_[c]["vsini"] = true;

            // Set vsini to 0 for all datasets
            for (size_t d = 0; d < datasets_.size(); ++d) {
                unified_params_[indexer_.index_of(c, static_cast<int>(d),
                                                  "vsini")] = 0.0;
            }
        }

        // Run a fit with vsini frozen
        std::set<std::string> fp = { "all", "continuum" };
        solve_stage(fp, 100);
    }
}


/* ------------------------------------------------------------------------- *
 *  Retire a secondary that the data cannot see.
 *
 *  ISIS's auto_freeze_sur_ratio (spectroscopy_automated.sl ~1157-1180): after
 *  the first free fit it measures each component's peak share of the
 *  composite,
 *
 *      contribution_k = max_lambda  s_k F_k(lambda) / sum_j s_j C_j(lambda)
 *
 *  and, if the fitted surface ratio came out below `sur_ratio_thres` or that
 *  share stayed below `c2_detection_thres`, sets the surface ratio to zero
 *  and freezes the whole component ("Freezing c2 contribution to zero").
 *
 *  It matters that the *whole* component is frozen, not just its surface
 *  ratio: at s_k = 0 that component's spectrum drops out of the residuals
 *  entirely, so every one of its other parameters has an identically-zero
 *  Jacobian column and JtJ is singular in eight directions at once.
 *
 *  Unlike ISIS this is off unless the config asks for it -- see
 *  Config::auto_freeze_sur_ratio.
 * ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::stage5b_auto_freeze_sur_ratio()
{
    const int n_components = static_cast<int>(model_.params.size());
    if (!config_.auto_freeze_sur_ratio || n_components < 2) return;

    /* ---- peak contribution of every component, over all spectra -------- */
    std::vector<double> contribution(n_components, 0.0);

    for (std::size_t d = 0; d < datasets_.size(); ++d) {
        const auto& ds  = datasets_[d];
        const int   np  = static_cast<int>(ds.obs.lambda.size());
        const int   did = static_cast<int>(d);

        std::vector<Vector> flux(n_components);
        Vector den = Vector::Zero(np);

        for (int c = 0; c < n_components; ++c) {
            const StellarParams sp =
                stellar_params_from(indexer_, unified_params_, c, did);

            Spectrum syn = compute_synthetic(model_.grids[c], sp, ds.obs.lambda,
                                             ds.resOffset, ds.resSlope,
                                             /*with_continuum=*/true);
            flux[c] = sp.sur_ratio * syn.flux;
            den    += sp.sur_ratio * syn.cont;
        }

        for (int c = 0; c < n_components; ++c)
            for (int i = 0; i < np; ++i) {
                if (!ds.obs.ignoreflag[i] || den[i] <= 0.0) continue;
                contribution[c] = std::max(contribution[c], flux[c][i] / den[i]);
            }
    }

    std::cout << "[Stage 5b] Max. contributions:";
    for (int c = 0; c < n_components; ++c)
        std::cout << " c" << (c + 1) << '=' << std::fixed << std::setprecision(3)
                  << contribution[c];
    std::cout << " (thres=" << config_.c2_detection_thres << ").\n";

    /* ---- retire the undetected ones ------------------------------------ */
    bool froze_any = false;

    for (int c = 1; c < n_components; ++c) {
        if (frozen_status_[c].at("sur_ratio")) continue;   // user already fixed it
        const int sr_idx = indexer_.index_of(c, 0, "sur_ratio");
        const double sr  = unified_params_[sr_idx];
        if (!(sr < config_.sur_ratio_thres ||
              contribution[c] <= config_.c2_detection_thres)) continue;

        std::cout << "[Stage 5b] Freezing c" << (c + 1)
                  << " contribution to zero.\n";
        for (std::size_t d = 0; d < datasets_.size(); ++d)
            unified_params_[indexer_.index_of(c, static_cast<int>(d),
                                              "sur_ratio")] = 0.0;
        /*  The *whole* component is frozen, not just its surface ratio: at
         *  s_k = 0 it drops out of the residuals, so every one of its
         *  parameters would have an identically-zero Jacobian column.       */
        for (const auto& ps : indexer_.params(c)) frozen_status_[c][ps.name] = true;
        froze_any = true;
    }

    if (froze_any)
        solve_stage({ "all", "continuum" }, 100);
}

/* ------------------------------------------------------------------------- */
/*  Stage-4 : iterative noise re-scaling  +  outlier rejection (fast IRLS)   */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::stage6_rescale_and_reject()
{
    const auto &P      = config_;
    const int   NDS    = static_cast<int>(datasets_.size());

    /* ---------- store pristine σ (per workflow, not per process) -------- */
    std::vector<ArrayXd> sigma0;
    sigma0.reserve(NDS);
    for (const auto &ds : datasets_)
        sigma0.emplace_back(ds.obs.sigma);

    /* ---------- scratch arrays per data set ---------------------------- */
    std::vector<ArrayXd> chi_arr (NDS);
    std::vector<ArrayXd> scl_arr (NDS);

    bool  weights_changed = true;
    int   pass            = 0;

    /* ======================   outer IRLS loop   ======================== */
    while (weights_changed && pass < P.nit_noise_max)
    {
        weights_changed = false;
        ++pass;

        /* =========   data set loop  (parallel if requested)   ========= */
        #pragma omp parallel for schedule(dynamic) if(nthreads_>1)
        for (int d = 0; d < NDS; ++d)
        {
            auto &ds   = datasets_[static_cast<std::size_t>(d)];
            const int nbin = ds.obs.flux.size();

            /* ---- make sure scratch buffers have correct size ---------- */
            if (chi_arr[d].size() != nbin) {
                chi_arr[d].resize(nbin);
                scl_arr[d].resize(nbin);
            }

            /* ---- reset σ to pristine values --------------------------- */
            ds.obs.sigma = sigma0[d];

            /* ---- model counts ----------------------------------------- */
            ArrayXd model = get_model_for_dataset(static_cast<std::size_t>(d));

            /* ---- χ array (ignored bins keep 0) ------------------------ */
            ArrayXd &chi = chi_arr[d];
            chi.setZero();
            for (int i = 0; i < nbin; ++i)
                if (ds.obs.ignoreflag[i] && ds.obs.sigma[i] > 0.0)
                    chi[i] = (ds.obs.flux[i] - model[i]) / ds.obs.sigma[i];

            /* ---- local scale factors ---------------------------------- */
            ArrayXd &scale = scl_arr[d];
            scale.setOnes();

            const int w  = P.width_box_px;
            const int ws = 2 * w;               /* smoothing half-width   */

            std::vector<double> neigh;
            neigh.reserve(2 * w);

            bool local_changed = false;

            for (int i = 0; i < nbin; ++i)
            if (ds.obs.ignoreflag[i])
            {
                /* ---- neighbourhood (excluding i) --------------------- */
                neigh.clear();
                const int lo = std::max(0, i - w);
                const int hi = std::min(nbin - 1, i + w);

                for (int j = lo; j <= hi; ++j)
                    if (j != i && ds.obs.ignoreflag[j])
                        neigh.push_back(chi[j]);

                if (neigh.size() < 2) continue;

                ArrayXd neighArr = Eigen::Map<ArrayXd>(neigh.data(),
                                                       neigh.size());
                const double mean  = neighArr.mean();
                const double sdev  = std::sqrt(
                        (neighArr - mean).square().sum() /
                        std::max<int>(neighArr.size() - 1, 1));

                scale[i] = sdev;                        /* local σ */

                /* ---- self-outlier test from 2nd pass on -------------- */
                if (pass > 1) {
                    const double delt = chi[i] - mean;
                    if (delt < -P.outlier_sigma_lo * sdev ||
                        delt >  P.outlier_sigma_hi * sdev)
                    {
                        ds.obs.ignoreflag[i] = 0;
                        local_changed        = true;
                    }
                }
            }

            /* ---- smooth scale with simple box filter ----------------- */
            ArrayXd tmp = scale;
            for (int i = 0; i < nbin; ++i)
            if (ds.obs.ignoreflag[i])
            {
                const int lo = std::max(0, i - ws);
                const int hi = std::min(nbin - 1, i + ws);
                tmp[i] = scale.segment(lo, hi - lo + 1).mean();
            }
            scale.swap(tmp);

            /* ---- convergence test ------------------------------------ */
            const int tot = std::count_if(ds.obs.ignoreflag.begin(),
                                          ds.obs.ignoreflag.end(),
                                          [](int v){return v!=0;});
            if (tot > 0)
            {
                int good = 0;
                for (int i = 0; i < nbin; ++i)
                    if (ds.obs.ignoreflag[i] &&
                        scale[i] > P.conv_range_lo &&
                        scale[i] < P.conv_range_hi)
                        ++good;

                if (good < P.conv_fraction * tot)
                    local_changed = true;
            }

            /* ---- apply the scaling ----------------------------------- */
            for (int i = 0; i < nbin; ++i)
                ds.obs.sigma[i] *= scale[i];

            /* ---- any change in this spectrum propagates globally ----- */
            if (local_changed)
                weights_changed = true;
        }  /* -------- end data-set loop -------------------------------- */

        /* -------- one warm-started LM step with new weights ----------- */
        std::set<std::string> fp = { "all", "continuum" };
        solve_stage(fp, 3);                          /* ≤ 3 LM iteration */
    }  /* =====================   end outer loop   ======================= */

    if (config_.verbose)
        std::cout << "[IterNoise] passes: " << pass << '\n';
}

void UnifiedFitWorkflow::stage7_final() {
    const int n_components = static_cast<int>(model_.params.size());
    const double boundary_tol = 1e-6;

    const ModelGrid::ParameterBounds gb = grid_bounds();

    // Do initial full fit
    std::cout << "[Stage 7] Final fit ...\n";
    stage4_full();

    // Check for boundary parameters
    bool any_at_boundary = false;
    std::vector<std::pair<int, std::string>> boundary_params;  // comp, name

    for (int c = 0; c < n_components; ++c) {
        const auto& table = indexer_.params(c);
        for (std::size_t p = 0; p < table.size(); ++p) {
            const auto& ps = table[p];

            /*  sur_ratio is deliberately excluded.  Freezing it alone at its
             *  lower bound of 0 would leave the rest of that component free
             *  while contributing nothing, i.e. a full set of exactly-zero
             *  Jacobian columns.  A vanishing component is retired as a whole,
             *  by stage5b_auto_freeze_sur_ratio.                             */
            if (ps.kind == ParamKind::SurRatio) continue;

            // Skip if already frozen
            if (frozen_status_[c].at(ps.name)) continue;

            const int idx = indexer_.get(c, 0, static_cast<int>(p)); // dataset 0
            const double val = unified_params_[idx];
            const auto lim = param_limits(ps, c, gb);
            const double lo = lim.first;
            const double hi = lim.second;

            // Check if at boundary
            bool at_lower = (val - lo) < boundary_tol * std::abs(lo + 1.0);
            bool at_upper = (hi - val) < boundary_tol * std::abs(hi + 1.0);

            if (at_lower || at_upper) {
                std::cout << "  Warning: Component " << (c+1)
                          << " parameter " << ps.name
                          << " at " << (at_lower ? "lower" : "upper")
                          << " grid boundary (" << val << ")\n";
                any_at_boundary = true;
                boundary_params.emplace_back(c, ps.name);
            }
        }
    }
    
    // If any parameters at boundary, freeze them and refit
    if (any_at_boundary) {
        std::cout << "\n[Stage 7] Detected " << boundary_params.size() 
                  << " parameter(s) at grid boundaries.\n";
        std::cout << "[Stage 7] Freezing boundary parameters and refitting...\n";
        
        // Freeze all boundary parameters
        for (const auto& [comp, param_name] : boundary_params) {
            frozen_status_[comp][param_name] = true;
            std::cout << "  Frozen: Component " << (comp+1) 
                      << " " << param_name << "\n";
        }
        
        stage4_full();  // with Powell refinement
    }
}    


void UnifiedFitWorkflow::report_boundary_parameters() const {
    // Get bounds from grids
    ModelGrid::ParameterBounds grid_bounds;
    if (!model_.grids.empty()) {
        grid_bounds = model_.grids[0].get_parameter_bounds();
    }
    
    const double tol = 1e-4;
    bool any_at_boundary = false;
    
    std::cout << "\n=== Boundary Check ===\n";
    
    for (size_t c = 0; c < model_.params.size(); ++c) {
        const auto& sp = model_.params[c];
        
        if (std::abs(sp.teff - grid_bounds.teff_min) < tol * grid_bounds.teff_min ||
            std::abs(sp.teff - grid_bounds.teff_max) < tol * grid_bounds.teff_max) {
            std::cout << "Component " << (c+1) << " Teff at grid boundary: " 
                     << sp.teff << " K\n";
            any_at_boundary = true;
        }
        
        if (std::abs(sp.logg - grid_bounds.logg_min) < tol ||
            std::abs(sp.logg - grid_bounds.logg_max) < tol) {
            std::cout << "Component " << (c+1) << " log g at grid boundary: " 
                     << sp.logg << "\n";
            any_at_boundary = true;
        }
        
        if (std::abs(sp.z - grid_bounds.z_min) < tol ||
            std::abs(sp.z - grid_bounds.z_max) < tol) {
            std::cout << "Component " << (c+1) << " [M/H] at grid boundary: " 
                     << sp.z << "\n";
            any_at_boundary = true;
        }
        
        // Add checks for other parameters as needed
    }
    
    if (any_at_boundary) {
        std::cout << "\n Warning: One or more parameters are at grid boundaries.\n"
                  << "   Consider whether the solution is physical or if a larger grid is needed.\n";
    }
}


/* ------------------------------------------------------------------------- */
/*  public “run” orchestrator                                                */
/* ------------------------------------------------------------------------- */
void UnifiedFitWorkflow::run()
{
    const std::string variant = stage_variant();
    const bool timing = stage_timing();

    const bool do_1  = variant != "direct";
    const bool do_23 = variant == "full";
    const bool do_6  = variant != "nonoise";

    auto t_start = std::chrono::steady_clock::now();
    auto mark = [&](const char* name) {
        if (!timing) return;
        const double t = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t_start).count();
        std::cout << "[stage-timing] " << name << " cum_t=" << std::fixed
                  << std::setprecision(2) << t
                  << "s chi2=" << std::setprecision(3) << summary_.final_chi2;
        for (int c = 0; c < static_cast<int>(model_.params.size()); ++c)
            std::cout << " c" << (c + 1)
                      << "_teff=" << unified_params_[indexer_.index_of(c, 0, "teff")]
                      << " c" << (c + 1)
                      << "_logg=" << unified_params_[indexer_.index_of(c, 0, "logg")]
                      << " c" << (c + 1)
                      << "_he=" << unified_params_[indexer_.index_of(c, 0, "he")];
        std::cout << '\n';
    };

    if (timing)
        std::cout << "[stage-timing] variant=" << variant << '\n';

    if (do_1) { std::cout << "[Stage 1] Continuum Fit ...\n";   stage1_continuum_only(); mark("stage1"); }
    if (do_23) {
        std::cout << "[Stage 2] Fitting Continuum + v_rad ...\n"; stage2_continuum_vrad(); mark("stage2");
        std::cout << "[Stage 3] Fitting Continuum + v_rad + T_eff + log(g) + [M/H] ...\n"; stage3_continuum_vrad_teff_logg_z(); mark("stage3");
    }
    std::cout << "[Stage 4] First Full Fit ...\n"; stage4_full(); mark("stage4");
    std::cout << "[Stage 5] Auto-freeze vsini if unmeasurable ...\n"; stage5_auto_freeze_vsini(); mark("stage5");
    if (config_.auto_freeze_sur_ratio && model_.params.size() > 1) {
        std::cout << "[Stage 5b] Auto-freeze undetected components ...\n";
        stage5b_auto_freeze_sur_ratio(); mark("stage5b");
    }
    if (do_6) { std::cout << "[Stage 6] Iterative Noise Rescaling and Outlier Rejection ...\n"; stage6_rescale_and_reject(); mark("stage6"); }
    std::cout << "[Stage 7] Final Fit ...\n"; stage7_final(); mark("stage7");
    final_uncertainties_ = summary_.param_uncertainties;

    /* update model structure with the final parameter values */
    sync_model_params();
}


Vector UnifiedFitWorkflow::get_model_for_dataset(size_t dataset_idx) const {
    return model_for_dataset(dataset_idx, -1);
}

/*  One component's own model: the same continuum and the same telluric
 *  transmission, but only component `only_component`'s stellar flux -- what
 *  the spectrum would look like if that component were the only star in it.
 *  The surface ratio cancels out of a single component's num/den, so this is
 *  the component's line profile at full depth rather than its diluted
 *  contribution to the composite.                                           */
Vector UnifiedFitWorkflow::get_component_model_for_dataset(size_t dataset_idx,
                                                           int component) const
{
    if (component < 0 || component >= static_cast<int>(model_.grids.size()))
        throw std::out_of_range("Invalid component index");
    return model_for_dataset(dataset_idx, component);
}

Vector UnifiedFitWorkflow::model_for_dataset(size_t dataset_idx,
                                             int only_component) const {
    if (dataset_idx >= datasets_.size()) {
        throw std::out_of_range("Invalid dataset index");
    }

    const auto& ds = datasets_[dataset_idx];
    const int n_points = ds.obs.lambda.size();
    
    // Extract current stellar parameters (dataset-specific)
    std::vector<StellarParams> stellar(model_.params.size());
    const int didx = static_cast<int>(dataset_idx);
    for (int c = 0; c < static_cast<int>(model_.params.size()); ++c)
        stellar[c] = stellar_params_from(indexer_, unified_params_, c, didx);
    
    /* -------- continuum anchors live in ONE big block at the very end --- *
     *  (the telluric block sits between them and the stellar parameters, so
     *   counting back from the end is still the way to find them)          */
    int total_cont = 0;
    for (const auto& dsi : datasets_) total_cont += dsi.cont_y.size();
    
    const int cont_block_start = static_cast<int>(unified_params_.size()) - total_cont;
    
    int cont_offset = 0;
    for (std::size_t j = 0; j < dataset_idx; ++j)
        cont_offset += datasets_[j].cont_y.size();
    
    Vector cont_y = Eigen::Map<const Vector>(
        unified_params_.data() + cont_block_start + cont_offset,
        ds.cont_y.size());
    
    // Build continuum
    AkimaSpline cont_spline(ds.cont_x, cont_y);
    Vector continuum = cont_spline(ds.obs.lambda);
    
    // Compute the normalised synthetic spectrum -- see
    // MultiDatasetCost::synth_of_dataset for why a multi-component model is
    // built from calibrated fluxes and summed continua rather than averaged.
    const bool composite = model_.grids.size() > 1 && only_component < 0;

    Vector model = Vector::Zero(n_points);

    if (!composite) {
        /*  A single-component fit, or one component of a composite one.  The
         *  latter still needs its own continuum divided out, because a
         *  component of a composite model is loaded with_continuum.          */
        const int c = (only_component < 0) ? 0 : only_component;
        if (model_.grids.size() > 1) {
            Spectrum synth = compute_synthetic(
                model_.grids[c], stellar[c], ds.obs.lambda,
                ds.resOffset, ds.resSlope, /*with_continuum=*/true);
            for (int i = 0; i < n_points; ++i)
                model[i] = (synth.cont[i] > 0.0) ? synth.flux[i] / synth.cont[i]
                                                 : 0.0;
        } else {
            model = compute_synthetic(model_.grids[c], stellar[c], ds.obs.lambda,
                                      ds.resOffset, ds.resSlope).flux;
        }
    } else {
        Vector num = Vector::Zero(n_points);
        Vector den = Vector::Zero(n_points);
        for (size_t c = 0; c < model_.grids.size(); ++c) {
            Spectrum synth = compute_synthetic(
                model_.grids[c], stellar[c], ds.obs.lambda,
                ds.resOffset, ds.resSlope, /*with_continuum=*/true);
            num += stellar[c].sur_ratio * synth.flux;
            den += stellar[c].sur_ratio * synth.cont;
        }
        for (int i = 0; i < n_points; ++i)
            model[i] = (den[i] > 0.0) ? num[i] / den[i] : 0.0;
    }

    // Apply continuum
    model = model.cwiseProduct(continuum);

    /*  ... and the atmosphere, when this spectrum fits one: ISIS's model is
     *  cspline * stellar * telluric.                                       */
    const int tb = telluric_param_offset(dataset_idx);
    if (tb >= 0 && model_.telluric) {
        const Vector tr = model_.telluric->transmission(
            ds.obs.lambda,
            unified_params_[tb + static_cast<int>(TelluricParam::Airmass)],
            unified_params_[tb + static_cast<int>(TelluricParam::Pwv)],
            unified_params_[tb + static_cast<int>(TelluricParam::Barycorr)],
            ds.resOffset, ds.resSlope);
        model = model.cwiseProduct(tr);
    }
    return model;
}

/* ------------------------------------------------------------------------- *
 *  Solver limits for every entry of the global parameter vector.  Only the
 *  stellar block has any: the telluric and continuum entries are fit policy
 *  with no grid behind them and report +-inf, so a consumer checking whether a
 *  value sits on a boundary never flags one of those.
 * ------------------------------------------------------------------------- */
std::vector<std::pair<double,double>>
UnifiedFitWorkflow::get_param_limits() const
{
    constexpr double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::pair<double,double>> lim(unified_params_.size(),
                                              { -kInf, kInf });

    const ModelGrid::ParameterBounds gb = grid_bounds();
    const int n_ds = static_cast<int>(datasets_.size());

    for (int c = 0; c < static_cast<int>(model_.params.size()); ++c) {
        const auto& table = indexer_.params(c);
        for (std::size_t p = 0; p < table.size(); ++p) {
            const auto lp = param_limits(table[p], c, gb);
            for (int d = 0; d < n_ds; ++d) {
                const int gidx = indexer_.get(c, d, static_cast<int>(p));
                if (gidx >= 0 && gidx < static_cast<int>(lim.size()))
                    lim[static_cast<std::size_t>(gidx)] = lp;
            }
        }
    }
    return lim;
}

} // namespace specfit