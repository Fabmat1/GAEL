
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
 *    GAEL_STAGES=<variant> -> run a reduced stage ladder:
 *        full     (default) 1,2,3,4,5,6,7
 *        noladder           1,    4,5,6,7   (drop the vrad / teff-logg-z steps)
 *        direct                  4,5,6,7   (no continuum-only pre-fit either)
 *        nonoise            1,2,3,4,5,  7   (drop iterative noise rescaling)
 * ------------------------------------------------------------------------- */
namespace {
std::string stage_variant()
{
    const char* v = std::getenv("GAEL_STAGES");
    return v ? std::string(v) : std::string("full");
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
    
    indexer_.build(n_components, n_datasets, config_.untie_params);
    
    /* ----------  collect initial parameters into one big vector -------- */
    unified_params_.resize(indexer_.total_stellar_params);
    for (int c = 0; c < n_components; ++c) {
        const auto& sp = model_.params[c];
        for (int d = 0; d < n_datasets; ++d) {
            unified_params_[ indexer_.get(c,d,0) ] = sp.vrad ;
            unified_params_[ indexer_.get(c,d,1) ] = sp.vsini;
            unified_params_[ indexer_.get(c,d,2) ] = sp.zeta ;
            unified_params_[ indexer_.get(c,d,3) ] = sp.teff ;
            unified_params_[ indexer_.get(c,d,4) ] = sp.logg ;
            unified_params_[ indexer_.get(c,d,5) ] = sp.xi   ;
            unified_params_[ indexer_.get(c,d,6) ] = sp.z    ;
            unified_params_[ indexer_.get(c,d,7) ] = sp.he   ;
        }
    }
    for (const auto& ds : datasets_)
        unified_params_.insert(unified_params_.end(),
                               ds.cont_y.begin(), ds.cont_y.end());
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

    const char* names[8] = {"vrad","vsini","zeta","teff",
        "logg","xi","z","he"};

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

        ds_infos.push_back(std::move(info));
        const int n_kept = std::accumulate(ds.obs.ignoreflag.begin(),
                               ds.obs.ignoreflag.end(), 0);
        total_residuals += n_kept;
        cont_offset     += ds.cont_y.size();
    }
    for (auto& g : model_.grids) grid_ptrs.push_back(&g);

    MultiDatasetCost cost(ds_infos, grid_ptrs, n_components,
                              indexer_,
                               total_residuals, cont_offset);

    const int Npar = stellar_total + cont_offset;

    /* ---- b)   build lower / upper bounds ------------------------------ */
    std::vector<double> lo(Npar, -1.0e10);
    std::vector<double> hi(Npar,  1.0e10);
    
    // Get bounds from the first grid (assuming all grids have same bounds)
    // If grids have different bounds, take the intersection
    ModelGrid::ParameterBounds grid_bounds;
    bool first_grid = true;
    
    for (auto& grid : model_.grids) {
        auto bounds = grid.get_parameter_bounds();
        if (first_grid) {
            grid_bounds = bounds;
            first_grid = false;
        } else {
            // Take intersection of bounds (most restrictive)
            grid_bounds.teff_min = std::max(grid_bounds.teff_min, bounds.teff_min);
            grid_bounds.teff_max = std::min(grid_bounds.teff_max, bounds.teff_max);
            grid_bounds.logg_min = std::max(grid_bounds.logg_min, bounds.logg_min);
            grid_bounds.logg_max = std::min(grid_bounds.logg_max, bounds.logg_max);
            grid_bounds.z_min = std::max(grid_bounds.z_min, bounds.z_min);
            grid_bounds.z_max = std::min(grid_bounds.z_max, bounds.z_max);
            grid_bounds.he_min = std::max(grid_bounds.he_min, bounds.he_min);
            grid_bounds.he_max = std::min(grid_bounds.he_max, bounds.he_max);
            grid_bounds.xi_min = std::max(grid_bounds.xi_min, bounds.xi_min);
            grid_bounds.xi_max = std::min(grid_bounds.xi_max, bounds.xi_max);
        }
    }
    
    // Apply grid bounds to parameter vectors
    for (int c = 0; c < n_components; ++c) {
        for (std::size_t d = 0; d < datasets_.size(); ++d) {
            // vrad
            int idx = indexer_.get(c, static_cast<int>(d), 0);
            lo[idx] = -1000.0; hi[idx] = 1000.0;
            
            // vsini
            idx = indexer_.get(c, static_cast<int>(d), 1);
            lo[idx] = 0.0; hi[idx] = 500.0;
            
            // zeta (no specific bounds)
            
            // teff
            idx = indexer_.get(c, static_cast<int>(d), 3);
            lo[idx] = grid_bounds.teff_min;
            hi[idx] = grid_bounds.teff_max;
            
            // logg
            idx = indexer_.get(c, static_cast<int>(d), 4);
            lo[idx] = grid_bounds.logg_min;
            hi[idx] = grid_bounds.logg_max;
            
            // xi
            idx = indexer_.get(c, static_cast<int>(d), 5);
            if (grid_bounds.xi_min > -1e9) {  // If grid has xi bounds
                lo[idx] = grid_bounds.xi_min;
                hi[idx] = grid_bounds.xi_max;
            }
            
            // z (metallicity)
            idx = indexer_.get(c, static_cast<int>(d), 6);
            lo[idx] = grid_bounds.z_min;
            hi[idx] = grid_bounds.z_max;
            
            // he
            idx = indexer_.get(c, static_cast<int>(d), 7);
            lo[idx] = grid_bounds.he_min;
            hi[idx] = grid_bounds.he_max;
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
        for (std::size_t d = 0; d < datasets_.size(); ++d)
            for (int p = 0; p < 8; ++p)
            {
                if (!(all_requested ||
                      free_params.count(token(names[p])))) continue;
                if (frz.at(names[p])) continue;
                mark_free(indexer_.get(c,static_cast<int>(d),p));
            }
    }

    // continuum anchors
    if (free_params.count("continuum"))
        for (int i = stellar_total; i < Npar; ++i)
            mark_free(i);

    /* ---- d)   run Levenberg–Marquardt -------------------------------- */
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(unified_params_.data(), Npar);
    
    // Ensure initial parameters are within bounds
    for (int i = 0; i < Npar; ++i) {
        x[i] = std::clamp(x[i], lo[i], hi[i]);
    }
    
    LMSolverOptions lm_opt;
    lm_opt.max_iterations = max_iterations;
    lm_opt.verbose        = false;
    
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
            
            int comp = -1, dataset = -1, param_type = -1;
            // Decode which parameter this is
            for (int c = 0; c < n_components && comp < 0; ++c) {
                for (int d = 0; d < static_cast<int>(datasets_.size()); ++d) {
                    for (int p = 0; p < 8; ++p) {
                        if (indexer_.get(c, d, p) == i) {
                            comp = c; dataset = d; param_type = p;
                            break;
                        }
                    }
                    if (comp >= 0) break;
                }
            }
            
            if (comp >= 0) {
                std::cout << "  Warning: Component " << (comp+1) 
                            << " parameter " << names[param_type]
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
    solve_stage( { "continuum" }, 100 );
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
    for (std::size_t c = 0; c < model_.params.size(); ++c) {
        model_.params[c].vrad  = unified_params_[ indexer_.get(c,0,0) ];
        model_.params[c].vsini = unified_params_[ indexer_.get(c,0,1) ];
        model_.params[c].zeta  = unified_params_[ indexer_.get(c,0,2) ];
        model_.params[c].teff  = unified_params_[ indexer_.get(c,0,3) ];
        model_.params[c].logg  = unified_params_[ indexer_.get(c,0,4) ];
        model_.params[c].xi    = unified_params_[ indexer_.get(c,0,5) ];
        model_.params[c].z     = unified_params_[ indexer_.get(c,0,6) ];
        model_.params[c].he    = unified_params_[ indexer_.get(c,0,7) ];
    }
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
        if (unified_params_[indexer_.get(c, 0, 1)] < vsini_thres)
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
                unified_params_[indexer_.get(c, static_cast<int>(d), 1)] = 0.0;
            }
        }

        // Run a fit with vsini frozen
        std::set<std::string> fp = { "all", "continuum" };
        solve_stage(fp, 100);
    }
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
    const char* names[8] = {"vrad","vsini","zeta","teff","logg","xi","z","he"};
    const double boundary_tol = 1e-6;
    
    // Get grid bounds
    ModelGrid::ParameterBounds grid_bounds;
    bool first_grid = true;
    for (auto& grid : model_.grids) {
        auto bounds = grid.get_parameter_bounds();
        if (first_grid) {
            grid_bounds = bounds;
            first_grid = false;
        } else {
            grid_bounds.teff_min = std::max(grid_bounds.teff_min, bounds.teff_min);
            grid_bounds.teff_max = std::min(grid_bounds.teff_max, bounds.teff_max);
            grid_bounds.logg_min = std::max(grid_bounds.logg_min, bounds.logg_min);
            grid_bounds.logg_max = std::min(grid_bounds.logg_max, bounds.logg_max);
            grid_bounds.z_min = std::max(grid_bounds.z_min, bounds.z_min);
            grid_bounds.z_max = std::min(grid_bounds.z_max, bounds.z_max);
            grid_bounds.he_min = std::max(grid_bounds.he_min, bounds.he_min);
            grid_bounds.he_max = std::min(grid_bounds.he_max, bounds.he_max);
            grid_bounds.xi_min = std::max(grid_bounds.xi_min, bounds.xi_min);
            grid_bounds.xi_max = std::min(grid_bounds.xi_max, bounds.xi_max);
        }
    }
    
    // Do initial full fit
    std::cout << "[Stage 7] Final fit ...\n";
    stage4_full(); 
    
    // Check for boundary parameters
    bool any_at_boundary = false;
    std::vector<std::tuple<int, int, std::string>> boundary_params;  // comp, param_idx, name
    
    for (int c = 0; c < n_components; ++c) {
        // Get bounds for each parameter type
        std::vector<std::pair<double, double>> param_bounds = {
            {-1000.0, 1000.0},           // vrad
            {0.0, 500.0},                // vsini
            {-1e10, 1e10},               // zeta (no bounds)
            {grid_bounds.teff_min, grid_bounds.teff_max},  // teff
            {grid_bounds.logg_min, grid_bounds.logg_max},  // logg
            {grid_bounds.xi_min, grid_bounds.xi_max},      // xi
            {grid_bounds.z_min, grid_bounds.z_max},        // z
            {grid_bounds.he_min, grid_bounds.he_max}       // he
        };
        
        for (int p = 0; p < 8; ++p) {
            // Skip if already frozen
            if (frozen_status_[c].at(names[p])) continue;
            
            int idx = indexer_.get(c, 0, p);  // Check first dataset
            double val = unified_params_[idx];
            double lo = param_bounds[p].first;
            double hi = param_bounds[p].second;
            
            // Check if at boundary
            bool at_lower = (val - lo) < boundary_tol * std::abs(lo + 1.0);
            bool at_upper = (hi - val) < boundary_tol * std::abs(hi + 1.0);
            
            if (at_lower || at_upper) {
                std::cout << "  Warning: Component " << (c+1) 
                          << " parameter " << names[p]
                          << " at " << (at_lower ? "lower" : "upper")
                          << " grid boundary (" << val << ")\n";
                any_at_boundary = true;
                boundary_params.push_back(std::make_tuple(c, p, std::string(names[p])));
            }
        }
    }
    
    // If any parameters at boundary, freeze them and refit
    if (any_at_boundary) {
        std::cout << "\n[Stage 7] Detected " << boundary_params.size() 
                  << " parameter(s) at grid boundaries.\n";
        std::cout << "[Stage 7] Freezing boundary parameters and refitting...\n";
        
        // Freeze all boundary parameters
        for (const auto& param_info : boundary_params) {
            int comp = std::get<0>(param_info);
            std::string param_name = std::get<2>(param_info);
            
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

    const bool do_1 = variant != "direct";
    const bool do_23 = (variant == "full" || variant == "nonoise");
    const bool do_6 = variant != "nonoise";

    auto t_start = std::chrono::steady_clock::now();
    auto mark = [&](const char* name) {
        if (!timing) return;
        const double t = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t_start).count();
        std::cout << "[stage-timing] " << name << " cum_t=" << std::fixed
                  << std::setprecision(2) << t
                  << "s chi2=" << std::setprecision(3) << summary_.final_chi2;
        for (std::size_t c = 0; c < model_.params.size(); ++c)
            std::cout << " c" << (c + 1)
                      << "_teff=" << unified_params_[indexer_.get(c, 0, 3)]
                      << " c" << (c + 1)
                      << "_logg=" << unified_params_[indexer_.get(c, 0, 4)]
                      << " c" << (c + 1)
                      << "_he=" << unified_params_[indexer_.get(c, 0, 7)];
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
    if (do_6) { std::cout << "[Stage 6] Iterative Noise Rescaling and Outlier Rejection ...\n"; stage6_rescale_and_reject(); mark("stage6"); }
    std::cout << "[Stage 7] Final Fit ...\n"; stage7_final(); mark("stage7");
    final_uncertainties_ = summary_.param_uncertainties;
    
    /* update model structure with the final parameter values */
    for (std::size_t c = 0; c < model_.params.size(); ++c) {
        model_.params[c].vrad  = unified_params_[ indexer_.get(c,0,0) ];
        model_.params[c].vsini = unified_params_[ indexer_.get(c,0,1) ];
        model_.params[c].zeta  = unified_params_[ indexer_.get(c,0,2) ];
        model_.params[c].teff  = unified_params_[ indexer_.get(c,0,3) ];
        model_.params[c].logg  = unified_params_[ indexer_.get(c,0,4) ];
        model_.params[c].xi    = unified_params_[ indexer_.get(c,0,5) ];
        model_.params[c].z     = unified_params_[ indexer_.get(c,0,6) ];
        model_.params[c].he    = unified_params_[ indexer_.get(c,0,7) ];
    }
}


Vector UnifiedFitWorkflow::get_model_for_dataset(size_t dataset_idx) const {
    if (dataset_idx >= datasets_.size()) {
        throw std::out_of_range("Invalid dataset index");
    }
    
    const auto& ds = datasets_[dataset_idx];
    const int n_points = ds.obs.lambda.size();
    
    // Extract current stellar parameters (dataset-specific)
    std::vector<StellarParams> stellar(model_.params.size());
    const int didx = static_cast<int>(dataset_idx);
    for (int c = 0; c < static_cast<int>(model_.params.size()); ++c) {
        stellar[c].vrad  = unified_params_[ indexer_.get(c,didx,0) ];
        stellar[c].vsini = unified_params_[ indexer_.get(c,didx,1) ];
        stellar[c].zeta  = unified_params_[ indexer_.get(c,didx,2) ];
        stellar[c].teff  = unified_params_[ indexer_.get(c,didx,3) ];
        stellar[c].logg  = unified_params_[ indexer_.get(c,didx,4) ];
        stellar[c].xi    = unified_params_[ indexer_.get(c,didx,5) ];
        stellar[c].z     = unified_params_[ indexer_.get(c,didx,6) ];
        stellar[c].he    = unified_params_[ indexer_.get(c,didx,7) ];
    }
    
    /* -------- continuum anchors live in ONE big block at the very end --- */
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
    
    // Compute synthetic spectrum
    Vector model = Vector::Zero(n_points);
    double weight_sum = 0.0;
    
    for (size_t c = 0; c < model_.grids.size(); ++c) {
        Spectrum synth = compute_synthetic(
            model_.grids[c], stellar[c], ds.obs.lambda,
            ds.resOffset, ds.resSlope);
        
        double weight = std::pow(stellar[c].teff, 4);
        model += weight * synth.flux;
        weight_sum += weight;
    }
    
    if (weight_sum > 0) {
        model /= weight_sum;
    }
    
    // Apply continuum
    return model.cwiseProduct(continuum);
}

} // namespace specfit