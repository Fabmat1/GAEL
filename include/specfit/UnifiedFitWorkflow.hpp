
#pragma once
#include "Types.hpp"
#include "CommonTypes.hpp"
#include "ModelGrid.hpp"
#include "SimpleLM.hpp"
#include <vector>
#include <memory>
#include <set>
#include <map>
#include "ParameterIndexer.hpp"
#include <tuple>

namespace specfit {

class UnifiedFitWorkflow {
public:
    struct Config
    {
        int    n_outlier_iterations = 3;
        bool   verbose              = true;
        bool   debug_plots          = false;
        std::tuple<double,double> chi_thresholds = {-2.0, 2.0};
        std::vector<std::string> untie_params;

        int    nit_noise_max            = 5;
        int    nit_fit_max              = 5;
        int    width_box_px             = 50;  // ISIS: width_of_box_filter_in_pixels
        double outlier_sigma_lo         = 2.0;
        double outlier_sigma_hi         = 2.0;
        double conv_range_lo            = 0.9;
        double conv_range_hi            = 1.1;
        double conv_fraction            = 0.9;

        /*  ISIS's auto_freeze_sur_ratio (on by default there, off here --
         *  GAEL only ever drops a component the user asked for if the user
         *  asked for that too).  When on, a secondary whose surface ratio or
         *  whose peak contribution to the composite falls below the
         *  thresholds below is retired: its surface ratio is set to zero and
         *  every one of its parameters is frozen.                           */
        bool   auto_freeze_sur_ratio    = false;
        double sur_ratio_thres          = 5.0;    // ISIS: sur_ratio_thres
        double c2_detection_thres       = 0.05;   // ISIS: c2_detection_thres

        // Called at the end of every fitting stage if set.
        // stage_index: 0-based counter across all solve_stage() calls.
        // Implementation-defined usage: the CLI hooks MultiPanelPlotter here.
        std::function<void(int stage_index,
                        const UnifiedFitWorkflow& wf)> on_stage_complete;
    };

    UnifiedFitWorkflow(std::vector<DataSet>& datasets,
                       SharedModel&           model,
                       const Config&          config,
                       const std::vector<std::map<std::string,bool>>& frozen_status,
                       int                    nthreads);

    void run();

    // Warm-started light refit for the continuum-jitter error ensemble:
    // settle the (re-seeded) continuum at the current stellar values, then one
    // joint continuum+stellar solve. Skips the progressive stages, the
    // iterative-noise/outlier rejection, vsini auto-freeze, and Powell.
    void quick_refit(int max_iterations = 60);

    Vector get_model_for_dataset(std::size_t dataset_idx) const;
    const LMSolverSummary& get_summary() const { return summary_; }
    const std::vector<double>& get_parameters()   const { return unified_params_; }
    const std::vector<double>& get_uncertainties() const { return final_uncertainties_; }
    const double& get_final_chi2() const { return summary_.final_chi2; }
    const std::vector<bool>& get_free_mask() const { return last_free_mask_; }

    const ParameterIndexer& get_indexer() const { return indexer_; }
    int  n_components() const { return static_cast<int>(model_.params.size()); }

    // total count of continuum parameters across all datasets
    int  n_continuum_params() const {
        int n = 0; for (const auto& d : datasets_) n += (int)d.cont_y.size();
        return n;
    }

private:
    void solve_stage(const std::set<std::string>& free_params,
                     int                          max_iterations,
                     bool                         add_powell = false);

    void stage1_continuum_only();
    void stage2_continuum_vrad();
    void stage3_continuum_vrad_teff_logg_z();
    void stage4_full(bool add_powell = false);
    void stage5_auto_freeze_vsini();
    void stage5b_auto_freeze_sur_ratio();
    void stage6_rescale_and_reject();
    void stage7_final();
    
    void report_boundary_parameters() const;
    double chi2_current() const;      //  <──  new

private:

    LMWorkspace lm_mem_;   // lives as long as the workflow lives
    std::vector<DataSet>& datasets_;
    SharedModel&          model_;
    Config                config_;
    std::vector<std::map<std::string,bool>> frozen_status_;
    int                   nthreads_;

    /* --- NEW : stellar-parameter mapping ---------------------------- */
    ParameterIndexer      indexer_;

    std::vector<double>   unified_params_;
    LMSolverSummary       summary_;
    std::vector<double>   final_uncertainties_;   // filled after stage 6
    std::vector<bool>  last_free_mask_;  
};


} // namespace specfit