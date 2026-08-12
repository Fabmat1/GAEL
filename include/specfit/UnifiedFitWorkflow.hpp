
#pragma once
#include "Types.hpp"
#include "CommonTypes.hpp"
#include "FitProgress.hpp"
#include "ModelGrid.hpp"
#include "SimpleLM.hpp"
#include <vector>
#include <memory>
#include <set>
#include <map>
#include "ParameterIndexer.hpp"
#include <tuple>
#include <utility>

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

        /*  Where to report progress, and which phase of that tracker's plan
         *  this workflow owns.  run() / quick_refit() expand that one phase
         *  into their own ladder, so the session does not have to know what
         *  the ladder looks like.  Null tracker == no reporting; the
         *  workflow is otherwise unchanged.  Not owned.                    */
        FitProgressTracker* progress       = nullptr;
        int                 progress_phase = -1;
        /*  Prefix for the phase labels this workflow creates, so that a
         *  warm refit run as part of an ensemble can say which one it is.
         *  Empty for the main fit, whose stage names stand on their own.  */
        std::string         progress_label;
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

    /*  Prior cost of one quick_refit(), in the same arbitrary units the
     *  progress tracker's phase weights use.  Lets the caller of an ensemble
     *  of warm refits weight them against the main fit before any of them
     *  has run; see FitProgress.hpp.                                       */
    double estimated_quick_refit_cost(int max_iterations = 60) const;

    Vector get_model_for_dataset(std::size_t dataset_idx) const;

    /*  One component's own model on the dataset's wavelength grid: the fitted
     *  continuum (and telluric transmission, when the spectrum fits one) times
     *  that component's normalised flux alone, undiluted by the other
     *  component's light.  Same thing as get_model_for_dataset for a
     *  single-component fit.                                                 */
    Vector get_component_model_for_dataset(std::size_t dataset_idx,
                                           int component) const;

    /*  Lower/upper solver limit of every entry of the global parameter vector;
     *  continuum and telluric entries report +-inf.  Lets a consumer see which
     *  side of its range a converged parameter is pinned against -- an
     *  abundance sitting at the low edge of its axis is an upper limit, not a
     *  measurement.                                                          */
    std::vector<std::pair<double,double>> get_param_limits() const;

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

    // total count of telluric parameters (3 per spectrum that fits them)
    int  n_telluric_params() const { return n_telluric_; }

    /*  Where dataset d's telluric parameters start in the global vector,
     *  or -1 if that spectrum has none.                                    */
    int  telluric_offset_of(std::size_t d) const
    { return telluric_param_offset(d); }

private:
    void solve_stage(const std::set<std::string>& free_params,
                     int                          max_iterations,
                     bool                         add_powell = false);

    void stage1_continuum_only();
    void stage2_continuum_vrad();
    void stage3_continuum_vrad_teff_logg_z();

    /*  The free-parameter sets of stages 2 and 3, so that the stage and its
     *  cost estimate cannot drift apart.                                   */
    std::set<std::string> free_params_stage2() const;
    std::set<std::string> free_params_stage3() const;
    void stage4_full(bool add_powell = false);
    void stage5_auto_freeze_vsini();
    void stage5b_auto_freeze_sur_ratio();
    void stage6_rescale_and_reject();
    void stage7_final();
    
    void report_boundary_parameters() const;
    double chi2_current() const;      //  <──  new

    /* ---- progress bookkeeping ---------------------------------------- *
     *  Prior cost of one solve_stage() call, in "dataset-synthetic-spectrum
     *  evaluations": the unit the fit actually spends its time in.  Used
     *  only to seed FitProgressTracker's phase weights, which it then
     *  recalibrates against the clock -- so this has to get the *ordering*
     *  of the stages right, not their absolute cost.                      */
    double solve_cost(const std::set<std::string>& free_params,
                      int max_iterations) const;

    /*  Turn the tracker's single placeholder phase into this run's ladder,
     *  filling phase_.  A no-op when there is no tracker.                  */
    void plan_stages(bool do_1, bool do_23, bool do_6);

    /*  Enter/leave the tracker phase that the stages below report into.  */
    void enter_phase(int id, const std::string& detail = {});
    void leave_phase();

    /*  Phase ids handed out by run() / quick_refit(); -1 when the tracker is
     *  inert or the phase does not apply to this run.                     */
    struct StagePhases {
        int stage1 = -1, stage2 = -1, stage3 = -1, stage4 = -1;
        int stage5 = -1, stage5b = -1;
        std::vector<int> stage6;
        int stage7 = -1, stage7_boundary = -1;
    } phase_;
    int active_phase_ = -1;

    /*  Grid coverage intersected over every component's grid. */
    ModelGrid::ParameterBounds grid_bounds() const;

    /*  Shared implementation of the two model getters: `only_component` < 0
     *  builds the composite model, otherwise just that component's.          */
    Vector model_for_dataset(std::size_t dataset_idx, int only_component) const;

    /*  Solver limits on one parameter: the grid answers for the axes it has,
     *  the rest is fit policy (ISIS's stellar_set_ranges).                   */
    std::pair<double,double> param_limits(
            const ParamSpec& ps, int comp,
            const ModelGrid::ParameterBounds& gb) const;

    /*  Copy the current solution back into model_.params. */
    void sync_model_params();

    /*  Absolute index of dataset d's first telluric parameter, or -1 when
     *  that spectrum does not fit a telluric component.                     */
    int telluric_param_offset(std::size_t d) const;

private:

    LMWorkspace lm_mem_;   // lives as long as the workflow lives
    std::vector<DataSet>& datasets_;
    SharedModel&          model_;
    Config                config_;
    std::vector<std::map<std::string,bool>> frozen_status_;
    int                   nthreads_;

    /* --- NEW : stellar-parameter mapping ---------------------------- */
    ParameterIndexer      indexer_;

    /* --- telluric block: [stellar][telluric][continuum] --------------- */
    int                   telluric_offset_ = 0;   // where the block starts
    int                   n_telluric_      = 0;   // 3 per enabled spectrum
    /*  Per dataset, per telluric parameter.  ISIS leaves all three free;
     *  a spectrum whose tellurics were already removed simply does not
     *  enable the component.                                              */
    std::vector<std::array<bool,3>> frozen_telluric_;

    std::vector<double>   unified_params_;
    LMSolverSummary       summary_;
    std::vector<double>   final_uncertainties_;   // filled after stage 6
    std::vector<bool>  last_free_mask_;  
};


} // namespace specfit