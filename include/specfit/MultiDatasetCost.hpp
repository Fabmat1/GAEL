#pragma once
#include "Types.hpp"
#include "ModelGrid.hpp"
#include "SyntheticModel.hpp"
#include "Spectrum.hpp"
#include "ParameterIndexer.hpp"
#include <Eigen/Core>
#include <vector>

namespace specfit {

/* ------------------------------------------------------------------------- */
/*  Helper that stores everything needed from one observed spectrum          */
/* ------------------------------------------------------------------------- */
struct DatasetInfo {
    Vector        lambda;
    Vector        flux;
    Vector        sigma;
    Vector        cont_x;
    double        resOffset;
    double        resSlope;
    int           cont_param_offset;   // where in the big parameter vector
    int           cont_param_count;
    std::vector<int>  ignoreflag;     // NEW
};

/* ------------------------------------------------------------------------- */
/*  Residual functor for ALL spectra at once                                 */
/* ------------------------------------------------------------------------- */
class MultiDatasetCost {
public:
    /* kept for legacy bounds; tracks the indexer so it cannot drift */
    static constexpr int kStellarParamsPerComp =
        ParameterIndexer::kNStellarParams;

    MultiDatasetCost(const std::vector<DatasetInfo>& datasets,
                     const std::vector<ModelGrid*>&  grids,
                     int  n_components,
                     const ParameterIndexer&         indexer,
                     int  total_residuals,
                     int  total_cont_params);       

    /* number of residuals produced */
    int numResiduals() const { return num_residuals_; }

    /* main entry: returns residuals and (optionally) full Jacobian */
    void operator()(const Eigen::VectorXd& parameters,
                    Eigen::VectorXd*       residuals,
                    Eigen::MatrixXd*       jacobians) const;

    /*  Which parameters the solver is actually free to move.  A frozen
     *  parameter's Jacobian column is never read -- the reduced system only
     *  contains free columns -- so differencing it is pure waste, and in a
     *  continuum-only stage that is every stellar parameter.  An empty mask
     *  (the default) means "assume everything is free".                     */
    void set_free_mask(const std::vector<bool>& mask) { free_mask_ = mask; }

    /*  Residual-row layout, so the solver can be told that a spectrum's
     *  continuum anchors cannot reach any other spectrum's rows.            */
    const std::vector<int>& row_offsets() const { return row_offset_; }
    const std::vector<int>& row_counts () const { return row_count_;  }

private:
    /* only residuals (re-used by numeric differentiation) */
    void compute_residuals(const Eigen::VectorXd& parameters,
                           Eigen::VectorXd&       residuals) const;

    /*  The three pieces a residual row is made of, split apart so that a
     *  finite difference only has to redo the piece that actually moved:
     *  the continuum depends on the anchor ordinates alone, the synthetic
     *  spectrum on the stellar parameters alone.                            */
    Vector continuum_of_dataset(const Eigen::VectorXd& p, std::size_t d) const;
    Vector synth_of_dataset    (const Eigen::VectorXd& p, std::size_t d) const;
    void   rows_of_dataset(std::size_t d,
                           const Vector& synth, const Vector& continuum,
                           Eigen::VectorXd& r) const;

    /* data */
    std::vector<DatasetInfo> datasets_;
    std::vector<ModelGrid*>  grids_;
    int                      n_components_;
    int                      n_total_params_;
    int                      base_cont_offset_;
    int                      num_residuals_;
    std::vector<int>         row_offset_;   // first residual row per spectrum
    std::vector<int>         row_count_;    // residual rows per spectrum

    /*  pixel -> residual row for each spectrum, -1 where the pixel is not
     *  fitted; lets a Jacobian column touch a wavelength range directly.    */
    std::vector<std::vector<int>> row_of_pixel_;

    /*  which spectra each stellar parameter feeds.  A tied parameter feeds
     *  all of them, an untied one (vrad, typically) exactly one -- and then
     *  its finite difference must not recompute the other spectra.          */
    std::vector<std::vector<int>> param_datasets_;

    std::vector<bool>        free_mask_;   // empty == everything free

    bool is_free(int j) const
    { return free_mask_.empty() ||
             (j < static_cast<int>(free_mask_.size()) && free_mask_[j]); }

    const ParameterIndexer&  indexer_;
};

} // namespace specfit
