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
    static constexpr int kStellarParamsPerComp = 8;   // kept for legacy bounds

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

private:
    /* only residuals (re-used by numeric differentiation) */
    void compute_residuals(const Eigen::VectorXd& parameters,
                           Eigen::VectorXd&       residuals) const;

    /* data */
    std::vector<DatasetInfo> datasets_;
    std::vector<ModelGrid*>  grids_;
    int                      n_components_;
    int                      n_total_params_;
    int                      base_cont_offset_;
    int                      num_residuals_;
    std::vector<int>         row_offset_;   // first residual row per spectrum

    const ParameterIndexer&  indexer_;
};

} // namespace specfit
