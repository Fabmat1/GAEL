#pragma once
#include "Types.hpp"
#include "ModelGrid.hpp"
#include "SyntheticModel.hpp"
#include "Spectrum.hpp"
#include "TelluricGrid.hpp"
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

    /*  Telluric transmission fitted on this spectrum.  `telluric_param_offset`
     *  is an *absolute* index into the global parameter vector (unlike the
     *  continuum offset, which is relative to the continuum block) and is
     *  meaningful only when the grid is set.                                */
    const TelluricGrid* telluric = nullptr;
    int           telluric_param_offset = -1;
};

/* ------------------------------------------------------------------------- */
/*  Residual functor for ALL spectra at once                                 */
/* ------------------------------------------------------------------------- */
class MultiDatasetCost {
public:
    MultiDatasetCost(const std::vector<DatasetInfo>& datasets,
                     const std::vector<ModelGrid*>&  grids,
                     int  n_components,
                     const ParameterIndexer&         indexer,
                     int  total_residuals,
                     int  total_cont_params,
                     int  total_telluric_params = 0);

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

    /*  Where each parameter's column goes in the matrix handed to operator().
     *
     *  Empty (the default) means "one column per parameter", i.e. the full
     *  n_total_params_-wide Jacobian.  Set to the solver's own full->reduced
     *  column map, the matrix is only as wide as there are free parameters and
     *  the column for parameter j is written at col_map[j] (-1 = frozen, never
     *  written).  That is exactly what the solver goes on to use, so it saves
     *  allocating, zeroing and then copying a second matrix of the same size:
     *  on an 18-arm metal fit the full Jacobian is 1.3 GB and only ~20 of its
     *  347 columns are dropped on the way into the reduced one.             */
    void set_column_map(const std::vector<int>& col_map) { col_map_ = col_map; }

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

    /*  Telluric transmission of one spectrum, or an empty vector when this
     *  spectrum does not fit one -- an empty vector means "no factor" rather
     *  than a vector of ones, so a fit without tellurics does no extra work
     *  and produces bit-identical residuals.                                */
    Vector telluric_of_dataset (const Eigen::VectorXd& p, std::size_t d) const;

    void   rows_of_dataset(std::size_t d,
                           const Vector& synth, const Vector& continuum,
                           const Vector& telluric,
                           Eigen::VectorXd& r) const;

    /* data */
    std::vector<DatasetInfo> datasets_;
    std::vector<ModelGrid*>  grids_;
    int                      n_components_;
    int                      n_total_params_;
    int                      base_cont_offset_;   // start of the continuum block
    int                      base_tell_offset_;   // start of the telluric block
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
    std::vector<int>         col_map_;     // empty == column j is column j

    bool is_free(int j) const
    { return free_mask_.empty() ||
             (j < static_cast<int>(free_mask_.size()) && free_mask_[j]); }

    /*  Column of the output matrix that parameter j is written to, or -1 when
     *  it has none.  Pairs with is_free(): a parameter that is not free has no
     *  column under a reduced map and is never differenced anyway.           */
    int column_of(int j) const
    {
        if (col_map_.empty()) return j;
        return (j < static_cast<int>(col_map_.size())) ? col_map_[j] : -1;
    }

    /*  Width of the matrix operator() fills. */
    int jacobian_cols() const
    {
        if (col_map_.empty()) return n_total_params_;
        int n = 0;
        for (int c : col_map_) n = std::max(n, c + 1);
        return n;
    }

    const ParameterIndexer&  indexer_;
};

} // namespace specfit
