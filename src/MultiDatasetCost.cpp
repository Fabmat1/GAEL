#include "specfit/MultiDatasetCost.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "specfit/Types.hpp"
#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace specfit {

/* ------------------------------------------------------------------ *
 *  The continuum is the Akima spline through (cont_x, cont_y).
 *
 *  This used to be modelled as sum_k a_k * makima(e_k) -- a LINEAR expansion
 *  in pre-computed "Akima basis" curves -- which let the anchor columns of the
 *  Jacobian be written down analytically.  makima is not a linear function of
 *  its ordinates, so that expansion is a different curve from makima(cont_x,a)
 *  (0.6% rms, 2% peak on a converged real fit), and makima(cont_x,a) is what
 *  get_model_for_dataset(), the reported continuum and the iterative-noise
 *  stage all use.  Fitting one curve and reporting another biased log g by
 *  -0.03 dex and Teff by +90 K against ISIS; evaluating the real spline here
 *  removes both and halves the scatter.  The anchor columns are numeric now,
 *  but only the owning spectrum's rows change, so they stay cheap.
 * ------------------------------------------------------------------ */
static inline
Vector spline_continuum(const Vector& cont_x, const Vector& cont_y,
                        const Vector& lambda)
{
    return AkimaSpline(cont_x, cont_y)(lambda);
}

MultiDatasetCost::MultiDatasetCost(const std::vector<DatasetInfo>& datasets,
                                   const std::vector<ModelGrid*>&  grids,
                                   int  n_components,
                                   const ParameterIndexer& indexer,
                                   int  /*total_residuals (unused)*/,
                                   int  total_cont_params)
    : datasets_(datasets)
    , grids_(grids)
    , n_components_(n_components)
    , n_total_params_(indexer.total_stellar_params + total_cont_params)
    , base_cont_offset_(indexer.total_stellar_params)
    , indexer_(indexer)
{
    /* how many points are *actually* fitted? -------------------------- */
    int kept_points = 0;
    for (const auto& ds : datasets_)
        for (size_t i = 0; i < ds.ignoreflag.size(); ++i)
            if (ds.ignoreflag[i] &&
                std::isfinite(ds.sigma[i]) &&
                ds.sigma[i] > 0.0)
                ++kept_points;

    num_residuals_ = kept_points;

    /* ----------  first residual row belonging to each spectrum --------- */
    row_offset_.resize(datasets_.size());
    int row = 0;
    for (std::size_t d = 0; d < datasets_.size(); ++d) {
        row_offset_[d] = row;
        const auto& ds = datasets_[d];
        for (std::size_t i = 0; i < ds.ignoreflag.size(); ++i)
            if (ds.ignoreflag[i] &&
                std::isfinite(ds.sigma[i]) &&
                ds.sigma[i] > 0.0)
                ++row;
    }
}

/* --------------------------------------------------------------------- */
/*  residuals only (no derivatives)                                      */
/* --------------------------------------------------------------------- */
/* --------------------------------------------------------------------- */
/*  (1)  residuals only                                                  */
/* --------------------------------------------------------------------- */
void MultiDatasetCost::compute_residuals(const Eigen::VectorXd& p,
                                         Eigen::VectorXd&       r) const
{
    r.setZero(num_residuals_);
    int row = 0;
    //std::cout << "[CostFunc](residuals) Entering loop.";

    /* ---------- loop over spectra ----------------------------------- */
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds  = datasets_[ds_idx];
        const int   np  = static_cast<int>(ds.lambda.size());
        const int   na  = ds.cont_param_count;

        //std::cout << "[CostFunc](residuals)(loop) Continuum." << std::endl;
        /* ---- continuum -------------------------------------------- */
        Eigen::Map<const Vector> cont_y(p.data() +
                                        base_cont_offset_ +
                                        ds.cont_param_offset, na);
        const Vector continuum =
            spline_continuum(ds.cont_x, Vector(cont_y), ds.lambda);

        //std::cout << "[CostFunc](residuals)(loop) Synthetic spectra." << std::endl;
        /* ---- synthetic composite spectrum ------------------------- */
        Vector synth  = Vector::Zero(np);
        double w_sum  = 0.0;

        for (int c = 0; c < n_components_; ++c) {
            StellarParams sp;
            const int d = static_cast<int>(ds_idx);
            sp.vrad  = p[indexer_.get(c,d,0)];
            sp.vsini = p[indexer_.get(c,d,1)];
            sp.zeta  = p[indexer_.get(c,d,2)];
            sp.teff  = p[indexer_.get(c,d,3)];
            sp.logg  = p[indexer_.get(c,d,4)];
            sp.xi    = p[indexer_.get(c,d,5)];
            sp.z     = p[indexer_.get(c,d,6)];
            sp.he    = p[indexer_.get(c,d,7)];

            Spectrum s = compute_synthetic(*grids_[c], sp,
                                            ds.lambda,
                                            ds.resOffset, ds.resSlope);

            const double w = std::pow(sp.teff, 4);
            synth         += w * s.flux;
            w_sum         += w;
        }
        if (w_sum > 0.0) synth.array() /= w_sum;

        //std::cout << "[CostFunc](residuals)(loop) Residuals." << std::endl;
        /* ---- χ residuals  (skip ignored points) ------------------- */
        for (int i = 0; i < np; ++i) {
            if (!ds.ignoreflag[i]) continue;

            const double sigma = ds.sigma[i];
            if (!std::isfinite(sigma) || sigma <= 0.0) continue;

            const double model = synth[i] * continuum[i];
            r[row++]           = (model - ds.flux[i]) / sigma;
        }
    }
}

/* --------------------------------------------------------------------- */
/*  (2)  residuals + Jacobian                                            */
/* --------------------------------------------------------------------- */
void MultiDatasetCost::operator()(const Eigen::VectorXd& parameters,
                                  Eigen::VectorXd*       residuals,
                                  Eigen::MatrixXd*       jacobians) const
{
    //std::cout << "[CostFunc] Beginning of Cost Function." << std::endl; 
    //std::cout << "[CostFunc] Calculating residuals." << std::endl; 
    /* ========== residual vector ==================================== */
    if (residuals) {
        residuals->resize(num_residuals_);
        compute_residuals(parameters, *residuals);
    }
    //std::cout << "[CostFunc] Calculated residuals. Reserving Space" << std::endl; 

    if (!jacobians) return;                       // user wants resid only
    jacobians->setZero(num_residuals_, n_total_params_);

    /* ========== common data for analytic continuum columns ========= */
    std::vector<Vector>  all_synth;  // one per dataset (to reuse later)
    std::vector<Vector>  all_sigma;

    all_synth.reserve(datasets_.size());
    all_sigma.reserve(datasets_.size());

    for (const auto& ds : datasets_)
        all_sigma.push_back(ds.sigma);            // cheap copy of ref

        //std::cout << "[CostFunc] Reserved Space. Entering Dataset Loop." << std::endl; 
    /* ---------------------------------------------------------------
       Build synthetic spectra & keep them – we need them twice:
       (a) for analytic Jacobian columns
       (b) for residual FD later (saves recomputation)
    ---------------------------------------------------------------- */
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds  = datasets_[ds_idx];
        const int   np  = static_cast<int>(ds.lambda.size());

        Vector synth  = Vector::Zero(np);
        double w_sum  = 0.0;

        for (int c = 0; c < n_components_; ++c) {
            StellarParams sp;
            const int d = static_cast<int>(ds_idx);
            sp.vrad  = parameters[indexer_.get(c,d,0)];
            sp.vsini = parameters[indexer_.get(c,d,1)];
            sp.zeta  = parameters[indexer_.get(c,d,2)];
            sp.teff  = parameters[indexer_.get(c,d,3)];
            sp.logg  = parameters[indexer_.get(c,d,4)];
            sp.xi    = parameters[indexer_.get(c,d,5)];
            sp.z     = parameters[indexer_.get(c,d,6)];
            sp.he    = parameters[indexer_.get(c,d,7)];

            //std::cout << "[CostFunc] (loop)Getting Synth spectrum." << std::endl; 
            Spectrum s = compute_synthetic(*grids_[c], sp,
                                            ds.lambda,
                                            ds.resOffset, ds.resSlope);
                                            //std::cout << "[CostFunc] (loop)Got Synth spectrum." << std::endl; 
            const double w = std::pow(sp.teff, 4);
            synth         += w * s.flux;
            w_sum         += w;
        }
        if (w_sum > 0.0) synth.array() /= w_sum;
        all_synth.emplace_back(std::move(synth));
    }
    //std::cout << "[CostFunc] Loop Done. Analytical Continuum eval." << std::endl; 

    const double eps_base = 1e-6;
    Eigen::VectorXd r0;
    compute_residuals(parameters, r0);

    /* ========== (A) continuum anchors ============================== *
     *  The Akima spline is not linear in its ordinates, so these columns
     *  are numeric.  Perturbing anchor k of spectrum d changes only that
     *  spectrum's rows, and leaves the synthetic spectra untouched, so one
     *  spline evaluation is all it costs.                                */
    for (std::size_t ds_idx = 0; ds_idx < datasets_.size(); ++ds_idx) {
        const auto& ds  = datasets_[ds_idx];
        const int   np  = static_cast<int>(ds.lambda.size());
        const int   na  = ds.cont_param_count;

        Eigen::Map<const Vector> cont_y(parameters.data() +
                                        base_cont_offset_ +
                                        ds.cont_param_offset, na);
        const Vector continuum =
            spline_continuum(ds.cont_x, Vector(cont_y), ds.lambda);

        for (int k = 0; k < na; ++k) {
            const int j_global = base_cont_offset_ + ds.cont_param_offset + k;

            Vector cy_eps = cont_y;
            const double h = eps_base * (std::abs(cy_eps[k]) + 1.0);
            cy_eps[k] += h;
            const Vector cont_eps =
                spline_continuum(ds.cont_x, cy_eps, ds.lambda);

            int row = row_offset_[ds_idx];
            for (int i = 0; i < np; ++i) {
                if (!ds.ignoreflag[i]) continue;
                const double sigma = all_sigma[ds_idx][i];
                if (!std::isfinite(sigma) || sigma <= 0.0) continue;

                const double dmodel =
                    all_synth[ds_idx][i] * (cont_eps[i] - continuum[i]);
                jacobians->coeffRef(row, j_global) = dmodel / (h * sigma);
                ++row;
            }
        }
    }

    /* ========== (B) FD for stellar parameters ====================== */
    for (int j = 0; j < base_cont_offset_; ++j)   // stellar parameters
    {
        double h = eps_base * (std::abs(parameters[j]) + 1.0);
        Eigen::VectorXd p_eps = parameters;
        p_eps[j] += h;

        Eigen::VectorXd r_eps;
        compute_residuals(p_eps, r_eps);

        jacobians->col(j) = (r_eps - r0) / h;
    }
    //std::cout << "[CostFunc] Finite Differences done." << std::endl;
}

} // namespace specfit