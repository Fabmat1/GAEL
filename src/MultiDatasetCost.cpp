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
                                   int  total_cont_params,
                                   int  total_telluric_params)
    : datasets_(datasets)
    , grids_(grids)
    , n_components_(n_components)
    , n_total_params_(indexer.total_stellar_params + total_telluric_params +
                      total_cont_params)
    , base_cont_offset_(indexer.total_stellar_params + total_telluric_params)
    , base_tell_offset_(indexer.total_stellar_params)
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
    row_count_ .resize(datasets_.size());
    row_of_pixel_.resize(datasets_.size());
    int row = 0;
    for (std::size_t d = 0; d < datasets_.size(); ++d) {
        row_offset_[d] = row;
        const auto& ds = datasets_[d];
        row_of_pixel_[d].assign(ds.ignoreflag.size(), -1);
        for (std::size_t i = 0; i < ds.ignoreflag.size(); ++i)
            if (ds.ignoreflag[i] &&
                std::isfinite(ds.sigma[i]) &&
                ds.sigma[i] > 0.0)
                row_of_pixel_[d][i] = row++;
        row_count_[d] = row - row_offset_[d];
    }

    /* ----------  which spectra does each stellar parameter feed? ------- */
    param_datasets_.assign(base_tell_offset_ > 0
                               ? static_cast<std::size_t>(base_tell_offset_)
                               : 0,
                           {});
    for (int c = 0; c < n_components_; ++c)
        for (int d = 0; d < static_cast<int>(datasets_.size()); ++d)
            for (int k = 0; k < indexer_.n_params(c); ++k) {
                const int j = indexer_.get(c, d, k);
                if (j < 0 || j >= base_tell_offset_) continue;
                auto& v = param_datasets_[static_cast<std::size_t>(j)];
                if (std::find(v.begin(), v.end(), d) == v.end())
                    v.push_back(d);
            }
}

/* --------------------------------------------------------------------- */
/*  the three factors of a residual row, computed separately              */
/* --------------------------------------------------------------------- */
Vector MultiDatasetCost::continuum_of_dataset(const Eigen::VectorXd& p,
                                              std::size_t            d) const
{
    const auto& ds = datasets_[d];
    Eigen::Map<const Vector> cont_y(p.data() + base_cont_offset_ +
                                        ds.cont_param_offset,
                                    ds.cont_param_count);
    return spline_continuum(ds.cont_x, Vector(cont_y), ds.lambda);
}

/* ------------------------------------------------------------------ *
 *  The normalised model spectrum of one dataset.
 *
 *  One component is the ordinary case: the grid's normalised flux is the
 *  model, full stop.
 *
 *  More than one component cannot be an average of normalised spectra --
 *  a line of the secondary is diluted by the *continuum flux* of the
 *  primary, which is a function of wavelength, not by a constant.  ISIS
 *  therefore sums the calibrated fluxes weighted by the fitted surface
 *  ratios and divides by the summed continua (spectroscopic_fitting.sl,
 *  `f_uni /= c_uni`):
 *
 *      n(lambda) = sum_k s_k F_k(lambda) / sum_k s_k C_k(lambda)
 *
 *  with s_1 == 1 by definition.  ISIS forms that sum on the union of the
 *  components' model grids and rebins the result onto the data bins; here
 *  each component's F_k and C_k are rebinned onto the data bins first (the
 *  same flux-conserving operator ISIS's rebinDensity applies) and the sum is
 *  formed there, which keeps every component independently cacheable.  The
 *  two differ only by mean(sum s F)/mean(sum s C) versus mean(sum s F/sum s C)
 *  inside a single bin, and a continuum is flat across one bin.
 * ------------------------------------------------------------------ */
Vector MultiDatasetCost::synth_of_dataset(const Eigen::VectorXd& p,
                                          std::size_t            d) const
{
    const auto& ds = datasets_[d];
    const int   np = static_cast<int>(ds.lambda.size());
    const bool  composite = n_components_ > 1;

    Vector num = Vector::Zero(np);
    Vector den = Vector::Zero(np);

    for (int c = 0; c < n_components_; ++c) {
        const int di = static_cast<int>(d);
        const StellarParams sp = stellar_params_from(indexer_, p, c, di);

        SpectrumPtr s = compute_synthetic_cached(*grids_[c], sp,
                                                  ds.lambda,
                                                  ds.resOffset, ds.resSlope,
                                                  composite);

        if (!composite) return s->flux;      // already normalised

        num += sp.sur_ratio * s->flux;
        den += sp.sur_ratio * s->cont;
    }

    Vector synth(np);
    for (int i = 0; i < np; ++i)
        synth[i] = (den[i] > 0.0) ? num[i] / den[i] : 0.0;
    return synth;
}

/* ------------------------------------------------------------------ *
 *  Telluric transmission of one spectrum.
 *
 *  ISIS models an unnormalised spectrum as `cspline * stellar * telluric`
 *  (initialize_telluric.sl), i.e. the atmosphere is one more multiplicative
 *  factor on the data bins.  An empty result means this spectrum has no
 *  telluric component at all.
 * ------------------------------------------------------------------ */
Vector MultiDatasetCost::telluric_of_dataset(const Eigen::VectorXd& p,
                                             std::size_t            d) const
{
    const auto& ds = datasets_[d];
    if (!ds.telluric || ds.telluric_param_offset < 0) return Vector();

    const int o = ds.telluric_param_offset;
    return ds.telluric->transmission(ds.lambda,
                                     p[o + static_cast<int>(TelluricParam::Airmass)],
                                     p[o + static_cast<int>(TelluricParam::Pwv)],
                                     p[o + static_cast<int>(TelluricParam::Barycorr)],
                                     ds.resOffset, ds.resSlope);
}

void MultiDatasetCost::rows_of_dataset(std::size_t      d,
                                       const Vector&    synth,
                                       const Vector&    continuum,
                                       const Vector&    telluric,
                                       Eigen::VectorXd& r) const
{
    const auto& ds = datasets_[d];
    const int   np = static_cast<int>(ds.lambda.size());
    const bool  has_tell = telluric.size() == ds.lambda.size();

    int row = row_offset_[d];
    for (int i = 0; i < np; ++i) {
        if (!ds.ignoreflag[i]) continue;

        const double sigma = ds.sigma[i];
        if (!std::isfinite(sigma) || sigma <= 0.0) continue;

        double model = synth[i] * continuum[i];
        if (has_tell) model *= telluric[i];
        r[row++]     = (model - ds.flux[i]) / sigma;
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

    for (std::size_t d = 0; d < datasets_.size(); ++d)
        rows_of_dataset(d, synth_of_dataset(p, d),
                           continuum_of_dataset(p, d),
                           telluric_of_dataset(p, d), r);
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

    const std::size_t nds = datasets_.size();

    /* ---------------------------------------------------------------
       The continuum depends only on the anchor ordinates and the
       synthetic spectrum only on the stellar parameters, so each is
       built once here and reused by every column below.  In particular
       the stellar finite differences in (B) no longer re-evaluate the
       continuum spline of every spectrum for every column -- that alone
       was ~40 % of all spline evaluations in a fit.
    ---------------------------------------------------------------- */
    std::vector<Vector> all_cont (nds);
    std::vector<Vector> all_synth(nds);
    std::vector<Vector> all_tell (nds);
    for (std::size_t d = 0; d < nds; ++d) {
        all_cont [d] = continuum_of_dataset(parameters, d);
        all_synth[d] = synth_of_dataset    (parameters, d);
        all_tell [d] = telluric_of_dataset (parameters, d);
    }

    const double eps_base = 1e-6;
    Eigen::VectorXd r0 = Eigen::VectorXd::Zero(num_residuals_);
    for (std::size_t d = 0; d < nds; ++d)
        rows_of_dataset(d, all_synth[d], all_cont[d], all_tell[d], r0);

    /* ========== (A) continuum anchors ============================== *
     *  The Akima spline is not linear in its ordinates, so these columns
     *  are numeric.  Perturbing anchor k of spectrum d changes only that
     *  spectrum's rows, and leaves the synthetic spectra untouched, so one
     *  spline evaluation is all it costs.
     *
     *  And not even a full one: a (modified) Akima ordinate reaches at most
     *  three knot intervals to either side -- interval j is built from
     *  y[j-2..j+3] through the slope stencil -- so outside that window the
     *  perturbed curve is bit-for-bit the unperturbed one and the column is
     *  exactly zero.  Evaluating only the window (a generous +/-4 knots)
     *  leaves every entry unchanged and cuts this block by ~3x.  The bound
     *  was checked numerically as well as derived: over 200 random knot
     *  layouts no difference ever appeared further than 3 intervals away.  */
    for (std::size_t ds_idx = 0; ds_idx < nds; ++ds_idx) {
        const auto&   ds        = datasets_[ds_idx];
        const int     na        = ds.cont_param_count;
        const int     nx        = static_cast<int>(ds.cont_x.size());
        const Vector& continuum = all_cont[ds_idx];
        const Vector& synth     = all_synth[ds_idx];
        const auto&   pix_row   = row_of_pixel_[ds_idx];

        Eigen::Map<const Vector> cont_y(parameters.data() +
                                        base_cont_offset_ +
                                        ds.cont_param_offset, na);

        const double* lam_b = ds.lambda.data();
        const double* lam_e = lam_b + ds.lambda.size();

        constexpr int kAnchorReach = 4;      // >= the 3 intervals of support

        for (int k = 0; k < na; ++k) {
            const int j_global = base_cont_offset_ + ds.cont_param_offset + k;
            if (!is_free(j_global)) continue;      // column is never read

            /* pixel window this anchor can possibly change */
            const int klo = k - kAnchorReach;
            const int khi = k + kAnchorReach;
            const Eigen::Index i0 =
                (klo <= 0) ? 0
                           : std::lower_bound(lam_b, lam_e, ds.cont_x[klo]) - lam_b;
            const Eigen::Index i1 =
                (khi >= nx - 1) ? ds.lambda.size()
                                : std::upper_bound(lam_b, lam_e, ds.cont_x[khi]) - lam_b;
            if (i1 <= i0) continue;          // nothing fitted in the window

            Vector cy_eps = cont_y;
            const double h = eps_base * (std::abs(cy_eps[k]) + 1.0);
            cy_eps[k] += h;

            const Vector lam_win  = ds.lambda.segment(i0, i1 - i0);
            const Vector cont_eps = spline_continuum(ds.cont_x, cy_eps, lam_win);

            const Vector& tell     = all_tell[ds_idx];
            const bool    has_tell = tell.size() == ds.lambda.size();

            for (Eigen::Index i = i0; i < i1; ++i) {
                const int row = pix_row[static_cast<std::size_t>(i)];
                if (row < 0) continue;       // pixel not fitted

                double dmodel = synth[i] * (cont_eps[i - i0] - continuum[i]);
                if (has_tell) dmodel *= tell[i];
                jacobians->coeffRef(row, j_global) =
                    dmodel / (h * ds.sigma[i]);
            }
        }
    }

    /* ========== (A2) telluric parameters =========================== *
     *  Airmass, pwv and barycorr belong to one spectrum, so -- exactly like
     *  that spectrum's continuum anchors -- their columns are zero everywhere
     *  else and only that spectrum's rows have to be redone.  The synthetic
     *  spectrum and the continuum are untouched by them, so neither is
     *  recomputed.                                                          */
    for (std::size_t d = 0; d < nds; ++d) {
        const auto& ds = datasets_[d];
        if (!ds.telluric || ds.telluric_param_offset < 0) continue;

        const Eigen::Index o = row_offset_[d];
        const Eigen::Index n = row_count_ [d];
        if (n <= 0) continue;

        Eigen::VectorXd p_eps;
        Eigen::VectorXd r_eps;
        for (int k = 0; k < kNTelluricParams; ++k) {
            const int j = ds.telluric_param_offset + k;
            if (!is_free(j)) continue;

            const double h = eps_base * (std::abs(parameters[j]) + 1.0);
            p_eps = parameters;
            p_eps[j] += h;

            r_eps = r0;
            rows_of_dataset(d, all_synth[d], all_cont[d],
                            telluric_of_dataset(p_eps, d), r_eps);

            jacobians->col(j).segment(o, n) =
                (r_eps.segment(o, n) - r0.segment(o, n)) / h;
        }
    }

    /* ========== (B) FD for stellar parameters ====================== *
     *  A tied parameter moves every spectrum, an untied one (vrad, by
     *  default) only its own.  Recomputing the untouched spectra produced
     *  residuals identical to r0 and hence exactly-zero Jacobian entries,
     *  so restricting the difference to the spectra the parameter actually
     *  feeds changes no number and skips 4/5 of the work per vrad column
     *  on a five-spectrum fit.                                           */
    Eigen::VectorXd p_eps;
    Eigen::VectorXd r_eps;
    for (int j = 0; j < base_tell_offset_; ++j)   // stellar parameters
    {
        if (!is_free(j)) continue;                // column is never read

        const auto& affected = param_datasets_[static_cast<std::size_t>(j)];
        if (affected.empty()) continue;           // feeds no spectrum

        const double h = eps_base * (std::abs(parameters[j]) + 1.0);
        p_eps = parameters;
        p_eps[j] += h;

        r_eps = r0;                               // untouched rows are identical
        for (int d : affected)
            rows_of_dataset(static_cast<std::size_t>(d),
                            synth_of_dataset(p_eps, static_cast<std::size_t>(d)),
                            all_cont[static_cast<std::size_t>(d)],
                            all_tell[static_cast<std::size_t>(d)],
                            r_eps);

        for (int d : affected) {
            const Eigen::Index o = row_offset_[static_cast<std::size_t>(d)];
            const Eigen::Index n = row_count_ [static_cast<std::size_t>(d)];
            if (n <= 0) continue;
            jacobians->col(j).segment(o, n) =
                (r_eps.segment(o, n) - r0.segment(o, n)) / h;
        }
    }
    //std::cout << "[CostFunc] Finite Differences done." << std::endl;
}

} // namespace specfit