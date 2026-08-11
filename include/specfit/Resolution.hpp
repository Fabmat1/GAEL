#pragma once
#include "Types.hpp"     // supplies Vector = Eigen::VectorXd

namespace specfit {

/**
 * Degrade a high-resolution spectrum to the requested variable resolving
 * power  R(λ) = resOffset + resSlope · λ .
 *
 * A wavelength-dependent Gaussian convolution is performed.
 */
Vector degrade_resolution(const Vector& lam,
                          const Vector& flux,
                          double        resOffset,
                          double        resSlope);

/**
 * The same convolution, but evaluated on a *separate* output grid.
 *
 * The integral still runs over every point of `lam_in`, so nothing of the
 * input is thrown away; only the points at which the result is reported
 * change.  That matters when the input grid is far finer than the convolved
 * result can possibly be: a metal-bearing model lives on the union of ~25
 * species grids (600 k points on the Feros grids) purely so that the product
 * of the line-ratio spectra is formed at full resolution, and evaluating a
 * Gaussian of width lambda/R at all 600 k of those points costs -- in weights
 * as much as in flops -- roughly forty times what it costs at a sampling that
 * actually resolves the answer.  ISIS makes the same split (convolve_syn
 * builds `clambda` at ~2R and convolves onto that).
 *
 * `lam_out` must be ascending and should lie inside the range of `lam_in`.
 * Passing `lam_in` itself reproduces the four-argument overload bit for bit.
 */
Vector degrade_resolution(const Vector& lam_in,
                          const Vector& flux,
                          const Vector& lam_out,
                          double        resOffset,
                          double        resSlope);

#ifdef GAEL_USE_CUDA
// Forward declaration of the GPU implementation (only compiled when nvcc exists)
Vector degrade_resolution_cuda(const Vector& lam,
                               const Vector& flux,
                               double        resOffset,
                               double        resSlope);
#endif

} // namespace specfit