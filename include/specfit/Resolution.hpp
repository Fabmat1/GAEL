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

#ifdef GAEL_USE_CUDA
// Forward declaration of the GPU implementation (only compiled when nvcc exists)
Vector degrade_resolution_cuda(const Vector& lam,
                               const Vector& flux,
                               double        resOffset,
                               double        resSlope);
#endif

} // namespace specfit