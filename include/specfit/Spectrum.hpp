#pragma once
#include "Types.hpp"
#include <string>
#include <vector>
#include <optional>

namespace specfit {

// SNR estimation result
struct SNRResult {
    Real noise;  // Noise level estimate
    Real snr;    // Signal-to-noise ratio
};

// SNR curve result (windowed SNR estimates)
struct SNRCurve {
    Vector lambda;  // Wavelength points where SNR was evaluated
    Vector noise;   // Local noise estimates
    Vector snr;     // Local SNR estimates
};

// Container for an (optionally rebinned) spectrum
struct Spectrum {
    Vector               lambda;       // Å
    Vector               flux;         // arbitrary units
    Vector               sigma;        // 1-σ uncertainties (same units as flux)

    /* 1 == use point, 0 == ignore point during the optimisation          *
     * The size is always identical to lambda.size()                      */
    std::vector<int>     ignoreflag;

    /* ------------------------------------------------------------------ *
     *  Multi-component (binary) extras -- empty for everything else.
     *
     *  A two-component fit cannot combine normalised spectra: the mixing
     *  weight is the ratio of the two *calibrated* continuum fluxes, which
     *  depends on wavelength.  ISIS therefore builds each component from the
     *  grid's "c" column (calibrated flux, erg/s/cm^2/A) and divides the sum
     *  by the summed continua (spectroscopic_fitting.sl, `f_uni /= c_uni`).
     *  When a spectrum is produced in that mode, `flux` carries the
     *  calibrated flux and `cont` the component's continuum.
     *
     *  `cont_den` is scratch that only exists on cached *grid corner*
     *  spectra: the continuum is interp(c)/interp(f) -- both columns are
     *  interpolated across the hypercube corners and only then divided, as
     *  ISIS does -- so the two sums have to be carried separately until the
     *  corner loop is finished.  ModelGrid::load_spectrum clears it.
     * ------------------------------------------------------------------ */
    Vector               cont;         // continuum, calibrated flux units
    Vector               cont_den;     // scratch: normalised flux ("f" column)


    // SNR estimation methods
    
    // Gaussian-based SNR estimation (Irrgang et al. 2014, A&A, 565, A63)
    SNRResult estimate_snr_gaussian(int neighbor = 2) const;
    
    // DER_SNR algorithm (Stoehr et al. 2008)
    SNRResult estimate_snr_der(int order = 3) const;
    
    // Windowed SNR estimation along the spectrum
    SNRCurve estimate_snr_curve(const std::string& method = "der_snr", 
                                 int window_size = 300,
                                 int neighbor = 2) const;
    
    // Convenience method that interpolates SNR curve back to original lambda grid
    void compute_snr_errors(Vector& errors_out,
                            Vector& snr_out,
                            const std::string& method = "der_snr",
                            int window_size = 300) const;
    
private:
    // Helper functions
    SNRResult estimate_snr_gaussian_impl(const Vector& flux_vec, int neighbor) const;
    SNRResult estimate_snr_der_impl(const Vector& flux_vec, int order) const;
    Vector compute_delta_distribution(const Vector& flux_vec, int neighbor) const;
};

/* unchanged */
Spectrum load_ascii(const std::string& path, bool three_col);

} // namespace specfit