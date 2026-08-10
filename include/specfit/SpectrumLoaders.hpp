// SpectrumLoaders.hpp
#pragma once
#include "specfit/Spectrum.hpp"          //  λ, flux, sigma container
#include <Eigen/Dense>
#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace specfit {

// ---------------------------------------------------------------------------
// generic helpers
// ---------------------------------------------------------------------------
using Vector   = Eigen::VectorXd;
using Real     = double;
using SpectrumLoader = std::function<Spectrum(const std::string&)>;


// stubs for formats that need FITS / proprietary readers --------------------
Spectrum load_ascii_2col       (const std::string& path);  
Spectrum load_ascii_3col       (const std::string& path);   
Spectrum load_sdss             (const std::string& path);
Spectrum load_sdss_v           (const std::string& path);
Spectrum load_lamost           (const std::string& path);
Spectrum load_lamost_dr8       (const std::string& path);
Spectrum load_feros            (const std::string& path);
Spectrum load_feros_phase3     (const std::string& path);
Spectrum load_uves             (const std::string& path);
Spectrum load_xshooter         (const std::string& path);
Spectrum load_xshooter_esoreflex(const std::string& path);
Spectrum load_fuse             (const std::string& path);
Spectrum load_4most            (const std::string& path);
Spectrum load_iraf             (const std::string& path);
Spectrum load_muse             (const std::string& path);

// main entry point -----------------------------------------------------------
Spectrum load_spectrum(const std::string& path,
                       const std::string& format = "auto");

// ---------------------------------------------------------------------------
//  Format-independent post-read clean-up, mirroring what ISIS does right after
//  read_spectrum: drop non-positive flux pixels, repair non-positive errors by
//  interpolating the usable ones, then divide flux and error by the median
//  flux.  Applied by the fit pipeline to every loader.
// ---------------------------------------------------------------------------
Spectrum sanitize_spectrum(const Spectrum& in);

} // namespace specfit